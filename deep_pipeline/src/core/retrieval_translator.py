"""Retrieval-Guided Translator.

Encodes both eICU and MIMIC into a shared latent space via a shared encoder,
then translates each eICU timestep using its K nearest MIMIC neighbors via
cross-attention.  Avoids global MMD (instance-level matching) and is fully
causal when temporal_attention_mode="causal".

Architecture:
    eICU → Shared Encoder (causal) → latent (B, T, d_latent)
        → Per-timestep backward-looking pool → query
        → Retrieve K nearest MIMIC windows from MemoryBank
        → CrossAttentionBlocks (per-timestep cross-attn + global causal self-attn)
        → Decoder → (B, T, F) output + residual
"""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.translator import AxialBlock

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Memory Bank — stores pre-encoded MIMIC latents for retrieval
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MemoryBank:
    """Holds pre-encoded MIMIC latent representations for nearest-neighbor retrieval.

    GPU-resident:
        window_latents: (N_windows, d_latent) — mean-pooled per non-overlapping window
    CPU-resident:
        timestep_latents: list of (T_i, d_latent) tensors per stay
        pad_masks: list of (T_i,) bool tensors per stay
        window_to_stay_idx: (N_windows,) — maps window → parent stay
        window_to_time_range: (N_windows, 2) — (start_t, end_t) per window
    """
    window_latents: torch.Tensor       # (N_windows, d_latent)  GPU
    timestep_latents: list             # list of (T_i, d_latent) CPU
    pad_masks: list                    # list of (T_i,) CPU
    window_to_stay_idx: torch.Tensor   # (N_windows,) long  CPU
    window_to_time_range: torch.Tensor # (N_windows, 2) long  CPU
    window_size: int


def build_memory_bank(
    encoder: nn.Module,
    target_loader,
    schema_resolver,
    device: str,
    window_size: int = 6,
    window_stride: int | None = None,
) -> MemoryBank:
    """Encode all MIMIC data and build the memory bank.

    Args:
        encoder: the shared encoder (callable with encode() method)
        target_loader: DataLoader over MIMIC training data
        schema_resolver: SchemaResolver instance
        device: CUDA device string
        window_size: window size for bank storage
        window_stride: stride between windows (default=window_size for non-overlapping,
                       set < window_size for overlapping dense banks)
    """
    if window_stride is None:
        window_stride = window_size
    encoder.eval()
    all_timestep_latents = []  # list of (T_i, d_latent) on CPU
    all_pad_masks = []         # list of (T_i,) on CPU

    with torch.no_grad():
        for batch in target_loader:
            batch = tuple(b.to(device) for b in batch)
            parts = schema_resolver.extract(batch)
            latent = encoder.encode(
                parts["X_val"], parts["X_miss"], parts["t_abs"],
                parts["M_pad"], parts["X_static"],
            )  # (B, T, d_latent)
            m_pad = parts["M_pad"].bool()

            for i in range(latent.shape[0]):
                all_timestep_latents.append(latent[i].cpu())    # (T, d_latent)
                all_pad_masks.append(m_pad[i].cpu())             # (T,)

    # Segment into windows (overlapping if stride < window_size) and mean-pool each
    window_latent_list = []
    window_to_stay = []
    window_to_range = []

    W = window_size
    stride = window_stride
    for stay_idx, (ts_lat, ts_mask) in enumerate(zip(all_timestep_latents, all_pad_masks)):
        T = ts_lat.shape[0]
        for start in range(0, T, stride):
            end = min(start + W, T)
            window_mask = ~ts_mask[start:end]  # True = valid
            if window_mask.any():
                valid_latents = ts_lat[start:end][window_mask]  # (n_valid, d_latent)
                window_latent_list.append(valid_latents.mean(dim=0))
                window_to_stay.append(stay_idx)
                window_to_range.append([start, end])

    if len(window_latent_list) == 0:
        raise RuntimeError("Memory bank is empty — no valid MIMIC timesteps found")

    window_latents = torch.stack(window_latent_list).to(device)  # (N_windows, d_latent)
    window_to_stay_idx = torch.tensor(window_to_stay, dtype=torch.long)
    window_to_time_range = torch.tensor(window_to_range, dtype=torch.long)

    logger.info(
        "Memory bank built: %d stays, %d windows (W=%d, stride=%d), %.1f MB GPU",
        len(all_timestep_latents),
        window_latents.shape[0],
        W,
        stride,
        window_latents.nelement() * 4 / 1e6,
    )
    return MemoryBank(
        window_latents=window_latents,
        timestep_latents=all_timestep_latents,
        pad_masks=all_pad_masks,
        window_to_stay_idx=window_to_stay_idx,
        window_to_time_range=window_to_time_range,
        window_size=W,
    )


def query_memory_bank(
    query_latents: torch.Tensor,
    query_pad_mask: torch.Tensor,
    bank: MemoryBank,
    k_neighbors: int = 16,
    retrieval_window: int = 6,
    importance_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-timestep retrieval from the memory bank.

    For each non-padded query timestep t, uses a backward-looking window
    [max(0, t-W+1), t+1] to pool a query vector, then finds K nearest
    MIMIC windows by Euclidean distance.

    Args:
        query_latents: (B, T, d_latent) — encoded eICU latents
        query_pad_mask: (B, T) — True = padded
        bank: MemoryBank with pre-encoded MIMIC windows
        k_neighbors: number of nearest windows to retrieve per timestep
        retrieval_window: backward-looking window size for query pooling
        importance_weights: optional (d_latent,) weights for weighted distance

    Returns:
        context: (B, T, K*W, d_latent) — retrieved MIMIC timestep latents
                 per query timestep. Padded query timesteps get zeros.
    """
    B, T, d_latent = query_latents.shape
    device = query_latents.device
    W = retrieval_window
    K = k_neighbors
    bank_W = bank.window_size

    # Build backward-looking pooled queries: for each timestep t, pool [max(0,t-W+1)..t+1]
    # Use cumsum trick for efficient backward-looking mean pool
    # Accumulate sum and count, then subtract to get window sums
    m_valid = (~query_pad_mask.bool()).unsqueeze(-1).float()  # (B, T, 1)
    lat_masked = query_latents * m_valid  # zero out padded

    # Cumulative sum along time dim
    cumsum = torch.cumsum(lat_masked, dim=1)   # (B, T, d_latent)
    cumcount = torch.cumsum(m_valid, dim=1)     # (B, T, 1)

    # For timestep t, backward window sum = cumsum[t] - cumsum[t-W] (if t >= W)
    # Pad a zero row at position -1
    zero_pad = torch.zeros(B, 1, d_latent, device=device, dtype=cumsum.dtype)
    zero_count = torch.zeros(B, 1, 1, device=device, dtype=cumcount.dtype)
    cumsum_padded = torch.cat([zero_pad, cumsum], dim=1)   # (B, T+1, d_latent)
    cumcount_padded = torch.cat([zero_count, cumcount], dim=1)  # (B, T+1, 1)

    # Indices: for t in [0..T-1], start = max(0, t-W+1), end = t+1
    t_idx = torch.arange(T, device=device)
    start_idx = torch.clamp(t_idx - W + 1, min=0)  # (T,)

    # window_sum[t] = cumsum_padded[t+1] - cumsum_padded[start_idx[t]]
    window_sum = cumsum_padded[:, t_idx + 1, :] - cumsum_padded[:, start_idx, :]  # (B, T, d_latent)
    window_count = cumcount_padded[:, t_idx + 1, :] - cumcount_padded[:, start_idx, :]  # (B, T, 1)
    queries = window_sum / window_count.clamp(min=1)  # (B, T, d_latent)

    # Flatten valid queries for batched distance computation
    valid_mask = ~query_pad_mask.bool()  # (B, T)
    # We compute for ALL timesteps (including padded) then mask afterward
    queries_flat = queries.reshape(B * T, d_latent)  # (B*T, d_latent)

    # Apply importance weighting for distance computation
    bank_vecs = bank.window_latents  # (N_windows, d_latent)
    if importance_weights is not None:
        w_sqrt = importance_weights.sqrt().unsqueeze(0)  # (1, d_latent)
        queries_weighted = queries_flat * w_sqrt
        bank_weighted = bank_vecs * w_sqrt
    else:
        queries_weighted = queries_flat
        bank_weighted = bank_vecs

    # Compute distances in chunks to avoid OOM with large banks
    # (B*T, d_latent) vs (N_windows, d_latent)
    chunk_size = 256
    all_topk_indices = []
    for chunk_start in range(0, B * T, chunk_size):
        chunk_end = min(chunk_start + chunk_size, B * T)
        q_chunk = queries_weighted[chunk_start:chunk_end]  # (chunk, d_latent)
        dists = torch.cdist(q_chunk, bank_weighted.unsqueeze(0).expand(q_chunk.shape[0], -1, -1).reshape(1, -1, d_latent).expand(q_chunk.shape[0], -1, -1).squeeze(0) if False else bank_weighted)
        # cdist: (chunk, N_windows)
        # Actually torch.cdist wants (B, M, D) and (B, N, D) or (M, D) and (N, D)
        # Use matmul-based distance for efficiency
        # ||q - b||^2 = ||q||^2 + ||b||^2 - 2*q@b^T
        pass
    # Actually let me simplify this. torch.cdist works with 2D tensors too.

    # Compute all distances at once if feasible, otherwise chunk
    N_windows = bank_weighted.shape[0]
    total_queries = B * T

    # For V100 with 32GB, (768, 480K) float32 = 1.4GB, fits fine
    # But let's chunk to be safe
    topk_k = min(K, N_windows)
    topk_indices = torch.zeros(total_queries, topk_k, dtype=torch.long, device=device)

    for chunk_start in range(0, total_queries, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_queries)
        q_chunk = queries_weighted[chunk_start:chunk_end]  # (C, d_latent)
        # dists: (C, N_windows)
        dists = torch.cdist(q_chunk.unsqueeze(0), bank_weighted.unsqueeze(0)).squeeze(0)
        _, chunk_topk = dists.topk(topk_k, dim=-1, largest=False)  # (C, K)
        topk_indices[chunk_start:chunk_end] = chunk_topk

    # Gather context: for each query timestep, fetch K windows' timestep latents
    # topk_indices: (B*T, K) — indices into bank.window_latents
    # For each window, get the parent stay and time range, then fetch timestep latents
    topk_flat = topk_indices.reshape(-1).cpu()  # (B*T*K,)
    stay_indices = bank.window_to_stay_idx[topk_flat]     # (B*T*K,)
    time_ranges = bank.window_to_time_range[topk_flat]    # (B*T*K, 2)

    # Gather timestep latents for each retrieved window
    context_list = []
    for i in range(topk_flat.shape[0]):
        s_idx = stay_indices[i].item()
        t_start = time_ranges[i, 0].item()
        t_end = time_ranges[i, 1].item()
        ts_lat = bank.timestep_latents[s_idx][t_start:t_end]  # (W_actual, d_latent) CPU
        # Pad to bank_W if shorter
        if ts_lat.shape[0] < bank_W:
            pad = torch.zeros(bank_W - ts_lat.shape[0], d_latent, dtype=ts_lat.dtype)
            ts_lat = torch.cat([ts_lat, pad], dim=0)
        context_list.append(ts_lat[:bank_W])  # ensure exactly bank_W

    # Stack and reshape: (B*T*K, bank_W, d_latent) → (B*T, K*bank_W, d_latent)
    context_all = torch.stack(context_list).to(device)  # (B*T*K, bank_W, d_latent)
    context = context_all.reshape(B * T, K * bank_W, d_latent)  # (B*T, K*W, d_latent)

    # Reshape to (B, T, K*W, d_latent) and zero out padded timesteps
    context = context.reshape(B, T, K * bank_W, d_latent)
    context = context.masked_fill(query_pad_mask[:, :, None, None], 0.0)

    return context


# ═══════════════════════════════════════════════════════════════════════
#  Cross-Attention Block
# ═══════════════════════════════════════════════════════════════════════

class CrossAttentionBlock(nn.Module):
    """Two-stage attention: per-timestep cross-attention to retrieved neighbors,
    then global causal self-attention over the full sequence."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 512,
        use_causal_self_attn: bool = True,
    ):
        super().__init__()
        self.use_causal_self_attn = use_causal_self_attn

        # 1. Per-timestep cross-attention: Q=eICU timestep, KV=MIMIC neighbors
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # 2. Global causal self-attention over full sequence
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.self_norm = nn.LayerNorm(d_model)

        # 3. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
        m_pad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, T, d_model) — current sequence representation
            context: (B, T, K*W, d_model) — per-timestep retrieved neighbor context
            m_pad: (B, T) — True = padded timestep

        Returns:
            h: (B, T, d_model) — updated representation
        """
        B, T, d_model = h.shape
        KW = context.shape[2]

        # ── 1. Per-timestep cross-attention ──
        # Reshape to (B*T, 1, d_model) for Q and (B*T, KW, d_model) for KV
        q = h.reshape(B * T, 1, d_model)
        kv = context.reshape(B * T, KW, d_model)

        # Zero out all-padded positions to avoid NaN in attention
        pad_flat = m_pad.bool().reshape(B * T)  # True = padded
        if pad_flat.any():
            q[pad_flat] = 0.0
            kv[pad_flat] = 0.0

        cross_out, _ = self.cross_attn(q, kv, kv)  # (B*T, 1, d_model)
        cross_out = cross_out.reshape(B, T, d_model)
        h = self.cross_norm(h + cross_out)
        h = h.masked_fill(m_pad[:, :, None], 0.0)

        # ── 2. Global causal self-attention ──
        attn_mask = None
        if self.use_causal_self_attn:
            attn_mask = torch.triu(
                torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1
            )  # True = blocked

        key_padding_mask = m_pad.bool()  # (B, T)
        # Guard all-padded rows
        all_padded = key_padding_mask.all(dim=-1)  # (B,)
        if all_padded.any():
            h[all_padded] = 0.0

        self_out, _ = self.self_attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        h = self.self_norm(h + self_out)
        h = h.masked_fill(m_pad[:, :, None], 0.0)

        # ── 3. FFN ──
        h = self.ffn_norm(h + self.ffn(h))
        h = h.masked_fill(m_pad[:, :, None], 0.0)

        return h


# ═══════════════════════════════════════════════════════════════════════
#  Retrieval Translator
# ═══════════════════════════════════════════════════════════════════════

class RetrievalTranslator(nn.Module):
    """Retrieval-guided translator: shared encoder + cross-attention to MIMIC
    neighbors + decoder with optional residual output.

    Reuses the same encoder/decoder architecture as SharedLatentTranslator
    (AxialBlock-based), with CrossAttentionBlocks inserted between encoder
    and decoder for per-timestep retrieval fusion.
    """

    def __init__(
        self,
        num_features: int,
        d_latent: int = 128,
        d_model: int = 128,
        d_time: int = 16,
        n_enc_layers: int = 4,
        n_dec_layers: int = 2,
        n_cross_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.2,
        out_dropout: float = 0.1,
        static_dim: int = 4,
        temporal_attention_mode: str = "causal",
        temporal_attention_window: int = 0,
        output_mode: str = "residual",
    ):
        super().__init__()
        if d_time % 2 != 0:
            raise ValueError("d_time must be even for sin/cos encoding")

        self.num_features = num_features
        self.d_latent = d_latent
        self.d_model = d_model
        self.d_time = d_time
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.n_cross_layers = n_cross_layers
        self.temporal_attention_window = temporal_attention_window
        self.temporal_attention_mode = temporal_attention_mode
        self.output_mode = output_mode
        use_causal = temporal_attention_mode == "causal"

        # ── Encoder (same as SharedLatentTranslator) ──
        self.triplet_proj = nn.Linear(3, 16)
        self.sensor_emb = nn.Parameter(torch.zeros(num_features, 16))
        nn.init.normal_(self.sensor_emb, mean=0.0, std=0.02)
        self.lift = nn.Linear(16, d_model)
        self.time_proj = nn.Linear(d_time, d_model)

        self.enc_blocks = nn.ModuleList([
            AxialBlock(d_model, n_heads, dropout, d_ff, use_causal, temporal_attention_window)
            for _ in range(n_enc_layers)
        ])
        self.enc_film = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_enc_layers * d_model),
        )
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.LayerNorm(d_latent),
        )

        # ── Feature Importance for retrieval distance weighting ──
        self.importance_logits = nn.Parameter(torch.zeros(d_latent))

        # ── Cross-attention blocks (new) ──
        # Project latent → d_model for cross-attention
        self.latent_to_cross = nn.Linear(d_latent, d_model)
        self.context_proj = nn.Linear(d_latent, d_model)

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout, d_ff, use_causal)
            for _ in range(n_cross_layers)
        ])

        # ── Decoder (same as SharedLatentTranslator) ──
        self.from_latent = nn.Linear(d_model, d_model)  # post-cross to decoder
        self.dec_feature_emb = nn.Parameter(torch.zeros(num_features, d_model))
        nn.init.normal_(self.dec_feature_emb, mean=0.0, std=0.02)

        self.dec_blocks = nn.ModuleList([
            AxialBlock(d_model, n_heads, dropout, d_ff, use_causal, temporal_attention_window)
            for _ in range(n_dec_layers)
        ])
        self.dec_film = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_dec_layers * d_model),
        )

        self.output_head = nn.Linear(d_model, 1)
        self.out_dropout = nn.Dropout(out_dropout)

        # Label prediction head (for pretraining)
        self.label_pred_head = nn.Sequential(
            nn.Linear(d_latent, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def get_importance_weights(self) -> torch.Tensor:
        """Return (d_latent,) importance weights in [0, 1]."""
        return torch.sigmoid(self.importance_logits)

    def encode(
        self,
        x_val: torch.Tensor,
        x_miss: torch.Tensor,
        t_abs: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Encode to shared latent space. Returns (B, T, d_latent)."""
        m_pad = m_pad.bool()
        B, T, F = x_val.shape

        # Triplet embedding
        t_abs_f = t_abs.to(dtype=x_val.dtype)
        time_delta = torch.zeros_like(t_abs_f)
        time_delta[:, 1:] = t_abs_f[:, 1:] - t_abs_f[:, :-1]
        time_delta = time_delta.masked_fill(m_pad, 0.0)

        td_feat = time_delta.unsqueeze(-1).expand(-1, -1, F)
        x_trip = torch.stack([x_val, x_miss, td_feat], dim=-1)  # (B, T, F, 3)
        h = self.triplet_proj(x_trip)  # (B, T, F, 16)
        h = h + self.sensor_emb.view(1, 1, F, 16)
        h = self.lift(h)  # (B, T, F, d_model)

        # Time encoding
        time_enc = self._time_encoding(t_abs_f)  # (B, T, d_time)
        time_enc = self.time_proj(time_enc)       # (B, T, d_model)
        h = h + time_enc[:, :, None, :]
        h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        # FiLM context
        ctx = self.enc_film(x_static).view(B, self.n_enc_layers, 2, self.d_model)

        # Encoder blocks
        for i, block in enumerate(self.enc_blocks):
            h, _ = block(h, m_pad)
            gamma = ctx[:, i, 0, :].unsqueeze(1).unsqueeze(1)
            beta = ctx[:, i, 1, :].unsqueeze(1).unsqueeze(1)
            h = gamma * h + beta
            h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        # Pool over features → (B, T, d_model), then project to latent
        h_pooled = h.mean(dim=2)  # (B, T, d_model)
        latent = self.to_latent(h_pooled)  # (B, T, d_latent)
        latent = latent.masked_fill(m_pad[:, :, None], 0.0)
        return latent

    def decode_with_context(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Decode from latent with retrieved MIMIC context via cross-attention.

        Args:
            latent: (B, T, d_latent) per-timestep latent from encoder
            context: (B, T, K*W, d_latent) retrieved neighbor latents
            m_pad: (B, T) padding mask
            x_static: (B, S) static features

        Returns: (B, T, F) decoded feature values (before residual addition).
        """
        m_pad = m_pad.bool()
        B, T, _ = latent.shape
        F = self.num_features

        # Project latent and context to d_model for cross-attention
        h = self.latent_to_cross(latent)     # (B, T, d_model)
        ctx_proj = self.context_proj(context)  # (B, T, K*W, d_model)

        # Cross-attention blocks
        for block in self.cross_blocks:
            h = block(h, ctx_proj, m_pad)

        # h is now (B, T, d_model) enriched with neighbor information
        # Proceed to standard decoder: broadcast to features
        h = self.from_latent(h)  # (B, T, d_model)
        h = h.unsqueeze(2).expand(-1, -1, F, -1)  # (B, T, F, d_model)
        h = h + self.dec_feature_emb.view(1, 1, F, self.d_model)
        h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        # FiLM context for decoder
        dec_ctx = self.dec_film(x_static).view(B, self.n_dec_layers, 2, self.d_model)

        for i, block in enumerate(self.dec_blocks):
            h, _ = block(h, m_pad)
            gamma = dec_ctx[:, i, 0, :].unsqueeze(1).unsqueeze(1)
            beta = dec_ctx[:, i, 1, :].unsqueeze(1).unsqueeze(1)
            h = gamma * h + beta
            h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        out = self.output_head(h).squeeze(-1)  # (B, T, F)
        out = self.out_dropout(out)
        out = out.masked_fill(m_pad[:, :, None], 0.0)
        return out

    def decode(
        self,
        latent: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Standard decode without cross-attention (for pretraining / MIMIC path).

        Uses zero context to bypass cross-attention (equivalent to no retrieval).
        """
        B, T, d_latent = latent.shape
        m_pad = m_pad.bool()
        F = self.num_features

        # Project latent to d_model
        h = self.latent_to_cross(latent)  # (B, T, d_model)

        # Create zero context (no neighbors) — cross-attn sees only zeros
        zero_context = torch.zeros(B, T, 1, self.d_model, device=latent.device, dtype=latent.dtype)
        for block in self.cross_blocks:
            h = block(h, zero_context, m_pad)

        # Standard decoder path
        h = self.from_latent(h)  # (B, T, d_model)
        h = h.unsqueeze(2).expand(-1, -1, F, -1)  # (B, T, F, d_model)
        h = h + self.dec_feature_emb.view(1, 1, F, self.d_model)
        h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        dec_ctx = self.dec_film(x_static).view(B, self.n_dec_layers, 2, self.d_model)

        for i, block in enumerate(self.dec_blocks):
            h, _ = block(h, m_pad)
            gamma = dec_ctx[:, i, 0, :].unsqueeze(1).unsqueeze(1)
            beta = dec_ctx[:, i, 1, :].unsqueeze(1).unsqueeze(1)
            h = gamma * h + beta
            h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        out = self.output_head(h).squeeze(-1)  # (B, T, F)
        out = self.out_dropout(out)
        out = out.masked_fill(m_pad[:, :, None], 0.0)
        return out

    def forward(
        self,
        x_val: torch.Tensor,
        x_miss: torch.Tensor,
        t_abs: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
        return_forecast: bool = False,
    ) -> torch.Tensor:
        """Full forward without retrieval (for pretraining / compatibility).

        Returns (B, T, F) translated features.
        """
        latent = self.encode(x_val, x_miss, t_abs, m_pad, x_static)
        x_out = self.decode(latent, m_pad, x_static)

        if self.output_mode == "residual":
            x_out = x_val + x_out
            x_out = x_out.masked_fill(m_pad.bool()[:, :, None], 0.0)
        return x_out

    def forward_with_retrieval(
        self,
        x_val: torch.Tensor,
        x_miss: torch.Tensor,
        t_abs: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
        context: torch.Tensor,
        latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward with pre-retrieved context.

        Args:
            context: (B, T, K*W, d_latent) from query_memory_bank
            latent: (B, T, d_latent) pre-computed encoder output. If provided,
                    skips encode() to avoid double-encode inconsistency
                    (different dropout masks between auxiliary and task losses).

        Returns:
            x_out: (B, T, F) translated features
            latent: (B, T, d_latent) encoded latents (for logging)
        """
        if latent is None:
            latent = self.encode(x_val, x_miss, t_abs, m_pad, x_static)
        x_decoded = self.decode_with_context(latent, context, m_pad, x_static)

        if self.output_mode == "residual":
            x_out = x_val + x_decoded
            x_out = x_out.masked_fill(m_pad.bool()[:, :, None], 0.0)
        else:
            x_out = x_decoded
        return x_out, latent

    def predict_labels(self, latent: torch.Tensor, m_pad: torch.Tensor) -> torch.Tensor:
        """Predict labels from latent (for pretraining)."""
        logits = self.label_pred_head(latent.float()).squeeze(-1)  # (B, T)
        logits = logits.masked_fill(m_pad.bool(), 0.0)
        return logits

    def set_temporal_mode(self, mode: str) -> None:
        """Switch all blocks between 'causal' and 'bidirectional'."""
        if mode not in {"causal", "bidirectional"}:
            raise ValueError(f"Invalid temporal mode: {mode}")
        causal = mode == "causal"
        for block in self.enc_blocks:
            block.use_causal_temporal_attention = causal
        for block in self.dec_blocks:
            block.use_causal_temporal_attention = causal
        for block in self.cross_blocks:
            block.use_causal_self_attn = causal
        self.temporal_attention_mode = mode

    def _time_encoding(self, t_abs: torch.Tensor) -> torch.Tensor:
        half_dim = self.d_time // 2
        if half_dim == 1:
            freq = torch.ones(1, device=t_abs.device, dtype=t_abs.dtype)
        else:
            freq = torch.exp(
                torch.arange(half_dim, device=t_abs.device, dtype=t_abs.dtype)
                * -(math.log(10000.0) / (half_dim - 1))
            )
        angles = t_abs.unsqueeze(-1) * freq.view(1, 1, half_dim)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
