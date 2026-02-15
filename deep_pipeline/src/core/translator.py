import logging
import math
import os
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class Translator(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
    
    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError


class IdentityTranslator(Translator):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self.input_size = input_size
        # Dummy learnable scalar so optimizer isn't empty
        self._dummy = nn.Parameter(torch.zeros(()))
    
    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        data = batch[0]
        return data + (self._dummy * 0.0)


class LinearRegressionTranslator(Translator):
    def __init__(
        self,
        input_size: int,
        source_feature_indices: Iterable[int],
        dynamic_feature_names: Iterable[str],
        output_feature_names: Iterable[str],
        target_feature_indices: Iterable[int] | None = None,
        source_missing_indicator_indices: Iterable[int | None] | None = None,
        target_missing_indicator_indices: Iterable[int | None] | None = None,
        use_missing_indicator_mask: bool = True,
    ):
        super().__init__(input_size=input_size)
        self.input_size = input_size
        self.source_feature_indices = list(source_feature_indices)
        self.dynamic_feature_names = list(dynamic_feature_names)
        self.target_feature_indices = list(target_feature_indices) if target_feature_indices is not None else None
        self.output_feature_names = list(output_feature_names)
        self.source_missing_indicator_indices = (
            list(source_missing_indicator_indices) if source_missing_indicator_indices is not None else []
        )
        self.target_missing_indicator_indices = (
            list(target_missing_indicator_indices) if target_missing_indicator_indices is not None else []
        )
        self.use_missing_indicator_mask = use_missing_indicator_mask
        self.a = None
        self.b = None
        self.tgt_mean = None
        self.tgt_std = None

    def _init_stats(self, num_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts = np.zeros(num_features, dtype=np.int64)
        sums = np.zeros(num_features, dtype=np.float64)
        sumsq = np.zeros(num_features, dtype=np.float64)
        return counts, sums, sumsq

    def _update_stats(
        self,
        data_np: np.ndarray,
        pad_mask: np.ndarray,
        feature_indices: list[int],
        missing_indicator_indices: list[int | None],
        counts: np.ndarray,
        sums: np.ndarray,
        sumsq: np.ndarray,
    ) -> None:
        for idx, feat_idx in enumerate(feature_indices):
            feature_vals = data_np[:, :, feat_idx]
            observed = pad_mask
            indicator_idx = missing_indicator_indices[idx] if idx < len(missing_indicator_indices) else None
            if indicator_idx is not None and self.use_missing_indicator_mask:
                indicator_vals = data_np[:, :, indicator_idx]
                observed = observed & (indicator_vals <= 0.5)
            if not observed.any():
                continue
            vals = feature_vals[observed]
            counts[idx] += vals.size
            sums[idx] += vals.sum()
            sumsq[idx] += (vals ** 2).sum()

    def _compute_mean_std(
        self,
        loader,
        feature_indices: list[int],
        missing_indicator_indices: list[int | None],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts, sums, sumsq = self._init_stats(len(feature_indices))
        for batch in loader:
            data, _, mask = batch
            data_np = data.detach().cpu().numpy()
            pad_mask = mask.detach().cpu().numpy().astype(bool)
            self._update_stats(data_np, pad_mask, feature_indices, missing_indicator_indices, counts, sums, sumsq)
        means = np.zeros_like(sums)
        stds = np.zeros_like(sums)
        valid = counts > 0
        means[valid] = sums[valid] / counts[valid]
        variances = np.zeros_like(sums)
        variances[valid] = (sumsq[valid] / counts[valid]) - (means[valid] ** 2)
        variances = np.maximum(variances, 0.0)
        stds[valid] = np.sqrt(variances[valid])
        return means, stds, counts

    def fit_from_loaders(self, train_loader, target_loader=None) -> "LinearRegressionTranslator":
        if target_loader is None:
            target_loader = train_loader
        if self.target_feature_indices is None:
            self.target_feature_indices = list(self.source_feature_indices)
        if not self.source_missing_indicator_indices:
            self.source_missing_indicator_indices = [None] * len(self.source_feature_indices)
        if not self.target_missing_indicator_indices:
            self.target_missing_indicator_indices = [None] * len(self.target_feature_indices)

        src_mean, src_std, src_count = self._compute_mean_std(
            train_loader, self.source_feature_indices, self.source_missing_indicator_indices
        )
        tgt_mean, tgt_std, tgt_count = self._compute_mean_std(
            target_loader, self.target_feature_indices, self.target_missing_indicator_indices
        )

        a = np.ones_like(src_mean)
        b = np.zeros_like(src_mean)
        skipped = 0
        zero_var = 0
        for i in range(len(src_mean)):
            if src_count[i] == 0 or tgt_count[i] == 0:
                skipped += 1
                continue
            if src_std[i] == 0 or tgt_std[i] == 0:
                zero_var += 1
                a[i] = 0.0
                b[i] = tgt_mean[i]
            else:
                a[i] = tgt_std[i] / src_std[i]
                b[i] = tgt_mean[i] - a[i] * src_mean[i]
        self.a = a
        self.b = b
        self.tgt_mean = tgt_mean
        self.tgt_std = tgt_std
        logging.info(
            "Fitted per-feature mapping for %d features (skipped=%d, zero_var=%d).",
            len(self.output_feature_names),
            skipped,
            zero_var,
        )
        return self

    def plot_feature_maps(
        self,
        source_loader,
        target_loader,
        output_dir: str | Path,
        max_points: int = 50000,
    ) -> None:
        if self.a is None or self.b is None:
            raise ValueError("Mapping not fitted. Call fit_from_loaders() first.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_features = len(self.source_feature_indices)
        collected_x = [[] for _ in range(num_features)]
        collected_y = [[] for _ in range(num_features)]

        for source_batch, target_batch in zip(source_loader, target_loader):
            src_data = source_batch[0].detach().cpu().numpy()
            tgt_data = target_batch[0].detach().cpu().numpy()
            src_mask = source_batch[2].detach().cpu().numpy().astype(bool)
            tgt_mask = target_batch[2].detach().cpu().numpy().astype(bool)
            for idx in range(num_features):
                if len(collected_x[idx]) >= max_points:
                    continue
                src_feat = src_data[:, :, self.source_feature_indices[idx]]
                tgt_feat = tgt_data[:, :, self.target_feature_indices[idx]]
                observed = src_mask & tgt_mask
                src_ind = (
                    self.source_missing_indicator_indices[idx]
                    if idx < len(self.source_missing_indicator_indices)
                    else None
                )
                tgt_ind = (
                    self.target_missing_indicator_indices[idx]
                    if idx < len(self.target_missing_indicator_indices)
                    else None
                )
                if src_ind is not None and self.use_missing_indicator_mask:
                    observed = observed & (src_data[:, :, src_ind] <= 0.5)
                if tgt_ind is not None and self.use_missing_indicator_mask:
                    observed = observed & (tgt_data[:, :, tgt_ind] <= 0.5)
                if not observed.any():
                    continue
                x_vals = src_feat[observed]
                y_vals = tgt_feat[observed]
                remaining = max_points - len(collected_x[idx])
                if x_vals.size > remaining:
                    x_vals = x_vals[:remaining]
                    y_vals = y_vals[:remaining]
                collected_x[idx].extend(x_vals.tolist())
                collected_y[idx].extend(y_vals.tolist())

        for idx, feature_name in enumerate(self.output_feature_names):
            if not collected_x[idx]:
                continue
            x = np.array(collected_x[idx])
            y = np.array(collected_y[idx])
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x, y, s=4, alpha=0.3)
            x_line = np.linspace(np.min(x), np.max(x), 100)
            y_line = self.a[idx] * x_line + self.b[idx]
            ax.plot(x_line, y_line, color="red", linewidth=1)
            ax.set_title(feature_name)
            ax.set_xlabel("eICU")
            ax.set_ylabel("MIMIC-IV")
            fig.tight_layout()
            fig.savefig(output_dir / f"{feature_name}.png", dpi=150)
            plt.close(fig)

    def translate_batch(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        data = batch[0]
        data_np = data.detach().cpu().numpy()
        batch_size, seq_len, feat_dim = data_np.shape
        if feat_dim != self.input_size:
            logging.warning("LinearRegressionTranslator input_size %d != batch features %d; updating.", self.input_size, feat_dim)
            self.input_size = feat_dim
        if self.a is None or self.b is None:
            raise ValueError("LinearRegressionTranslator not fitted. Call fit_from_loaders() first.")
        if self.tgt_mean is None or self.tgt_std is None:
            raise ValueError("LinearRegressionTranslator missing target stats. Refit the model.")
        output_np = data_np.copy()
        pad_mask = batch[2].detach().cpu().numpy().astype(bool)
        for idx, feat_idx in enumerate(self.source_feature_indices):
            observed = pad_mask
            indicator_idx = None
            if idx < len(self.source_missing_indicator_indices):
                indicator_idx = self.source_missing_indicator_indices[idx]
            if indicator_idx is not None and self.use_missing_indicator_mask:
                indicator_vals = data_np[:, :, indicator_idx]
                observed = observed & (indicator_vals <= 0.5)
            if not observed.any():
                continue
            vals = output_np[:, :, feat_idx]
            vals[observed] = self.a[idx] * vals[observed] + self.b[idx]
            # Normalize to target (MIMIC-IV) stats for all non-padding positions
            if self.tgt_std[idx] > 0:
                vals[pad_mask] = (vals[pad_mask] - self.tgt_mean[idx]) / self.tgt_std[idx]
            else:
                vals[pad_mask] = 0.0
            output_np[:, :, feat_idx] = vals
        output = torch.from_numpy(output_np).to(data.device).to(data.dtype)
        return output

    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.translate_batch(batch)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "input_size": self.input_size,
                "source_feature_indices": self.source_feature_indices,
                "dynamic_feature_names": self.dynamic_feature_names,
                "target_feature_indices": self.target_feature_indices,
                "output_feature_names": self.output_feature_names,
                "source_missing_indicator_indices": self.source_missing_indicator_indices,
                "target_missing_indicator_indices": self.target_missing_indicator_indices,
                "use_missing_indicator_mask": self.use_missing_indicator_mask,
                "a": self.a,
                "b": self.b,
                "tgt_mean": self.tgt_mean,
                "tgt_std": self.tgt_std,
            },
            path,
        )

    def load(self, path: str) -> "LinearRegressionTranslator":
        state = joblib.load(path)
        self.input_size = state.get("input_size", self.input_size)
        self.source_feature_indices = list(state["source_feature_indices"])
        self.dynamic_feature_names = list(state.get("dynamic_feature_names") or [])
        self.target_feature_indices = list(state.get("target_feature_indices") or [])
        self.output_feature_names = list(state["output_feature_names"])
        self.source_missing_indicator_indices = list(state.get("source_missing_indicator_indices") or [])
        self.target_missing_indicator_indices = list(state.get("target_missing_indicator_indices") or [])
        self.use_missing_indicator_mask = state.get("use_missing_indicator_mask", self.use_missing_indicator_mask)
        self.a = state.get("a")
        self.b = state.get("b")
        self.tgt_mean = state.get("tgt_mean")
        self.tgt_std = state.get("tgt_std")
        return self


class AxialBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, d_ff: int, use_causal_temporal_attention: bool,
                 temporal_attention_window: int = 0):
        super().__init__()
        self.var_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temp_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_var = nn.LayerNorm(d_model)
        self.norm_temp = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.use_causal_temporal_attention = use_causal_temporal_attention
        self.temporal_attention_window = temporal_attention_window
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, h: torch.Tensor, m_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_features, d_model = h.shape
        h_var = h.reshape(batch_size * seq_len, num_features, d_model)
        attn_out, _ = self.var_attn(h_var, h_var, h_var, need_weights=False)
        h_var = self.norm_var(h_var + self.dropout(attn_out))
        h = h_var.reshape(batch_size, seq_len, num_features, d_model)

        h_temp = h.permute(0, 2, 1, 3).reshape(batch_size * num_features, seq_len, d_model).contiguous()
        key_padding_mask = m_pad.unsqueeze(1).expand(batch_size, num_features, seq_len).reshape(batch_size * num_features, seq_len).contiguous()
        if os.environ.get("YAIB_TRANSLATOR_DEBUG") == "1":
            if key_padding_mask.dtype is not torch.bool:
                raise RuntimeError(f"key_padding_mask dtype {key_padding_mask.dtype} (expected bool)")
            if key_padding_mask.shape != (batch_size * num_features, seq_len):
                raise RuntimeError(
                    f"key_padding_mask shape {tuple(key_padding_mask.shape)} "
                    f"(expected {(batch_size * num_features, seq_len)})"
                )
            if key_padding_mask.device != h_temp.device:
                raise RuntimeError("key_padding_mask device mismatch with h_temp")
            if not torch.isfinite(h_temp).all():
                bad = (~torch.isfinite(h_temp)).any().item()
                raise RuntimeError(f"h_temp has non-finite values: {bad}")
            if not torch.isfinite(h).all():
                bad = (~torch.isfinite(h)).any().item()
                raise RuntimeError(f"h has non-finite values: {bad}")
        # Guard against all-padded rows (can trigger CUDA kernel faults in some MHA kernels).
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            h_temp[all_pad] = 0
            key_padding_mask[all_pad, 0] = False
            if os.environ.get("YAIB_TRANSLATOR_DEBUG") == "1":
                logging.warning("All-padded sequences in temporal attention: %d", int(all_pad.sum().item()))
        temporal_attn_mask = None
        if self.use_causal_temporal_attention:
            temporal_attn_mask = torch.ones((seq_len, seq_len), device=h_temp.device, dtype=torch.bool).triu_(1)
            if self.temporal_attention_window > 0:
                # Block positions more than W-1 steps in the past
                temporal_attn_mask |= torch.ones((seq_len, seq_len), device=h_temp.device, dtype=torch.bool).tril_(-self.temporal_attention_window)
        attn_out, _ = self.temp_attn(
            h_temp,
            h_temp,
            h_temp,
            attn_mask=temporal_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h_temp = self.norm_temp(h_temp + self.dropout(attn_out))
        h = h_temp.reshape(batch_size, num_features, seq_len, d_model).permute(0, 2, 1, 3)

        h_ffn = self.ffn(h)
        h = self.norm_ffn(h + self.dropout(h_ffn))
        return h, key_padding_mask


class EHRTranslator(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_latent: int = 16,
        d_model: int = 128,
        d_time: int = 16,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.2,
        out_dropout: float = 0.1,
        static_dim: int = 4,
        temporal_attention_mode: str = "bidirectional",
        temporal_attention_window: int = 0,
    ):
        super().__init__()
        if d_time % 2 != 0:
            raise ValueError("d_time must be even for sin/cos encoding")
        self.num_features = num_features
        self.d_latent = d_latent
        self.d_model = d_model
        self.d_time = d_time
        self.n_layers = n_layers
        self.temporal_attention_window = temporal_attention_window
        if temporal_attention_mode not in {"bidirectional", "causal"}:
            raise ValueError(
                f"Unsupported temporal_attention_mode '{temporal_attention_mode}'. "
                "Expected one of: 'bidirectional', 'causal'."
            )
        self.temporal_attention_mode = temporal_attention_mode
        use_causal_temporal_attention = temporal_attention_mode == "causal"

        self.triplet_proj = nn.Linear(3, d_latent)
        self.sensor_emb = nn.Parameter(torch.zeros(num_features, d_latent))
        nn.init.normal_(self.sensor_emb, mean=0.0, std=0.02)
        self.lift = nn.Linear(d_latent, d_model)
        self.time_proj = nn.Linear(d_time, d_model)

        self.blocks = nn.ModuleList(
            [
                AxialBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    d_ff=d_ff,
                    use_causal_temporal_attention=use_causal_temporal_attention,
                    temporal_attention_window=temporal_attention_window,
                )
                for _ in range(n_layers)
            ]
        )

        self.film_mlp = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_layers * d_model),
        )

        self.delta_head = nn.Linear(d_model, 1)
        self.forecast_head = nn.Linear(d_model, 1)
        self.out_dropout = nn.Dropout(out_dropout)
        self._last_temporal_key_padding_mask = None

        # MLM pretraining components (lazy-initialized)
        self.mask_embedding: nn.Parameter | None = None
        self.reconstruction_head: nn.Linear | None = None

    def forward(
        self,
        x_val: torch.Tensor,
        x_miss: torch.Tensor,
        t_abs: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
        return_forecast: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        m_pad = m_pad.bool()
        if x_val.shape[-1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {x_val.shape[-1]}")
        if x_miss.shape != x_val.shape:
            raise ValueError(f"x_miss shape {tuple(x_miss.shape)} does not match x_val {tuple(x_val.shape)}")
        if os.environ.get("YAIB_TRANSLATOR_DEBUG") == "1":
            if m_pad.shape != x_val.shape[:2]:
                raise RuntimeError(f"M_pad shape {tuple(m_pad.shape)} does not match (B,T) {tuple(x_val.shape[:2])}")
            if not torch.isfinite(x_val).all():
                raise RuntimeError("x_val contains non-finite values")
            if not torch.isfinite(x_miss).all():
                raise RuntimeError("x_miss contains non-finite values")
            if not torch.isfinite(t_abs).all():
                raise RuntimeError("t_abs contains non-finite values")
        t_abs = t_abs.to(dtype=x_val.dtype)
        time_delta = torch.zeros_like(t_abs)
        time_delta[:, 1:] = t_abs[:, 1:] - t_abs[:, :-1]
        time_delta = time_delta.masked_fill(m_pad, 0.0)

        time_delta_feat = time_delta.unsqueeze(-1).expand(-1, -1, self.num_features)
        x_trip = torch.stack([x_val, x_miss, time_delta_feat], dim=-1)
        h = self.triplet_proj(x_trip)
        h = h + self.sensor_emb.view(1, 1, self.num_features, self.d_latent)
        h = self.lift(h)

        time_enc = self._time_encoding(t_abs)
        time_enc = self.time_proj(time_enc)
        h = h + time_enc[:, :, None, :]
        h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        context = self.film_mlp(x_static)
        context = context.view(x_static.shape[0], self.n_layers, 2, self.d_model)

        for layer_idx, block in enumerate(self.blocks):
            h, key_padding_mask = block(h, m_pad)
            gamma = context[:, layer_idx, 0, :].unsqueeze(1).unsqueeze(1)
            beta = context[:, layer_idx, 1, :].unsqueeze(1).unsqueeze(1)
            h = gamma * h + beta
            h = h.masked_fill(m_pad[:, :, None, None], 0.0)
            self._last_temporal_key_padding_mask = key_padding_mask

        delta = self.delta_head(h).squeeze(-1)
        delta = self.out_dropout(delta)
        x_val_out = x_val + delta
        x_val_out = x_val_out.masked_fill(m_pad[:, :, None], 0.0)

        if not return_forecast:
            return x_val_out

        x_forecast = self.forecast_head(h).squeeze(-1)   # (B, T, F)
        x_forecast = x_forecast.masked_fill(m_pad[:, :, None], 0.0)
        return x_val_out, x_forecast

    def set_temporal_mode(self, mode: str) -> None:
        """Switch all AxialBlocks between 'causal' and 'bidirectional' attention."""
        if mode not in {"causal", "bidirectional"}:
            raise ValueError(f"Invalid temporal mode: {mode}")
        causal = mode == "causal"
        for block in self.blocks:
            block.use_causal_temporal_attention = causal
        self.temporal_attention_mode = mode

    def init_mlm_head(self) -> None:
        """Initialize MLM-specific parameters (mask embedding + reconstruction head)."""
        if self.mask_embedding is None:
            self.mask_embedding = nn.Parameter(torch.zeros(self.d_model))
            nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)
        if self.reconstruction_head is None:
            self.reconstruction_head = nn.Linear(self.d_model, 1)

    def discard_mlm_head(self) -> None:
        """Free MLM-specific parameters after pretraining."""
        self.mask_embedding = None
        self.reconstruction_head = None

    def forward_mlm(
        self,
        x_val: torch.Tensor,
        x_miss: torch.Tensor,
        t_abs: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
        mlm_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for MLM pretraining. Returns reconstructed feature values.

        Args:
            x_val: (B, T, F) feature values (masked timesteps zeroed out).
            x_miss: (B, T, F) missingness indicators.
            t_abs: (B, T) absolute timestamps.
            m_pad: (B, T) padding mask.
            x_static: (B, S) static features.
            mlm_mask: (B, T) bool — True for timesteps to reconstruct.

        Returns:
            x_reconstructed: (B, T, F) reconstructed feature values.
        """
        m_pad = m_pad.bool()
        t_abs = t_abs.to(dtype=x_val.dtype)
        time_delta = torch.zeros_like(t_abs)
        time_delta[:, 1:] = t_abs[:, 1:] - t_abs[:, :-1]
        time_delta = time_delta.masked_fill(m_pad, 0.0)

        time_delta_feat = time_delta.unsqueeze(-1).expand(-1, -1, self.num_features)
        x_trip = torch.stack([x_val, x_miss, time_delta_feat], dim=-1)
        h = self.triplet_proj(x_trip)
        h = h + self.sensor_emb.view(1, 1, self.num_features, self.d_latent)
        h = self.lift(h)

        # Add mask embedding at masked positions (broadcast across features)
        if self.mask_embedding is not None:
            mask_expand = mlm_mask[:, :, None, None].expand_as(h)
            h = h + mask_expand * self.mask_embedding.view(1, 1, 1, self.d_model)

        time_enc = self._time_encoding(t_abs)
        time_enc = self.time_proj(time_enc)
        h = h + time_enc[:, :, None, :]
        h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        context = self.film_mlp(x_static)
        context = context.view(x_static.shape[0], self.n_layers, 2, self.d_model)

        for layer_idx, block in enumerate(self.blocks):
            h, _ = block(h, m_pad)
            gamma = context[:, layer_idx, 0, :].unsqueeze(1).unsqueeze(1)
            beta = context[:, layer_idx, 1, :].unsqueeze(1).unsqueeze(1)
            h = gamma * h + beta
            h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        x_reconstructed = self.reconstruction_head(h).squeeze(-1)  # (B, T, F)
        x_reconstructed = x_reconstructed.masked_fill(m_pad[:, :, None], 0.0)
        return x_reconstructed

    def _time_encoding(self, t_abs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = t_abs.shape
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
