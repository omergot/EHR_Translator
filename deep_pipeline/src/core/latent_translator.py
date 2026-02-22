"""Shared Latent Space Translator.

Maps both eICU (source) and MIMIC (target) into a shared latent space via a shared
encoder, then decodes from that space to produce MIMIC-like features.

Architecture:
    Encoder: triplet_proj → AxialBlocks → pool over features → latent (B, T, d_latent)
    Decoder: broadcast to features → AxialBlocks → output head (B, T, F)
"""

import logging
import math
import os

import torch
import torch.nn as nn

from src.core.translator import AxialBlock


class SharedLatentTranslator(nn.Module):
    """Encoder-decoder translator with shared latent bottleneck."""

    def __init__(
        self,
        num_features: int,
        d_latent: int = 64,
        d_model: int = 128,
        d_time: int = 16,
        n_enc_layers: int = 3,
        n_dec_layers: int = 2,
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
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.temporal_attention_window = temporal_attention_window
        self.temporal_attention_mode = temporal_attention_mode
        use_causal = temporal_attention_mode == "causal"

        # ── Encoder ──────────────────────────────────────────────
        # Triplet projection: (value, missingness, time_delta) → d_latent_emb
        self.triplet_proj = nn.Linear(3, 16)
        self.sensor_emb = nn.Parameter(torch.zeros(num_features, 16))
        nn.init.normal_(self.sensor_emb, mean=0.0, std=0.02)
        self.lift = nn.Linear(16, d_model)
        self.time_proj = nn.Linear(d_time, d_model)

        self.enc_blocks = nn.ModuleList([
            AxialBlock(d_model, n_heads, dropout, d_ff, use_causal, temporal_attention_window)
            for _ in range(n_enc_layers)
        ])

        # FiLM for encoder
        self.enc_film = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_enc_layers * d_model),
        )

        # Project from (B, T, F, d_model) → (B, T, d_latent)
        # Pool over features, then project
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.LayerNorm(d_latent),
        )

        # ── Decoder ──────────────────────────────────────────────
        # Project from (B, T, d_latent) → (B, T, F, d_model)
        self.from_latent = nn.Linear(d_latent, d_model)
        self.dec_feature_emb = nn.Parameter(torch.zeros(num_features, d_model))
        nn.init.normal_(self.dec_feature_emb, mean=0.0, std=0.02)

        self.dec_blocks = nn.ModuleList([
            AxialBlock(d_model, n_heads, dropout, d_ff, use_causal, temporal_attention_window)
            for _ in range(n_dec_layers)
        ])

        # FiLM for decoder
        self.dec_film = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_dec_layers * d_model),
        )

        self.output_head = nn.Linear(d_model, 1)
        self.out_dropout = nn.Dropout(out_dropout)

        # Label prediction head: latent → logits (bypasses frozen LSTM)
        self.label_pred_head = nn.Sequential(
            nn.Linear(d_latent, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def encode(
        self,
        x_val: torch.Tensor,
        x_miss: torch.Tensor,
        t_abs: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Encode features to shared latent space.

        Returns: (B, T, d_latent) latent representation per timestep.
        """
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
        time_enc = self.time_proj(time_enc)  # (B, T, d_model)
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
        # Use mean over non-padded features
        h_pooled = h.mean(dim=2)  # (B, T, d_model)
        latent = self.to_latent(h_pooled)  # (B, T, d_latent)
        latent = latent.masked_fill(m_pad[:, :, None], 0.0)
        return latent

    def decode(
        self,
        latent: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Decode from latent space to feature space.

        Args:
            latent: (B, T, d_latent) per-timestep latent
            m_pad: (B, T) padding mask
            x_static: (B, S) static features

        Returns: (B, T, F) translated feature values.
        """
        m_pad = m_pad.bool()
        B, T, _ = latent.shape
        F = self.num_features

        # Project latent → (B, T, d_model) then broadcast to features
        h = self.from_latent(latent)  # (B, T, d_model)
        h = h.unsqueeze(2).expand(-1, -1, F, -1)  # (B, T, F, d_model)

        # Add per-feature embedding so decoder can differentiate features
        h = h + self.dec_feature_emb.view(1, 1, F, self.d_model)
        h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        # FiLM context for decoder
        ctx = self.dec_film(x_static).view(B, self.n_dec_layers, 2, self.d_model)

        # Decoder blocks
        for i, block in enumerate(self.dec_blocks):
            h, _ = block(h, m_pad)
            gamma = ctx[:, i, 0, :].unsqueeze(1).unsqueeze(1)
            beta = ctx[:, i, 1, :].unsqueeze(1).unsqueeze(1)
            h = gamma * h + beta
            h = h.masked_fill(m_pad[:, :, None, None], 0.0)

        # Output: per-feature value
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
        """Full forward: encode → decode.

        Returns translated feature values (B, T, F).
        Compatible with EHRTranslator interface.
        """
        latent = self.encode(x_val, x_miss, t_abs, m_pad, x_static)
        x_out = self.decode(latent, m_pad, x_static)
        return x_out

    def predict_labels(self, latent: torch.Tensor, m_pad: torch.Tensor) -> torch.Tensor:
        """Predict labels from latent representation.

        Args:
            latent: (B, T, d_latent) per-timestep latent
            m_pad: (B, T) padding mask

        Returns: (B, T) logits for binary classification.
        """
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
