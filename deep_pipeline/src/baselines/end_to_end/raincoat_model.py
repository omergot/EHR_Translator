"""RAINCOAT: Domain Adaptation for Time Series (He et al., ICML 2023).

End-to-end implementation based on https://github.com/mims-harvard/Raincoat

Architecture:
- Temporal encoder: 3-layer CNN (Conv1d -> BN -> ReLU -> MaxPool)
- Frequency encoder: SpectralConv1d (FFT -> learnable complex weights -> IFFT -> amplitude/phase)
- Decoder: Transpose CNN for reconstruction
- Classifier: Linear (temporal_dim + freq_dim) -> 1

Training losses:
- Classification: weighted BCE on source (weight 0.5)
- Sinkhorn alignment: on concatenated temporal+frequency features (weight 0.5)
- Reconstruction: L1 loss (weight 1e-4)
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .trainer_base import E2EBaselineTrainer
from ..components import SinkhornDivergence


# ---------------------------------------------------------------------------
# Temporal encoder (CNN)
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """3-layer 1D CNN encoder for temporal features.

    Input: (B, C_in, L)
    Output: (B, H)  where H = temporal_dim
    """

    def __init__(self, num_inputs: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_inputs, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (global_features, sequence_features).

        global_features: (B, H) for classification
        sequence_features: (B, H, L') for reconstruction
        """
        h = self.dropout(F.relu(self.bn1(self.conv1(x))))
        h = F.max_pool1d(h, 2)
        h = self.dropout(F.relu(self.bn2(self.conv2(h))))
        h = F.max_pool1d(h, 2)
        h = self.dropout(F.relu(self.bn3(self.conv3(h))))
        seq_features = h  # (B, H, L')
        global_features = self.pool(h).squeeze(-1)  # (B, H)
        return global_features, seq_features


# ---------------------------------------------------------------------------
# Frequency encoder (SpectralConv)
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """Learnable spectral convolution via FFT.

    Applies learnable complex weights in the frequency domain.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, L) -> (B, C_out, L)."""
        B, C, L = x.shape
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, L//2+1)
        n_freq = x_ft.shape[-1]
        modes = min(self.modes, n_freq)

        # Apply learnable complex weights to first `modes` frequencies
        weights = torch.complex(
            self.weights_real[:, :, :modes],
            self.weights_imag[:, :, :modes],
        )  # (C_in, C_out, modes)

        out_ft = torch.zeros(B, self.out_channels, n_freq,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :modes] = torch.einsum("bcf,cof->bof", x_ft[:, :, :modes], weights)

        # IFFT
        out = torch.fft.irfft(out_ft, n=L, dim=-1)  # (B, C_out, L)
        return out


class FrequencyEncoder(nn.Module):
    """Frequency-domain encoder using learnable spectral convolutions.

    Input: (B, C_in, L)
    Output: (B, H_freq)
    """

    def __init__(self, num_inputs: int, hidden_dim: int = 64, modes: int = 16):
        super().__init__()
        self.spec_conv1 = SpectralConv1d(num_inputs, hidden_dim, modes)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.spec_conv2 = SpectralConv1d(hidden_dim, hidden_dim, modes)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, H_freq)."""
        h = F.relu(self.bn1(self.spec_conv1(x)))
        h = F.relu(self.bn2(self.spec_conv2(h)))
        return self.pool(h).squeeze(-1)  # (B, H_freq)


# ---------------------------------------------------------------------------
# Decoder (reconstruction)
# ---------------------------------------------------------------------------

class TemporalDecoder(nn.Module):
    """Transpose CNN decoder for reconstruction.

    Reconstructs input from temporal encoder's sequence features.
    """

    def __init__(self, hidden_dim: int, output_channels: int, target_len: int):
        super().__init__()
        self.target_len = target_len
        self.deconv1 = nn.ConvTranspose1d(hidden_dim, hidden_dim * 2, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.deconv2 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.deconv3 = nn.ConvTranspose1d(hidden_dim, output_channels, kernel_size=3,
                                           padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, H, L') -> (B, C_out, target_len)."""
        x = F.relu(self.bn1(self.deconv1(h)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        # Interpolate to target length
        if x.size(2) != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# RAINCOAT model
# ---------------------------------------------------------------------------

class RAINCOATModel(nn.Module):
    """RAINCOAT end-to-end model.

    Components:
    - Temporal encoder (CNN)
    - Frequency encoder (SpectralConv)
    - Decoder (Transpose CNN)
    - Classifier (Linear)
    """

    def __init__(self, config: dict):
        super().__init__()
        training = config.get("training", {})

        self.num_inputs = training.get("num_input_channels", 96)
        hidden_dim = training.get("hidden_dim", 64)
        freq_modes = training.get("freq_modes", 16)
        self.seq_len = training.get("seq_len", 48)
        num_static = 4

        self.temporal_encoder = TemporalEncoder(self.num_inputs, hidden_dim)
        self.frequency_encoder = FrequencyEncoder(self.num_inputs, hidden_dim, freq_modes)

        # Decoder: reconstructs from temporal features
        self.decoder = TemporalDecoder(hidden_dim, self.num_inputs, self.seq_len)

        # Classifier: temporal + frequency + static -> logit
        feat_dim = self.temporal_encoder.out_dim + self.frequency_encoder.out_dim + num_static
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        logging.info(
            "[RAINCOAT] inputs=%d hidden=%d modes=%d seq_len=%d feat_dim=%d",
            self.num_inputs, hidden_dim, freq_modes, self.seq_len, feat_dim,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input into temporal and frequency features.

        Returns:
            t_feat: (B, H_t) temporal features
            f_feat: (B, H_f) frequency features
            t_seq: (B, H_t, L') temporal sequence features (for decoder)
        """
        t_feat, t_seq = self.temporal_encoder(x)
        f_feat = self.frequency_encoder(x)
        return t_feat, f_feat, t_seq

    def decode(self, t_seq: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from temporal sequence features.

        Returns: (B, C, L) reconstructed input.
        """
        return self.decoder(t_seq)

    def predict(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Predict logits. x: (B, C, L), static: (B, S) -> (B,)."""
        t_feat, f_feat, _ = self.encode(x)
        combined = torch.cat([t_feat, f_feat, static], dim=1)
        return self.classifier(combined).squeeze(-1)

    def get_alignment_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get concatenated temporal+frequency features for Sinkhorn alignment."""
        t_feat, f_feat, _ = self.encode(x)
        return torch.cat([t_feat, f_feat], dim=1)


class RAINCOATTrainer(E2EBaselineTrainer):
    """Trainer for RAINCOAT end-to-end baseline."""

    def __init__(self, model: RAINCOATModel, source_train_loader, target_train_loader,
                 source_val_loader, config, device="cuda"):
        super().__init__(model, source_train_loader, target_train_loader,
                         source_val_loader, config, device)

        training = config.get("training", {})
        self.lambda_cls = training.get("lambda_cls", 0.5)
        self.lambda_align = training.get("lambda_align", 0.5)
        self.lambda_recon = training.get("lambda_recon", 1e-4)
        sinkhorn_eps = training.get("sinkhorn_eps", 0.1)
        sinkhorn_iters = training.get("sinkhorn_iters", 50)

        # Use the FIXED SinkhornDivergence from components.py (with L2 normalization)
        self.sinkhorn = SinkhornDivergence(
            eps=sinkhorn_eps, max_iters=sinkhorn_iters, max_samples=4096,
        )

        # Single optimizer for all parameters
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        self._target_iter = iter(target_train_loader)

    def _get_target_batch(self):
        try:
            return next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self.target_train_loader)
            return next(self._target_iter)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        model = self.model

        total_metrics = {
            "cls_loss": 0.0, "align_loss": 0.0,
            "recon_loss": 0.0, "total_loss": 0.0,
        }
        n_batches = 0

        for src_x, src_y, src_static in self.source_train_loader:
            tgt_x, _, tgt_static = self._get_target_batch()

            src_x = src_x.to(self.device)
            src_y = src_y.to(self.device)
            src_static = src_static.to(self.device)
            tgt_x = tgt_x.to(self.device)

            with autocast(enabled=self.use_amp):
                # Encode source
                src_t, src_f, src_seq = model.encode(src_x)
                # Encode target
                tgt_t, tgt_f, tgt_seq = model.encode(tgt_x)

                # Classification loss (source only)
                combined_src = torch.cat([src_t, src_f, src_static], dim=1)
                logits = model.classifier(combined_src).squeeze(-1)
                cls_loss = self.classification_loss(logits, src_y)

                # Sinkhorn alignment on concatenated temporal+frequency features
                src_align = torch.cat([src_t, src_f], dim=1).float()
                tgt_align = torch.cat([tgt_t, tgt_f], dim=1).float().detach()
                align_loss = self.sinkhorn(src_align, tgt_align)

                # Reconstruction loss (both domains)
                src_recon = model.decode(src_seq)
                tgt_recon = model.decode(tgt_seq)
                recon_loss = (
                    F.l1_loss(src_recon, src_x) +
                    F.l1_loss(tgt_recon, tgt_x)
                ) * 0.5

                # Total
                loss = (
                    self.lambda_cls * cls_loss +
                    self.lambda_align * align_loss +
                    self.lambda_recon * recon_loss
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_metrics["cls_loss"] += cls_loss.item()
            total_metrics["align_loss"] += align_loss.item()
            total_metrics["recon_loss"] += recon_loss.item()
            total_metrics["total_loss"] += loss.item()
            n_batches += 1

        for k in total_metrics:
            total_metrics[k] /= max(n_batches, 1)

        return total_metrics
