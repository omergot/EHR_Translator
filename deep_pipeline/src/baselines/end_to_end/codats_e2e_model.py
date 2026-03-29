"""CoDATS: Convolutional Deep Domain Adaptation for Time Series (Wilson et al., 2020).

End-to-end implementation with ORIGINAL architecture (not our translator backbone).

Architecture:
- Encoder: 3-layer 1D CNN with CAUSAL (left-only) padding
  - Conv1d(in, 128, k=5) + BN + ReLU
  - Conv1d(128, 256, k=5) + BN + ReLU
  - Conv1d(256, 128, k=3) + BN + ReLU
  - NO MaxPool (preserves temporal resolution for per-timestep prediction)
- Task classifier: Linear(128, 1) applied per-timestep for AKI/sepsis, pooled for mortality
- Domain discriminator: GRL + MLP(128, 256, 1) applied per-timestep

Training losses:
- Source classification: per-timestep masked BCE (AKI/sepsis) or per-stay BCE (mortality)
- Domain adversarial: per-timestep BCE on both domains

Key design choice: NO MaxPool. The original CoDATS uses MaxPool because it does
per-segment classification. For per-timestep prediction (AKI/sepsis), MaxPool
would halve temporal resolution. Without MaxPool, the CNN produces (B, 128, L)
at the same length as input.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW

from .trainer_base import E2EBaselineTrainer
from ..components import GradientReversalLayer, DomainDiscriminator, grl_lambda_schedule


class CoDATS1DCNNCausal(nn.Module):
    """Causal 1D CNN encoder (no MaxPool, left-only padding).

    Uses F.pad(x, (kernel_size-1, 0)) before each Conv1d(padding=0)
    to ensure each output timestep only depends on current and past inputs.

    Input: (B, C, L)
    Output: (B, H, L) per-timestep, or (B, H) pooled
    """

    def __init__(self, num_inputs: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Layer 1: (num_inputs) -> 128, kernel=5
        self.conv1 = nn.Conv1d(num_inputs, hidden_dim, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Layer 2: 128 -> 256, kernel=5
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)

        # Layer 3: 256 -> 128, kernel=3
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, return_sequence: bool = True) -> torch.Tensor:
        """Encode with causal padding.

        Args:
            x: (B, C, L) channels-first input.
            return_sequence: If True, return (B, H, L). If False, return (B, H).

        Returns:
            (B, H, L) per-timestep features or (B, H) pooled.
        """
        # Layer 1: causal pad (k-1=4 on left, 0 on right)
        h = F.pad(x, (4, 0))
        h = self.dropout(F.relu(self.bn1(self.conv1(h))))

        # Layer 2: causal pad (k-1=4 on left, 0 on right)
        h = F.pad(h, (4, 0))
        h = self.dropout(F.relu(self.bn2(self.conv2(h))))

        # Layer 3: causal pad (k-1=2 on left, 0 on right)
        h = F.pad(h, (2, 0))
        h = self.dropout(F.relu(self.bn3(self.conv3(h))))

        # h: (B, hidden_dim, L) — same temporal length as input

        if return_sequence:
            return h  # (B, H, L)
        else:
            return h.mean(dim=2)  # Global average pooling -> (B, H)


class CoDATSModel(nn.Module):
    """CoDATS: Causal CNN encoder + classifier + GRL discriminator.

    For per-timestep tasks (AKI, sepsis): classifier and discriminator operate
    at each timestep independently.
    For per-stay tasks (mortality): pool encoder output before classifier.
    """

    def __init__(self, config: dict):
        super().__init__()
        training = config.get("training", {})

        self.num_inputs = training.get("num_input_channels", 96)
        self.hidden_size = training.get("hidden_size", 128)
        dropout = training.get("dropout", 0.2)
        num_static = 4

        # CNN encoder
        self.encoder = CoDATS1DCNNCausal(
            num_inputs=self.num_inputs,
            hidden_dim=self.hidden_size,
            dropout=dropout,
        )

        # Static feature projection
        self.static_proj = nn.Linear(num_static, self.hidden_size)

        # Task classifier: Linear(H, 1)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )

        # Domain discriminator: GRL + MLP
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.discriminator = DomainDiscriminator(
            input_dim=self.hidden_size,
            hidden_dim=256,
            n_layers=3,
            dropout=0.3,
        )

        logging.info(
            "[CoDATS-E2E] inputs=%d hidden=%d dropout=%.2f (causal CNN, no MaxPool)",
            self.num_inputs, self.hidden_size, dropout,
        )

    def encode(self, x: torch.Tensor, static: torch.Tensor,
               return_sequence: bool = True) -> torch.Tensor:
        """Encode input.

        Args:
            x: (B, C, L) channels-first.
            static: (B, S) static features.
            return_sequence: If True, return (B, L, H). If False, return (B, H).

        Returns:
            (B, L, H) or (B, H).
        """
        if return_sequence:
            h_seq = self.encoder(x, return_sequence=True)  # (B, H, L)
            h_seq = h_seq.transpose(1, 2)  # (B, L, H)
            # Add static context
            static_emb = self.static_proj(static)  # (B, H)
            h_seq = h_seq + static_emb.unsqueeze(1)  # broadcast
            return h_seq
        else:
            h = self.encoder(x, return_sequence=False)  # (B, H)
            static_emb = self.static_proj(static)  # (B, H)
            return h + static_emb

    def predict(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Per-stay prediction (mortality). x: (B, C, L), static: (B, S) -> (B,)."""
        h = self.encode(x, static, return_sequence=False)  # (B, H)
        return self.classifier(h).squeeze(-1)  # (B,)

    def predict_per_timestep(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Per-timestep prediction (AKI/sepsis). x: (B, C, L), static: (B, S) -> (B, L)."""
        h_seq = self.encode(x, static, return_sequence=True)  # (B, L, H)
        logits = self.classifier(h_seq)  # (B, L, 1)
        return logits.squeeze(-1)  # (B, L)


class CoDATSTrainer(E2EBaselineTrainer):
    """Trainer for CoDATS end-to-end baseline."""

    def __init__(self, model: CoDATSModel, source_train_loader, target_train_loader,
                 source_val_loader, config, device="cuda", target_val_loader=None):
        super().__init__(model, source_train_loader, target_train_loader,
                         source_val_loader, config, device, target_val_loader=target_val_loader)

        training = config.get("training", {})
        self.lambda_adversarial = training.get("lambda_adversarial", 1.0)
        self.lambda_cls = training.get("lambda_cls", 1.0)

        # Single optimizer
        self.optimizer = AdamW(
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

        # Progressive GRL lambda (Ganin et al. 2016, also used in CoDATS)
        grl_lam = grl_lambda_schedule(epoch, self.epochs)
        model.grl.set_lambda(grl_lam)

        total_metrics = {
            "cls_loss": 0.0, "adv_loss": 0.0, "total_loss": 0.0, "grl_lam": grl_lam,
        }
        n_batches = 0

        for src_batch in self.source_train_loader:
            src_x = src_batch[0].to(self.device)
            src_y = src_batch[1].to(self.device)
            src_static = src_batch[2].to(self.device)
            src_vmask = src_batch[3].to(self.device)

            tgt_batch = self._get_target_batch()
            tgt_x = tgt_batch[0].to(self.device)
            tgt_static = tgt_batch[2].to(self.device)
            tgt_vmask = tgt_batch[3].to(self.device)

            with autocast(enabled=self.use_amp):
                # === Classification loss (source only) ===
                if src_y.dim() > 1:
                    # Per-timestep: AKI / sepsis
                    src_h_seq = model.encode(src_x, src_static, return_sequence=True)  # (B, L, H)
                    logits_ts = model.classifier(src_h_seq).squeeze(-1)  # (B, L)
                    valid = src_vmask & (src_y >= 0)
                    if valid.sum() > 0:
                        cls_loss = F.binary_cross_entropy_with_logits(
                            logits_ts[valid], src_y[valid],
                            pos_weight=self._pos_weight,
                        )
                    else:
                        cls_loss = logits_ts.new_tensor(0.0)

                    # === Domain adversarial loss (per-timestep) ===
                    tgt_h_seq = model.encode(tgt_x, tgt_static, return_sequence=True)  # (B, L, H)

                    # Flatten valid timesteps
                    src_h_valid = src_h_seq[src_vmask].float()  # (N_s, H)
                    tgt_h_valid = tgt_h_seq[tgt_vmask].float()  # (N_t, H)

                    if src_h_valid.size(0) > 0 and tgt_h_valid.size(0) > 0:
                        h_all = torch.cat([src_h_valid, tgt_h_valid], dim=0)
                        h_rev = model.grl(h_all)
                        d_logits = model.discriminator(h_rev).squeeze(-1)
                        d_labels = torch.cat([
                            torch.zeros(src_h_valid.size(0), device=self.device),
                            torch.ones(tgt_h_valid.size(0), device=self.device),
                        ])
                        adv_loss = F.binary_cross_entropy_with_logits(d_logits, d_labels)
                    else:
                        adv_loss = src_x.new_tensor(0.0)
                else:
                    # Per-stay: mortality
                    src_h = model.encode(src_x, src_static, return_sequence=False)  # (B, H)
                    logits = model.classifier(src_h).squeeze(-1)  # (B,)
                    cls_loss = self.classification_loss(logits, src_y)

                    # Domain adversarial (pooled)
                    tgt_h = model.encode(tgt_x, tgt_static, return_sequence=False)  # (B, H)
                    h_all = torch.cat([src_h, tgt_h], dim=0).float()
                    h_rev = model.grl(h_all)
                    d_logits = model.discriminator(h_rev).squeeze(-1)
                    d_labels = torch.cat([
                        torch.zeros(src_h.size(0), device=self.device),
                        torch.ones(tgt_h.size(0), device=self.device),
                    ])
                    adv_loss = F.binary_cross_entropy_with_logits(d_logits, d_labels)

                # Total loss
                loss = self.lambda_cls * cls_loss + self.lambda_adversarial * adv_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_metrics["cls_loss"] += cls_loss.item()
            total_metrics["adv_loss"] += adv_loss.item()
            total_metrics["total_loss"] += loss.item()
            n_batches += 1

        for k in total_metrics:
            if k != "grl_lam":
                total_metrics[k] /= max(n_batches, 1)

        return total_metrics
