"""Deep CORAL: Correlation Alignment (Sun & Saenko, ECCV 2016).

End-to-end implementation with ORIGINAL architecture (not our translator backbone).

Architecture:
- Encoder: 2-layer LSTM (same as DANN for fair comparison)
  - input_size = num_features (96/100), hidden_size = 128, num_layers = 2
- Task classifier: Linear(128, 1) applied per-timestep for AKI/sepsis, pooled for mortality
- CORAL loss: Second-order covariance alignment on LSTM hidden states

Training losses:
- Source classification: per-timestep masked BCE (AKI/sepsis) or per-stay BCE (mortality)
- CORAL loss: covariance matching between source and target hidden states
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW

from .trainer_base import E2EBaselineTrainer
from .dann_e2e_model import DANNEncoder
from ..components import coral_loss


class CORALModel(nn.Module):
    """Deep CORAL: LSTM encoder + classifier + covariance alignment.

    Same LSTM encoder as DANN for fair comparison. No GRL or discriminator.
    Alignment is achieved through CORAL loss on encoder hidden states.
    """

    def __init__(self, config: dict):
        super().__init__()
        training = config.get("training", {})

        self.num_inputs = training.get("num_input_channels", 96)
        self.hidden_size = training.get("hidden_size", 128)
        num_layers = training.get("num_lstm_layers", 2)
        dropout = training.get("dropout", 0.2)

        # Same encoder architecture as DANN
        self.encoder = DANNEncoder(
            input_size=self.num_inputs,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Task classifier: Linear(H, 1)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )

        logging.info(
            "[CORAL-E2E] inputs=%d hidden=%d lstm_layers=%d dropout=%.2f",
            self.num_inputs, self.hidden_size, num_layers, dropout,
        )

    def encode(self, x: torch.Tensor, static: torch.Tensor,
               return_sequence: bool = True) -> torch.Tensor:
        """Encode input. Returns (B, L, H) or (B, H)."""
        return self.encoder(x, static, return_sequence=return_sequence)

    def predict(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Per-stay prediction (mortality). x: (B, C, L), static: (B, S) -> (B,)."""
        h = self.encode(x, static, return_sequence=False)  # (B, H)
        return self.classifier(h).squeeze(-1)  # (B,)

    def predict_per_timestep(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Per-timestep prediction (AKI/sepsis). x: (B, C, L), static: (B, S) -> (B, L)."""
        h_seq = self.encode(x, static, return_sequence=True)  # (B, L, H)
        logits = self.classifier(h_seq)  # (B, L, 1)
        return logits.squeeze(-1)  # (B, L)


class CORALTrainer(E2EBaselineTrainer):
    """Trainer for Deep CORAL end-to-end baseline."""

    def __init__(self, model: CORALModel, source_train_loader, target_train_loader,
                 source_val_loader, config, device="cuda", target_val_loader=None):
        super().__init__(model, source_train_loader, target_train_loader,
                         source_val_loader, config, device, target_val_loader=target_val_loader)

        training = config.get("training", {})
        self.lambda_coral = training.get("lambda_coral", 1.0)
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

        total_metrics = {
            "cls_loss": 0.0, "coral_loss": 0.0, "total_loss": 0.0,
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

                    # === CORAL loss on per-timestep hidden states ===
                    tgt_h_seq = model.encode(tgt_x, tgt_static, return_sequence=True)  # (B, L, H)

                    # Flatten valid timesteps from both domains
                    src_h_valid = src_h_seq[src_vmask].float()  # (N_s, H)
                    tgt_h_valid = tgt_h_seq[tgt_vmask].float()  # (N_t, H)

                    l_coral = coral_loss(src_h_valid, tgt_h_valid)

                else:
                    # Per-stay: mortality
                    src_h = model.encode(src_x, src_static, return_sequence=False)  # (B, H)
                    logits = model.classifier(src_h).squeeze(-1)  # (B,)
                    cls_loss = self.classification_loss(logits, src_y)

                    # CORAL on pooled hidden states
                    tgt_h = model.encode(tgt_x, tgt_static, return_sequence=False)  # (B, H)
                    l_coral = coral_loss(src_h.float(), tgt_h.float())

                # Total loss
                loss = self.lambda_cls * cls_loss + self.lambda_coral * l_coral

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_metrics["cls_loss"] += cls_loss.item()
            total_metrics["coral_loss"] += l_coral.item()
            total_metrics["total_loss"] += loss.item()
            n_batches += 1

        for k in total_metrics:
            total_metrics[k] /= max(n_batches, 1)

        return total_metrics
