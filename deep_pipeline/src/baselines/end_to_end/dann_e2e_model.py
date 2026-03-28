"""DANN: Domain-Adversarial Neural Network (Ganin et al., JMLR 2016).

End-to-end implementation with ORIGINAL architecture (not our translator backbone).

Architecture:
- Encoder: 2-layer LSTM (naturally causal, per-timestep hidden states)
  - input_size = num_features (96/100), hidden_size = 128, num_layers = 2
- Task classifier: Linear(128, 1) applied per-timestep for AKI/sepsis, pooled for mortality
- Domain discriminator: GRL + MLP(128, 256, 1) applied per-timestep
- GRL lambda: progressive schedule (Ganin et al. 2016)

Training losses:
- Source classification: per-timestep masked BCE (AKI/sepsis) or per-stay BCE (mortality)
- Domain adversarial: per-timestep BCE on both domains
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


class DANNEncoder(nn.Module):
    """2-layer LSTM encoder. Naturally causal, per-timestep output.

    Input: (B, C, L) — channels-first format from E2EDataset
    Output: (B, L, H) per-timestep hidden states, or (B, H) pooled
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, num_static: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_static = num_static

        # Project static features to broadcast per-timestep
        self.static_proj = nn.Linear(num_static, hidden_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,  # Causal: forward-only
        )

        # Layer norm on LSTM output for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, static: torch.Tensor,
                return_sequence: bool = True) -> torch.Tensor:
        """Encode input sequence.

        Args:
            x: (B, C, L) channels-first input.
            static: (B, S) static features.
            return_sequence: If True, return (B, L, H). If False, return (B, H) pooled.

        Returns:
            Hidden states: (B, L, H) or (B, H).
        """
        # Transpose to (B, L, C) for LSTM
        x_seq = x.transpose(1, 2)  # (B, L, C)
        B, L, C = x_seq.shape

        # Run LSTM
        h_seq, _ = self.lstm(x_seq)  # (B, L, H)
        h_seq = self.layer_norm(h_seq)

        # Add static context: project static and add to each timestep
        static_emb = self.static_proj(static)  # (B, H)
        h_seq = h_seq + static_emb.unsqueeze(1)  # (B, L, H) broadcast

        if return_sequence:
            return h_seq  # (B, L, H)
        else:
            # Pool: mean over valid timesteps (use last hidden state)
            return h_seq[:, -1, :]  # (B, H) — last timestep


class DANNModel(nn.Module):
    """DANN: LSTM encoder + task classifier + GRL domain discriminator.

    For per-timestep tasks (AKI, sepsis): classifier and discriminator operate
    at each timestep independently.
    For per-stay tasks (mortality): pool encoder output before classifier.
    """

    def __init__(self, config: dict):
        super().__init__()
        training = config.get("training", {})

        self.num_inputs = training.get("num_input_channels", 96)
        self.hidden_size = training.get("hidden_size", 128)
        num_layers = training.get("num_lstm_layers", 2)
        dropout = training.get("dropout", 0.2)

        # Encoder
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

        # Domain discriminator: GRL + MLP
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.discriminator = DomainDiscriminator(
            input_dim=self.hidden_size,
            hidden_dim=256,
            n_layers=3,
            dropout=0.3,
        )

        logging.info(
            "[DANN-E2E] inputs=%d hidden=%d lstm_layers=%d dropout=%.2f",
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

    def discriminate_sequence(self, h_seq: torch.Tensor) -> torch.Tensor:
        """Per-timestep domain discrimination. h_seq: (B, L, H) -> (B*L, 1)."""
        B, L, H = h_seq.shape
        h_flat = h_seq.reshape(B * L, H)  # (B*L, H)
        h_rev = self.grl(h_flat)  # Apply GRL
        return self.discriminator(h_rev)  # (B*L, 1)

    def discriminate_pooled(self, h: torch.Tensor) -> torch.Tensor:
        """Pooled domain discrimination. h: (B, H) -> (B, 1)."""
        h_rev = self.grl(h)
        return self.discriminator(h_rev)  # (B, 1)


class DANNTrainer(E2EBaselineTrainer):
    """Trainer for DANN end-to-end baseline."""

    def __init__(self, model: DANNModel, source_train_loader, target_train_loader,
                 source_val_loader, config, device="cuda"):
        super().__init__(model, source_train_loader, target_train_loader,
                         source_val_loader, config, device)

        training = config.get("training", {})
        self.lambda_adversarial = training.get("lambda_adversarial", 1.0)
        self.lambda_cls = training.get("lambda_cls", 1.0)

        # Single optimizer for all parameters (Ganin et al. standard)
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

        # Progressive GRL lambda (Ganin et al. 2016)
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

                    # Flatten valid timesteps from both domains
                    src_h_valid = src_h_seq[src_vmask].float()  # (N_s, H)
                    tgt_h_valid = tgt_h_seq[tgt_vmask].float()  # (N_t, H)

                    if src_h_valid.size(0) > 0 and tgt_h_valid.size(0) > 0:
                        h_all = torch.cat([src_h_valid, tgt_h_valid], dim=0)  # (N_s+N_t, H)
                        h_rev = model.grl(h_all)
                        d_logits = model.discriminator(h_rev).squeeze(-1)  # (N_s+N_t,)
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
