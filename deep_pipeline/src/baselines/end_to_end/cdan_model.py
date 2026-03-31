"""CDAN: Conditional Domain Adversarial Network (Long et al., NeurIPS 2018).

End-to-end implementation. Conditions the domain discriminator on classifier
predictions via multilinear conditioning (outer product of features × predictions).

Architecture: Same LSTM encoder + classifier as DANN. Discriminator receives
h ⊗ p where h = encoder hidden state, p = sigmoid(classifier(h)).
For binary classification with scalar output, the conditioning doubles the
discriminator input: concat(h * p, h * (1-p)) → (2H,) input.

Reference: Long et al., "Conditional Adversarial Domain Adaptation", NeurIPS 2018.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW

from .dann_e2e_model import DANNEncoder, DANNTrainer
from ..components import GradientReversalLayer, DomainDiscriminator, grl_lambda_schedule


class CDANModel(nn.Module):
    """CDAN: LSTM encoder + classifier + conditional domain discriminator."""

    def __init__(self, config: dict):
        super().__init__()
        training = config.get("training", {})

        self.num_inputs = training.get("num_input_channels", 96)
        self.hidden_size = training.get("hidden_size", 128)
        num_layers = training.get("num_lstm_layers", 2)
        dropout = training.get("dropout", 0.2)
        use_layer_norm = training.get("use_layer_norm", True)
        use_static_proj = training.get("use_static_proj", True)

        # Same encoder as DANN
        self.encoder = DANNEncoder(
            input_size=self.num_inputs,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_static_proj=use_static_proj,
        )

        # Task classifier: Linear(H, 1)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )

        # Conditional domain discriminator: input is h ⊗ p → (2H,)
        # For binary: concat(h * sigmoid(logit), h * (1 - sigmoid(logit)))
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.discriminator = DomainDiscriminator(
            input_dim=self.hidden_size * 2,  # 2H due to conditioning
            hidden_dim=256,
            n_layers=3,
            dropout=0.3,
        )

        logging.info(
            "[CDAN-E2E] inputs=%d hidden=%d lstm_layers=%d dropout=%.2f ln=%s static_proj=%s",
            self.num_inputs, self.hidden_size, num_layers, dropout, use_layer_norm, use_static_proj,
        )

    def encode(self, x, static, return_sequence=True):
        return self.encoder(x, static, return_sequence=return_sequence)

    def predict(self, x, static):
        h = self.encode(x, static, return_sequence=False)
        return self.classifier(h).squeeze(-1)

    def predict_per_timestep(self, x, static):
        h_seq = self.encode(x, static, return_sequence=True)
        logits = self.classifier(h_seq)
        return logits.squeeze(-1)

    def _condition(self, h: torch.Tensor) -> torch.Tensor:
        """Multilinear conditioning: concat(h * p, h * (1-p)) where p = sigmoid(classifier(h)).

        Args:
            h: (N, H) hidden states.
        Returns:
            (N, 2H) conditioned features.
        """
        logits = self.classifier(h).squeeze(-1)  # (N,)
        p = torch.sigmoid(logits).unsqueeze(-1)   # (N, 1)
        return torch.cat([h * p, h * (1 - p)], dim=-1)  # (N, 2H)


class CDANTrainer(DANNTrainer):
    """Trainer for CDAN. Inherits from DANNTrainer, overrides adversarial loss computation."""

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        model = self.model

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
                if src_y.dim() > 1:
                    # Per-timestep
                    src_h_seq = model.encode(src_x, src_static, return_sequence=True)
                    logits_ts = model.classifier(src_h_seq).squeeze(-1)
                    valid = src_vmask & (src_y >= 0)
                    if valid.sum() > 0:
                        cls_loss = F.binary_cross_entropy_with_logits(
                            logits_ts[valid], src_y[valid],
                            pos_weight=self._pos_weight,
                        )
                    else:
                        cls_loss = logits_ts.new_tensor(0.0)

                    tgt_h_seq = model.encode(tgt_x, tgt_static, return_sequence=True)

                    src_h_valid = src_h_seq[src_vmask].float()
                    tgt_h_valid = tgt_h_seq[tgt_vmask].float()

                    if src_h_valid.size(0) > 0 and tgt_h_valid.size(0) > 0:
                        h_all = torch.cat([src_h_valid, tgt_h_valid], dim=0)
                        # CDAN: condition on classifier predictions
                        h_cond = model._condition(h_all)  # (N, 2H)
                        h_rev = model.grl(h_cond)
                        d_logits = model.discriminator(h_rev).squeeze(-1)
                        d_labels = torch.cat([
                            torch.zeros(src_h_valid.size(0), device=self.device),
                            torch.ones(tgt_h_valid.size(0), device=self.device),
                        ])
                        adv_loss = F.binary_cross_entropy_with_logits(d_logits, d_labels)
                    else:
                        adv_loss = src_x.new_tensor(0.0)
                else:
                    # Per-stay
                    src_h = model.encode(src_x, src_static, return_sequence=False)
                    logits = model.classifier(src_h).squeeze(-1)
                    cls_loss = self.classification_loss(logits, src_y)

                    tgt_h = model.encode(tgt_x, tgt_static, return_sequence=False)
                    h_all = torch.cat([src_h, tgt_h], dim=0).float()
                    h_cond = model._condition(h_all)
                    h_rev = model.grl(h_cond)
                    d_logits = model.discriminator(h_rev).squeeze(-1)
                    d_labels = torch.cat([
                        torch.zeros(src_h.size(0), device=self.device),
                        torch.ones(tgt_h.size(0), device=self.device),
                    ])
                    adv_loss = F.binary_cross_entropy_with_logits(d_logits, d_labels)

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
