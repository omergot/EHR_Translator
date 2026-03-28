"""ACON: Adversarial CONtrastive domain adaptation (Liu et al., NeurIPS 2024).

End-to-end implementation based on https://github.com/mingyangliu1024/ACON

Architecture:
- Temporal encoder: 3-layer 1D CNN (AdaTime standard)
- Frequency encoder: multi-period reshaping + 1D CNN
- Temporal classifier: MLP t_feat -> num_classes
- Frequency classifier: MLP f_feat -> num_classes
- Domain discriminator: MLP (t_feat_dim * f_feat_dim) -> hidden -> 1

Training losses:
- Classification: CE on temporal + frequency source predictions
- Domain adversarial: on temporal-frequency correlation subspace
- Conditional entropy: on target predictions (weight 0.01)
- Temporal alignment: KL(source_t || target_t)
- Frequency alignment: KL(source_f || target_f)
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .trainer_base import E2EBaselineTrainer
from ..components import GradientReversalLayer


# ---------------------------------------------------------------------------
# Temporal encoder (3-layer CNN, AdaTime standard)
# ---------------------------------------------------------------------------

class TemporalEncoderCNN(nn.Module):
    """3-layer 1D CNN temporal encoder.

    Input: (B, C_in, L) -> Output: (B, H)
    """

    def __init__(self, num_inputs: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        # Causal (left-only) padding: pad = kernel_size - 1, then truncate right
        self.conv1 = nn.Conv1d(num_inputs, hidden_dim, kernel_size=5, padding=0)
        self.pad1 = 4
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=0)
        self.pad2 = 4
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=0)
        self.pad3 = 2
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, return_seq: bool = False) -> torch.Tensor:
        """x: (B, C, L) -> (B, H). If return_seq, also return (B, H, L') before pooling.
        Uses causal (left-only) padding to prevent time-travel."""
        h = F.pad(x, (self.pad1, 0))
        h = self.dropout(F.relu(self.bn1(self.conv1(h))))
        h = F.max_pool1d(h, 2)
        h = F.pad(h, (self.pad2, 0))
        h = self.dropout(F.relu(self.bn2(self.conv2(h))))
        h = F.max_pool1d(h, 2)
        h = F.pad(h, (self.pad3, 0))
        h = self.dropout(F.relu(self.bn3(self.conv3(h))))
        if return_seq:
            return self.pool(h).squeeze(-1), h  # (B, H), (B, H, L')
        return self.pool(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Frequency encoder (multi-period reshaping + CNN)
# ---------------------------------------------------------------------------

class FrequencyEncoder(nn.Module):
    """Frequency encoder using multi-period decomposition.

    Reshapes input into multiple periods, applies 1D CNN on each, then aggregates.
    This captures periodic patterns at different frequencies.

    Input: (B, C_in, L) -> Output: (B, H_f)
    """

    def __init__(self, num_inputs: int, hidden_dim: int = 64, periods: tuple = (2, 4, 8)):
        super().__init__()
        self.periods = periods
        self.out_dim = hidden_dim

        # One small CNN per period
        self.period_convs = nn.ModuleList()
        for p in periods:
            self.period_convs.append(nn.Sequential(
                nn.Conv1d(num_inputs, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            ))

        # Aggregate features from all periods
        self.aggregate = nn.Sequential(
            nn.Linear(hidden_dim * len(periods), hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, H_f)."""
        B, C, L = x.shape
        period_feats = []

        for i, p in enumerate(self.periods):
            # Subsample at period p: take every p-th timestep
            x_sub = x[:, :, ::p]  # (B, C, L//p)
            if x_sub.size(2) < 2:
                # Too short, zero-pad
                x_sub = F.pad(x_sub, (0, 2 - x_sub.size(2)))
            feat = self.period_convs[i](x_sub).squeeze(-1)  # (B, H)
            period_feats.append(feat)

        combined = torch.cat(period_feats, dim=1)  # (B, H * n_periods)
        return self.aggregate(combined)  # (B, H_f)


# ---------------------------------------------------------------------------
# ACON model
# ---------------------------------------------------------------------------

class ACONModel(nn.Module):
    """ACON end-to-end model.

    Components:
    - Temporal encoder (3-layer CNN)
    - Frequency encoder (multi-period + CNN)
    - Temporal classifier (MLP)
    - Frequency classifier (MLP)
    - Domain discriminator (on outer-product correlation subspace)
    """

    def __init__(self, config: dict):
        super().__init__()
        training = config.get("training", {})

        self.num_inputs = training.get("num_input_channels", 96)
        hidden_dim = training.get("hidden_dim", 64)
        freq_hidden = training.get("freq_hidden_dim", 64)
        num_static = 4

        self.temporal_encoder = TemporalEncoderCNN(self.num_inputs, hidden_dim)
        self.frequency_encoder = FrequencyEncoder(self.num_inputs, freq_hidden)

        t_dim = self.temporal_encoder.out_dim
        f_dim = self.frequency_encoder.out_dim

        # Temporal classifier
        self.temporal_classifier = nn.Sequential(
            nn.Linear(t_dim + num_static, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        # Frequency classifier
        self.frequency_classifier = nn.Sequential(
            nn.Linear(f_dim + num_static, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        # Domain discriminator on correlation subspace
        # Outer product: (t_dim) x (f_dim) -> (t_dim * f_dim)
        # Use bilinear pooling with dimensionality reduction
        self.t_proj = nn.Linear(t_dim, 32)
        self.f_proj = nn.Linear(f_dim, 32)
        corr_dim = 32 * 32  # outer product dimension

        self.grl = GradientReversalLayer(lambda_=1.0)
        self.discriminator = nn.Sequential(
            nn.Linear(corr_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        logging.info(
            "[ACON] inputs=%d t_dim=%d f_dim=%d corr_dim=%d",
            self.num_inputs, t_dim, f_dim, corr_dim,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input into temporal and frequency features.

        Returns:
            t_feat: (B, t_dim)
            f_feat: (B, f_dim)
        """
        t_feat = self.temporal_encoder(x)
        f_feat = self.frequency_encoder(x)
        return t_feat, f_feat

    def correlation_features(self, t_feat: torch.Tensor,
                              f_feat: torch.Tensor) -> torch.Tensor:
        """Compute temporal-frequency correlation subspace features.

        Uses reduced-dimension outer product.
        """
        t_proj = self.t_proj(t_feat)  # (B, 32)
        f_proj = self.f_proj(f_feat)  # (B, 32)
        # Outer product: (B, 32, 1) x (B, 1, 32) -> (B, 32, 32) -> (B, 1024)
        corr = torch.bmm(t_proj.unsqueeze(2), f_proj.unsqueeze(1))
        return corr.view(corr.size(0), -1)

    def predict(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Predict logits (ensemble of temporal + frequency classifiers).

        x: (B, C, L), static: (B, S) -> (B,)
        """
        t_feat, f_feat = self.encode(x)
        t_logits = self.temporal_classifier(
            torch.cat([t_feat, static], dim=1)
        ).squeeze(-1)
        f_logits = self.frequency_classifier(
            torch.cat([f_feat, static], dim=1)
        ).squeeze(-1)
        # Average logits from both classifiers
        return (t_logits + f_logits) / 2.0

    def predict_per_timestep(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Per-timestep prediction. x: (B, C, L), static: (B, S) -> (B, L)."""
        B, C, L = x.shape
        # Get temporal sequence features before pooling (B, H, L')
        _, t_seq = self.temporal_encoder(x, return_seq=True)
        # Upsample to original L
        t_up = F.interpolate(t_seq, size=L, mode="nearest")  # (B, H_t, L)
        # Broadcast static
        s_exp = static.unsqueeze(2).expand(-1, -1, L)  # (B, S, L)
        t_cat = torch.cat([t_up, s_exp], dim=1).permute(0, 2, 1)  # (B, L, H_t+S)
        t_logits = self.temporal_classifier(t_cat).squeeze(-1)  # (B, L)
        # Frequency: only global features, broadcast to all timesteps
        f_feat = self.frequency_encoder(x)  # (B, H_f)
        f_exp = torch.cat([f_feat, static], dim=1)  # (B, H_f+S)
        f_logit = self.frequency_classifier(f_exp).squeeze(-1)  # (B,)
        f_logits = f_logit.unsqueeze(1).expand(-1, L)  # (B, L) broadcast
        return (t_logits + f_logits) / 2.0

    def discriminate(self, corr_feat: torch.Tensor) -> torch.Tensor:
        """Domain discrimination on correlation features with GRL."""
        return self.discriminator(self.grl(corr_feat)).squeeze(-1)


class ACONTrainer(E2EBaselineTrainer):
    """Trainer for ACON end-to-end baseline."""

    def __init__(self, model: ACONModel, source_train_loader, target_train_loader,
                 source_val_loader, config, device="cuda"):
        super().__init__(model, source_train_loader, target_train_loader,
                         source_val_loader, config, device)

        training = config.get("training", {})
        self.lambda_cls = training.get("lambda_cls", 1.0)
        self.lambda_adversarial = training.get("lambda_adversarial", 1.0)
        self.lambda_entropy = training.get("lambda_entropy", 0.01)
        self.lambda_t_align = training.get("lambda_t_align", 0.1)
        self.lambda_f_align = training.get("lambda_f_align", 0.1)

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

    def _grl_lambda(self, epoch: int) -> float:
        p = epoch / max(self.epochs, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

    @staticmethod
    def _conditional_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Conditional entropy minimization on target predictions.

        Encourages confident predictions on target domain.
        """
        probs = torch.sigmoid(logits)
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-7
        entropy = -(
            probs * torch.log(probs + eps) +
            (1 - probs) * torch.log(1 - probs + eps)
        )
        return entropy.mean()

    @staticmethod
    def _kl_alignment(src_feat: torch.Tensor,
                      tgt_feat: torch.Tensor) -> torch.Tensor:
        """KL divergence-based feature alignment.

        Approximates distributions as Gaussians and computes KL.
        """
        # Source stats
        src_mean = src_feat.mean(dim=0)
        src_var = src_feat.var(dim=0, unbiased=True) + 1e-6

        # Target stats
        tgt_mean = tgt_feat.mean(dim=0)
        tgt_var = tgt_feat.var(dim=0, unbiased=True) + 1e-6

        # KL(source || target) for diagonal Gaussians
        kl = 0.5 * (
            (src_var / tgt_var).log() +
            (tgt_var + (tgt_mean - src_mean).pow(2)) / src_var -
            1.0
        ).sum()

        return torch.relu(kl)  # Clip negative values from numerical issues

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        model = self.model

        # Update GRL lambda
        grl_lam = self._grl_lambda(epoch)
        model.grl.set_lambda(grl_lam)

        total_metrics = {
            "cls_loss": 0.0, "adv_loss": 0.0, "ent_loss": 0.0,
            "t_align": 0.0, "f_align": 0.0, "total_loss": 0.0,
        }
        n_batches = 0

        for src_batch in self.source_train_loader:
            src_x, src_y, src_static = src_batch[0], src_batch[1], src_batch[2]
            tgt_batch = self._get_target_batch()
            tgt_x, tgt_static = tgt_batch[0], tgt_batch[2]

            src_x = src_x.to(self.device)
            src_y = src_y.to(self.device)
            src_static = src_static.to(self.device)
            tgt_x = tgt_x.to(self.device)
            tgt_static = tgt_static.to(self.device)

            with autocast(enabled=self.use_amp):
                # Encode both domains
                src_t, src_f = model.encode(src_x)
                tgt_t, tgt_f = model.encode(tgt_x)

                # Classification loss (source, both classifiers)
                if src_y.dim() > 1:
                    # Per-timestep classification: predict at every timestep
                    src_vmask = src_batch[3].to(self.device)
                    logits_ts = model.predict_per_timestep(src_x, src_static)  # (B, L)
                    valid = src_vmask & (src_y >= 0)
                    if valid.sum() > 0:
                        cls_loss = F.binary_cross_entropy_with_logits(
                            logits_ts[valid], src_y[valid],
                            pos_weight=self._pos_weight)
                    else:
                        cls_loss = logits_ts.new_tensor(0.0)
                else:
                    # Per-stay classification: global pooled features
                    src_t_logits = model.temporal_classifier(
                        torch.cat([src_t, src_static], dim=1)
                    ).squeeze(-1)
                    src_f_logits = model.frequency_classifier(
                        torch.cat([src_f, src_static], dim=1)
                    ).squeeze(-1)
                    cls_loss = (
                        self.classification_loss(src_t_logits, src_y) +
                        self.classification_loss(src_f_logits, src_y)
                    ) * 0.5

                # Domain adversarial on correlation subspace
                src_corr = model.correlation_features(src_t, src_f)
                tgt_corr = model.correlation_features(tgt_t, tgt_f)
                corr_all = torch.cat([src_corr, tgt_corr], dim=0)
                d_logits = model.discriminate(corr_all)
                d_labels = torch.cat([
                    torch.zeros(src_corr.size(0), device=self.device),
                    torch.ones(tgt_corr.size(0), device=self.device),
                ])
                adv_loss = F.binary_cross_entropy_with_logits(d_logits, d_labels)

                # Conditional entropy on target predictions
                tgt_t_logits = model.temporal_classifier(
                    torch.cat([tgt_t, tgt_static], dim=1)
                ).squeeze(-1)
                tgt_f_logits = model.frequency_classifier(
                    torch.cat([tgt_f, tgt_static], dim=1)
                ).squeeze(-1)
                ent_loss = (
                    self._conditional_entropy(tgt_t_logits) +
                    self._conditional_entropy(tgt_f_logits)
                ) * 0.5

                # Feature alignment (KL divergence)
                t_align = self._kl_alignment(src_t.float(), tgt_t.float().detach())
                f_align = self._kl_alignment(src_f.float(), tgt_f.float().detach())

                # Total loss
                loss = (
                    self.lambda_cls * cls_loss +
                    self.lambda_adversarial * adv_loss +
                    self.lambda_entropy * ent_loss +
                    self.lambda_t_align * t_align +
                    self.lambda_f_align * f_align
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_metrics["cls_loss"] += cls_loss.item()
            total_metrics["adv_loss"] += adv_loss.item()
            total_metrics["ent_loss"] += ent_loss.item()
            total_metrics["t_align"] += t_align.item()
            total_metrics["f_align"] += f_align.item()
            total_metrics["total_loss"] += loss.item()
            n_batches += 1

        for k in total_metrics:
            total_metrics[k] /= max(n_batches, 1)

        return total_metrics
