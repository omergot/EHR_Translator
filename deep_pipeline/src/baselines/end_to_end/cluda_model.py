"""CLUDA: Contrastive Learning for Unsupervised Domain Adaptation (Ozyurt et al., ICLR 2023).

End-to-end implementation based on https://github.com/oezyurty/CLUDA

Architecture:
- TCN encoder: 5 layers, 64 channels, kernel_size=3, dilation doubling
- Momentum encoder: same architecture, EMA m=0.999
- Projector: MLP 64 -> 256 -> 64
- Predictor: MLP (64+4_static) -> 256 -> 1
- Discriminator: MLP 64 -> 256 -> 1
- Memory queue: size 8192

Training losses:
- Source contrastive (InfoNCE with queue)
- Target contrastive (InfoNCE with queue)
- Cross-domain NN contrastive
- Adversarial discriminator (GRL)
- Prediction (weighted BCE on source)
"""

import copy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW

from .trainer_base import E2EBaselineTrainer
from ..components import GradientReversalLayer


# ---------------------------------------------------------------------------
# TCN building blocks
# ---------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """Single TCN block with dilated causal convolutions and residual connection."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, C_out, L)."""
        out = self.conv1(x)
        # Remove future timesteps for causal convolution
        if self._padding > 0:
            out = out[:, :, :-self._padding]
        out = self.relu(self.bn1(out))
        out = self.dropout1(out)

        out = self.conv2(out)
        if self._padding > 0:
            out = out[:, :, :-self._padding]
        out = self.relu(self.bn2(out))
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """TCN encoder: stack of dilated causal convolution blocks.

    Default: 5 layers, 64 channels, kernel_size=3, dilation=2^layer.
    """

    def __init__(self, num_inputs: int, num_channels: int = 64,
                 num_layers: int = 5, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = num_inputs if i == 0 else num_channels
            layers.append(TemporalBlock(
                in_ch, num_channels, kernel_size,
                dilation=2 ** i, dropout=dropout,
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, L) -> (B, C_out, L)."""
        return self.network(x)


# ---------------------------------------------------------------------------
# CLUDA model
# ---------------------------------------------------------------------------

class CLUDAModel(nn.Module):
    """CLUDA end-to-end model.

    Components:
    - Online encoder (TCN)
    - Momentum encoder (TCN, EMA updated)
    - Projector (MLP)
    - Predictor/classifier (MLP)
    - Domain discriminator (MLP + GRL)
    - Memory queues for source and target
    """

    def __init__(self, config: dict):
        super().__init__()
        training = config.get("training", {})

        self.num_channels = training.get("tcn_channels", 64)
        self.num_layers = training.get("tcn_layers", 5)
        self.kernel_size = training.get("tcn_kernel_size", 3)
        self.proj_dim = training.get("proj_dim", 64)
        self.temperature = training.get("temperature", 0.07)
        self.momentum = training.get("momentum", 0.999)
        self.queue_size = training.get("queue_size", 8192)
        num_static = 4

        # Determine input channels from config
        # Default: 96 (48 dynamic + 48 MI) for mortality, 100 for AKI/sepsis
        self.num_inputs = training.get("num_input_channels", 96)

        # Online encoder
        self.encoder = TemporalConvNet(
            self.num_inputs, self.num_channels, self.num_layers,
            self.kernel_size, dropout=0.1,
        )

        # Momentum encoder (copy, no grad)
        self.momentum_encoder = copy.deepcopy(self.encoder)
        for p in self.momentum_encoder.parameters():
            p.requires_grad = False

        # Projector: maps encoder output to contrastive space
        self.projector = nn.Sequential(
            nn.Linear(self.num_channels, 256),
            nn.ReLU(),
            nn.Linear(256, self.proj_dim),
        )

        # Classifier: takes global-pooled encoder features + static -> logit
        self.classifier = nn.Sequential(
            nn.Linear(self.num_channels + num_static, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        # Domain discriminator (GRL -> MLP -> 1)
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.discriminator = nn.Sequential(
            nn.Linear(self.num_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        # Memory queues
        self.register_buffer("queue_source", torch.randn(self.queue_size, self.proj_dim))
        self.register_buffer("queue_target", torch.randn(self.queue_size, self.proj_dim))
        self.register_buffer("queue_source_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_target_ptr", torch.zeros(1, dtype=torch.long))

        # Normalize queue
        self.queue_source = F.normalize(self.queue_source, dim=1)
        self.queue_target = F.normalize(self.queue_target, dim=1)

        logging.info(
            "[CLUDA] inputs=%d tcn_ch=%d layers=%d kernel=%d proj=%d queue=%d",
            self.num_inputs, self.num_channels, self.num_layers,
            self.kernel_size, self.proj_dim, self.queue_size,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with online encoder. x: (B, C, L) -> (B, H)."""
        h = self.encoder(x)  # (B, H, L)
        return h.mean(dim=2)  # global average pooling -> (B, H)

    @torch.no_grad()
    def encode_momentum(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with momentum encoder. x: (B, C, L) -> (B, H)."""
        h = self.momentum_encoder(x)
        return h.mean(dim=2)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Project encoder features to contrastive space."""
        return F.normalize(self.projector(h), dim=1)

    def predict(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Predict logits for classification. x: (B, C, L), static: (B, S) -> (B,)."""
        h = self.encode(x)
        h_cat = torch.cat([h, static], dim=1)
        return self.classifier(h_cat).squeeze(-1)

    def predict_per_timestep(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Per-timestep prediction. x: (B, C, L), static: (B, S) -> (B, L)."""
        h_seq = self.encoder(x)  # (B, H, L) — skip pooling
        B, H, L = h_seq.shape
        # Broadcast static to each timestep
        static_exp = static.unsqueeze(2).expand(-1, -1, L)  # (B, S, L)
        h_cat = torch.cat([h_seq, static_exp], dim=1)  # (B, H+S, L)
        # Apply classifier at each timestep
        h_cat = h_cat.permute(0, 2, 1)  # (B, L, H+S)
        logits = self.classifier(h_cat)  # (B, L, 1)
        return logits.squeeze(-1)  # (B, L)

    def discriminate(self, h: torch.Tensor) -> torch.Tensor:
        """Domain discrimination with GRL. h: (B, H) -> (B,)."""
        return self.discriminator(self.grl(h)).squeeze(-1)

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """EMA update of momentum encoder."""
        for p_online, p_mom in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            p_mom.data = self.momentum * p_mom.data + (1.0 - self.momentum) * p_online.data

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor, queue_name: str):
        """Enqueue projected features into memory queue."""
        queue = getattr(self, f"queue_{queue_name}")
        ptr_buf = getattr(self, f"queue_{queue_name}_ptr")
        ptr = int(ptr_buf.item())
        batch_size = keys.size(0)
        if batch_size > self.queue_size:
            keys = keys[:self.queue_size]
            batch_size = self.queue_size
        if ptr + batch_size > self.queue_size:
            remaining = self.queue_size - ptr
            queue[ptr:] = keys[:remaining]
            queue[:batch_size - remaining] = keys[remaining:]
        else:
            queue[ptr:ptr + batch_size] = keys
        ptr_buf[0] = (ptr + batch_size) % self.queue_size

    def contrastive_loss(self, q: torch.Tensor, k: torch.Tensor,
                         queue: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss with positive key and negative queue.

        q: (B, D) query features (online encoder + projector).
        k: (B, D) key features (momentum encoder + projector).
        queue: (K, D) negative keys from memory queue.
        """
        # Positive: (B,)
        pos = (q * k).sum(dim=1, keepdim=True) / self.temperature  # (B, 1)
        # Negatives: (B, K)
        neg = q @ queue.t() / self.temperature  # (B, K)
        # Logits: (B, 1+K)
        logits = torch.cat([pos, neg], dim=1)
        # Labels: positives are at index 0
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)

    def cross_domain_loss(self, h_source: torch.Tensor,
                          h_target: torch.Tensor) -> torch.Tensor:
        """Cross-domain NN contrastive loss.

        For each source sample, find nearest target sample and use it as positive.
        """
        z_s = F.normalize(self.projector(h_source), dim=1)
        z_t = F.normalize(self.projector(h_target.detach()), dim=1)

        # Find NN: cosine similarity
        sim = z_s @ z_t.t()  # (B_s, B_t)
        nn_idx = sim.argmax(dim=1)  # (B_s,)

        # Positive: NN target for each source
        pos = (z_s * z_t[nn_idx]).sum(dim=1, keepdim=True) / self.temperature
        # Negative: all other targets
        neg = sim / self.temperature  # (B_s, B_t) — includes the positive
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(z_s.size(0), dtype=torch.long, device=z_s.device)
        return F.cross_entropy(logits, labels)


class CLUDATrainer(E2EBaselineTrainer):
    """Trainer for CLUDA end-to-end baseline."""

    def __init__(self, model: CLUDAModel, source_train_loader, target_train_loader,
                 source_val_loader, config, device="cuda"):
        super().__init__(model, source_train_loader, target_train_loader,
                         source_val_loader, config, device)

        training = config.get("training", {})
        self.lambda_contrastive = training.get("lambda_contrastive", 0.5)
        self.lambda_cross = training.get("lambda_cross", 0.5)
        self.lambda_adversarial = training.get("lambda_adversarial", 0.1)
        self.lambda_pred = training.get("lambda_pred", 1.0)
        self.warmup_steps = training.get("warmup_steps", 500)

        # Separate optimizers for encoder+predictor vs discriminator
        enc_params = (
            list(model.encoder.parameters()) +
            list(model.projector.parameters()) +
            list(model.classifier.parameters())
        )
        self.optimizer = AdamW(enc_params, lr=self.lr, weight_decay=self.weight_decay)
        disc_lr = training.get("discriminator_lr", self.lr)
        self.disc_optimizer = AdamW(
            model.discriminator.parameters(), lr=disc_lr, weight_decay=self.weight_decay,
        )

        self._global_step = 0
        self._target_iter = iter(target_train_loader)

    def _get_target_batch(self):
        try:
            return next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self.target_train_loader)
            return next(self._target_iter)

    def _grl_lambda(self, epoch: int) -> float:
        """Progressive GRL schedule."""
        p = epoch / max(self.epochs, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        model = self.model

        # Update GRL lambda
        grl_lam = self._grl_lambda(epoch)
        model.grl.set_lambda(grl_lam)

        total_metrics = {
            "pred_loss": 0.0, "src_ctr": 0.0, "tgt_ctr": 0.0,
            "cross_ctr": 0.0, "adv_loss": 0.0, "total_loss": 0.0,
        }
        n_batches = 0

        for src_batch in self.source_train_loader:
            src_x, src_y, src_static = src_batch[0], src_batch[1], src_batch[2]
            tgt_batch = self._get_target_batch()
            tgt_x = tgt_batch[0]

            src_x = src_x.to(self.device)
            src_y = src_y.to(self.device)
            src_static = src_static.to(self.device)
            tgt_x = tgt_x.to(self.device)

            with autocast(enabled=self.use_amp):
                # Encode (online)
                h_src = model.encode(src_x)  # (B, H)
                h_tgt = model.encode(tgt_x)  # (B, H)

                # Encode (momentum)
                k_src = model.encode_momentum(src_x)
                k_tgt = model.encode_momentum(tgt_x)

                # Project
                z_src = model.project(h_src)
                z_tgt = model.project(h_tgt)
                z_k_src = model.project(k_src)
                z_k_tgt = model.project(k_tgt)

                # Contrastive losses
                src_ctr = model.contrastive_loss(z_src, z_k_src, model.queue_source)
                tgt_ctr = model.contrastive_loss(z_tgt, z_k_tgt, model.queue_target)

                # Cross-domain NN contrastive
                cross_ctr = model.cross_domain_loss(h_src, h_tgt)

                # Prediction loss (source only)
                logits = model.classifier(
                    torch.cat([h_src, src_static], dim=1)
                ).squeeze(-1)
                # For per-timestep labels (B, L), aggregate to per-segment for training
                train_y = src_y
                if train_y.dim() > 1:
                    train_y = train_y.clamp(min=0).max(dim=1).values
                pred_loss = self.classification_loss(logits, train_y)

                # Adversarial loss
                h_all = torch.cat([h_src, h_tgt], dim=0)
                d_logits = model.discriminate(h_all)
                d_labels = torch.cat([
                    torch.zeros(h_src.size(0), device=self.device),
                    torch.ones(h_tgt.size(0), device=self.device),
                ])
                adv_loss = F.binary_cross_entropy_with_logits(d_logits, d_labels)

                # Total loss
                loss = (
                    self.lambda_pred * pred_loss +
                    self.lambda_contrastive * (src_ctr + tgt_ctr) +
                    self.lambda_cross * cross_ctr +
                    self.lambda_adversarial * adv_loss
                )

            # Update encoder + classifier
            self.optimizer.zero_grad()
            self.disc_optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.step(self.disc_optimizer)
            self.scaler.update()

            # EMA update momentum encoder
            model._update_momentum_encoder()

            # Enqueue momentum features
            model._enqueue(z_k_src.detach(), "source")
            model._enqueue(z_k_tgt.detach(), "target")

            total_metrics["pred_loss"] += pred_loss.item()
            total_metrics["src_ctr"] += src_ctr.item()
            total_metrics["tgt_ctr"] += tgt_ctr.item()
            total_metrics["cross_ctr"] += cross_ctr.item()
            total_metrics["adv_loss"] += adv_loss.item()
            total_metrics["total_loss"] += loss.item()
            n_batches += 1
            self._global_step += 1

        for k in total_metrics:
            total_metrics[k] /= max(n_batches, 1)

        return total_metrics
