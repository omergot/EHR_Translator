"""Trainer for the retrieval translator on AdaTime benchmarks.

Adapts RetrievalTranslatorTrainer for per-sequence classification with
shorter fixed-length sequences and fewer features.

Key differences from EHR pipeline:
  1. Per-sequence cross-entropy loss (not per-timestep BCE/MSE)
  2. No missing indicators, no generated features, no variable-length padding
  3. Fixed 128-timestep sequences, 3-9 channels
  4. Smaller memory bank (hundreds of samples, not thousands of stays)
  5. No bounds CSV (use data-derived bounds instead)
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from src.core.retrieval_translator import (
    RetrievalTranslator,
    build_memory_bank,
    query_memory_bank,
    MemoryBank,
)
from .adapter import AdaTimeSchemaResolver, AdaTimeRuntime
from .target_model import LSTMClassifier, AdaTimeCNNClassifier

logger = logging.getLogger(__name__)


class AdaTimeRetrievalTrainer:
    """Retrieval translator trainer adapted for AdaTime benchmarks.

    Phase 1: Autoencoder pretrain on target domain (reconstruct target data)
    Phase 2: Task-guided training with retrieval (cross-entropy on frozen model)
    """

    def __init__(
        self,
        frozen_model: LSTMClassifier,
        translator: RetrievalTranslator,
        schema_resolver: AdaTimeSchemaResolver,
        target_train_loader: DataLoader,
        num_classes: int = 6,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_recon: float = 0.1,
        lambda_range: float = 0.1,
        lambda_smooth: float = 0.0,
        lambda_importance_reg: float = 0.01,
        lambda_fidelity: float = 0.01,
        pretrain_epochs: int = 10,
        k_neighbors: int = 8,
        retrieval_window: int = 4,
        memory_refresh_epochs: int = 5,
        early_stopping_patience: int = 10,
        use_last_epoch: bool = False,
        run_dir: str = "runs/adatime",
        device: str = "cuda",
        optimizer_type: str = "adamw",
        optimizer_betas: tuple = (0.9, 0.999),
    ):
        self.frozen_model = frozen_model.to(device)
        self.translator = translator.to(device)
        self.schema_resolver = schema_resolver
        self.target_train_loader = target_train_loader
        self._target_iter = iter(target_train_loader)
        self.num_classes = num_classes
        self.device = device

        # Loss weights
        self.lambda_recon = lambda_recon
        self.lambda_range = lambda_range
        self.lambda_smooth = lambda_smooth
        self.lambda_importance_reg = lambda_importance_reg
        self.lambda_fidelity = lambda_fidelity

        # Training params
        self.pretrain_epochs = pretrain_epochs
        self.k_neighbors = k_neighbors
        self.retrieval_window = retrieval_window
        self.memory_refresh_epochs = memory_refresh_epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_last_epoch = use_last_epoch  # AdaTime convention: use last epoch, not best

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        OptimClass = Adam if optimizer_type.lower() == "adam" else AdamW
        self.optimizer = OptimClass(
            self.translator.parameters(), lr=learning_rate,
            weight_decay=weight_decay, betas=optimizer_betas,
        )
        self.scaler = GradScaler(enabled=device.startswith("cuda"))

        # Snapshot frozen model params for verification
        self._frozen_param_snapshot = {
            name: param.detach().clone()
            for name, param in self.frozen_model.named_parameters()
        }

        # Compute data-derived feature bounds for range loss
        self._compute_feature_bounds()

        # State
        self.best_val_metric = 0.0  # accuracy (higher is better)
        self.best_state = None
        self.history = []
        self.memory_bank: Optional[MemoryBank] = None

    def _compute_feature_bounds(self):
        """Compute per-feature min/max from target training data for range loss."""
        all_data = []
        with torch.no_grad():
            for batch in self.target_train_loader:
                x = batch[0]  # (B, T, C)
                all_data.append(x)
        all_data = torch.cat(all_data, dim=0)  # (N, T, C)
        # Use 0.1th and 99.9th percentiles
        flat = all_data.reshape(-1, all_data.shape[-1])  # (N*T, C)
        self.lower_bounds = torch.quantile(flat, 0.001, dim=0).to(self.device)
        self.upper_bounds = torch.quantile(flat, 0.999, dim=0).to(self.device)
        logger.info(
            "Feature bounds: lower=[%.3f, %.3f], upper=[%.3f, %.3f]",
            self.lower_bounds.min().item(), self.lower_bounds.max().item(),
            self.upper_bounds.min().item(), self.upper_bounds.max().item(),
        )

    def _next_target_batch(self):
        try:
            batch = next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self.target_train_loader)
            batch = next(self._target_iter)
        return tuple(b.to(self.device) for b in batch)

    def _verify_frozen(self):
        """Verify frozen model weights haven't changed."""
        for name, param in self.frozen_model.named_parameters():
            diff = (param.detach() - self._frozen_param_snapshot[name].to(param.device)).abs().max().item()
            if diff > 0:
                raise RuntimeError(f"FROZEN MODEL CORRUPTED: {name} changed by {diff:.2e}")
        logger.info("[verify] Frozen model integrity OK")

    def _range_loss(self, x_translated: torch.Tensor) -> torch.Tensor:
        """Penalize translated values outside feature bounds."""
        below = F.relu(self.lower_bounds - x_translated)
        above = F.relu(x_translated - self.upper_bounds)
        return (below + above).mean()

    def _smoothness_loss(self, x_translated: torch.Tensor) -> torch.Tensor:
        """Temporal smoothness: penalize large jumps between adjacent timesteps."""
        diff = x_translated[:, 1:, :] - x_translated[:, :-1, :]
        return diff.abs().mean()

    def _fidelity_loss(self, x_translated: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """Input fidelity: keep translation close to original input."""
        return F.mse_loss(x_translated, x_original)

    def _task_loss(
        self,
        x_translated: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss through frozen model.

        Args:
            x_translated: (B, T, C) translated source data
            labels: (B,) class labels
        """
        logits = self.frozen_model(x_translated)  # (B, num_classes)
        return F.cross_entropy(logits, labels)

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 1: Autoencoder pretrain on target data
    # ═══════════════════════════════════════════════════════════════════

    def _pretrain_epoch(self):
        """One epoch of autoencoder pretraining on target domain."""
        self.translator.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.target_train_loader:
            batch = tuple(b.to(self.device) for b in batch)
            parts = self.schema_resolver.extract(batch)

            x_val = parts["X_val"]
            x_miss = parts["X_miss"]
            t_abs = parts["t_abs"]
            m_pad = parts["M_pad"]
            x_static = parts["X_static"]

            # Forward through encoder -> decoder (no retrieval)
            x_recon = self.translator(x_val, x_miss, t_abs, m_pad, x_static)

            # Reconstruction loss
            loss = F.mse_loss(x_recon, x_val)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.translator.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def pretrain(self):
        """Phase 1: Autoencoder pretraining."""
        if self.pretrain_epochs <= 0:
            logger.info("Skipping Phase 1 pretrain (pretrain_epochs=0)")
            return

        logger.info("=== Phase 1: Autoencoder pretrain (%d epochs) ===", self.pretrain_epochs)
        pretrain_ckpt = self.run_dir / "pretrain_checkpoint.pt"

        # Check for existing pretrain checkpoint
        if pretrain_ckpt.exists():
            state = torch.load(pretrain_ckpt, map_location=self.device, weights_only=False)
            self.translator.load_state_dict(state["translator"])
            logger.info("Loaded pretrain checkpoint from %s", pretrain_ckpt)
            return

        for epoch in range(1, self.pretrain_epochs + 1):
            loss = self._pretrain_epoch()
            if epoch % 5 == 0 or epoch == 1:
                logger.info("[Pretrain] Epoch %d/%d: recon_loss=%.6f", epoch, self.pretrain_epochs, loss)

        # Save pretrain checkpoint
        torch.save({"translator": self.translator.state_dict()}, pretrain_ckpt)
        logger.info("Saved pretrain checkpoint to %s", pretrain_ckpt)

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 2: Task-guided training with retrieval
    # ═══════════════════════════════════════════════════════════════════

    def _build_memory_bank(self):
        """Build memory bank from target training data."""
        self.memory_bank = build_memory_bank(
            encoder=self.translator,
            target_loader=self.target_train_loader,
            schema_resolver=self.schema_resolver,
            device=self.device,
            window_size=self.retrieval_window,
        )
        logger.info("Memory bank rebuilt: %d windows", self.memory_bank.window_latents.shape[0])

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """One epoch of Phase 2 training."""
        self.translator.train()
        # CRITICAL: Frozen LSTM must be in train() mode for cuDNN RNN backward
        self.frozen_model.train()
        losses_sum = {"task": 0.0, "fidelity": 0.0, "range": 0.0, "smooth": 0.0, "imp_reg": 0.0, "total": 0.0}
        n_batches = 0

        for batch in train_loader:
            batch = tuple(b.to(self.device) for b in batch)
            parts = self.schema_resolver.extract(batch)

            x_val = parts["X_val"]
            x_miss = parts["X_miss"]
            t_abs = parts["t_abs"]
            m_pad = parts["M_pad"]
            x_static = parts["X_static"]
            y = parts["y"][:, 0]  # Per-sequence label

            # Encode source data
            latent = self.translator.encode(x_val, x_miss, t_abs, m_pad, x_static)

            # Retrieve from memory bank
            importance_w = self.translator.get_importance_weights()
            context = query_memory_bank(
                latent.detach(),  # CRITICAL: detach before querying (safety rule)
                m_pad, self.memory_bank,
                k_neighbors=self.k_neighbors,
                retrieval_window=self.retrieval_window,
                importance_weights=importance_w,
            )

            # Translate with retrieval context
            x_translated, _ = self.translator.forward_with_retrieval(
                x_val, x_miss, t_abs, m_pad, x_static, context, latent=latent,
            )

            # Task loss: cross-entropy through frozen model
            task_loss = self._task_loss(x_translated, y)

            # Fidelity loss
            fidelity_loss = self._fidelity_loss(x_translated, x_val)

            # Range loss
            range_loss = self._range_loss(x_translated)

            # Smoothness loss
            smooth_loss = self._smoothness_loss(x_translated) if self.lambda_smooth > 0 else torch.tensor(0.0)

            # Importance regularization
            imp_reg = importance_w.mean() if self.lambda_importance_reg > 0 else torch.tensor(0.0)

            # Total loss
            total_loss = (
                task_loss
                + self.lambda_fidelity * fidelity_loss
                + self.lambda_range * range_loss
                + self.lambda_smooth * smooth_loss
                + self.lambda_importance_reg * imp_reg
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.translator.parameters(), 1.0)
            self.optimizer.step()

            losses_sum["task"] += task_loss.item()
            losses_sum["fidelity"] += fidelity_loss.item()
            losses_sum["range"] += range_loss.item()
            losses_sum["smooth"] += smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else 0
            losses_sum["imp_reg"] += imp_reg.item() if isinstance(imp_reg, torch.Tensor) else 0
            losses_sum["total"] += total_loss.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on source validation data using frozen model."""
        self.translator.eval()
        # NOTE: Frozen model stays in train() mode (cuDNN RNN backward requires it).
        # Since dropout is disabled and batchnorm is frozen, train() == eval() for frozen model.

        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(b.to(self.device) for b in batch)
                parts = self.schema_resolver.extract(batch)

                x_val = parts["X_val"]
                x_miss = parts["X_miss"]
                t_abs = parts["t_abs"]
                m_pad = parts["M_pad"]
                x_static = parts["X_static"]
                y = parts["y"][:, 0]

                # Encode and retrieve
                latent = self.translator.encode(x_val, x_miss, t_abs, m_pad, x_static)
                importance_w = self.translator.get_importance_weights()
                context = query_memory_bank(
                    latent, m_pad, self.memory_bank,
                    k_neighbors=self.k_neighbors,
                    retrieval_window=self.retrieval_window,
                    importance_weights=importance_w,
                )
                x_translated, _ = self.translator.forward_with_retrieval(
                    x_val, x_miss, t_abs, m_pad, x_static, context, latent=latent,
                )

                logits = self.frozen_model(x_translated)
                loss = F.cross_entropy(logits, y)
                preds = logits.argmax(dim=-1)

                correct += (preds == y).sum().item()
                total += y.shape[0]
                val_loss += loss.item() * y.shape[0]

        acc = correct / max(total, 1)
        avg_loss = val_loss / max(total, 1)

        return {"val_acc": acc, "val_loss": avg_loss}

    def train(
        self,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """Full training: Phase 1 (pretrain) + Phase 2 (task-guided retrieval)."""
        # Phase 1
        self.pretrain()

        # Phase 2
        logger.info("=== Phase 2: Task-guided retrieval training (%d epochs) ===", epochs)

        # Build initial memory bank
        self._build_memory_bank()

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Rebuild memory bank periodically
            if epoch > 1 and (epoch - 1) % self.memory_refresh_epochs == 0:
                self._build_memory_bank()

            # Train
            train_losses = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._validate(val_loader)

            elapsed = time.time() - t0

            # Log
            logger.info(
                "[Epoch %d/%d] task=%.4f fid=%.4f range=%.4f | val_acc=%.4f val_loss=%.4f | %.1fs",
                epoch, epochs,
                train_losses["task"], train_losses["fidelity"], train_losses["range"],
                val_metrics["val_acc"], val_metrics["val_loss"],
                elapsed,
            )

            # Save history
            self.history.append({
                "epoch": epoch,
                **train_losses,
                **val_metrics,
            })

            # Best model tracking (by val accuracy)
            if val_metrics["val_acc"] > self.best_val_metric:
                self.best_val_metric = val_metrics["val_acc"]
                self.best_state = {k: v.clone() for k, v in self.translator.state_dict().items()}
                patience_counter = 0
                logger.info("  -> New best val_acc=%.4f", self.best_val_metric)

                # Save best checkpoint
                torch.save({
                    "translator": self.translator.state_dict(),
                    "best_val_acc": self.best_val_metric,
                    "epoch": epoch,
                }, self.run_dir / "best_checkpoint.pt")
            else:
                patience_counter += 1
                if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.early_stopping_patience)
                    break

        # Restore best model
        if not self.use_last_epoch and self.best_state is not None:
            self.translator.load_state_dict(self.best_state)
            logger.info("Restored best model (val_acc=%.4f)", self.best_val_metric)
        elif self.use_last_epoch:
            logger.info("AdaTime protocol: using last epoch model (not best val)")

        # Verify frozen model integrity
        self._verify_frozen()

        # Save final checkpoint
        torch.save({
            "translator": self.translator.state_dict(),
            "best_val_acc": self.best_val_metric,
            "history": self.history,
        }, self.run_dir / "final_checkpoint.pt")
        logger.info("Training complete. Best val_acc=%.4f", self.best_val_metric)


class AdaTimeCNNRetrievalTrainer:
    """Retrieval translator trainer for target→source translation with frozen source CNN.

    Direction flip from AdaTimeRetrievalTrainer:
      - Frozen model: CNN trained on SOURCE domain
      - Phase 1: Autoencoder pretrain on SOURCE domain (not target)
      - Memory bank: SOURCE domain windows (not target)
      - Phase 2 task: translate TARGET data → source-like → classify with frozen source CNN
      - Validation: evaluate translated target data through frozen source CNN

    This matches AdaTime's published setup where all baselines train a CNN on source
    and try to adapt target data to the source distribution.
    """

    def __init__(
        self,
        frozen_model: AdaTimeCNNClassifier,
        translator: RetrievalTranslator,
        schema_resolver: AdaTimeSchemaResolver,
        source_train_loader: DataLoader,
        num_classes: int = 6,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_recon: float = 0.1,
        lambda_range: float = 0.1,
        lambda_smooth: float = 0.0,
        lambda_importance_reg: float = 0.01,
        lambda_fidelity: float = 0.01,
        pretrain_epochs: int = 10,
        k_neighbors: int = 8,
        retrieval_window: int = 4,
        memory_refresh_epochs: int = 5,
        early_stopping_patience: int = 10,
        best_metric: str = "val_acc",
        use_last_epoch: bool = False,
        run_dir: str = "runs/adatime_cnn",
        pretrain_fallback_dir: str = None,
        device: str = "cuda",
        optimizer_type: str = "adamw",
        optimizer_betas: tuple = (0.9, 0.999),
    ):
        """Initialize the CNN-based retrieval trainer.

        Args:
            frozen_model: Source CNN classifier (frozen, trained on source domain)
            translator: RetrievalTranslator that will translate target→source-like
            schema_resolver: Schema resolver for batch format extraction
            source_train_loader: SOURCE domain training DataLoader (for Phase 1 + memory bank)
            num_classes: Number of output classes
            learning_rate: Translator optimizer learning rate
            weight_decay: Optimizer weight decay
            lambda_recon: Reconstruction loss weight (Phase 1)
            lambda_range: Range constraint loss weight
            lambda_smooth: Temporal smoothness loss weight
            lambda_importance_reg: Importance weight regularization
            lambda_fidelity: Input fidelity loss weight
            pretrain_epochs: Phase 1 autoencoder pretrain epochs
            k_neighbors: Number of retrieval neighbors
            retrieval_window: Retrieval window size
            memory_refresh_epochs: Rebuild memory bank every N epochs
            early_stopping_patience: Epochs without improvement before stopping
            best_metric: Metric for early stopping/best model selection.
                "val_acc" (higher is better) or "val_loss" (lower is better).
                val_loss is recommended for small test sets where accuracy is coarse.
            run_dir: Directory for checkpoints
            pretrain_fallback_dir: Fallback directory to look for pretrain_checkpoint.pt
                when it doesn't exist in run_dir. Used by variant runs to reuse pretrain
                from the base experiment.
            device: Device ('cuda' or 'cpu')
            optimizer_type: "adam" or "adamw"
            optimizer_betas: Optimizer beta parameters
        """
        self.frozen_model = frozen_model.to(device)
        self.translator = translator.to(device)
        self.schema_resolver = schema_resolver
        self.source_train_loader = source_train_loader
        self.num_classes = num_classes
        self.device = device

        # Loss weights
        self.lambda_recon = lambda_recon
        self.lambda_range = lambda_range
        self.lambda_smooth = lambda_smooth
        self.lambda_importance_reg = lambda_importance_reg
        self.lambda_fidelity = lambda_fidelity

        # Training params
        self.pretrain_epochs = pretrain_epochs
        self.k_neighbors = k_neighbors
        self.retrieval_window = retrieval_window
        self.memory_refresh_epochs = memory_refresh_epochs
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = best_metric  # "val_acc" or "val_loss"
        self.use_last_epoch = use_last_epoch  # AdaTime convention: use last epoch, not best
        self.pretrain_fallback_dir = Path(pretrain_fallback_dir) if pretrain_fallback_dir else None

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer for translator
        OptimClass = Adam if optimizer_type.lower() == "adam" else AdamW
        self.optimizer = OptimClass(
            self.translator.parameters(), lr=learning_rate,
            weight_decay=weight_decay, betas=optimizer_betas,
        )
        self.scaler = GradScaler(enabled=device.startswith("cuda"))

        # Snapshot frozen model params for verification
        self._frozen_param_snapshot = {
            name: param.detach().clone()
            for name, param in self.frozen_model.named_parameters()
        }

        # Compute feature bounds from SOURCE data (since translator maps target → source-like)
        self._compute_feature_bounds_from_source()

        # State
        # For val_acc: higher is better, init to 0.0
        # For val_loss: lower is better, init to inf
        self.best_val_metric = 0.0 if best_metric == "val_acc" else float("inf")
        self.best_state = None
        self.history = []
        self.memory_bank: Optional[MemoryBank] = None

    def _compute_feature_bounds_from_source(self):
        """Compute per-feature min/max from SOURCE training data for range loss.

        We use source bounds because the translator output should resemble source data.
        """
        all_data = []
        with torch.no_grad():
            for batch in self.source_train_loader:
                x = batch[0]  # (B, T, C)
                all_data.append(x)
        all_data = torch.cat(all_data, dim=0)  # (N, T, C)
        flat = all_data.reshape(-1, all_data.shape[-1])  # (N*T, C)
        self.lower_bounds = torch.quantile(flat, 0.001, dim=0).to(self.device)
        self.upper_bounds = torch.quantile(flat, 0.999, dim=0).to(self.device)
        logger.info(
            "Source feature bounds: lower=[%.3f, %.3f], upper=[%.3f, %.3f]",
            self.lower_bounds.min().item(), self.lower_bounds.max().item(),
            self.upper_bounds.min().item(), self.upper_bounds.max().item(),
        )

    def _verify_frozen(self):
        """Verify frozen model weights haven't changed."""
        for name, param in self.frozen_model.named_parameters():
            diff = (param.detach() - self._frozen_param_snapshot[name].to(param.device)).abs().max().item()
            if diff > 0:
                raise RuntimeError(f"FROZEN MODEL CORRUPTED: {name} changed by {diff:.2e}")
        logger.info("[verify] Frozen CNN model integrity OK")

    def _range_loss(self, x_translated: torch.Tensor) -> torch.Tensor:
        """Penalize translated values outside source feature bounds."""
        below = F.relu(self.lower_bounds - x_translated)
        above = F.relu(x_translated - self.upper_bounds)
        return (below + above).mean()

    def _smoothness_loss(self, x_translated: torch.Tensor) -> torch.Tensor:
        """Temporal smoothness: penalize large jumps between adjacent timesteps."""
        diff = x_translated[:, 1:, :] - x_translated[:, :-1, :]
        return diff.abs().mean()

    def _fidelity_loss(self, x_translated: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """Input fidelity: keep translation close to original input."""
        return F.mse_loss(x_translated, x_original)

    def _task_loss(self, x_translated: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss through frozen source CNN.

        Args:
            x_translated: (B, T, C) translated target data (should look source-like)
            labels: (B,) class labels (from target domain)
        """
        logits = self.frozen_model(x_translated)  # (B, num_classes)
        return F.cross_entropy(logits, labels)

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 1: Autoencoder pretrain on SOURCE domain
    # ═══════════════════════════════════════════════════════════════════

    def _pretrain_epoch(self) -> float:
        """One epoch of autoencoder pretraining on source domain.

        Phase 1 pretrains on SOURCE data so the encoder/decoder learns
        source domain structure. This primes the memory bank for source-style retrieval.
        """
        self.translator.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.source_train_loader:
            batch = tuple(b.to(self.device) for b in batch)
            parts = self.schema_resolver.extract(batch)

            x_val = parts["X_val"]
            x_miss = parts["X_miss"]
            t_abs = parts["t_abs"]
            m_pad = parts["M_pad"]
            x_static = parts["X_static"]

            # Autoencoder: encode source data -> reconstruct source data
            x_recon = self.translator(x_val, x_miss, t_abs, m_pad, x_static)
            loss = F.mse_loss(x_recon, x_val)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.translator.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def pretrain(self):
        """Phase 1: Autoencoder pretraining on SOURCE domain."""
        if self.pretrain_epochs <= 0:
            logger.info("Skipping Phase 1 pretrain (pretrain_epochs=0)")
            return

        logger.info(
            "=== Phase 1: Autoencoder pretrain on SOURCE domain (%d epochs) ===",
            self.pretrain_epochs,
        )
        pretrain_ckpt = self.run_dir / "pretrain_checkpoint.pt"

        # Reuse existing pretrain checkpoint if present
        if pretrain_ckpt.exists():
            state = torch.load(pretrain_ckpt, map_location=self.device, weights_only=False)
            self.translator.load_state_dict(state["translator"])
            logger.info("Loaded pretrain checkpoint from %s", pretrain_ckpt)
            return

        # Fall back to base dir pretrain checkpoint (for variant runs)
        if self.pretrain_fallback_dir is not None:
            fallback_ckpt = self.pretrain_fallback_dir / "pretrain_checkpoint.pt"
            if fallback_ckpt.exists():
                state = torch.load(fallback_ckpt, map_location=self.device, weights_only=False)
                self.translator.load_state_dict(state["translator"])
                logger.info("Loaded pretrain checkpoint from fallback: %s", fallback_ckpt)
                # Save a copy to the variant dir so future reruns find it locally
                torch.save(state, pretrain_ckpt)
                return

        for epoch in range(1, self.pretrain_epochs + 1):
            loss = self._pretrain_epoch()
            if epoch % 5 == 0 or epoch == 1:
                logger.info("[Pretrain] Epoch %d/%d: recon_loss=%.6f", epoch, self.pretrain_epochs, loss)

        torch.save({"translator": self.translator.state_dict()}, pretrain_ckpt)
        logger.info("Saved pretrain checkpoint to %s", pretrain_ckpt)

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 2: Task-guided training
    # ═══════════════════════════════════════════════════════════════════

    def _build_memory_bank(self):
        """Build memory bank from SOURCE training data.

        The memory bank stores source domain window latents.
        During inference, target data latents are used to query source-domain neighbors,
        and the retrieved context guides the decoder to produce source-like outputs.
        """
        self.memory_bank = build_memory_bank(
            encoder=self.translator,
            target_loader=self.source_train_loader,  # Use SOURCE data for bank
            schema_resolver=self.schema_resolver,
            device=self.device,
            window_size=self.retrieval_window,
        )
        logger.info(
            "Memory bank rebuilt from SOURCE data: %d windows",
            self.memory_bank.window_latents.shape[0],
        )

    def _train_epoch(self, target_train_loader: DataLoader) -> Dict[str, float]:
        """One epoch of Phase 2 training on TARGET domain data.

        Translates target data to be source-like, then evaluates through frozen source CNN.
        """
        self.translator.train()
        # CRITICAL: Frozen model must be in train() mode for cuDNN BatchNorm backward
        self.frozen_model.train()
        # But keep BN and Dropout in eval mode (freeze() sets this)
        for m in self.frozen_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.Dropout)):
                m.eval()

        losses_sum = {"task": 0.0, "fidelity": 0.0, "range": 0.0, "smooth": 0.0, "imp_reg": 0.0, "total": 0.0}
        n_batches = 0

        for batch in target_train_loader:
            batch = tuple(b.to(self.device) for b in batch)
            parts = self.schema_resolver.extract(batch)

            x_val = parts["X_val"]       # (B, T, C) target data
            x_miss = parts["X_miss"]
            t_abs = parts["t_abs"]
            m_pad = parts["M_pad"]
            x_static = parts["X_static"]
            y = parts["y"][:, 0]          # (B,) target labels

            # Encode target data
            latent = self.translator.encode(x_val, x_miss, t_abs, m_pad, x_static)

            # Query SOURCE memory bank with target latents
            # CRITICAL: detach before querying memory bank
            importance_w = self.translator.get_importance_weights()
            context = query_memory_bank(
                latent.detach(),
                m_pad, self.memory_bank,
                k_neighbors=self.k_neighbors,
                retrieval_window=self.retrieval_window,
                importance_weights=importance_w,
            )

            # Translate target → source-like using retrieved source context
            x_translated, _ = self.translator.forward_with_retrieval(
                x_val, x_miss, t_abs, m_pad, x_static, context, latent=latent,
            )

            # Task loss: cross-entropy on translated target through frozen source CNN
            task_loss = self._task_loss(x_translated, y)

            # Fidelity loss: keep translated close to original target input
            fidelity_loss = self._fidelity_loss(x_translated, x_val)

            # Range loss: translated output should be within source feature bounds
            range_loss = self._range_loss(x_translated)

            # Optional smoothness
            smooth_loss = self._smoothness_loss(x_translated) if self.lambda_smooth > 0 else torch.tensor(0.0)

            # Importance regularization
            imp_reg = importance_w.mean() if self.lambda_importance_reg > 0 else torch.tensor(0.0)

            total_loss = (
                task_loss
                + self.lambda_fidelity * fidelity_loss
                + self.lambda_range * range_loss
                + self.lambda_smooth * smooth_loss
                + self.lambda_importance_reg * imp_reg
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.translator.parameters(), 1.0)
            self.optimizer.step()

            losses_sum["task"] += task_loss.item()
            losses_sum["fidelity"] += fidelity_loss.item()
            losses_sum["range"] += range_loss.item()
            losses_sum["smooth"] += smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else 0.0
            losses_sum["imp_reg"] += imp_reg.item() if isinstance(imp_reg, torch.Tensor) else 0.0
            losses_sum["total"] += total_loss.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on target validation data (translated through frozen source CNN)."""
        self.translator.eval()
        # Frozen CNN: train() mode but BN/Dropout in eval()
        self.frozen_model.train()
        for m in self.frozen_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.Dropout)):
                m.eval()

        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(b.to(self.device) for b in batch)
                parts = self.schema_resolver.extract(batch)

                x_val = parts["X_val"]
                x_miss = parts["X_miss"]
                t_abs = parts["t_abs"]
                m_pad = parts["M_pad"]
                x_static = parts["X_static"]
                y = parts["y"][:, 0]

                # Encode and retrieve from source memory bank
                latent = self.translator.encode(x_val, x_miss, t_abs, m_pad, x_static)
                importance_w = self.translator.get_importance_weights()
                context = query_memory_bank(
                    latent, m_pad, self.memory_bank,
                    k_neighbors=self.k_neighbors,
                    retrieval_window=self.retrieval_window,
                    importance_weights=importance_w,
                )
                x_translated, _ = self.translator.forward_with_retrieval(
                    x_val, x_miss, t_abs, m_pad, x_static, context, latent=latent,
                )

                logits = self.frozen_model(x_translated)
                loss = F.cross_entropy(logits, y)
                preds = logits.argmax(dim=-1)

                correct += (preds == y).sum().item()
                total += y.shape[0]
                val_loss += loss.item() * y.shape[0]

        acc = correct / max(total, 1)
        return {"val_acc": acc, "val_loss": val_loss / max(total, 1)}

    def train(
        self,
        epochs: int,
        target_train_loader: DataLoader,
        target_val_loader: DataLoader,
    ):
        """Full training: Phase 1 (pretrain on source) + Phase 2 (translate target→source-like).

        Args:
            epochs: Number of Phase 2 training epochs
            target_train_loader: Target domain training DataLoader (Phase 2 task)
            target_val_loader: Target domain validation DataLoader (Phase 2 evaluation)
        """
        # Phase 1: pretrain autoencoder on source domain
        self.pretrain()

        # Phase 2: task-guided translation
        logger.info(
            "=== Phase 2: Task-guided retrieval training target→source (%d epochs) ===",
            epochs,
        )

        # Build initial SOURCE memory bank
        self._build_memory_bank()

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Rebuild memory bank periodically
            if epoch > 1 and (epoch - 1) % self.memory_refresh_epochs == 0:
                self._build_memory_bank()

            # Train: translate target → source-like
            train_losses = self._train_epoch(target_train_loader)

            # Validate: translated target through frozen source CNN
            val_metrics = self._validate(target_val_loader)

            elapsed = time.time() - t0

            logger.info(
                "[Epoch %d/%d] task=%.4f fid=%.4f range=%.4f | val_acc=%.4f val_loss=%.4f | %.1fs",
                epoch, epochs,
                train_losses["task"], train_losses["fidelity"], train_losses["range"],
                val_metrics["val_acc"], val_metrics["val_loss"],
                elapsed,
            )

            self.history.append({"epoch": epoch, **train_losses, **val_metrics})

            # Determine if current epoch is a new best
            current_metric = val_metrics[self.best_metric]
            if self.best_metric == "val_acc":
                is_new_best = current_metric > self.best_val_metric
            else:  # val_loss: lower is better
                is_new_best = current_metric < self.best_val_metric

            if is_new_best:
                self.best_val_metric = current_metric
                self.best_state = {k: v.clone() for k, v in self.translator.state_dict().items()}
                patience_counter = 0
                logger.info(
                    "  -> New best %s=%.4f (val_acc=%.4f val_loss=%.4f)",
                    self.best_metric, self.best_val_metric,
                    val_metrics["val_acc"], val_metrics["val_loss"],
                )

                torch.save({
                    "translator": self.translator.state_dict(),
                    "best_val_acc": val_metrics["val_acc"],
                    "best_val_loss": val_metrics["val_loss"],
                    "best_metric": self.best_metric,
                    "best_metric_value": self.best_val_metric,
                    "epoch": epoch,
                }, self.run_dir / "best_checkpoint.pt")
            else:
                patience_counter += 1
                if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.early_stopping_patience)
                    break

        if not self.use_last_epoch and self.best_state is not None:
            self.translator.load_state_dict(self.best_state)
            logger.info("Restored best model (%s=%.4f)", self.best_metric, self.best_val_metric)
        elif self.use_last_epoch:
            logger.info("AdaTime protocol: using last epoch model (not best val)")

        self._verify_frozen()

        torch.save({
            "translator": self.translator.state_dict(),
            "best_val_acc": self.best_val_metric if self.best_metric == "val_acc" else 0.0,
            "history": self.history,
        }, self.run_dir / "final_checkpoint.pt")
        logger.info("Training complete. Best %s=%.4f", self.best_metric, self.best_val_metric)


class AdaTimeFrozenDANNTrainer:
    """Frozen-model DANN baseline for AdaTime (fair comparison).

    Trains a small feature adapter (MLP) that transforms source features
    to match target domain, evaluated through frozen target LSTM.
    Domain adversarial training with gradient reversal.
    """

    def __init__(
        self,
        frozen_model: LSTMClassifier,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        lambda_domain: float = 1.0,
        device: str = "cuda",
    ):
        self.frozen_model = frozen_model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.lambda_domain = lambda_domain

        # Feature adapter: transforms source features to target-like features
        self.adapter = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_channels),
        ).to(device)

        # Domain discriminator on LSTM hidden states
        lstm_hidden = frozen_model.hidden_dim
        self.discriminator = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        ).to(device)

        self.optimizer_adapter = torch.optim.Adam(self.adapter.parameters(), lr=learning_rate)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)

    def train(
        self,
        epochs: int,
        source_train_loader: DataLoader,
        target_train_loader: DataLoader,
        source_val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Train the DANN adapter."""
        best_acc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.adapter.train()
            self.discriminator.train()

            target_iter = iter(target_train_loader)

            for batch in source_train_loader:
                x_src = batch[0].to(self.device)  # (B, T, C)
                y_src = batch[1][:, 0].to(self.device)

                try:
                    trg_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_train_loader)
                    trg_batch = next(target_iter)
                x_trg = trg_batch[0].to(self.device)

                B_src = x_src.shape[0]
                B_trg = x_trg.shape[0]

                # ── Step 1: Update discriminator ──
                B, T, C = x_src.shape
                with torch.no_grad():
                    x_adapted_detached = self.adapter(x_src.reshape(-1, C)).reshape(B, T, C)
                    src_features = self.frozen_model.extract_features(x_adapted_detached).float()
                    trg_features = self.frozen_model.extract_features(x_trg).float()

                domain_labels = torch.cat([
                    torch.zeros(B_src, dtype=torch.long, device=self.device),
                    torch.ones(B_trg, dtype=torch.long, device=self.device),
                ])
                domain_preds = self.discriminator(torch.cat([src_features.detach(), trg_features.detach()]))
                disc_loss = F.cross_entropy(domain_preds, domain_labels)

                self.optimizer_disc.zero_grad()
                disc_loss.backward()
                self.optimizer_disc.step()

                # ── Step 2: Update adapter (task loss + confuse discriminator) ──
                x_adapted = self.adapter(x_src.reshape(-1, C)).reshape(B, T, C)
                logits = self.frozen_model(x_adapted)
                task_loss = F.cross_entropy(logits, y_src)

                src_features_grad = self.frozen_model.extract_features(x_adapted).float()
                # Flipped labels: adapter wants discriminator to think source is target
                flipped_labels = torch.ones(B_src, dtype=torch.long, device=self.device)
                domain_preds_adapter = self.discriminator(src_features_grad)
                domain_confusion_loss = F.cross_entropy(domain_preds_adapter, flipped_labels)

                adapter_loss = task_loss + self.lambda_domain * domain_confusion_loss

                self.optimizer_adapter.zero_grad()
                adapter_loss.backward()
                self.optimizer_adapter.step()

            # Validate
            self.adapter.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in source_val_loader:
                    x = batch[0].to(self.device)
                    y = batch[1][:, 0].to(self.device)
                    B, T, C = x.shape
                    x_adapted = self.adapter(x.reshape(-1, C)).reshape(B, T, C)
                    logits = self.frozen_model(x_adapted)
                    correct += (logits.argmax(-1) == y).sum().item()
                    total += y.shape[0]

            val_acc = correct / max(total, 1)

            if epoch % 5 == 0 or epoch == 1:
                logger.info("[DANN] Epoch %d/%d: val_acc=%.4f", epoch, epochs, val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.clone() for k, v in self.adapter.state_dict().items()}

        if best_state:
            self.adapter.load_state_dict(best_state)

        return {"best_val_acc": best_acc}

    def translate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter to input data."""
        self.adapter.eval()
        B, T, C = x.shape
        with torch.no_grad():
            return self.adapter(x.reshape(-1, C)).reshape(B, T, C)


class ChunkedAdaTimeCNNRetrievalTrainer:
    """Retrieval translator trainer for full-length SSC/MFD sequences using chunking.

    Strategy:
      - Full 3000-timestep SSC sequences are split into non-overlapping 128-timestep chunks
      - Each chunk is independently translated by the retrieval translator
      - Translated chunks are concatenated and passed to the frozen CNN (adaptive pool handles any length)
      - Memory bank is built from source-domain chunks (same 128-timestep window as translator)
      - Phase 1 autoencoder pretrain processes source data as chunks

    Direction: TARGET → source-like (same as AdaTimeCNNRetrievalTrainer).
      - Frozen model: CNN trained on SOURCE domain
      - Memory bank: SOURCE domain chunks
      - Phase 2 task: translate TARGET chunks → source-like → classify with frozen source CNN
    """

    def __init__(
        self,
        frozen_model: "AdaTimeCNNClassifier",
        translator: RetrievalTranslator,
        schema_resolver: "AdaTimeSchemaResolver",
        source_train_loader: DataLoader,
        num_classes: int = 5,
        chunk_size: int = 128,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_recon: float = 0.1,
        lambda_range: float = 0.1,
        lambda_smooth: float = 0.0,
        lambda_importance_reg: float = 0.01,
        lambda_fidelity: float = 0.01,
        pretrain_epochs: int = 10,
        k_neighbors: int = 8,
        retrieval_window: int = 4,
        memory_refresh_epochs: int = 5,
        early_stopping_patience: int = 10,
        use_last_epoch: bool = False,
        run_dir: str = "runs/adatime_cnn",
        device: str = "cuda",
        optimizer_type: str = "adamw",
        optimizer_betas: tuple = (0.9, 0.999),
        context_aware: bool = False,
        drop_last_chunk: bool = False,
    ):
        """Initialize the chunked CNN retrieval trainer.

        Args:
            frozen_model: Source CNN classifier (frozen, trained on source domain)
            translator: RetrievalTranslator that will translate target chunks → source-like
            schema_resolver: Schema resolver for batch format extraction
            source_train_loader: SOURCE domain DataLoader with full-length sequences
                (for Phase 1 pretrain and memory bank, processed as chunks)
            num_classes: Number of output classes
            chunk_size: Size of each chunk (default 128). Source sequences are split into
                non-overlapping chunks of this size. Partial final chunks are padded.
            learning_rate: Translator optimizer learning rate
            weight_decay: Optimizer weight decay
            lambda_recon: Reconstruction loss weight (Phase 1)
            lambda_range: Range constraint loss weight
            lambda_smooth: Temporal smoothness loss weight
            lambda_importance_reg: Importance weight regularization
            lambda_fidelity: Input fidelity loss weight
            pretrain_epochs: Phase 1 autoencoder pretrain epochs
            k_neighbors: Number of retrieval neighbors
            retrieval_window: Retrieval window size
            memory_refresh_epochs: Rebuild memory bank every N epochs
            early_stopping_patience: Epochs without improvement before stopping
            run_dir: Directory for checkpoints
            device: Device ('cuda' or 'cpu')
            optimizer_type: "adam" or "adamw"
            optimizer_betas: Optimizer beta parameters
            context_aware: If True, each chunk's encoder sees the previous chunk as left
                context (2*chunk_size input, only current chunk output kept). This lets the
                encoder's self-attention see across chunk boundaries.
            drop_last_chunk: If True, drop partial final chunk instead of
                padding. Default False (pad with zeros). Both reproduce results
                within noise (65.9 vs 66.0 on SSC).
        """
        self.frozen_model = frozen_model.to(device)
        self.translator = translator.to(device)
        self.schema_resolver = schema_resolver
        self.source_train_loader = source_train_loader
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.context_aware = context_aware
        self.drop_last_chunk = drop_last_chunk
        self.device = device

        # Loss weights
        self.lambda_recon = lambda_recon
        self.lambda_range = lambda_range
        self.lambda_smooth = lambda_smooth
        self.lambda_importance_reg = lambda_importance_reg
        self.lambda_fidelity = lambda_fidelity

        # Training params
        self.pretrain_epochs = pretrain_epochs
        self.k_neighbors = k_neighbors
        self.retrieval_window = retrieval_window
        self.memory_refresh_epochs = memory_refresh_epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_last_epoch = use_last_epoch  # AdaTime convention: use last epoch, not best

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer for translator
        OptimClass = Adam if optimizer_type.lower() == "adam" else AdamW
        self.optimizer = OptimClass(
            self.translator.parameters(), lr=learning_rate,
            weight_decay=weight_decay, betas=optimizer_betas,
        )
        self.scaler = GradScaler(enabled=device.startswith("cuda"))

        # Snapshot frozen model params for verification
        self._frozen_param_snapshot = {
            name: param.detach().clone()
            for name, param in self.frozen_model.named_parameters()
        }

        # Bank window size: use retrieval_window (e.g., 8) NOT chunk_size.
        # With window_size=8, each 128-timestep chunk → 128/8=16 windows.
        # Context shape: (B*chunk_size, K*8, d_latent) → manageable.
        # retrieval_window must be <= chunk_size.
        self._bank_window_size = min(retrieval_window, chunk_size)

        # Compute feature bounds from SOURCE data chunks
        self._compute_feature_bounds_from_source()

        if self.context_aware:
            logger.info("Context-aware chunking ENABLED: encoder sees previous chunk as left context")

        # State
        self.best_val_metric = 0.0
        self.best_state = None
        self.history = []
        self.memory_bank: Optional[MemoryBank] = None

    # ═══════════════════════════════════════════════════════════════════
    #  Chunking utilities
    # ═══════════════════════════════════════════════════════════════════

    def _split_into_chunks(self, x: torch.Tensor) -> torch.Tensor:
        """Split a batch of sequences into chunks.

        Args:
            x: (B, T, C) full-length sequences

        Returns:
            chunks: (B * n_chunks, chunk_size, C). If drop_last_chunk=True,
                partial final chunk is dropped. Otherwise it is zero-padded.
        """
        B, T, C = x.shape
        n_full_chunks = T // self.chunk_size
        remainder = T % self.chunk_size

        if n_full_chunks == 0 and remainder == 0:
            raise ValueError(
                f"Sequence length {T} is shorter than chunk_size {self.chunk_size}. "
                f"Use a smaller chunk_size."
            )

        if self.drop_last_chunk or remainder == 0:
            # Drop partial final chunk (original behavior)
            n_chunks = n_full_chunks
            x = x[:, :n_chunks * self.chunk_size, :]
        else:
            # Pad last partial chunk to full chunk_size
            pad_size = self.chunk_size - remainder
            x = F.pad(x, (0, 0, 0, pad_size))  # pad timestep dim: (B, T+pad, C)
            n_chunks = n_full_chunks + 1

        # Reshape to chunks
        chunks = x.reshape(B, n_chunks, self.chunk_size, C)
        # Flatten batch and chunk dims: (B * n_chunks, chunk_size, C)
        return chunks.reshape(B * n_chunks, self.chunk_size, C)

    def _translate_sequence_chunked(
        self,
        x: torch.Tensor,
        x_miss: torch.Tensor,
        t_abs_seq: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Translate a batch of full-length sequences by chunking.

        Each sequence is split into chunks (padding the last partial chunk).
        When context_aware=True, each chunk's encoder also sees the previous
        chunk as left context, but only the current chunk's output is kept.

        Args:
            x: (B, T, C) full-length sequences
            x_miss: (B, T, C) missing indicators (all zeros for AdaTime)
            t_abs_seq: (B, T) absolute time indices
            m_pad: (B, T) padding mask (all False for AdaTime)
            x_static: (B, S) static features

        Returns:
            x_translated: (B, T, C) translated sequences (same length as input)
        """
        B, T_original, C = x.shape
        n_full_chunks = T_original // self.chunk_size
        remainder = T_original % self.chunk_size
        if self.drop_last_chunk or remainder == 0:
            n_chunks = n_full_chunks
        else:
            n_chunks = n_full_chunks + 1

        if not self.context_aware:
            # ── Standard (non-context-aware) path: all chunks at once ──
            x_chunks = self._split_into_chunks(x)
            x_miss_chunks = self._split_into_chunks(x_miss)

            # Align t_abs and m_pad to match chunk count
            if self.drop_last_chunk or remainder == 0:
                t_abs_trimmed = t_abs_seq[:, :n_chunks * self.chunk_size]
                m_pad_trimmed = m_pad[:, :n_chunks * self.chunk_size]
                t_abs_chunks = t_abs_trimmed.reshape(B * n_chunks, self.chunk_size)
                m_pad_chunks = m_pad_trimmed.reshape(B * n_chunks, self.chunk_size)
            else:
                pad_size = self.chunk_size - remainder
                t_abs_padded = F.pad(t_abs_seq, (0, pad_size))
                m_pad_padded = F.pad(m_pad, (0, pad_size), value=True)  # True = padded
                t_abs_chunks = t_abs_padded.reshape(B * n_chunks, self.chunk_size)
                m_pad_chunks = m_pad_padded.reshape(B * n_chunks, self.chunk_size)

            # Repeat static features for each chunk
            x_static_chunks = x_static.unsqueeze(1).expand(B, n_chunks, -1).reshape(
                B * n_chunks, x_static.shape[-1]
            )

            # Encode all chunks at once
            latent_chunks = self.translator.encode(
                x_chunks, x_miss_chunks, t_abs_chunks, m_pad_chunks, x_static_chunks,
            )

            # Retrieve from chunk-level latent bank
            context = self._query_chunk_bank(latent_chunks.detach())

            # Translate chunks
            x_translated_chunks, _ = self.translator.forward_with_retrieval(
                x_chunks, x_miss_chunks, t_abs_chunks, m_pad_chunks, x_static_chunks,
                context, latent=latent_chunks,
            )

            # Reshape back and strip padding / trim to original length
            x_translated = x_translated_chunks.reshape(B, n_chunks * self.chunk_size, C)
            if n_chunks * self.chunk_size != T_original:
                x_translated = x_translated[:, :T_original, :]
            return x_translated

        else:
            # ── Context-aware path: each chunk sees previous chunk as left context ──
            # Pad the full sequence so all chunks are chunk_size aligned
            if remainder > 0:
                pad_size = self.chunk_size - remainder
                x_padded = F.pad(x, (0, 0, 0, pad_size))
                x_miss_padded = F.pad(x_miss, (0, 0, 0, pad_size))
                t_abs_padded = F.pad(t_abs_seq, (0, pad_size))
                m_pad_padded = F.pad(m_pad, (0, pad_size), value=True)
            else:
                x_padded = x
                x_miss_padded = x_miss
                t_abs_padded = t_abs_seq
                m_pad_padded = m_pad

            T_padded = n_chunks * self.chunk_size
            ctx_size = self.chunk_size  # left context = one full chunk

            translated_chunks = []
            for i in range(n_chunks):
                chunk_start = i * self.chunk_size
                chunk_end = chunk_start + self.chunk_size
                ctx_start = max(0, chunk_start - ctx_size)

                # Extract context + current chunk window
                x_win = x_padded[:, ctx_start:chunk_end, :]       # (B, win_len, C)
                x_miss_win = x_miss_padded[:, ctx_start:chunk_end, :]
                t_abs_win = t_abs_padded[:, ctx_start:chunk_end]
                m_pad_win = m_pad_padded[:, ctx_start:chunk_end]

                win_len = x_win.shape[1]
                x_static_exp = x_static  # (B, S) — no expansion needed

                # Encode the full window (context + current chunk)
                latent_win = self.translator.encode(
                    x_win, x_miss_win, t_abs_win, m_pad_win, x_static_exp,
                )

                # Retrieve using only the current chunk's latent (last chunk_size timesteps)
                latent_chunk = latent_win[:, -self.chunk_size:, :]
                context = self._query_chunk_bank(latent_chunk.detach())

                # Translate the full window
                x_trans_win, _ = self.translator.forward_with_retrieval(
                    x_win, x_miss_win, t_abs_win, m_pad_win, x_static_exp,
                    # Context needs to match win_len, not just chunk_size.
                    # Pad context on the left with zeros for the context timesteps.
                    self._expand_context_for_window(context, win_len),
                    latent=latent_win,
                )

                # Keep only the current chunk portion (last chunk_size timesteps)
                translated_chunks.append(x_trans_win[:, -self.chunk_size:, :])

            # Concatenate and strip padding
            x_translated = torch.cat(translated_chunks, dim=1)  # (B, T_padded, C)
            if remainder > 0:
                x_translated = x_translated[:, :T_original, :]
            return x_translated

    def _expand_context_for_window(
        self, context: torch.Tensor, win_len: int,
    ) -> torch.Tensor:
        """Expand chunk-level context to match a larger window length.

        Context is (B, chunk_size, K, d). For the context-aware window of
        win_len > chunk_size, we prepend zero-context for the left-context timesteps.

        Args:
            context: (B, chunk_size, K, d_latent) retrieved context for the current chunk
            win_len: Total window length (context_len + chunk_size)

        Returns:
            expanded: (B, win_len, K, d_latent)
        """
        if win_len == context.shape[1]:
            return context
        B, T_ctx, K, d = context.shape
        prefix_len = win_len - T_ctx
        prefix = torch.zeros(B, prefix_len, K, d, device=context.device, dtype=context.dtype)
        return torch.cat([prefix, context], dim=1)

    def _build_chunk_loader(self, full_length_loader: DataLoader) -> DataLoader:
        """Build a DataLoader that yields individual chunks instead of full sequences.

        Used for Phase 1 pretrain: source sequences are split into chunks and
        each chunk is treated as an independent sample.

        Args:
            full_length_loader: DataLoader yielding (B, T, C) full-length sequences

        Returns:
            A DataLoader where each item is a (chunk_size, C) chunk
        """
        from torch.utils.data import TensorDataset

        all_chunks = []
        all_labels = []
        all_statics = []

        with torch.no_grad():
            for batch in full_length_loader:
                x, y_seq, pad_mask, static = batch  # (B, T, C), (B, T), (B, T), (B, S)
                B, T, C = x.shape
                n_full = T // self.chunk_size
                remainder = T % self.chunk_size
                if self.drop_last_chunk or remainder == 0:
                    n_chunks = n_full
                else:
                    n_chunks = n_full + 1
                if n_chunks == 0:
                    continue
                # Split: (B * n_chunks, chunk_size, C)
                x_chunked = self._split_into_chunks(x)
                # Labels: use per-sequence label for all chunks of that sequence
                y_seq_label = y_seq[:, 0]  # (B,)
                y_repeated = y_seq_label.unsqueeze(1).expand(B, n_chunks).reshape(B * n_chunks)
                y_chunks_seq = y_repeated.unsqueeze(1).expand(B * n_chunks, self.chunk_size)
                # Pad mask: True = valid for full chunks, last chunk has partial padding
                pad_chunks = torch.ones(B * n_chunks, self.chunk_size, dtype=torch.bool)
                if remainder > 0 and not self.drop_last_chunk:
                    # Mark padded timesteps in the last chunk of each sequence as invalid
                    pad_size = self.chunk_size - remainder
                    for b in range(B):
                        last_chunk_idx = b * n_chunks + (n_chunks - 1)
                        pad_chunks[last_chunk_idx, remainder:] = False
                # Static: repeat for each chunk
                static_chunks = static.unsqueeze(1).expand(B, n_chunks, -1).reshape(
                    B * n_chunks, static.shape[-1]
                )

                all_chunks.append(x_chunked)
                all_labels.append(y_chunks_seq)
                all_statics.append(static_chunks)

        if not all_chunks:
            raise ValueError("No chunks extracted from loader — check chunk_size vs sequence length")

        all_chunks = torch.cat(all_chunks, dim=0)      # (N_chunks, chunk_size, C)
        all_labels = torch.cat(all_labels, dim=0)      # (N_chunks, chunk_size)
        # AdaTime convention: pad_mask True = valid. Schema resolver inverts → M_pad=False=not padded.
        all_pad = torch.ones(all_chunks.shape[0], self.chunk_size, dtype=torch.bool)
        all_statics = torch.cat(all_statics, dim=0)    # (N_chunks, S)

        chunk_ds = TensorDataset(all_chunks, all_labels, all_pad, all_statics)
        # Use larger batch size for chunk DataLoader — each chunk is independent,
        # so we can use 32 regardless of the original sequence batch_size.
        chunk_batch_size = max(32, full_length_loader.batch_size or 32)
        return DataLoader(
            chunk_ds, batch_size=chunk_batch_size,
            shuffle=True, drop_last=False, num_workers=0,
        )

    def _compute_feature_bounds_from_source(self):
        """Compute per-feature min/max from SOURCE chunks for range loss."""
        all_data = []
        with torch.no_grad():
            for batch in self.source_train_loader:
                x = batch[0]  # (B, T, C)
                B, T, C = x.shape
                n_chunks = T // self.chunk_size
                if n_chunks == 0:
                    continue
                x_chunks = self._split_into_chunks(x)  # (B*n_chunks, chunk_size, C)
                all_data.append(x_chunks)
        all_data = torch.cat(all_data, dim=0)  # (N, chunk_size, C)
        flat = all_data.reshape(-1, all_data.shape[-1])  # (N*chunk_size, C)
        self.lower_bounds = torch.quantile(flat, 0.001, dim=0).to(self.device)
        self.upper_bounds = torch.quantile(flat, 0.999, dim=0).to(self.device)
        logger.info(
            "Source chunk feature bounds: lower=[%.3f, %.3f], upper=[%.3f, %.3f]",
            self.lower_bounds.min().item(), self.lower_bounds.max().item(),
            self.upper_bounds.min().item(), self.upper_bounds.max().item(),
        )

    def _verify_frozen(self):
        """Verify frozen model weights haven't changed."""
        for name, param in self.frozen_model.named_parameters():
            diff = (param.detach() - self._frozen_param_snapshot[name].to(param.device)).abs().max().item()
            if diff > 0:
                raise RuntimeError(f"FROZEN MODEL CORRUPTED: {name} changed by {diff:.2e}")
        logger.info("[verify] Frozen CNN model integrity OK")

    def _range_loss(self, x_translated: torch.Tensor) -> torch.Tensor:
        below = F.relu(self.lower_bounds - x_translated)
        above = F.relu(x_translated - self.upper_bounds)
        return (below + above).mean()

    def _smoothness_loss(self, x_translated: torch.Tensor) -> torch.Tensor:
        diff = x_translated[:, 1:, :] - x_translated[:, :-1, :]
        return diff.abs().mean()

    def _fidelity_loss(self, x_translated: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_translated, x_original)

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 1: Autoencoder pretrain on SOURCE domain chunks
    # ═══════════════════════════════════════════════════════════════════

    def _pretrain_epoch(self, chunk_loader: DataLoader) -> float:
        """One epoch of autoencoder pretraining on source-domain chunks."""
        self.translator.train()
        total_loss = 0.0
        n_batches = 0

        for batch in chunk_loader:
            batch = tuple(b.to(self.device) for b in batch)
            parts = self.schema_resolver.extract(batch)

            x_val = parts["X_val"]    # (B_chunk, chunk_size, C)
            x_miss = parts["X_miss"]
            t_abs = parts["t_abs"]
            m_pad = parts["M_pad"]
            x_static = parts["X_static"]

            x_recon = self.translator(x_val, x_miss, t_abs, m_pad, x_static)
            loss = F.mse_loss(x_recon, x_val)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.translator.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def pretrain(self):
        """Phase 1: Autoencoder pretraining on source-domain chunks."""
        if self.pretrain_epochs <= 0:
            logger.info("Skipping Phase 1 pretrain (pretrain_epochs=0)")
            return

        logger.info(
            "=== Phase 1: Autoencoder pretrain on SOURCE chunks (%d epochs, chunk_size=%d) ===",
            self.pretrain_epochs, self.chunk_size,
        )
        pretrain_ckpt = self.run_dir / "pretrain_checkpoint.pt"

        if pretrain_ckpt.exists():
            state = torch.load(pretrain_ckpt, map_location=self.device, weights_only=False)
            self.translator.load_state_dict(state["translator"])
            logger.info("Loaded pretrain checkpoint from %s", pretrain_ckpt)
            return

        # Build chunk DataLoader from source sequences
        logger.info("Building source chunk DataLoader for Phase 1...")
        chunk_loader = self._build_chunk_loader(self.source_train_loader)
        logger.info("Phase 1: %d chunk batches per epoch", len(chunk_loader))

        for epoch in range(1, self.pretrain_epochs + 1):
            loss = self._pretrain_epoch(chunk_loader)
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "[Pretrain] Epoch %d/%d: recon_loss=%.6f", epoch, self.pretrain_epochs, loss,
                )

        torch.save({"translator": self.translator.state_dict()}, pretrain_ckpt)
        logger.info("Saved pretrain checkpoint to %s", pretrain_ckpt)

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 2: Build memory bank from SOURCE chunks
    # ═══════════════════════════════════════════════════════════════════

    def _build_chunk_latent_bank(self):
        """Build a simple chunk-level latent bank (one vector per source chunk).

        Unlike the standard MemoryBank (which creates W windows per chunk),
        this builds a flat (N_chunks, d_latent) tensor of mean-pooled chunk latents.
        This keeps memory manageable: 31648 × d_latent for all 10 SSC scenarios.

        Retrieval: for each target chunk, find K nearest source chunks by
        Euclidean distance in latent space and return their full latent sequences.
        """
        chunk_loader = self._build_chunk_loader(self.source_train_loader)
        chunk_latents = []
        chunk_data = []

        self.translator.eval()
        with torch.no_grad():
            for batch in chunk_loader:
                batch = tuple(b.to(self.device) for b in batch)
                parts = self.schema_resolver.extract(batch)
                x_val = parts["X_val"]
                x_miss = parts["X_miss"]
                t_abs = parts["t_abs"]
                m_pad = parts["M_pad"]
                x_static = parts["X_static"]

                # Encode: (B_chunk, T, d_latent)
                latent = self.translator.encode(x_val, x_miss, t_abs, m_pad, x_static)
                # Mean-pool to get one vector per chunk: (B_chunk, d_latent)
                mean_latent = latent.mean(dim=1)
                chunk_latents.append(mean_latent.cpu())
                chunk_data.append(latent.cpu())  # Store full latent sequences

        self._chunk_bank_latents = torch.cat(chunk_latents, dim=0).to(self.device)  # (N, d_latent)
        self._chunk_bank_sequences = torch.cat(chunk_data, dim=0)  # (N, T, d_latent) — stays on CPU
        logger.info(
            "Chunk latent bank built: %d chunks, bank_latents=%s (GPU %.1f MB)",
            self._chunk_bank_latents.shape[0],
            tuple(self._chunk_bank_latents.shape),
            self._chunk_bank_latents.numel() * 4 / 1e6,
        )

    def _query_chunk_bank(self, query_latent: torch.Tensor) -> torch.Tensor:
        """Query the chunk-level latent bank.

        Args:
            query_latent: (B_chunks, T, d_latent) encoded target chunk latents

        Returns:
            context: (B_chunks, T, K, d_latent) retrieved source latent sequences
                     per-timestep, each timestep gets K source chunks as context.
        """
        B_chunks, T, d = query_latent.shape
        K = self.k_neighbors
        N = self._chunk_bank_latents.shape[0]

        # Mean-pool query: (B_chunks, d_latent)
        query_mean = query_latent.mean(dim=1)  # (B_chunks, d_latent)

        # Compute L2 distances: (B_chunks, N)
        # Use chunked computation to avoid OOM with large N
        CHUNK = 512
        topk_indices = torch.zeros(B_chunks, K, dtype=torch.long, device=self.device)
        for start in range(0, B_chunks, CHUNK):
            end = min(start + CHUNK, B_chunks)
            q_chunk = query_mean[start:end]  # (c, d)
            # (c, d) vs (N, d) -> (c, N)
            diff = q_chunk.unsqueeze(1) - self._chunk_bank_latents.unsqueeze(0)  # (c, N, d)
            dists = diff.pow(2).sum(dim=-1)  # (c, N)
            _, top_idx = dists.topk(K, dim=-1, largest=False)  # (c, K)
            topk_indices[start:end] = top_idx

        # Gather source chunk latent sequences: (B_chunks, K, T, d_latent)
        # bank_sequences is on CPU to save GPU memory
        bank_seqs = self._chunk_bank_sequences  # (N, T, d_latent) on CPU
        # Move top-K sequences to GPU: (B_chunks, K, T, d_latent)
        gathered = bank_seqs[topk_indices.cpu()].to(self.device)  # (B_chunks, K, T, d)

        # Context: for each query timestep, provide K source chunk latent sequences
        # Reshape to (B_chunks, T, K*T_bank, d_latent) — but that's K*T=K*128 entries
        # Instead use per-timestep: (B_chunks, T, K, d_latent) using mean-pooled bank seqs
        # This gives each query timestep one context vector per retrieved chunk
        gathered_mean = gathered.mean(dim=2)  # (B_chunks, K, d) — mean-pool source chunk latents
        # Expand to per-timestep: (B_chunks, T, K, d_latent)
        context = gathered_mean.unsqueeze(1).expand(B_chunks, T, K, d)  # (B_chunks, T, K, d)
        return context

    def _build_memory_bank(self):
        """Build memory bank: delegates to _build_chunk_latent_bank."""
        self._build_chunk_latent_bank()

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 2 training
    # ═══════════════════════════════════════════════════════════════════

    def _train_epoch(self, target_train_loader: DataLoader) -> Dict[str, float]:
        """One epoch of Phase 2: translate TARGET full sequences by chunking."""
        self.translator.train()
        self.frozen_model.train()
        for m in self.frozen_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.Dropout)):
                m.eval()

        losses_sum = {
            "task": 0.0, "fidelity": 0.0, "range": 0.0,
            "smooth": 0.0, "imp_reg": 0.0, "total": 0.0,
        }
        n_batches = 0

        for batch in target_train_loader:
            batch = tuple(b.to(self.device) for b in batch)
            parts = self.schema_resolver.extract(batch)

            x_val = parts["X_val"]        # (B, T, C) full-length target
            x_miss = parts["X_miss"]      # (B, T, C)
            t_abs = parts["t_abs"]        # (B, T)
            m_pad = parts["M_pad"]        # (B, T)
            x_static = parts["X_static"]  # (B, S)
            y = parts["y"][:, 0]          # (B,) per-sequence label

            B, T, C = x_val.shape
            if T < self.chunk_size:
                continue

            # Translate full sequence via chunking
            x_translated = self._translate_sequence_chunked(
                x_val, x_miss, t_abs, m_pad, x_static,
            )
            # x_translated: (B, T', C) — may be shorter than input if drop_last_chunk

            # Trim original to match translated length (drop_last_chunk trims tail)
            T_trans = x_translated.shape[1]
            x_val_matched = x_val[:, :T_trans, :]

            # Task loss: cross-entropy through frozen source CNN
            # CNN uses AdaptiveAvgPool1d so any length works
            task_loss = F.cross_entropy(self.frozen_model(x_translated), y)

            # Fidelity loss: translated vs original (matched length)
            fidelity_loss = self._fidelity_loss(x_translated, x_val_matched)

            # Range loss
            range_loss = self._range_loss(x_translated)

            # Smoothness (optional)
            smooth_loss = (
                self._smoothness_loss(x_translated) if self.lambda_smooth > 0
                else torch.tensor(0.0, device=self.device)
            )

            # Importance regularization
            importance_w = self.translator.get_importance_weights()
            imp_reg = (
                importance_w.mean() if self.lambda_importance_reg > 0
                else torch.tensor(0.0, device=self.device)
            )

            total_loss = (
                task_loss
                + self.lambda_fidelity * fidelity_loss
                + self.lambda_range * range_loss
                + self.lambda_smooth * smooth_loss
                + self.lambda_importance_reg * imp_reg
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.translator.parameters(), 1.0)
            self.optimizer.step()

            losses_sum["task"] += task_loss.item()
            losses_sum["fidelity"] += fidelity_loss.item()
            losses_sum["range"] += range_loss.item()
            losses_sum["smooth"] += smooth_loss.item()
            losses_sum["imp_reg"] += imp_reg.item()
            losses_sum["total"] += total_loss.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on TARGET validation data (chunked translation → frozen source CNN)."""
        self.translator.eval()
        self.frozen_model.train()
        for m in self.frozen_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.Dropout)):
                m.eval()

        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(b.to(self.device) for b in batch)
                parts = self.schema_resolver.extract(batch)

                x_val = parts["X_val"]
                x_miss = parts["X_miss"]
                t_abs = parts["t_abs"]
                m_pad = parts["M_pad"]
                x_static = parts["X_static"]
                y = parts["y"][:, 0]

                B, T, C = x_val.shape
                if T < self.chunk_size:
                    continue

                x_translated = self._translate_sequence_chunked(
                    x_val, x_miss, t_abs, m_pad, x_static,
                )

                logits = self.frozen_model(x_translated)
                loss = F.cross_entropy(logits, y)
                preds = logits.argmax(dim=-1)

                correct += (preds == y).sum().item()
                total += y.shape[0]
                val_loss += loss.item() * y.shape[0]

        acc = correct / max(total, 1)
        return {"val_acc": acc, "val_loss": val_loss / max(total, 1)}

    def train(
        self,
        epochs: int,
        target_train_loader: DataLoader,
        target_val_loader: DataLoader,
    ):
        """Full training: Phase 1 (pretrain on source chunks) + Phase 2 (chunked translation).

        Args:
            epochs: Number of Phase 2 training epochs
            target_train_loader: Target domain training DataLoader (full-length sequences)
            target_val_loader: Target domain validation DataLoader (full-length sequences)
        """
        # Phase 1: pretrain on source chunks
        self.pretrain()

        # Phase 2: task-guided chunked translation
        logger.info(
            "=== Phase 2: Chunked translation target→source (%d epochs, chunk_size=%d) ===",
            epochs, self.chunk_size,
        )

        # Build initial SOURCE memory bank from chunks
        self._build_memory_bank()

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Rebuild memory bank periodically
            if epoch > 1 and (epoch - 1) % self.memory_refresh_epochs == 0:
                self._build_memory_bank()

            train_losses = self._train_epoch(target_train_loader)
            val_metrics = self._validate(target_val_loader)
            elapsed = time.time() - t0

            logger.info(
                "[Epoch %d/%d] task=%.4f fid=%.4f range=%.4f | val_acc=%.4f val_loss=%.4f | %.1fs",
                epoch, epochs,
                train_losses["task"], train_losses["fidelity"], train_losses["range"],
                val_metrics["val_acc"], val_metrics["val_loss"],
                elapsed,
            )

            self.history.append({"epoch": epoch, **train_losses, **val_metrics})

            if val_metrics["val_acc"] > self.best_val_metric:
                self.best_val_metric = val_metrics["val_acc"]
                self.best_state = {k: v.clone() for k, v in self.translator.state_dict().items()}
                patience_counter = 0
                logger.info("  -> New best val_acc=%.4f", self.best_val_metric)

                torch.save({
                    "translator": self.translator.state_dict(),
                    "best_val_acc": self.best_val_metric,
                    "epoch": epoch,
                    "chunk_size": self.chunk_size,
                }, self.run_dir / "best_checkpoint.pt")
            else:
                patience_counter += 1
                if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.early_stopping_patience)
                    break

        if not self.use_last_epoch and self.best_state is not None:
            self.translator.load_state_dict(self.best_state)
            logger.info("Restored best model (val_acc=%.4f)", self.best_val_metric)
        elif self.use_last_epoch:
            logger.info("AdaTime protocol: using last epoch model (not best val)")

        self._verify_frozen()

        torch.save({
            "translator": self.translator.state_dict(),
            "best_val_acc": self.best_val_metric,
            "history": self.history,
            "chunk_size": self.chunk_size,
        }, self.run_dir / "final_checkpoint.pt")
        logger.info("Training complete. Best val_acc=%.4f", self.best_val_metric)
