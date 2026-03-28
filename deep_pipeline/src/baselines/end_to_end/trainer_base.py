"""Shared trainer base for end-to-end DA baselines (CLUDA, RAINCOAT, ACON).

Provides: training loop, early stopping, checkpointing (best by val AUROC),
logging, class-weighted BCE loss, AMP support.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader


class E2EBaselineTrainer:
    """Base trainer for end-to-end DA baselines.

    Subclasses must implement ``_train_epoch(epoch)`` which returns a dict of
    training metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        source_train_loader: DataLoader,
        target_train_loader: DataLoader,
        source_val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.source_train_loader = source_train_loader
        self.target_train_loader = target_train_loader
        self.source_val_loader = source_val_loader
        self.device = device
        self.config = config

        training = config.get("training", {})
        self.epochs = training.get("epochs", 50)
        self.lr = training.get("lr", 1e-3)
        self.weight_decay = training.get("weight_decay", 1e-4)
        self.early_stopping_patience = training.get("early_stopping_patience", 15)

        run_dir = config.get("output", {}).get("run_dir", "runs/e2e_baseline")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # AMP
        self.scaler = GradScaler(enabled=device.startswith("cuda"))
        self.use_amp = device.startswith("cuda")

        # Class-weighted BCE: compute weights from source training set
        self._pos_weight = self._compute_pos_weight(source_train_loader)
        logging.info("[E2ETrainer] pos_weight=%.4f", self._pos_weight.item())

        # Tracking
        self.best_val_auroc = -float("inf")
        self.best_epoch = -1
        self._patience_counter = 0

    def _compute_pos_weight(self, loader: DataLoader) -> torch.Tensor:
        """Compute pos_weight for weighted BCE from label distribution."""
        total_pos = 0
        total_neg = 0
        for batch in loader:
            labels = batch[1]  # y is second element
            valid = labels >= 0
            total_pos += (labels[valid] > 0.5).sum().item()
            total_neg += (labels[valid] <= 0.5).sum().item()
        if total_pos == 0:
            return torch.tensor(1.0, device=self.device)
        weight = total_neg / max(total_pos, 1)
        return torch.tensor(min(weight, 20.0), device=self.device)  # cap at 20x

    def classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted BCE with logits."""
        return F.binary_cross_entropy_with_logits(
            logits.view(-1), labels.view(-1).float(),
            pos_weight=self._pos_weight,
        )

    def train(self, epochs: int | None = None):
        """Main training loop with early stopping and checkpointing."""
        epochs = epochs or self.epochs
        logging.info("[E2ETrainer] Starting training for %d epochs", epochs)

        for epoch in range(epochs):
            t0 = time.time()
            self.model.train()

            train_metrics = self._train_epoch(epoch)
            train_time = time.time() - t0

            # Validate
            val_metrics = self._validate()

            # Log
            parts = [f"Ep {epoch+1}/{epochs}"]
            for k, v in train_metrics.items():
                parts.append(f"{k}={v:.4f}")
            parts.append(f"val_auroc={val_metrics['auroc']:.4f}")
            parts.append(f"val_aucpr={val_metrics['aucpr']:.4f}")
            parts.append(f"val_loss={val_metrics['loss']:.4f}")
            parts.append(f"t={train_time:.1f}s")
            logging.info("[E2ETrainer] %s", "  ".join(parts))

            # Checkpoint best model
            improved = val_metrics["auroc"] > self.best_val_auroc
            if improved:
                self.best_val_auroc = val_metrics["auroc"]
                self.best_epoch = epoch + 1
                self._patience_counter = 0
                self._save_checkpoint("best_checkpoint.pt", epoch, val_metrics)
                logging.info(
                    "[E2ETrainer] New best: AUROC=%.4f at epoch %d",
                    self.best_val_auroc, self.best_epoch,
                )
            else:
                self._patience_counter += 1

            # Save latest checkpoint every epoch for resume
            self._save_checkpoint("latest_checkpoint.pt", epoch, val_metrics)

            # Early stopping
            if (
                self.early_stopping_patience > 0
                and self._patience_counter >= self.early_stopping_patience
            ):
                logging.info(
                    "[E2ETrainer] Early stopping at epoch %d (patience=%d, best=%d)",
                    epoch + 1, self.early_stopping_patience, self.best_epoch,
                )
                break

        # Load best checkpoint for evaluation
        best_path = self.run_dir / "best_checkpoint.pt"
        if best_path.exists():
            self._load_checkpoint(best_path)
            logging.info("[E2ETrainer] Loaded best model from epoch %d", self.best_epoch)

    def _train_epoch(self, epoch: int) -> dict:
        """Implement in subclass. Returns dict of training metrics."""
        raise NotImplementedError

    def _eval_loop(self, loader: DataLoader) -> dict:
        """Shared eval loop for validate and test. Handles both per-stay and per-timestep."""
        self.model.eval()
        all_probs = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                x, labels, static, vmask = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                static = static.to(self.device)
                vmask = vmask.to(self.device)

                if labels.dim() == 1:
                    # Per-stay: scalar labels (B,)
                    logits = self.model.predict(x, static)  # (B,)
                    loss = self.classification_loss(logits, labels)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    lbls = labels.cpu().numpy()
                    valid = lbls >= 0
                else:
                    # Per-timestep: (B, L) labels
                    logits = self.model.predict_per_timestep(x, static)  # (B, L)
                    # Mask invalid timesteps
                    valid_mask = vmask & (labels >= 0)
                    if valid_mask.sum() > 0:
                        loss = F.binary_cross_entropy_with_logits(
                            logits[valid_mask], labels[valid_mask],
                            pos_weight=self._pos_weight,
                        )
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                    probs = torch.sigmoid(logits).cpu().numpy().ravel()
                    lbls = labels.cpu().numpy().ravel()
                    valid_np = (vmask & (labels >= 0)).cpu().numpy().ravel()
                    valid = valid_np.astype(bool)

                if valid.sum() > 0:
                    all_probs.append(probs[valid] if probs.ndim == 1 else probs[valid])
                    all_labels.append(lbls[valid] if lbls.ndim == 1 else lbls[valid])
                total_loss += loss.item()
                n_batches += 1

        if len(all_labels) == 0:
            return {"auroc": 0.0, "aucpr": 0.0, "loss": float("inf")}

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.5
        try:
            aucpr = average_precision_score(all_labels, all_probs)
        except ValueError:
            aucpr = 0.0

        return {
            "auroc": auroc,
            "aucpr": aucpr,
            "loss": total_loss / max(n_batches, 1),
        }

    @torch.no_grad()
    def _validate(self) -> dict:
        """Evaluate on source validation set."""
        return self._eval_loop(self.source_val_loader)

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluate on test set. Returns dict with auroc, aucpr, loss."""
        return self._eval_loop(test_loader)

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict):
        path = self.run_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "best_val_auroc": self.best_val_auroc,
            "best_epoch": self.best_epoch,
        }, path)

    def _load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_val_auroc = ckpt.get("best_val_auroc", -float("inf"))
        self.best_epoch = ckpt.get("best_epoch", -1)
