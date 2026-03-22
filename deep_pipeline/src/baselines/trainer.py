"""Trainer for DA baselines (DANN, CORAL, CoDATS) in the frozen-LSTM setting.

Follows the TransformerTranslatorTrainer pattern from src/core/train.py,
adding domain-adversarial / CORAL losses on LSTM hidden states.
"""
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..adapters.yaib import YAIBRuntime
from ..core.schema import SchemaResolver
from ..core.hidden_extractor import HiddenStateExtractor
from ..core.train import _create_lr_scheduler, verify_baseline_determinism
from .components import (
    DomainDiscriminator,
    GradientReversalLayer,
    coral_loss,
    grl_lambda_schedule,
)


class DABaselineTrainer:
    """Unified trainer for DANN, CORAL, and CoDATS baselines.

    Differences from TransformerTranslatorTrainer:
    - Extracts LSTM hidden states via HiddenStateExtractor
    - Adds adversarial loss (DANN/CoDATS) or CORAL loss
    - Separate discriminator optimizer (DANN/CoDATS)
    - GRL with progressive lambda schedule
    """

    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: nn.Module,
        schema_resolver: SchemaResolver,
        bounds_csv: Path,
        da_method: str,
        target_train_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_fidelity: float = 0.1,
        lambda_range: float = 0.5,
        lambda_adversarial: float = 0.2,
        lambda_coral: float = 1.0,
        discriminator_hidden_dim: int = 256,
        discriminator_lr: float = 1e-4,
        grl_schedule: bool = True,
        early_stopping_patience: int = 0,
        best_metric: str = "val_task",
        run_dir: Path | None = None,
        device: str = "cuda",
        training_config: dict | None = None,
    ) -> None:
        self.yaib_runtime = yaib_runtime
        self.schema_resolver = schema_resolver
        self.translator = translator.to(device)
        self.device = device
        self.da_method = da_method
        self.lambda_fidelity = lambda_fidelity
        self.lambda_range = lambda_range
        self.lambda_adversarial = lambda_adversarial
        self.lambda_coral = lambda_coral
        self.grl_schedule = grl_schedule
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = best_metric
        self.run_dir = Path(run_dir) if run_dir else Path("runs/da_baseline")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = GradScaler(enabled=device.startswith("cuda"))
        self._training_config = training_config or {}

        # Translator optimizer
        self.optimizer = AdamW(self.translator.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # V6 features
        self.grad_clip_norm = self._training_config.get("grad_clip_norm", 0.0)
        self.accumulate_grad_batches = max(1, self._training_config.get("accumulate_grad_batches", 1))
        self.lr_scheduler_type = self._training_config.get("lr_scheduler", None)
        self.lr_min = self._training_config.get("lr_min", 0.0)
        self.lr_warmup_epochs = self._training_config.get("lr_warmup_epochs", 0)

        # Target data
        self.target_train_loader = target_train_loader
        self._target_iter = iter(target_train_loader)

        # Frozen baseline setup
        self.yaib_runtime.load_baseline_model()
        if hasattr(self.yaib_runtime, "_model") and self.yaib_runtime._model is not None:
            self.yaib_runtime._model = self.yaib_runtime._model.to(device)
            for param in self.yaib_runtime._model.parameters():
                param.requires_grad = False
            self._apply_baseline_speed_safe_mode()

        # Hidden state extractor (hooks into frozen LSTM)
        self.hidden_extractor = HiddenStateExtractor(self.yaib_runtime._model)

        # Feature bounds for range loss
        self.lower_bounds, self.upper_bounds = self._load_feature_bounds(
            bounds_csv, schema_resolver.dynamic_features
        )
        self.lower_bounds = self.lower_bounds.to(device)
        self.upper_bounds = self.upper_bounds.to(device)

        # Domain discriminator (DANN / CoDATS)
        self.discriminator = None
        self.disc_optimizer = None
        self.grl = None
        if da_method in ("dann", "codats"):
            # Detect hidden dim from the LSTM
            hidden_dim = self._detect_hidden_dim()
            self.discriminator = DomainDiscriminator(
                input_dim=hidden_dim,
                hidden_dim=discriminator_hidden_dim,
            ).to(device)
            self.disc_optimizer = Adam(self.discriminator.parameters(), lr=discriminator_lr)
            self.grl = GradientReversalLayer(lambda_=1.0)
            logging.info(
                "[DA] %s: discriminator on hidden_dim=%d, disc_lr=%.2e, grl_schedule=%s",
                da_method, hidden_dim, discriminator_lr, grl_schedule,
            )
        elif da_method == "coral":
            hidden_dim = self._detect_hidden_dim()
            logging.info("[DA] CORAL: covariance matching on hidden_dim=%d", hidden_dim)

        # Feature gate (optional)
        self.feature_gate = None
        if self._training_config.get("feature_gate", False):
            from ..core.feature_gate import FeatureGate
            num_features = len(schema_resolver.dynamic_features)
            self.feature_gate = FeatureGate(num_features).to(device)
            self.optimizer.add_param_group({"params": self.feature_gate.parameters()})
            logging.info("Feature gate enabled (%d features)", num_features)

        # Cross-domain normalization
        self.renorm_scale = None
        self.renorm_offset = None

        # Tracking
        self.best_val = float("inf")
        self.best_state = None
        self.history: list[dict[str, float]] = []
        self._logged_train_batch = False

        logging.info("[DA] Trainer initialized: method=%s, lambda_fid=%.3f, lambda_range=%.3f",
                     da_method, lambda_fidelity, lambda_range)

    def _detect_hidden_dim(self) -> int:
        """Detect LSTM hidden dimension from model architecture."""
        model = self.yaib_runtime._model
        if hasattr(model, "rnn") and hasattr(model.rnn, "hidden_size"):
            return model.rnn.hidden_size
        if hasattr(model, "lstm") and hasattr(model.lstm, "hidden_size"):
            return model.lstm.hidden_size
        # Fallback: try to get from logit layer input size
        if hasattr(model, "logit") and hasattr(model.logit, "in_features"):
            return model.logit.in_features
        logging.warning("Could not detect hidden_dim, defaulting to 128")
        return 128

    def _load_feature_bounds(self, bounds_csv: Path, feature_names: list[str]):
        """Load feature bounds from CSV (same as TransformerTranslatorTrainer)."""
        import pandas as pd
        df = pd.read_csv(bounds_csv)
        df = df.set_index("feature")
        columns = set(df.columns)

        lower_candidates = ["p0.1", "p_001", "q001", "q01", "q05"]
        upper_candidates = ["p99.9", "p_999", "q999", "q99", "q95"]

        def pick(base):
            for suffix in ("_a", "_b", ""):
                key = f"{base}{suffix}" if suffix else base
                if key in columns:
                    return key
            return None

        lower_col = next((pick(c) for c in lower_candidates if pick(c)), None)
        upper_col = next((pick(c) for c in upper_candidates if pick(c)), None)
        if lower_col is None or upper_col is None:
            raise ValueError(f"Bounds CSV missing percentile columns. Found: {sorted(columns)}")

        lower = torch.tensor(df.loc[feature_names, lower_col].to_numpy(), dtype=torch.float32)
        upper = torch.tensor(df.loc[feature_names, upper_col].to_numpy(), dtype=torch.float32)
        return lower, upper

    def _apply_baseline_speed_safe_mode(self):
        """Same as TransformerTranslatorTrainer: train mode + disable dropout/BN."""
        model = self.yaib_runtime._model
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LSTM):
                if module.dropout > 0.0 and module.num_layers > 1:
                    module.dropout = 0.0
        model.train()
        from torch.nn.modules.dropout import _DropoutNd
        from torch.nn.modules.batchnorm import _BatchNorm

        def force_stateless(m):
            if isinstance(m, (_DropoutNd, torch.nn.Dropout)):
                m.eval()
            elif isinstance(m, _BatchNorm):
                m.eval()
                m.track_running_stats = False
        model.apply(force_stateless)

    def set_renorm_params(self, scale: torch.Tensor, offset: torch.Tensor):
        self.renorm_scale = scale.to(self.device)
        self.renorm_offset = offset.to(self.device)
        logging.info("Cross-domain renormalization enabled")

    def _apply_renorm(self, x_val: torch.Tensor, m_pad: torch.Tensor) -> torch.Tensor:
        if self.renorm_scale is None:
            return x_val
        x = x_val * self.renorm_scale.view(1, 1, -1) + self.renorm_offset.view(1, 1, -1)
        return x.masked_fill(m_pad.unsqueeze(-1).bool(), 0.0)

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return values[mask].mean()
        return values.new_tensor(0.0)

    def _next_target_batch(self):
        try:
            batch = next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self.target_train_loader)
            batch = next(self._target_iter)
        return tuple(b.to(self.device) for b in batch)

    def _get_hidden_states(self, yaib_batch, label_mask):
        """Forward through frozen LSTM and capture hidden states."""
        logits = self.yaib_runtime.forward(yaib_batch)
        h = self.hidden_extractor.hidden_states  # (B, T, H)
        return logits, h

    def _run_epoch(self, loader: DataLoader, epoch: int, total_epochs: int) -> dict[str, float]:
        self.translator.train()
        if self.discriminator is not None:
            self.discriminator.train()

        # Update GRL lambda
        if self.grl is not None and self.grl_schedule:
            grl_lam = grl_lambda_schedule(epoch, total_epochs)
            self.grl.set_lambda(grl_lam)
            if epoch == 0 or (epoch + 1) % 5 == 0:
                logging.info("[GRL] epoch %d: lambda=%.4f", epoch + 1, grl_lam)

        totals = {"total": 0.0, "task": 0.0, "fidelity": 0.0, "range": 0.0, "da": 0.0, "disc_acc": 0.0}
        num_batches = 0

        for batch in loader:
            batch = tuple(b.to(self.device) for b in batch)
            parts = self.schema_resolver.extract(batch)
            parts["X_val"] = self._apply_renorm(parts["X_val"], parts["M_pad"])

            use_amp = self.scaler.is_enabled()

            # --- Source forward ---
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                x_val_out = self.translator(
                    parts["X_val"], parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], parts["X_static"],
                )
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out, parts["X_miss"], parts["X_static"],
                    m_pad=parts["M_pad"],
                )

            label_mask = parts["M_label"].bool()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits_src, h_source = self._get_hidden_states(
                    (x_yaib_translated, parts["y"], label_mask), label_mask
                )

            # Task loss
            l_task = self.yaib_runtime.compute_loss(
                logits_src.float(), (x_yaib_translated, parts["y"], label_mask)
            ).float()

            # Fidelity loss
            mask = (~parts["M_pad"]).bool()
            diff = (x_val_out.float() - parts["X_val"].float()) ** 2
            if self.feature_gate is not None:
                gate = self.feature_gate()
                fid_weight = (1.0 - 0.5 * gate).view(1, 1, -1)
                diff = diff * fid_weight
            l_fidelity = self._masked_mean(diff.sum(dim=-1), mask).float()

            # Range loss
            upper = self.upper_bounds.view(1, 1, -1)
            lower = self.lower_bounds.view(1, 1, -1)
            over = torch.relu(x_val_out.float() - upper)
            under = torch.relu(lower - x_val_out.float())
            l_range = self._masked_mean((over ** 2 + under ** 2).sum(dim=-1), mask).float()

            # --- Target forward (raw, no translator) ---
            target_batch = self._next_target_batch()
            target_parts = self.schema_resolver.extract(target_batch)
            target_label_mask = target_parts["M_label"].bool()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                with torch.no_grad():
                    _, h_target = self._get_hidden_states(
                        (target_parts["X_yaib"], target_parts["y"], target_label_mask),
                        target_label_mask,
                    )

            # Flatten hidden states to non-padded timesteps
            src_mask = (~parts["M_pad"]).bool()
            tgt_mask = (~target_parts["M_pad"]).bool()
            h_s = h_source[src_mask].float()   # (N_s, H)
            h_t = h_target[tgt_mask].float()   # (N_t, H)

            # --- DA-specific loss ---
            l_da = x_val_out.new_tensor(0.0)
            disc_acc = 0.0

            if self.da_method in ("dann", "codats") and self.discriminator is not None:
                # Adversarial loss for translator (with GRL)
                h_s_grl = self.grl(h_s)
                pred_src = self.discriminator(h_s_grl)
                pred_tgt = self.discriminator(h_t.detach())
                l_da = (
                    F.binary_cross_entropy_with_logits(pred_src, torch.zeros_like(pred_src))
                    + F.binary_cross_entropy_with_logits(pred_tgt, torch.ones_like(pred_tgt))
                ) / 2.0

                # Discriminator accuracy for monitoring
                with torch.no_grad():
                    acc_s = (pred_src.detach() < 0).float().mean()
                    acc_t = (pred_tgt.detach() > 0).float().mean()
                    disc_acc = ((acc_s + acc_t) / 2.0).item()

            elif self.da_method == "coral":
                l_da = coral_loss(h_s, h_t.detach())

            # --- Total loss ---
            l_total = (
                l_task
                + self.lambda_fidelity * l_fidelity
                + self.lambda_range * l_range
            )
            if self.da_method in ("dann", "codats"):
                l_total = l_total + self.lambda_adversarial * l_da
            elif self.da_method == "coral":
                l_total = l_total + self.lambda_coral * l_da

            # --- Backward + optimizer step (translator) ---
            if num_batches % self.accumulate_grad_batches == 0:
                self.optimizer.zero_grad()

            if use_amp:
                self.scaler.scale(l_total / self.accumulate_grad_batches).backward()
                if (num_batches + 1) % self.accumulate_grad_batches == 0:
                    if self.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in self.optimizer.param_groups for p in g["params"]],
                            self.grad_clip_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                (l_total / self.accumulate_grad_batches).backward()
                if (num_batches + 1) % self.accumulate_grad_batches == 0:
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in self.optimizer.param_groups for p in g["params"]],
                            self.grad_clip_norm,
                        )
                    self.optimizer.step()

            # --- Discriminator update step (separate, no GRL) ---
            if self.da_method in ("dann", "codats") and self.discriminator is not None:
                self.disc_optimizer.zero_grad()
                # Detach hidden states for discriminator-only update
                pred_src_d = self.discriminator(h_s.detach())
                pred_tgt_d = self.discriminator(h_t.detach())
                l_disc = (
                    F.binary_cross_entropy_with_logits(pred_src_d, torch.zeros_like(pred_src_d))
                    + F.binary_cross_entropy_with_logits(pred_tgt_d, torch.ones_like(pred_tgt_d))
                ) / 2.0
                l_disc.backward()
                self.disc_optimizer.step()

            totals["total"] += l_total.item()
            totals["task"] += l_task.item()
            totals["fidelity"] += l_fidelity.item()
            totals["range"] += l_range.item()
            totals["da"] += l_da.item()
            totals["disc_acc"] += disc_acc
            num_batches += 1

        # Flush remaining accumulated gradients
        if self.accumulate_grad_batches > 1 and num_batches % self.accumulate_grad_batches != 0:
            if use_amp:
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.optimizer.param_groups for p in g["params"]],
                        self.grad_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.optimizer.param_groups for p in g["params"]],
                        self.grad_clip_norm,
                    )
                self.optimizer.step()

        if num_batches == 0:
            return {k: float("inf") for k in totals}
        return {k: v / num_batches for k, v in totals.items()}

    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.translator.eval()
        if self.discriminator is not None:
            self.discriminator.eval()

        totals = {"total": 0.0, "task": 0.0, "fidelity": 0.0, "range": 0.0, "da": 0.0, "disc_acc": 0.0}
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                batch = tuple(b.to(self.device) for b in batch)
                parts = self.schema_resolver.extract(batch)
                parts["X_val"] = self._apply_renorm(parts["X_val"], parts["M_pad"])

                x_val_out = self.translator(
                    parts["X_val"], parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], parts["X_static"],
                )
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out, parts["X_miss"], parts["X_static"],
                    m_pad=parts["M_pad"],
                )

                label_mask = parts["M_label"].bool()
                logits_src, h_source = self._get_hidden_states(
                    (x_yaib_translated, parts["y"], label_mask), label_mask
                )

                l_task = self.yaib_runtime.compute_loss(
                    logits_src.float(), (x_yaib_translated, parts["y"], label_mask)
                ).float()

                mask = (~parts["M_pad"]).bool()
                diff = (x_val_out.float() - parts["X_val"].float()) ** 2
                if self.feature_gate is not None:
                    gate = self.feature_gate()
                    fid_weight = (1.0 - 0.5 * gate).view(1, 1, -1)
                    diff = diff * fid_weight
                l_fidelity = self._masked_mean(diff.sum(dim=-1), mask).float()

                upper = self.upper_bounds.view(1, 1, -1)
                lower = self.lower_bounds.view(1, 1, -1)
                over = torch.relu(x_val_out.float() - upper)
                under = torch.relu(lower - x_val_out.float())
                l_range = self._masked_mean((over ** 2 + under ** 2).sum(dim=-1), mask).float()

                # DA loss on validation
                target_batch = self._next_target_batch()
                target_parts = self.schema_resolver.extract(target_batch)
                target_label_mask = target_parts["M_label"].bool()
                _, h_target = self._get_hidden_states(
                    (target_parts["X_yaib"], target_parts["y"], target_label_mask),
                    target_label_mask,
                )

                src_mask = (~parts["M_pad"]).bool()
                tgt_mask = (~target_parts["M_pad"]).bool()
                h_s = h_source[src_mask].float()
                h_t = h_target[tgt_mask].float()

                l_da = x_val_out.new_tensor(0.0)
                disc_acc = 0.0
                if self.da_method in ("dann", "codats") and self.discriminator is not None:
                    pred_src = self.discriminator(h_s)
                    pred_tgt = self.discriminator(h_t)
                    l_da = (
                        F.binary_cross_entropy_with_logits(pred_src, torch.zeros_like(pred_src))
                        + F.binary_cross_entropy_with_logits(pred_tgt, torch.ones_like(pred_tgt))
                    ) / 2.0
                    acc_s = (pred_src < 0).float().mean()
                    acc_t = (pred_tgt > 0).float().mean()
                    disc_acc = ((acc_s + acc_t) / 2.0).item()
                elif self.da_method == "coral":
                    l_da = coral_loss(h_s, h_t)

                l_total = l_task + self.lambda_fidelity * l_fidelity + self.lambda_range * l_range
                if self.da_method in ("dann", "codats"):
                    l_total = l_total + self.lambda_adversarial * l_da
                elif self.da_method == "coral":
                    l_total = l_total + self.lambda_coral * l_da

                totals["total"] += l_total.item()
                totals["task"] += l_task.item()
                totals["fidelity"] += l_fidelity.item()
                totals["range"] += l_range.item()
                totals["da"] += l_da.item()
                totals["disc_acc"] += disc_acc
                num_batches += 1

        if num_batches == 0:
            return {k: float("inf") for k in totals}
        return {k: v / num_batches for k, v in totals.items()}

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # Baseline determinism check
        if hasattr(self.yaib_runtime, "_model") and self.yaib_runtime._model is not None:
            try:
                sample_batch = next(iter(train_loader))
                sample_batch = tuple(b.to(self.device) for b in sample_batch)
                verify_baseline_determinism(self.yaib_runtime, sample_batch, self.device)
            except StopIteration:
                logging.warning("Baseline integrity check skipped: empty train_loader.")

        # Create LR scheduler
        self.scheduler = _create_lr_scheduler(
            self.optimizer, self.lr_scheduler_type, epochs,
            lr_min=self.lr_min, warmup_epochs=self.lr_warmup_epochs,
        )

        epochs_without_improvement = 0

        for epoch in range(epochs):
            logging.info(
                "Epoch %d/%d [%s] - training (lr=%.2e)",
                epoch + 1, epochs, self.da_method.upper(),
                self.optimizer.param_groups[0]["lr"],
            )
            train_metrics = self._run_epoch(train_loader, epoch=epoch, total_epochs=epochs)

            logging.info("Epoch %d/%d - validating", epoch + 1, epochs)
            val_metrics = self._validate(val_loader)

            logging.info(
                "Epoch %d/%d - train: total=%.4f task=%.4f fid=%.4f range=%.4f da=%.4f disc_acc=%.2f",
                epoch + 1, epochs,
                train_metrics["total"], train_metrics["task"],
                train_metrics["fidelity"], train_metrics["range"],
                train_metrics["da"], train_metrics["disc_acc"],
            )
            logging.info(
                "Epoch %d/%d - val: total=%.4f task=%.4f fid=%.4f range=%.4f da=%.4f disc_acc=%.2f",
                epoch + 1, epochs,
                val_metrics["total"], val_metrics["task"],
                val_metrics["fidelity"], val_metrics["range"],
                val_metrics["da"], val_metrics["disc_acc"],
            )

            self.history.append({
                "epoch": epoch + 1,
                "lr": self.optimizer.param_groups[0]["lr"],
                "train_total": train_metrics["total"],
                "train_task": train_metrics["task"],
                "train_fidelity": train_metrics["fidelity"],
                "train_range": train_metrics["range"],
                "train_da": train_metrics["da"],
                "train_disc_acc": train_metrics["disc_acc"],
                "val_total": val_metrics["total"],
                "val_task": val_metrics["task"],
                "val_fidelity": val_metrics["fidelity"],
                "val_range": val_metrics["range"],
                "val_da": val_metrics["da"],
                "val_disc_acc": val_metrics["disc_acc"],
            })

            # LR scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["task"])
                else:
                    self.scheduler.step()

            # Checkpoint best model
            candidate = val_metrics["total"] if self.best_metric == "val_total" else val_metrics["task"]
            if candidate < self.best_val:
                self.best_val = candidate
                self.best_state = self.translator.state_dict()
                checkpoint = {
                    "epoch": epoch,
                    "translator_state_dict": self.best_state,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "renorm_scale": self.renorm_scale,
                    "renorm_offset": self.renorm_offset,
                    "da_method": self.da_method,
                }
                if self.discriminator is not None:
                    checkpoint["discriminator_state_dict"] = self.discriminator.state_dict()
                torch.save(checkpoint, self.run_dir / "best_translator.pt")
                logging.info("Saved new best checkpoint to %s", self.run_dir / "best_translator.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save latest checkpoint for resume
            latest = {
                "epoch": epoch,
                "translator_state_dict": self.translator.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val": self.best_val,
                "best_state": self.best_state,
                "renorm_scale": self.renorm_scale,
                "renorm_offset": self.renorm_offset,
                "da_method": self.da_method,
            }
            if self.discriminator is not None:
                latest["discriminator_state_dict"] = self.discriminator.state_dict()
                latest["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()
            if self.scheduler is not None:
                latest["scheduler_state_dict"] = self.scheduler.state_dict()
            torch.save(latest, self.run_dir / "latest_checkpoint.pt")

            if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
                logging.info("Early stopping after %d epochs without improvement", epochs_without_improvement)
                break

        # Load best model
        if self.best_state is not None:
            self.translator.load_state_dict(self.best_state)

        # Save loss history
        self._save_history()
        self._plot_losses()

    def _save_history(self):
        """Save training history to CSV."""
        import pandas as pd
        if self.history:
            df = pd.DataFrame(self.history)
            df.to_csv(self.run_dir / "loss_history.csv", index=False)
            logging.info("Saved loss history to %s", self.run_dir / "loss_history.csv")

    def _plot_losses(self):
        """Plot training curves."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not self.history:
                return

            epochs = [h["epoch"] for h in self.history]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Task loss
            axes[0, 0].plot(epochs, [h["train_task"] for h in self.history], label="train")
            axes[0, 0].plot(epochs, [h["val_task"] for h in self.history], label="val")
            axes[0, 0].set_title("Task Loss")
            axes[0, 0].legend()

            # DA loss
            axes[0, 1].plot(epochs, [h["train_da"] for h in self.history], label="train")
            axes[0, 1].plot(epochs, [h["val_da"] for h in self.history], label="val")
            axes[0, 1].set_title(f"{self.da_method.upper()} Loss")
            axes[0, 1].legend()

            # Discriminator accuracy (DANN/CoDATS)
            axes[1, 0].plot(epochs, [h["train_disc_acc"] for h in self.history], label="train")
            axes[1, 0].plot(epochs, [h["val_disc_acc"] for h in self.history], label="val")
            axes[1, 0].set_title("Disc Accuracy")
            axes[1, 0].axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
            axes[1, 0].legend()

            # Fidelity + Range
            axes[1, 1].plot(epochs, [h["train_fidelity"] for h in self.history], label="fidelity")
            axes[1, 1].plot(epochs, [h["train_range"] for h in self.history], label="range")
            axes[1, 1].set_title("Regularization Losses")
            axes[1, 1].legend()

            plt.tight_layout()
            plt.savefig(self.run_dir / "loss_curves.png", dpi=100)
            plt.close()
            logging.info("Saved loss curves to %s", self.run_dir / "loss_curves.png")
        except Exception as e:
            logging.warning("Could not plot losses: %s", e)
