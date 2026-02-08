import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
import time

from ..adapters.yaib import YAIBRuntime
from ..core.translator import Translator
from ..core.schema import SchemaResolver


def verify_baseline_determinism(
    yaib_runtime: YAIBRuntime,
    sample_batch: tuple[torch.Tensor, ...],
    device: str,
) -> None:
    """
    Verifies that the baseline model is deterministic in its current mode.
    """
    model = getattr(yaib_runtime, "_model", None)
    if model is None:
        return

    print("\n[Safety Check] Verifying baseline determinism (cuDNN Enabled)...")

    x = sample_batch[0].to(device)

    with torch.no_grad():
        out1 = model(x)
        if isinstance(out1, tuple):
            out1 = out1[0]
        out2 = model(x)
        if isinstance(out2, tuple):
            out2 = out2[0]

    diff = (out1 - out2).abs().max().item()
    if diff > 1e-6:
        raise RuntimeError(
            f"CRITICAL: Baseline is stochastic (Diff: {diff:.2e}). "
            "Internal LSTM dropout or other noise source is active."
        )
    print(f"[Safety Check] PASSED. Baseline is deterministic (Max Diff: {diff:.2e}).\n")


class TranslatorTrainer:
    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: Translator,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.yaib_runtime = yaib_runtime
        self.translator = translator.to(device)
        self.device = device
        self.optimizer = Adam(self.translator.parameters(), lr=learning_rate)
        
        self.best_val_loss = float('inf')
        self.best_translator_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.translator.train()
        total_loss = 0.0
        num_batches = 0
        total_elements = 0
        for batch in train_loader:
            batch = tuple(b.to(self.device) for b in batch)
            
            self.optimizer.zero_grad()
            if num_batches % 200 == 0:
                logging.info(f"Training Batch number: {num_batches}/{len(train_loader)}")
            translated_data = self.translator(batch)
            baseline_outputs = self.yaib_runtime.forward((translated_data, batch[1], batch[2]))
            loss = self.yaib_runtime.compute_loss(baseline_outputs, (translated_data, batch[1], batch[2]))
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float("inf")
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.translator.eval()
        all_probs = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if num_batches % 200 == 0:
                    logging.info(f"Validating Batch number: {num_batches}/{len(val_loader)}")
                batch = tuple(b.to(self.device) for b in batch)
                translated_data = self.translator(batch)
                baseline_outputs = self.yaib_runtime.forward((translated_data, batch[1], batch[2]))
                mask = batch[2].to(baseline_outputs.device).bool()
                prediction = torch.masked_select(
                    baseline_outputs, mask.unsqueeze(-1)
                ).reshape(-1, baseline_outputs.shape[-1])
                target = torch.masked_select(batch[1].to(baseline_outputs.device), mask)

                if baseline_outputs.shape[-1] > 1:
                    prediction_proba = torch.softmax(prediction, dim=-1)[:, 1]
                else:
                    prediction_proba = torch.sigmoid(prediction).squeeze(-1)

                all_probs.append(prediction_proba.detach().cpu())
                all_targets.append(target.detach().cpu())
                loss = self.yaib_runtime.compute_loss(baseline_outputs, (translated_data, batch[1], batch[2]))
                total_loss += loss.item()
                num_batches += 1
        
        if not all_probs:
            return {"AUCROC": 0.0, "AUCPR": 0.0, "loss": float("inf")}

        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

        try:
            auroc = roc_auc_score(targets, probs)
        except ValueError:
            auroc = 0.0
        try:
            auprc = average_precision_score(targets, probs)
        except ValueError:
            auprc = 0.0
        loss = total_loss / num_batches if num_batches > 0 else float("inf")

        return {"AUCROC": auroc, "AUCPR": auprc, "loss": loss}
    
    def train(
        self,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Optional[Path] = None,
        patience: int = 10,
    ):
        logging.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs} - Training...")
            train_loss = self.train_epoch(train_loader)
            logging.info(f"Epoch {epoch+1}/{epochs} - Validating...")
            val_metrics = self.validate(val_loader)
            
            logging.info(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUCROC: {val_metrics['AUCROC']:.4f}, "
                f"Val AUCPR: {val_metrics['AUCPR']:.4f}"
            )
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_translator_state = self.translator.state_dict().copy()
                logging.info(f"New best validation loss: {self.best_val_loss:.4f}")
                
                if checkpoint_dir:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / "best_translator.pt"
                    torch.save({
                        'epoch': epoch,
                        'translator_state_dict': self.best_translator_state,
                        'val_loss': self.best_val_loss,
                        'val_metrics': val_metrics,
                    }, checkpoint_path)
                    logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        if self.best_translator_state:
            self.translator.load_state_dict(self.best_translator_state)
            logging.info("Loaded best translator weights")


class TransformerTranslatorTrainer:
    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: nn.Module,
        schema_resolver: SchemaResolver,
        bounds_csv: Path,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_fidelity: float = 0.01,
        lambda_range: float = 1e-3,
        best_metric: str = "val_total",
        run_dir: Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.yaib_runtime = yaib_runtime
        self.schema_resolver = schema_resolver
        self.translator = translator.to(device)
        self.device = device
        self.optimizer = AdamW(self.translator.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.lambda_fidelity = lambda_fidelity
        self.lambda_range = lambda_range
        self.best_metric = best_metric
        self.run_dir = Path(run_dir) if run_dir is not None else Path("runs/translator")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = GradScaler(enabled=self.device.startswith("cuda"))
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        self.yaib_runtime.load_baseline_model()
        if hasattr(self.yaib_runtime, "_model") and self.yaib_runtime._model is not None:
            self.yaib_runtime._model = self.yaib_runtime._model.to(device)
            for param in self.yaib_runtime._model.parameters():
                param.requires_grad = False
            self._apply_baseline_speed_safe_mode()

        self.lower_bounds, self.upper_bounds = self._load_feature_bounds(bounds_csv, schema_resolver.dynamic_features)
        self.lower_bounds = self.lower_bounds.to(device)
        self.upper_bounds = self.upper_bounds.to(device)

        self.best_val = float("inf")
        self.best_state = None
        self.history: list[dict[str, float]] = []
        self._logged_train_batch = False
        self._logged_val_batch = False

    def _load_feature_bounds(self, bounds_csv: Path, feature_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        import pandas as pd

        df = pd.read_csv(bounds_csv)
        if "feature" not in df.columns:
            raise ValueError("Bounds CSV must include a 'feature' column.")
        columns = set(df.columns)

        lower_candidates = ["p0.1", "p_001", "q001", "q01", "q05"]
        upper_candidates = ["p99.9", "p_999", "q999", "q99", "q95"]

        def pick(base: str) -> str | None:
            if f"{base}_a" in columns:
                return f"{base}_a"
            if f"{base}_b" in columns:
                logging.warning(f"Using {base}_b as fallback for {base}")
                return f"{base}_b"
            if base in columns:
                logging.warning(f"Using {base} as fallback for {base}")
                return base
            return None

        lower_col = None
        for cand in lower_candidates:
            lower_col = pick(cand)
            if lower_col:
                break
        upper_col = None
        for cand in upper_candidates:
            upper_col = pick(cand)
            if upper_col:
                break
        if lower_col is None or upper_col is None:
            raise ValueError(
                f"Bounds CSV missing required percentile columns. Found columns: {sorted(columns)}"
            )

        if "p0.1" not in lower_col and "p99.9" not in upper_col:
            logging.info(
                "Using fallback bounds columns (%s, %s) from %s", lower_col, upper_col, bounds_csv
            )

        df = df.set_index("feature")
        missing = [name for name in feature_names if name not in df.index]
        if missing:
            raise ValueError(f"Bounds CSV missing features: {missing}")
        lower = torch.tensor(df.loc[feature_names, lower_col].to_numpy(), dtype=torch.float32)
        upper = torch.tensor(df.loc[feature_names, upper_col].to_numpy(), dtype=torch.float32)
        return lower, upper

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return values[mask].mean()
        return values.new_tensor(0.0)

    def _log_debug_batch(self, parts: dict[str, torch.Tensor], rebuilt: torch.Tensor, split: str) -> None:
        logging.info(
            "[debug] %s batch shapes X_val=%s X_miss=%s X_static=%s t_abs=%s M_pad=%s rebuilt=%s",
            split,
            tuple(parts["X_val"].shape),
            tuple(parts["X_miss"].shape),
            tuple(parts["X_static"].shape),
            tuple(parts["t_abs"].shape),
            tuple(parts["M_pad"].shape),
            tuple(rebuilt.shape),
        )
        x_yaib = parts["X_yaib"]
        x_miss = parts["X_miss"]
        m_pad = parts["M_pad"]
        miss_from_rebuild = rebuilt[:, :, self.schema_resolver.indices.missing]
        assert torch.allclose(miss_from_rebuild, x_miss), "X_miss changed during rebuild"
        static_idx = self.schema_resolver.indices.static
        if static_idx:
            static_from_yaib = x_yaib[:, :, static_idx]
            static_expected = parts["X_static"].unsqueeze(1).expand_as(static_from_yaib)
            valid = (~m_pad).unsqueeze(-1).expand_as(static_from_yaib)
            assert torch.allclose(static_from_yaib[valid], static_expected[valid]), "X_static changed"
        if hasattr(self.translator, "_last_temporal_key_padding_mask"):
            key_mask = self.translator._last_temporal_key_padding_mask
            expected = m_pad.unsqueeze(1).expand(m_pad.shape[0], len(self.schema_resolver.indices.dynamic), m_pad.shape[1])
            expected = expected.reshape(m_pad.shape[0] * len(self.schema_resolver.indices.dynamic), m_pad.shape[1])
            assert key_mask is not None and key_mask.shape == expected.shape, "Temporal attention mask not applied"
            assert torch.equal(key_mask, expected), "Temporal attention mask mismatch"

    def _apply_baseline_speed_safe_mode(self) -> None:
        model = getattr(self.yaib_runtime, "_model", None)
        if model is None:
            return
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LSTM):
                if module.dropout > 0.0 and module.num_layers > 1:
                    logging.warning(
                        "FOUND INTERNAL LSTM DROPOUT (%.3f) in '%s'. Forcing to 0.0.",
                        module.dropout,
                        name,
                    )
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

    def _run_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.translator.train()
        totals = {"total": 0.0, "task": 0.0, "fidelity": 0.0, "range": 0.0}
        num_batches = 0
        logged_this_epoch = False
        for batch in loader:
            batch = tuple(b.to(self.device) for b in batch)
            parts = self.schema_resolver.extract(batch)
            use_amp = self.scaler.is_enabled()
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t_translator = time.time()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                x_val_out = self.translator(
                    parts["X_val"],
                    parts["X_miss"],
                    parts["t_abs"],
                    parts["M_pad"],
                    parts["X_static"],
                )
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out, parts["X_miss"], parts["X_static"]
                )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            translator_time = time.time() - t_translator
            if not self._logged_train_batch:
                self._log_debug_batch(parts, x_yaib_translated, "train")
                assert torch.allclose(x_val_out[parts["M_pad"]], x_val_out.new_zeros(())), "Padded rows not zero"
                self._logged_train_batch = True

            label_mask = parts["M_label"].bool()
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t_baseline = time.time()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = self.yaib_runtime.forward((x_yaib_translated, parts["y"], label_mask))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            baseline_time = time.time() - t_baseline
            logits_fp32 = logits.float()
            l_task = self.yaib_runtime.compute_loss(
                logits_fp32, (x_yaib_translated, parts["y"], label_mask)
            ).float()
            mask = (~parts["M_pad"]).bool()
            diff = (x_val_out.float() - parts["X_val"].float()) ** 2
            l_fidelity = self._masked_mean(diff.sum(dim=-1), mask).float()
            upper = self.upper_bounds.view(1, 1, -1)
            lower = self.lower_bounds.view(1, 1, -1)
            over = torch.relu(x_val_out.float() - upper)
            under = torch.relu(lower - x_val_out.float())
            l_range = self._masked_mean((over ** 2 + under ** 2).sum(dim=-1), mask).float()
            l_total = l_task + (self.lambda_fidelity * l_fidelity) + (self.lambda_range * l_range)

            self.optimizer.zero_grad()
            if use_amp:
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_backward = time.time()
                self.scaler.scale(l_total).backward()
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                backward_time = time.time() - t_backward
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_backward = time.time()
                l_total.backward()
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                backward_time = time.time() - t_backward
                self.optimizer.step()

            if not logged_this_epoch:
                logging.info(
                    "[perf] translator forward: %.3fs baseline forward: %.3fs backward: %.3fs",
                    translator_time,
                    baseline_time,
                    backward_time,
                )
                logging.info(
                    "[debug] X_val device=%s translator_device=%s",
                    parts["X_val"].device,
                    next(self.translator.parameters()).device,
                )
                logging.info(
                    "[debug] x_yaib_translated device=%s dtype=%s logits device=%s dtype=%s",
                    x_yaib_translated.device,
                    x_yaib_translated.dtype,
                    logits.device,
                    logits.dtype,
                )
                logged_this_epoch = True

            totals["total"] += l_total.item()
            totals["task"] += l_task.item()
            totals["fidelity"] += l_fidelity.item()
            totals["range"] += l_range.item()
            num_batches += 1

        if num_batches == 0:
            return {k: float("inf") for k in totals}
        return {k: v / num_batches for k, v in totals.items()}

    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.translator.eval()
        totals = {"total": 0.0, "task": 0.0, "fidelity": 0.0, "range": 0.0}
        num_batches = 0
        with torch.no_grad():
            for batch in loader:
                batch = tuple(b.to(self.device) for b in batch)
                parts = self.schema_resolver.extract(batch)
                x_val_out = self.translator(
                    parts["X_val"],
                    parts["X_miss"],
                    parts["t_abs"],
                    parts["M_pad"],
                    parts["X_static"],
                )
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out, parts["X_miss"], parts["X_static"]
                )
                if not self._logged_val_batch:
                    self._log_debug_batch(parts, x_yaib_translated, "val")
                    assert torch.allclose(x_val_out[parts["M_pad"]], x_val_out.new_zeros(())), "Padded rows not zero"
                    self._logged_val_batch = True

                label_mask = parts["M_label"].bool()
                logits = self.yaib_runtime.forward((x_yaib_translated, parts["y"], label_mask))
                l_task = self.yaib_runtime.compute_loss(logits, (x_yaib_translated, parts["y"], label_mask))
                mask = (~parts["M_pad"]).bool()
                diff = (x_val_out - parts["X_val"]) ** 2
                l_fidelity = self._masked_mean(diff.sum(dim=-1), mask)
                over = torch.relu(x_val_out - self.upper_bounds.view(1, 1, -1))
                under = torch.relu(self.lower_bounds.view(1, 1, -1) - x_val_out)
                l_range = self._masked_mean((over ** 2 + under ** 2).sum(dim=-1), mask)
                l_total = l_task + (self.lambda_fidelity * l_fidelity) + (self.lambda_range * l_range)

                totals["total"] += l_total.item()
                totals["task"] += l_task.item()
                totals["fidelity"] += l_fidelity.item()
                totals["range"] += l_range.item()
                num_batches += 1

        if num_batches == 0:
            return {k: float("inf") for k in totals}
        return {k: v / num_batches for k, v in totals.items()}

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader) -> None:
        if hasattr(self.yaib_runtime, "_model") and self.yaib_runtime._model is not None:
            try:
                sample_batch = next(iter(train_loader))
                sample_batch = tuple(b.to(self.device) for b in sample_batch)
                verify_baseline_determinism(self.yaib_runtime, sample_batch, self.device)
            except StopIteration:
                logging.warning("Baseline integrity check skipped: empty train_loader.")
        for epoch in range(epochs):
            logging.info("Epoch %d/%d - training", epoch + 1, epochs)
            train_metrics = self._run_epoch(train_loader)
            logging.info("Epoch %d/%d - validating", epoch + 1, epochs)
            val_metrics = self._validate(val_loader)

            logging.info(
                "Epoch %d/%d - train_total=%.4f train_task=%.4f train_fidelity=%.4f train_range=%.4f",
                epoch + 1,
                epochs,
                train_metrics["total"],
                train_metrics["task"],
                train_metrics["fidelity"],
                train_metrics["range"],
            )
            logging.info(
                "Epoch %d/%d - val_total=%.4f val_task=%.4f val_fidelity=%.4f val_range=%.4f",
                epoch + 1,
                epochs,
                val_metrics["total"],
                val_metrics["task"],
                val_metrics["fidelity"],
                val_metrics["range"],
            )

            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_total": train_metrics["total"],
                    "train_task": train_metrics["task"],
                    "train_fidelity": train_metrics["fidelity"],
                    "train_range": train_metrics["range"],
                    "val_total": val_metrics["total"],
                    "val_task": val_metrics["task"],
                    "val_fidelity": val_metrics["fidelity"],
                    "val_range": val_metrics["range"],
                }
            )

            candidate = val_metrics["total"] if self.best_metric == "val_total" else val_metrics["task"]
            if candidate < self.best_val:
                self.best_val = candidate
                self.best_state = self.translator.state_dict()
                checkpoint = {
                    "epoch": epoch,
                    "translator_state_dict": self.best_state,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                }
                torch.save(checkpoint, self.run_dir / "best_translator.pt")
                logging.info("Saved new best checkpoint to %s", self.run_dir / "best_translator.pt")

        if self.best_state is not None:
            self.translator.load_state_dict(self.best_state)

        self._plot_losses()

    def _plot_losses(self) -> None:
        if not self.history:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in self.history]
        train_total = [row["train_total"] for row in self.history]
        val_total = [row["val_total"] for row in self.history]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, train_total, label="train_total")
        ax.plot(epochs, val_total, label="val_total")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.run_dir / "loss_curve.png", dpi=150)
        plt.close(fig)
