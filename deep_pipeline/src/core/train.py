import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import Adam, AdamW

from torch.utils.data import DataLoader
import time

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR, SequentialLR

from ..adapters.yaib import YAIBRuntime
from ..core.translator import Translator
from ..core.schema import SchemaResolver
from ..core.mmd import multi_kernel_mmd


def _create_lr_scheduler(optimizer, scheduler_type, total_epochs, lr_min=0.0, warmup_epochs=0):
    """Create an LR scheduler (cosine or plateau) with optional linear warmup."""
    if scheduler_type is None:
        return None
    warmup_sched = None
    if warmup_epochs > 0:
        warmup_sched = LambdaLR(
            optimizer, lr_lambda=lambda ep: min(1.0, (ep + 1) / warmup_epochs)
        )
    if scheduler_type == "cosine":
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        main_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=lr_min)
        if warmup_sched is not None:
            return SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[warmup_epochs])
        return main_sched
    elif scheduler_type == "plateau":
        # ReduceLROnPlateau doesn't compose well with warmup, skip warmup for this type
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=lr_min)
    else:
        logging.warning("Unknown lr_scheduler type '%s', ignoring", scheduler_type)
        return None


def _compute_loader_stats(loader: DataLoader, schema_resolver: "SchemaResolver", device: str):
    """Compute per-dynamic-feature mean and std from a DataLoader."""
    sum_val = None
    sum_sq = None
    count = None
    with torch.no_grad():
        for batch in loader:
            batch = tuple(b.to(device, non_blocking=True) for b in batch)
            parts = schema_resolver.extract(batch)
            x = parts["X_val"]        # (B, T, F)
            mask = (~parts["M_pad"]).unsqueeze(-1).expand_as(x).float()
            if sum_val is None:
                sum_val = (x * mask).sum(dim=(0, 1))
                sum_sq = ((x ** 2) * mask).sum(dim=(0, 1))
                count = mask.sum(dim=(0, 1))
            else:
                sum_val += (x * mask).sum(dim=(0, 1))
                sum_sq += ((x ** 2) * mask).sum(dim=(0, 1))
                count += mask.sum(dim=(0, 1))
    mean = sum_val / count.clamp(min=1)
    std = ((sum_sq / count.clamp(min=1)) - mean ** 2).clamp(min=1e-8).sqrt()
    return mean, std


def compute_renorm_params(
    source_loader: DataLoader,
    target_loader: DataLoader,
    schema_resolver: "SchemaResolver",
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute affine transform to renormalize source data to target statistics.

    Returns (scale, offset) such that: x_renorm = x_source * scale + offset
    maps source distribution → target distribution via z-score matching:
        x_renorm = (x_source - src_mean) / src_std * tgt_std + tgt_mean

    This ensures E[x_renorm] = tgt_mean, Std[x_renorm] = tgt_std.
    """
    src_mean, src_std = _compute_loader_stats(source_loader, schema_resolver, device)
    tgt_mean, tgt_std = _compute_loader_stats(target_loader, schema_resolver, device)
    # Correct z-score mapping: x_renorm = x * (tgt_std/src_std) + (tgt_mean - src_mean*tgt_std/src_std)
    scale = tgt_std / src_std.clamp(min=1e-8)
    offset = tgt_mean - src_mean * scale
    logging.info(
        "Renorm params computed: scale range [%.4f, %.4f], offset range [%.4f, %.4f]",
        scale.min().item(), scale.max().item(), offset.min().item(), offset.max().item(),
    )
    return scale.to(device), offset.to(device)


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


def _gradient_diagnostic(
    optimizer: torch.optim.Optimizer,
    translator: nn.Module,
    x_out: torch.Tensor,
    l_task: torch.Tensor,
    l_reg: torch.Tensor,
    l_range: torch.Tensor,
    parts: dict,
    epoch: int,
    batch_idx: int,
    reg_label: str = "fid",
) -> None:
    """Log gradient magnitude, cosine alignment, and per-timestep gradient analysis.

    Called periodically during training to diagnose task vs regularization gradient
    dynamics.  Performs 3+1 extra backward passes (retain_graph=True) — caller must
    zero_grad before the real backward.
    """
    all_params = list(translator.parameters())

    def _grad_vec(loss_val):
        optimizer.zero_grad()
        if x_out.grad is not None:
            x_out.grad = None
        loss_val.backward(retain_graph=True)
        grads = [p.grad.detach().flatten() if p.grad is not None
                 else torch.zeros(p.numel(), device=x_out.device)
                 for p in all_params]
        return torch.cat(grads)

    task_vec = _grad_vec(l_task)
    reg_vec = _grad_vec(l_reg)
    range_vec = _grad_vec(l_range)

    task_norm = task_vec.norm().item()
    reg_norm = reg_vec.norm().item()
    range_norm = range_vec.norm().item()

    cos_task_reg = F.cosine_similarity(task_vec.unsqueeze(0), reg_vec.unsqueeze(0)).item()

    logging.info(
        "[grad-diag] epoch=%d batch=%d task_grad=%.6f %s_grad=%.6f range_grad=%.6f "
        "ratio_%s/task=%.2f ratio_range/task=%.2f cos_task_%s=%.4f",
        epoch, batch_idx, task_norm, reg_label, reg_norm, range_norm,
        reg_label, reg_norm / (task_norm + 1e-12),
        range_norm / (task_norm + 1e-12),
        reg_label, cos_task_reg,
    )

    # Per-timestep gradient analysis
    if x_out.grad is not None:
        x_out.grad = None
    optimizer.zero_grad()
    l_task.backward(retain_graph=True)
    if x_out.grad is not None:
        grad = x_out.grad.detach().float()
        label_m = parts["M_label"].bool()
        pad_m = parts["M_pad"]
        y = parts["y"]
        pos_mask = (y >= 1) & label_m
        neg_mask = (y < 1) & label_m
        unlabeled_mask = (~label_m) & (~pad_m)
        grad_per_ts = grad.norm(dim=-1)
        pos_norm = grad_per_ts[pos_mask].mean().item() if pos_mask.any() else 0.0
        neg_norm = grad_per_ts[neg_mask].mean().item() if neg_mask.any() else 0.0
        unlabeled_norm = grad_per_ts[unlabeled_mask].mean().item() if unlabeled_mask.any() else 0.0
        logging.info(
            "[grad-ts] epoch=%d batch=%d pos_norm=%.6f neg_norm=%.6f unlabeled_norm=%.6f "
            "ratio_pos/neg=%.2f n_pos=%d n_neg=%d n_unlabeled=%d n_pad=%d",
            epoch, batch_idx, pos_norm, neg_norm, unlabeled_norm,
            pos_norm / (neg_norm + 1e-12),
            pos_mask.sum().item(), neg_mask.sum().item(),
            unlabeled_mask.sum().item(), pad_m.sum().item(),
        )
    else:
        logging.info(
            "[grad-ts] epoch=%d batch=%d x_out.grad is None (no task grad reached output)",
            epoch, batch_idx,
        )


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
            batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
            
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
                batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
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
        lambda_forecast: float = 0.0,
        lambda_mmd: float = 0.0,
        lambda_mmd_transition: float = 0.0,
        target_train_loader: DataLoader | None = None,
        early_stopping_patience: int = 0,
        best_metric: str = "val_total",
        run_dir: Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        training_config: dict | None = None,
        freeze_baseline: bool = True,
    ) -> None:
        self.yaib_runtime = yaib_runtime
        self.schema_resolver = schema_resolver
        self.translator = translator.to(device)
        self.device = device
        self.freeze_baseline = freeze_baseline
        self.optimizer = AdamW(self.translator.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.lambda_fidelity = lambda_fidelity
        self.lambda_range = lambda_range
        self.lambda_forecast = lambda_forecast
        self.lambda_mmd = lambda_mmd
        self.lambda_mmd_transition = lambda_mmd_transition
        self.target_train_loader = target_train_loader
        self._target_iter = iter(target_train_loader) if target_train_loader is not None else None
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = best_metric
        self.run_dir = Path(run_dir) if run_dir is not None else Path("runs/translator")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = GradScaler(enabled=self.device.startswith("cuda"))

        self.yaib_runtime.load_baseline_model()
        if hasattr(self.yaib_runtime, "_model") and self.yaib_runtime._model is not None:
            self.yaib_runtime._model = self.yaib_runtime._model.to(device)
            if self.freeze_baseline:
                for param in self.yaib_runtime._model.parameters():
                    param.requires_grad = False
                self._apply_baseline_speed_safe_mode()
            else:
                # Unfrozen baseline (fine-tuning): add LSTM params to optimizer
                # and keep model in train() mode with active dropout
                baseline_params = list(self.yaib_runtime._model.parameters())
                self.optimizer.add_param_group({"params": baseline_params})
                self.yaib_runtime._model.train()
                logging.info(
                    "Baseline UNFROZEN: %d params added to optimizer (fine-tune mode)",
                    sum(p.numel() for p in baseline_params),
                )

        self.lower_bounds, self.upper_bounds = self._load_feature_bounds(bounds_csv, schema_resolver.dynamic_features)
        self.lower_bounds = self.lower_bounds.to(device)
        self.upper_bounds = self.upper_bounds.to(device)

        self.best_val = float("inf")
        self.best_state = None
        self.history: list[dict[str, float]] = []
        self._logged_train_batch = False
        self._logged_val_batch = False

        # Store training config for experiment-specific features
        self._training_config = training_config or {}

        # V6: LR scheduling, grad clipping, grad accumulation
        self.lr_scheduler_type = self._training_config.get("lr_scheduler", None)
        self.lr_min = self._training_config.get("lr_min", 0.0)
        self.lr_warmup_epochs = self._training_config.get("lr_warmup_epochs", 0)
        self.grad_clip_norm = self._training_config.get("grad_clip_norm", 0.0)
        self.accumulate_grad_batches = max(1, self._training_config.get("accumulate_grad_batches", 1))

        # MIMIC target task loss
        self.lambda_target_task = self._training_config.get("lambda_target_task", 0.0)
        if self.lambda_target_task > 0:
            logging.info("Target task loss enabled: lambda_target_task=%.4f", self.lambda_target_task)

        # Feature gate for weighted fidelity (optional)
        self.feature_gate = None
        if self._training_config.get("feature_gate", False):
            from ..core.feature_gate import FeatureGate
            num_features = len(schema_resolver.dynamic_features)
            self.feature_gate = FeatureGate(num_features).to(device)
            # Add gate params to optimizer
            self.optimizer.add_param_group({"params": self.feature_gate.parameters()})
            logging.info("Feature gate enabled for fidelity weighting (%d features)", num_features)

        # Cross-domain normalization
        self.renorm_scale = None   # (F,) tensor or None
        self.renorm_offset = None  # (F,) tensor or None

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

    def _next_target_batch(self) -> tuple:
        """Get next batch from the cycling target (MIMIC) data iterator."""
        try:
            batch = next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self.target_train_loader)
            batch = next(self._target_iter)
        return tuple(b.to(self.device) for b in batch)

    def set_renorm_params(self, scale: torch.Tensor, offset: torch.Tensor):
        self.renorm_scale = scale.to(self.device)
        self.renorm_offset = offset.to(self.device)
        logging.info("Cross-domain renormalization enabled for source data")

    def _apply_renorm(self, x_val: torch.Tensor, m_pad: torch.Tensor) -> torch.Tensor:
        if self.renorm_scale is None:
            return x_val
        x = x_val * self.renorm_scale.view(1, 1, -1) + self.renorm_offset.view(1, 1, -1)
        return x.masked_fill(m_pad.unsqueeze(-1).bool(), 0.0)

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

    def _run_epoch(self, loader: DataLoader, epoch: int = 0) -> dict[str, float]:
        self.translator.train()
        totals = {"total": 0.0, "task": 0.0, "fidelity": 0.0, "range": 0.0, "forecast": 0.0, "mmd": 0.0, "mmd_trans": 0.0, "target_task": 0.0}
        use_forecast = self.lambda_forecast > 0
        use_mmd = self.lambda_mmd > 0 and self.target_train_loader is not None
        use_mmd_transition = self.lambda_mmd_transition > 0 and self.target_train_loader is not None
        use_target_task = self.lambda_target_task > 0 and self.target_train_loader is not None
        num_batches = 0
        logged_this_epoch = False
        for batch in loader:
            batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
            parts = self.schema_resolver.extract(batch)
            parts["X_val"] = self._apply_renorm(parts["X_val"], parts["M_pad"])
            use_amp = self.scaler.is_enabled()
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t_translator = time.time()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                result = self.translator(
                    parts["X_val"],
                    parts["X_miss"],
                    parts["t_abs"],
                    parts["M_pad"],
                    parts["X_static"],
                    return_forecast=use_forecast,
                )
                if use_forecast:
                    x_val_out, x_forecast = result
                else:
                    x_val_out = result
                # Hard-clamp to feature bounds before LSTM (prevents NaN on OOD source data)
                x_val_out_clamped = x_val_out.float().clamp(
                    min=self.lower_bounds.view(1, 1, -1),
                    max=self.upper_bounds.view(1, 1, -1),
                )
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out_clamped, parts["X_miss"], parts["X_static"],
                    m_pad=parts["M_pad"],
                )
            # Retain grad on translator output for per-timestep gradient analysis
            _grad_diag_interval = getattr(self, "_grad_diag_interval", 1)
            _do_grad_diag = (epoch == 0 and num_batches <= 3) or \
                            (epoch > 0 and epoch % _grad_diag_interval == 0 and num_batches == 0)
            if _do_grad_diag:
                x_val_out.retain_grad()
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
            if self.feature_gate is not None:
                gate = self.feature_gate()  # (F,)
                fid_weight = (1.0 - 0.5 * gate).view(1, 1, -1)  # less fidelity where gate is high
                diff = diff * fid_weight
            l_fidelity = self._masked_mean(diff.sum(dim=-1), mask).float()
            upper = self.upper_bounds.view(1, 1, -1)
            lower = self.lower_bounds.view(1, 1, -1)
            over = torch.relu(x_val_out.float() - upper)
            under = torch.relu(lower - x_val_out.float())
            l_range = self._masked_mean((over ** 2 + under ** 2).sum(dim=-1), mask).float()
            l_forecast = x_val_out.new_tensor(0.0)
            if use_forecast:
                forecast_target = parts["X_val"][:, 1:, :]
                forecast_pred = x_forecast[:, :-1, :]
                valid = ~parts["M_pad"][:, :-1] & ~parts["M_pad"][:, 1:]
                if valid.any():
                    l_forecast = self._masked_mean(
                        (forecast_pred.float() - forecast_target.float()).pow(2).sum(dim=-1), valid
                    ).float()

            # MMD losses
            l_mmd = x_val_out.new_tensor(0.0)
            l_mmd_trans = x_val_out.new_tensor(0.0)
            if use_mmd or use_mmd_transition:
                target_batch = self._next_target_batch()
                target_parts = self.schema_resolver.extract(target_batch)
                source_mask = ~parts["M_pad"]
                target_mask = ~target_parts["M_pad"]
                if use_mmd:
                    source_features = x_val_out[source_mask]
                    target_features = target_parts["X_val"][target_mask]
                    l_mmd = multi_kernel_mmd(source_features.float(), target_features.float())
                if use_mmd_transition:
                    source_delta = x_val_out[:, 1:, :] - x_val_out[:, :-1, :]
                    target_delta = target_parts["X_val"][:, 1:, :] - target_parts["X_val"][:, :-1, :]
                    source_trans_mask = source_mask[:, :-1] & source_mask[:, 1:]
                    target_trans_mask = target_mask[:, :-1] & target_mask[:, 1:]
                    if source_trans_mask.any() and target_trans_mask.any():
                        l_mmd_trans = multi_kernel_mmd(
                            source_delta[source_trans_mask].float(),
                            target_delta[target_trans_mask].float(),
                        )

            # Target task loss: pass MIMIC through translator → frozen LSTM → MIMIC labels
            l_target_task = x_val_out.new_tensor(0.0)
            if use_target_task:
                if not (use_mmd or use_mmd_transition):
                    target_batch = self._next_target_batch()
                    target_parts = self.schema_resolver.extract(target_batch)
                tgt_val_out = self.translator(
                    target_parts["X_val"], target_parts["X_miss"],
                    target_parts["t_abs"], target_parts["M_pad"], target_parts["X_static"],
                    return_forecast=False,
                )
                tgt_yaib = self.schema_resolver.rebuild(
                    target_parts["X_yaib"], tgt_val_out.float(), target_parts["X_miss"], target_parts["X_static"],
                    m_pad=target_parts["M_pad"],
                )
                tgt_label_mask = target_parts["M_label"].bool()
                tgt_logits = self.yaib_runtime.forward((tgt_yaib, target_parts["y"], tgt_label_mask))
                l_target_task = self.yaib_runtime.compute_loss(
                    tgt_logits.float(), (tgt_yaib, target_parts["y"], tgt_label_mask)
                ).float()

            l_total = (
                l_task
                + (self.lambda_fidelity * l_fidelity)
                + (self.lambda_range * l_range)
                + (self.lambda_forecast * l_forecast)
                + (self.lambda_mmd * l_mmd)
                + (self.lambda_mmd_transition * l_mmd_trans)
                + (self.lambda_target_task * l_target_task)
            )

            # Gradient magnitude diagnostic (periodic: epoch 0 detailed, then batch 0 at intervals)
            if _do_grad_diag:
                _gradient_diagnostic(
                    self.optimizer, self.translator, x_val_out,
                    l_task, self.lambda_fidelity * l_fidelity, self.lambda_range * l_range,
                    parts, epoch, num_batches, reg_label="fid",
                )

            if num_batches % self.accumulate_grad_batches == 0:
                self.optimizer.zero_grad()
            if use_amp:
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_backward = time.time()
                self.scaler.scale(l_total / self.accumulate_grad_batches).backward()
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                backward_time = time.time() - t_backward
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
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_backward = time.time()
                (l_total / self.accumulate_grad_batches).backward()
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                backward_time = time.time() - t_backward
                if (num_batches + 1) % self.accumulate_grad_batches == 0:
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in self.optimizer.param_groups for p in g["params"]],
                            self.grad_clip_norm,
                        )
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
            totals["forecast"] += l_forecast.item()
            totals["mmd"] += l_mmd.item()
            totals["mmd_trans"] += l_mmd_trans.item()
            totals["target_task"] += l_target_task.item()
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
        totals = {"total": 0.0, "task": 0.0, "fidelity": 0.0, "range": 0.0, "forecast": 0.0, "mmd": 0.0, "mmd_trans": 0.0, "target_task": 0.0}
        use_forecast = self.lambda_forecast > 0
        use_mmd = self.lambda_mmd > 0 and self.target_train_loader is not None
        use_mmd_transition = self.lambda_mmd_transition > 0 and self.target_train_loader is not None
        use_target_task = self.lambda_target_task > 0 and self.target_train_loader is not None
        num_batches = 0
        with torch.no_grad():
            for batch in loader:
                batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
                parts = self.schema_resolver.extract(batch)
                parts["X_val"] = self._apply_renorm(parts["X_val"], parts["M_pad"])
                result = self.translator(
                    parts["X_val"],
                    parts["X_miss"],
                    parts["t_abs"],
                    parts["M_pad"],
                    parts["X_static"],
                    return_forecast=use_forecast,
                )
                if use_forecast:
                    x_val_out, x_forecast = result
                else:
                    x_val_out = result
                x_val_out_clamped = x_val_out.float().clamp(
                    min=self.lower_bounds.view(1, 1, -1),
                    max=self.upper_bounds.view(1, 1, -1),
                )
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out_clamped, parts["X_miss"], parts["X_static"],
                    m_pad=parts["M_pad"],
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
                if self.feature_gate is not None:
                    gate = self.feature_gate()
                    fid_weight = (1.0 - 0.5 * gate).view(1, 1, -1)
                    diff = diff * fid_weight
                l_fidelity = self._masked_mean(diff.sum(dim=-1), mask)
                over = torch.relu(x_val_out - self.upper_bounds.view(1, 1, -1))
                under = torch.relu(self.lower_bounds.view(1, 1, -1) - x_val_out)
                l_range = self._masked_mean((over ** 2 + under ** 2).sum(dim=-1), mask)
                l_forecast = x_val_out.new_tensor(0.0)
                if use_forecast:
                    forecast_target = parts["X_val"][:, 1:, :]
                    forecast_pred = x_forecast[:, :-1, :]
                    valid = ~parts["M_pad"][:, :-1] & ~parts["M_pad"][:, 1:]
                    if valid.any():
                        l_forecast = self._masked_mean(
                            (forecast_pred - forecast_target).pow(2).sum(dim=-1), valid
                        )

                # MMD losses (validation uses target train loader for distribution reference)
                l_mmd = x_val_out.new_tensor(0.0)
                l_mmd_trans = x_val_out.new_tensor(0.0)
                if use_mmd or use_mmd_transition:
                    target_batch = self._next_target_batch()
                    target_parts = self.schema_resolver.extract(target_batch)
                    source_mask = ~parts["M_pad"]
                    target_mask = ~target_parts["M_pad"]
                    if use_mmd:
                        source_features = x_val_out[source_mask]
                        target_features = target_parts["X_val"][target_mask]
                        l_mmd = multi_kernel_mmd(source_features.float(), target_features.float())
                    if use_mmd_transition:
                        source_delta = x_val_out[:, 1:, :] - x_val_out[:, :-1, :]
                        target_delta = target_parts["X_val"][:, 1:, :] - target_parts["X_val"][:, :-1, :]
                        source_trans_mask = source_mask[:, :-1] & source_mask[:, 1:]
                        target_trans_mask = target_mask[:, :-1] & target_mask[:, 1:]
                        if source_trans_mask.any() and target_trans_mask.any():
                            l_mmd_trans = multi_kernel_mmd(
                                source_delta[source_trans_mask].float(),
                                target_delta[target_trans_mask].float(),
                            )

                # Target task loss (validation)
                l_target_task = x_val_out.new_tensor(0.0)
                if use_target_task:
                    if not (use_mmd or use_mmd_transition):
                        target_batch = self._next_target_batch()
                        target_parts = self.schema_resolver.extract(target_batch)
                    tgt_val_out = self.translator(
                        target_parts["X_val"], target_parts["X_miss"],
                        target_parts["t_abs"], target_parts["M_pad"], target_parts["X_static"],
                        return_forecast=False,
                    )
                    tgt_yaib = self.schema_resolver.rebuild(
                        target_parts["X_yaib"], tgt_val_out.float(), target_parts["X_miss"], target_parts["X_static"],
                        m_pad=target_parts["M_pad"],
                    )
                    tgt_label_mask = target_parts["M_label"].bool()
                    tgt_logits = self.yaib_runtime.forward((tgt_yaib, target_parts["y"], tgt_label_mask))
                    l_target_task = self.yaib_runtime.compute_loss(
                        tgt_logits.float(), (tgt_yaib, target_parts["y"], tgt_label_mask)
                    ).float()

                l_total = (
                    l_task
                    + (self.lambda_fidelity * l_fidelity)
                    + (self.lambda_range * l_range)
                    + (self.lambda_forecast * l_forecast)
                    + (self.lambda_mmd * l_mmd)
                    + (self.lambda_mmd_transition * l_mmd_trans)
                    + (self.lambda_target_task * l_target_task)
                )

                totals["total"] += l_total.item()
                totals["task"] += l_task.item()
                totals["fidelity"] += l_fidelity.item()
                totals["range"] += l_range.item()
                totals["forecast"] += l_forecast.item()
                totals["mmd"] += l_mmd.item()
                totals["mmd_trans"] += l_mmd_trans.item()
                totals["target_task"] += l_target_task.item()
                num_batches += 1

        if num_batches == 0:
            return {k: float("inf") for k in totals}
        return {k: v / num_batches for k, v in totals.items()}

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader) -> None:
        if self.freeze_baseline and hasattr(self.yaib_runtime, "_model") and self.yaib_runtime._model is not None:
            try:
                sample_batch = next(iter(train_loader))
                sample_batch = tuple(b.to(self.device) for b in sample_batch)
                verify_baseline_determinism(self.yaib_runtime, sample_batch, self.device)
            except StopIteration:
                logging.warning("Baseline integrity check skipped: empty train_loader.")
        self._grad_diag_interval = max(1, epochs // 4) if epochs > 1 else 1
        logging.info("Gradient diagnostics interval: every %d epochs", self._grad_diag_interval)
        epochs_without_improvement = 0
        self.scheduler = _create_lr_scheduler(
            self.optimizer, self.lr_scheduler_type, epochs,
            lr_min=self.lr_min, warmup_epochs=self.lr_warmup_epochs,
        )
        for epoch in range(epochs):
            logging.info("Epoch %d/%d - training (lr=%.2e)", epoch + 1, epochs, self.optimizer.param_groups[0]["lr"])
            train_metrics = self._run_epoch(train_loader, epoch=epoch)
            logging.info("Epoch %d/%d - validating", epoch + 1, epochs)
            val_metrics = self._validate(val_loader)

            logging.info(
                "Epoch %d/%d - train_total=%.4f train_task=%.4f train_fidelity=%.4f train_range=%.4f train_forecast=%.4f train_mmd=%.4f train_mmd_trans=%.4f train_target_task=%.4f",
                epoch + 1,
                epochs,
                train_metrics["total"],
                train_metrics["task"],
                train_metrics["fidelity"],
                train_metrics["range"],
                train_metrics["forecast"],
                train_metrics["mmd"],
                train_metrics["mmd_trans"],
                train_metrics.get("target_task", 0.0),
            )
            logging.info(
                "Epoch %d/%d - val_total=%.4f val_task=%.4f val_fidelity=%.4f val_range=%.4f val_forecast=%.4f val_mmd=%.4f val_mmd_trans=%.4f val_target_task=%.4f",
                epoch + 1,
                epochs,
                val_metrics["total"],
                val_metrics["task"],
                val_metrics["fidelity"],
                val_metrics["range"],
                val_metrics["forecast"],
                val_metrics["mmd"],
                val_metrics["mmd_trans"],
                val_metrics.get("target_task", 0.0),
            )

            self.history.append(
                {
                    "epoch": epoch + 1,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "train_total": train_metrics["total"],
                    "train_task": train_metrics["task"],
                    "train_fidelity": train_metrics["fidelity"],
                    "train_range": train_metrics["range"],
                    "train_forecast": train_metrics["forecast"],
                    "train_mmd": train_metrics["mmd"],
                    "train_mmd_trans": train_metrics["mmd_trans"],
                    "train_target_task": train_metrics.get("target_task", 0.0),
                    "val_total": val_metrics["total"],
                    "val_task": val_metrics["task"],
                    "val_fidelity": val_metrics["fidelity"],
                    "val_range": val_metrics["range"],
                    "val_forecast": val_metrics["forecast"],
                    "val_mmd": val_metrics["mmd"],
                    "val_mmd_trans": val_metrics["mmd_trans"],
                    "val_target_task": val_metrics.get("target_task", 0.0),
                }
            )

            # Step LR scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["task"])
                else:
                    self.scheduler.step()

            candidate = val_metrics["total"] if self.best_metric == "val_total" else val_metrics["task"]
            import math
            if math.isnan(candidate):
                candidate = val_metrics.get("target_task", val_metrics.get("fidelity", float("inf")))
                if epoch == 0:
                    logging.warning(
                        "val metric is NaN — falling back to alternative metric (%.6f) "
                        "for best checkpoint selection", candidate,
                    )
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
                }
                # Save baseline model state when fine-tuning (unfrozen)
                if not self.freeze_baseline and hasattr(self.yaib_runtime, "_model") and self.yaib_runtime._model is not None:
                    checkpoint["model_state_dict"] = self.yaib_runtime._model.state_dict()
                torch.save(checkpoint, self.run_dir / "best_translator.pt")
                logging.info("Saved new best checkpoint to %s", self.run_dir / "best_translator.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
                logging.info("Early stopping after %d epochs without improvement", epochs_without_improvement)
                break

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

        train_task = [row["train_task"] for row in self.history]
        val_task = [row["val_task"] for row in self.history]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, train_task, label="train_task")
        ax.plot(epochs, val_task, label="val_task")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Task Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.run_dir / "task_loss_curve.png", dpi=150)
        plt.close(fig)

        # Fidelity loss curve
        train_fidelity = [row.get("train_fidelity", 0) for row in self.history]
        val_fidelity = [row.get("val_fidelity", 0) for row in self.history]
        if any(v > 0 for v in train_fidelity + val_fidelity):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs, train_fidelity, label="train_fidelity")
            ax.plot(epochs, val_fidelity, label="val_fidelity")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Fidelity Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.run_dir / "fidelity_loss_curve.png", dpi=150)
            plt.close(fig)

        # MMD loss curve (only if MMD was used)
        if any(row.get("train_mmd", 0) > 0 or row.get("val_mmd", 0) > 0 for row in self.history):
            train_mmd = [row.get("train_mmd", 0) for row in self.history]
            val_mmd = [row.get("val_mmd", 0) for row in self.history]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs, train_mmd, label="train_mmd")
            ax.plot(epochs, val_mmd, label="val_mmd")
            train_mmd_trans = [row.get("train_mmd_trans", 0) for row in self.history]
            val_mmd_trans = [row.get("val_mmd_trans", 0) for row in self.history]
            if any(v > 0 for v in train_mmd_trans + val_mmd_trans):
                ax.plot(epochs, train_mmd_trans, label="train_mmd_trans", linestyle="--")
                ax.plot(epochs, val_mmd_trans, label="val_mmd_trans", linestyle="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MMD Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.run_dir / "mmd_loss_curve.png", dpi=150)
            plt.close(fig)


class LatentTranslatorTrainer:
    """Trainer for SharedLatentTranslator with pretraining + joint alignment training."""

    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: nn.Module,
        schema_resolver: SchemaResolver,
        bounds_csv: Path,
        target_train_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_align: float = 0.5,
        lambda_recon: float = 0.1,
        lambda_range: float = 0.5,
        pretrain_epochs: int = 10,
        early_stopping_patience: int = 5,
        best_metric: str = "val_task",
        run_dir: Path | None = None,
        device: str = "cuda",
        training_config: dict | None = None,
    ) -> None:
        self.yaib_runtime = yaib_runtime
        self.schema_resolver = schema_resolver
        self.translator = translator.to(device)
        self.device = device
        self.lambda_align = lambda_align
        self.lambda_recon = lambda_recon
        self.lambda_range = lambda_range
        self.pretrain_epochs = pretrain_epochs
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = best_metric
        self.run_dir = Path(run_dir) if run_dir else Path("runs/shared_latent")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.target_train_loader = target_train_loader
        self._target_iter = iter(target_train_loader)

        # Cross-domain normalization
        self.renorm_scale = None   # (F,) tensor or None
        self.renorm_offset = None  # (F,) tensor or None

        # MIMIC target task loss and latent label prediction
        _tc = training_config or {}
        self._training_config = _tc
        self.task_type = _tc.get("task_type", "classification")
        self.lambda_target_task = _tc.get("lambda_target_task", 0.0)
        self.lambda_label_pred = _tc.get("lambda_label_pred", 0.0)
        self.lambda_contrastive_align = _tc.get("lambda_contrastive_align", 0.0)
        self.recon_positive_boost = _tc.get("recon_positive_boost", 0.0)

        # V6: LR scheduling, grad clipping, grad accumulation
        self.lr_scheduler_type = _tc.get("lr_scheduler", None)
        self.lr_min = _tc.get("lr_min", 0.0)
        self.lr_warmup_epochs = _tc.get("lr_warmup_epochs", 0)
        self.grad_clip_norm = _tc.get("grad_clip_norm", 0.0)
        self.accumulate_grad_batches = max(1, _tc.get("accumulate_grad_batches", 1))

        if self.lambda_target_task > 0:
            logging.info("Target task loss enabled: lambda_target_task=%.4f", self.lambda_target_task)
        if self.lambda_label_pred > 0:
            logging.info("Latent label prediction enabled: lambda_label_pred=%.4f", self.lambda_label_pred)
        if self.lambda_contrastive_align > 0:
            logging.info("Contrastive alignment enabled: lambda=%.4f", self.lambda_contrastive_align)
        if self.recon_positive_boost > 0:
            logging.info("Positive-weighted reconstruction enabled: boost=%.1f", self.recon_positive_boost)

        # Load and freeze baseline (before feature gate, so LSTM weights are available)
        self.yaib_runtime.load_baseline_model()
        if hasattr(self.yaib_runtime, "_model") and self.yaib_runtime._model is not None:
            self.yaib_runtime._model = self.yaib_runtime._model.to(device)
            for param in self.yaib_runtime._model.parameters():
                param.requires_grad = False
            model = self.yaib_runtime._model
            # Must be in train mode for cudnn RNN backward, but force
            # dropout/batchnorm to eval for deterministic inference
            for name, module in model.named_modules():
                if hasattr(module, "dropout") and isinstance(getattr(module, "dropout"), float):
                    if module.dropout > 0:
                        module.dropout = 0.0
            model.train()
            from torch.nn.modules.dropout import _DropoutNd
            from torch.nn.modules.batchnorm import _BatchNorm
            def _force_stateless(m):
                if isinstance(m, (_DropoutNd, nn.Dropout)):
                    m.eval()
                elif isinstance(m, _BatchNorm):
                    m.eval()
                    m.track_running_stats = False
            model.apply(_force_stateless)

        # Feature gate for weighted reconstruction (optional)
        self.feature_gate = None
        if _tc.get("feature_gate", False):
            from ..core.feature_gate import FeatureGate
            num_features = len(schema_resolver.dynamic_features)
            init_logits = None
            if _tc.get("lstm_informed_gate", False) and hasattr(self.yaib_runtime, "_model"):
                from ..core.lstm_importance import extract_lstm_feature_importance
                init_logits = extract_lstm_feature_importance(
                    self.yaib_runtime._model,
                    num_dynamic_features=num_features,
                    dynamic_feature_offset=0,
                )
                logging.info("LSTM-informed gate initialization enabled")
            self.feature_gate = FeatureGate(num_features, init_logits=init_logits).to(device)
            logging.info("Feature gate enabled for reconstruction weighting (%d features)", num_features)

        params = list(self.translator.parameters())
        if self.feature_gate is not None:
            params += list(self.feature_gate.parameters())
        self.optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.scaler = GradScaler(enabled=device.startswith("cuda"))

        # Snapshot frozen baseline parameters for post-training verification
        self._baseline_param_snapshot = {
            name: param.detach().clone()
            for name, param in self.yaib_runtime._model.named_parameters()
        }

        # Feature bounds for range loss
        self.lower_bounds, self.upper_bounds = self._load_feature_bounds(
            bounds_csv, schema_resolver.dynamic_features
        )
        self.lower_bounds = self.lower_bounds.to(device)
        self.upper_bounds = self.upper_bounds.to(device)

        self.best_val = float("inf")
        self.best_state = None
        self.history: list[dict] = []

    def _label_pred_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type == "regression":
            return F.mse_loss(logits.float(), targets.float())
        return F.binary_cross_entropy_with_logits(logits.float(), targets.float())

    def _load_feature_bounds(self, bounds_csv, feature_names):
        import pandas as pd
        df = pd.read_csv(bounds_csv)
        df = df.set_index("feature")
        cols = set(df.columns)
        lower_col = next((c for c in ["p0.1_a", "p0.1", "p_001_a", "q001"] if c in cols), None)
        upper_col = next((c for c in ["p99.9_a", "p99.9", "p_999_a", "q999"] if c in cols), None)
        if lower_col is None or upper_col is None:
            raise ValueError(f"Bounds CSV missing percentile columns. Found: {sorted(cols)}")
        lower = torch.tensor(df.loc[feature_names, lower_col].to_numpy(), dtype=torch.float32)
        upper = torch.tensor(df.loc[feature_names, upper_col].to_numpy(), dtype=torch.float32)
        return lower, upper

    def _verify_baseline_frozen(self) -> None:
        """Verify that frozen LSTM parameters are exactly unchanged after training."""
        model = self.yaib_runtime._model
        n_params = 0
        max_diff = 0.0
        for name, param in model.named_parameters():
            snapshot = self._baseline_param_snapshot[name]
            diff = (param.detach() - snapshot.to(param.device)).abs().max().item()
            max_diff = max(max_diff, diff)
            n_params += param.numel()
            if diff > 0:
                raise RuntimeError(
                    f"FROZEN LSTM CORRUPTED: parameter '{name}' changed by {diff:.2e} "
                    f"during training. model.train() mode may have caused weight updates."
                )
        logging.info(
            "[verify] Frozen LSTM integrity OK — %d parameters, max_diff=%.2e",
            n_params, max_diff,
        )

    def _next_target_batch(self):
        try:
            batch = next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self.target_train_loader)
            batch = next(self._target_iter)
        return tuple(b.to(self.device) for b in batch)

    def set_renorm_params(self, scale: torch.Tensor, offset: torch.Tensor):
        self.renorm_scale = scale.to(self.device)
        self.renorm_offset = offset.to(self.device)
        logging.info("Cross-domain renormalization enabled for source data (SL)")

    def _apply_renorm(self, x_val: torch.Tensor, m_pad: torch.Tensor) -> torch.Tensor:
        if self.renorm_scale is None:
            return x_val
        x = x_val * self.renorm_scale.view(1, 1, -1) + self.renorm_offset.view(1, 1, -1)
        return x.masked_fill(m_pad.unsqueeze(-1).bool(), 0.0)

    def _pretrain_epoch(self, target_loader: DataLoader) -> dict:
        """Autoencoder pretraining on MIMIC target data, optionally with label prediction."""
        self.translator.train()
        total_recon = 0.0
        total_label_pred = 0.0
        n_batches = 0

        for batch in target_loader:
            batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
            parts = self.schema_resolver.extract(batch)

            with torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")):
                # Encode → decode for reconstruction
                latent = self.translator.encode(
                    parts["X_val"], parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], parts["X_static"],
                )
                x_out = self.translator.decode(latent, parts["M_pad"], parts["X_static"])
                mask = ~parts["M_pad"].bool()
                diff = (x_out.float() - parts["X_val"].float()) ** 2
                l_recon = diff.sum(dim=-1)[mask].mean() if mask.any() else diff.new_tensor(0.0)

                # Label prediction from latent (teaches encoder task-relevant features)
                l_label_pred = latent.new_tensor(0.0)
                if self.lambda_label_pred > 0:
                    label_logits = self.translator.predict_labels(latent, parts["M_pad"])
                    label_mask = parts["M_label"].bool()
                    if label_mask.any():
                        l_label_pred = self._label_pred_loss(
                            label_logits[label_mask],
                            parts["y"][label_mask],
                        )

                loss = l_recon + self.lambda_label_pred * l_label_pred

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_recon += l_recon.item()
            total_label_pred += l_label_pred.item()
            n_batches += 1

        return {
            "pretrain_recon": total_recon / max(n_batches, 1),
            "pretrain_label_pred": total_label_pred / max(n_batches, 1),
        }

    def _run_epoch(self, train_loader: DataLoader, epoch: int = 0) -> dict:
        """Joint training: task + alignment + reconstruction + range + target_task + label_pred."""
        self.translator.train()
        totals = {"total": 0.0, "task": 0.0, "align": 0.0, "recon": 0.0, "range": 0.0,
                  "target_task": 0.0, "label_pred": 0.0, "contrastive": 0.0}
        n_batches = 0

        for batch in train_loader:
            batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
            parts = self.schema_resolver.extract(batch)
            parts["X_val"] = self._apply_renorm(parts["X_val"], parts["M_pad"])

            _grad_diag_interval = getattr(self, "_grad_diag_interval", 1)
            _do_grad_diag = (epoch == 0 and n_batches <= 3) or \
                            (epoch > 0 and epoch % _grad_diag_interval == 0 and n_batches == 0)

            with torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")):
                # ── Source (eICU) path ──
                src_latent = self.translator.encode(
                    parts["X_val"], parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], parts["X_static"],
                )
                x_out = self.translator.decode(src_latent, parts["M_pad"], parts["X_static"])
                if _do_grad_diag:
                    x_out.retain_grad()

                # Hard-clamp to feature bounds before LSTM (prevents NaN on OOD source data)
                x_out_clamped = x_out.float().clamp(
                    min=self.lower_bounds.view(1, 1, -1),
                    max=self.upper_bounds.view(1, 1, -1),
                )
                # Rebuild YAIB batch and get task loss
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_out_clamped, parts["X_miss"], parts["X_static"],
                    m_pad=parts["M_pad"],
                )
                label_mask = parts["M_label"].bool()
                logits = self.yaib_runtime.forward(
                    (x_yaib_translated, parts["y"], label_mask)
                )
                l_task = self.yaib_runtime.compute_loss(
                    logits, (x_yaib_translated, parts["y"], label_mask)
                )

                # ── Target (MIMIC) path ──
                tgt_batch = self._next_target_batch()
                tgt_parts = self.schema_resolver.extract(tgt_batch)
                tgt_latent = self.translator.encode(
                    tgt_parts["X_val"], tgt_parts["X_miss"], tgt_parts["t_abs"],
                    tgt_parts["M_pad"], tgt_parts["X_static"],
                )

                # Alignment loss: MMD in latent space
                src_mask = ~parts["M_pad"].bool()
                tgt_mask = ~tgt_parts["M_pad"].bool()
                src_z = src_latent[src_mask].float()  # (N, d_latent)
                tgt_z = tgt_latent[tgt_mask].float()  # (M, d_latent)
                l_align = multi_kernel_mmd(src_z, tgt_z) if src_z.shape[0] > 1 and tgt_z.shape[0] > 1 else src_z.new_tensor(0.0)

                # Reconstruction loss: decode MIMIC and compare
                tgt_out = self.translator.decode(tgt_latent, tgt_parts["M_pad"], tgt_parts["X_static"])
                tgt_diff = (tgt_out.float() - tgt_parts["X_val"].float()) ** 2
                if self.feature_gate is not None:
                    gate = self.feature_gate()  # (F,)
                    tgt_diff = tgt_diff * gate.view(1, 1, -1)
                tgt_recon_per_ts = tgt_diff.sum(dim=-1)  # (B, T)
                if self.recon_positive_boost > 0 and self.task_type != "regression":
                    tgt_labels = tgt_parts["y"].float()
                    tgt_label_available = tgt_parts["M_label"].bool()
                    pos_mask = tgt_label_available & (tgt_labels > 0.5)
                    ts_weight = torch.ones_like(tgt_recon_per_ts)
                    ts_weight[pos_mask] = 1.0 + self.recon_positive_boost
                    tgt_recon_per_ts = tgt_recon_per_ts * ts_weight
                l_recon = tgt_recon_per_ts[tgt_mask].mean() if tgt_mask.any() else tgt_diff.new_tensor(0.0)

                # Range loss on source output
                upper = self.upper_bounds.view(1, 1, -1)
                lower = self.lower_bounds.view(1, 1, -1)
                over = torch.relu(x_out.float() - upper)
                under = torch.relu(lower - x_out.float())
                range_penalty = (over ** 2 + under ** 2).sum(dim=-1)
                l_range = range_penalty[src_mask].mean() if src_mask.any() else range_penalty.new_tensor(0.0)

                # Target task loss: decoded MIMIC → frozen LSTM → MIMIC labels
                l_target_task = tgt_out.new_tensor(0.0)
                if self.lambda_target_task > 0:
                    tgt_yaib = self.schema_resolver.rebuild(
                        tgt_parts["X_yaib"], tgt_out.float(), tgt_parts["X_miss"], tgt_parts["X_static"],
                        m_pad=tgt_parts["M_pad"],
                    )
                    tgt_label_mask = tgt_parts["M_label"].bool()
                    tgt_logits = self.yaib_runtime.forward((tgt_yaib, tgt_parts["y"], tgt_label_mask))
                    l_target_task = self.yaib_runtime.compute_loss(
                        tgt_logits.float(), (tgt_yaib, tgt_parts["y"], tgt_label_mask)
                    ).float()

                # Label prediction from latent (both domains)
                l_label_pred = src_latent.new_tensor(0.0)
                if self.lambda_label_pred > 0:
                    # Source labels
                    src_label_logits = self.translator.predict_labels(src_latent, parts["M_pad"])
                    src_label_mask = parts["M_label"].bool()
                    if src_label_mask.any():
                        l_src_lp = self._label_pred_loss(
                            src_label_logits[src_label_mask],
                            parts["y"][src_label_mask],
                        )
                    else:
                        l_src_lp = src_latent.new_tensor(0.0)
                    # Target labels
                    tgt_label_logits = self.translator.predict_labels(tgt_latent, tgt_parts["M_pad"])
                    tgt_label_mask_lp = tgt_parts["M_label"].bool()
                    if tgt_label_mask_lp.any():
                        l_tgt_lp = self._label_pred_loss(
                            tgt_label_logits[tgt_label_mask_lp],
                            tgt_parts["y"][tgt_label_mask_lp],
                        )
                    else:
                        l_tgt_lp = tgt_latent.new_tensor(0.0)
                    l_label_pred = (l_src_lp + l_tgt_lp) / 2.0

                # Cross-domain contrastive alignment (optional)
                l_contrastive = src_latent.new_tensor(0.0)
                if self.lambda_contrastive_align > 0:
                    src_valid = ~parts["M_pad"].bool() & parts["M_label"].bool()
                    tgt_valid = ~tgt_parts["M_pad"].bool() & tgt_parts["M_label"].bool()
                    src_lats = src_latent[src_valid].float()
                    src_labs = parts["y"][src_valid]
                    tgt_lats = tgt_latent[tgt_valid].float()
                    tgt_labs = tgt_parts["y"][tgt_valid]
                    n_terms = 0
                    src_pos = src_lats[src_labs > 0.5]
                    src_neg = src_lats[src_labs < 0.5]
                    tgt_pos = tgt_lats[tgt_labs > 0.5]
                    tgt_neg = tgt_lats[tgt_labs < 0.5]
                    if src_pos.shape[0] > 0 and tgt_pos.shape[0] > 0:
                        l_contrastive = l_contrastive + F.mse_loss(src_pos.mean(0), tgt_pos.mean(0))
                        n_terms += 1
                    if src_neg.shape[0] > 0 and tgt_neg.shape[0] > 0:
                        l_contrastive = l_contrastive + F.mse_loss(src_neg.mean(0), tgt_neg.mean(0))
                        n_terms += 1
                    if n_terms > 0:
                        l_contrastive = l_contrastive / n_terms

                l_total = (
                    l_task
                    + self.lambda_align * l_align
                    + self.lambda_recon * l_recon
                    + self.lambda_range * l_range
                    + self.lambda_target_task * l_target_task
                    + self.lambda_label_pred * l_label_pred
                    + self.lambda_contrastive_align * l_contrastive
                )

            if _do_grad_diag:
                _gradient_diagnostic(
                    self.optimizer, self.translator, x_out,
                    l_task, self.lambda_recon * l_recon, self.lambda_range * l_range,
                    parts, epoch, n_batches, reg_label="recon",
                )

            if n_batches % self.accumulate_grad_batches == 0:
                self.optimizer.zero_grad()
            self.scaler.scale(l_total / self.accumulate_grad_batches).backward()
            if (n_batches + 1) % self.accumulate_grad_batches == 0:
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.optimizer.param_groups for p in g["params"]],
                        self.grad_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            totals["total"] += l_total.item()
            totals["task"] += l_task.item()
            totals["align"] += l_align.item()
            totals["recon"] += l_recon.item()
            totals["range"] += l_range.item()
            totals["target_task"] += l_target_task.item()
            totals["label_pred"] += l_label_pred.item()
            totals["contrastive"] += l_contrastive.item()
            n_batches += 1

        # Flush remaining accumulated gradients
        if self.accumulate_grad_batches > 1 and n_batches % self.accumulate_grad_batches != 0:
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    self.grad_clip_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> dict:
        """Validation pass."""
        self.translator.eval()
        totals = {"total": 0.0, "task": 0.0, "align": 0.0, "recon": 0.0, "range": 0.0,
                  "target_task": 0.0, "label_pred": 0.0, "contrastive": 0.0}
        n_batches = 0

        for batch in val_loader:
            batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
            parts = self.schema_resolver.extract(batch)
            parts["X_val"] = self._apply_renorm(parts["X_val"], parts["M_pad"])

            with torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")):
                src_latent = self.translator.encode(
                    parts["X_val"], parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], parts["X_static"],
                )
                x_out = self.translator.decode(src_latent, parts["M_pad"], parts["X_static"])

                x_out_clamped = x_out.float().clamp(
                    min=self.lower_bounds.view(1, 1, -1),
                    max=self.upper_bounds.view(1, 1, -1),
                )
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_out_clamped, parts["X_miss"], parts["X_static"],
                    m_pad=parts["M_pad"],
                )
                label_mask = parts["M_label"].bool()
                logits = self.yaib_runtime.forward(
                    (x_yaib_translated, parts["y"], label_mask)
                )
                l_task = self.yaib_runtime.compute_loss(
                    logits, (x_yaib_translated, parts["y"], label_mask)
                )

                # Alignment with target
                tgt_batch = self._next_target_batch()
                tgt_parts = self.schema_resolver.extract(tgt_batch)
                tgt_latent = self.translator.encode(
                    tgt_parts["X_val"], tgt_parts["X_miss"], tgt_parts["t_abs"],
                    tgt_parts["M_pad"], tgt_parts["X_static"],
                )
                src_mask = ~parts["M_pad"].bool()
                tgt_mask = ~tgt_parts["M_pad"].bool()
                src_z = src_latent[src_mask].float()
                tgt_z = tgt_latent[tgt_mask].float()
                l_align = multi_kernel_mmd(src_z, tgt_z) if src_z.shape[0] > 1 and tgt_z.shape[0] > 1 else src_z.new_tensor(0.0)

                tgt_out = self.translator.decode(tgt_latent, tgt_parts["M_pad"], tgt_parts["X_static"])
                tgt_diff = (tgt_out.float() - tgt_parts["X_val"].float()) ** 2
                if self.feature_gate is not None:
                    gate = self.feature_gate()
                    tgt_diff = tgt_diff * gate.view(1, 1, -1)
                l_recon = tgt_diff.sum(dim=-1)[tgt_mask].mean() if tgt_mask.any() else tgt_diff.new_tensor(0.0)

                upper = self.upper_bounds.view(1, 1, -1)
                lower = self.lower_bounds.view(1, 1, -1)
                over = torch.relu(x_out.float() - upper)
                under = torch.relu(lower - x_out.float())
                range_penalty = (over ** 2 + under ** 2).sum(dim=-1)
                l_range = range_penalty[src_mask].mean() if src_mask.any() else range_penalty.new_tensor(0.0)

                # Target task loss (validation)
                l_target_task = tgt_out.new_tensor(0.0)
                if self.lambda_target_task > 0:
                    tgt_yaib = self.schema_resolver.rebuild(
                        tgt_parts["X_yaib"], tgt_out.float(), tgt_parts["X_miss"], tgt_parts["X_static"],
                        m_pad=tgt_parts["M_pad"],
                    )
                    tgt_label_mask = tgt_parts["M_label"].bool()
                    tgt_logits = self.yaib_runtime.forward((tgt_yaib, tgt_parts["y"], tgt_label_mask))
                    l_target_task = self.yaib_runtime.compute_loss(
                        tgt_logits.float(), (tgt_yaib, tgt_parts["y"], tgt_label_mask)
                    ).float()

                # Label prediction (validation)
                l_label_pred = src_latent.new_tensor(0.0)
                if self.lambda_label_pred > 0:
                    src_label_logits = self.translator.predict_labels(src_latent, parts["M_pad"])
                    src_label_mask = parts["M_label"].bool()
                    if src_label_mask.any():
                        l_src_lp = self._label_pred_loss(
                            src_label_logits[src_label_mask],
                            parts["y"][src_label_mask],
                        )
                    else:
                        l_src_lp = src_latent.new_tensor(0.0)
                    tgt_label_logits = self.translator.predict_labels(tgt_latent, tgt_parts["M_pad"])
                    tgt_label_mask_lp = tgt_parts["M_label"].bool()
                    if tgt_label_mask_lp.any():
                        l_tgt_lp = self._label_pred_loss(
                            tgt_label_logits[tgt_label_mask_lp],
                            tgt_parts["y"][tgt_label_mask_lp],
                        )
                    else:
                        l_tgt_lp = tgt_latent.new_tensor(0.0)
                    l_label_pred = (l_src_lp + l_tgt_lp) / 2.0

                # Cross-domain contrastive alignment (validation)
                l_contrastive = src_latent.new_tensor(0.0)
                if self.lambda_contrastive_align > 0:
                    src_valid = ~parts["M_pad"].bool() & parts["M_label"].bool()
                    tgt_valid = ~tgt_parts["M_pad"].bool() & tgt_parts["M_label"].bool()
                    src_lats = src_latent[src_valid].float()
                    src_labs = parts["y"][src_valid]
                    tgt_lats = tgt_latent[tgt_valid].float()
                    tgt_labs = tgt_parts["y"][tgt_valid]
                    n_terms = 0
                    src_pos = src_lats[src_labs > 0.5]
                    src_neg = src_lats[src_labs < 0.5]
                    tgt_pos = tgt_lats[tgt_labs > 0.5]
                    tgt_neg = tgt_lats[tgt_labs < 0.5]
                    if src_pos.shape[0] > 0 and tgt_pos.shape[0] > 0:
                        l_contrastive = l_contrastive + F.mse_loss(src_pos.mean(0), tgt_pos.mean(0))
                        n_terms += 1
                    if src_neg.shape[0] > 0 and tgt_neg.shape[0] > 0:
                        l_contrastive = l_contrastive + F.mse_loss(src_neg.mean(0), tgt_neg.mean(0))
                        n_terms += 1
                    if n_terms > 0:
                        l_contrastive = l_contrastive / n_terms

                l_total = (
                    l_task
                    + self.lambda_align * l_align
                    + self.lambda_recon * l_recon
                    + self.lambda_range * l_range
                    + self.lambda_target_task * l_target_task
                    + self.lambda_label_pred * l_label_pred
                    + self.lambda_contrastive_align * l_contrastive
                )

            totals["total"] += l_total.item()
            totals["task"] += l_task.item()
            totals["align"] += l_align.item()
            totals["recon"] += l_recon.item()
            totals["range"] += l_range.item()
            totals["target_task"] += l_target_task.item()
            totals["label_pred"] += l_label_pred.item()
            totals["contrastive"] += l_contrastive.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # Phase 1: Autoencoder pretraining on MIMIC
        pretrain_path = self.run_dir / "pretrain_checkpoint.pt"
        if pretrain_path.exists():
            ckpt = torch.load(pretrain_path, map_location=self.device, weights_only=True)
            self.translator.load_state_dict(ckpt["translator_state_dict"])
            logging.info("Loaded pretrain checkpoint from %s — skipping Phase 1", pretrain_path)
        elif self.pretrain_epochs > 0:
            logging.info("=== Phase 1: Autoencoder pretraining on MIMIC (%d epochs) ===", self.pretrain_epochs)
            for ep in range(self.pretrain_epochs):
                metrics = self._pretrain_epoch(self.target_train_loader)
                lp_str = ""
            if self.lambda_label_pred > 0:
                lp_str = " label_pred=%.4f" % metrics.get("pretrain_label_pred", 0.0)
            logging.info("Pretrain epoch %d/%d - recon=%.4f%s", ep + 1, self.pretrain_epochs, metrics["pretrain_recon"], lp_str)

            # Save pretrain checkpoint
            pretrain_path = self.run_dir / "pretrain_checkpoint.pt"
            torch.save({"translator_state_dict": self.translator.state_dict()}, pretrain_path)
            logging.info("Saved pretrain checkpoint to %s", pretrain_path)

            # Reset optimizer after pretraining and free GPU memory
            params = list(self.translator.parameters())
            if self.feature_gate is not None:
                params += list(self.feature_gate.parameters())
            self.optimizer = AdamW(params, lr=self.optimizer.defaults["lr"],
                                   weight_decay=self.optimizer.defaults["weight_decay"])
            self.scaler = GradScaler(enabled=self.device.startswith("cuda"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("=== Phase 1 complete. Starting Phase 2: Joint training ===")

        # Phase 2: Joint training
        logging.info("=== Phase 2: Joint training (%d epochs) ===", epochs)
        self._grad_diag_interval = max(1, epochs // 4) if epochs > 1 else 1
        logging.info("Gradient diagnostics interval: every %d epochs", self._grad_diag_interval)
        epochs_without_improvement = 0
        self.scheduler = _create_lr_scheduler(
            self.optimizer, self.lr_scheduler_type, epochs,
            lr_min=self.lr_min, warmup_epochs=self.lr_warmup_epochs,
        )
        for epoch in range(epochs):
            train_metrics = self._run_epoch(train_loader, epoch=epoch)
            val_metrics = self._validate(val_loader)

            logging.info(
                "Epoch %d/%d - train: total=%.4f task=%.4f align=%.4f recon=%.4f range=%.4f target_task=%.4f label_pred=%.4f (lr=%.2e)",
                epoch + 1, epochs, train_metrics["total"], train_metrics["task"],
                train_metrics["align"], train_metrics["recon"], train_metrics["range"],
                train_metrics.get("target_task", 0.0), train_metrics.get("label_pred", 0.0),
                self.optimizer.param_groups[0]["lr"],
            )
            logging.info(
                "Epoch %d/%d - val: total=%.4f task=%.4f align=%.4f recon=%.4f range=%.4f target_task=%.4f label_pred=%.4f",
                epoch + 1, epochs, val_metrics["total"], val_metrics["task"],
                val_metrics["align"], val_metrics["recon"], val_metrics["range"],
                val_metrics.get("target_task", 0.0), val_metrics.get("label_pred", 0.0),
            )

            self.history.append({
                "epoch": epoch + 1,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            # Step LR scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["task"])
                else:
                    self.scheduler.step()

            candidate = val_metrics["task"] if self.best_metric == "val_task" else val_metrics["total"]
            import math
            if math.isnan(candidate):
                candidate = val_metrics.get("target_task", float("inf"))
                if epoch == 0:
                    logging.warning(
                        "val_task is NaN — falling back to val_target_task (%.6f) "
                        "for best checkpoint selection", candidate,
                    )
            if candidate < self.best_val:
                self.best_val = candidate
                self.best_state = self.translator.state_dict()
                torch.save({
                    "epoch": epoch,
                    "translator_state_dict": self.best_state,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "renorm_scale": self.renorm_scale,
                    "renorm_offset": self.renorm_offset,
                }, self.run_dir / "best_translator.pt")
                logging.info("Saved new best checkpoint to %s", self.run_dir / "best_translator.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
                logging.info("Early stopping after %d epochs without improvement", epochs_without_improvement)
                break

        if self.best_state is not None:
            self.translator.load_state_dict(self.best_state)

        # Verify frozen LSTM stayed exactly the same
        self._verify_baseline_frozen()
        logging.info("Shared latent translator training completed")


class RetrievalTranslatorTrainer:
    """Trainer for RetrievalTranslator: Phase 1 pretrain + Phase 2 retrieval-guided training."""

    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: nn.Module,
        schema_resolver: SchemaResolver,
        bounds_csv: Path,
        target_train_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_recon: float = 0.1,
        lambda_range: float = 0.5,
        lambda_smooth: float = 0.1,
        lambda_importance_reg: float = 0.01,
        lambda_align: float = 0.0,
        pretrain_epochs: int = 10,
        k_neighbors: int = 16,
        retrieval_window: int = 6,
        memory_refresh_epochs: int = 5,
        early_stopping_patience: int = 5,
        best_metric: str = "val_task",
        run_dir: Path | None = None,
        device: str = "cuda",
        training_config: dict | None = None,
    ) -> None:
        self.yaib_runtime = yaib_runtime
        self.schema_resolver = schema_resolver
        self.translator = translator.to(device)
        self.device = device
        self.lambda_recon = lambda_recon
        self.lambda_range = lambda_range
        self.lambda_smooth = lambda_smooth
        self.lambda_importance_reg = lambda_importance_reg
        self.lambda_align = lambda_align
        self.pretrain_epochs = pretrain_epochs
        self.k_neighbors = k_neighbors
        self.retrieval_window = retrieval_window
        self.memory_refresh_epochs = memory_refresh_epochs
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = best_metric
        self.run_dir = Path(run_dir) if run_dir else Path("runs/retrieval")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.target_train_loader = target_train_loader
        self._target_iter = iter(target_train_loader)

        # Cross-domain normalization
        self.renorm_scale = None
        self.renorm_offset = None

        # MIMIC target task loss and latent label prediction
        _tc = training_config or {}
        self._training_config = _tc
        self.task_type = _tc.get("task_type", "classification")
        self.window_stride = _tc.get("window_stride", None)  # None = window_size (non-overlapping)
        self.lambda_target_task = _tc.get("lambda_target_task", 0.0)
        self.lambda_label_pred = _tc.get("lambda_label_pred", 0.0)
        self.importance_reg_type = _tc.get("importance_reg_type", "l1")
        self.lambda_contrastive_align = _tc.get("lambda_contrastive_align", 0.0)
        self.recon_positive_boost = _tc.get("recon_positive_boost", 0.0)

        # V6: LR scheduling, grad clipping, self-retrieval, grad accumulation
        self.lr_scheduler_type = _tc.get("lr_scheduler", None)
        self.lr_min = _tc.get("lr_min", 0.0)
        self.lr_warmup_epochs = _tc.get("lr_warmup_epochs", 0)
        self.grad_clip_norm = _tc.get("grad_clip_norm", 0.0)
        self.phase1_self_retrieval = _tc.get("phase1_self_retrieval", False)
        self.phase1_memory_refresh_epochs = _tc.get(
            "phase1_memory_refresh_epochs", self.memory_refresh_epochs
        )
        self.accumulate_grad_batches = max(1, _tc.get("accumulate_grad_batches", 1))

        # V7 (NeurIPS): TCR — target context reconstruction through full retrieval pipeline
        self.lambda_tcr = _tc.get("lambda_tcr", 0.0)
        # V7 (NeurIPS): FJS — Jacobian-initialized feature gate
        self.jacobian_init = _tc.get("jacobian_init", False)

        # V7: Adaptive memory bank refresh
        self.adaptive_refresh = _tc.get("adaptive_refresh", False)
        self.adaptive_refresh_patience = max(1, _tc.get("adaptive_refresh_patience", 2))
        self.adaptive_refresh_metric = _tc.get("adaptive_refresh_metric", "align")
        self.adaptive_refresh_min_delta = _tc.get("adaptive_refresh_min_delta", 0.001)

        # V8: Class-Conditional Retrieval (CCR)
        self.ccr_alpha = _tc.get("ccr_alpha", 0.0)
        if self.ccr_alpha > 0:
            logging.info("CCR (class-conditional retrieval) enabled: alpha=%.2f", self.ccr_alpha)

        # V8: Adaptive Fidelity Scheduling (AFS)
        self.fidelity_schedule = _tc.get("fidelity_schedule", None)
        self.fidelity_decay_start_epoch = _tc.get("fidelity_decay_start_epoch", 5)
        self.fidelity_decay_end_epoch = _tc.get("fidelity_decay_end_epoch", 40)
        self.fidelity_min_ratio = max(0.01, _tc.get("fidelity_min_ratio", 0.1))
        self._initial_lambda_recon = self.lambda_recon
        if self.fidelity_schedule:
            logging.info(
                "AFS enabled: schedule=%s, start=%d, end=%d, min_ratio=%.2f (lambda_recon: %.4f → %.4f)",
                self.fidelity_schedule, self.fidelity_decay_start_epoch,
                self.fidelity_decay_end_epoch, self.fidelity_min_ratio,
                self.lambda_recon, self.lambda_recon * self.fidelity_min_ratio,
            )
        self._last_refresh_epoch = -999
        self._metric_at_last_refresh = None
        self._refresh_history = []

        # Guard: if adaptive + align metric but no MMD loss, fall back to recon
        if self.adaptive_refresh and self.adaptive_refresh_metric == "align" and self.lambda_align <= 0:
            logging.warning(
                "adaptive_refresh_metric='align' but lambda_align=0 (val_align always 0). "
                "Falling back to metric='recon'."
            )
            self.adaptive_refresh_metric = "recon"

        if self.lambda_tcr > 0:
            logging.info("TCR (target context reconstruction) enabled: lambda_tcr=%.4f", self.lambda_tcr)
        if self.lambda_target_task > 0:
            logging.info("Target task loss enabled: lambda_target_task=%.4f", self.lambda_target_task)
        if self.lambda_label_pred > 0:
            logging.info("Latent label prediction enabled: lambda_label_pred=%.4f", self.lambda_label_pred)
        if self.importance_reg_type != "l1":
            logging.info("Importance reg type: %s", self.importance_reg_type)
        if self.lambda_contrastive_align > 0:
            logging.info("Contrastive alignment enabled: lambda=%.4f", self.lambda_contrastive_align)
        if self.recon_positive_boost > 0:
            logging.info("Positive-weighted reconstruction enabled: boost=%.1f", self.recon_positive_boost)
        if self.lambda_align > 0:
            logging.info("Latent MMD alignment enabled: lambda_align=%.4f", self.lambda_align)
        if self.phase1_self_retrieval:
            logging.info("Phase 1 self-retrieval enabled (refresh every %d epochs)", self.phase1_memory_refresh_epochs)
        if self.lr_scheduler_type:
            logging.info("LR scheduler: %s (min=%.1e, warmup=%d)", self.lr_scheduler_type, self.lr_min, self.lr_warmup_epochs)
        if self.grad_clip_norm > 0:
            logging.info("Gradient clipping: max_norm=%.1f", self.grad_clip_norm)
        if self.accumulate_grad_batches > 1:
            logging.info("Gradient accumulation: %d batches", self.accumulate_grad_batches)
        if self.adaptive_refresh:
            logging.info(
                "Adaptive memory refresh enabled: metric=%s, min_delta=%.4f, patience=%d",
                self.adaptive_refresh_metric, self.adaptive_refresh_min_delta,
                self.adaptive_refresh_patience,
            )

        # Load and freeze baseline (before feature gate, so LSTM weights are available)
        self.yaib_runtime.load_baseline_model()
        if hasattr(self.yaib_runtime, "_model") and self.yaib_runtime._model is not None:
            self.yaib_runtime._model = self.yaib_runtime._model.to(device)
            for param in self.yaib_runtime._model.parameters():
                param.requires_grad = False
            model = self.yaib_runtime._model
            for name, module in model.named_modules():
                if hasattr(module, "dropout") and isinstance(getattr(module, "dropout"), float):
                    if module.dropout > 0:
                        module.dropout = 0.0
            model.train()
            from torch.nn.modules.dropout import _DropoutNd
            from torch.nn.modules.batchnorm import _BatchNorm
            def _force_stateless(m):
                if isinstance(m, (_DropoutNd, nn.Dropout)):
                    m.eval()
                elif isinstance(m, _BatchNorm):
                    m.eval()
                    m.track_running_stats = False
            model.apply(_force_stateless)

        # Feature gate for weighted reconstruction (optional)
        self.feature_gate = None
        if _tc.get("feature_gate", False):
            from ..core.feature_gate import FeatureGate
            num_features = len(schema_resolver.dynamic_features)
            init_logits = None
            if self.jacobian_init and hasattr(self.yaib_runtime, "_model"):
                init_logits = self._compute_jacobian_sensitivity(num_features)
                logging.info("FJS: Jacobian-initialized feature gate enabled")
            elif _tc.get("lstm_informed_gate", False) and hasattr(self.yaib_runtime, "_model"):
                from ..core.lstm_importance import extract_lstm_feature_importance
                init_logits = extract_lstm_feature_importance(
                    self.yaib_runtime._model,
                    num_dynamic_features=num_features,
                    dynamic_feature_offset=0,
                )
                logging.info("LSTM-informed gate initialization enabled")
            self.feature_gate = FeatureGate(num_features, init_logits=init_logits).to(device)
            logging.info("Feature gate enabled for reconstruction weighting (%d features)", num_features)

        params = list(self.translator.parameters())
        if self.feature_gate is not None:
            params += list(self.feature_gate.parameters())
        self.optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.scaler = GradScaler(enabled=device.startswith("cuda"))

        # Snapshot frozen baseline parameters for post-training verification
        self._baseline_param_snapshot = {
            name: param.detach().clone()
            for name, param in self.yaib_runtime._model.named_parameters()
        }

        # Feature bounds for range loss
        self.lower_bounds, self.upper_bounds = self._load_feature_bounds(
            bounds_csv, schema_resolver.dynamic_features
        )
        self.lower_bounds = self.lower_bounds.to(device)
        self.upper_bounds = self.upper_bounds.to(device)

        self.best_val = float("inf")
        self.best_state = None
        self.history: list[dict] = []
        self.memory_bank = None

        # torch.compile: JIT compile translator sub-modules
        if _tc.get("use_torch_compile", False):
            logging.info("[torch.compile] Compiling translator sub-modules (mode=default, dynamic=True)")
            self.translator.encode = torch.compile(
                self.translator.encode, mode="default", dynamic=True
            )
            self.translator.decode_with_context = torch.compile(
                self.translator.decode_with_context, mode="default", dynamic=True
            )
            self.translator.decode = torch.compile(
                self.translator.decode, mode="default", dynamic=True
            )

    def _compute_jacobian_sensitivity(self, num_features: int) -> torch.Tensor:
        """FJS: Compute frozen model's per-feature input sensitivity via Jacobian.

        Returns init_logits (num_features,) suitable for FeatureGate initialization.
        Computes |∂loss/∂x_yaib| averaged over MIMIC batches, then extracts the
        dynamic feature columns to get per-translator-feature sensitivity.
        """
        from itertools import islice
        logging.info("FJS: Computing frozen model Jacobian sensitivity...")
        sensitivities = []
        model = self.yaib_runtime._model
        was_training = model.training

        for batch in islice(self.target_train_loader, 50):
            batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
            parts = self.schema_resolver.extract(batch)
            # Work directly on the YAIB-format tensor to avoid rebuild grad issues
            x_yaib = parts["X_yaib"].clone().detach().float().requires_grad_(True)
            label_mask = parts["M_label"].bool()
            with torch.enable_grad():
                model.train()  # cuDNN RNN backward requires train mode
                logits = self.yaib_runtime.forward((x_yaib, parts["y"], label_mask))
                loss = self.yaib_runtime.compute_loss(
                    logits.float(), (x_yaib, parts["y"], label_mask)
                ).float()
                grad = torch.autograd.grad(loss, x_yaib)[0]  # (B, T, F_yaib)
            mask = ~parts["M_pad"].bool()
            abs_grad = grad.abs().float()
            abs_grad[~mask] = 0.0
            n_valid = mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            # Per-YAIB-feature mean absolute gradient: (B, F) / (B, 1) broadcasts
            feat_sens = (abs_grad.sum(dim=1) / n_valid).mean(dim=0)  # (F_yaib,)
            # Extract only the dynamic feature columns (first num_features)
            sensitivities.append(feat_sens[:num_features].detach())

        if was_training:
            model.train()
        if not sensitivities:
            logging.warning("FJS: No valid batches, falling back to zero init")
            return None
        S = torch.stack(sensitivities).mean(0)  # (num_features,)
        S = S / S.max().clamp(min=1e-8)  # normalize to [0, 1]
        # Convert to logits: logit(S) = log(S / (1-S)), clamped for stability
        S_clamped = S.clamp(0.05, 0.95)
        init_logits = torch.log(S_clamped / (1.0 - S_clamped))
        logging.info("FJS: Sensitivity range [%.4f, %.4f], logit range [%.4f, %.4f]",
                      S.min().item(), S.max().item(), init_logits.min().item(), init_logits.max().item())
        return init_logits.cpu()

    def _label_pred_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type == "regression":
            return F.mse_loss(logits.float(), targets.float())
        return F.binary_cross_entropy_with_logits(logits.float(), targets.float())

    def _load_feature_bounds(self, bounds_csv, feature_names):
        import pandas as pd
        df = pd.read_csv(bounds_csv)
        df = df.set_index("feature")
        cols = set(df.columns)
        lower_col = next((c for c in ["p0.1_a", "p0.1", "p_001_a", "q001"] if c in cols), None)
        upper_col = next((c for c in ["p99.9_a", "p99.9", "p_999_a", "q999"] if c in cols), None)
        if lower_col is None or upper_col is None:
            raise ValueError(f"Bounds CSV missing percentile columns. Found: {sorted(cols)}")
        lower = torch.tensor(df.loc[feature_names, lower_col].to_numpy(), dtype=torch.float32)
        upper = torch.tensor(df.loc[feature_names, upper_col].to_numpy(), dtype=torch.float32)
        return lower, upper

    def _verify_baseline_frozen(self) -> None:
        model = self.yaib_runtime._model
        n_params = 0
        max_diff = 0.0
        for name, param in model.named_parameters():
            snapshot = self._baseline_param_snapshot[name]
            diff = (param.detach() - snapshot.to(param.device)).abs().max().item()
            max_diff = max(max_diff, diff)
            n_params += param.numel()
            if diff > 0:
                raise RuntimeError(
                    f"FROZEN LSTM CORRUPTED: parameter '{name}' changed by {diff:.2e} "
                    f"during training."
                )
        logging.info("[verify] Frozen LSTM integrity OK — %d parameters, max_diff=%.2e", n_params, max_diff)

    def _next_target_batch(self):
        try:
            batch = next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self.target_train_loader)
            batch = next(self._target_iter)
        return tuple(b.to(self.device) for b in batch)

    def set_renorm_params(self, scale: torch.Tensor, offset: torch.Tensor):
        self.renorm_scale = scale.to(self.device)
        self.renorm_offset = offset.to(self.device)
        logging.info("Cross-domain renormalization enabled for source data (retrieval)")

    def _apply_renorm(self, x_val: torch.Tensor, m_pad: torch.Tensor) -> torch.Tensor:
        if self.renorm_scale is None:
            return x_val
        x = x_val * self.renorm_scale.view(1, 1, -1) + self.renorm_offset.view(1, 1, -1)
        return x.masked_fill(m_pad.unsqueeze(-1).bool(), 0.0)

    def _clamp_to_bounds(self, x_out: torch.Tensor) -> torch.Tensor:
        """Hard-clamp translated features to target feature bounds.

        Prevents extreme translator outputs (e.g. from OOD source data
        like HiRID LoS) from causing NaN in the frozen LSTM.  The soft
        range loss alone cannot prevent occasional extreme values during
        early training.  For well-behaved experiments (translator outputs
        already within bounds) this is a differentiable no-op.
        """
        return x_out.clamp(
            min=self.lower_bounds.view(1, 1, -1),
            max=self.upper_bounds.view(1, 1, -1),
        )

    def _compute_importance_reg(self, importance_w) -> torch.Tensor:
        """Compute importance regularization on importance weights."""
        if importance_w is None:
            return torch.tensor(0.0, device=self.device)
        if self.importance_reg_type == "entropy":
            # Entropy reg: encourages decisive 0/1 without collapse
            per_dim_ent = -(
                importance_w * (importance_w + 1e-8).log()
                + (1 - importance_w) * (1 - importance_w + 1e-8).log()
            )
            return per_dim_ent.mean()
        else:
            # Default L1 for backward compat
            return importance_w.abs().mean()

    def _compute_scheduled_lambda_recon(self, epoch: int) -> float:
        """Compute lambda_recon for the current epoch based on fidelity schedule (AFS)."""
        if self.fidelity_schedule is None:
            return self._initial_lambda_recon
        start = self.fidelity_decay_start_epoch
        end = self.fidelity_decay_end_epoch
        min_val = self._initial_lambda_recon * self.fidelity_min_ratio
        if epoch < start:
            return self._initial_lambda_recon
        if epoch >= end:
            return min_val
        progress = (epoch - start) / max(end - start, 1)
        if self.fidelity_schedule == "linear_decay":
            return self._initial_lambda_recon + (min_val - self._initial_lambda_recon) * progress
        elif self.fidelity_schedule == "cosine_decay":
            import math
            return min_val + (self._initial_lambda_recon - min_val) * 0.5 * (1 + math.cos(math.pi * progress))
        return self._initial_lambda_recon

    def _build_memory_bank(self) -> None:
        """Build/rebuild the MIMIC memory bank using the current encoder."""
        import gc
        from ..core.retrieval_translator import build_memory_bank
        logging.info("Building memory bank (encoding all MIMIC data)...")
        # Free old memory bank before building new one to reduce peak CPU RAM.
        # For LoS (65K stays, 757K windows), the old bank holds ~2.5 GB of CPU
        # tensors that would otherwise coexist with the new bank during rebuild.
        if self.memory_bank is not None:
            self.memory_bank = None
            gc.collect()
        self.memory_bank = build_memory_bank(
            encoder=self.translator,
            target_loader=self.target_train_loader,
            schema_resolver=self.schema_resolver,
            device=self.device,
            window_size=self.retrieval_window,
            window_stride=self.window_stride,
            store_labels=(self.ccr_alpha > 0),
        )
        self.translator.train()  # restore train mode after build

    def _get_refresh_metric(self, val_metrics: dict) -> float:
        """Extract the tracked metric value from validation metrics."""
        if self.adaptive_refresh_metric == "both":
            return val_metrics.get("align", 0.0) + val_metrics.get("recon", 0.0)
        elif self.adaptive_refresh_metric == "align":
            return val_metrics.get("align", 0.0)
        else:  # "recon"
            return val_metrics.get("recon", 0.0)

    def _update_refresh_baseline(self, val_metrics: dict | None) -> None:
        """Record the metric value at the time of last refresh."""
        if val_metrics is None:
            self._metric_at_last_refresh = None
            return
        self._metric_at_last_refresh = self._get_refresh_metric(val_metrics)

    def _should_refresh_memory_bank(self, epoch: int, val_metrics: dict | None) -> tuple[bool, str]:
        """Decide whether to rebuild the memory bank.

        Returns: (should_refresh, reason)
        """
        if self.memory_bank is None:
            return True, "initial_build"

        # Fixed interval mode (backward compat)
        if not self.adaptive_refresh:
            if epoch % self.memory_refresh_epochs == 0:
                return True, "fixed_interval"
            return False, "fixed_not_due"

        # Adaptive mode: check validation metric improvement
        epochs_since = epoch - self._last_refresh_epoch
        if epochs_since < self.adaptive_refresh_patience:
            return False, "patience"

        if val_metrics is None or self._metric_at_last_refresh is None:
            return True, "no_baseline"

        # Lower is better for both align and recon
        current = self._get_refresh_metric(val_metrics)
        improvement = self._metric_at_last_refresh - current  # positive = improved
        if improvement > self.adaptive_refresh_min_delta:
            return True, f"improved_{self.adaptive_refresh_metric}"

        return False, "no_improvement"

    def _pretrain_epoch(self, target_loader: DataLoader) -> dict:
        """Autoencoder pretraining on MIMIC target data.

        When phase1_self_retrieval is enabled, builds a MIMIC memory bank from the
        encoder's own representations and uses it to provide real context to
        cross-attention blocks during pretraining (instead of zero tensors).
        """
        from ..core.retrieval_translator import query_memory_bank
        self.translator.train()
        total_recon = 0.0
        total_label_pred = 0.0
        n_batches = 0
        use_self_retrieval = self.phase1_self_retrieval and self.memory_bank is not None

        for batch in target_loader:
            batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
            parts = self.schema_resolver.extract(batch)

            with torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")):
                latent = self.translator.encode(
                    parts["X_val"], parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], parts["X_static"],
                )

                if use_self_retrieval:
                    # Self-retrieval: use the memory bank to provide real context
                    importance_w = self.translator.get_importance_weights()
                    context = query_memory_bank(
                        latent.detach(),  # safety: prevent backprop through k-NN
                        parts["M_pad"],
                        self.memory_bank,
                        k_neighbors=self.k_neighbors,
                        retrieval_window=self.retrieval_window,
                        importance_weights=importance_w.detach(),
                    )
                    x_out = self.translator.decode_with_context(
                        latent, context, parts["M_pad"], parts["X_static"],
                    )
                else:
                    # Standard zero-context decode (backward compat)
                    x_out = self.translator.decode(latent, parts["M_pad"], parts["X_static"])

                mask = ~parts["M_pad"].bool()
                diff = (x_out.float() - parts["X_val"].float()) ** 2
                l_recon = diff.sum(dim=-1)[mask].mean() if mask.any() else diff.new_tensor(0.0)

                l_label_pred = latent.new_tensor(0.0)
                if self.lambda_label_pred > 0:
                    label_logits = self.translator.predict_labels(latent, parts["M_pad"])
                    label_mask = parts["M_label"].bool()
                    if label_mask.any():
                        l_label_pred = self._label_pred_loss(
                            label_logits[label_mask],
                            parts["y"][label_mask],
                        )

                loss = l_recon + self.lambda_label_pred * l_label_pred

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_recon += l_recon.item()
            total_label_pred += l_label_pred.item()
            n_batches += 1

        return {
            "pretrain_recon": total_recon / max(n_batches, 1),
            "pretrain_label_pred": total_label_pred / max(n_batches, 1),
        }

    def _run_epoch(self, train_loader: DataLoader, epoch: int = 0) -> dict:
        """Phase 2 training: retrieval-guided translation."""
        from ..core.retrieval_translator import query_memory_bank
        self.translator.train()
        totals = {"total": 0.0, "task": 0.0, "align": 0.0, "recon": 0.0, "range": 0.0,
                  "smooth": 0.0, "importance_reg": 0.0, "target_task": 0.0,
                  "label_pred": 0.0, "contrastive": 0.0, "tcr": 0.0}
        n_batches = 0

        # AFS: update lambda_recon based on fidelity schedule
        self.lambda_recon = self._compute_scheduled_lambda_recon(epoch)

        for batch in train_loader:
            batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
            parts = self.schema_resolver.extract(batch)
            parts["X_val"] = self._apply_renorm(parts["X_val"], parts["M_pad"])

            _grad_diag_interval = getattr(self, "_grad_diag_interval", 1)
            _gc_enabled = getattr(self.translator, "gradient_checkpointing", False)
            _do_grad_diag = not _gc_enabled and (
                (epoch == 0 and n_batches <= 3) or
                (epoch > 0 and epoch % _grad_diag_interval == 0 and n_batches == 0)
            )

            with torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")):
                # ── Source (eICU) path with retrieval ──
                src_latent = self.translator.encode(
                    parts["X_val"], parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], parts["X_static"],
                )

                # CCR: compute soft label predictions for class-conditional retrieval
                _ccr_label_probs = None
                if self.ccr_alpha > 0 and self.memory_bank is not None:
                    with torch.no_grad():
                        _lp_logits = self.translator.predict_labels(src_latent.detach(), parts["M_pad"])
                        _ccr_label_probs = torch.sigmoid(_lp_logits.float())

                # Per-timestep retrieval from memory bank (skip if no cross-attention)
                if self.translator.n_cross_layers > 0:
                    importance_w = self.translator.get_importance_weights()
                    context = query_memory_bank(
                        src_latent.detach(),  # detach queries for retrieval (not for encoding)
                        parts["M_pad"],
                        self.memory_bank,
                        k_neighbors=self.k_neighbors,
                        retrieval_window=self.retrieval_window,
                        importance_weights=importance_w.detach(),
                        query_label_probs=_ccr_label_probs,
                        ccr_alpha=self.ccr_alpha,
                    )
                else:
                    B, T, D = src_latent.shape
                    context = torch.zeros(B, T, 1, D, device=src_latent.device)
                    importance_w = None  # no retrieval → skip importance reg

                # Forward with retrieved context (reuse src_latent to avoid double-encode)
                x_out, _ = self.translator.forward_with_retrieval(
                    parts["X_val"], parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], parts["X_static"],
                    context,
                    latent=src_latent,
                )
                if _do_grad_diag:
                    x_out.retain_grad()

                # Task loss via frozen LSTM
                # Hard-clamp to feature bounds before LSTM (prevents NaN on OOD source data)
                x_out_clamped = self._clamp_to_bounds(x_out.float())
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_out_clamped, parts["X_miss"], parts["X_static"],
                    m_pad=parts["M_pad"],
                )
                label_mask = parts["M_label"].bool()
                logits = self.yaib_runtime.forward(
                    (x_yaib_translated, parts["y"], label_mask)
                )
                l_task = self.yaib_runtime.compute_loss(
                    logits.float(), (x_yaib_translated, parts["y"], label_mask)
                ).float()

                # ── Target (MIMIC) path: reconstruction ──
                tgt_batch = self._next_target_batch()
                tgt_parts = self.schema_resolver.extract(tgt_batch)
                tgt_latent = self.translator.encode(
                    tgt_parts["X_val"], tgt_parts["X_miss"], tgt_parts["t_abs"],
                    tgt_parts["M_pad"], tgt_parts["X_static"],
                )
                tgt_out = self.translator.decode(tgt_latent, tgt_parts["M_pad"], tgt_parts["X_static"])
                tgt_mask = ~tgt_parts["M_pad"].bool()
                tgt_diff = (tgt_out.float() - tgt_parts["X_val"].float()) ** 2
                if self.feature_gate is not None:
                    gate = self.feature_gate()  # (F,)
                    tgt_diff = tgt_diff * gate.view(1, 1, -1)
                # Per-feature recon → per-timestep scalar
                tgt_recon_per_ts = tgt_diff.sum(dim=-1)  # (B, T)
                # Positive-weighted reconstruction: boost recon loss at positive timesteps
                if self.recon_positive_boost > 0 and self.task_type != "regression":
                    tgt_labels = tgt_parts["y"].float()
                    tgt_label_available = tgt_parts["M_label"].bool()
                    pos_mask = tgt_label_available & (tgt_labels > 0.5)
                    ts_weight = torch.ones_like(tgt_recon_per_ts)
                    ts_weight[pos_mask] = 1.0 + self.recon_positive_boost
                    tgt_recon_per_ts = tgt_recon_per_ts * ts_weight
                l_recon = tgt_recon_per_ts[tgt_mask].mean() if tgt_mask.any() else tgt_diff.new_tensor(0.0)

                # ── TCR: Target Context Reconstruction ──
                # Route MIMIC through full retrieval pipeline (cross-attention gets gradient)
                l_tcr = tgt_out.new_tensor(0.0)
                if self.lambda_tcr > 0 and self.memory_bank is not None:
                    tgt_context = query_memory_bank(
                        tgt_latent.detach(),
                        tgt_parts["M_pad"],
                        self.memory_bank,
                        k_neighbors=self.k_neighbors,
                        retrieval_window=self.retrieval_window,
                        importance_weights=importance_w.detach(),
                    )
                    tgt_out_full, _ = self.translator.forward_with_retrieval(
                        tgt_parts["X_val"], tgt_parts["X_miss"], tgt_parts["t_abs"],
                        tgt_parts["M_pad"], tgt_parts["X_static"],
                        tgt_context, latent=tgt_latent,
                    )
                    l_tcr = F.mse_loss(
                        tgt_out_full[tgt_mask].float(),
                        tgt_parts["X_val"][tgt_mask].float(),
                    )

                # Range loss on source output
                src_mask = ~parts["M_pad"].bool()
                upper = self.upper_bounds.view(1, 1, -1)
                lower = self.lower_bounds.view(1, 1, -1)
                over = torch.relu(x_out.float() - upper)
                under = torch.relu(lower - x_out.float())
                range_penalty = (over ** 2 + under ** 2).sum(dim=-1)
                l_range = range_penalty[src_mask].mean() if src_mask.any() else range_penalty.new_tensor(0.0)

                # Temporal smoothness: penalize jumps between consecutive timesteps
                if x_out.shape[1] > 1:
                    diffs = (x_out[:, 1:, :].float() - x_out[:, :-1, :].float()) ** 2
                    smooth_mask = src_mask[:, 1:] & src_mask[:, :-1]
                    l_smooth = diffs.sum(dim=-1)[smooth_mask].mean() if smooth_mask.any() else diffs.new_tensor(0.0)
                else:
                    l_smooth = x_out.new_tensor(0.0)

                # Importance regularization
                l_importance_reg = self._compute_importance_reg(importance_w)

                # Target task loss (optional)
                l_target_task = tgt_out.new_tensor(0.0)
                if self.lambda_target_task > 0:
                    tgt_yaib = self.schema_resolver.rebuild(
                        tgt_parts["X_yaib"], tgt_out.float(), tgt_parts["X_miss"], tgt_parts["X_static"],
                        m_pad=tgt_parts["M_pad"],
                    )
                    tgt_label_mask = tgt_parts["M_label"].bool()
                    tgt_logits = self.yaib_runtime.forward((tgt_yaib, tgt_parts["y"], tgt_label_mask))
                    l_target_task = self.yaib_runtime.compute_loss(
                        tgt_logits.float(), (tgt_yaib, tgt_parts["y"], tgt_label_mask)
                    ).float()

                # Latent MMD alignment (distributional domain alignment)
                l_align = src_latent.new_tensor(0.0)
                if self.lambda_align > 0:
                    src_mask_align = ~parts["M_pad"].bool()
                    tgt_mask_align = ~tgt_parts["M_pad"].bool()
                    src_z = src_latent[src_mask_align].float()
                    tgt_z = tgt_latent[tgt_mask_align].float().detach()
                    if src_z.shape[0] > 1 and tgt_z.shape[0] > 1:
                        l_align = multi_kernel_mmd(src_z, tgt_z)

                # Label prediction from latent (both domains)
                l_label_pred = src_latent.new_tensor(0.0)
                if self.lambda_label_pred > 0:
                    src_lp_logits = self.translator.predict_labels(src_latent, parts["M_pad"])
                    src_lp_mask = parts["M_label"].bool()
                    l_src_lp = (self._label_pred_loss(
                        src_lp_logits[src_lp_mask], parts["y"][src_lp_mask]
                    ) if src_lp_mask.any() else src_latent.new_tensor(0.0))

                    tgt_lp_logits = self.translator.predict_labels(tgt_latent, tgt_parts["M_pad"])
                    tgt_lp_mask = tgt_parts["M_label"].bool()
                    l_tgt_lp = (self._label_pred_loss(
                        tgt_lp_logits[tgt_lp_mask], tgt_parts["y"][tgt_lp_mask]
                    ) if tgt_lp_mask.any() else tgt_latent.new_tensor(0.0))

                    l_label_pred = (l_src_lp + l_tgt_lp) / 2.0

                # Cross-domain contrastive alignment (optional)
                l_contrastive = src_latent.new_tensor(0.0)
                if self.lambda_contrastive_align > 0:
                    src_valid = ~parts["M_pad"].bool() & parts["M_label"].bool()
                    tgt_valid = ~tgt_parts["M_pad"].bool() & tgt_parts["M_label"].bool()
                    src_lats = src_latent[src_valid].float()
                    src_labs = parts["y"][src_valid]
                    tgt_lats = tgt_latent[tgt_valid].float()
                    tgt_labs = tgt_parts["y"][tgt_valid]
                    n_terms = 0
                    src_pos = src_lats[src_labs > 0.5]
                    src_neg = src_lats[src_labs < 0.5]
                    tgt_pos = tgt_lats[tgt_labs > 0.5]
                    tgt_neg = tgt_lats[tgt_labs < 0.5]
                    if src_pos.shape[0] > 0 and tgt_pos.shape[0] > 0:
                        l_contrastive = l_contrastive + F.mse_loss(src_pos.mean(0), tgt_pos.mean(0))
                        n_terms += 1
                    if src_neg.shape[0] > 0 and tgt_neg.shape[0] > 0:
                        l_contrastive = l_contrastive + F.mse_loss(src_neg.mean(0), tgt_neg.mean(0))
                        n_terms += 1
                    if n_terms > 0:
                        l_contrastive = l_contrastive / n_terms

                # Guard against NaN task loss (e.g. LSTM overflow on
                # out-of-distribution source data like HiRID LoS).  Replace
                # NaN with 0 so the remaining loss components can still drive
                # gradient updates.
                if torch.isnan(l_task):
                    if n_batches == 0 and epoch == 0:
                        logging.warning(
                            "l_task is NaN — replacing with 0.0 for this batch "
                            "(frozen LSTM may overflow on translated source data)"
                        )
                    l_task = l_task.new_tensor(0.0)

                l_total = (
                    l_task
                    + self.lambda_align * l_align
                    + self.lambda_recon * l_recon
                    + self.lambda_range * l_range
                    + self.lambda_smooth * l_smooth
                    + self.lambda_importance_reg * l_importance_reg
                    + self.lambda_target_task * l_target_task
                    + self.lambda_label_pred * l_label_pred
                    + self.lambda_contrastive_align * l_contrastive
                    + self.lambda_tcr * l_tcr
                )

            if _do_grad_diag:
                _gradient_diagnostic(
                    self.optimizer, self.translator, x_out,
                    l_task, self.lambda_recon * l_recon, self.lambda_range * l_range,
                    parts, epoch, n_batches, reg_label="recon",
                )

            if n_batches % self.accumulate_grad_batches == 0:
                self.optimizer.zero_grad()
            self.scaler.scale(l_total / self.accumulate_grad_batches).backward()
            if (n_batches + 1) % self.accumulate_grad_batches == 0:
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.optimizer.param_groups for p in g["params"]],
                        self.grad_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            totals["total"] += l_total.item()
            totals["task"] += l_task.item()
            totals["align"] += l_align.item()
            totals["recon"] += l_recon.item()
            totals["range"] += l_range.item()
            totals["smooth"] += l_smooth.item()
            totals["importance_reg"] += l_importance_reg.item()
            totals["target_task"] += l_target_task.item()
            totals["label_pred"] += l_label_pred.item()
            totals["contrastive"] += l_contrastive.item()
            totals["tcr"] += l_tcr.item()
            n_batches += 1

        # Flush remaining accumulated gradients
        if self.accumulate_grad_batches > 1 and n_batches % self.accumulate_grad_batches != 0:
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    self.grad_clip_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> dict:
        """Validation pass with retrieval."""
        from ..core.retrieval_translator import query_memory_bank
        self.translator.eval()
        totals = {"total": 0.0, "task": 0.0, "align": 0.0, "recon": 0.0, "range": 0.0,
                  "smooth": 0.0, "importance_reg": 0.0, "target_task": 0.0,
                  "label_pred": 0.0, "contrastive": 0.0, "tcr": 0.0}
        n_batches = 0

        importance_w = self.translator.get_importance_weights()

        for batch in val_loader:
            batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
            parts = self.schema_resolver.extract(batch)
            parts["X_val"] = self._apply_renorm(parts["X_val"], parts["M_pad"])

            with torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")):
                src_latent = self.translator.encode(
                    parts["X_val"], parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], parts["X_static"],
                )

                # CCR: compute soft label predictions for class-conditional retrieval
                _ccr_label_probs = None
                if self.ccr_alpha > 0 and self.memory_bank is not None:
                    _lp_logits = self.translator.predict_labels(src_latent, parts["M_pad"])
                    _ccr_label_probs = torch.sigmoid(_lp_logits.float())

                if self.translator.n_cross_layers > 0:
                    context = query_memory_bank(
                        src_latent,
                        parts["M_pad"],
                        self.memory_bank,
                        k_neighbors=self.k_neighbors,
                        retrieval_window=self.retrieval_window,
                        importance_weights=importance_w,
                        query_label_probs=_ccr_label_probs,
                        ccr_alpha=self.ccr_alpha,
                    )
                else:
                    B, T, D = src_latent.shape
                    context = torch.zeros(B, T, 1, D, device=src_latent.device)

                x_out, _ = self.translator.forward_with_retrieval(
                    parts["X_val"], parts["X_miss"], parts["t_abs"],
                    parts["M_pad"], parts["X_static"],
                    context,
                    latent=src_latent,
                )

                # Hard-clamp before LSTM (validation path)
                x_out_clamped = self._clamp_to_bounds(x_out.float())
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_out_clamped, parts["X_miss"], parts["X_static"],
                    m_pad=parts["M_pad"],
                )
                label_mask = parts["M_label"].bool()
                logits = self.yaib_runtime.forward(
                    (x_yaib_translated, parts["y"], label_mask)
                )
                l_task = self.yaib_runtime.compute_loss(
                    logits.float(), (x_yaib_translated, parts["y"], label_mask)
                ).float()

                # Target reconstruction
                tgt_batch = self._next_target_batch()
                tgt_parts = self.schema_resolver.extract(tgt_batch)
                tgt_latent = self.translator.encode(
                    tgt_parts["X_val"], tgt_parts["X_miss"], tgt_parts["t_abs"],
                    tgt_parts["M_pad"], tgt_parts["X_static"],
                )
                tgt_out = self.translator.decode(tgt_latent, tgt_parts["M_pad"], tgt_parts["X_static"])
                tgt_mask = ~tgt_parts["M_pad"].bool()
                tgt_diff = (tgt_out.float() - tgt_parts["X_val"].float()) ** 2
                if self.feature_gate is not None:
                    gate = self.feature_gate()  # (F,)
                    tgt_diff = tgt_diff * gate.view(1, 1, -1)
                l_recon = tgt_diff.sum(dim=-1)[tgt_mask].mean() if tgt_mask.any() else tgt_diff.new_tensor(0.0)

                src_mask = ~parts["M_pad"].bool()
                upper = self.upper_bounds.view(1, 1, -1)
                lower = self.lower_bounds.view(1, 1, -1)
                over = torch.relu(x_out.float() - upper)
                under = torch.relu(lower - x_out.float())
                range_penalty = (over ** 2 + under ** 2).sum(dim=-1)
                l_range = range_penalty[src_mask].mean() if src_mask.any() else range_penalty.new_tensor(0.0)

                if x_out.shape[1] > 1:
                    diffs = (x_out[:, 1:, :].float() - x_out[:, :-1, :].float()) ** 2
                    smooth_mask = src_mask[:, 1:] & src_mask[:, :-1]
                    l_smooth = diffs.sum(dim=-1)[smooth_mask].mean() if smooth_mask.any() else diffs.new_tensor(0.0)
                else:
                    l_smooth = x_out.new_tensor(0.0)

                l_importance_reg = self._compute_importance_reg(importance_w)

                l_target_task = tgt_out.new_tensor(0.0)
                if self.lambda_target_task > 0:
                    tgt_yaib = self.schema_resolver.rebuild(
                        tgt_parts["X_yaib"], tgt_out.float(), tgt_parts["X_miss"], tgt_parts["X_static"],
                        m_pad=tgt_parts["M_pad"],
                    )
                    tgt_label_mask = tgt_parts["M_label"].bool()
                    tgt_logits = self.yaib_runtime.forward((tgt_yaib, tgt_parts["y"], tgt_label_mask))
                    l_target_task = self.yaib_runtime.compute_loss(
                        tgt_logits.float(), (tgt_yaib, tgt_parts["y"], tgt_label_mask)
                    ).float()

                # Latent MMD alignment (validation)
                l_align = src_latent.new_tensor(0.0)
                if self.lambda_align > 0:
                    src_mask_align = ~parts["M_pad"].bool()
                    tgt_mask_align = ~tgt_parts["M_pad"].bool()
                    src_z = src_latent[src_mask_align].float()
                    tgt_z = tgt_latent[tgt_mask_align].float()
                    if src_z.shape[0] > 1 and tgt_z.shape[0] > 1:
                        l_align = multi_kernel_mmd(src_z, tgt_z)

                # Label prediction from latent (validation)
                l_label_pred = src_latent.new_tensor(0.0)
                if self.lambda_label_pred > 0:
                    src_lp_logits = self.translator.predict_labels(src_latent, parts["M_pad"])
                    src_lp_mask = parts["M_label"].bool()
                    l_src_lp = (self._label_pred_loss(
                        src_lp_logits[src_lp_mask], parts["y"][src_lp_mask]
                    ) if src_lp_mask.any() else src_latent.new_tensor(0.0))

                    tgt_lp_logits = self.translator.predict_labels(tgt_latent, tgt_parts["M_pad"])
                    tgt_lp_mask = tgt_parts["M_label"].bool()
                    l_tgt_lp = (self._label_pred_loss(
                        tgt_lp_logits[tgt_lp_mask], tgt_parts["y"][tgt_lp_mask]
                    ) if tgt_lp_mask.any() else tgt_latent.new_tensor(0.0))

                    l_label_pred = (l_src_lp + l_tgt_lp) / 2.0

                # Replace NaN task loss with 0 in total (so val_total stays
                # valid for checkpoint selection fallback), but keep the raw
                # NaN in the task bucket so logging reflects the actual issue.
                l_task_safe = l_task if not torch.isnan(l_task) else l_task.new_tensor(0.0)

                l_total = (
                    l_task_safe
                    + self.lambda_align * l_align
                    + self.lambda_recon * l_recon
                    + self.lambda_range * l_range
                    + self.lambda_smooth * l_smooth
                    + self.lambda_importance_reg * l_importance_reg
                    + self.lambda_target_task * l_target_task
                    + self.lambda_label_pred * l_label_pred
                )

            totals["total"] += l_total.item()
            totals["task"] += l_task.item()
            totals["align"] += l_align.item()
            totals["recon"] += l_recon.item()
            totals["range"] += l_range.item()
            totals["smooth"] += l_smooth.item()
            totals["importance_reg"] += l_importance_reg.item()
            totals["target_task"] += l_target_task.item()
            totals["label_pred"] += l_label_pred.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def _log_retrieval_metrics(self, epoch: int) -> None:
        """Log retrieval quality metrics."""
        if self.memory_bank is None:
            return
        bank = self.memory_bank
        logging.info(
            "[retrieval] Epoch %d — bank: %d windows, mean_nn_dist: computed per-batch during training",
            epoch, bank.window_latents.shape[0],
        )
        # Log importance weights
        iw = self.translator.get_importance_weights().detach().cpu()
        top_vals, top_idx = iw.topk(min(10, iw.shape[0]))
        entropy = -(iw * (iw + 1e-8).log() + (1 - iw) * (1 - iw + 1e-8).log()).sum().item()
        logging.info(
            "[importance] Epoch %d — entropy=%.4f, top10_dims=%s, top10_weights=%s",
            epoch,
            entropy,
            top_idx.tolist(),
            [f"{v:.3f}" for v in top_vals.tolist()],
        )

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # Phase 1: Autoencoder pretraining on MIMIC
        pretrain_path = self.run_dir / "pretrain_checkpoint.pt"
        if pretrain_path.exists():
            ckpt = torch.load(pretrain_path, map_location=self.device, weights_only=True)
            self.translator.load_state_dict(ckpt["translator_state_dict"])
            logging.info("Loaded pretrain checkpoint from %s — skipping Phase 1", pretrain_path)
        elif self.pretrain_epochs > 0:
            sr_label = " with self-retrieval" if self.phase1_self_retrieval else ""
            logging.info("=== Phase 1: Autoencoder pretraining on MIMIC (%d epochs%s) ===",
                         self.pretrain_epochs, sr_label)
            for ep in range(self.pretrain_epochs):
                # Build/refresh memory bank for self-retrieval
                if self.phase1_self_retrieval:
                    if self.memory_bank is None or (ep % self.phase1_memory_refresh_epochs == 0):
                        self._build_memory_bank()

                metrics = self._pretrain_epoch(self.target_train_loader)
                lp_str = ""
                if self.lambda_label_pred > 0:
                    lp_str = " label_pred=%.4f" % metrics.get("pretrain_label_pred", 0.0)
                logging.info(
                    "Pretrain epoch %d/%d - recon=%.4f%s",
                    ep + 1, self.pretrain_epochs, metrics["pretrain_recon"], lp_str,
                )

            # Force Phase 2 to build a fresh memory bank (encoder changed during pretrain)
            self.memory_bank = None

            torch.save({"translator_state_dict": self.translator.state_dict()}, pretrain_path)
            logging.info("Saved pretrain checkpoint to %s", pretrain_path)

            # Reset optimizer after pretraining
            params = list(self.translator.parameters())
            if self.feature_gate is not None:
                params += list(self.feature_gate.parameters())
            self.optimizer = AdamW(
                params,
                lr=self.optimizer.defaults["lr"],
                weight_decay=self.optimizer.defaults["weight_decay"],
            )
            self.scaler = GradScaler(enabled=self.device.startswith("cuda"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("=== Phase 1 complete. Starting Phase 2: Retrieval-guided training ===")

        # Phase 2: Retrieval-guided training
        logging.info("=== Phase 2: Retrieval-guided training (%d epochs) ===", epochs)
        self._grad_diag_interval = max(1, epochs // 4) if epochs > 1 else 1
        logging.info("Gradient diagnostics interval: every %d epochs", self._grad_diag_interval)
        epochs_without_improvement = 0
        start_epoch = 0

        # Create LR scheduler for Phase 2
        self.scheduler = _create_lr_scheduler(
            self.optimizer, self.lr_scheduler_type, epochs,
            lr_min=self.lr_min, warmup_epochs=self.lr_warmup_epochs,
        )

        # --- Resume from checkpoint ---
        resume_path = self.run_dir / "latest_checkpoint.pt"
        if resume_path.exists():
            rk = torch.load(resume_path, map_location=self.device, weights_only=False)
            self.translator.load_state_dict(rk["translator_state_dict"])
            self.optimizer.load_state_dict(rk["optimizer_state_dict"])
            self.scaler.load_state_dict(rk["scaler_state_dict"])
            self.best_val = rk["best_val"]
            self.best_state = rk.get("best_state")
            self.history = rk.get("history", [])
            epochs_without_improvement = rk.get("epochs_without_improvement", 0)
            start_epoch = rk["epoch"] + 1
            if self.feature_gate is not None and "feature_gate_state_dict" in rk:
                self.feature_gate.load_state_dict(rk["feature_gate_state_dict"])
            if rk.get("renorm_scale") is not None:
                self.renorm_scale = rk["renorm_scale"]
                self.renorm_offset = rk["renorm_offset"]
            if self.scheduler is not None and "scheduler_state_dict" in rk:
                self.scheduler.load_state_dict(rk["scheduler_state_dict"])
            ar_state = rk.get("adaptive_refresh_state")
            if ar_state:
                self._last_refresh_epoch = ar_state["last_refresh_epoch"]
                self._metric_at_last_refresh = ar_state.get("metric_at_last_refresh")
                self._refresh_history = ar_state.get("refresh_history", [])
            logging.info(
                "Resumed from checkpoint epoch %d (best_val=%.6f, no_improve=%d)",
                start_epoch, self.best_val, epochs_without_improvement,
            )

        # Initial memory bank build (before loop) — skip if no cross-attention (C1 ablation)
        if self.memory_bank is None and self.translator.n_cross_layers > 0:
            self._build_memory_bank()
            self._last_refresh_epoch = start_epoch
            self._update_refresh_baseline(None)
            self._refresh_history.append({"epoch": start_epoch, "reason": "initial_build"})

        import time as _time
        for epoch in range(start_epoch, epochs):
            t_epoch = _time.time()
            train_metrics = self._run_epoch(train_loader, epoch=epoch)
            val_metrics = self._validate(val_loader)

            epoch_time = _time.time() - t_epoch
            cur_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("[timing] Epoch %d/%d: %.1fs", epoch + 1, epochs, epoch_time)
            logging.info(
                "Epoch %d/%d - train: total=%.4f task=%.4f align=%.4f recon=%.4f range=%.4f smooth=%.4f imp_reg=%.4f target_task=%.4f label_pred=%.4f (lr=%.2e)",
                epoch + 1, epochs, train_metrics["total"], train_metrics["task"],
                train_metrics.get("align", 0.0), train_metrics["recon"], train_metrics["range"],
                train_metrics["smooth"], train_metrics["importance_reg"],
                train_metrics.get("target_task", 0.0), train_metrics.get("label_pred", 0.0),
                cur_lr,
            )
            logging.info(
                "Epoch %d/%d - val: total=%.4f task=%.4f align=%.4f recon=%.4f range=%.4f smooth=%.4f imp_reg=%.4f target_task=%.4f label_pred=%.4f",
                epoch + 1, epochs, val_metrics["total"], val_metrics["task"],
                val_metrics.get("align", 0.0), val_metrics["recon"], val_metrics["range"],
                val_metrics["smooth"], val_metrics["importance_reg"],
                val_metrics.get("target_task", 0.0), val_metrics.get("label_pred", 0.0),
            )

            self._log_retrieval_metrics(epoch + 1)

            self.history.append({
                "epoch": epoch + 1,
                "lr": cur_lr,
                "bank_last_refresh_epoch": self._last_refresh_epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            # Step LR scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["task"])
                else:
                    self.scheduler.step()

            # Adaptive / fixed memory bank refresh (AFTER validation) — skip if no cross-attention
            should_refresh, refresh_reason = self._should_refresh_memory_bank(epoch, val_metrics)
            if should_refresh and self.translator.n_cross_layers > 0:
                self._build_memory_bank()
                self._last_refresh_epoch = epoch
                self._update_refresh_baseline(val_metrics)
                self._refresh_history.append({
                    "epoch": epoch, "reason": refresh_reason,
                    "val_align": val_metrics.get("align", 0.0),
                    "val_recon": val_metrics.get("recon", 0.0),
                })
                logging.info(
                    "[adaptive-refresh] Epoch %d: rebuilt (reason=%s, val_align=%.6f, val_recon=%.6f)",
                    epoch, refresh_reason, val_metrics.get("align", 0.0), val_metrics.get("recon", 0.0),
                )
            else:
                logging.info(
                    "[adaptive-refresh] Epoch %d: skipped (reason=%s, val_align=%.6f, val_recon=%.6f)",
                    epoch, refresh_reason, val_metrics.get("align", 0.0), val_metrics.get("recon", 0.0),
                )

            candidate = val_metrics["task"] if self.best_metric == "val_task" else val_metrics["total"]
            # Fallback: when val_task is NaN (e.g. LSTM overflow on HiRID
            # source data), use val_target_task for checkpoint selection.
            import math
            if math.isnan(candidate):
                candidate = val_metrics.get("target_task", float("inf"))
                if epoch == 0:
                    logging.warning(
                        "val_task is NaN — falling back to val_target_task (%.6f) "
                        "for best checkpoint selection", candidate,
                    )
            if candidate < self.best_val:
                self.best_val = candidate
                self.best_state = self.translator.state_dict()
                ckpt_data = {
                    "epoch": epoch,
                    "translator_state_dict": self.best_state,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "renorm_scale": self.renorm_scale,
                    "renorm_offset": self.renorm_offset,
                    "bank_last_refresh_epoch": self._last_refresh_epoch,
                }
                if self.feature_gate is not None:
                    ckpt_data["feature_gate_state_dict"] = self.feature_gate.state_dict()
                torch.save(ckpt_data, self.run_dir / "best_translator.pt")
                logging.info("Saved new best checkpoint to %s", self.run_dir / "best_translator.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save resume checkpoint every epoch
            resume_ckpt = {
                "epoch": epoch,
                "translator_state_dict": self.translator.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_val": self.best_val,
                "best_state": self.best_state,
                "history": self.history,
                "epochs_without_improvement": epochs_without_improvement,
                "renorm_scale": self.renorm_scale,
                "renorm_offset": self.renorm_offset,
            }
            if self.feature_gate is not None:
                resume_ckpt["feature_gate_state_dict"] = self.feature_gate.state_dict()
            if self.scheduler is not None:
                resume_ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
            resume_ckpt["adaptive_refresh_state"] = {
                "last_refresh_epoch": self._last_refresh_epoch,
                "metric_at_last_refresh": self._metric_at_last_refresh,
                "refresh_history": self._refresh_history,
            }
            torch.save(resume_ckpt, resume_path)

            if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
                logging.info("Early stopping after %d epochs without improvement", epochs_without_improvement)
                break

        if self.best_state is not None:
            self.translator.load_state_dict(self.best_state)

        # Clean exit — remove resume checkpoint
        if resume_path.exists():
            resume_path.unlink()
            logging.info("Training completed — removed resume checkpoint")

        # Log adaptive refresh summary
        if self._refresh_history:
            total = len(self._refresh_history)
            reasons = [r["reason"] for r in self._refresh_history]
            adaptive_count = sum(1 for r in reasons if r.startswith("improved_"))
            logging.info(
                "[adaptive-refresh] Summary: %d total refreshes (%d initial, %d adaptive, %d fixed/other). "
                "Epochs without refresh: %d/%d",
                total, reasons.count("initial_build"), adaptive_count,
                total - reasons.count("initial_build") - adaptive_count,
                epochs - total, epochs,
            )

        self._verify_baseline_frozen()
        self._plot_losses()
        logging.info("Retrieval translator training completed")

    def _plot_losses(self) -> None:
        if not self.history:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in self.history]
        for key in ["total", "task", "recon"]:
            train_vals = [row.get(f"train_{key}", 0) for row in self.history]
            val_vals = [row.get(f"val_{key}", 0) for row in self.history]
            if any(v > 0 for v in train_vals + val_vals):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(epochs, train_vals, label=f"train_{key}")
                ax.plot(epochs, val_vals, label=f"val_{key}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(f"{key.title()} Loss")
                ax.legend()
                fig.tight_layout()
                fig.savefig(self.run_dir / f"{key}_loss_curve.png", dpi=150)
                plt.close(fig)
