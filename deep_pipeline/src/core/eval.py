import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import numpy as np

from ..adapters.yaib import YAIBRuntime
from ..core.translator import Translator
from ..core.schema import SchemaResolver
from ..core.io_parquet import reconstruct_parquet_from_batches, write_translated_parquet


def _save_predictions(probs: np.ndarray, targets: np.ndarray, output_path: Path, suffix: str = "") -> None:
    """Save predictions to .predictions.npz alongside the output parquet."""
    stem = output_path.stem
    fname = f"{stem}{suffix}.predictions.npz"
    npz_path = output_path.with_name(fname)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, probs=probs, targets=targets)
    logging.info("Saved predictions (%d samples) to %s", len(probs), npz_path)


def _compute_calibration_metrics(targets: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Compute Brier score and Expected Calibration Error (ECE)."""
    from sklearn.metrics import brier_score_loss

    brier = brier_score_loss(targets, probs)

    # ECE: bin predictions, compare mean predicted vs observed fraction of positives
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:  # last bin includes right edge
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = targets[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    ece /= len(probs)

    return {"brier": float(brier), "ece": float(ece)}


def _compute_regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: MAE, MSE, RMSE, R2."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(mse ** 0.5),
        "R2": float(r2_score(targets, predictions)),
    }


def load_feature_bounds(
    bounds_csv: Path, feature_names: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-feature lower/upper bounds from a CSV file.

    Standalone version of the trainer's ``_load_feature_bounds`` so that
    the evaluator can clamp translated outputs without a trainer instance.
    """
    df = pd.read_csv(bounds_csv)
    if "feature" not in df.columns:
        raise ValueError("Bounds CSV must include a 'feature' column.")
    columns = set(df.columns)

    lower_candidates = ["p0.1", "p_001", "q001", "q01", "q05"]
    upper_candidates = ["p99.9", "p_999", "q999", "q99", "q95"]

    def _pick(base: str) -> Optional[str]:
        if f"{base}_a" in columns:
            return f"{base}_a"
        if f"{base}_b" in columns:
            return f"{base}_b"
        if base in columns:
            return base
        return None

    lower_col = None
    for cand in lower_candidates:
        lower_col = _pick(cand)
        if lower_col:
            break
    upper_col = None
    for cand in upper_candidates:
        upper_col = _pick(cand)
        if upper_col:
            break
    if lower_col is None or upper_col is None:
        raise ValueError(
            f"Bounds CSV missing required percentile columns. Found columns: {sorted(columns)}"
        )

    df = df.set_index("feature")
    missing = [name for name in feature_names if name not in df.index]
    if missing:
        raise ValueError(f"Bounds CSV missing features: {missing}")
    lower = torch.tensor(df.loc[feature_names, lower_col].to_numpy(), dtype=torch.float32)
    upper = torch.tensor(df.loc[feature_names, upper_col].to_numpy(), dtype=torch.float32)
    return lower, upper


class TranslatorEvaluator:
    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: Translator,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        task_type: str = "classification",
    ):
        self.yaib_runtime = yaib_runtime
        self.translator = translator.to(device)
        self.device = device
        self.task_type = task_type
    
    def translate_and_evaluate(
        self,
        test_loader: DataLoader,
        output_parquet_path: Optional[Path] = None,
    ) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
        self.translator.eval()

        all_probs = []
        all_targets = []
        all_batches = []
        total_loss = 0.0
        num_batches = 0
        translated_batches = []
        stay_id_batches = []

        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(b.to(self.device) for b in batch)
                translated_data = self.translator(batch)
                baseline_outputs = self.yaib_runtime.forward((translated_data, batch[1], batch[2]))

                mask = batch[2].to(baseline_outputs.device).bool()
                prediction = torch.masked_select(
                    baseline_outputs, mask.unsqueeze(-1)
                ).reshape(-1, baseline_outputs.shape[-1])
                target = torch.masked_select(batch[1].to(baseline_outputs.device), mask)

                if self.task_type == "regression":
                    raw_pred = prediction[:, 0] if prediction.shape[-1] >= 1 else prediction.squeeze(-1)
                    all_probs.append(raw_pred.detach().cpu())
                else:
                    if baseline_outputs.shape[-1] > 1:
                        prediction_proba = torch.softmax(prediction, dim=-1)[:, 1]
                    else:
                        prediction_proba = torch.sigmoid(prediction).squeeze(-1)
                    all_probs.append(prediction_proba.detach().cpu())

                all_targets.append(target.detach().cpu())
                total_loss += self.yaib_runtime.compute_loss(
                    baseline_outputs, (translated_data, batch[1], batch[2])
                ).item()
                num_batches += 1
                all_batches.append(batch)
                translated_batches.append(translated_data)
                stay_id_batches.append(None)

        probs_np, targets_np = None, None
        if not all_probs:
            avg_metrics = {"MAE": float("inf"), "MSE": float("inf"), "loss": float("inf")} if self.task_type == "regression" else {"AUCROC": 0.0, "AUCPR": 0.0, "loss": float("inf")}
        else:
            probs_np = torch.cat(all_probs).numpy()
            targets_np = torch.cat(all_targets).numpy()

            if self.task_type == "regression":
                avg_metrics = _compute_regression_metrics(targets_np, probs_np)
                avg_metrics["loss"] = total_loss / num_batches if num_batches > 0 else float("inf")
            else:
                from sklearn.metrics import roc_auc_score, average_precision_score

                try:
                    auroc = roc_auc_score(targets_np, probs_np)
                except ValueError:
                    auroc = 0.0
                try:
                    auprc = average_precision_score(targets_np, probs_np)
                except ValueError:
                    auprc = 0.0

                avg_metrics = {
                    "AUCROC": auroc,
                    "AUCPR": auprc,
                    "loss": total_loss / num_batches if num_batches > 0 else float("inf"),
                }
                try:
                    cal = _compute_calibration_metrics(targets_np, probs_np)
                    avg_metrics.update(cal)
                except Exception as e:
                    logging.warning("Calibration metrics failed: %s", e)

        if output_parquet_path:
            self.export_translated_parquet(
                all_batches,
                translated_batches,
                stay_id_batches,
                output_parquet_path,
            )

        return avg_metrics, probs_np, targets_np
    
    def export_translated_parquet(
        self,
        batches: List[Tuple[torch.Tensor, ...]],
        translated_batches: List[torch.Tensor],
        stay_id_batches: List[Optional[torch.Tensor]],
        output_path: Path,
    ):
        if self.yaib_runtime._data is None:
            self.yaib_runtime.load_data()
        
        from icu_benchmarks.data.constants import DataSplit, DataSegment
        feature_names = self.yaib_runtime._data[DataSplit.train][DataSegment.features].columns
        feature_names = [
            col for col in feature_names
            if col != self.yaib_runtime.vars["GROUP"]
            and col != self.yaib_runtime.vars.get("SEQUENCE", "")
        ]
        
        translated_df = reconstruct_parquet_from_batches(
            batches,
            translated_batches,
            stay_id_batches,
            self.yaib_runtime.vars,
            feature_names,
            None
        )
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        translated_df.write_parquet(output_path)
        logging.info(f"Exported translated parquet to {output_path}")
    
    def evaluate_original_vs_translated(
        self,
        test_loader: DataLoader,
        output_parquet_path: Path,
    ) -> Dict[str, Dict[str, float]]:
        logging.info("Evaluating original test data...")
        original_metrics, orig_probs, orig_targets = self._evaluate_without_translator(test_loader)

        logging.info("Evaluating translated test data...")
        translated_metrics, trans_probs, trans_targets = self.translate_and_evaluate(test_loader, output_parquet_path)

        if output_parquet_path:
            out_path = Path(output_parquet_path)
            if trans_probs is not None:
                _save_predictions(trans_probs, trans_targets, out_path)
            if orig_probs is not None:
                _save_predictions(orig_probs, orig_targets, out_path, suffix=".original")

        return {
            "original": original_metrics,
            "translated": translated_metrics,
        }
    
    def _evaluate_without_translator(self, test_loader: DataLoader) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
        all_probs = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(b.to(self.device) for b in batch)
                baseline_outputs = self.yaib_runtime.forward(batch)
                mask = batch[2].to(baseline_outputs.device).bool()
                prediction = torch.masked_select(
                    baseline_outputs, mask.unsqueeze(-1)
                ).reshape(-1, baseline_outputs.shape[-1])
                target = torch.masked_select(batch[1].to(baseline_outputs.device), mask)

                if self.task_type == "regression":
                    raw_pred = prediction[:, 0] if prediction.shape[-1] >= 1 else prediction.squeeze(-1)
                    all_probs.append(raw_pred.detach().cpu())
                else:
                    if baseline_outputs.shape[-1] > 1:
                        prediction_proba = torch.softmax(prediction, dim=-1)[:, 1]
                    else:
                        prediction_proba = torch.sigmoid(prediction).squeeze(-1)
                    all_probs.append(prediction_proba.detach().cpu())

                all_targets.append(target.detach().cpu())
                total_loss += self.yaib_runtime.compute_loss(baseline_outputs, batch).item()
                num_batches += 1

        if not all_probs:
            empty = {"MAE": float("inf"), "MSE": float("inf"), "loss": float("inf")} if self.task_type == "regression" else {"AUCROC": 0.0, "AUCPR": 0.0, "loss": float("inf")}
            return empty, None, None

        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        if self.task_type == "regression":
            metrics = _compute_regression_metrics(targets, probs)
            metrics["loss"] = total_loss / num_batches if num_batches > 0 else float("inf")
        else:
            from sklearn.metrics import roc_auc_score, average_precision_score

            try:
                auroc = roc_auc_score(targets, probs)
            except ValueError:
                auroc = 0.0
            try:
                auprc = average_precision_score(targets, probs)
            except ValueError:
                auprc = 0.0

            metrics = {
                "AUCROC": auroc,
                "AUCPR": auprc,
                "loss": total_loss / num_batches if num_batches > 0 else float("inf"),
            }
        try:
            cal = _compute_calibration_metrics(targets, probs)
            metrics.update(cal)
        except Exception as e:
            logging.warning("Calibration metrics failed: %s", e)
        return metrics, probs, targets


class TransformerTranslatorEvaluator:
    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: torch.nn.Module,
        schema_resolver: SchemaResolver,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        renorm_scale: Optional[torch.Tensor] = None,
        renorm_offset: Optional[torch.Tensor] = None,
        task_type: str = "classification",
        lower_bounds: Optional[torch.Tensor] = None,
        upper_bounds: Optional[torch.Tensor] = None,
    ):
        self.yaib_runtime = yaib_runtime
        self.translator = translator.to(device)
        self.schema_resolver = schema_resolver
        self.device = device
        self.renorm_scale = renorm_scale.to(device) if renorm_scale is not None else None
        self.renorm_offset = renorm_offset.to(device) if renorm_offset is not None else None
        self.task_type = task_type
        self.lower_bounds = lower_bounds.to(device) if lower_bounds is not None else None
        self.upper_bounds = upper_bounds.to(device) if upper_bounds is not None else None

    def _apply_renorm(self, x_val: torch.Tensor, m_pad: torch.Tensor) -> torch.Tensor:
        if self.renorm_scale is None:
            return x_val
        x = x_val * self.renorm_scale.view(1, 1, -1) + self.renorm_offset.view(1, 1, -1)
        return x.masked_fill(m_pad.unsqueeze(-1).bool(), 0.0)

    def _clamp_to_bounds(self, x_out: torch.Tensor) -> torch.Tensor:
        """Hard-clamp translated features to feature bounds (eval path)."""
        if self.lower_bounds is None or self.upper_bounds is None:
            return x_out
        return x_out.clamp(
            min=self.lower_bounds.view(1, 1, -1),
            max=self.upper_bounds.view(1, 1, -1),
        )

    def translate_to_parquet(
        self,
        test_loader: DataLoader,
        output_parquet_path: Path,
        export_full_sequence: bool = True,
    ) -> None:
        self.translator.eval()
        batches = []
        translated_batches = []
        stay_id_batches = []
        time_batches = []
        sample_indices = []
        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(b.to(self.device) for b in batch)
                parts = self.schema_resolver.extract(batch)
                parts["X_val"] = self._apply_renorm(parts["X_val"], parts["M_pad"])
                x_val_out = self.translator(
                    parts["X_val"],
                    parts["X_miss"],
                    parts["t_abs"],
                    parts["M_pad"],
                    parts["X_static"],
                )
                x_val_out = self._clamp_to_bounds(x_val_out)
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out, parts["X_miss"], parts["X_static"],
                    m_pad=parts["M_pad"],
                )
                if export_full_sequence:
                    # Export all non-padded steps, not just the label mask (which is often only last-timestep).
                    export_mask = (~parts["M_pad"]).to(dtype=torch.bool)
                    batches.append((batch[0], batch[1], export_mask))
                else:
                    batches.append(batch)
                translated_batches.append(x_yaib_translated)
                stay_id_batches.append(None)
                sample_indices.append(batch[0].shape[0])

        if output_parquet_path:
            stay_id_batches, time_batches = self._build_id_time_batches(
                test_loader, sample_indices
            )
            self._export_translated_parquet(
                batches,
                translated_batches,
                stay_id_batches,
                time_batches,
                output_parquet_path,
            )

    def evaluate_original(self, test_loader: DataLoader) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
        all_probs = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(b.to(self.device) for b in batch)
                yaib_batch = batch[:3]
                outputs = self.yaib_runtime.forward(yaib_batch)
                mask = yaib_batch[2].to(outputs.device).bool()
                prediction = torch.masked_select(outputs, mask.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
                target = torch.masked_select(yaib_batch[1].to(outputs.device), mask)

                if self.task_type == "regression":
                    raw_pred = prediction[:, 0] if prediction.shape[-1] >= 1 else prediction.squeeze(-1)
                    all_probs.append(raw_pred.detach().cpu())
                else:
                    if outputs.shape[-1] > 1:
                        prediction_proba = torch.softmax(prediction, dim=-1)[:, 1]
                    else:
                        prediction_proba = torch.sigmoid(prediction).squeeze(-1)
                    all_probs.append(prediction_proba.detach().cpu())

                all_targets.append(target.detach().cpu())
                total_loss += self.yaib_runtime.compute_loss(outputs, yaib_batch).item()
                num_batches += 1

        if not all_probs:
            empty = {"MAE": float("inf"), "MSE": float("inf"), "loss": float("inf")} if self.task_type == "regression" else {"AUCROC": 0.0, "AUCPR": 0.0, "loss": float("inf")}
            return empty, None, None

        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        if self.task_type == "regression":
            metrics = _compute_regression_metrics(targets, probs)
            metrics["loss"] = total_loss / num_batches if num_batches > 0 else float("inf")
        else:
            from sklearn.metrics import roc_auc_score, average_precision_score

            try:
                auroc = roc_auc_score(targets, probs)
            except ValueError:
                auroc = 0.0
            try:
                auprc = average_precision_score(targets, probs)
            except ValueError:
                auprc = 0.0

            metrics = {
                "AUCROC": auroc,
                "AUCPR": auprc,
                "loss": total_loss / num_batches if num_batches > 0 else float("inf"),
            }
            try:
                cal = _compute_calibration_metrics(targets, probs)
                metrics.update(cal)
            except Exception as e:
                logging.warning("Calibration metrics failed: %s", e)
        return metrics, probs, targets

    def translate_and_evaluate(
        self,
        test_loader: DataLoader,
        output_parquet_path: Optional[Path] = None,
        sample_output_dir: Optional[Path] = None,
        sample_size: int = 1000,
        export_full_sequence: bool = True,
    ) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
        self.translator.eval()
        all_probs = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        batches = []
        translated_batches = []
        stay_id_batches = []
        time_batches = []
        sample_indices = []
        sample_before: List[torch.Tensor] = []
        sample_after: List[torch.Tensor] = []
        remaining_samples = sample_size

        # D2: Per-feature delta accumulators (running stats to avoid storing all deltas)
        all_deltas_sum = None
        all_deltas_sq_sum = None
        all_deltas_abs_max = None
        all_deltas_near_zero_count = None
        n_valid_timesteps = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(b.to(self.device) for b in batch)
                parts = self.schema_resolver.extract(batch)
                parts["X_val"] = self._apply_renorm(parts["X_val"], parts["M_pad"])
                x_val_out = self.translator(
                    parts["X_val"],
                    parts["X_miss"],
                    parts["t_abs"],
                    parts["M_pad"],
                    parts["X_static"],
                )
                x_val_out = self._clamp_to_bounds(x_val_out)
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out, parts["X_miss"], parts["X_static"],
                    m_pad=parts["M_pad"],
                )

                # D2: Accumulate per-feature delta stats
                delta = (x_val_out - parts["X_val"]).detach()  # (B, T, F)
                valid = ~parts["M_pad"]  # (B, T)
                delta_valid = delta[valid]  # (N_valid, F)
                if delta_valid.shape[0] > 0:
                    if all_deltas_sum is None:
                        n_features = delta_valid.shape[1]
                        all_deltas_sum = torch.zeros(n_features, device=self.device)
                        all_deltas_sq_sum = torch.zeros(n_features, device=self.device)
                        all_deltas_abs_max = torch.zeros(n_features, device=self.device)
                        all_deltas_near_zero_count = torch.zeros(n_features, device=self.device)
                    all_deltas_sum += delta_valid.sum(dim=0)
                    all_deltas_sq_sum += (delta_valid ** 2).sum(dim=0)
                    all_deltas_abs_max = torch.max(all_deltas_abs_max, delta_valid.abs().max(dim=0).values)
                    all_deltas_near_zero_count += (delta_valid.abs() < 1e-4).sum(dim=0).float()
                    n_valid_timesteps += delta_valid.shape[0]

                logits = self.yaib_runtime.forward((x_yaib_translated, parts["y"], parts["M_label"]))
                mask = parts["M_label"].to(logits.device).bool()
                prediction = torch.masked_select(logits, mask.unsqueeze(-1)).reshape(-1, logits.shape[-1])
                target = torch.masked_select(parts["y"].to(logits.device), mask)

                if self.task_type == "regression":
                    raw_pred = prediction[:, 0] if prediction.shape[-1] >= 1 else prediction.squeeze(-1)
                    all_probs.append(raw_pred.detach().cpu())
                elif logits.shape[-1] > 1:
                    prediction_proba = torch.softmax(prediction, dim=-1)[:, 1]
                    all_probs.append(prediction_proba.detach().cpu())
                else:
                    prediction_proba = torch.sigmoid(prediction).squeeze(-1)
                    all_probs.append(prediction_proba.detach().cpu())
                all_targets.append(target.detach().cpu())
                total_loss += self.yaib_runtime.compute_loss(
                    logits, (x_yaib_translated, parts["y"], mask)
                ).item()
                num_batches += 1

                if output_parquet_path is not None:
                    if export_full_sequence:
                        export_mask = (~parts["M_pad"]).to(dtype=torch.bool)
                        batches.append((batch[0], batch[1], export_mask))
                    else:
                        batches.append(batch)
                    translated_batches.append(x_yaib_translated)
                    stay_id_batches.append(None)
                    sample_indices.append(batch[0].shape[0])
                if sample_output_dir is not None and remaining_samples > 0:
                    valid_mask = ~parts["M_pad"]
                    flat_before = parts["X_yaib"][valid_mask]
                    flat_after = x_yaib_translated[valid_mask]
                    take = min(remaining_samples, flat_before.shape[0])
                    if take > 0:
                        sample_before.append(flat_before[:take].detach().cpu())
                        sample_after.append(flat_after[:take].detach().cpu())
                        remaining_samples -= take

        if output_parquet_path is not None:
            stay_id_batches, time_batches = self._build_id_time_batches(
                test_loader, sample_indices
            )
            self._export_translated_parquet(
                batches,
                translated_batches,
                stay_id_batches,
                time_batches,
                output_parquet_path,
            )
        if sample_output_dir is not None and sample_before:
            self._save_translation_samples(sample_before, sample_after, sample_output_dir)

        # D2: Log per-feature delta analysis
        if all_deltas_sum is not None and n_valid_timesteps > 0:
            try:
                mean_delta = all_deltas_sum / n_valid_timesteps
                var_delta = (all_deltas_sq_sum / n_valid_timesteps) - mean_delta ** 2
                std_delta = var_delta.clamp(min=0).sqrt()
                frac_near_zero = all_deltas_near_zero_count / n_valid_timesteps
                mean_abs_delta = (all_deltas_sq_sum / n_valid_timesteps).sqrt()  # RMS as proxy for mean |delta|

                feature_names = self.schema_resolver.dynamic_features
                n_features = len(feature_names)

                # Sort by mean |delta| (descending) for top modified
                sorted_idx = mean_abs_delta.argsort(descending=True)
                top_k = min(5, n_features)

                logging.info("[delta-analysis] Per-feature translation delta stats (n_timesteps=%d):", n_valid_timesteps)
                logging.info("[delta-analysis] Top-%d most modified features:", top_k)
                for i in range(top_k):
                    idx = sorted_idx[i].item()
                    logging.info(
                        "[delta-analysis]   %s: mean=%.6f std=%.6f abs_max=%.6f frac_near_zero=%.4f",
                        feature_names[idx], mean_delta[idx].item(), std_delta[idx].item(),
                        all_deltas_abs_max[idx].item(), frac_near_zero[idx].item(),
                    )
                logging.info("[delta-analysis] Top-%d least modified features:", top_k)
                for i in range(top_k):
                    idx = sorted_idx[n_features - 1 - i].item()
                    logging.info(
                        "[delta-analysis]   %s: mean=%.6f std=%.6f abs_max=%.6f frac_near_zero=%.4f",
                        feature_names[idx], mean_delta[idx].item(), std_delta[idx].item(),
                        all_deltas_abs_max[idx].item(), frac_near_zero[idx].item(),
                    )
            except Exception as e:
                logging.warning("[delta-analysis] Failed to compute delta stats: %s", e)

        if not all_probs:
            empty = {"MAE": float("inf"), "MSE": float("inf"), "loss": float("inf")} if self.task_type == "regression" else {"AUCROC": 0.0, "AUCPR": 0.0, "loss": float("inf")}
            return empty, None, None

        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        if self.task_type == "regression":
            metrics = _compute_regression_metrics(targets, probs)
            metrics["loss"] = total_loss / num_batches if num_batches > 0 else float("inf")
        else:
            from sklearn.metrics import roc_auc_score, average_precision_score

            try:
                auroc = roc_auc_score(targets, probs)
            except ValueError:
                auroc = 0.0
            try:
                auprc = average_precision_score(targets, probs)
            except ValueError:
                auprc = 0.0

            metrics = {
                "AUCROC": auroc,
                "AUCPR": auprc,
                "loss": total_loss / num_batches if num_batches > 0 else float("inf"),
            }
            try:
                cal = _compute_calibration_metrics(targets, probs)
                metrics.update(cal)
            except Exception as e:
                logging.warning("Calibration metrics failed: %s", e)
        return metrics, probs, targets

    def evaluate_original_vs_translated(
        self,
        test_loader: DataLoader,
        output_parquet_path: Optional[Path] = None,
        sample_output_dir: Optional[Path] = None,
        sample_size: int = 1000,
        export_full_sequence: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        logging.info("Evaluating original test data...")
        original_metrics, orig_probs, orig_targets = self.evaluate_original(test_loader)
        logging.info("Evaluating translated test data...")
        translated_metrics, trans_probs, trans_targets = self.translate_and_evaluate(
            test_loader,
            output_parquet_path,
            sample_output_dir=sample_output_dir,
            sample_size=sample_size,
            export_full_sequence=export_full_sequence,
        )

        if output_parquet_path:
            out_path = Path(output_parquet_path)
            if trans_probs is not None:
                _save_predictions(trans_probs, trans_targets, out_path)
            if orig_probs is not None:
                _save_predictions(orig_probs, orig_targets, out_path, suffix=".original")

        return {"original": original_metrics, "translated": translated_metrics}

    def _export_translated_parquet(
        self,
        batches: List[Tuple[torch.Tensor, ...]],
        translated_batches: List[torch.Tensor],
        stay_id_batches: List[Optional[torch.Tensor]],
        time_batches: Optional[List[Optional[List[List[float]]]]],
        output_path: Path,
    ) -> None:
        feature_names = list(self.schema_resolver.feature_names)
        if translated_batches:
            expected_dim = translated_batches[0].shape[-1]
            if len(feature_names) != expected_dim:
                sequence_col = self.yaib_runtime.vars.get("SEQUENCE")
                if sequence_col and sequence_col in feature_names and len(feature_names) - 1 == expected_dim:
                    keep_idx = [i for i, name in enumerate(feature_names) if name != sequence_col]
                    feature_names = [feature_names[i] for i in keep_idx]
                    translated_batches = [batch[:, :, keep_idx] for batch in translated_batches]
                else:
                    raise ValueError(
                        f"Feature name count ({len(feature_names)}) does not match translated tensor "
                        f"dimension ({expected_dim})."
                    )

        translated_df = reconstruct_parquet_from_batches(
            batches,
            translated_batches,
            stay_id_batches,
            self.yaib_runtime.vars,
            feature_names,
            time_batches,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        translated_df.write_parquet(output_path)
        logging.info("Exported translated parquet to %s", output_path)

    def _save_translation_samples(
        self,
        sample_before: List[torch.Tensor],
        sample_after: List[torch.Tensor],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        before = torch.cat(sample_before, dim=0)
        after = torch.cat(sample_after, dim=0)
        feature_names = list(self.schema_resolver.feature_names)
        if before.shape[1] != len(feature_names):
            sequence_col = self.yaib_runtime.vars.get("SEQUENCE")
            if sequence_col and sequence_col in feature_names and len(feature_names) - 1 == before.shape[1]:
                keep_idx = [i for i, name in enumerate(feature_names) if name != sequence_col]
                feature_names = [feature_names[i] for i in keep_idx]
                before = before[:, keep_idx]
                after = after[:, keep_idx]
            else:
                raise ValueError(
                    f"Sample tensor dimension ({before.shape[1]}) does not match feature names "
                    f"({len(feature_names)})."
                )
        cols = {}
        before_np = before.numpy()
        after_np = after.numpy()
        for idx, name in enumerate(feature_names):
            cols[f"{name}_before"] = before_np[:, idx]
            cols[f"{name}_after"] = after_np[:, idx]
        sample_df = pd.DataFrame(cols)
        sample_df.to_csv(output_dir / "translation_samples.csv", index=False)

    def _build_id_time_batches(
        self,
        loader: DataLoader,
        batch_sizes: List[int],
    ) -> tuple[List[Optional[torch.Tensor]], List[Optional[List[List[float]]]]]:
        dataset = loader.dataset
        base_dataset, index_map = self._resolve_dataset_indices(dataset)
        group_col = base_dataset.vars.get("GROUP") if hasattr(base_dataset, "vars") else None
        seq_col = base_dataset.vars.get("SEQUENCE") if hasattr(base_dataset, "vars") else None
        if not group_col or not hasattr(base_dataset, "outcome_df"):
            return [None] * len(batch_sizes), [None] * len(batch_sizes)

        try:
            stay_ids_list = base_dataset.outcome_df[group_col].unique().to_list()
            stay_ids = [stay_ids_list[idx] for idx in index_map]
        except Exception as exc:
            # Catches IndexError (index_map vs stay_ids mismatch),
            # polars.exceptions.ColumnNotFoundError (group_col not in outcome_df),
            # and any other data-schema issues.  This is a non-critical metadata
            # path — the parquet just won't have stay_id/time columns.
            logging.warning(
                "Failed to build stay_id mapping in _build_id_time_batches: %s. "
                "Falling back to no stay_id/time metadata (parquet export may lack these columns).",
                exc,
            )
            return [None] * len(batch_sizes), [None] * len(batch_sizes)

        time_map: dict[int, List[float]] = {}
        if seq_col and hasattr(base_dataset, "row_indicators"):
            try:
                import polars as pl
                stay_set = set(int(sid) for sid in stay_ids)
                ri = base_dataset.row_indicators
                if isinstance(ri, pl.DataFrame) and seq_col in ri.columns and group_col in ri.columns:
                    df = (
                        ri.filter(pl.col(group_col).is_in(list(stay_set)))
                        .sort([group_col, seq_col])
                        .group_by(group_col)
                        .agg(pl.col(seq_col).alias("_time"))
                    )
                    time_map = {
                        int(row[group_col]): row["_time"] for row in df.iter_rows(named=True)
                    }
            except Exception:
                time_map = {}

        stay_id_batches: List[Optional[torch.Tensor]] = []
        time_batches: List[Optional[List[List[float]]]] = []
        cursor = 0
        for batch_size in batch_sizes:
            batch_ids = stay_ids[cursor : cursor + batch_size]
            cursor += batch_size
            stay_id_batches.append(
                torch.tensor(batch_ids, dtype=torch.long)
                if batch_ids
                else None
            )
            if time_map:
                time_batches.append([time_map.get(int(sid), []) for sid in batch_ids])
            else:
                time_batches.append(None)

        return stay_id_batches, time_batches

    def _resolve_dataset_indices(self, dataset):
        indices = list(range(len(dataset)))
        while True:
            if hasattr(dataset, "_indices") and hasattr(dataset, "_dataset"):
                indices = [dataset._indices[i] for i in indices]
                dataset = dataset._dataset
            elif isinstance(dataset, Subset):
                indices = [dataset.indices[i] for i in indices]
                dataset = dataset.dataset
            else:
                break
        return dataset, indices


class RetrievalTranslatorWrapper(torch.nn.Module):
    """Wraps a RetrievalTranslator + MemoryBank so forward() uses full retrieval.

    The existing TransformerTranslatorEvaluator calls self.translator(...) with
    (x_val, x_miss, t_abs, m_pad, x_static). This wrapper intercepts that call
    and routes through encode → query_memory_bank → forward_with_retrieval,
    returning only x_out (matching the expected signature).
    """

    def __init__(self, translator, memory_bank, k_neighbors: int = 16, retrieval_window: int = 6):
        super().__init__()
        self.translator = translator
        self.memory_bank = memory_bank
        self.k_neighbors = k_neighbors
        self.retrieval_window = retrieval_window

    def forward(self, x_val, x_miss, t_abs, m_pad, x_static, return_forecast=False):
        from ..core.retrieval_translator import query_memory_bank

        # Encode source data
        latent = self.translator.encode(x_val, x_miss, t_abs, m_pad, x_static)

        # Query memory bank with detached latents (same as training)
        importance_weights = self.translator.get_importance_weights()
        context = query_memory_bank(
            query_latents=latent.detach(),
            query_pad_mask=m_pad,
            bank=self.memory_bank,
            k_neighbors=self.k_neighbors,
            retrieval_window=self.retrieval_window,
            importance_weights=importance_weights.detach() if importance_weights is not None else None,
        )

        # Full forward with retrieval context (reuse latent to avoid double-encode)
        x_out, _ = self.translator.forward_with_retrieval(
            x_val, x_miss, t_abs, m_pad, x_static, context,
            latent=latent,
        )
        return x_out
