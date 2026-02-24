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


class TranslatorEvaluator:
    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: Translator,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.yaib_runtime = yaib_runtime
        self.translator = translator.to(device)
        self.device = device
    
    def translate_and_evaluate(
        self,
        test_loader: DataLoader,
        output_parquet_path: Optional[Path] = None,
    ) -> Dict[str, float]:
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

        if not all_probs:
            avg_metrics = {"AUCROC": 0.0, "AUCPR": 0.0, "loss": float("inf")}
        else:
            probs = torch.cat(all_probs).numpy()
            targets = torch.cat(all_targets).numpy()

            from sklearn.metrics import roc_auc_score, average_precision_score

            try:
                auroc = roc_auc_score(targets, probs)
            except ValueError:
                auroc = 0.0
            try:
                auprc = average_precision_score(targets, probs)
            except ValueError:
                auprc = 0.0

            avg_metrics = {
                "AUCROC": auroc,
                "AUCPR": auprc,
                "loss": total_loss / num_batches if num_batches > 0 else float("inf"),
            }
            try:
                cal = _compute_calibration_metrics(targets, probs)
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
        
        return avg_metrics
    
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
        original_metrics = self._evaluate_without_translator(test_loader)
        
        logging.info("Evaluating translated test data...")
        translated_metrics = self.translate_and_evaluate(test_loader, output_parquet_path)
        
        return {
            "original": original_metrics,
            "translated": translated_metrics,
        }
    
    def _evaluate_without_translator(self, test_loader: DataLoader) -> Dict[str, float]:
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

                if baseline_outputs.shape[-1] > 1:
                    prediction_proba = torch.softmax(prediction, dim=-1)[:, 1]
                else:
                    prediction_proba = torch.sigmoid(prediction).squeeze(-1)

                all_probs.append(prediction_proba.detach().cpu())
                all_targets.append(target.detach().cpu())
                total_loss += self.yaib_runtime.compute_loss(baseline_outputs, batch).item()
                num_batches += 1

        if not all_probs:
            return {"AUCROC": 0.0, "AUCPR": 0.0, "loss": float("inf")}

        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

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
        return metrics


class TransformerTranslatorEvaluator:
    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: torch.nn.Module,
        schema_resolver: SchemaResolver,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        renorm_scale: Optional[torch.Tensor] = None,
        renorm_offset: Optional[torch.Tensor] = None,
    ):
        self.yaib_runtime = yaib_runtime
        self.translator = translator.to(device)
        self.schema_resolver = schema_resolver
        self.device = device
        self.renorm_scale = renorm_scale.to(device) if renorm_scale is not None else None
        self.renorm_offset = renorm_offset.to(device) if renorm_offset is not None else None

    def _apply_renorm(self, x_val: torch.Tensor, m_pad: torch.Tensor) -> torch.Tensor:
        if self.renorm_scale is None:
            return x_val
        x = x_val * self.renorm_scale.view(1, 1, -1) + self.renorm_offset.view(1, 1, -1)
        return x.masked_fill(m_pad.unsqueeze(-1).bool(), 0.0)

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
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out, parts["X_miss"], parts["X_static"]
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

    def evaluate_original(self, test_loader: DataLoader) -> Dict[str, float]:
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

                if outputs.shape[-1] > 1:
                    prediction_proba = torch.softmax(prediction, dim=-1)[:, 1]
                else:
                    prediction_proba = torch.sigmoid(prediction).squeeze(-1)

                all_probs.append(prediction_proba.detach().cpu())
                all_targets.append(target.detach().cpu())
                total_loss += self.yaib_runtime.compute_loss(outputs, yaib_batch).item()
                num_batches += 1

        if not all_probs:
            return {"AUCROC": 0.0, "AUCPR": 0.0, "loss": float("inf")}

        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

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
        return metrics

    def translate_and_evaluate(
        self,
        test_loader: DataLoader,
        output_parquet_path: Optional[Path] = None,
        sample_output_dir: Optional[Path] = None,
        sample_size: int = 1000,
        export_full_sequence: bool = True,
    ) -> Dict[str, float]:
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
                x_yaib_translated = self.schema_resolver.rebuild(
                    parts["X_yaib"], x_val_out, parts["X_miss"], parts["X_static"]
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

                if logits.shape[-1] > 1:
                    prediction_proba = torch.softmax(prediction, dim=-1)[:, 1]
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
            return {"AUCROC": 0.0, "AUCPR": 0.0, "loss": float("inf")}

        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

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
        return metrics

    def evaluate_original_vs_translated(
        self,
        test_loader: DataLoader,
        output_parquet_path: Optional[Path] = None,
        sample_output_dir: Optional[Path] = None,
        sample_size: int = 1000,
        export_full_sequence: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        logging.info("Evaluating original test data...")
        original_metrics = self.evaluate_original(test_loader)
        logging.info("Evaluating translated test data...")
        translated_metrics = self.translate_and_evaluate(
            test_loader,
            output_parquet_path,
            sample_output_dir=sample_output_dir,
            sample_size=sample_size,
            export_full_sequence=export_full_sequence,
        )
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

        stay_ids_series = base_dataset.outcome_df[group_col].unique()
        stay_ids = [stay_ids_series[idx] for idx in index_map]

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
