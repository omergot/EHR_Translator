import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..adapters.yaib import YAIBRuntime
from ..core.translator import Translator
from ..core.schema import SchemaResolver
from ..core.io_parquet import reconstruct_parquet_from_batches, write_translated_parquet


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

        return {
            "AUCROC": auroc,
            "AUCPR": auprc,
            "loss": total_loss / num_batches if num_batches > 0 else float("inf"),
        }


class TransformerTranslatorEvaluator:
    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: torch.nn.Module,
        schema_resolver: SchemaResolver,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.yaib_runtime = yaib_runtime
        self.translator = translator.to(device)
        self.schema_resolver = schema_resolver
        self.device = device

    def translate_to_parquet(self, test_loader: DataLoader, output_parquet_path: Path) -> None:
        self.translator.eval()
        batches = []
        translated_batches = []
        stay_id_batches = []
        with torch.no_grad():
            for batch in test_loader:
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
                batches.append(batch)
                translated_batches.append(x_yaib_translated)
                stay_id_batches.append(None)

        if output_parquet_path:
            self._export_translated_parquet(batches, translated_batches, stay_id_batches, output_parquet_path)

    def evaluate_original(self, test_loader: DataLoader) -> Dict[str, float]:
        all_probs = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(b.to(self.device) for b in batch)
                outputs = self.yaib_runtime.forward(batch)
                mask = batch[2].to(outputs.device).bool()
                prediction = torch.masked_select(outputs, mask.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
                target = torch.masked_select(batch[1].to(outputs.device), mask)

                if outputs.shape[-1] > 1:
                    prediction_proba = torch.softmax(prediction, dim=-1)[:, 1]
                else:
                    prediction_proba = torch.sigmoid(prediction).squeeze(-1)

                all_probs.append(prediction_proba.detach().cpu())
                all_targets.append(target.detach().cpu())
                total_loss += self.yaib_runtime.compute_loss(outputs, batch).item()
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

        return {
            "AUCROC": auroc,
            "AUCPR": auprc,
            "loss": total_loss / num_batches if num_batches > 0 else float("inf"),
        }

    def translate_and_evaluate(
        self,
        test_loader: DataLoader,
        output_parquet_path: Optional[Path] = None,
        sample_output_dir: Optional[Path] = None,
        sample_size: int = 1000,
    ) -> Dict[str, float]:
        self.translator.eval()
        all_probs = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        batches = []
        translated_batches = []
        stay_id_batches = []
        sample_before: List[torch.Tensor] = []
        sample_after: List[torch.Tensor] = []
        remaining_samples = sample_size

        with torch.no_grad():
            for batch in test_loader:
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
                    batches.append(batch)
                    translated_batches.append(x_yaib_translated)
                    stay_id_batches.append(None)
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
            self._export_translated_parquet(batches, translated_batches, stay_id_batches, output_parquet_path)
        if sample_output_dir is not None and sample_before:
            self._save_translation_samples(sample_before, sample_after, sample_output_dir)

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

        return {
            "AUCROC": auroc,
            "AUCPR": auprc,
            "loss": total_loss / num_batches if num_batches > 0 else float("inf"),
        }

    def evaluate_original_vs_translated(
        self,
        test_loader: DataLoader,
        output_parquet_path: Optional[Path] = None,
        sample_output_dir: Optional[Path] = None,
        sample_size: int = 1000,
    ) -> Dict[str, Dict[str, float]]:
        logging.info("Evaluating original test data...")
        original_metrics = self.evaluate_original(test_loader)
        logging.info("Evaluating translated test data...")
        translated_metrics = self.translate_and_evaluate(
            test_loader,
            output_parquet_path,
            sample_output_dir=sample_output_dir,
            sample_size=sample_size,
        )
        return {"original": original_metrics, "translated": translated_metrics}

    def _export_translated_parquet(
        self,
        batches: List[Tuple[torch.Tensor, ...]],
        translated_batches: List[torch.Tensor],
        stay_id_batches: List[Optional[torch.Tensor]],
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
