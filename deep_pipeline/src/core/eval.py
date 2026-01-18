import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import torch
from torch.utils.data import DataLoader

from ..adapters.yaib import YAIBRuntime
from ..core.translator import Translator
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
        
        all_outputs = []
        all_batches = []
        translated_batches = []
        stay_id_batches = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(b.to(self.device) for b in batch)
                translated_data = self.translator(batch)
                baseline_outputs = self.yaib_runtime.forward((translated_data, batch[1], batch[2]))
                
                all_outputs.append(baseline_outputs)
                all_batches.append(batch)
                translated_batches.append(translated_data)
                stay_id_batches.append(None)
        
        metrics = {}
        for outputs, batch in zip(all_outputs, all_batches):
            batch_metrics = self.yaib_runtime.compute_metrics(outputs, batch, split="test")
            for key, value in batch_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
        
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
        all_outputs = []
        all_batches = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(b.to(self.device) for b in batch)
                baseline_outputs = self.yaib_runtime.forward(batch)
                all_outputs.append(baseline_outputs)
                all_batches.append(batch)
        
        metrics = {}
        for outputs, batch in zip(all_outputs, all_batches):
            batch_metrics = self.yaib_runtime.compute_metrics(outputs, batch, split="test")
            for key, value in batch_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        return {key: sum(values) / len(values) for key, values in metrics.items()}

