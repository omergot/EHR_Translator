import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gin
import numpy as np
import polars as pl
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

yaib_path = Path(__file__).parent.parent.parent.parent.parent / "YAIB"
if not yaib_path.exists():
    yaib_path = Path(__file__).parent.parent.parent.parent.parent.parent / "YAIB"
sys.path.insert(0, str(yaib_path))

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment, DataSplit
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.models.train import load_model, train_common
from icu_benchmarks.tuning import hyperparameters  # registers gin configurables used by DLTuning.gin


import os
import importlib.util

from contextlib import contextmanager

@contextmanager
def _pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

def _find_yaib_root(task_config_path: str) -> str:
    p = Path('/bigdata/omerg/Thesis/YAIB').resolve()
    return str(p)


def import_yaib_run_module():
    run_path = Path('/bigdata/omerg/Thesis/YAIB/icu_benchmarks/run.py')
    spec = importlib.util.spec_from_file_location("yaib_run", str(run_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # registers @gin.configurable("Run")
    return module

class YAIBRuntime:
    def __init__(
        self,
        data_dir: Path,
        baseline_model_dir: Path,
        task_config: Path,
        model_name: str,
        vars: Dict[str, Any],
        file_names: Dict[str, str],
        seed: int = 42,
        batch_size: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.baseline_model_dir = Path(baseline_model_dir)
        self.task_config = Path(task_config)
        self.model_name = model_name
        self.vars = vars
        self.file_names = file_names
        self.seed = seed
        self.batch_size = batch_size
        
        self._data: Optional[Dict[str, Dict[str, pl.DataFrame]]] = None
        self._model: Optional[Any] = None
        self._preprocessor = None
        self._mode: Optional[RunMode] = None

    def _setup_gin_search_paths(self, task_config_path: str):
        gin.clear_config()

        task_path = Path(task_config_path).resolve()

        # YAIB root should be the directory that contains "configs/"
        # If task_config is .../YAIB/configs/prediction_models/LSTM.gin
        # then yaib_root = .../YAIB
        yaib_root = task_path
        while yaib_root != yaib_root.parent and not (yaib_root / "configs").is_dir():
            yaib_root = yaib_root.parent

        if not (yaib_root / "configs").is_dir():
            raise FileNotFoundError(f"Could not find YAIB root containing 'configs/' from: {task_path}")

        gin.add_config_file_search_path(str(yaib_root))
        gin.add_config_file_search_path(str(yaib_root / "configs"))

        # Optional but often helpful if YAIB code assumes repo-root relative paths elsewhere
        # os.chdir(str(yaib_root))


    def wrap_load_gin_config(self, config_path: str):
        yaib_root = _find_yaib_root(config_path)
        gin.add_config_file_search_path(yaib_root)
        gin.add_config_file_search_path(str(Path(yaib_root) / "configs"))

        with _pushd(yaib_root):
            gin.parse_config_file(config_path)
        
    def load_data(self) -> Dict[str, Dict[str, pl.DataFrame]]:
        if self._data is not None:
            return self._data
            
        logging.info("Loading and preprocessing data using YAIB...")
        gin.clear_config()
        self.wrap_load_gin_config(self.task_config)
        yaib_run_module = import_yaib_run_module()

        if (self.baseline_model_dir / "train_config.gin").exists():
            self.wrap_load_gin_config(self.baseline_model_dir / "train_config.gin")
        
        self._data = preprocess_data(
            data_dir=self.data_dir,
            file_names=self.file_names,
            vars=self.vars,
            seed=self.seed,
            load_cache=True,
            generate_cache=False,
        )
        
        self._mode = RunMode.classification
        logging.info(f"Data loaded: train={len(self._data[DataSplit.train][DataSegment.features])} rows, "
                    f"val={len(self._data[DataSplit.val][DataSegment.features])} rows, "
                    f"test={len(self._data[DataSplit.test][DataSegment.features])} rows")
        return self._data
    
    def load_baseline_model(self):
        if self._model is not None:
            return self._model
        
        
        from icu_benchmarks.models import DLModel, MLModelClassifier, MLModelRegression
        from icu_benchmarks.models.dl_models import GRUNet, LSTMNet, TemporalConvNet, Transformer
        
        model_map = {
            "GRU": GRUNet,
            "LSTM": LSTMNet,
            "TCN": TemporalConvNet,
            "Transformer": Transformer,
        }
        
        if self.model_name not in model_map:
            raise ValueError(f"Unknown model: {self.model_name}. Supported: {list(model_map.keys())}")
        
        model_class = model_map[self.model_name]
        
        self._model = load_model(model_class, self.baseline_model_dir, pl_model=True)
        
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()
        
        logging.info(f"Baseline model loaded and frozen: {type(self._model).__name__}")
        return self._model
    
    def create_dataset(self, split: str, ram_cache: bool = True) -> PredictionPolarsDataset:
        if self._data is None:
            self.load_data()
        
        return PredictionPolarsDataset(
            data=self._data,
            split=split,
            # vars=self.vars,
            ram_cache=ram_cache,
        )
    
    def create_dataloader(self, dataset: PredictionPolarsDataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            drop_last=True,
        )
    
    def compute_loss(self, outputs: torch.Tensor, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self._model is None:
            self.load_baseline_model()
        
        labels, mask = batch[1], batch[2]
        
        if isinstance(self._model, LightningModule):
            if hasattr(self._model, 'step_fn'):
                data = batch[0]
                element = (data, labels, mask)
                loss = self._model.step_fn(element, step_prefix="")
                return loss
            else:
                if hasattr(self._model, 'loss'):
                    prediction = torch.masked_select(outputs, mask.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
                    target = torch.masked_select(labels, mask)
                    device = next(self._model.parameters()).device if hasattr(self._model, 'parameters') else torch.device("cpu")
                    if outputs.shape[-1] > 1 and self._model.run_mode == RunMode.classification:
                        loss_weights = self._model.loss_weights.to(device) if hasattr(self._model, 'loss_weights') else None
                        loss = self._model.loss(prediction, target.long(), weight=loss_weights)
                    elif self._model.run_mode == RunMode.regression:
                        loss = self._model.loss(prediction[:, 0], target.float())
                    else:
                        raise ValueError(f"Unsupported run mode: {self._model.run_mode}")
                    return loss
                else:
                    raise AttributeError("Model has no loss computation method")
        else:
            raise NotImplementedError("ML models not yet supported for loss computation")
    
    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self._model is None:
            self.load_baseline_model()
        
        data = batch[0]
        device = next(self._model.parameters()).device if hasattr(self._model, 'parameters') else torch.device("cpu")
        
        if isinstance(data, list):
            data = [d.float().to(device) for d in data]
        else:
            data = data.float().to(device)
        
        with torch.no_grad():
            outputs = self._model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        return outputs
    
    def compute_metrics(
        self,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, ...],
        split: str = "test"
    ) -> Dict[str, float]:
        if self._model is None:
            self.load_baseline_model()
        
        data, labels, mask = batch[0], batch[1], batch[2]
        
        if isinstance(self._model, LightningModule):
            prediction = torch.masked_select(outputs, mask.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
            target = torch.masked_select(labels, mask)
            
            if outputs.shape[-1] > 1:
                prediction_proba = torch.softmax(prediction, dim=-1)[:, 1].cpu().numpy()
            else:
                prediction_proba = torch.sigmoid(prediction).cpu().numpy()
            
            target_np = target.cpu().numpy()
            
            from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
            
            try:
                auroc = roc_auc_score(target_np, prediction_proba)
            except ValueError:
                auroc = 0.0
            
            try:
                auprc = average_precision_score(target_np, prediction_proba)
            except ValueError:
                auprc = 0.0
            
            try:
                loss = log_loss(target_np, prediction_proba)
            except ValueError:
                loss = float('inf')
            
            return {
                "AUCROC": auroc,
                "AUCPR": auprc,
                "loss": loss,
            }
        else:
            raise NotImplementedError("ML models not yet supported for metrics computation")
    
    def export_batch_to_parquet_format(
        self,
        translated_data: torch.Tensor,
        batch: Tuple[torch.Tensor, ...],
        stay_ids: Optional[torch.Tensor] = None,
    ) -> pl.DataFrame:
        if self._data is None:
            self.load_data()
        
        original_data, labels, mask = batch[0], batch[1], batch[2]
        
        translated_np = translated_data.detach().cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        batch_size, seq_len, num_features = translated_np.shape
        
        feature_names = self._data[DataSplit.train][DataSegment.features].columns
        feature_names = [col for col in feature_names if col != self.vars["GROUP"] and col != self.vars.get("SEQUENCE", "")]
        
        rows = []
        for b in range(batch_size):
            for t in range(seq_len):
                if mask_np[b, t]:
                    row = {self.vars["GROUP"]: stay_ids[b].item() if stay_ids is not None else b}
                    if self.vars.get("SEQUENCE"):
                        row[self.vars["SEQUENCE"]] = t
                    for f_idx, f_name in enumerate(feature_names):
                        if f_idx < num_features:
                            row[f_name] = float(translated_np[b, t, f_idx])
                    rows.append(row)
        
        return pl.DataFrame(rows)

