import logging
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gin
import numpy as np
import polars as pl
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset, Subset

yaib_path = Path(__file__).parent.parent.parent.parent.parent / "YAIB"
if not yaib_path.exists():
    yaib_path = Path(__file__).parent.parent.parent.parent.parent.parent / "YAIB"
sys.path.insert(0, str(yaib_path))

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment, DataSplit
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.models.train import load_model, train_common
import icu_benchmarks.models as yaib_models
from icu_benchmarks.tuning import hyperparameters  # registers gin configurables used by DLTuning.gin


import os
import importlib.util

from contextlib import contextmanager


class _CachedSubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = list(indices)
        self._cached_dataset = [dataset[i] for i in self._indices]
        if hasattr(dataset, "get_feature_names"):
            self.get_feature_names = dataset.get_feature_names
        if hasattr(dataset, "vars"):
            self.vars = dataset.vars

    def __len__(self) -> int:
        return len(self._cached_dataset)

    def __getitem__(self, idx):
        return self._cached_dataset[idx]

@contextmanager
def _pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

def _find_yaib_root(task_config_path: str) -> str:
    # Derive YAIB root from the task_config path: .../YAIB/configs/tasks/X.gin -> .../YAIB
    if task_config_path:
        p = Path(task_config_path).resolve()
        for parent in p.parents:
            if parent.name == "YAIB" or (parent / "icu_benchmarks").is_dir():
                return str(parent)
    # Fallback: walk up from this file looking for a sibling YAIB/
    # (works in both main tree and git worktrees)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "YAIB"
        if candidate.is_dir() and (candidate / "icu_benchmarks").is_dir():
            return str(candidate)
    # Last resort: original hardcoded depth
    repo_root = Path(__file__).resolve().parent.parent.parent
    yaib_path = repo_root.parent.parent / "YAIB"
    return str(yaib_path.resolve())


def import_yaib_run_module(task_config_path: str = ""):
    yaib_root = _find_yaib_root(task_config_path) if task_config_path else _find_yaib_root("")
    run_path = Path(yaib_root) / "icu_benchmarks" / "run.py"
    spec = importlib.util.spec_from_file_location("yaib_run", str(run_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # registers @gin.configurable("Run")
    return module


def _get_gin_param(name: str, default):
    try:
        return gin.query_parameter(name)
    except Exception:
        return default


def _is_yaib_run_registered() -> bool:
    try:
        gin.get_configurable("yaib_run.Run")
        return True
    except Exception:
        return False

class YAIBRuntime:
    def __init__(
        self,
        data_dir: Path,
        baseline_model_dir: Path,
        task_config: Path,
        model_config: Optional[Path],
        model_name: str,
        vars: Dict[str, Any],
        file_names: Dict[str, str],
        seed: int = 42,
        batch_size: int = 1,
        percentile_outliers_csv: Optional[Path] = None,
    ):
        self.data_dir = Path(data_dir)
        self.baseline_model_dir = Path(baseline_model_dir)
        self.task_config = Path(task_config)
        self.model_config = Path(model_config) if model_config else None
        self.model_name = model_name
        self.vars = vars
        self.file_names = file_names
        self.seed = seed
        self.batch_size = batch_size
        self.percentile_outliers_csv = Path(percentile_outliers_csv) if percentile_outliers_csv else None
        
        self._data: Optional[Dict[str, Dict[str, pl.DataFrame]]] = None
        self._model: Optional[Any] = None
        self._preprocessor = None
        self._mode: Optional[RunMode] = None
        self._loss_weight_set = False
        self._trained_columns_set = False
        self._logged_test_stats = False
        self._is_ml_model = False
        self._custom_loss_fn = None

        self.setup_yaib_environment()

    def set_custom_loss(self, loss_fn):
        """Set a custom loss function to replace the default model loss."""
        self._custom_loss_fn = loss_fn
        logging.info("[yaib] Custom loss function set: %s", type(loss_fn).__name__)

    def setup_yaib_environment(self):
        gin.clear_config()
        self._mode = RunMode.classification
        if not _is_yaib_run_registered():
            import_yaib_run_module(str(self.task_config))
        self.wrap_load_gin_config(self.task_config)
        if self.model_config is not None:
            self.wrap_load_gin_config(self.model_config)
        if (self.baseline_model_dir / "train_config.gin").exists():
            self.wrap_load_gin_config(self.baseline_model_dir / "train_config.gin")
        # Auto-detect run mode from gin config (e.g. Regression.gin sets Run.mode)
        self._mode = self._detect_run_mode()

    def _detect_run_mode(self) -> RunMode:
        """Auto-detect RunMode from gin config (Regression.gin sets Run.mode = 'Regression')."""
        try:
            mode_str = gin.query_parameter("Run.mode")
            if hasattr(mode_str, '__str__'):
                mode_str = str(mode_str).strip("'\"")
            if mode_str == "Regression" or mode_str == str(RunMode.regression):
                logging.info("[yaib] Auto-detected RunMode.regression from gin config")
                return RunMode.regression
        except Exception:
            pass
        return RunMode.classification

    @property
    def run_mode(self) -> RunMode:
        return self._mode

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
        
    def _repair_outcome_sequence_column(self) -> None:
        """Add a per-stay row-index SEQUENCE column to outc.parquet if missing.

        YAIB's check_sanitize_data() dedupes the outcome by (GROUP, SEQUENCE)
        when SEQUENCE is present, otherwise by GROUP alone (collapsing
        per-timestep labels to one row per stay). HiRID LoS shipped without a
        SEQUENCE column even though it has per-timestep labels, which silently
        collapses LoS to per-stay during preprocessing. Mirror the eICU LoS
        convention (time = 0..N-1 per stay, in row order) so YAIB treats the
        outcome as dynamic. Safe no-op for per-stay tasks (rows == stays) and
        for outcomes that already have the SEQUENCE column.
        """
        seq_col = self.vars.get("SEQUENCE")
        group_col = self.vars.get("GROUP")
        if not seq_col or not group_col:
            return
        outc_name = self.file_names.get("OUTCOME")
        if not outc_name:
            return
        outc_path = self.data_dir / outc_name
        if not outc_path.exists():
            return
        try:
            outc = pl.read_parquet(outc_path)
        except Exception as exc:
            logging.info("[yaib] outcome repair: read failed (%s) — skipping", exc)
            return
        if seq_col in outc.columns or group_col not in outc.columns:
            return
        n_rows = len(outc)
        n_stays = outc[group_col].n_unique()
        if n_rows == n_stays:
            # Per-stay outcome (KF-style); SEQUENCE legitimately absent.
            return
        backup = outc_path.with_suffix(outc_path.suffix + ".bak_no_time")
        if not backup.exists():
            import shutil
            shutil.copy2(outc_path, backup)
        repaired = outc.with_columns(
            (pl.col(group_col).cum_count().over(group_col) - 1)
            .cast(pl.Float64)
            .alias(seq_col)
        )
        tmp = outc_path.with_suffix(outc_path.suffix + ".tmp")
        repaired.write_parquet(tmp)
        os.replace(tmp, outc_path)
        logging.warning(
            "[yaib] Repaired outcome at %s: added '%s' column "
            "(rows=%d, stays=%d). Backup saved to %s.",
            outc_path, seq_col, n_rows, n_stays, backup,
        )

    def load_data(self, scaling_override: Optional[bool] = None, load_cache: bool = True) -> Dict[str, Dict[str, pl.DataFrame]]:
        if self._data is not None:
            return self._data

        # HiRID LoS outc.parquet ships without a SEQUENCE column, which causes
        # YAIB's check_sanitize_data() to dedupe by stay_id only, collapsing the
        # per-timestep LoS labels to one-per-stay (KF-style). Detect and repair
        # in place. Idempotent; only triggers when outc has multiple rows per
        # stay AND the SEQUENCE column is missing.
        self._repair_outcome_sequence_column()
        logging.info("Loading and preprocessing data using YAIB...")
        prev_scaling = None
        prev_reg_scaling = None
        if scaling_override is not None:
            try:
                prev_scaling = gin.query_parameter("base_classification_preprocessor.scaling")
            except Exception:
                prev_scaling = None
            try:
                prev_reg_scaling = gin.query_parameter("base_regression_preprocessor.scaling")
            except Exception:
                prev_reg_scaling = None
            gin.bind_parameter("base_classification_preprocessor.scaling", scaling_override)
            gin.bind_parameter("base_regression_preprocessor.scaling", scaling_override)

        cv_repetitions = _get_gin_param("execute_repeated_cv.cv_repetitions", 5)
        cv_folds = _get_gin_param("execute_repeated_cv.cv_folds", 5)
        repetition_index = _get_gin_param("execute_repeated_cv.repetition_index", 0)
        fold_index = _get_gin_param("execute_repeated_cv.fold_index", 0)
        train_size = _get_gin_param("execute_repeated_cv.train_size", None)
        complete_train = _get_gin_param("execute_repeated_cv.complete_train", False)
        logging.info(
            "[debug] cv params "
            f"cv_repetitions={cv_repetitions} cv_folds={cv_folds} "
            f"repetition_index={repetition_index} fold_index={fold_index} "
            f"train_size={train_size} complete_train={complete_train}"
        )
        
        self._data = preprocess_data(
            data_dir=self.data_dir,
            file_names=self.file_names,
            vars=self.vars,
            seed=self.seed,
            cv_repetitions=cv_repetitions,
            cv_folds=cv_folds,
            repetition_index=repetition_index,
            fold_index=fold_index,
            train_size=train_size,
            complete_train=complete_train,
            load_cache=load_cache,
            generate_cache=not load_cache,
            percentile_outliers_csv=self.percentile_outliers_csv,
            export_feature_stats=False,
            runmode=self._mode,
        )
        if scaling_override is not None:
            if prev_scaling is not None:
                gin.bind_parameter("base_classification_preprocessor.scaling", prev_scaling)
            if prev_reg_scaling is not None:
                gin.bind_parameter("base_regression_preprocessor.scaling", prev_reg_scaling)
        
        logging.info(f"Data loaded: train={len(self._data[DataSplit.train][DataSegment.features])} rows, "
                    f"val={len(self._data[DataSplit.val][DataSegment.features])} rows, "
                    f"test={len(self._data[DataSplit.test][DataSegment.features])} rows")
        return self._data
    
    def load_baseline_model(self, freeze: bool = True):
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

        model_class = model_map.get(self.model_name)
        if model_class is None and hasattr(yaib_models, self.model_name):
            model_class = getattr(yaib_models, self.model_name)
            self._is_ml_model = True
        if model_class is None:
            raise ValueError(
                f"Unknown model: {self.model_name}. Supported: {list(model_map.keys())} "
                f"+ ML models in icu_benchmarks.models"
            )

        self._model = load_model(model_class, self.baseline_model_dir, pl_model=True)

        if isinstance(self._model, LightningModule):
            if freeze:
                for param in self._model.parameters():
                    param.requires_grad = False
                self._model.eval()
            else:
                # Keep model trainable — used for fine-tuning upper bound baseline
                for param in self._model.parameters():
                    param.requires_grad = True
                self._model.train()
                logging.info("Baseline model loaded UNFROZEN (fine-tune mode)")
        else:
            self._is_ml_model = True
            if not hasattr(self._model, "requires_backprop"):
                self._model.requires_backprop = False

        if freeze:
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
    
    def create_dataloader(
        self,
        split: str,
        shuffle: bool = False,
        ram_cache: bool = True,
        subset_fraction: float | None = None,
        subset_seed: int = 42,
        drop_last: bool | None = None,
    ) -> DataLoader:
        self.load_baseline_model()
        train_dataset = None
        if not self._loss_weight_set and hasattr(self._model, 'set_weight') and self._mode != RunMode.regression:
            train_dataset = self.create_dataset(DataSplit.train, ram_cache=False)
            self._model.set_weight('balanced', train_dataset)
            self._loss_weight_set = True
            if hasattr(self._model, 'loss_weights'):
                loss_weights = getattr(self._model, 'loss_weights', None)
                if loss_weights is not None:
                    lw = loss_weights.detach().cpu()
                    logging.info(
                        f"[debug] loss_weights stats min={lw.min().item():.6f} "
                        f"max={lw.max().item():.6f} mean={lw.mean().item():.6f}"
                    )
        if not self._trained_columns_set and hasattr(self._model, 'set_trained_columns'):
            if train_dataset is None:
                train_dataset = self.create_dataset(DataSplit.train, ram_cache=False)
            self._model.set_trained_columns(train_dataset.get_feature_names())
            self._trained_columns_set = True
        base_ram_cache = ram_cache and subset_fraction is None
        dataset = self.create_dataset(split, ram_cache=base_ram_cache)
        dataset_to_use = dataset
        if subset_fraction is not None and subset_fraction < 1.0:
            generator = torch.Generator()
            generator.manual_seed(subset_seed)
            indices = self._stratified_subset_indices(dataset, subset_fraction, generator)
            if ram_cache:
                dataset_to_use = _CachedSubsetDataset(dataset, indices)
            else:
                dataset_to_use = Subset(dataset, indices)
            if hasattr(dataset, "get_feature_names") and not hasattr(dataset_to_use, "get_feature_names"):
                dataset_to_use.get_feature_names = dataset.get_feature_names
            if hasattr(dataset, "vars") and not hasattr(dataset_to_use, "vars"):
                dataset_to_use.vars = dataset.vars
            if drop_last is None:
                drop_last = (split == DataSplit.train)

        if drop_last is None:
            drop_last = (split == DataSplit.train)

        batch_size = self.batch_size
        if split == DataSplit.test and subset_fraction is None and 1==2:
            batch_size = min(self.batch_size * 4, len(dataset_to_use))

        if hasattr(self._model, 'run_mode'):
            self._model.run_mode = self._mode
            logging.info(f"Set run_mode to {self._model.run_mode} (overriding checkpoint value)")
        # Handle case where loaded ML models are raw sklearn models without YAIB wrapper methods# Handle case where loaded ML models are raw sklearn models without YAIB wrapper methods
        # Add requires_backprop=False for sklearn models that don't have this attribute
        if not hasattr(self._model, 'requires_backprop'):
            self._model.requires_backprop = False
        if hasattr(self._model, 'set_trained_columns') and not self._trained_columns_set:
            self._model.set_trained_columns(dataset.get_feature_names())
            self._trained_columns_set = True
        loader = DataLoader(
            dataset_to_use,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=True,
        )
        self._log_dataset_stats(dataset, split)
        self._log_split_hash(dataset_to_use, split, subset_fraction)
        logging.info(
            f"[debug] dataloader split={split} batch_size={batch_size} "
            f"shuffle={shuffle} drop_last={drop_last} num_workers={loader.num_workers} "
            f"num_batches={len(loader)}"
        )
        if split == DataSplit.test and not self._logged_test_stats:
            self._log_test_batch_stats(loader)
            self._logged_test_stats = True
        return loader

    def _log_dataset_stats(self, dataset: PredictionPolarsDataset, split: str) -> None:
        if not hasattr(dataset, "outcome_df"):
            return
        label_col = self.vars.get("LABEL")
        if not label_col or label_col not in dataset.outcome_df.columns:
            return
        try:
            if self._mode == RunMode.regression:
                # Continuous labels: log min/max/mean/std instead of 0/1 counts
                label_series = dataset.outcome_df[label_col].cast(pl.Float64)
                logging.info(
                    "[debug] split=%s (regression) label_stats: min=%.4f max=%.4f mean=%.4f std=%.4f n=%d",
                    split,
                    label_series.min(),
                    label_series.max(),
                    label_series.mean(),
                    label_series.std(),
                    len(label_series),
                )
            else:
                counts = dataset.outcome_df[label_col].value_counts(parallel=True)
                labels = counts.get_column(label_col).to_list()
                count_col = counts.columns[1] if len(counts.columns) > 1 else counts.columns[0]
                freqs = counts.get_column(count_col).to_list()
                stats = {str(label): int(freq) for label, freq in zip(labels, freqs)}
                total = sum(stats.values())
                pos = stats.get("1", stats.get(1, 0))
                neg = stats.get("0", stats.get(0, 0))
                pos_rate = (pos / total) if total > 0 else 0.0
                logging.info(
                    f"[debug] split={split} label_counts={stats} "
                    f"pos={pos} neg={neg} pos_rate={pos_rate:.6f} total={total}"
                )
        except Exception as exc:
            logging.info(f"[debug] split={split} label_counts=unavailable ({exc})")

    def _log_split_hash(
        self,
        dataset: PredictionPolarsDataset,
        split: str,
        subset_fraction: float | None,
    ) -> None:
        if os.getenv("YAIB_SPLIT_HASH", "0") != "1":
            return
        if not hasattr(dataset, "outcome_df"):
            logging.info(f"[debug] split={split} stay_id_hash=unavailable (no outcome_df)")
            return
        group_col = self.vars.get("GROUP")
        if not group_col or group_col not in dataset.outcome_df.columns:
            return
        try:
            ids = dataset.outcome_df[group_col].unique().to_list()
            ids = sorted(ids)
            digest = hashlib.md5()
            for val in ids:
                digest.update(str(val).encode("utf-8"))
                digest.update(b",")
            logging.info(
                "[debug] split=%s stay_id_hash=%s num_stays=%d subset_fraction=%s",
                split,
                digest.hexdigest(),
                len(ids),
                subset_fraction,
            )
            output_dir = Path(os.getenv("YAIB_SPLIT_HASH_DIR", str(Path(__file__).resolve().parent.parent.parent / "yaib_split_ids")))
            output_dir.mkdir(parents=True, exist_ok=True)
            dataset_tag = f"{self.data_dir.parent.name}_{self.data_dir.name}"
            subset_tag = "full" if subset_fraction is None else f"{subset_fraction:.4f}".replace(".", "p")
            output_path = output_dir / f"stay_ids_{dataset_tag}_{split}_{subset_tag}_{digest.hexdigest()}.txt"
            with output_path.open("w", encoding="utf-8") as handle:
                for stay_id in ids:
                    handle.write(f"{stay_id}\n")
            logging.info("[debug] wrote stay_id list to %s", output_path)
        except Exception as exc:
            logging.info(f"[debug] split={split} stay_id_hash=unavailable ({exc})")

    def _stratified_subset_indices(
        self,
        dataset: PredictionPolarsDataset,
        fraction: float,
        generator: torch.Generator,
    ) -> list[int]:
        """Return a stratified subset of dataset indices that preserves the
        label positive rate.  Falls back to uniform random if labels are
        unavailable (e.g. regression tasks).

        The *generator* is consumed once (for the seed) so callers still
        advance their RNG state deterministically.
        """
        from sklearn.model_selection import StratifiedShuffleSplit

        subset_size = max(1, int(len(dataset) * fraction))
        # Consume the generator once to derive a deterministic seed
        rng_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())

        group_col = self.vars.get("GROUP")
        label_col = self.vars.get("LABEL")
        can_stratify = (
            self._mode != RunMode.regression
            and hasattr(dataset, "outcome_df")
            and group_col
            and label_col
            and label_col in dataset.outcome_df.columns
        )

        if can_stratify:
            stay_ids = dataset.outcome_df[group_col].unique().to_list()
            stay_label_df = dataset.outcome_df.group_by(group_col).agg(
                pl.col(label_col).max().alias("_pos")
            )
            label_map = dict(
                zip(stay_label_df[group_col].to_list(), stay_label_df["_pos"].to_list())
            )
            labels = [label_map.get(sid, 0) for sid in stay_ids]

            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=subset_size, random_state=rng_seed,
            )
            indices, _ = next(splitter.split(range(len(dataset)), labels))
            indices = indices.tolist()

            n_pos_full = sum(1 for lb in labels if lb == 1)
            n_pos_sub = sum(1 for i in indices if labels[i] == 1)
            n_total = len(indices)
            pos_rate_full = n_pos_full / len(stay_ids) if stay_ids else 0
            pos_rate_sub = n_pos_sub / n_total if n_total > 0 else 0
            logging.info(
                "[debug] stratified subset: %d stays -> %d (pos=%d neg=%d) "
                "pos_rate_full=%.4f pos_rate_subset=%.4f",
                len(stay_ids), n_total, n_pos_sub, n_total - n_pos_sub,
                pos_rate_full, pos_rate_sub,
            )
        else:
            # Fallback: uniform random (regression or missing labels)
            np_rng = np.random.RandomState(rng_seed)
            indices = np_rng.choice(len(dataset), size=subset_size, replace=False).tolist()
            logging.info(
                "[debug] uniform subset (no labels): %d stays -> %d",
                len(dataset), len(indices),
            )

        return indices

    def _log_test_batch_stats(self, loader: DataLoader) -> None:
        try:
            batch = next(iter(loader))
        except StopIteration:
            logging.info("[debug] test batch stats unavailable: empty loader")
            return
        device = torch.device("cpu")
        if hasattr(self._model, "parameters"):
            device = next(self._model.parameters()).device
        batch = tuple(b.to(device) for b in batch)
        data, labels, mask = batch
        with torch.no_grad():
            outputs = self.forward((data, labels, mask))
        mask = mask.to(outputs.device).bool()
        prediction = torch.masked_select(outputs, mask.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
        target = torch.masked_select(labels.to(outputs.device), mask)
        if self._mode == RunMode.regression:
            raw_pred = prediction[:, 0] if prediction.shape[-1] >= 1 else prediction.squeeze(-1)
            logging.info(
                "[debug] test_batch prediction stats (regression) "
                f"min={raw_pred.min().item():.6f} max={raw_pred.max().item():.6f} "
                f"mean={raw_pred.mean().item():.6f} std={raw_pred.std().item():.6f}"
            )
            logging.info(
                "[debug] test_batch target stats (regression) "
                f"min={target.min().item():.6f} max={target.max().item():.6f} "
                f"mean={target.mean().item():.6f} std={target.std().item():.6f}"
            )
        else:
            if outputs.shape[-1] > 1:
                probs = torch.softmax(prediction, dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(prediction).squeeze(-1)
            logging.info(
                "[debug] test_batch logits stats "
                f"min={prediction.min().item():.6f} max={prediction.max().item():.6f} "
                f"mean={prediction.mean().item():.6f} std={prediction.std().item():.6f}"
            )
            logging.info(
                "[debug] test_batch prob stats "
                f"min={probs.min().item():.6f} max={probs.max().item():.6f} "
                f"mean={probs.mean().item():.6f} std={probs.std().item():.6f}"
            )
            positives = int((target > 0.5).sum().item())
            total = int(target.numel())
            logging.info(f"[debug] test_batch targets pos={positives} neg={total - positives} total={total}")
    
    def compute_loss(self, outputs: torch.Tensor, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self._model is None:
            self.load_baseline_model()
        
        labels, mask = batch[1], batch[2]
        
        if isinstance(self._model, LightningModule):
            if hasattr(self._model, 'loss'):
                device = outputs.device
                mask = mask.to(device).bool()
                labels = labels.to(device)
                prediction = torch.masked_select(outputs, mask.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
                target = torch.masked_select(labels, mask)
                run_mode = getattr(self._model, "run_mode", RunMode.classification)
                if outputs.shape[-1] > 1 and run_mode == RunMode.classification:
                    if self._custom_loss_fn is not None:
                        loss = self._custom_loss_fn(prediction, target.long())
                    else:
                        loss_weights = getattr(self._model, "loss_weights", None)
                        if loss_weights is not None:
                            loss_weights = loss_weights.to(device)
                        loss = self._model.loss(prediction, target.long(), weight=loss_weights)
                elif run_mode == RunMode.regression:
                    loss = self._model.loss(prediction[:, 0], target.float())
                else:
                    raise ValueError(f"Unsupported run mode: {run_mode}")
                return loss
            raise AttributeError("Model has no loss computation method")
        elif self._is_ml_model:
            mask = mask.to(outputs.device).bool()
            labels = labels.to(outputs.device)
            prediction = torch.masked_select(outputs, mask.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
            target = torch.masked_select(labels, mask)

            run_mode = self._mode or RunMode.classification
            if run_mode == RunMode.regression:
                loss = torch.mean((prediction[:, 0] - target.float()) ** 2)
                return loss

            if prediction.shape[-1] > 1:
                probs = torch.softmax(prediction, dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(prediction).squeeze(-1)

            from sklearn.metrics import log_loss

            try:
                loss_value = log_loss(target.detach().cpu().numpy(), probs.detach().cpu().numpy())
            except ValueError:
                loss_value = float("inf")
            return outputs.new_tensor(loss_value)
        else:
            raise NotImplementedError("ML models not yet supported for loss computation")
    
    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self._model is None:
            self.load_baseline_model()
        
        data = batch[0]
        device = next(self._model.parameters()).device if hasattr(self._model, "parameters") else torch.device("cpu")

        if isinstance(data, list):
            data = [d.float().to(device) for d in data]
        else:
            data = data.float().to(device)

        if self._is_ml_model:
            mask = batch[2].to(device).bool()
            flat_data = data.reshape(-1, data.shape[-1])
            flat_mask = mask.reshape(-1)
            if flat_mask.sum() == 0:
                return data.new_zeros((*data.shape[:2], 1))
            features = flat_data[flat_mask].detach().cpu().numpy()
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(features)
            elif hasattr(self._model, "decision_function"):
                scores = self._model.decision_function(features)
                if scores.ndim == 1:
                    proba = 1 / (1 + np.exp(-scores))
                else:
                    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
                    proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            else:
                preds = self._model.predict(features)
                proba = preds

            eps = 1e-6
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] > 1:
                log_probs = np.log(np.clip(proba, eps, 1.0))
                out_dim = proba.shape[1]
            else:
                if isinstance(proba, np.ndarray):
                    p1 = proba.reshape(-1)
                else:
                    p1 = np.asarray(proba).reshape(-1)
                p1 = np.clip(p1, eps, 1 - eps)
                logit = np.log(p1 / (1 - p1))
                log_probs = logit.reshape(-1, 1)
                out_dim = 1

            output = data.new_zeros((flat_data.shape[0], out_dim))
            output[flat_mask] = torch.from_numpy(log_probs).to(output.device, output.dtype)
            return output.view(data.shape[0], data.shape[1], out_dim)

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
        device = outputs.device
        labels = labels.to(device)
        mask = mask.to(device)
        
        prediction = torch.masked_select(outputs, mask.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
        target = torch.masked_select(labels, mask)
        target_np = target.cpu().numpy()

        run_mode = getattr(self._model, "run_mode", self._mode) if isinstance(self._model, LightningModule) else (self._mode or RunMode.classification)

        if run_mode == RunMode.regression:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            pred_np = prediction[:, 0].cpu().numpy() if prediction.shape[-1] >= 1 else prediction.squeeze(-1).cpu().numpy()
            mae = mean_absolute_error(target_np, pred_np)
            mse = mean_squared_error(target_np, pred_np)
            return {
                "MAE": float(mae),
                "MSE": float(mse),
                "RMSE": float(mse ** 0.5),
                "R2": float(r2_score(target_np, pred_np)),
                "loss": float(mse),
            }

        if isinstance(self._model, LightningModule) or self._is_ml_model:
            if outputs.shape[-1] > 1:
                prediction_proba = torch.softmax(prediction, dim=-1)[:, 1].cpu().numpy()
            else:
                prediction_proba = torch.sigmoid(prediction).cpu().numpy()

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
                loss = float("inf")
            return {"AUCROC": auroc, "AUCPR": auprc, "loss": loss}
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
