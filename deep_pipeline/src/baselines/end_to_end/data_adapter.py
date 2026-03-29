"""Data adapter: converts YAIB data to fixed-length (B, C, L) tensors for E2E baselines.

Each sample is a fixed-length window of the last `seq_len` timesteps from a stay,
zero-padded on the left for shorter stays.  Channels = 48 dynamic features +
48 missingness indicators = 96.  Static features (4) are provided separately.

Uses the same YAIB data splits (seed 2222) as all other experiments for fair comparison.

**Source/target convention (CRITICAL):**
E2E DA baselines train on MIMIC labels (source) and align with eICU (target).
Evaluation is on eICU test (target domain) — the model has never seen eICU labels.
This matches the standard DA protocol and makes comparison fair with our translator
approach (which also evaluates on eICU test through the frozen MIMIC LSTM).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ...adapters.yaib import YAIBRuntime
from ...core.static_utils import build_static_matrix_for_dataset, load_static_with_recipe
from icu_benchmarks.data.constants import DataSplit


class E2EDataset(Dataset):
    """Fixed-length time-series dataset for end-to-end baselines.

    Each sample returns:
        x:      (C, L) float32 — 96 channels x seq_len timesteps
        y:      scalar or (L,) float32 — label (per-stay: scalar; per-timestep: full sequence)
        static: (S,) float32 — static features (age, sex, height, weight)
        mask:   (L,) bool — True for valid (non-padded) timesteps
    """

    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        split: str,
        seq_len: int = 48,
        static_recipe_path: Optional[Path] = None,
        label_mode: str = "per_stay",
    ):
        """
        Args:
            yaib_runtime: Initialized YAIBRuntime with data loaded.
            split: "train", "val", or "test".
            seq_len: Fixed sequence length (timesteps).
            static_recipe_path: Path to static feature recipe for baking.
            label_mode: "per_stay" (mortality) or "per_timestep" (AKI/sepsis).
                        For per_timestep, the label is max(timestep_labels_in_window).
        """
        self.seq_len = seq_len
        self.label_mode = label_mode

        # Create YAIB dataset to iterate over stays
        self._yaib_dataset = yaib_runtime.create_dataset(split, ram_cache=True)

        # Build static features matrix
        self._static = None
        if static_recipe_path is not None:
            try:
                data_dir = yaib_runtime.data_dir
                file_names = yaib_runtime.file_names
                group_col = yaib_runtime.vars["GROUP"]
                static_features = yaib_runtime.vars.get("STATIC", [])
                static_df = load_static_with_recipe(
                    data_dir=data_dir,
                    file_names=file_names,
                    group_col=group_col,
                    static_features=static_features,
                    recipe_path=static_recipe_path,
                )
                self._static = build_static_matrix_for_dataset(
                    dataset=self._yaib_dataset,
                    static_df=static_df,
                    group_col=group_col,
                    static_features=static_features,
                )
            except Exception as e:
                logging.warning("[E2EDataset] Failed to load static features: %s", e)

        # Pre-cache all samples to avoid repeated YAIB __getitem__ overhead
        self._cache = []
        for idx in range(len(self._yaib_dataset)):
            data, labels, mask = self._yaib_dataset[idx]
            # data: (T, F) where F = num_features (96 for mortality, 100 for AKI/sepsis)
            # labels: (T,) — for per-stay: last timestep has label, rest are -1; padded positions are 0
            # mask: (T,) — YAIB's "pad_mask": True at labeled timesteps only (NOT a pure validity mask!
            #   YAIB sets mask=False at both padded AND unlabeled-but-valid timesteps)
            self._cache.append((data, labels, mask))

        # Determine number of dynamic features (exclude static if > 96)
        sample_data = self._cache[0][0]
        self.num_features = sample_data.shape[1]

        logging.info(
            "[E2EDataset] split=%s n_samples=%d seq_len=%d features=%d label_mode=%s",
            split, len(self._cache), seq_len, self.num_features, label_mode,
        )

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int):
        data, labels, mask = self._cache[idx]

        # data is (T, F), mask is (T,) bool
        T, F = data.shape

        # IMPORTANT: YAIB's mask conflates padding and labeling.
        # For per-stay tasks (mortality), YAIB sets mask=False at unlabeled
        # timesteps (where labels were NaN -> -1), not just at padded positions.
        # This means mask is True at ONLY the labeled timestep, discarding all
        # other real timesteps.
        #
        # To get the actual non-padding mask:
        # - labels == -1 means real timestep without label (was NaN)
        # - mask == True means real timestep with label
        # - labels == 0 AND mask == False in the padding zone means padding
        # So: valid = (labels < 0) | mask  captures all real timesteps.
        valid_mask = (labels < 0) | mask.bool()
        valid_len = valid_mask.sum().item()

        if valid_len == 0:
            # Edge case: all padded — return zeros
            x = torch.zeros(F, self.seq_len, dtype=torch.float32)
            y_scalar = torch.tensor(0.0, dtype=torch.float32)
            y_seq = torch.full((self.seq_len,), -1.0, dtype=torch.float32)
            vmask = torch.zeros(self.seq_len, dtype=torch.bool)
            static = torch.zeros(4, dtype=torch.float32)
            if self._static is not None:
                static = self._static[idx]
            if self.label_mode == "per_stay":
                return x, y_scalar, static, vmask
            else:
                return x, y_seq, static, vmask

        # Extract valid data: take last seq_len timesteps of valid portion
        valid_data = data[valid_mask]  # (valid_len, F)
        valid_labels = labels[valid_mask]  # (valid_len,)

        # Take last seq_len timesteps
        if valid_len >= self.seq_len:
            window_data = valid_data[-self.seq_len:]  # (seq_len, F)
            window_labels = valid_labels[-self.seq_len:]
        else:
            # Pad on the left with zeros
            pad_len = self.seq_len - valid_len
            window_data = torch.cat([
                torch.zeros(pad_len, F, dtype=data.dtype),
                valid_data,
            ], dim=0)
            window_labels = torch.cat([
                torch.full((pad_len,), -1.0, dtype=labels.dtype),
                valid_labels,
            ], dim=0)

        # Transpose to (F, seq_len) = (C, L) for Conv1d
        x = window_data.t()  # (F, seq_len)

        # Validity mask: True for non-padded timesteps in the window.
        # We know exactly how many valid timesteps are in the window from the
        # extraction logic above: min(valid_len, seq_len) real timesteps at the
        # right, with left-padding if valid_len < seq_len.
        if valid_len >= self.seq_len:
            vmask = torch.ones(self.seq_len, dtype=torch.bool)
        else:
            vmask = torch.cat([
                torch.zeros(self.seq_len - valid_len, dtype=torch.bool),
                torch.ones(valid_len, dtype=torch.bool),
            ])

        # Extract label
        if self.label_mode == "per_stay":
            # Last valid label (mortality: single label per stay)
            valid_lbls = window_labels[window_labels >= 0]
            if len(valid_lbls) > 0:
                y = valid_lbls[-1].float()
            else:
                y = torch.tensor(0.0, dtype=torch.float32)
        else:
            # Per-timestep: return full sequence labels (not aggregated)
            # Invalid positions have -1, valid have 0 or 1
            y = window_labels.float()  # (seq_len,)

        # Static features
        if self._static is not None:
            static = self._static[idx].float()
        else:
            static = torch.zeros(4, dtype=torch.float32)

        return x, y, static, vmask


class YAIBToE2EAdapter:
    """Adapter that creates E2E-compatible DataLoaders from a config dict.

    Uses YAIBRuntime to load data with the same splits (seed 2222), then
    wraps in E2EDataset for the fixed-length (B, C, L) format needed by
    CLUDA/RAINCOAT/ACON.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda")
        self.seed = config.get("seed", 2222)
        training = config.get("training", {})
        self.batch_size = training.get("batch_size", 64)
        self.seq_len = training.get("seq_len", 48)
        self.static_recipe_path = Path(config.get("paths", {}).get(
            "static_recipe",
            "/bigdata/omerg/Thesis/cohort_data/sepsis/eicu/preproc/static_recipe",
        ))

        # Determine label mode from task
        self.label_mode = self._infer_label_mode(config)

        # E2E DA convention: train on MIMIC labels (source), align with eICU (target).
        # Config: data_dir = eICU, target_data_dir = MIMIC (same as translator configs).
        # We SWAP them here so source=MIMIC (labeled) and target=eICU (eval domain).
        self._source_runtime = self._build_runtime(config["target_data_dir"])  # MIMIC = source
        self._target_runtime = self._build_runtime(config["data_dir"])         # eICU = target

    def _infer_label_mode(self, config: dict) -> str:
        """Infer label mode from config path or task config."""
        config_path = config.get("_config_path", "")
        data_dir = config.get("data_dir", "")
        combined = (config_path + data_dir).lower()
        if "mortality" in combined:
            return "per_stay"
        return "per_timestep"

    def _build_runtime(self, data_dir: str) -> YAIBRuntime:
        """Build a YAIBRuntime for a given data directory."""
        import copy
        cfg = copy.deepcopy(self.config)
        return YAIBRuntime(
            data_dir=Path(data_dir),
            baseline_model_dir=Path(cfg["baseline_model_dir"]),
            task_config=Path(cfg["task_config"]),
            model_config=Path(cfg["model_config"]) if cfg.get("model_config") else None,
            model_name=cfg["model_name"],
            vars=copy.deepcopy(cfg["vars"]),
            file_names=copy.deepcopy(cfg["file_names"]),
            seed=self.seed,
            batch_size=self.batch_size,
        )

    def _make_dataset(self, runtime: YAIBRuntime, split: str) -> E2EDataset:
        runtime.load_data()
        return E2EDataset(
            yaib_runtime=runtime,
            split=split,
            seq_len=self.seq_len,
            static_recipe_path=self.static_recipe_path,
            label_mode=self.label_mode,
        )

    def get_loaders(self) -> tuple:
        """Return (source_train, source_val, source_test, target_train, target_val, target_test).

        Source = MIMIC (labeled, for training classifier).
        Target = eICU (for DA alignment and evaluation).
        All use the same YAIB splits (seed 2222).
        """
        source_train_ds = self._make_dataset(self._source_runtime, DataSplit.train)
        source_val_ds = self._make_dataset(self._source_runtime, DataSplit.val)
        source_test_ds = self._make_dataset(self._source_runtime, DataSplit.test)
        target_train_ds = self._make_dataset(self._target_runtime, DataSplit.train)
        target_val_ds = self._make_dataset(self._target_runtime, DataSplit.val)
        target_test_ds = self._make_dataset(self._target_runtime, DataSplit.test)

        source_train_loader = DataLoader(
            source_train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4, drop_last=True, pin_memory=True,
        )
        source_val_loader = DataLoader(
            source_val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4, drop_last=False, pin_memory=True,
        )
        source_test_loader = DataLoader(
            source_test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4, drop_last=False, pin_memory=True,
        )
        target_train_loader = DataLoader(
            target_train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4, drop_last=True, pin_memory=True,
        )
        target_val_loader = DataLoader(
            target_val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4, drop_last=False, pin_memory=True,
        )
        target_test_loader = DataLoader(
            target_test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4, drop_last=False, pin_memory=True,
        )

        logging.info(
            "[E2EAdapter] Source(MIMIC) train=%d val=%d test=%d  Target(eICU) train=%d val=%d test=%d  batch_size=%d",
            len(source_train_ds), len(source_val_ds), len(source_test_ds),
            len(target_train_ds), len(target_val_ds), len(target_test_ds), self.batch_size,
        )

        return (source_train_loader, source_val_loader, source_test_loader,
                target_train_loader, target_val_loader, target_test_loader)

    def get_no_adapt_runtime(self) -> YAIBRuntime:
        """Return the source runtime for computing no-adaptation baseline
        (frozen MIMIC LSTM on raw eICU test set)."""
        return self._source_runtime

    @property
    def num_channels(self) -> int:
        """Number of input channels (dynamic features including MI)."""
        # Load a sample to determine
        self._source_runtime.load_data()
        ds = self._source_runtime.create_dataset(DataSplit.train, ram_cache=False)
        sample = ds[0]
        return sample[0].shape[1]  # (T, F) -> F

    @property
    def num_static(self) -> int:
        return 4  # age, sex, height, weight
