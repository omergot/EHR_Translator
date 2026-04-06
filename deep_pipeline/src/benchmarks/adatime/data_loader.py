"""Data loading for AdaTime benchmark datasets.

Loads AdaTime datasets (HAR, HHAR, WISDM, SSC/EEG, FD/MFD) and converts them
to the format expected by our translator pipeline:
  - Input shape: (batch, timesteps, features) -- timesteps-first
  - Batch tuple: (data, labels, pad_mask, static_features)

AdaTime native format: (N, channels, timesteps) -- channels-first

Long-sequence datasets (SSC: 3000, FD: 5120) are downsampled via average
pooling to keep sequence length manageable for transformer self-attention.
When full_length=True, no downsampling is applied — the full sequence is
returned as-is, to be chunked downstream by ChunkedAdaTimeCNNRetrievalTrainer.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Dataset configurations (mirroring AdaTime's configs/data_model_configs.py)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DatasetConfig:
    """Configuration for an AdaTime dataset."""
    name: str
    input_channels: int
    sequence_len: int
    num_classes: int
    class_names: list
    # List of (source_domain_id, target_domain_id) scenarios
    scenarios: list

    # For normalization
    normalize: bool = True
    # Whether to drop last incomplete batch during training (matches AdaTime's drop_last)
    drop_last: bool = True

    # CNN architecture params (must match AdaTime's per-dataset configs exactly)
    kernel_size: int = 5
    stride: int = 1
    mid_channels: int = 64
    final_out_channels: int = 128
    dropout: float = 0.5
    features_len: int = 1


DATASET_CONFIGS = {
    "HAR": DatasetConfig(
        name="HAR",
        input_channels=9,
        sequence_len=128,
        num_classes=6,
        class_names=["walk", "upstairs", "downstairs", "sit", "stand", "lie"],
        scenarios=[
            ("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"),
            ("12", "16"), ("18", "27"), ("20", "5"), ("24", "8"),
            ("28", "27"), ("30", "20"),
        ],
        drop_last=True,  # AdaTime: drop_last=True for HAR
    ),
    "HHAR": DatasetConfig(
        name="HHAR",
        input_channels=3,
        sequence_len=128,
        num_classes=6,
        class_names=["bike", "sit", "stand", "walk", "stairs_up", "stairs_down"],
        scenarios=[
            ("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5"),
            ("5", "0"), ("6", "1"), ("7", "4"), ("8", "3"), ("0", "2"),
        ],
        drop_last=True,  # AdaTime: drop_last=True for HHAR
    ),
    "WISDM": DatasetConfig(
        name="WISDM",
        input_channels=3,
        sequence_len=128,
        num_classes=6,
        class_names=["walk", "jog", "sit", "stand", "upstairs", "downstairs"],
        scenarios=[
            ("7", "18"), ("20", "30"), ("35", "31"), ("17", "23"), ("6", "19"),
            ("2", "11"), ("33", "12"), ("5", "26"), ("28", "4"), ("23", "32"),
        ],
        drop_last=False,  # AdaTime: drop_last=False for WISDM
    ),
    "SSC": DatasetConfig(
        name="EEG",  # Folder name in AdaTime data directory
        input_channels=1,
        sequence_len=3000,
        num_classes=5,
        class_names=["W", "N1", "N2", "N3", "REM"],
        scenarios=[
            ("0", "11"), ("7", "18"), ("9", "14"), ("12", "5"), ("16", "1"),
            ("3", "19"), ("18", "12"), ("13", "17"), ("5", "15"), ("6", "2"),
        ],
        # AdaTime EEG-specific CNN params (configs/data_model_configs.py class EEG)
        kernel_size=25,
        stride=6,
        mid_channels=32,
        dropout=0.2,
    ),
    "MFD": DatasetConfig(
        name="FD",  # Folder name in AdaTime data directory
        input_channels=1,
        sequence_len=5120,
        num_classes=3,
        class_names=["Healthy", "D1", "D2"],
        scenarios=[
            ("0", "1"), ("0", "3"), ("1", "0"), ("1", "2"), ("1", "3"),
            ("2", "1"), ("2", "3"), ("3", "0"), ("3", "1"), ("3", "2"),
        ],
        # AdaTime FD-specific CNN params (configs/data_model_configs.py class FD)
        kernel_size=32,
        stride=6,
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for a named dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


# ═══════════════════════════════════════════════════════════════════════
#  Dataset class (converts AdaTime format to our pipeline format)
# ═══════════════════════════════════════════════════════════════════════

class AdaTimeDataset(Dataset):
    """Wraps an AdaTime domain's data for our translator pipeline.

    Converts from AdaTime format (N, channels, timesteps) to our format
    (N, timesteps, channels) and provides batch tuples:
        (data, labels, pad_mask, static_features)

    Since AdaTime samples are fixed-length, pad_mask is all-True (no padding)
    and static features are zeros (no static info in HAR).

    For long-sequence datasets (SSC=3000, MFD=5120), average-pooling
    downsampling is applied to reduce sequence length to max_seq_len.
    """

    def __init__(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
        config: DatasetConfig,
        normalize: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        """
        Args:
            samples: (N, channels, timesteps) float tensor
            labels: (N,) long tensor with class indices
            config: Dataset configuration
            normalize: Whether to normalize per-channel
            max_seq_len: If set and sequence_len > max_seq_len, downsample
                via average pooling to this length.
        """
        # Transpose: (N, C, T) -> (N, T, C) to match our pipeline
        if samples.dim() == 2:
            samples = samples.unsqueeze(1)
        if samples.shape[1] == config.input_channels:
            # channels-first -> timesteps-first
            self.data = samples.permute(0, 2, 1).float()  # (N, T, C)
        else:
            # Already in correct format
            self.data = samples.float()

        self.labels = labels.long()
        self.config = config
        self.num_features = config.input_channels

        # Downsample long sequences via average pooling
        original_len = self.data.shape[1]
        if max_seq_len is not None and original_len > max_seq_len:
            # Average pooling: (N, T, C) -> (N, C, T) -> pool -> (N, C, T') -> (N, T', C)
            pool_factor = original_len // max_seq_len
            actual_len = (original_len // pool_factor)
            # Trim to exact multiple
            trim_len = actual_len * pool_factor
            trimmed = self.data[:, :trim_len, :]  # (N, trim_len, C)
            # Reshape and average
            N, _, C = trimmed.shape
            self.data = trimmed.reshape(N, actual_len, pool_factor, C).mean(dim=2)  # (N, actual_len, C)
            self.seq_len = actual_len
            logger.info(
                "Downsampled %s: %d -> %d timesteps (pool_factor=%d)",
                config.name, original_len, self.seq_len, pool_factor,
            )
        else:
            self.seq_len = original_len

        # Per-channel normalization
        if normalize:
            # Compute mean/std over (N, T) for each channel
            mean = self.data.mean(dim=(0, 1), keepdim=True)  # (1, 1, C)
            std = self.data.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
            self.data = (self.data - mean) / std
            self._mean = mean.squeeze()
            self._std = std.squeeze()
        else:
            self._mean = torch.zeros(self.num_features)
            self._std = torch.ones(self.num_features)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]                    # (T, C)
        y = self.labels[idx]                   # scalar
        # Per-sequence label: expand to (T,) for compatibility with per-timestep pipeline
        # The label is the same for all timesteps (per-sequence classification)
        y_seq = y.expand(self.seq_len)         # (T,)
        # Pad mask: all True (no padding in fixed-length sequences)
        pad_mask = torch.ones(self.seq_len, dtype=torch.bool)  # (T,)
        # Static features: zeros (no static info)
        static = torch.zeros(4, dtype=torch.float32)  # (4,) matching our pipeline's static_dim
        return x, y_seq, pad_mask, static

    def get_class_label(self, idx):
        """Get the per-sequence class label (not expanded)."""
        return self.labels[idx].item()


# ═══════════════════════════════════════════════════════════════════════
#  Data loading functions
# ═══════════════════════════════════════════════════════════════════════

def load_domain_data(
    data_path: str,
    dataset_name: str,
    domain_id: str,
    split: str = "train",
    folder_name: Optional[str] = None,
) -> dict:
    """Load raw AdaTime domain data from .pt file.

    Args:
        data_path: Root path to datasets (e.g., /path/to/AdaTime/data)
        dataset_name: Dataset name (HAR, HHAR, WISDM, SSC, MFD)
        domain_id: Domain identifier (e.g., "2", "11")
        split: "train" or "test"
        folder_name: Override folder name (e.g., "EEG" for SSC, "FD" for MFD).
            If None, uses dataset_name.

    Returns:
        dict with "samples" and "labels" tensors
    """
    folder = folder_name or dataset_name
    file_path = os.path.join(data_path, folder, f"{split}_{domain_id}.pt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Download AdaTime datasets from: https://researchdata.ntu.edu.sg/\n"
            f"Expected structure: {data_path}/{folder}/{{train,test}}_{{domain_id}}.pt"
        )
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    logger.info(
        "Loaded %s/%s_%s.pt: samples=%s, labels=%s",
        folder, split, domain_id,
        tuple(data["samples"].shape) if isinstance(data["samples"], (torch.Tensor, np.ndarray)) else "?",
        tuple(data["labels"].shape) if isinstance(data["labels"], (torch.Tensor, np.ndarray)) else "?",
    )
    return data


def create_dataloaders(
    data_path: str,
    dataset_name: str,
    source_id: str,
    target_id: str,
    batch_size: int = 32,
    val_fraction: float = 0.0,
    seed: int = 42,
    normalize: bool = True,
    max_seq_len: Optional[int] = None,
    full_length: bool = False,
) -> dict:
    """Create train/val/test DataLoaders for a source->target scenario.

    Matches AdaTime's data loading exactly:
      - NO validation split carved from source training data. AdaTime trains
        on the full source train set (val_fraction=0.0 by default).
      - Source train uses dataset's drop_last setting (True for HAR/HHAR, False for WISDM).
      - Target train split kept for our translator training (still uses val_fraction for target).
      - Normalization: z-score per channel using (N, T) mean/std from each split independently.

    Args:
        data_path: Root directory containing dataset folders
        dataset_name: Logical dataset name (HAR, HHAR, WISDM, SSC, MFD)
        source_id: Source domain identifier
        target_id: Target domain identifier
        batch_size: Batch size for DataLoaders
        val_fraction: Fraction of TARGET training data for validation split.
            Source training data is always used in full (AdaTime convention).
            Set to 0.0 (default) to also use full target training data.
        seed: Random seed for train/val split
        normalize: Whether to normalize per-channel
        max_seq_len: If set, downsample sequences longer than this via avg pooling
        full_length: If True, ignore max_seq_len and return full-length sequences
            (no downsampling). Used by ChunkedAdaTimeCNNRetrievalTrainer which
            handles chunking internally.

    Returns dict with keys:
        - source_train: DataLoader for FULL source training data (no val split)
        - source_val: DataLoader for source test data (used as source validation)
        - target_train: DataLoader for target training data
        - target_val: DataLoader for target validation data (split from train if val_fraction>0)
        - target_test: DataLoader for target test data (evaluation)
    """
    config = get_dataset_config(dataset_name)
    # config.name is the folder name on disk (e.g., "EEG" for SSC, "FD" for MFD)
    folder_name = config.name

    # Load source domain
    src_train_raw = load_domain_data(data_path, dataset_name, source_id, "train", folder_name=folder_name)
    src_test_raw = load_domain_data(data_path, dataset_name, source_id, "test", folder_name=folder_name)

    # Load target domain
    trg_train_raw = load_domain_data(data_path, dataset_name, target_id, "train", folder_name=folder_name)
    trg_test_raw = load_domain_data(data_path, dataset_name, target_id, "test", folder_name=folder_name)

    # Convert numpy to tensor if needed
    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    # When full_length=True, bypass max_seq_len downsampling
    effective_max_seq_len = None if full_length else max_seq_len

    # Create full source dataset (NOT split — AdaTime uses full train set)
    src_full = AdaTimeDataset(
        _to_tensor(src_train_raw["samples"]),
        _to_tensor(src_train_raw["labels"]),
        config, normalize=normalize, max_seq_len=effective_max_seq_len,
    )
    # Source test set (used as "val" for our purposes, but not for CNN training)
    src_test_ds = AdaTimeDataset(
        _to_tensor(src_test_raw["samples"]),
        _to_tensor(src_test_raw["labels"]),
        config, normalize=normalize, max_seq_len=effective_max_seq_len,
    )
    # Full target train dataset
    trg_full = AdaTimeDataset(
        _to_tensor(trg_train_raw["samples"]),
        _to_tensor(trg_train_raw["labels"]),
        config, normalize=normalize, max_seq_len=effective_max_seq_len,
    )
    trg_test = AdaTimeDataset(
        _to_tensor(trg_test_raw["samples"]),
        _to_tensor(trg_test_raw["labels"]),
        config, normalize=normalize, max_seq_len=effective_max_seq_len,
    )

    # Source: use FULL training set (no val split) — matches AdaTime
    src_train_ds = src_full
    src_val_ds = src_test_ds  # Source val = source test (for reference only)

    # Target: optionally split into train/val for translator training
    generator = torch.Generator().manual_seed(seed)
    if val_fraction > 0.0:
        trg_val_size = max(1, int(len(trg_full) * val_fraction))
        trg_train_size = len(trg_full) - trg_val_size
        trg_train_ds, trg_val_ds = random_split(trg_full, [trg_train_size, trg_val_size], generator=generator)
    else:
        trg_train_ds = trg_full
        trg_val_ds = trg_test  # Use test as val when no split

    logger.info(
        "Scenario %s->%s: src_train=%d (full, no val split), trg_train=%d, trg_val=%d, trg_test=%d",
        source_id, target_id,
        len(src_train_ds),
        len(trg_train_ds), len(trg_val_ds), len(trg_test),
    )

    # Effective sequence length (may differ from config.sequence_len if downsampled)
    effective_seq_len = src_full.seq_len

    # Use dataset's drop_last setting for training loaders (matches AdaTime per-dataset config)
    src_drop_last = config.drop_last

    return {
        "source_train": DataLoader(src_train_ds, batch_size=batch_size, shuffle=True, drop_last=src_drop_last, num_workers=0),
        "source_val": DataLoader(src_val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0),
        "target_train": DataLoader(trg_train_ds, batch_size=batch_size, shuffle=True, drop_last=config.drop_last, num_workers=0),
        "target_val": DataLoader(trg_val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0),
        "target_test": DataLoader(trg_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0),
        "config": config,
        "effective_seq_len": effective_seq_len,
        "source_normalization": {"mean": src_full._mean, "std": src_full._std},
        "target_normalization": {"mean": trg_full._mean, "std": trg_full._std},
    }
