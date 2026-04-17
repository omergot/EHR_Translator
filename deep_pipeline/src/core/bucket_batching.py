"""Bucket batching for variable-length sequences.

Groups sequences of similar length into batches, then truncates padding to
per-batch max. Reduces memory/compute from O(global_maxlen²) to O(batch_maxlen²)
for attention layers.
"""

import logging
import random
from typing import Optional

import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate


def compute_sequence_lengths(dataset) -> list[int]:
    """Compute actual (non-padded) length for every sample in a dataset.

    Works with PredictionPolarsDataset, StaticAugmentedDataset,
    _CachedSubsetDataset, and Subset wrappers.

    Returns list of int lengths (one per sample).

    Note: item[2] from YAIB is a *label* mask, not a *pad* mask.  For
    per-timestep tasks (AKI, Sepsis, eICU-LoS) every valid timestep has
    a label, so label_mask.sum() == actual length and the two are
    interchangeable.  For HiRID-LoS (and potentially other regression
    tasks) labels exist only at the *last* valid timestep, so
    label_mask.sum() == 1 regardless of actual length.

    We therefore compute length from the *data* tensor: the position of
    the last non-zero row.  This is robust to both pad-mask and
    label-mask conventions.
    """
    lengths = []
    for i in range(len(dataset)):
        item = dataset[i]
        data = item[0]  # (T, F) dynamic features
        if isinstance(data, torch.Tensor):
            # Find last timestep with any non-zero feature value.
            # YAIB right-pads with zeros, so trailing all-zero rows are padding.
            nonzero_mask = data.abs().sum(dim=-1) > 0  # (T,) bool
            if nonzero_mask.any():
                last_valid = int(nonzero_mask.nonzero(as_tuple=False)[-1].item()) + 1
                lengths.append(last_valid)
            else:
                lengths.append(1)  # safety: at least 1
        else:
            # Fallback for non-tensor data
            lengths.append(data.shape[0] if hasattr(data, 'shape') else 1)
    return lengths


class BucketBatchSampler(Sampler):
    """BatchSampler that groups sequences by length for efficient padding.

    Algorithm:
      1. Sample indices (with optional oversampling via weights)
      2. Sort within pools of pool_factor * batch_size by sequence length
      3. Group into batches of batch_size
      4. Shuffle batch order

    Compatible with WeightedRandomSampler (oversampling) — pass sample_weights
    to combine oversampling with length-bucketing in a single sampler.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        drop_last: bool = True,
        sample_weights: Optional[list[float]] = None,
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        pool_factor: int = 100,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sample_weights = sample_weights
        self.num_samples = num_samples or len(lengths)
        self.shuffle = shuffle
        self.pool_factor = pool_factor

    def __iter__(self):
        # Step 1: Get indices (with optional oversampling)
        if self.sample_weights is not None:
            indices = list(WeightedRandomSampler(
                self.sample_weights, self.num_samples, replacement=True,
            ))
        elif self.shuffle:
            indices = list(range(len(self.lengths)))
            random.shuffle(indices)
        else:
            indices = list(range(len(self.lengths)))

        # Step 2: Sort within pools by length
        pool_size = self.batch_size * self.pool_factor
        batches = []
        for i in range(0, len(indices), pool_size):
            pool = indices[i:i + pool_size]
            pool.sort(key=lambda idx: self.lengths[idx])
            for j in range(0, len(pool), self.batch_size):
                batch = pool[j:j + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # Step 3: Shuffle batch order
        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def variable_length_collate(batch):
    """Collate function that truncates sequences to max actual length in batch.

    Expects items as (data, labels, mask[, static]) tuples where:
      data: (T, F) — dynamic features
      labels: (T,) — per-timestep labels
      mask: (T,) — label mask (1=has label, 0=no label)
      static: (S,) — static features (optional, not truncated)

    Actual sequence length is derived from data (last non-zero row), not
    from the mask, because the mask may be a sparse label mask (e.g.
    HiRID LoS regression has only 1 label per stay at the last timestep).
    """
    # Find max actual length from data (last non-zero row)
    actual_lengths = []
    for item in batch:
        data = item[0]
        if isinstance(data, torch.Tensor):
            nonzero = data.abs().sum(dim=-1) > 0
            if nonzero.any():
                actual_lengths.append(int(nonzero.nonzero(as_tuple=False)[-1].item()) + 1)
            else:
                actual_lengths.append(1)
        else:
            actual_lengths.append(data.shape[0] if hasattr(data, 'shape') else 1)
    max_len = max(actual_lengths)

    if max_len <= 0:
        max_len = 1  # Safety: at least 1 timestep

    # Truncate time-dimension tensors (first 3 elements), keep static as-is
    truncated = []
    for item in batch:
        new_item = []
        for k, t in enumerate(item):
            if k < 3:  # data, labels, mask — have time dimension
                new_item.append(t[:max_len])
            else:  # static — no time dimension
                new_item.append(t)
        truncated.append(tuple(new_item))

    return default_collate(truncated)


def apply_bucket_batching(
    loader: DataLoader,
    batch_size: int,
    oversampling_factor: float = 0,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Replace a DataLoader with one using bucket batching + variable-length collation.

    Args:
        loader: Original DataLoader (its dataset is reused).
        batch_size: Batch size.
        oversampling_factor: If > 0, apply weighted oversampling for positive samples.
        shuffle: Whether to shuffle batch order.
        drop_last: Whether to drop last incomplete batch.

    Returns:
        New DataLoader with BucketBatchSampler and variable_length_collate.
    """
    dataset = loader.dataset
    logging.info("Computing sequence lengths for bucket batching (%d samples)...", len(dataset))
    lengths = compute_sequence_lengths(dataset)

    if lengths:
        sorted_lens = sorted(lengths)
        logging.info(
            "Sequence lengths: min=%d, p25=%d, median=%d, p75=%d, p95=%d, max=%d, mean=%.1f",
            sorted_lens[0],
            sorted_lens[len(sorted_lens) // 4],
            sorted_lens[len(sorted_lens) // 2],
            sorted_lens[3 * len(sorted_lens) // 4],
            sorted_lens[int(0.95 * len(sorted_lens))],
            sorted_lens[-1],
            sum(sorted_lens) / len(sorted_lens),
        )

    # Optionally compute oversampling weights
    sample_weights = None
    if oversampling_factor > 0:
        labels = []
        for i in range(len(dataset)):
            stay_labels = dataset[i][1]  # (data, labels, mask[, static])
            labels.append(int(stay_labels.max() >= 1))
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        sample_weights = [oversampling_factor if lbl == 1 else 1.0 for lbl in labels]
        logging.info(
            "Bucket batching + oversampling: factor=%.1f, %d pos / %d neg, eff_pos_rate=%.1f%%",
            oversampling_factor, n_pos, n_neg,
            100 * oversampling_factor * n_pos / (oversampling_factor * n_pos + n_neg),
        )

    sampler = BucketBatchSampler(
        lengths,
        batch_size=batch_size,
        drop_last=drop_last,
        sample_weights=sample_weights,
        shuffle=shuffle,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=loader.num_workers,
        collate_fn=variable_length_collate,
        pin_memory=True,
        # persistent_workers disabled: batch_sampler + persistent_workers has a bug in
        # PyTorch 2.4 (local, cu121) where worker iterator doesn't reset between epochs,
        # causing stale indices / data corruption that compounds over 3500+ batches/epoch
        # (AKI/Sepsis VLB). Not an issue on PyTorch 2.6 (3090, cu118). Mortality passes
        # because it has ~35 batches/epoch and the corruption doesn't accumulate enough.
        persistent_workers=False,
    )
