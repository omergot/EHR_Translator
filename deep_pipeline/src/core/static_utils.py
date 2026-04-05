
from __future__ import annotations

from pathlib import Path
import logging
from typing import Iterable, Tuple

import polars as pl
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from icu_benchmarks.data.preprocessor import restore_recipe
from recipys.ingredients import Ingredients


class StaticAugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset, static_matrix: torch.Tensor) -> None:
        self.dataset = dataset
        self.static_matrix = static_matrix
        for attr in ("get_feature_names", "vars", "outcome_df", "row_indicators"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        data, labels, mask = self.dataset[idx]
        return data, labels, mask, self.static_matrix[idx]


def _resolve_dataset_indices(dataset: Dataset) -> tuple[Dataset, list[int]]:
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


def load_static_with_recipe(
    *,
    data_dir: Path,
    file_names: dict,
    group_col: str,
    static_features: Iterable[str],
    recipe_path: Path,
) -> pl.DataFrame:
    static_path = data_dir / file_names["STATIC"]
    if not static_path.exists():
        raise FileNotFoundError(f"Static parquet not found: {static_path}")
    raw_df = pl.read_parquet(static_path)
    for col in ("age", "height", "weight"):
        if col in raw_df.columns:
            stats = raw_df.select(
                pl.col(col).mean().alias("mean"),
                pl.col(col).std(ddof=0).alias("std"),
            ).row(0)
            logging.info("[static raw] %s mean=%.6f std=%.6f", col, stats[0], stats[1])
    recipe = restore_recipe(recipe_path)
    if not hasattr(recipe, "data") and hasattr(recipe, "roles"):
        recipe.data = Ingredients(raw_df, roles=recipe.roles, check_roles=False)
    baked = recipe.bake(raw_df)
    if isinstance(baked, pd.DataFrame):
        baked = pl.from_pandas(baked)
    if group_col not in baked.columns:
        if group_col in raw_df.columns:
            baked = baked.with_columns(raw_df[group_col])
        else:
            raise ValueError(f"Group column '{group_col}' missing in static data")

    static_features = list(static_features)
    missing = [col for col in static_features if col not in baked.columns]
    if missing:
        raise ValueError(f"Static recipe missing columns: {missing}")

    baked = baked.select([group_col] + static_features)
    return baked


def build_static_matrix_for_dataset(
    dataset: Dataset,
    static_df: pl.DataFrame,
    group_col: str,
    static_features: Iterable[str],
) -> torch.Tensor:
    base_dataset, index_map = _resolve_dataset_indices(dataset)
    if not hasattr(base_dataset, "outcome_df"):
        raise ValueError("Dataset missing outcome_df; cannot align static features")

    stay_ids_list = base_dataset.outcome_df[group_col].unique().to_list()
    try:
        stay_ids = [stay_ids_list[idx] for idx in index_map]
    except IndexError:
        logging.warning(
            "Index mismatch in static feature alignment: index_map max=%d but %d unique stay IDs. "
            "Skipping static feature augmentation.",
            max(index_map) if index_map else -1, len(stay_ids_list),
        )
        return None
    order_df = pl.DataFrame({group_col: stay_ids, "_order": list(range(len(stay_ids)))})

    if static_df[group_col].dtype != order_df[group_col].dtype:
        static_df = static_df.with_columns(pl.col(group_col).cast(order_df[group_col].dtype))

    static_df = static_df.group_by(group_col).last()
    joined = order_df.join(static_df, on=group_col, how="left").sort("_order")

    static_features = list(static_features)
    null_mask = joined.select(
        pl.any_horizontal([pl.col(col).is_null() for col in static_features]).alias("missing")
    )["missing"]
    if null_mask.sum() > 0:
        missing_count = int(null_mask.sum())
        raise ValueError(f"Static features missing for {missing_count} stays")

    matrix = joined.select(static_features).to_numpy().astype("float32")
    return torch.from_numpy(matrix)
