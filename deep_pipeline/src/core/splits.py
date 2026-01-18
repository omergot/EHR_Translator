import logging
import sys
from pathlib import Path
from typing import Dict

import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

yaib_path = Path(__file__).parent.parent.parent.parent.parent / "YAIB"
if not yaib_path.exists():
    yaib_path = Path(__file__).parent.parent.parent.parent.parent.parent / "YAIB"
sys.path.insert(0, str(yaib_path))

from icu_benchmarks.data.constants import DataSegment, DataSplit


def split_by_stay_id(
    data: Dict[str, pl.DataFrame],
    vars: Dict[str, str],
    train_size: float = 0.8,
    seed: int = 42,
    runmode: str = "classification",
) -> Dict[str, Dict[str, pl.DataFrame]]:
    group_col = vars["GROUP"]
    label_col = vars["LABEL"]
    
    outcome_df = data[DataSegment.outcome]
    stays = outcome_df[group_col].unique().to_list()
    
    if runmode == "classification" and label_col in outcome_df.columns:
        labels = outcome_df.group_by(group_col).agg(pl.col(label_col).max())[label_col].to_list()
        splitter = StratifiedShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
        train_indices, val_indices = list(splitter.split(stays, labels))[0]
    else:
        splitter = ShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
        train_indices, val_indices = list(splitter.split(stays))[0]
    
    train_stays = [stays[i] for i in train_indices]
    val_stays = [stays[i] for i in val_indices]
    
    data_split = {}
    for split_name, stay_list in [(DataSplit.train, train_stays), (DataSplit.val, val_stays)]:
        data_split[split_name] = {}
        stay_df = pl.DataFrame({group_col: stay_list})
        
        for segment_name, segment_df in data.items():
            data_split[split_name][segment_name] = segment_df.join(
                stay_df, on=group_col, how="inner"
            )
    
    data_split[DataSplit.test] = data_split[DataSplit.val].copy()
    
    logging.info(f"Split data: train={len(train_stays)} stays, val={len(val_stays)} stays")
    return data_split


