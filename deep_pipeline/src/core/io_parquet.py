import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import torch

yaib_path = Path(__file__).parent.parent.parent.parent.parent / "YAIB"
if not yaib_path.exists():
    yaib_path = Path(__file__).parent.parent.parent.parent.parent.parent / "YAIB"
sys.path.insert(0, str(yaib_path))

from icu_benchmarks.data.constants import DataSegment, DataSplit


def write_translated_parquet(
    translated_dataframes: List[pl.DataFrame],
    output_path: Path,
    vars: Dict[str, str],
    file_names: Dict[str, str],
):
    if not translated_dataframes:
        raise ValueError("No translated data to write")
    
    combined_df = pl.concat(translated_dataframes)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    features_file = output_path / file_names.get("DYNAMIC", "dyn.parquet")
    combined_df.write_parquet(features_file)
    logging.info(f"Written translated features to {features_file}")
    
    return features_file


def reconstruct_parquet_from_batches(
    batches: List[tuple],
    translated_batches: List[torch.Tensor],
    stay_id_batches: List[Optional[torch.Tensor]],
    vars: Dict[str, str],
    feature_names: List[str],
    time_batches: Optional[List[Optional[List[List[float]]]]] = None,
) -> pl.DataFrame:
    all_rows = []
    
    for batch_idx, (batch, translated, stay_ids) in enumerate(zip(batches, translated_batches, stay_id_batches)):
        data, labels, mask = batch[0], batch[1], batch[2]
        translated_np = translated.detach().cpu().numpy()
        mask_np = mask.cpu().numpy()
        time_rows = None
        if time_batches is not None and batch_idx < len(time_batches):
            time_rows = time_batches[batch_idx]
        
        batch_size, seq_len, num_features = translated_np.shape
        
        for b in range(batch_size):
            stay_id = stay_ids[b].item() if stay_ids is not None else (batch_idx * batch_size + b)
            time_seq = None
            if time_rows is not None and b < len(time_rows):
                time_seq = time_rows[b]
            for t in range(seq_len):
                if mask_np[b, t]:
                    row = {vars["GROUP"]: stay_id}
                    if vars.get("SEQUENCE"):
                        if time_seq is not None and t < len(time_seq):
                            row[vars["SEQUENCE"]] = time_seq[t]
                        else:
                            row[vars["SEQUENCE"]] = t
                    for f_idx, f_name in enumerate(feature_names):
                        if f_idx < num_features:
                            row[f_name] = float(translated_np[b, t, f_idx])
                    all_rows.append(row)
    
    return pl.DataFrame(all_rows)
