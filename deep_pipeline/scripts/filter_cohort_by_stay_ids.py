#!/usr/bin/env python3
import argparse
from pathlib import Path

import polars as pl


def _load_unique_ids(path: Path, col: str) -> set[int]:
    ids = (
        pl.scan_parquet(path)
        .select(col)
        .unique()
        .collect()
        .get_column(col)
        .to_list()
    )
    return {int(x) for x in ids if x is not None}


def _filter_and_write(src_path: Path, out_path: Path, col: str, stay_ids: list[int]) -> int:
    df = (
        pl.scan_parquet(src_path)
        .filter(pl.col(col).is_in(stay_ids))
        .collect()
    )
    df.write_parquet(out_path)
    return df.height


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter cohort parquet files by stay_id list.")
    parser.add_argument(
        "--ids-parquet",
        required=True,
        type=Path,
        help="Parquet file containing stay_id column (e.g., translated output).",
    )
    parser.add_argument(
        "--ids-col",
        default="stay_id",
        help="Column name for stay_id in ids parquet.",
    )
    parser.add_argument(
        "--src-dir",
        required=True,
        type=Path,
        help="Source cohort directory containing dyn/sta/outc parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory to write filtered parquet files.",
    )
    args = parser.parse_args()

    ids_parquet = args.ids_parquet
    src_dir = args.src_dir
    out_dir = args.out_dir
    stay_col = args.ids_col

    dyn_path = src_dir / "dyn.parquet"
    sta_path = src_dir / "sta.parquet"
    outc_path = src_dir / "outc.parquet"

    for p in (ids_parquet, dyn_path, sta_path, outc_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)

    ids_set = _load_unique_ids(ids_parquet, stay_col)
    src_ids = _load_unique_ids(outc_path, stay_col)
    mutual = sorted(src_ids & ids_set)

    count_ids = len(ids_set)
    count_src = len(src_ids)
    count_mutual = len(mutual)
    pct_of_ids = (count_mutual / count_ids * 100) if count_ids else 0.0
    pct_of_src = (count_mutual / count_src * 100) if count_src else 0.0

    print(f"stay_ids in ids parquet: {count_ids}")
    print(f"stay_ids in source cohort: {count_src}")
    print(f"mutual stay_ids (used for testing): {count_mutual}")
    print(f"mutual % of ids parquet: {pct_of_ids:.2f}%")
    print(f"mutual % of source cohort: {pct_of_src:.2f}%")

    if not mutual:
        raise RuntimeError("No mutual stay_ids found; nothing to write.")

    rows_dyn = _filter_and_write(dyn_path, out_dir / "dyn.parquet", stay_col, mutual)
    rows_sta = _filter_and_write(sta_path, out_dir / "sta.parquet", stay_col, mutual)
    rows_outc = _filter_and_write(outc_path, out_dir / "outc.parquet", stay_col, mutual)

    print(f"written rows: dyn={rows_dyn} sta={rows_sta} outc={rows_outc}")


if __name__ == "__main__":
    main()
