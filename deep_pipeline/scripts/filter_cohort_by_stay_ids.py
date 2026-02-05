#!/usr/bin/env python3
import argparse
from pathlib import Path

import polars as pl


def _parse_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _load_unique_ids(path: Path, col: str) -> set[int]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        ids = (
            pl.scan_parquet(path)
            .select(col)
            .unique()
            .collect()
            .get_column(col)
            .to_list()
        )
        return {int(x) for x in ids if x is not None}
    if suffix == ".csv":
        ids = (
            pl.scan_csv(path)
            .select(col)
            .unique()
            .collect()
            .get_column(col)
            .to_list()
        )
        return {int(x) for x in ids if x is not None}
    if suffix == ".txt":
        ids: set[int] = set()
        for line in path.read_text().splitlines():
            parsed = _parse_int(line)
            if parsed is not None:
                ids.add(parsed)
        return ids
    raise ValueError(f"Unsupported ids file type: {path.suffix}")


def _filter_and_write(src_path: Path, out_path: Path, col: str, stay_ids: list[int]) -> int:
    df = (
        pl.scan_parquet(src_path)
        .filter(pl.col(col).is_in(stay_ids))
        .collect()
    )
    df.write_parquet(out_path)
    return df.height


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter cohort parquet files by stay_id list."
    )
    parser.add_argument(
        "--ids-file",
        required=True,
        type=Path,
        help="Parquet/CSV/TXT file containing stay_id values.",
    )
    parser.add_argument(
        "--ids-col",
        default="stay_id",
        help="Column name for stay_id in ids parquet/csv.",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        help="Source cohort directory containing dyn/sta/outc parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory to write filtered parquet files.",
    )
    parser.add_argument(
        "--compare-file",
        type=Path,
        help="Second ids file (parquet/csv/txt) for report-only intersection.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only report intersection stats between --ids-file and --compare-file.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--intersection",
        action="store_true",
        help="Keep only stay_ids present in both the ids file and source cohort (default).",
    )
    mode_group.add_argument(
        "--no-intersection",
        action="store_true",
        help="Select stay_ids present in ids file but missing from source cohort.",
    )
    args = parser.parse_args()

    ids_file = args.ids_file
    src_dir = args.src_dir
    out_dir = args.out_dir
    stay_col = args.ids_col

    if args.report_only:
        if args.compare_file is None:
            raise ValueError("--compare-file is required when using --report-only.")
        for p in (ids_file, args.compare_file):
            if not p.exists():
                raise FileNotFoundError(f"Missing file: {p}")

        ids_set = _load_unique_ids(ids_file, stay_col)
        compare_set = _load_unique_ids(args.compare_file, stay_col)
        mutual = sorted(ids_set & compare_set)

        count_ids = len(ids_set)
        count_compare = len(compare_set)
        count_mutual = len(mutual)
        pct_of_ids = (count_mutual / count_ids * 100) if count_ids else 0.0
        pct_of_compare = (count_mutual / count_compare * 100) if count_compare else 0.0

        print(f"stay_ids in ids file: {count_ids}")
        print(f"stay_ids in compare file: {count_compare}")
        print(f"intersection stay_ids: {count_mutual}")
        print(f"intersection % of ids file: {pct_of_ids:.2f}%")
        print(f"intersection % of compare file: {pct_of_compare:.2f}%")
        return

    if src_dir is None or out_dir is None:
        raise ValueError("--src-dir and --out-dir are required unless --report-only is used.")

    dyn_path = src_dir / "dyn.parquet"
    sta_path = src_dir / "sta.parquet"
    outc_path = src_dir / "outc.parquet"

    for p in (ids_file, dyn_path, sta_path, outc_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)

    ids_set = _load_unique_ids(ids_file, stay_col)
    src_ids = _load_unique_ids(outc_path, stay_col)
    if args.no_intersection:
        mutual = sorted(src_ids - ids_set)
        mode_label = "src_only"
    else:
        mutual = sorted(src_ids & ids_set)
        mode_label = "intersection"

    count_ids = len(ids_set)
    count_src = len(src_ids)
    count_mutual = len(mutual)
    pct_of_ids = (count_mutual / count_ids * 100) if count_ids else 0.0
    pct_of_src = (count_mutual / count_src * 100) if count_src else 0.0

    print(f"stay_ids in ids file: {count_ids}")
    print(f"stay_ids in source cohort: {count_src}")
    print(f"{mode_label} stay_ids: {count_mutual}")
    print(f"{mode_label} % of ids file: {pct_of_ids:.2f}%")
    print(f"{mode_label} % of source cohort: {pct_of_src:.2f}%")

    if not mutual:
        raise RuntimeError("No stay_ids found for selected mode; nothing to write.")

    rows_dyn = _filter_and_write(dyn_path, out_dir / "dyn.parquet", stay_col, mutual)
    rows_sta = _filter_and_write(sta_path, out_dir / "sta.parquet", stay_col, mutual)
    rows_outc = _filter_and_write(outc_path, out_dir / "outc.parquet", stay_col, mutual)

    print(f"written rows: dyn={rows_dyn} sta={rows_sta} outc={rows_outc}")


if __name__ == "__main__":
    main()
