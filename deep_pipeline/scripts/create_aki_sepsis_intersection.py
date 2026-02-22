#!/usr/bin/env python3
"""Create intersection dataset: AKI eICU stays that also exist in sepsis eICU.

Filters sepsis eICU parquets (dyn, outc, sta) to only include stays
present in both AKI and sepsis cohorts. Output is written to a new directory
that can be used as data_dir for evaluating AKI-trained translators on sepsis.
"""

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Create AKI-sepsis intersection dataset")
    parser.add_argument(
        "--aki-data-dir",
        type=str,
        default="/bigdata/omerg/Thesis/cohort_data/aki/eicu",
        help="Path to AKI eICU cohort data",
    )
    parser.add_argument(
        "--sepsis-data-dir",
        type=str,
        default="/bigdata/omerg/Thesis/cohort_data/sepsis/eicu",
        help="Path to sepsis eICU cohort data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/bigdata/omerg/Thesis/cohort_data/sepsis/eicu_aki_intersection",
        help="Output directory for filtered parquets",
    )
    args = parser.parse_args()

    aki_dir = Path(args.aki_data_dir)
    sepsis_dir = Path(args.sepsis_data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load stay IDs from both cohorts
    aki_stays = set(
        pl.read_parquet(aki_dir / "outc.parquet", columns=["stay_id"])["stay_id"]
        .unique()
        .to_list()
    )
    sepsis_stays = set(
        pl.read_parquet(sepsis_dir / "outc.parquet", columns=["stay_id"])["stay_id"]
        .unique()
        .to_list()
    )

    intersection = sorted(aki_stays & sepsis_stays)
    log.info(
        "AKI stays: %d, Sepsis stays: %d, Intersection: %d",
        len(aki_stays), len(sepsis_stays), len(intersection),
    )

    # Filter sepsis parquets to intersection
    for fname in ["dyn.parquet", "outc.parquet", "sta.parquet"]:
        src_path = sepsis_dir / fname
        if not src_path.exists():
            log.warning("Skipping %s (not found)", src_path)
            continue
        df = pl.read_parquet(src_path)
        original_rows = len(df)
        df_filtered = df.filter(pl.col("stay_id").is_in(intersection))
        filtered_rows = len(df_filtered)
        n_stays = df_filtered["stay_id"].n_unique()
        df_filtered.write_parquet(out_dir / fname)
        log.info(
            "%s: %d → %d rows (%d stays)",
            fname, original_rows, filtered_rows, n_stays,
        )

    # Copy preproc directory if it exists (needed for static recipe)
    preproc_src = sepsis_dir / "preproc"
    preproc_dst = out_dir / "preproc"
    if preproc_src.exists() and not preproc_dst.exists():
        import shutil
        shutil.copytree(preproc_src, preproc_dst)
        log.info("Copied preproc/ directory")

    # Print label statistics for the intersection
    outc = pl.read_parquet(out_dir / "outc.parquet")
    n_total_ts = len(outc)
    n_pos_ts = int(outc["label"].sum())
    n_stays = outc["stay_id"].n_unique()
    pos_stays = outc.group_by("stay_id").agg(pl.col("label").max()).filter(pl.col("label") >= 1)
    n_pos_stays = len(pos_stays)
    log.info(
        "Intersection stats: %d stays, %d pos stays (%.1f%%), "
        "%d timesteps, %d pos timesteps (%.2f%%)",
        n_stays, n_pos_stays, 100 * n_pos_stays / n_stays,
        n_total_ts, n_pos_ts, 100 * n_pos_ts / n_total_ts,
    )


if __name__ == "__main__":
    main()
