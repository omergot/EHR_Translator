#!/usr/bin/env python3
"""
Create a filtered eICU sepsis cohort with AKI-like label density.

The original eICU sepsis cohort has ~123k stays with only 4.57% per-stay positive
rate and 1.13% per-timestep positive rate. This makes sepsis translation extremely
difficult due to weak task signal (gradient bottleneck).

This script creates a filtered cohort by:
- Keeping ALL 5,639 positive stays (any stay with at least one label=1 timestep)
- Randomly sampling 7,300 negative stays (seed=2222)
- Target: ~12,939 stays, ~43.6% per-stay positive, ~12% per-TS positive
  (matching AKI label density where shared latent translation works)

Usage:
    python scripts/create_filtered_sepsis_cohort.py
"""

import logging
import shutil
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
SRC_DIR = Path("/bigdata/omerg/Thesis/cohort_data/sepsis/eicu")
DST_DIR = Path("/bigdata/omerg/Thesis/cohort_data/sepsis/eicu_filtered_aki_density")

# Parameters
SEED = 2222
N_NEGATIVE_SAMPLE = 7_300


def main():
    logger.info("=" * 70)
    logger.info("Creating filtered eICU sepsis cohort with AKI-like label density")
    logger.info("=" * 70)

    # ---- Step 1: Read outcome data and identify positive/negative stays ----
    logger.info(f"Reading {SRC_DIR / 'outc.parquet'}")
    outc = pl.read_parquet(SRC_DIR / "outc.parquet")
    logger.info(f"  Original outc shape: {outc.shape}")
    logger.info(f"  Original stays: {outc['stay_id'].n_unique()}")

    # Positive stays: any row with label > 0
    positive_stay_ids = outc.filter(pl.col("label") > 0)["stay_id"].unique().sort()
    n_positive = len(positive_stay_ids)
    logger.info(f"  Positive stays (any label>0): {n_positive}")

    # Negative stays: all labels = 0 for the stay
    all_stay_ids = outc["stay_id"].unique().sort()
    negative_stay_ids = all_stay_ids.filter(~all_stay_ids.is_in(positive_stay_ids))
    n_negative = len(negative_stay_ids)
    logger.info(f"  Negative stays (all label=0): {n_negative}")

    assert n_positive == 5_639, f"Expected 5,639 positive stays, got {n_positive}"

    # ---- Step 2: Sample negative stays ----
    rng = np.random.RandomState(SEED)
    sampled_neg_indices = rng.choice(n_negative, size=N_NEGATIVE_SAMPLE, replace=False)
    sampled_neg_ids = negative_stay_ids.gather(sampled_neg_indices)
    logger.info(f"  Sampled {N_NEGATIVE_SAMPLE} negative stays (seed={SEED})")

    # Combined stay IDs
    keep_ids = pl.concat([positive_stay_ids, sampled_neg_ids]).unique().sort()
    logger.info(f"  Total stays to keep: {len(keep_ids)}")

    # ---- Step 3: Create output directory ----
    DST_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Output directory: {DST_DIR}")

    # ---- Step 4: Filter and save outc.parquet ----
    outc_filtered = outc.filter(pl.col("stay_id").is_in(keep_ids))
    outc_filtered.write_parquet(DST_DIR / "outc.parquet")
    logger.info(f"  Saved outc.parquet: {outc_filtered.shape}")

    # ---- Step 5: Filter and save dyn.parquet ----
    logger.info(f"Reading {SRC_DIR / 'dyn.parquet'}")
    dyn = pl.read_parquet(SRC_DIR / "dyn.parquet")
    dyn_filtered = dyn.filter(pl.col("stay_id").is_in(keep_ids))
    dyn_filtered.write_parquet(DST_DIR / "dyn.parquet")
    logger.info(f"  Saved dyn.parquet: {dyn_filtered.shape} (from {dyn.shape})")

    # ---- Step 6: Filter and save sta.parquet ----
    logger.info(f"Reading {SRC_DIR / 'sta.parquet'}")
    sta = pl.read_parquet(SRC_DIR / "sta.parquet")
    sta_filtered = sta.filter(pl.col("stay_id").is_in(keep_ids))
    sta_filtered.write_parquet(DST_DIR / "sta.parquet")
    logger.info(f"  Saved sta.parquet: {sta_filtered.shape} (from {sta.shape})")

    # ---- Step 7: Copy preproc directory ----
    src_preproc = SRC_DIR / "preproc"
    dst_preproc = DST_DIR / "preproc"
    if dst_preproc.exists():
        shutil.rmtree(dst_preproc)
    shutil.copytree(src_preproc, dst_preproc)
    logger.info(f"  Copied preproc directory: {src_preproc} -> {dst_preproc}")

    # ---- Step 8: Verification ----
    logger.info("")
    logger.info("=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)

    total_stays = outc_filtered["stay_id"].n_unique()
    pos_stays_verify = outc_filtered.filter(pl.col("label") > 0)["stay_id"].unique()
    neg_stays_verify = total_stays - len(pos_stays_verify)

    total_timesteps = len(outc_filtered)
    positive_timesteps = outc_filtered["label"].sum()
    per_ts_pos_rate = positive_timesteps / total_timesteps * 100
    per_stay_pos_rate = len(pos_stays_verify) / total_stays * 100

    logger.info(f"  Total stays:              {total_stays}")
    logger.info(f"  Positive stays:           {len(pos_stays_verify)}")
    logger.info(f"  Negative stays:           {neg_stays_verify}")
    logger.info(f"  Total timesteps:          {total_timesteps}")
    logger.info(f"  Positive timesteps:       {positive_timesteps}")
    logger.info(f"  Per-timestep pos rate:    {per_ts_pos_rate:.2f}%")
    logger.info(f"  Per-stay pos rate:        {per_stay_pos_rate:.2f}%")
    logger.info("")
    logger.info("  TARGET COMPARISON:")
    logger.info(f"    Stays:      {total_stays} / 12,939 target")
    logger.info(f"    Per-stay:   {per_stay_pos_rate:.1f}% / 43.6% target")
    logger.info(f"    Per-TS:     {per_ts_pos_rate:.2f}% / ~12% target")
    logger.info("")

    # Compare to original
    orig_ts_rate = 72649 / 6416789 * 100
    orig_stay_rate = 5639 / 123412 * 100
    logger.info("  ORIGINAL vs FILTERED:")
    logger.info(f"    Stays:      123,412 -> {total_stays} ({total_stays/123412*100:.1f}%)")
    logger.info(f"    Per-stay:   {orig_stay_rate:.2f}% -> {per_stay_pos_rate:.2f}% ({per_stay_pos_rate/orig_stay_rate:.1f}x)")
    logger.info(f"    Per-TS:     {orig_ts_rate:.2f}% -> {per_ts_pos_rate:.2f}% ({per_ts_pos_rate/orig_ts_rate:.1f}x)")
    logger.info("")
    logger.info("Done!")


if __name__ == "__main__":
    main()
