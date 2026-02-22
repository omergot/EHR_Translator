#!/usr/bin/env python3
"""
Create a filtered AKI-sepsis intersection dataset that matches AKI's per-timestep
positive rate (~12%).

The original intersection has ~1.15% per-timestep positive rate (sepsis labels).
This script filters out negative stays (keeping all positive stays) until the
per-timestep positive rate reaches ~12%, matching AKI's label density.

Strategy:
  1. Load outc.parquet, compute per-stay statistics
  2. Keep ALL positive stays (stays with at least one positive timestep)
  3. Sort negative stays by n_total_ts ascending (shortest first)
  4. Greedily add negative stays until adding the next one would drop the rate below 12%
  5. Filter all parquet files to the selected stay_ids
  6. Copy preproc/ directory from source
  7. Print before/after statistics

Usage:
    python scripts/create_aki_filtered_intersection.py
"""

import shutil
from pathlib import Path

import polars as pl

# === Configuration ===
SEED = 2222
TARGET_POSITIVE_RATE = 0.12  # ~12% per-timestep positive rate (AKI-like)

SRC_DIR = Path("/bigdata/omerg/Thesis/cohort_data/sepsis/eicu_aki_intersection")
DST_DIR = Path("/bigdata/omerg/Thesis/cohort_data/sepsis/eicu_aki_intersection_filtered")


def main():
    print("=" * 70)
    print("Creating filtered AKI-sepsis intersection dataset")
    print(f"Source: {SRC_DIR}")
    print(f"Target: {DST_DIR}")
    print(f"Target per-TS positive rate: {TARGET_POSITIVE_RATE:.1%}")
    print(f"Random seed: {SEED}")
    print("=" * 70)

    # --- Load data ---
    print("\nLoading outc.parquet...")
    outc = pl.read_parquet(SRC_DIR / "outc.parquet")

    # --- Compute per-stay statistics ---
    stay_stats = outc.group_by("stay_id").agg(
        pl.col("label").count().alias("n_total_ts"),
        pl.col("label").sum().alias("n_positive_ts"),
    ).with_columns(
        (pl.col("n_positive_ts") > 0).alias("has_positive"),
    )

    positive_stays = stay_stats.filter(pl.col("has_positive"))
    negative_stays = stay_stats.filter(~pl.col("has_positive"))

    total_positive_ts = positive_stays["n_positive_ts"].sum()
    total_ts_positive_stays = positive_stays["n_total_ts"].sum()

    # --- Print original statistics ---
    print("\n--- Original Dataset Statistics ---")
    print(f"Total stays:        {stay_stats.height:,}")
    print(f"Positive stays:     {positive_stays.height:,} ({positive_stays.height / stay_stats.height:.1%})")
    print(f"Negative stays:     {negative_stays.height:,} ({negative_stays.height / stay_stats.height:.1%})")
    print(f"Total timesteps:    {outc.height:,}")
    print(f"Positive timesteps: {total_positive_ts:,} ({total_positive_ts / outc.height:.2%})")
    print(f"Positive stays TS:  {total_ts_positive_stays:,}")
    print(f"Negative stays TS:  {negative_stays['n_total_ts'].sum():,}")

    # --- Compute how many negative stays to keep ---
    # We want: total_positive_ts / (total_ts_positive_stays + negative_ts_budget) >= TARGET_RATE
    # => negative_ts_budget <= total_positive_ts / TARGET_RATE - total_ts_positive_stays
    max_total_ts = total_positive_ts / TARGET_POSITIVE_RATE
    max_negative_ts = max_total_ts - total_ts_positive_stays

    print(f"\n--- Budget Calculation ---")
    print(f"Max total TS for {TARGET_POSITIVE_RATE:.0%} rate: {max_total_ts:,.0f}")
    print(f"TS from positive stays:                  {total_ts_positive_stays:,}")
    print(f"Max negative TS budget:                  {max_negative_ts:,.0f}")

    if max_negative_ts <= 0:
        print("\nWARNING: Even with only positive stays, the rate is below target!")
        rate_pos_only = total_positive_ts / total_ts_positive_stays
        print(f"Rate with positive stays only: {rate_pos_only:.2%}")
        print("Using positive stays only.")
        selected_negative_ids = []
        selected_negative_ts = 0
    else:
        # Sort negative stays by n_total_ts ascending (shortest first) to pack more stays
        # Use random shuffle within same-length groups for fairness
        negative_stays_sorted = negative_stays.with_columns(
            pl.col("stay_id").shuffle(seed=SEED).alias("_rand_order")
        ).sort(["n_total_ts", "_rand_order"])

        # Greedily add negative stays
        cumsum_ts = negative_stays_sorted["n_total_ts"].cum_sum()
        # Find how many we can add without exceeding budget
        mask = cumsum_ts <= int(max_negative_ts)
        n_selected = mask.sum()

        selected_negative = negative_stays_sorted.head(n_selected)
        selected_negative_ids = selected_negative["stay_id"].to_list()
        selected_negative_ts = selected_negative["n_total_ts"].sum()

        print(f"\nSelected {n_selected:,} of {negative_stays.height:,} negative stays")
        print(f"Selected negative TS: {selected_negative_ts:,} (budget: {max_negative_ts:,.0f})")

    # --- Build final stay_id set ---
    positive_ids = positive_stays["stay_id"].to_list()
    all_selected_ids = set(positive_ids + selected_negative_ids)

    final_total_ts = total_ts_positive_stays + selected_negative_ts
    final_rate = total_positive_ts / final_total_ts if final_total_ts > 0 else 0

    print(f"\n--- Filtered Dataset Statistics ---")
    print(f"Total stays:        {len(all_selected_ids):,}")
    print(f"Positive stays:     {len(positive_ids):,} ({len(positive_ids) / len(all_selected_ids):.1%})")
    print(f"Negative stays:     {len(selected_negative_ids):,} ({len(selected_negative_ids) / len(all_selected_ids):.1%})")
    print(f"Total timesteps:    {final_total_ts:,}")
    print(f"Positive timesteps: {total_positive_ts:,} ({final_rate:.2%})")

    # --- Filter and save ---
    DST_DIR.mkdir(parents=True, exist_ok=True)

    selected_ids_series = pl.Series("stay_id", list(all_selected_ids))

    # Filter outc
    print("\nFiltering and saving outc.parquet...")
    outc_filtered = outc.filter(pl.col("stay_id").is_in(selected_ids_series))
    outc_filtered.write_parquet(DST_DIR / "outc.parquet")
    print(f"  outc: {outc.height:,} -> {outc_filtered.height:,} rows")

    # Filter dyn
    print("Loading and filtering dyn.parquet...")
    dyn = pl.read_parquet(SRC_DIR / "dyn.parquet")
    dyn_filtered = dyn.filter(pl.col("stay_id").is_in(selected_ids_series))
    dyn_filtered.write_parquet(DST_DIR / "dyn.parquet")
    print(f"  dyn: {dyn.height:,} -> {dyn_filtered.height:,} rows")

    # Filter sta
    print("Loading and filtering sta.parquet...")
    sta = pl.read_parquet(SRC_DIR / "sta.parquet")
    sta_filtered = sta.filter(pl.col("stay_id").is_in(selected_ids_series))
    sta_filtered.write_parquet(DST_DIR / "sta.parquet")
    print(f"  sta: {sta.height:,} -> {sta_filtered.height:,} rows")

    # Copy preproc directory
    dst_preproc = DST_DIR / "preproc"
    if dst_preproc.exists():
        shutil.rmtree(dst_preproc)
    print("Copying preproc/ directory...")
    shutil.copytree(SRC_DIR / "preproc", dst_preproc)
    print(f"  Copied {len(list(dst_preproc.iterdir()))} files")

    # --- Verification ---
    print("\n--- Verification ---")
    outc_check = pl.read_parquet(DST_DIR / "outc.parquet")
    n_stays_check = outc_check["stay_id"].n_unique()
    n_pos_ts_check = (outc_check["label"] == 1).sum()
    n_total_ts_check = outc_check.height
    rate_check = n_pos_ts_check / n_total_ts_check

    print(f"Saved stays:        {n_stays_check:,}")
    print(f"Saved timesteps:    {n_total_ts_check:,}")
    print(f"Saved positive TS:  {n_pos_ts_check:,}")
    print(f"Per-TS positive rate: {rate_check:.2%}")
    print(f"Target rate:          {TARGET_POSITIVE_RATE:.2%}")

    # Check all files have matching stay counts
    dyn_check = pl.read_parquet(DST_DIR / "dyn.parquet")
    sta_check = pl.read_parquet(DST_DIR / "sta.parquet")
    assert dyn_check["stay_id"].n_unique() == n_stays_check, "dyn stay count mismatch!"
    assert sta_check["stay_id"].n_unique() == n_stays_check, "sta stay count mismatch!"
    assert dyn_check.height == outc_check.height, "dyn/outc row count mismatch!"
    print("All consistency checks passed.")

    print(f"\nFiltered dataset saved to: {DST_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
