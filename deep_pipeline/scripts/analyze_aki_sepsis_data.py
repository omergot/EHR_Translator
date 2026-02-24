#!/usr/bin/env python3
"""
Analyze AKI and Sepsis data statistics for both eICU (source) and MIMIC (target).
Computes per-stay and per-timestep label statistics for comparison.
"""

import polars as pl
import numpy as np

BASE = "/bigdata/omerg/Thesis/cohort_data"

DATASETS = {
    "AKI eICU (source)": {
        "dyn": f"{BASE}/aki/eicu/dyn.parquet",
        "outc": f"{BASE}/aki/eicu/outc.parquet",
        "sta": f"{BASE}/aki/eicu/sta.parquet",
    },
    "AKI MIMIC (target)": {
        "dyn": f"{BASE}/aki/miiv/dyn.parquet",
        "outc": f"{BASE}/aki/miiv/outc.parquet",
        "sta": f"{BASE}/aki/miiv/sta.parquet",
    },
    "Sepsis eICU (source)": {
        "dyn": f"{BASE}/sepsis/eicu/dyn.parquet",
        "outc": f"{BASE}/sepsis/eicu/outc.parquet",
        "sta": f"{BASE}/sepsis/eicu/sta.parquet",
    },
    "Sepsis MIMIC (target)": {
        "dyn": f"{BASE}/sepsis/miiv/dyn.parquet",
        "outc": f"{BASE}/sepsis/miiv/outc.parquet",
        "sta": f"{BASE}/sepsis/miiv/sta.parquet",
    },
}


def analyze_dataset(name, paths):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # Load data
    outc = pl.read_parquet(paths["outc"])
    dyn = pl.read_parquet(paths["dyn"])

    # Show schema
    print(f"\nOutcome columns: {outc.columns}")
    print(f"Outcome shape: {outc.shape}")
    print(f"Dynamic columns: {dyn.columns[:5]}... ({len(dyn.columns)} total)")
    print(f"Dynamic shape: {dyn.shape}")

    # Identify the stay ID column and label column
    id_col = outc.columns[0]  # Usually 'stay_id' or similar
    print(f"ID column: {id_col}")

    # Find label column (last column that isn't id or time)
    label_candidates = [c for c in outc.columns if c not in [id_col, "time"]]
    print(f"Label candidates: {label_candidates}")

    # Use the last label candidate (usually the outcome)
    label_col = label_candidates[-1]
    print(f"Using label column: {label_col}")

    # Show first few rows
    print(f"\nFirst 5 outcome rows:")
    print(outc.head(5))

    # Check if this is per-timestep (has 'time' column) or per-stay
    is_per_timestep = "time" in outc.columns
    print(f"\nPer-timestep labels: {is_per_timestep}")

    if is_per_timestep:
        # ----- Per-timestep analysis -----
        total_timesteps = len(outc)
        stays = outc[id_col].unique()
        total_stays = len(stays)

        # Per-stay statistics
        stay_stats = outc.group_by(id_col).agg([
            pl.col(label_col).sum().alias("pos_ts"),
            pl.col(label_col).count().alias("total_ts"),
            (pl.col(label_col).sum() > 0).alias("has_positive"),
        ])

        positive_stays = stay_stats.filter(pl.col("has_positive")).height
        negative_stays = total_stays - positive_stays
        per_stay_pos_rate = positive_stays / total_stays

        # Per-timestep positive rate
        total_pos_ts = outc[label_col].sum()
        per_ts_pos_rate = total_pos_ts / total_timesteps

        # Per-timestep positive rate within positive stays only
        pos_stay_data = stay_stats.filter(pl.col("has_positive"))
        total_ts_in_pos_stays = pos_stay_data["total_ts"].sum()
        total_pos_ts_in_pos_stays = pos_stay_data["pos_ts"].sum()
        per_ts_pos_rate_in_pos_stays = total_pos_ts_in_pos_stays / total_ts_in_pos_stays if total_ts_in_pos_stays > 0 else 0

        # Mean positive timesteps per positive stay
        mean_pos_ts_per_pos_stay = pos_stay_data["pos_ts"].mean() if positive_stays > 0 else 0

        # Mean total timesteps per stay
        mean_total_ts = stay_stats["total_ts"].mean()

        # Also compute for dynamic data (to check padding)
        dyn_stay_stats = dyn.group_by(id_col).agg([
            pl.col(dyn.columns[2]).count().alias("dyn_ts"),
        ])
        mean_dyn_ts = dyn_stay_stats["dyn_ts"].mean()

        print(f"\n--- Results ---")
        print(f"Total stays:                    {total_stays:,}")
        print(f"Positive stays (>=1 pos TS):    {positive_stays:,}")
        print(f"Negative stays (all zero):      {negative_stays:,}")
        print(f"Per-stay positive rate:          {per_stay_pos_rate:.4f} ({per_stay_pos_rate*100:.2f}%)")
        print(f"")
        print(f"Total timesteps (outcome):      {total_timesteps:,}")
        print(f"Total positive timesteps:       {int(total_pos_ts):,}")
        print(f"Per-TS positive rate (all):     {per_ts_pos_rate:.4f} ({per_ts_pos_rate*100:.2f}%)")
        print(f"Per-TS pos rate (pos stays):    {per_ts_pos_rate_in_pos_stays:.4f} ({per_ts_pos_rate_in_pos_stays*100:.2f}%)")
        print(f"")
        print(f"Mean pos TS per positive stay:  {mean_pos_ts_per_pos_stay:.2f}")
        print(f"Mean total TS per stay (outc):  {mean_total_ts:.2f}")
        print(f"Mean total TS per stay (dyn):   {mean_dyn_ts:.2f}")

        # Distribution of positive timesteps per positive stay
        pos_ts_values = pos_stay_data["pos_ts"].to_numpy()
        if len(pos_ts_values) > 0:
            print(f"\nPositive TS per positive stay distribution:")
            print(f"  Min: {np.min(pos_ts_values)}, Median: {np.median(pos_ts_values):.0f}, "
                  f"Mean: {np.mean(pos_ts_values):.1f}, Max: {np.max(pos_ts_values)}")
            for pct in [25, 50, 75, 90, 95]:
                print(f"  P{pct}: {np.percentile(pos_ts_values, pct):.0f}")

        # Distribution of total timesteps per stay
        total_ts_values = stay_stats["total_ts"].to_numpy()
        print(f"\nTotal TS per stay distribution:")
        print(f"  Min: {np.min(total_ts_values)}, Median: {np.median(total_ts_values):.0f}, "
              f"Mean: {np.mean(total_ts_values):.1f}, Max: {np.max(total_ts_values)}")
        for pct in [25, 50, 75, 90, 95]:
            print(f"  P{pct}: {np.percentile(total_ts_values, pct):.0f}")

        return {
            "total_stays": total_stays,
            "positive_stays": positive_stays,
            "negative_stays": negative_stays,
            "per_stay_pos_rate": per_stay_pos_rate,
            "total_timesteps": total_timesteps,
            "total_pos_ts": int(total_pos_ts),
            "per_ts_pos_rate": float(per_ts_pos_rate),
            "per_ts_pos_rate_in_pos_stays": float(per_ts_pos_rate_in_pos_stays),
            "mean_pos_ts_per_pos_stay": float(mean_pos_ts_per_pos_stay),
            "mean_total_ts": float(mean_total_ts),
            "mean_ts_pos_stays": float(pos_stay_data["total_ts"].mean()),
            "mean_ts_neg_stays": float(stay_stats.filter(~pl.col("has_positive"))["total_ts"].mean()),
        }
    else:
        # Per-stay labels (like mortality)
        total_stays = len(outc)
        positive_stays = int(outc[label_col].sum())
        negative_stays = total_stays - positive_stays
        per_stay_pos_rate = positive_stays / total_stays

        print(f"\n--- Results (per-stay labels) ---")
        print(f"Total stays:                    {total_stays:,}")
        print(f"Positive stays:                 {positive_stays:,}")
        print(f"Negative stays:                 {negative_stays:,}")
        print(f"Per-stay positive rate:          {per_stay_pos_rate:.4f} ({per_stay_pos_rate*100:.2f}%)")

        return {
            "total_stays": total_stays,
            "positive_stays": positive_stays,
            "negative_stays": negative_stays,
            "per_stay_pos_rate": per_stay_pos_rate,
        }


def main():
    results = {}
    for name, paths in DATASETS.items():
        results[name] = analyze_dataset(name, paths)

    # --- Comparison table ---
    print(f"\n\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")

    header = f"{'Metric':<40} {'AKI eICU':>12} {'AKI MIMIC':>12} {'Sep eICU':>12} {'Sep MIMIC':>12}"
    print(header)
    print("-" * len(header))

    keys = ["AKI eICU (source)", "AKI MIMIC (target)", "Sepsis eICU (source)", "Sepsis MIMIC (target)"]

    def fmt(val, is_pct=False):
        if val is None:
            return "N/A"
        if is_pct:
            return f"{val*100:.2f}%"
        if isinstance(val, float):
            return f"{val:.2f}"
        return f"{val:,}"

    metrics = [
        ("Total stays", "total_stays", False),
        ("Positive stays", "positive_stays", False),
        ("Negative stays", "negative_stays", False),
        ("Per-stay positive rate", "per_stay_pos_rate", True),
        ("Total timesteps", "total_timesteps", False),
        ("Total positive TS", "total_pos_ts", False),
        ("Per-TS positive rate (all)", "per_ts_pos_rate", True),
        ("Per-TS pos rate (pos stays only)", "per_ts_pos_rate_in_pos_stays", True),
        ("Mean pos TS / positive stay", "mean_pos_ts_per_pos_stay", False),
        ("Mean total TS / stay", "mean_total_ts", False),
    ]

    for label, key, is_pct in metrics:
        vals = []
        for k in keys:
            v = results[k].get(key)
            vals.append(fmt(v, is_pct))
        print(f"{label:<40} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

    # --- Subsampling analysis for sepsis ---
    print(f"\n\n{'='*70}")
    print(f"  SUBSAMPLING ANALYSIS: Making Sepsis Match AKI's Label Density")
    print(f"{'='*70}")

    sep_src = results["Sepsis eICU (source)"]
    aki_src = results["AKI eICU (source)"]

    sep_pos_stays = sep_src["positive_stays"]
    sep_neg_stays = sep_src["negative_stays"]
    sep_total_stays = sep_src["total_stays"]

    aki_per_stay_rate = aki_src["per_stay_pos_rate"]
    aki_per_ts_rate = aki_src["per_ts_pos_rate"]

    print(f"\nCurrent sepsis: {sep_pos_stays:,} positive stays, {sep_neg_stays:,} negative stays")
    print(f"Current sepsis per-stay rate: {sep_src['per_stay_pos_rate']*100:.2f}%")
    print(f"Current sepsis per-TS rate: {sep_src['per_ts_pos_rate']*100:.2f}%")
    print(f"")
    print(f"AKI per-stay rate: {aki_per_stay_rate*100:.2f}%")
    print(f"AKI per-TS rate: {aki_per_ts_rate*100:.2f}%")

    # To match AKI's per-stay positive rate:
    neg_stays_to_keep = int(sep_pos_stays * (1 - aki_per_stay_rate) / aki_per_stay_rate)

    print(f"\n--- To match AKI per-stay positive rate ({aki_per_stay_rate*100:.2f}%) ---")
    print(f"Keep all {sep_pos_stays:,} positive stays")
    print(f"Keep {neg_stays_to_keep:,} negative stays (out of {sep_neg_stays:,}, remove {sep_neg_stays - neg_stays_to_keep:,})")
    print(f"New total: {sep_pos_stays + neg_stays_to_keep:,} stays")
    print(f"New per-stay rate: {sep_pos_stays / (sep_pos_stays + neg_stays_to_keep) * 100:.2f}%")

    # More precise per-TS estimation using actual mean TS for pos/neg stays
    mean_ts_pos = sep_src["mean_ts_pos_stays"]
    mean_ts_neg = sep_src["mean_ts_neg_stays"]
    total_pos_ts = sep_src["total_pos_ts"]

    new_total_ts = (sep_pos_stays * mean_ts_pos) + (neg_stays_to_keep * mean_ts_neg)
    new_per_ts_rate = total_pos_ts / new_total_ts
    print(f"Estimated new per-TS rate: {new_per_ts_rate*100:.2f}%")
    print(f"  (using mean TS/pos stay={mean_ts_pos:.1f}, mean TS/neg stay={mean_ts_neg:.1f})")

    # What if we want to match AKI per-TS rate exactly?
    target_per_ts = aki_per_ts_rate
    neg_for_ts_match = (total_pos_ts / target_per_ts - sep_pos_stays * mean_ts_pos) / mean_ts_neg
    neg_for_ts_match = max(0, int(neg_for_ts_match))

    print(f"\n--- To match AKI per-TS positive rate ({aki_per_ts_rate*100:.2f}%) ---")
    print(f"Keep all {sep_pos_stays:,} positive stays")
    print(f"Keep {neg_for_ts_match:,} negative stays (out of {sep_neg_stays:,}, remove {sep_neg_stays - neg_for_ts_match:,})")
    print(f"New total: {sep_pos_stays + neg_for_ts_match:,} stays")
    if neg_for_ts_match > 0:
        new_total_ts2 = (sep_pos_stays * mean_ts_pos) + (neg_for_ts_match * mean_ts_neg)
        print(f"New per-TS rate: {total_pos_ts / new_total_ts2 * 100:.2f}%")
        print(f"New per-stay rate: {sep_pos_stays / (sep_pos_stays + neg_for_ts_match) * 100:.2f}%")
    else:
        print(f"Cannot reach {aki_per_ts_rate*100:.2f}% per-TS even with only positive stays")
        pos_only_rate = total_pos_ts / (sep_pos_stays * mean_ts_pos)
        print(f"Max possible per-TS rate (pos stays only): {pos_only_rate*100:.2f}%")

    # --- Oversampling analysis ---
    print(f"\n\n{'='*70}")
    print(f"  OVERSAMPLING COMPARISON")
    print(f"{'='*70}")
    print(f"\nCurrent sepsis oversampling_factor=20 gives effective per-stay rate:")
    eff_sep = (sep_pos_stays * 20) / (sep_pos_stays * 20 + sep_neg_stays)
    print(f"  Sepsis OF=20: {eff_sep*100:.2f}% effective per-stay rate")
    aki_pos = aki_src["positive_stays"]
    aki_neg = aki_src["negative_stays"]
    print(f"  AKI native:   {aki_per_stay_rate*100:.2f}% per-stay rate (no oversampling needed)")
    # What OF would sepsis need to match AKI native rate?
    of_for_aki_match = sep_neg_stays * aki_per_stay_rate / (sep_pos_stays * (1 - aki_per_stay_rate))
    print(f"  Sepsis OF to match AKI per-stay rate: {of_for_aki_match:.1f}")

    # --- Label density comparison ---
    print(f"\n\n{'='*70}")
    print(f"  KEY INSIGHT: LABEL DENSITY COMPARISON")
    print(f"{'='*70}")

    sep_per_ts_in_pos = sep_src["per_ts_pos_rate_in_pos_stays"]
    aki_per_ts_in_pos = aki_src["per_ts_pos_rate_in_pos_stays"]

    print(f"\n{'Metric':<45} {'AKI eICU':>12} {'Sepsis eICU':>12} {'Ratio':>8}")
    print("-" * 80)
    print(f"{'Per-stay positive rate':<45} {aki_per_stay_rate*100:>11.2f}% {sep_src['per_stay_pos_rate']*100:>11.2f}% {aki_per_stay_rate/sep_src['per_stay_pos_rate']:>7.1f}x")
    print(f"{'Per-TS positive rate (all stays)':<45} {aki_per_ts_rate*100:>11.2f}% {sep_src['per_ts_pos_rate']*100:>11.2f}% {aki_per_ts_rate/sep_src['per_ts_pos_rate']:>7.1f}x")
    print(f"{'Per-TS positive rate (pos stays only)':<45} {aki_per_ts_in_pos*100:>11.2f}% {sep_per_ts_in_pos*100:>11.2f}% {aki_per_ts_in_pos/sep_per_ts_in_pos:>7.1f}x")
    print(f"{'Mean pos TS per positive stay':<45} {aki_src['mean_pos_ts_per_pos_stay']:>12.1f} {sep_src['mean_pos_ts_per_pos_stay']:>12.1f} {aki_src['mean_pos_ts_per_pos_stay']/sep_src['mean_pos_ts_per_pos_stay']:>7.1f}x")
    print(f"{'Mean total TS per stay':<45} {aki_src['mean_total_ts']:>12.1f} {sep_src['mean_total_ts']:>12.1f} {aki_src['mean_total_ts']/sep_src['mean_total_ts']:>7.1f}x")

    # --- Source vs Target comparison ---
    print(f"\n\n{'='*70}")
    print(f"  SOURCE vs TARGET COMPARISON (for alignment loss)")
    print(f"{'='*70}")

    for task in ["AKI", "Sepsis"]:
        src = results[f"{task} eICU (source)"]
        tgt = results[f"{task} MIMIC (target)"]
        print(f"\n--- {task} ---")
        print(f"{'Metric':<40} {'eICU (src)':>12} {'MIMIC (tgt)':>12} {'Ratio':>8}")
        print("-" * 75)
        print(f"{'Total stays':<40} {src['total_stays']:>12,} {tgt['total_stays']:>12,} {src['total_stays']/tgt['total_stays']:>7.2f}x")
        print(f"{'Positive stays':<40} {src['positive_stays']:>12,} {tgt['positive_stays']:>12,} {src['positive_stays']/tgt['positive_stays']:>7.2f}x")
        print(f"{'Per-stay positive rate':<40} {src['per_stay_pos_rate']*100:>11.2f}% {tgt['per_stay_pos_rate']*100:>11.2f}%")
        if "per_ts_pos_rate" in src:
            print(f"{'Per-TS positive rate':<40} {src['per_ts_pos_rate']*100:>11.2f}% {tgt['per_ts_pos_rate']*100:>11.2f}%")
            print(f"{'Mean total TS / stay':<40} {src['mean_total_ts']:>12.1f} {tgt['mean_total_ts']:>12.1f}")


if __name__ == "__main__":
    main()
