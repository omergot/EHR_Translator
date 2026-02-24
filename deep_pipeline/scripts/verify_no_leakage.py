#!/usr/bin/env python3
"""
Verify whether there is train/test data leakage in the sepsis filtered experiment.

Scenario:
  1. TRAINED translator on FILTERED eICU cohort (12,939 stays), split with seed=2222
  2. EVALUATED on ORIGINAL full eICU cohort (123,412 stays), split with seed=2222

Risk: YAIB splits each dataset independently using stay_id ordering + sklearn splitters.
      Stays in filtered TRAIN could end up in original TEST.

This script reproduces the exact YAIB splitting logic from
  icu_benchmarks/data/split_process_data.py::make_single_split_polars()
with the same parameters:
  cv_repetitions=2, cv_folds=8, repetition_index=0, fold_index=0,
  train_size=0.8, seed=2222
"""

import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def simulate_yaib_split(outc_path: str, seed: int = 2222):
    """Reproduce the exact YAIB splitting for a given outcome parquet."""
    outc = pl.read_parquet(outc_path)

    # Exactly mirrors make_single_split_polars:
    # stays = pl.Series(name=id, values=data[DataSegment.outcome][id].unique())
    stays = pl.Series(name="stay_id", values=outc["stay_id"].unique())

    # labels: data[DataSegment.outcome].group_by(id).max().sort(id)[vars[VarType.label]]
    labels = outc.group_by("stay_id").max().sort("stay_id")["label"]

    # Outer split: StratifiedShuffleSplit(cv_repetitions=2, train_size=0.8, random_state=2222)
    outer_cv = StratifiedShuffleSplit(n_splits=2, train_size=0.8, random_state=seed)
    # repetition_index=0
    dev_indices, test_indices = list(outer_cv.split(stays, labels))[0]
    dev_stays = stays[dev_indices]

    # Inner split: StratifiedKFold(cv_folds=8, shuffle=True, random_state=2222)
    inner_cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
    # fold_index=0
    train_indices, val_indices = list(inner_cv.split(dev_stays, labels[dev_indices]))[0]

    train_ids = set(dev_stays[train_indices].to_list())
    val_ids = set(dev_stays[val_indices].to_list())
    test_ids = set(stays[test_indices].to_list())

    return train_ids, val_ids, test_ids


def main():
    full_outc_path = "/bigdata/omerg/Thesis/cohort_data/sepsis/eicu/outc.parquet"
    filtered_outc_path = "/bigdata/omerg/Thesis/cohort_data/sepsis/eicu_filtered_aki_density/outc.parquet"

    print("=" * 80)
    print("DATA LEAKAGE VERIFICATION")
    print("Filtered eICU -> trained translator, Original eICU -> evaluated")
    print("=" * 80)

    # Load cohort stats
    full_outc = pl.read_parquet(full_outc_path)
    filtered_outc = pl.read_parquet(filtered_outc_path)

    full_stays = set(full_outc["stay_id"].unique().to_list())
    filtered_stays = set(filtered_outc["stay_id"].unique().to_list())

    print(f"\nFull eICU cohort:     {len(full_stays):,} unique stays")
    print(f"Filtered eICU cohort: {len(filtered_stays):,} unique stays")
    print(f"Overlap (filtered is subset): {len(filtered_stays & full_stays):,} stays")
    print(f"Filtered-only (not in full):  {len(filtered_stays - full_stays):,} stays")

    assert filtered_stays.issubset(full_stays), "ERROR: Filtered is NOT a subset of full!"
    print("CONFIRMED: Filtered cohort is a strict subset of the full cohort.")

    # Simulate splits
    print("\n--- Simulating YAIB splits (seed=2222, cv_rep=2, cv_folds=8, rep=0, fold=0, train_size=0.8) ---")

    f_train, f_val, f_test = simulate_yaib_split(filtered_outc_path)
    o_train, o_val, o_test = simulate_yaib_split(full_outc_path)

    print(f"\nFiltered split: train={len(f_train):,} | val={len(f_val):,} | test={len(f_test):,}")
    print(f"Original split: train={len(o_train):,} | val={len(o_val):,} | test={len(o_test):,}")

    # The critical check: filtered_train stays that appear in original_test
    leaked_train_to_test = f_train & o_test
    leaked_val_to_test = f_val & o_test
    leaked_trainval_to_test = (f_train | f_val) & o_test

    print("\n" + "=" * 80)
    print("LEAKAGE ANALYSIS")
    print("=" * 80)

    print(f"\nFiltered TRAIN stays in Original TEST: {len(leaked_train_to_test):,} / {len(f_train):,} "
          f"({100 * len(leaked_train_to_test) / len(f_train):.2f}%)")
    print(f"Filtered VAL stays in Original TEST:   {len(leaked_val_to_test):,} / {len(f_val):,} "
          f"({100 * len(leaked_val_to_test) / len(f_val):.2f}%)")
    print(f"Filtered TRAIN+VAL in Original TEST:   {len(leaked_trainval_to_test):,} / {len(f_train | f_val):,} "
          f"({100 * len(leaked_trainval_to_test) / len(f_train | f_val):.2f}%)")

    # Also check: what fraction of the original test set is contaminated?
    contaminated_test = leaked_trainval_to_test & o_test
    print(f"\nContaminated Original TEST stays:      {len(contaminated_test):,} / {len(o_test):,} "
          f"({100 * len(contaminated_test) / len(o_test):.2f}%)")

    # Bonus: check how filtered test overlaps with original test
    shared_test = f_test & o_test
    print(f"\nFiltered TEST in Original TEST:        {len(shared_test):,} / {len(f_test):,} "
          f"({100 * len(shared_test) / len(f_test):.2f}%)")

    # Detailed breakdown: where did filtered train stays end up in the original split?
    f_train_in_o_train = f_train & o_train
    f_train_in_o_val = f_train & o_val
    f_train_in_o_test = f_train & o_test
    f_train_nowhere = f_train - (o_train | o_val | o_test)

    print(f"\n--- Where do filtered TRAIN stays land in the original split? ---")
    print(f"  -> Original TRAIN: {len(f_train_in_o_train):,} ({100 * len(f_train_in_o_train) / len(f_train):.2f}%)")
    print(f"  -> Original VAL:   {len(f_train_in_o_val):,} ({100 * len(f_train_in_o_val) / len(f_train):.2f}%)")
    print(f"  -> Original TEST:  {len(f_train_in_o_test):,} ({100 * len(f_train_in_o_test) / len(f_train):.2f}%)")
    print(f"  -> Nowhere:        {len(f_train_nowhere):,}")

    # Count leaked timesteps (rows) to understand severity
    print("\n--- Timestep-level impact ---")
    # How many rows in the original test set come from leaked stays?
    full_test_df = full_outc.filter(pl.col("stay_id").is_in(list(o_test)))
    leaked_rows = full_outc.filter(pl.col("stay_id").is_in(list(leaked_trainval_to_test)))
    print(f"Original TEST total timesteps:        {len(full_test_df):,}")
    print(f"Leaked (filtered train+val) timesteps: {len(leaked_rows):,} "
          f"({100 * len(leaked_rows) / len(full_test_df):.2f}% of original test)")

    # Verdict
    print("\n" + "=" * 80)
    if len(leaked_trainval_to_test) > 0:
        print("VERDICT: DATA LEAKAGE CONFIRMED")
        print(f"  {len(leaked_trainval_to_test):,} stays ({100 * len(leaked_trainval_to_test) / len(o_test):.1f}% "
              f"of original test) were used during training on the filtered cohort.")
        print("  The +0.0805 AUCROC improvement on the original test set is NOT trustworthy.")
        print("  The model may have memorized patterns from these stays during training.")
    else:
        print("VERDICT: NO DATA LEAKAGE")
        print("  No filtered train/val stays appear in the original test set.")
    print("=" * 80)


if __name__ == "__main__":
    main()
