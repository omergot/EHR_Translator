#!/usr/bin/env python3
"""Aggregate AdaTime CNN results from individual scenario JSON files into
the master adatime_cnn_results.json.

Usage:
    python scripts/aggregate_adatime_cnn_results.py --datasets SSC MFD
    python scripts/aggregate_adatime_cnn_results.py  # aggregates all datasets
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.adatime.data_loader import DATASET_CONFIGS

CNN_RUNS_DIR = PROJECT_ROOT / "runs" / "adatime_cnn"
RESULTS_JSON = PROJECT_ROOT / "experiments" / "results" / "adatime_cnn_results.json"


def load_existing_results():
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return {"datasets": {}, "summary": {}}


def aggregate_dataset(dataset_name: str) -> dict:
    """Load all scenario results for a dataset."""
    ds_config = DATASET_CONFIGS[dataset_name]
    dataset_dir = CNN_RUNS_DIR / dataset_name
    if not dataset_dir.exists():
        print(f"  No results directory for {dataset_name}: {dataset_dir}")
        return {}

    all_results = {}
    for src_id, trg_id in ds_config.scenarios:
        scenario_key = f"{src_id}_to_{trg_id}"
        results_file = dataset_dir / scenario_key / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                all_results[scenario_key] = json.load(f)
            print(f"  Loaded {scenario_key}: {list(all_results[scenario_key].keys())}")
        else:
            print(f"  Missing: {results_file}")

    return all_results


def compute_summary(dataset_results: dict) -> dict:
    """Compute mean/std across scenarios for each method."""
    methods = set()
    for scenario_res in dataset_results.values():
        methods.update(scenario_res.keys())

    summary = {}
    for method in sorted(methods):
        accs = [r[method]["accuracy"] for r in dataset_results.values() if method in r]
        f1s = [r[method]["f1"] for r in dataset_results.values() if method in r and "f1" in r[method]]
        aurocs = [r[method]["auroc"] for r in dataset_results.values() if method in r and "auroc" in r[method]]

        if accs:
            summary[method] = {
                "accuracy": float(np.mean(accs)) * 100,
                "accuracy_std": float(np.std(accs)) * 100,
                "f1": float(np.mean(f1s)) * 100 if f1s else 0.0,
                "f1_std": float(np.std(f1s)) * 100 if f1s else 0.0,
                "auroc": float(np.mean(aurocs)) * 100 if aurocs else 0.0,
                "auroc_std": float(np.std(aurocs)) * 100 if aurocs else 0.0,
                "n": len(accs),
            }
    return summary


def print_summary(dataset_name: str, summary: dict):
    """Print a nice summary table."""
    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset_name}")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'Acc':>8} {'±':>5} {'F1':>8} {'AUROC':>8}")
    print(f"{'-'*70}")
    for method, stats in summary.items():
        print(f"{method:<30} {stats['accuracy']:>7.2f}% ±{stats['accuracy_std']:>4.2f}  "
              f"{stats['f1']:>7.2f}%  {stats['auroc']:>7.2f}%  (n={stats['n']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()),
                        default=None, help="Datasets to aggregate (default: all with results)")
    args = parser.parse_args()

    datasets = args.datasets or list(DATASET_CONFIGS.keys())

    # Load existing results
    existing = load_existing_results()

    for dataset_name in datasets:
        print(f"\nAggregating {dataset_name}...")
        dataset_results = aggregate_dataset(dataset_name)
        if not dataset_results:
            print(f"  No results found for {dataset_name}, skipping.")
            continue

        existing["datasets"][dataset_name] = dataset_results
        summary = compute_summary(dataset_results)
        existing["summary"][dataset_name] = summary
        print_summary(dataset_name, summary)

    # Save updated results
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
