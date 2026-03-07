#!/usr/bin/env python3
"""Aggregate multi-seed experiment results for EHR Translator.

Usage:
    # From prediction files
    python scripts/aggregate_seeds.py \
        runs/aki_sl_fg_tseed1337/eval.predictions.npz \
        runs/aki_sl_fg_tseed7777/eval.predictions.npz \
        runs/aki_sl_featgate_full/eval.predictions.npz

    # With baseline for delta computation
    python scripts/aggregate_seeds.py \
        --baseline runs/aki_sl_featgate_full/eval.original.predictions.npz \
        runs/aki_sl_fg_tseed1337/eval.predictions.npz \
        runs/aki_sl_fg_tseed7777/eval.predictions.npz

Computes mean ± std, t-distribution CI, and range across seeds.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_metrics(probs, targets):
    try:
        auroc = roc_auc_score(targets, probs)
    except ValueError:
        auroc = float("nan")
    try:
        aucpr = average_precision_score(targets, probs)
    except ValueError:
        aucpr = float("nan")
    return {"AUCROC": auroc, "AUCPR": aucpr}


def aggregate(metric_list: list[dict], confidence: float = 0.95) -> dict:
    """Aggregate metrics across seeds with t-distribution CI."""
    results = {}
    metric_names = metric_list[0].keys()
    n = len(metric_list)

    for name in metric_names:
        values = np.array([m[name] for m in metric_list])
        valid = values[~np.isnan(values)]
        if len(valid) < 2:
            results[name] = {
                "mean": float(np.nanmean(values)),
                "std": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n": len(valid),
                "values": values.tolist(),
            }
            continue

        mean = float(np.mean(valid))
        std = float(np.std(valid, ddof=1))
        se = std / np.sqrt(len(valid))
        t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=len(valid) - 1)
        ci_lo = mean - t_crit * se
        ci_hi = mean + t_crit * se

        results[name] = {
            "mean": mean,
            "std": std,
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "range": float(np.max(valid) - np.min(valid)),
            "n": len(valid),
            "values": valid.tolist(),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument("predictions", nargs="+", help="Paths to .predictions.npz files")
    parser.add_argument("--baseline", help="Path to baseline .predictions.npz for delta computation")
    parser.add_argument("--confidence", type=float, default=0.95, help="CI confidence level")
    args = parser.parse_args()

    metric_list = []
    baseline_metrics = None

    if args.baseline:
        bl = np.load(args.baseline)
        baseline_metrics = compute_metrics(bl["probs"], bl["targets"])
        print(f"Baseline: AUCROC={baseline_metrics['AUCROC']:.4f}, AUCPR={baseline_metrics['AUCPR']:.4f}")
        print()

    for path in args.predictions:
        data = np.load(path)
        metrics = compute_metrics(data["probs"], data["targets"])
        metric_list.append(metrics)
        label = Path(path).parent.name
        print(f"  {label}: AUCROC={metrics['AUCROC']:.4f}, AUCPR={metrics['AUCPR']:.4f}")

    print(f"\n--- Aggregate ({len(metric_list)} seeds, {args.confidence*100:.0f}% CI) ---")
    agg = aggregate(metric_list, args.confidence)

    for name, vals in agg.items():
        if np.isnan(vals.get("std", float("nan"))):
            print(f"  {name}: {vals['mean']:.4f} (insufficient seeds for CI)")
        else:
            print(f"  {name}: {vals['mean']:.4f} ± {vals['std']:.4f} [{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}] range={vals['range']:.4f}")

    if baseline_metrics:
        print(f"\n--- Deltas (vs baseline) ---")
        delta_list = []
        for m in metric_list:
            delta_list.append({k: m[k] - baseline_metrics[k] for k in m})
        delta_agg = aggregate(delta_list, args.confidence)
        for name, vals in delta_agg.items():
            if np.isnan(vals.get("std", float("nan"))):
                print(f"  Δ{name}: {vals['mean']:+.4f} (insufficient seeds for CI)")
            else:
                print(f"  Δ{name}: {vals['mean']:+.4f} ± {vals['std']:.4f} [{vals['ci_lo']:+.4f}, {vals['ci_hi']:+.4f}]")


if __name__ == "__main__":
    main()
