#!/usr/bin/env python3
"""Bootstrap CIs for AdaTime benchmark results.

Computes bootstrap confidence intervals over cross-subject scenarios for
each AdaTime dataset. Supports both single-seed and multi-seed analysis.

Usage:
    # All datasets, best config only
    python scripts/bootstrap_adatime_ci.py

    # Specific dataset
    python scripts/bootstrap_adatime_ci.py --datasets HAR HHAR

    # Include multi-seed analysis
    python scripts/bootstrap_adatime_ci.py --multi-seed

    # Custom replicates
    python scripts/bootstrap_adatime_ci.py --n-replicates 2000
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "adatime_cnn"

# Best protocol-compliant config per dataset
BEST_CONFIGS = {
    "HAR": {
        "run_dir": "HAR_v5_k24",
        "seed_pattern": "HAR_best_s{seed}",
        "n_seeds": 5,
        "best_baseline": "DIRT-T",
        "best_baseline_mf1": 93.7,
    },
    "HHAR": {
        "run_dir": "HHAR_v4_cross3",
        "seed_pattern": "HHAR_best_s{seed}",
        "n_seeds": 5,
        "best_baseline": "CoTMix",
        "best_baseline_mf1": 84.5,
    },
    "WISDM": {
        "run_dir": "WISDM_v4_lr67",
        "seed_pattern": "WISDM_best_s{seed}",
        "n_seeds": 5,
        "best_baseline": "CoTMix",
        "best_baseline_mf1": 66.3,
    },
    "SSC": {
        "run_dir": "SSC_full_v5_nopretrain_d64",
        "seed_pattern": "SSC_full_best_s{seed}",
        "n_seeds": 5,
        "best_baseline": "MMDA",
        "best_baseline_mf1": 63.5,
    },
    "MFD": {
        "run_dir": "MFD_full_v5_nopretrain",
        "seed_pattern": "MFD_full_best_res_s{seed}",
        "n_seeds": 3,
        "best_baseline": "DIRT-T",
        "best_baseline_mf1": 92.8,
    },
}


def load_scenario_results(run_dir: Path) -> dict:
    """Load per-scenario results from a run directory.

    Returns dict of scenario_name -> {source_only_f1, translator_f1, ...}
    """
    results = {}
    for scenario_dir in sorted(run_dir.iterdir()):
        results_file = scenario_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            results[scenario_dir.name] = data
    return results


def extract_f1_arrays(scenario_results: dict) -> tuple:
    """Extract source_only and translator F1 arrays from scenario results.

    Handles both `translator_cnn` (standard) and `translator_cnn_full` (chunked).
    Returns (source_f1_array, translator_f1_array, scenario_names)
    """
    source_f1s = []
    translator_f1s = []
    names = []
    for name, data in sorted(scenario_results.items()):
        # Translator key: standard or chunked (full-length)
        trans_key = None
        if "translator_cnn" in data:
            trans_key = "translator_cnn"
        elif "translator_cnn_full" in data:
            trans_key = "translator_cnn_full"

        if "source_only_cnn" in data and trans_key:
            source_f1s.append(data["source_only_cnn"]["f1"])
            translator_f1s.append(data[trans_key]["f1"])
            names.append(name)
    return np.array(source_f1s), np.array(translator_f1s), names


def bootstrap_mean_ci(values: np.ndarray, n_replicates: int = 2000,
                      seed: int = 42, ci: float = 0.95) -> dict:
    """Bootstrap CI for the mean of values."""
    rng = np.random.RandomState(seed)
    n = len(values)
    point = float(np.mean(values))

    boot_means = np.zeros(n_replicates)
    for i in range(n_replicates):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(values[idx])

    alpha = (1 - ci) / 2
    ci_lo = float(np.percentile(boot_means, alpha * 100))
    ci_hi = float(np.percentile(boot_means, (1 - alpha) * 100))

    return {
        "point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_width": ci_hi - ci_lo,
        "std": float(np.std(boot_means)),
        "n": n,
    }


def bootstrap_paired_diff(values_a: np.ndarray, values_b: np.ndarray,
                          n_replicates: int = 2000, seed: int = 42,
                          ci: float = 0.95) -> dict:
    """Bootstrap CI for the paired difference (A - B)."""
    rng = np.random.RandomState(seed)
    n = len(values_a)
    assert len(values_b) == n

    diffs = values_a - values_b
    point = float(np.mean(diffs))

    boot_diffs = np.zeros(n_replicates)
    for i in range(n_replicates):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs[i] = np.mean(diffs[idx])

    alpha = (1 - ci) / 2
    ci_lo = float(np.percentile(boot_diffs, alpha * 100))
    ci_hi = float(np.percentile(boot_diffs, (1 - alpha) * 100))

    # p-value: fraction of replicates where diff <= 0
    p_value = float(np.mean(boot_diffs <= 0))

    return {
        "point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_width": ci_hi - ci_lo,
        "p_value": p_value,
        "significant": ci_lo > 0,
        "n": n,
    }


def bootstrap_multiseed(seed_f1_arrays: list, n_replicates: int = 2000,
                        seed: int = 42, ci: float = 0.95) -> dict:
    """Bootstrap CI pooling across seeds and scenarios.

    Each entry in seed_f1_arrays is a 1D array of per-scenario F1 for one seed.
    We stack into (n_seeds, n_scenarios) and bootstrap over both dimensions.
    """
    rng = np.random.RandomState(seed)
    # Stack: (n_seeds, n_scenarios)
    matrix = np.array(seed_f1_arrays)
    n_seeds, n_scenarios = matrix.shape

    # Point estimate: mean over all (seed, scenario) pairs
    point = float(np.mean(matrix))

    boot_means = np.zeros(n_replicates)
    for i in range(n_replicates):
        # Resample seeds
        seed_idx = rng.choice(n_seeds, size=n_seeds, replace=True)
        # Resample scenarios
        scen_idx = rng.choice(n_scenarios, size=n_scenarios, replace=True)
        boot_means[i] = np.mean(matrix[np.ix_(seed_idx, scen_idx)])

    alpha = (1 - ci) / 2
    ci_lo = float(np.percentile(boot_means, alpha * 100))
    ci_hi = float(np.percentile(boot_means, (1 - alpha) * 100))

    return {
        "point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_width": ci_hi - ci_lo,
        "std": float(np.std(boot_means)),
        "n_seeds": n_seeds,
        "n_scenarios": n_scenarios,
    }


def format_ci(result: dict, scale: float = 100.0, metric: str = "MF1") -> str:
    """Format CI result as string."""
    p = result["point"] * scale
    lo = result["ci_lo"] * scale
    hi = result["ci_hi"] * scale
    w = result["ci_width"] * scale
    return f"{metric}: {p:.1f} [{lo:.1f}, {hi:.1f}] (width={w:.1f})"


def format_diff(result: dict, scale: float = 100.0) -> str:
    """Format paired diff result as string."""
    p = result["point"] * scale
    lo = result["ci_lo"] * scale
    hi = result["ci_hi"] * scale
    pv = result["p_value"]
    sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
    return f"Δ MF1: {p:+.1f} [{lo:+.1f}, {hi:+.1f}] p={pv:.4f} {sig}"


def analyze_dataset(dataset: str, config: dict, n_replicates: int,
                    multi_seed: bool) -> dict:
    """Run full bootstrap analysis for one dataset."""
    log.info(f"\n{'='*60}")
    log.info(f"  {dataset}")
    log.info(f"{'='*60}")

    results = {"dataset": dataset}

    # --- Single best config ---
    run_dir = RUNS_DIR / config["run_dir"]
    if not run_dir.exists():
        log.info(f"  SKIP: {run_dir} not found")
        return results

    scenario_results = load_scenario_results(run_dir)
    source_f1, translator_f1, names = extract_f1_arrays(scenario_results)

    if len(source_f1) == 0:
        log.info(f"  SKIP: no scenario results found")
        return results

    log.info(f"  Config: {config['run_dir']} ({len(names)} scenarios)")

    # Source-only CI
    src_ci = bootstrap_mean_ci(source_f1, n_replicates)
    log.info(f"  Source-only  {format_ci(src_ci)}")
    results["source_only"] = src_ci

    # Translator CI
    trans_ci = bootstrap_mean_ci(translator_f1, n_replicates)
    log.info(f"  Translator   {format_ci(trans_ci)}")
    results["translator"] = trans_ci

    # Paired difference
    diff_ci = bootstrap_paired_diff(translator_f1, source_f1, n_replicates)
    log.info(f"  {format_diff(diff_ci)}")
    results["improvement"] = diff_ci

    # Compare vs published baseline
    baseline_mf1 = config["best_baseline_mf1"]
    baseline_name = config["best_baseline"]
    trans_point = trans_ci["point"] * 100
    gap = trans_point - baseline_mf1
    log.info(f"  vs {baseline_name} ({baseline_mf1:.1f}): {gap:+.1f}")
    results["vs_baseline"] = {
        "name": baseline_name,
        "mf1": baseline_mf1,
        "gap": gap,
    }

    # --- Multi-seed analysis ---
    if multi_seed:
        seed_translator_f1s = []
        seed_source_f1s = []
        n_seeds = config["n_seeds"]
        available_seeds = 0

        for s in range(n_seeds):
            seed_dir = RUNS_DIR / config["seed_pattern"].format(seed=s)
            if not seed_dir.exists():
                continue
            sr = load_scenario_results(seed_dir)
            sf1, tf1, _ = extract_f1_arrays(sr)
            if len(tf1) == len(names):  # must have all scenarios
                seed_translator_f1s.append(tf1)
                seed_source_f1s.append(sf1)
                available_seeds += 1

        if available_seeds >= 2:
            log.info(f"\n  Multi-seed ({available_seeds} seeds × {len(names)} scenarios):")

            # Translator multi-seed CI
            trans_ms = bootstrap_multiseed(seed_translator_f1s, n_replicates)
            log.info(f"  Translator   {format_ci(trans_ms)}")
            results["translator_multiseed"] = trans_ms

            # Source multi-seed CI
            src_ms = bootstrap_multiseed(seed_source_f1s, n_replicates)
            log.info(f"  Source-only  {format_ci(src_ms)}")
            results["source_only_multiseed"] = src_ms

            # Paired multi-seed: pool all (seed, scenario) diffs
            all_trans = np.concatenate(seed_translator_f1s)
            all_src = np.concatenate(seed_source_f1s)
            diff_ms = bootstrap_paired_diff(all_trans, all_src, n_replicates)
            log.info(f"  {format_diff(diff_ms)} (pooled {len(all_trans)} pairs)")
            results["improvement_multiseed"] = diff_ms

            # Per-seed means
            per_seed = [float(np.mean(tf1)) * 100 for tf1 in seed_translator_f1s]
            log.info(f"  Per-seed MF1: {', '.join(f'{v:.1f}' for v in per_seed)}")
            log.info(f"  Seed mean±std: {np.mean(per_seed):.1f}±{np.std(per_seed):.1f}")
            results["per_seed_mf1"] = per_seed
        else:
            log.info(f"\n  Multi-seed: only {available_seeds} seeds available (need ≥2)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for AdaTime results")
    parser.add_argument("--datasets", nargs="+", default=list(BEST_CONFIGS.keys()),
                        choices=list(BEST_CONFIGS.keys()))
    parser.add_argument("--n-replicates", type=int, default=2000)
    parser.add_argument("--multi-seed", action="store_true",
                        help="Include multi-seed bootstrap analysis")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    log.info(f"AdaTime Bootstrap CIs ({args.n_replicates} replicates, 95% CI)")

    all_results = {}
    for dataset in args.datasets:
        config = BEST_CONFIGS[dataset]
        result = analyze_dataset(dataset, config, args.n_replicates, args.multi_seed)
        all_results[dataset] = result

    # Summary table
    log.info(f"\n{'='*60}")
    log.info("  SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'Dataset':<8} {'MF1':>6} {'95% CI':>16} {'Δ vs SO':>8} {'p-val':>8} {'vs Best E2E':>12}")
    log.info("-" * 60)

    for ds in args.datasets:
        r = all_results[ds]
        if "translator" not in r:
            continue
        t = r["translator"]
        imp = r.get("improvement", {})
        vs = r.get("vs_baseline", {})
        mf1 = t["point"] * 100
        ci = f"[{t['ci_lo']*100:.1f}, {t['ci_hi']*100:.1f}]"
        delta = f"{imp.get('point', 0)*100:+.1f}" if imp else "—"
        pv = f"{imp.get('p_value', 1):.4f}" if imp else "—"
        gap = f"{vs.get('gap', 0):+.1f} vs {vs.get('name', '?')}" if vs else "—"
        log.info(f"{ds:<8} {mf1:>6.1f} {ci:>16} {delta:>8} {pv:>8} {gap:>12}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = json.loads(json.dumps(all_results, default=convert))
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        log.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
