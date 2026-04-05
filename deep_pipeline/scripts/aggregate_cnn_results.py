#!/usr/bin/env python3
"""Aggregate AdaTime CNN results and produce comparison table.

Reads per-scenario results from runs/adatime_cnn/{dataset}/{src}_to_{trg}/results.json
and saves a combined JSON + markdown comparison table.

Usage:
    python scripts/aggregate_cnn_results.py
    python scripts/aggregate_cnn_results.py --datasets HAR HHAR WISDM
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

WORKTREE = Path(__file__).resolve().parent.parent

# AdaTime published numbers (from van de Water et al., ICLR 2024 / AdaTime paper)
ADATIME_PUBLISHED = {
    "HAR": {
        "source_only": 72.09,
        "DANN": 80.49,
        "DIRT-T": 83.27,
        "Deep_Coral": 79.99,
        "CDAN": 82.68,
        "CoDATS": 83.45,
        "best_adatime": 83.45,
    },
    "HHAR": {
        "source_only": 70.17,
        "DANN": 77.88,
        "DIRT-T": 81.09,
        "Deep_Coral": 79.07,
        "CDAN": 81.16,
        "CoDATS": 82.95,
        "best_adatime": 82.95,
    },
    "WISDM": {
        "source_only": 48.53,
        "DANN": 56.21,
        "DIRT-T": 63.17,
        "Deep_Coral": 64.28,
        "CDAN": 57.12,
        "CoDATS": 67.34,
        "best_adatime": 67.34,
    },
}

DATASET_SCENARIOS = {
    "HAR": [
        ("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"),
        ("12", "16"), ("18", "27"), ("20", "5"), ("24", "8"),
        ("28", "27"), ("30", "20"),
    ],
    "HHAR": [
        ("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5"),
        ("5", "0"), ("6", "1"), ("7", "4"), ("8", "3"), ("0", "2"),
    ],
    "WISDM": [
        ("7", "18"), ("20", "30"), ("35", "31"), ("17", "23"), ("6", "19"),
        ("2", "11"), ("33", "12"), ("5", "26"), ("28", "4"), ("23", "32"),
    ],
}


def load_dataset_results(dataset: str, base_dir: Path) -> dict:
    """Load all per-scenario results for a dataset."""
    scenarios = DATASET_SCENARIOS.get(dataset, [])
    results = {}
    missing = []

    for src, trg in scenarios:
        scenario_key = f"{src}_to_{trg}"
        result_file = base_dir / dataset / scenario_key / "results.json"

        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            results[scenario_key] = data
            logger.info("  %s: loaded (methods: %s)", scenario_key, list(data.keys()))
        else:
            missing.append(scenario_key)
            logger.warning("  %s: MISSING (%s)", scenario_key, result_file)

    if missing:
        logger.warning("Missing %d/%d scenarios for %s: %s", len(missing), len(scenarios), dataset, missing)

    return results


def compute_mean_metrics(results: dict, method_key: str) -> dict:
    """Compute mean metrics across all scenarios for a method."""
    accs = []
    f1s = []
    aurocs = []

    for scenario, scenario_results in results.items():
        if method_key in scenario_results:
            m = scenario_results[method_key]
            accs.append(m.get("accuracy", 0) * 100)  # Convert to percentage
            f1s.append(m.get("f1", 0) * 100)
            aurocs.append(m.get("auroc", 0) * 100)

    if not accs:
        return {"accuracy": None, "f1": None, "auroc": None, "n": 0}

    return {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "auroc": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "n": len(accs),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate AdaTime CNN results")
    parser.add_argument("--datasets", nargs="+", default=["HAR", "HHAR", "WISDM"])
    parser.add_argument("--cnn-dir", default=None, help="Override CNN results directory")
    args = parser.parse_args()

    cnn_base = Path(args.cnn_dir) if args.cnn_dir else WORKTREE / "runs" / "adatime_cnn"
    output_dir = WORKTREE / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dataset_results = {}
    all_means = {}

    for dataset in args.datasets:
        logger.info("\n=== %s ===", dataset)
        dataset_results = load_dataset_results(dataset, cnn_base)
        all_dataset_results[dataset] = dataset_results

        # Compute means for both methods
        all_means[dataset] = {
            "source_only_cnn": compute_mean_metrics(dataset_results, "source_only_cnn"),
            "translator_cnn": compute_mean_metrics(dataset_results, "translator_cnn"),
        }

    # Save combined JSON
    output_json = output_dir / "adatime_cnn_results.json"
    combined = {
        "datasets": all_dataset_results,
        "summary": all_means,
    }
    with open(output_json, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info("\nSaved to %s", output_json)

    # Generate comparison table
    table_lines = [
        "# AdaTime CNN Results — Comparison Table",
        "",
        "| Method | Constraint | HAR MF1 | HHAR MF1 | WISDM MF1 |",
        "|--------|-----------|---------|----------|-----------|",
    ]

    # Published AdaTime numbers
    for method_name, key in [
        ("Source-only (AdaTime published)", "source_only"),
        ("DANN (AdaTime published)", "DANN"),
        ("Deep CORAL (AdaTime published)", "Deep_Coral"),
        ("CDAN (AdaTime published)", "CDAN"),
        ("CoDATS (AdaTime published)", "CoDATS"),
        ("DIRT-T (AdaTime published)", "DIRT-T"),
    ]:
        values = []
        for ds in args.datasets:
            pub = ADATIME_PUBLISHED.get(ds, {})
            v = pub.get(key, None)
            values.append(f"{v:.2f}" if v is not None else "—")
        table_lines.append(
            f"| {method_name} | None | {' | '.join(values)} |"
        )

    table_lines.append("|--------|-----------|---------|----------|-----------|")

    # Our CNN results
    for method_key, label in [
        ("source_only_cnn", "Source-only (Ours, frozen CNN)"),
        ("translator_cnn", "**Translator (Ours, frozen CNN)**"),
    ]:
        values = []
        for ds in args.datasets:
            m = all_means.get(ds, {}).get(method_key, {})
            f1_mean = m.get("f1", None)
            f1_std = m.get("f1_std", None)
            n = m.get("n", 0)
            if f1_mean is not None and n > 0:
                values.append(f"{f1_mean:.2f} ±{f1_std:.2f} (n={n})")
            else:
                values.append("— (pending)")
        table_lines.append(
            f"| {label} | Frozen CNN | {' | '.join(values)} |"
        )

    table_lines += [
        "",
        "## Per-Scenario Results",
        "",
        "MF1 = Macro-F1 (%)",
        "",
    ]

    for dataset in args.datasets:
        table_lines.append(f"### {dataset}")
        table_lines.append("")
        table_lines.append("| Scenario | Source-only MF1 | Translator MF1 | Delta |")
        table_lines.append("|----------|----------------|----------------|-------|")

        for src, trg in DATASET_SCENARIOS.get(dataset, []):
            key = f"{src}_to_{trg}"
            scenario_res = all_dataset_results.get(dataset, {}).get(key, {})
            src_f1 = scenario_res.get("source_only_cnn", {}).get("f1", None)
            tr_f1 = scenario_res.get("translator_cnn", {}).get("f1", None)

            src_str = f"{src_f1 * 100:.2f}" if src_f1 is not None else "—"
            tr_str = f"{tr_f1 * 100:.2f}" if tr_f1 is not None else "—"

            if src_f1 is not None and tr_f1 is not None:
                delta = (tr_f1 - src_f1) * 100
                delta_str = f"{delta:+.2f}"
            else:
                delta_str = "—"

            table_lines.append(f"| {src}→{trg} | {src_str} | {tr_str} | {delta_str} |")

        table_lines.append("")

    table_str = "\n".join(table_lines)
    table_path = output_dir / "adatime_cnn_comparison.md"
    with open(table_path, "w") as f:
        f.write(table_str)
    logger.info("Comparison table saved to %s", table_path)

    # Print summary to console
    logger.info("\n" + "=" * 70)
    logger.info("CNN RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("%-30s %8s %8s %8s", "Method", "HAR MF1", "HHAR MF1", "WISDM MF1")
    logger.info("-" * 70)
    for method_key, label in [
        ("source_only_cnn", "Source-only (frozen CNN)"),
        ("translator_cnn", "Translator (frozen CNN)"),
    ]:
        vals = []
        for ds in args.datasets:
            m = all_means.get(ds, {}).get(method_key, {})
            f1_mean = m.get("f1", None)
            n = m.get("n", 0)
            vals.append(f"{f1_mean:.2f}%" if f1_mean is not None and n > 0 else "—")
        logger.info("%-30s %8s %8s %8s", label, *vals)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
