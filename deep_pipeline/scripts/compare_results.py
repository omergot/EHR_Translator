#!/usr/bin/env python3
"""Compare experiment results and produce ranked leaderboards.

Merges full experiment results with screening results into unified tables.

Usage:
    python scripts/compare_results.py --task aki
    python scripts/compare_results.py --task aki --paradigm retrieval
    python scripts/compare_results.py --task sepsis --include-screening
    python scripts/compare_results.py --exps aki_v5_cross3 aki_v5_stride3 aki_v5_k24
    python scripts/compare_results.py --all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_convergence import infer_task, infer_paradigm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "experiments" / "results"
SCREEN_RESULTS_DIR = REPO / "experiments" / "screening_results"
QUEUE_PATH = REPO / "experiments" / "queue.yaml"

# Baselines per task (frozen LSTM on eICU data, no translation)
BASELINES = {
    "mortality": {"AUCROC": 0.8079, "AUCPR": 0.2965},
    "aki":       {"AUCROC": 0.8558, "AUCPR": 0.5678},
    "sepsis":    {"AUCROC": 0.7159, "AUCPR": 0.0600},
    "los":       {"MAE": 0.2527},
    "kf":        {"MAE": 0.0330},
}

# Classification vs regression tasks
REGRESSION_TASKS = {"los", "kf"}


def load_full_results() -> dict:
    """Load all full experiment results from experiments/results/*.json."""
    results = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if data.get("status") != "ok":
                continue
            # Skip screening results that ended up here
            if f.stem.startswith("screen_") or f.stem.startswith("cal_"):
                continue
            results[f.stem] = data
        except (json.JSONDecodeError, OSError):
            pass
    return results


def load_screening_results() -> dict:
    """Load all screening results from experiments/screening_results/*.json."""
    results = {}
    if not SCREEN_RESULTS_DIR.exists():
        return results
    for f in sorted(SCREEN_RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            results[f.stem] = data
        except (json.JSONDecodeError, OSError):
            pass
    return results


def load_queue_results() -> dict:
    """Load results embedded in queue entries."""
    if not QUEUE_PATH.exists():
        return {}
    queue = yaml.safe_load(QUEUE_PATH.read_text())
    results = {}
    for exp in queue.get("experiments", []):
        if exp.get("results"):
            results[exp["name"]] = {
                "difference": exp["results"],
                "config": exp.get("config", ""),
                "status": exp.get("status", ""),
            }
    return results


def _get_paradigm_from_queue(name: str) -> str:
    """Look up paradigm from queue config.

    Result file names may include task suffix (e.g., aki_v5_cross3_aki)
    while queue entries don't (aki_v5_cross3). Try both.
    """
    if not QUEUE_PATH.exists():
        return infer_paradigm("", name)
    queue = yaml.safe_load(QUEUE_PATH.read_text())

    # Try exact match, then stripped of task suffix
    candidates = [name]
    for task in ("mortality", "aki", "sepsis", "los", "kf", "kidney_function"):
        if name.endswith(f"_{task}"):
            candidates.append(name[: -(len(task) + 1)])
            break

    for exp in queue.get("experiments", []):
        if exp.get("name") in candidates:
            config_path = REPO / exp.get("config", "")
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                    ttype = config.get("translator", {}).get("type", "")
                    return infer_paradigm(ttype, name)
                except (json.JSONDecodeError, OSError):
                    pass
            break
    return infer_paradigm("", name)


def build_leaderboard(task: str, paradigm: str = None,
                      include_screening: bool = False,
                      specific_exps: list = None) -> list:
    """Build a ranked list of experiments for a task (+ optional paradigm filter).

    Returns list of dicts: {name, aucroc_delta, aucpr_delta, status, paradigm, screen_info}
    """
    full_results = load_full_results()
    queue_results = load_queue_results()
    screen_results = load_screening_results() if include_screening else {}

    entries = []

    # Process full results
    for name, data in full_results.items():
        result_task = data.get("task", infer_task(name))
        if result_task != task:
            continue

        exp_paradigm = _get_paradigm_from_queue(name)
        if paradigm and exp_paradigm != paradigm:
            continue

        if specific_exps and name not in specific_exps:
            # Also check without task suffix
            base = name.rsplit(f"_{task}", 1)[0] if name.endswith(f"_{task}") else name
            if base not in specific_exps:
                continue

        diff = data.get("difference", {})
        entry = {
            "name": name,
            "paradigm": exp_paradigm,
            "status": "done",
            "screen_info": None,
        }

        if task in REGRESSION_TASKS:
            entry["mae_delta"] = diff.get("MAE", None)
            entry["mse_delta"] = diff.get("MSE", None)
            entry["r2_delta"] = diff.get("R2", None)
        else:
            entry["aucroc_delta"] = diff.get("AUCROC", None)
            entry["aucpr_delta"] = diff.get("AUCPR", None)
            entry["loss_delta"] = diff.get("loss", None)

        entries.append(entry)

    # Also check queue for results not yet in JSON files
    for name, data in queue_results.items():
        q_task = infer_task(name)
        if q_task != task:
            continue
        # Skip if already in full results
        if any(e["name"] == name or e["name"] == f"{name}_{task}" for e in entries):
            continue
        if name.startswith("screen_") or name.startswith("cal_"):
            continue

        exp_paradigm = _get_paradigm_from_queue(name)
        if paradigm and exp_paradigm != paradigm:
            continue
        if specific_exps and name not in specific_exps:
            continue

        diff = data.get("difference", {})
        entry = {
            "name": name,
            "paradigm": exp_paradigm,
            "status": data.get("status", "done"),
            "screen_info": None,
        }

        if task in REGRESSION_TASKS:
            entry["mae_delta"] = diff.get("MAE", None)
        else:
            entry["aucroc_delta"] = diff.get("AUCROC", None)
            entry["aucpr_delta"] = diff.get("AUCPR", None)

        entries.append(entry)

    # Process screening results
    if include_screening:
        for name, data in screen_results.items():
            if data.get("task") != task:
                continue
            exp_paradigm = data.get("paradigm", "unknown")
            if paradigm and exp_paradigm != paradigm:
                continue

            orig_config = Path(data.get("original_config", "")).stem
            if specific_exps and orig_config not in specific_exps:
                continue

            entry = {
                "name": orig_config or name,
                "paradigm": exp_paradigm,
                "status": "screened",
                "screen_info": {
                    "percentile": data.get("reference_percentile"),
                    "recommendation": data.get("recommendation"),
                    "val_task_final": data.get("val_task_final"),
                    "epochs": data.get("screening_epochs"),
                },
            }

            if task in REGRESSION_TASKS:
                entry["mae_delta"] = None
            else:
                entry["aucroc_delta"] = None
                entry["aucpr_delta"] = None

            entries.append(entry)

    # Sort by primary metric
    if task in REGRESSION_TASKS:
        # MAE delta: more negative = better
        entries.sort(key=lambda x: x.get("mae_delta") or 0)
    else:
        # AUCROC delta: higher = better
        entries.sort(key=lambda x: -(x.get("aucroc_delta") or -999))

    return entries


def print_leaderboard(entries: list, task: str, paradigm: str = None):
    """Print a formatted leaderboard table."""
    if not entries:
        print(f"  No experiments found for {task}" +
              (f"/{paradigm}" if paradigm else ""))
        return

    paradigm_str = f" / {paradigm}" if paradigm else ""
    print(f"\n{'='*90}")
    print(f"  {task.upper()}{paradigm_str} Leaderboard")
    print(f"{'='*90}")

    is_regression = task in REGRESSION_TASKS

    if is_regression:
        baseline_mae = BASELINES.get(task, {}).get("MAE")
        if baseline_mae:
            print(f"  Baseline MAE: {baseline_mae:.4f}")
        print(f"\n  {'Rank':<5} {'Experiment':<40} {'MAE Δ':>10} {'Status':>10} {'Screen':>15}")
        print(f"  {'-'*82}")
    else:
        baseline = BASELINES.get(task, {})
        if baseline:
            print(f"  Baselines: AUCROC={baseline.get('AUCROC', '?')}, "
                  f"AUCPR={baseline.get('AUCPR', '?')}")
        print(f"\n  {'Rank':<5} {'Experiment':<40} {'AUCROC Δ':>10} {'AUCPR Δ':>10} "
              f"{'Status':>10} {'Screen':>15}")
        print(f"  {'-'*92}")

    for i, entry in enumerate(entries, 1):
        name = entry["name"]
        status = entry["status"]
        screen = entry.get("screen_info")

        if is_regression:
            mae = entry.get("mae_delta")
            mae_str = f"{mae:+.4f}" if mae is not None else "-"
            screen_str = ""
            if screen:
                rec = screen.get("recommendation", "?")
                pct = screen.get("percentile", "?")
                screen_str = f"{rec} (p{pct})"
            print(f"  {i:<5} {name:<40} {mae_str:>10} {status:>10} {screen_str:>15}")
        else:
            aucroc = entry.get("aucroc_delta")
            aucpr = entry.get("aucpr_delta")
            aucroc_str = f"{aucroc:+.4f}" if aucroc is not None else "-"
            aucpr_str = f"{aucpr:+.4f}" if aucpr is not None else "-"
            screen_str = ""
            if screen:
                rec = screen.get("recommendation", "?")
                pct = screen.get("percentile", "?")
                screen_str = f"{rec} (p{pct})"
            print(f"  {i:<5} {name:<40} {aucroc_str:>10} {aucpr_str:>10} "
                  f"{status:>10} {screen_str:>15}")

    print(f"\n  Total: {len(entries)} experiments")

    # Breakdown by paradigm
    paradigms = {}
    for e in entries:
        p = e.get("paradigm", "unknown")
        paradigms[p] = paradigms.get(p, 0) + 1
    if len(paradigms) > 1:
        parts = [f"{k}={v}" for k, v in sorted(paradigms.items())]
        print(f"  By paradigm: {', '.join(parts)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiment results and produce leaderboards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", type=str,
                        choices=["mortality", "aki", "sepsis", "los", "kf"],
                        help="Task to show leaderboard for")
    parser.add_argument("--paradigm", type=str,
                        choices=["delta", "sl", "retrieval"],
                        help="Filter by paradigm")
    parser.add_argument("--include-screening", action="store_true",
                        help="Include screening results in leaderboard")
    parser.add_argument("--exps", nargs="+",
                        help="Show only specific experiments")
    parser.add_argument("--all", action="store_true",
                        help="Show leaderboards for all tasks")
    parser.add_argument("--top", type=int, default=0,
                        help="Show only top N entries")

    args = parser.parse_args()

    if args.all:
        for task in ["mortality", "aki", "sepsis", "los", "kf"]:
            entries = build_leaderboard(task, args.paradigm,
                                       args.include_screening)
            if entries:
                if args.top:
                    entries = entries[:args.top]
                print_leaderboard(entries, task, args.paradigm)
        return

    if not args.task and not args.exps:
        parser.print_help()
        sys.exit(1)

    if args.exps and not args.task:
        # Try to infer task from first experiment name
        args.task = infer_task(args.exps[0])
        if args.task == "unknown":
            parser.error("Cannot infer task from experiment names. Use --task.")

    entries = build_leaderboard(
        args.task, args.paradigm, args.include_screening, args.exps
    )
    if args.top:
        entries = entries[:args.top]
    print_leaderboard(entries, args.task, args.paradigm)


if __name__ == "__main__":
    main()
