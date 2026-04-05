#!/usr/bin/env python3
"""Calibrate screening: validate that short runs predict final outcome rankings.

Runs a set of known experiments for a reduced number of epochs and checks
whether the val_task ranking at the screening epoch correlates with the
known final AUCROC delta ranking.

Usage:
    python scripts/calibrate_screening.py --task aki --paradigm retrieval --epochs 5
    python scripts/calibrate_screening.py --task sepsis --paradigm retrieval --epochs 3
    python scripts/calibrate_screening.py --task aki --paradigm retrieval --epochs 5 --submit
    python scripts/calibrate_screening.py --results  # show results from completed calibration

All GPU work goes through experiments/queue.yaml + gpu_scheduler.py.
"""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import yaml
    from scipy import stats
except ImportError:
    print("Required: pip install pyyaml scipy")
    sys.exit(1)

# Reuse log parsing from analyze_convergence
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_convergence import (
    parse_log, parse_all_logs, parse_all_results,
    infer_task, infer_paradigm, get_val_task_trajectory,
    ExperimentLog,
)
from manage_pretrain import find_match, auto_copy, REPO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

QUEUE_PATH = REPO / "experiments" / "queue.yaml"
CALIBRATION_DIR = REPO / "experiments" / "calibration"
CALIBRATION_RESULTS = CALIBRATION_DIR / "calibration_results.json"

# Known experiment configs and their final AUCROC deltas
# These are populated dynamically from experiments/results/ and queue.yaml
KNOWN_CONFIGS = {
    "aki_retrieval": {
        "aki_v5_cross3":       {"config": "configs/aki_v5_cross3.json",       "aucroc_delta": 0.0556},
        "aki_v5_stride3":      {"config": "configs/aki_v5_stride3.json",      "aucroc_delta": 0.0498},
        "aki_v5_k24":          {"config": "configs/aki_v5_k24.json",          "aucroc_delta": 0.0491},
        "aki_v5_stride1":      {"config": "configs/aki_v5_stride1.json",      "aucroc_delta": 0.0448},
        "aki_v5_fix_only":     {"config": "configs/aki_v5_fix_only.json",     "aucroc_delta": 0.0446},
    },
    "sepsis_retrieval": {
        "sepsis_retr_v4_mmd":        {"config": "configs/sepsis_retr_v4_mmd.json",        "aucroc_delta": 0.0512},
        "sepsis_retr_fg_no_smooth":  {"config": "configs/sepsis_retr_fg_no_smooth.json",  "aucroc_delta": 0.0488},
        "sepsis_retr_v5_cross3":     {"config": "configs/sepsis_retr_v5_cross3.json",     "aucroc_delta": 0.0448},
        "sepsis_retrieval_full":     {"config": "configs/sepsis_retrieval_full.json",     "aucroc_delta": 0.0330},
    },
}


def _load_queue() -> dict:
    with open(QUEUE_PATH) as f:
        return yaml.safe_load(f)


def _save_queue(queue: dict):
    """Atomic write to queue file."""
    tmp = str(QUEUE_PATH) + ".tmp"
    with open(tmp, "w") as f:
        f.write("# ============================================================\n")
        f.write("# Experiment Queue — edit this file to manage experiments\n")
        f.write("# Reorder pending experiments by moving entries up/down\n")
        f.write("# The scheduler runs top-to-bottom through pending experiments\n")
        f.write("# ============================================================\n\n")
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=120)
    import os
    os.rename(tmp, str(QUEUE_PATH))


def _discover_known_experiments(task: str, paradigm: str) -> dict:
    """Discover known experiments from results and queue for a task+paradigm."""
    # First check hardcoded configs
    key = f"{task}_{paradigm}"
    if key in KNOWN_CONFIGS:
        # Verify configs exist
        known = {}
        for name, info in KNOWN_CONFIGS[key].items():
            config_path = REPO / info["config"]
            if config_path.exists():
                known[name] = info
            else:
                log.warning(f"Config not found for {name}: {config_path}")
        if known:
            return known

    # Fall back to discovering from results + queue
    queue = _load_queue()
    results_dir = REPO / "experiments" / "results"
    known = {}

    for exp in queue.get("experiments", []):
        if exp.get("status") not in ("done",):
            continue
        results = exp.get("results", {})
        if "AUCROC" not in results:
            continue

        name = exp["name"]
        config_rel = exp.get("config", "")
        config_path = REPO / config_rel

        if not config_path.exists():
            continue

        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        exp_task = infer_task(name)
        ttype = config.get("translator", {}).get("type", "")
        exp_paradigm = infer_paradigm(ttype, name)

        if exp_task == task and exp_paradigm == paradigm:
            known[name] = {
                "config": config_rel,
                "aucroc_delta": results["AUCROC"],
            }

    return known


def create_screening_config(original_config_path: str, screening_epochs: int,
                            run_name: str) -> Path:
    """Create a config with reduced epochs for screening."""
    config_path = REPO / original_config_path
    config = json.loads(config_path.read_text())

    # Override epochs
    config["training"]["epochs"] = screening_epochs

    # Set screening run_dir
    run_dir = f"runs/calibration_{run_name}"
    config["output"]["run_dir"] = run_dir
    config["output"]["log_file"] = f"{run_dir}/run.log"

    # Disable early stopping for screening (want exact epoch count)
    config["training"]["early_stopping_patience"] = screening_epochs + 10

    # Write screening config
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    screen_config_path = CALIBRATION_DIR / f"{run_name}.json"
    screen_config_path.write_text(json.dumps(config, indent=2))

    return screen_config_path


def submit_calibration(task: str, paradigm: str, screening_epochs: int,
                       server: str = "local"):
    """Create screening configs and add calibration entries to the queue."""
    known = _discover_known_experiments(task, paradigm)
    if len(known) < 3:
        log.error(f"Need at least 3 known experiments for calibration, found {len(known)}")
        sys.exit(1)

    log.info(f"Calibration for {task}/{paradigm}: {len(known)} experiments, "
             f"{screening_epochs} epochs each")

    queue = _load_queue()
    experiments = queue.get("experiments", [])
    existing_names = {e["name"] for e in experiments}

    submitted = []
    for name, info in known.items():
        cal_name = f"cal_{name}_ep{screening_epochs}"
        if cal_name in existing_names:
            log.info(f"  {cal_name} already in queue, skipping")
            continue

        # Create screening config
        screen_config = create_screening_config(
            info["config"], screening_epochs, cal_name
        )
        log.info(f"  Created config: {screen_config}")

        # Auto-copy pretrain checkpoint
        dest = auto_copy(str(screen_config))
        if dest:
            log.info(f"  Pretrain checkpoint: {dest}")
        else:
            log.warning(f"  No pretrain checkpoint found for {cal_name}")

        # Add to queue
        entry = {
            "name": cal_name,
            "config": str(screen_config.relative_to(REPO)),
            "output": f"experiments/results/{cal_name}.parquet",
            "status": "calibration",
            "server": server,
            "notes": f"Calibration: {name} @ {screening_epochs}ep (known AUCROC Δ={info['aucroc_delta']:+.4f})",
        }

        # Insert before non-pending entries
        insert_idx = len(experiments)
        for i, exp in enumerate(experiments):
            if exp.get("status") not in ("pending", "screening", "calibration"):
                insert_idx = i
                break
        experiments.insert(insert_idx, entry)
        submitted.append(cal_name)

    _save_queue(queue)
    log.info(f"Submitted {len(submitted)} calibration experiments to queue")
    print(f"\nSubmitted {len(submitted)} calibration experiments.")
    print("Run the scheduler to execute them: python scripts/gpu_scheduler.py")

    # Save calibration metadata
    meta = {
        "task": task,
        "paradigm": paradigm,
        "screening_epochs": screening_epochs,
        "known_experiments": known,
        "submitted": submitted,
        "server": server,
    }
    meta_path = CALIBRATION_DIR / f"calibration_meta_{task}_{paradigm}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"Saved calibration metadata to {meta_path}")


def collect_calibration_results(task: str, paradigm: str):
    """Collect results from completed calibration runs and compute correlations."""
    meta_path = CALIBRATION_DIR / f"calibration_meta_{task}_{paradigm}.json"
    if not meta_path.exists():
        log.error(f"No calibration metadata found at {meta_path}")
        log.error("Run with --submit first to create calibration experiments")
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    known = meta["known_experiments"]
    screening_epochs = meta["screening_epochs"]

    # Parse logs for calibration experiments
    log_dir = REPO / "experiments" / "logs"
    results = {}

    for orig_name, info in known.items():
        cal_name = f"cal_{orig_name}_ep{screening_epochs}"
        task_suffix = infer_task(orig_name)

        # Try to find the log
        log_path = log_dir / f"{cal_name}_{task_suffix}.log"
        if not log_path.exists():
            log.warning(f"Log not found: {log_path}")
            continue

        exp_log = parse_log(log_path)
        if exp_log is None:
            log.warning(f"Failed to parse log: {log_path}")
            continue

        traj = get_val_task_trajectory(exp_log)
        if not traj:
            log.warning(f"No val_task trajectory for {cal_name}")
            continue

        results[orig_name] = {
            "val_task_trajectory": traj,
            "known_aucroc_delta": info["aucroc_delta"],
        }

    if len(results) < 3:
        log.error(f"Need at least 3 completed calibration runs, found {len(results)}")
        print(f"\nOnly {len(results)}/{len(known)} calibration runs completed.")
        print("Wait for more runs to finish, then run --results again.")
        return

    # Compute Spearman correlation at various epoch checkpoints
    names = sorted(results.keys())
    known_deltas = [results[n]["known_aucroc_delta"] for n in names]
    all_epochs = set()
    for r in results.values():
        all_epochs.update(r["val_task_trajectory"].keys())

    print(f"\n{'='*70}")
    print(f"  Calibration Results: {task}/{paradigm}")
    print(f"  {len(results)} experiments, screening {screening_epochs} epochs")
    print(f"{'='*70}\n")

    # Ranking comparison table
    print("  Known AUCROC Δ ranking (higher = better):")
    for i, n in enumerate(sorted(names, key=lambda x: results[x]["known_aucroc_delta"], reverse=True), 1):
        print(f"    {i}. {n:<30} AUCROC Δ = {results[n]['known_aucroc_delta']:+.4f}")

    print()

    # Correlation at each epoch
    correlations = {}
    for ep in sorted(all_epochs):
        val_tasks = []
        valid_deltas = []
        valid_names = []
        for n in names:
            traj = results[n]["val_task_trajectory"]
            if ep in traj:
                val_tasks.append(traj[ep])
                valid_deltas.append(results[n]["known_aucroc_delta"])
                valid_names.append(n)

        if len(val_tasks) < 3:
            continue

        # val_task is a loss (lower = better), AUCROC delta (higher = better)
        # So we expect NEGATIVE Spearman correlation
        rho, pval = stats.spearmanr(val_tasks, valid_deltas)
        correlations[ep] = {"rho": rho, "pval": pval, "n": len(val_tasks)}

    print("  Spearman ρ(val_task, AUCROC Δ) by epoch:")
    print(f"  {'Epoch':<8} {'ρ':>8} {'p-value':>10} {'n':>4} {'Signal':>10}")
    print(f"  {'-'*44}")
    for ep in sorted(correlations.keys()):
        c = correlations[ep]
        # Negative ρ means lower val_task → higher AUCROC Δ (good)
        signal = "STRONG" if c["rho"] <= -0.7 else "MODERATE" if c["rho"] <= -0.5 else "WEAK"
        if c["pval"] > 0.1:
            signal = "NS"
        print(f"  {ep:<8} {c['rho']:>8.3f} {c['pval']:>10.4f} {c['n']:>4} {signal:>10}")

    # Epoch-specific ranking at screening_epochs
    if screening_epochs in correlations:
        c = correlations[screening_epochs]
        print(f"\n  At screening epoch {screening_epochs}: ρ = {c['rho']:.3f}, p = {c['pval']:.4f}")
        if abs(c["rho"]) >= 0.5:
            print(f"  ✓ Screening is USABLE (|ρ| = {abs(c['rho']):.3f} ≥ 0.5)")
        else:
            print(f"  ✗ Screening is UNRELIABLE (|ρ| = {abs(c['rho']):.3f} < 0.5)")

        # Print val_task ranking at screening epoch
        print(f"\n  Val_task ranking at epoch {screening_epochs} (lower = better):")
        ranked = []
        for n in names:
            traj = results[n]["val_task_trajectory"]
            if screening_epochs in traj:
                ranked.append((n, traj[screening_epochs], results[n]["known_aucroc_delta"]))
        ranked.sort(key=lambda x: x[1])
        for i, (n, vt, delta) in enumerate(ranked, 1):
            print(f"    {i}. {n:<30} val_task={vt:.4f}  (AUCROC Δ={delta:+.4f})")

    print()

    # Save results
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "task": task,
        "paradigm": paradigm,
        "screening_epochs": screening_epochs,
        "n_experiments": len(results),
        "correlations": {str(k): v for k, v in correlations.items()},
        "per_experiment": {
            n: {
                "val_task_trajectory": {str(k): v for k, v in r["val_task_trajectory"].items()},
                "known_aucroc_delta": r["known_aucroc_delta"],
            }
            for n, r in results.items()
        },
    }
    output_path = CALIBRATION_DIR / f"calibration_results_{task}_{paradigm}.json"
    output_path.write_text(json.dumps(output, indent=2))
    log.info(f"Saved calibration results to {output_path}")


def show_all_results():
    """Show results from all completed calibration runs."""
    for f in sorted(CALIBRATION_DIR.glob("calibration_results_*.json")):
        data = json.loads(f.read_text())
        task = data["task"]
        paradigm = data["paradigm"]
        ep = data["screening_epochs"]
        n = data["n_experiments"]
        corr = data.get("correlations", {})
        ep_corr = corr.get(str(ep), {})
        rho = ep_corr.get("rho", float("nan"))
        print(f"  {task}/{paradigm}: {n} experiments, ρ={rho:.3f} at epoch {ep}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate screening protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", type=str, choices=["aki", "sepsis", "mortality", "los", "kf"],
                        help="Task to calibrate")
    parser.add_argument("--paradigm", type=str, choices=["retrieval", "sl", "delta"],
                        help="Paradigm to calibrate")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of screening epochs (default: 5)")
    parser.add_argument("--server", type=str, default="local",
                        help="Server to run calibration on (default: local)")
    parser.add_argument("--submit", action="store_true",
                        help="Create configs and submit calibration jobs to queue")
    parser.add_argument("--results", action="store_true",
                        help="Collect and display calibration results")
    parser.add_argument("--results-all", action="store_true",
                        help="Show results from all calibration runs")

    args = parser.parse_args()

    if args.results_all:
        show_all_results()
        return

    if args.results:
        if not args.task or not args.paradigm:
            parser.error("--results requires --task and --paradigm")
        collect_calibration_results(args.task, args.paradigm)
        return

    if args.submit:
        if not args.task or not args.paradigm:
            parser.error("--submit requires --task and --paradigm")
        submit_calibration(args.task, args.paradigm, args.epochs, args.server)
        return

    # Default: show known experiments
    if args.task and args.paradigm:
        known = _discover_known_experiments(args.task, args.paradigm)
        print(f"\nKnown {args.task}/{args.paradigm} experiments ({len(known)}):")
        for name, info in sorted(known.items(), key=lambda x: x[1]["aucroc_delta"], reverse=True):
            config_exists = (REPO / info["config"]).exists()
            status = "OK" if config_exists else "MISSING CONFIG"
            print(f"  {name:<35} AUCROC Δ={info['aucroc_delta']:+.4f}  [{status}]")
        print(f"\nUse --submit to create calibration experiments")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
