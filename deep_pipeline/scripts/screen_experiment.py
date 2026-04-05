#!/usr/bin/env python3
"""Screen an experiment config with reduced epochs to predict viability.

Equivalent of autoresearch's "run for N epochs and check val_task": creates a
screening config (reduced epochs + pretrain reuse), submits it through the
experiment queue, waits for completion, then compares the result against
the reference distribution from existing experiments.

Usage:
    python scripts/screen_experiment.py --config configs/aki_v5_new_idea.json
    python scripts/screen_experiment.py --config configs/sepsis_new.json --epochs 5
    python scripts/screen_experiment.py --config configs/aki_new.json --server 3090
    python scripts/screen_experiment.py --config configs/aki_new.json --submit-only
    python scripts/screen_experiment.py --collect screen_aki_new_ep5

All GPU work goes through experiments/queue.yaml + gpu_scheduler.py.
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import yaml
    from scipy import stats
except ImportError:
    print("Required: pip install pyyaml scipy")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_convergence import (
    parse_log, parse_all_logs, infer_task, infer_paradigm,
    get_val_task_trajectory, ExperimentLog,
)
from manage_pretrain import auto_copy, fingerprint_from_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
QUEUE_PATH = REPO / "experiments" / "queue.yaml"
SCREEN_DIR = REPO / "experiments" / "screening"
SCREEN_RESULTS_DIR = REPO / "experiments" / "screening_results"

# Default screening epochs per task+paradigm (calibrated values)
DEFAULT_SCREENING_EPOCHS = {
    ("aki", "retrieval"): 5,
    ("aki", "sl"): 3,
    ("sepsis", "retrieval"): 3,
    ("sepsis", "sl"): 5,
    ("sepsis", "delta"): 5,
    ("los", "retrieval"): 5,
    ("kf", "retrieval"): 5,
}

# Tasks where screening is not recommended (no early signal)
NO_SCREEN_TASKS = {"mortality"}


def _load_queue() -> dict:
    with open(QUEUE_PATH) as f:
        return yaml.safe_load(f)


def _save_queue(queue: dict):
    tmp = str(QUEUE_PATH) + ".tmp"
    with open(tmp, "w") as f:
        f.write("# ============================================================\n")
        f.write("# Experiment Queue — edit this file to manage experiments\n")
        f.write("# Reorder pending experiments by moving entries up/down\n")
        f.write("# The scheduler runs top-to-bottom through pending experiments\n")
        f.write("# ============================================================\n\n")
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=120)
    os.rename(tmp, str(QUEUE_PATH))


def _infer_task_from_config(config: dict, config_path: str) -> str:
    """Infer task from config, falling back to data_dir."""
    task = infer_task(Path(config_path).stem)
    if task == "unknown":
        task = infer_task(config.get("data_dir", ""))
    return task


def _infer_paradigm_from_config(config: dict) -> str:
    """Infer paradigm from config translator type."""
    ttype = config.get("translator", {}).get("type", "")
    return infer_paradigm(ttype, "")


def create_screening_config(config_path: str, screening_epochs: int,
                            screen_name: str) -> Path:
    """Create a reduced-epoch config for screening."""
    abs_path = Path(config_path)
    if not abs_path.is_absolute():
        abs_path = REPO / config_path

    config = json.loads(abs_path.read_text())

    # Override epochs
    config["training"]["epochs"] = screening_epochs

    # Disable early stopping (want exact epoch count)
    config["training"]["early_stopping_patience"] = screening_epochs + 10

    # Set screening run_dir
    run_dir = f"runs/{screen_name}"
    config["output"]["run_dir"] = run_dir
    config["output"]["log_file"] = f"{run_dir}/run.log"

    # Write screening config
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    screen_config_path = SCREEN_DIR / f"{screen_name}.json"
    screen_config_path.write_text(json.dumps(config, indent=2))

    return screen_config_path


def build_reference_distribution(task: str, paradigm: str,
                                 screening_epoch: int) -> dict:
    """Build reference distribution of val_task at screening_epoch from existing logs.

    Returns dict with percentiles, median, values, and experiment names.
    """
    all_logs = parse_all_logs()

    # Filter to matching task + paradigm, full (non-debug) runs only
    matching = []
    for exp in all_logs:
        if exp.is_debug:
            continue
        if exp.task != task:
            continue
        exp_paradigm = infer_paradigm(exp.translator_type, exp.name)
        if exp_paradigm != paradigm:
            continue
        # Skip calibration/screening runs themselves
        if exp.name.startswith("cal_") or exp.name.startswith("screen_"):
            continue
        matching.append(exp)

    if not matching:
        return {"n": 0, "values": [], "names": []}

    # Extract val_task at screening_epoch
    values = []
    names = []
    for exp in matching:
        traj = get_val_task_trajectory(exp)
        if screening_epoch in traj:
            values.append(traj[screening_epoch])
            names.append(exp.name)

    if not values:
        return {"n": 0, "values": [], "names": []}

    values_arr = np.array(values)
    return {
        "n": len(values),
        "values": values,
        "names": names,
        "median": float(np.median(values_arr)),
        "mean": float(np.mean(values_arr)),
        "std": float(np.std(values_arr)),
        "min": float(np.min(values_arr)),
        "max": float(np.max(values_arr)),
        "p25": float(np.percentile(values_arr, 25)),
        "p75": float(np.percentile(values_arr, 75)),
    }


def compute_percentile(val_task: float, reference: dict) -> float:
    """Compute percentile of val_task within reference distribution.

    Lower percentile = lower val_task = better (val_task is a loss).
    Returns percentile 0-100.
    """
    if reference["n"] == 0:
        return 50.0  # No reference, assume middle

    values = reference["values"]
    # Count how many reference values are <= val_task
    n_below_or_equal = sum(1 for v in values if v <= val_task)
    return 100.0 * n_below_or_equal / len(values)


def recommend(percentile: float) -> str:
    """Map percentile to recommendation."""
    if percentile < 30:
        return "ACCEPT"
    elif percentile <= 70:
        return "UNCERTAIN"
    else:
        return "REJECT"


def submit_screening(config_path: str, screening_epochs: Optional[int] = None,
                     server: str = "local", force: bool = False) -> Optional[str]:
    """Create screening config and submit to queue. Returns screen name or None."""
    abs_path = Path(config_path)
    if not abs_path.is_absolute():
        abs_path = REPO / config_path

    config = json.loads(abs_path.read_text())
    task = _infer_task_from_config(config, config_path)
    paradigm = _infer_paradigm_from_config(config)

    # Check if task supports screening
    if task in NO_SCREEN_TASKS and not force:
        log.warning(f"Task '{task}' has no reliable screening signal.")
        log.warning("Use --force to run anyway, or run full experiment directly.")
        return None

    # Determine screening epochs
    if screening_epochs is None:
        screening_epochs = DEFAULT_SCREENING_EPOCHS.get(
            (task, paradigm), 5  # default fallback
        )

    # Generate screen name
    config_stem = Path(config_path).stem
    screen_name = f"screen_{config_stem}_ep{screening_epochs}"

    # Check if already in queue
    queue = _load_queue()
    existing_names = {e["name"] for e in queue.get("experiments", [])}
    if screen_name in existing_names:
        log.info(f"Screening {screen_name} already in queue")
        return screen_name

    log.info(f"Screening {task}/{paradigm}: {config_stem} @ {screening_epochs} epochs")

    # Create screening config
    screen_config = create_screening_config(config_path, screening_epochs, screen_name)
    log.info(f"Created screening config: {screen_config}")

    # Auto-copy pretrain checkpoint
    dest = auto_copy(str(screen_config))
    if dest:
        log.info(f"Pretrain checkpoint: {dest}")
    else:
        ttype = config.get("translator", {}).get("type", "")
        if ttype in ("shared_latent", "retrieval"):
            log.warning("No pretrain checkpoint found — Phase 1 will run from scratch")

    # Add to queue
    entry = {
        "name": screen_name,
        "config": str(screen_config.relative_to(REPO)),
        "output": f"experiments/results/{screen_name}.parquet",
        "status": "screening",
        "server": server,
        "notes": f"Screening: {config_stem} @ {screening_epochs}ep",
        "original_config": config_path,
    }

    experiments = queue.get("experiments", [])
    # Insert before non-pending entries
    insert_idx = len(experiments)
    for i, exp in enumerate(experiments):
        if exp.get("status") not in ("pending", "screening", "calibration"):
            insert_idx = i
            break
    experiments.insert(insert_idx, entry)

    _save_queue(queue)
    log.info(f"Added {screen_name} to queue at position {insert_idx}")
    return screen_name


def wait_for_completion(screen_name: str, poll_interval: int = 30,
                        timeout: int = 7200) -> bool:
    """Poll queue until screening experiment completes. Returns True if done."""
    start = time.time()
    while time.time() - start < timeout:
        queue = _load_queue()
        for exp in queue.get("experiments", []):
            if exp.get("name") == screen_name:
                status = exp.get("status", "")
                if status in ("done", "screening_done", "calibration_done"):
                    return True
                if status == "failed":
                    log.error(f"Screening {screen_name} failed: {exp.get('error', 'unknown')}")
                    return False
                break
        else:
            log.error(f"Screening {screen_name} not found in queue")
            return False

        elapsed = int(time.time() - start)
        log.info(f"Waiting for {screen_name}... ({elapsed}s elapsed)")
        time.sleep(poll_interval)

    log.error(f"Timeout waiting for {screen_name} after {timeout}s")
    return False


def collect_screening_result(screen_name: str) -> Optional[dict]:
    """Collect results from a completed screening run."""
    # Find the queue entry
    queue = _load_queue()
    entry = None
    for exp in queue.get("experiments", []):
        if exp.get("name") == screen_name:
            entry = exp
            break

    if entry is None:
        log.error(f"Screen {screen_name} not found in queue")
        return None

    status = entry.get("status", "")
    if status not in ("done", "screening_done", "calibration_done"):
        log.error(f"Screen {screen_name} not completed (status={status})")
        return None

    # Parse the screening log
    task = infer_task(screen_name)
    log_path = REPO / "experiments" / "logs" / f"{screen_name}_{task}.log"
    if not log_path.exists():
        log.error(f"Log not found: {log_path}")
        return None

    exp_log = parse_log(log_path)
    if exp_log is None:
        log.error(f"Failed to parse log: {log_path}")
        return None

    traj = get_val_task_trajectory(exp_log)
    if not traj:
        log.error(f"No val_task trajectory in {log_path}")
        return None

    # Get screening config to determine epochs and paradigm
    screen_config_path = entry.get("config", "")
    if screen_config_path:
        config = json.loads((REPO / screen_config_path).read_text())
        paradigm = _infer_paradigm_from_config(config)
        screening_epochs = config["training"]["epochs"]
    else:
        paradigm = infer_paradigm(exp_log.translator_type, screen_name)
        screening_epochs = max(traj.keys()) if traj else 0

    # Get val_task at final screening epoch
    final_epoch = max(traj.keys())
    val_task_final = traj[final_epoch]

    # Build reference distribution
    reference = build_reference_distribution(task, paradigm, final_epoch)
    percentile = compute_percentile(val_task_final, reference)
    rec = recommend(percentile)

    # Build result
    original_config = entry.get("original_config", screen_config_path)
    result = {
        "screen_name": screen_name,
        "original_config": original_config,
        "task": task,
        "paradigm": paradigm,
        "screening_epochs": screening_epochs,
        "val_task_trajectory": {str(k): v for k, v in sorted(traj.items())},
        "val_task_final": val_task_final,
        "reference_percentile": round(percentile, 1),
        "recommendation": rec,
        "reference_n": reference["n"],
        "reference_median": reference.get("median"),
        "reference_min": reference.get("min"),
        "reference_max": reference.get("max"),
        "queue_results": entry.get("results", {}),
        "timestamp": datetime.now().isoformat(),
    }

    # Save screening result
    SCREEN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = SCREEN_RESULTS_DIR / f"{screen_name}.json"
    result_path.write_text(json.dumps(result, indent=2))
    log.info(f"Saved screening result to {result_path}")

    return result


def print_result(result: dict):
    """Pretty-print a screening result."""
    rec = result["recommendation"]
    rec_emoji = {"ACCEPT": ">>>", "UNCERTAIN": "???", "REJECT": "XXX"}.get(rec, "---")

    print(f"\n{'='*60}")
    print(f"  Screening Result: {result['screen_name']}")
    print(f"{'='*60}")
    print(f"  Config:      {result['original_config']}")
    print(f"  Task:        {result['task']} / {result['paradigm']}")
    print(f"  Epochs:      {result['screening_epochs']}")
    print(f"  val_task:    {result['val_task_final']:.4f}")
    print(f"  Percentile:  {result['reference_percentile']}% "
          f"(n={result['reference_n']} reference experiments)")
    if result.get("reference_median"):
        print(f"  Ref median:  {result['reference_median']:.4f} "
              f"(range: {result['reference_min']:.4f} - {result['reference_max']:.4f})")
    print(f"  Decision:    [{rec_emoji}] {rec}")

    # Val_task trajectory
    traj = result.get("val_task_trajectory", {})
    if traj:
        print(f"\n  Val_task trajectory:")
        for ep, v in sorted(traj.items(), key=lambda x: int(x[0])):
            print(f"    Epoch {ep}: {v:.4f}")

    # Final metrics if available
    queue_results = result.get("queue_results", {})
    if queue_results:
        print(f"\n  Final metrics:")
        for k, v in queue_results.items():
            print(f"    {k}: {v:+.4f}")

    print()


def screen_and_wait(config_path: str, screening_epochs: Optional[int] = None,
                    server: str = "local", force: bool = False,
                    timeout: int = 7200) -> Optional[dict]:
    """Full screening pipeline: submit, wait, collect, evaluate."""
    screen_name = submit_screening(config_path, screening_epochs, server, force)
    if screen_name is None:
        return None

    print(f"\nSubmitted {screen_name} to queue.")
    print("Waiting for scheduler to run it...")
    print("(Make sure gpu_scheduler.py is running in another terminal)")

    if not wait_for_completion(screen_name, timeout=timeout):
        return None

    result = collect_screening_result(screen_name)
    if result is None:
        return None

    print_result(result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Screen experiment viability with reduced epochs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str,
                        help="Config file to screen")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override screening epochs (default: task-specific)")
    parser.add_argument("--server", type=str, default="local",
                        help="Server to run on (default: local)")
    parser.add_argument("--force", action="store_true",
                        help="Force screening even for tasks without reliable signal")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Max wait time in seconds (default: 7200 = 2h)")
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit to queue without waiting")
    parser.add_argument("--collect", type=str, metavar="SCREEN_NAME",
                        help="Collect results for a completed screening run")
    parser.add_argument("--no-wait", action="store_true",
                        help="Alias for --submit-only")

    args = parser.parse_args()

    if args.collect:
        result = collect_screening_result(args.collect)
        if result:
            print_result(result)
        else:
            sys.exit(1)
        return

    if not args.config:
        parser.print_help()
        sys.exit(1)

    if args.submit_only or args.no_wait:
        screen_name = submit_screening(
            args.config, args.epochs, args.server, args.force
        )
        if screen_name:
            print(f"\nSubmitted {screen_name} to queue.")
            print("Collect results later with:")
            print(f"  python scripts/screen_experiment.py --collect {screen_name}")
        else:
            sys.exit(1)
    else:
        result = screen_and_wait(
            args.config, args.epochs, args.server, args.force, args.timeout
        )
        if result is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
