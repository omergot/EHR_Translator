#!/usr/bin/env python3
"""GPU Experiment Scheduler — reads experiments/queue.yaml, manages GPU assignment.

Usage:
    python scripts/gpu_scheduler.py              # Start scheduler daemon
    python scripts/gpu_scheduler.py --status     # Show queue status
    python scripts/gpu_scheduler.py --dry-run    # Show what would launch
    python scripts/gpu_scheduler.py --add --name NAME --config PATH [--notes TEXT]

Designed to run in a tmux session. Graceful shutdown with Ctrl+C.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml")
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
QUEUE_PATH = REPO / "experiments" / "queue.yaml"
LOG_DIR = REPO / "experiments" / "logs"
RESULTS_DIR = REPO / "experiments" / "results"
SCHEDULER_LOG = LOG_DIR / "scheduler.log"

# Track running subprocesses for cleanup
_running_procs: dict[str, subprocess.Popen] = {}
_shutdown = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [scheduler] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(SCHEDULER_LOG, mode="a"),
        ],
    )


# ---------------------------------------------------------------------------
# Queue I/O
# ---------------------------------------------------------------------------

def load_queue() -> dict:
    if not QUEUE_PATH.exists():
        logging.error(f"Queue file not found: {QUEUE_PATH}")
        sys.exit(1)
    with open(QUEUE_PATH) as f:
        return yaml.safe_load(f)


def save_queue(queue: dict):
    """Atomic write: tmp file + rename to prevent corruption."""
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(QUEUE_PATH) + ".tmp"
    with open(tmp, "w") as f:
        # Write header comment manually, then dump YAML
        f.write("# ============================================================\n")
        f.write("# Experiment Queue — edit this file to manage experiments\n")
        f.write("# Reorder pending experiments by moving entries up/down\n")
        f.write("# The scheduler runs top-to-bottom through pending experiments\n")
        f.write("# ============================================================\n\n")
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=120)
    os.rename(tmp, str(QUEUE_PATH))


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def get_free_gpus(threshold_mb: int) -> list[int]:
    """Return GPU indices with memory usage below threshold."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            timeout=10,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logging.warning(f"nvidia-smi failed: {e}")
        return []

    free = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue
        idx, mem = int(parts[0].strip()), int(parts[1].strip())
        if mem < threshold_mb:
            free.append(idx)
    return free


def get_max_gpus(settings: dict) -> int:
    """Return max GPUs allowed based on time of day and day of week."""
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    is_weekend_day = weekday in (4, 5)  # Fri, Sat
    if 9 <= hour < 21 and not is_weekend_day:
        return settings.get("day_max_gpus", 2)
    return settings.get("night_max_gpus", 3)


def select_gpus(settings: dict, free_gpus: list[int], running_gpus: set[int]) -> list[int]:
    """Select available GPUs respecting priority and time limits."""
    max_gpus = get_max_gpus(settings)
    priority = settings.get("gpu_priority", [0, 1, 2, 3])

    # GPUs that are free AND not running our experiments, in priority order
    available = [g for g in priority if g in free_gpus and g not in running_gpus]

    # How many more can we launch?
    slots = max(0, max_gpus - len(running_gpus))
    return available[:slots]


# ---------------------------------------------------------------------------
# Task inference
# ---------------------------------------------------------------------------

def infer_task(config_path: str) -> str:
    """Infer task name from config filename."""
    name = Path(config_path).stem.lower()
    for task in ["mortality", "aki", "sepsis"]:
        if task in name:
            return task
    return "unknown"


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def pid_is_alive(pid: int) -> bool:
    """Check if a process with given PID is still running (not zombie)."""
    try:
        status_path = f"/proc/{pid}/status"
        if os.path.exists(status_path):
            with open(status_path) as f:
                for line in f:
                    if line.startswith("State:"):
                        # Z = zombie, X = dead
                        state = line.split()[1]
                        return state not in ("Z", "X")
        # Fallback: signal check
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def recover_stale(experiments: list[dict]):
    """On startup, mark 'running' experiments with dead PIDs as failed."""
    for exp in experiments:
        if exp.get("status") == "running":
            pid = exp.get("pid")
            if pid and not pid_is_alive(pid):
                logging.warning(
                    f"Stale experiment '{exp['name']}' (PID {pid} dead), marking failed"
                )
                exp["status"] = "failed"
                exp["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                exp["error"] = "Process died (scheduler restart recovery)"


# ---------------------------------------------------------------------------
# Experiment launch & monitoring
# ---------------------------------------------------------------------------

def launch_experiment(exp: dict, gpu: int) -> subprocess.Popen:
    """Launch an experiment on the specified GPU."""
    task = infer_task(exp["config"])
    # Log file: {name}_{task}.log for collect_result.py compatibility
    # Always include task suffix so collect_result.py can find the log
    log_name = f"{exp['name']}_{task}.log"
    log_path = LOG_DIR / log_name

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EHR_LOG_FILE"] = str(log_path)

    config_path = str(REPO / exp["config"])
    output_path = str(REPO / exp["output"])

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(REPO / "run.py"), "train_and_eval",
        "--config", config_path,
        "--output_parquet", output_path,
    ]

    logging.info(f"Launching '{exp['name']}' on GPU {gpu}: {' '.join(cmd)}")
    logging.info(f"  Log: {log_path}")

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
        cwd=str(REPO),
    )

    exp["status"] = "running"
    exp["gpu"] = gpu
    exp["pid"] = proc.pid
    exp["started"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _running_procs[exp["name"]] = proc
    return proc


def collect_results(exp: dict):
    """Collect results using collect_result.py after experiment finishes."""
    task = infer_task(exp["config"])
    if task == "unknown":
        logging.warning(f"Cannot collect results for '{exp['name']}': unknown task")
        return

    try:
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "collect_result.py"),
             exp["name"], task],
            cwd=str(REPO),
            timeout=30,
            capture_output=True,
        )
    except Exception as e:
        logging.warning(f"collect_result.py failed for '{exp['name']}': {e}")
        return

    result_path = RESULTS_DIR / f"{exp['name']}_{task}.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            diff = data.get("difference", {})
            if diff:
                exp["results"] = diff
                logging.info(f"  Results for '{exp['name']}': {diff}")
        except json.JSONDecodeError:
            logging.warning(f"Malformed results JSON for '{exp['name']}'")


def check_running(experiments: list[dict]):
    """Check running experiments and update status on completion."""
    for exp in experiments:
        if exp.get("status") != "running":
            continue

        name = exp["name"]
        proc = _running_procs.get(name)

        if proc is not None:
            retcode = proc.poll()
            if retcode is None:
                continue  # Still running
            finished_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            exp["finished"] = finished_ts

            if retcode == 0:
                logging.info(f"Experiment '{name}' completed successfully")
                exp["status"] = "done"
                collect_results(exp)
            else:
                logging.error(f"Experiment '{name}' failed (exit code {retcode})")
                exp["status"] = "failed"
                exp["error"] = f"Exit code {retcode}"

            _running_procs.pop(name, None)
        else:
            # No proc tracked — check PID directly (e.g. after scheduler restart)
            pid = exp.get("pid")
            if pid and not pid_is_alive(pid):
                logging.warning(f"Experiment '{name}' PID {pid} no longer running")
                exp["status"] = "failed"
                exp["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                exp["error"] = "Process died (detected during monitoring)"
                _running_procs.pop(name, None)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def scheduler_loop(dry_run: bool = False):
    """Main scheduler loop — poll GPUs, launch pending experiments."""
    global _shutdown

    queue = load_queue()
    settings = queue.get("settings", {})
    experiments = queue.get("experiments", [])

    # Recover stale entries on startup
    recover_stale(experiments)
    if not dry_run:
        save_queue(queue)

    logging.info(
        f"Scheduler started — {sum(1 for e in experiments if e.get('status') == 'pending')} pending, "
        f"{sum(1 for e in experiments if e.get('status') == 'running')} running, "
        f"{sum(1 for e in experiments if e.get('status') == 'done')} done"
    )

    poll_interval = settings.get("poll_interval", 60)
    threshold_mb = settings.get("gpu_free_threshold_mb", 1000)

    while not _shutdown:
        # Reload queue (may have been edited externally)
        try:
            queue = load_queue()
            settings = queue.get("settings", {})
            experiments = queue.get("experiments", [])
        except Exception as e:
            logging.error(f"Failed to reload queue: {e}")
            time.sleep(poll_interval)
            continue

        # Check running experiments
        check_running(experiments)

        # Find running GPUs
        running_gpus = set()
        for exp in experiments:
            if exp.get("status") == "running" and "gpu" in exp:
                running_gpus.add(exp["gpu"])

        # Get free GPUs
        free_gpus = get_free_gpus(threshold_mb)
        available = select_gpus(settings, free_gpus, running_gpus)

        # Find pending experiments
        pending = [e for e in experiments if e.get("status") == "pending"]

        if dry_run:
            _print_dry_run(settings, free_gpus, running_gpus, available, pending, experiments)
            return

        # Launch as many as we have GPU slots
        launched = 0
        for gpu in available:
            if not pending:
                break
            exp = pending.pop(0)
            config_path = REPO / exp["config"]
            if not config_path.exists():
                logging.error(
                    f"Config not found for '{exp['name']}': {config_path}. Skipping."
                )
                exp["status"] = "failed"
                exp["error"] = f"Config not found: {exp['config']}"
                continue
            launch_experiment(exp, gpu)
            launched += 1

        # Save updated queue
        save_queue(queue)

        if launched:
            logging.info(f"Launched {launched} experiment(s) this cycle")

        # Check if all done
        remaining = sum(1 for e in experiments
                        if e.get("status") in ("pending", "running"))
        if remaining == 0:
            logging.info("All experiments completed. Scheduler exiting.")
            return

        time.sleep(poll_interval)

    logging.info("Scheduler shut down by signal")


def _print_dry_run(settings, free_gpus, running_gpus, available, pending, experiments):
    """Print what the scheduler would do without launching anything."""
    hour = datetime.now().hour
    period = "daytime" if 9 <= hour < 21 else "nighttime"
    max_gpus = get_max_gpus(settings)

    print(f"\n{'='*60}")
    print(f"  GPU Scheduler — Dry Run  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*60}")
    print(f"  Period:        {period} (max {max_gpus} GPUs)")
    print(f"  Free GPUs:     {free_gpus}")
    print(f"  Running GPUs:  {running_gpus or '{none}'}")
    print(f"  Available:     {available}")
    print(f"  Pending:       {len(pending)} experiment(s)")
    print()

    if available and pending:
        print("  Would launch:")
        for gpu, exp in zip(available, pending):
            print(f"    GPU {gpu} <- {exp['name']}  ({exp['config']})")
    elif not available:
        print("  No GPU slots available.")
    elif not pending:
        print("  No pending experiments.")

    # Summary table
    print(f"\n  {'Status':<10} {'Count'}")
    print(f"  {'-'*20}")
    for status in ["pending", "running", "done", "failed"]:
        count = sum(1 for e in experiments if e.get("status") == status)
        if count:
            print(f"  {status:<10} {count}")
    print()


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status():
    """Print a human-readable status table."""
    queue = load_queue()
    experiments = queue.get("experiments", [])
    settings = queue.get("settings", {})

    hour = datetime.now().hour
    period = "daytime" if 9 <= hour < 21 else "nighttime"
    max_gpus = get_max_gpus(settings)

    threshold_mb = settings.get("gpu_free_threshold_mb", 1000)
    free_gpus = get_free_gpus(threshold_mb)

    print(f"\n{'='*70}")
    print(f"  Experiment Queue Status  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"  Period: {period} (max {max_gpus} GPUs)  |  Free GPUs: {free_gpus}")
    print(f"{'='*70}")

    if not experiments:
        print("  No experiments in queue.")
        print()
        return

    # Group by status
    for status in ["running", "pending", "done", "failed"]:
        group = [e for e in experiments if e.get("status") == status]
        if not group:
            continue

        status_label = {
            "running": "RUNNING",
            "pending": "PENDING",
            "done":    "DONE",
            "failed":  "FAILED",
        }[status]
        print(f"\n  [{status_label}]")

        for exp in group:
            name = exp["name"]
            config = exp.get("config", "")

            if status == "running":
                gpu = exp.get("gpu", "?")
                pid = exp.get("pid", "?")
                started = exp.get("started", "?")
                alive = "alive" if exp.get("pid") and pid_is_alive(exp["pid"]) else "DEAD"
                print(f"    {name:<35} GPU {gpu}  PID {pid} ({alive})  started {started}")

            elif status == "pending":
                notes = exp.get("notes", "")
                print(f"    {name:<35} {notes}")

            elif status == "done":
                finished = exp.get("finished", "?")
                results = exp.get("results", {})
                result_str = "  ".join(f"{k}: {v:+.4f}" for k, v in results.items()) if results else ""
                print(f"    {name:<35} finished {finished}  {result_str}")

            elif status == "failed":
                error = exp.get("error", "unknown")
                print(f"    {name:<35} error: {error}")

    print()


# ---------------------------------------------------------------------------
# Add experiment
# ---------------------------------------------------------------------------

def add_experiment(name: str, config: str, notes: str = ""):
    """Add a new pending experiment to the queue."""
    queue = load_queue()
    experiments = queue.get("experiments", [])

    # Check for duplicate name
    existing_names = {e["name"] for e in experiments}
    if name in existing_names:
        logging.error(f"Experiment '{name}' already exists in queue")
        sys.exit(1)

    # Verify config exists
    config_abs = REPO / config
    if not config_abs.exists():
        logging.warning(f"Config file not found: {config_abs}")

    task = infer_task(config)
    output = f"experiments/results/{name}.parquet"

    entry = {
        "name": name,
        "config": config,
        "output": output,
        "status": "pending",
    }
    if notes:
        entry["notes"] = notes

    # Insert before first non-pending entry (or at end)
    insert_idx = len(experiments)
    for i, exp in enumerate(experiments):
        if exp.get("status") not in ("pending",):
            insert_idx = i
            break
    experiments.insert(insert_idx, entry)

    save_queue(queue)
    logging.info(f"Added experiment '{name}' ({task}) at position {insert_idx}")
    print(f"Added '{name}' to queue (position {insert_idx}, task={task})")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(signum, frame):
    global _shutdown
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    _shutdown = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU Experiment Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--status", action="store_true",
                        help="Show queue status and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would launch without running")
    parser.add_argument("--add", action="store_true",
                        help="Add experiment to queue")
    parser.add_argument("--name", type=str, help="Experiment name (for --add)")
    parser.add_argument("--config", type=str, help="Config path (for --add)")
    parser.add_argument("--notes", type=str, default="", help="Notes (for --add)")

    args = parser.parse_args()

    setup_logging()

    if args.status:
        show_status()
        return

    if args.add:
        if not args.name or not args.config:
            parser.error("--add requires --name and --config")
        add_experiment(args.name, args.config, args.notes)
        return

    if args.dry_run:
        scheduler_loop(dry_run=True)
        return

    # Daemon mode
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logging.info("Starting GPU scheduler daemon (Ctrl+C to stop)")
    try:
        scheduler_loop()
    except KeyboardInterrupt:
        logging.info("Interrupted, shutting down...")
    finally:
        # Wait briefly for any running processes (don't kill them)
        if _running_procs:
            logging.info(
                f"{len(_running_procs)} experiment(s) still running — "
                "they will continue in background"
            )


if __name__ == "__main__":
    main()
