#!/usr/bin/env python3
"""Autonomous experiment iteration — propose, screen, accept/reject, repeat.

Inspired by Karpathy's autoresearch: mutate hyperparameters from the current
best config, screen with reduced epochs, accept/reject, and iterate.
All GPU work goes through experiments/queue.yaml + gpu_scheduler.py.

Usage:
    python scripts/autoresearch.py --task aki --paradigm retrieval --budget 12h
    python scripts/autoresearch.py --task sepsis --paradigm retrieval --budget 6h --server 3090
    python scripts/autoresearch.py --status
    python scripts/autoresearch.py --history --task aki

Architecture:
    ConfigGenerator  — proposes configs by mutating one hyperparameter
    ScreenRunner     — submits screening jobs via screen_experiment.py
    ResultTracker    — maintains JSON log of all proposals + outcomes
    DecisionEngine   — accept/reject/uncertain based on screening percentile
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_convergence import infer_task, infer_paradigm
from manage_pretrain import auto_copy
from screen_experiment import (
    submit_screening, wait_for_completion, collect_screening_result,
    print_result, DEFAULT_SCREENING_EPOCHS, SCREEN_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
QUEUE_PATH = REPO / "experiments" / "queue.yaml"
AUTORESEARCH_DIR = REPO / "experiments" / "autoresearch"

# Hyperparameter search spaces per paradigm
SEARCH_SPACES = {
    "retrieval": {
        "n_cross_layers": [1, 2, 3, 4],
        "lambda_align": [0.0, 0.1, 0.25, 0.5, 1.0],
        "k_neighbors": [8, 12, 16, 24, 32],
        "retrieval_window": [4, 6, 8, 12],
        "window_stride": [None, 1, 3, 6],
        "lambda_label_pred": [0.0, 0.1, 0.25, 0.5],
        "lambda_recon": [0.05, 0.1, 0.2],
        "lr": [5e-5, 1e-4, 2e-4],
        "lambda_smooth": [0.0, 0.05, 0.1, 0.2],
        "lambda_importance_reg": [0.0, 0.01, 0.05],
        "memory_refresh_epochs": [1, 3, 5, 10],
    },
    "sl": {
        "lambda_align": [0.0, 0.1, 0.25, 0.5, 1.0],
        "lambda_recon": [0.05, 0.1, 0.2, 0.5],
        "lambda_label_pred": [0.0, 0.1, 0.25, 0.5],
        "lr": [5e-5, 1e-4, 2e-4],
        "lambda_target_task": [0.0, 0.25, 0.5, 1.0],
    },
    "delta": {
        "lambda_fidelity": [0.005, 0.01, 0.02, 0.05],
        "lambda_range": [0.0005, 0.001, 0.002],
        "lr": [5e-5, 1e-4, 2e-4],
        "lambda_target_task": [0.0, 0.25, 0.5],
    },
}

# Keys that live in training vs translator config
TRAINING_KEYS = {
    "n_cross_layers", "lambda_align", "k_neighbors", "retrieval_window",
    "window_stride", "lambda_label_pred", "lambda_recon", "lr",
    "lambda_smooth", "lambda_importance_reg", "memory_refresh_epochs",
    "lambda_fidelity", "lambda_range", "lambda_target_task",
    "lambda_mmd", "feature_gate", "output_mode",
}


class ConfigGenerator:
    """Generate new configs by mutating one hyperparameter from a seed config."""

    def __init__(self, seed_config_path: str, paradigm: str, rng: random.Random):
        self.seed_path = Path(seed_config_path)
        if not self.seed_path.is_absolute():
            self.seed_path = REPO / seed_config_path
        self.seed_config = json.loads(self.seed_path.read_text())
        self.paradigm = paradigm
        self.rng = rng
        self.search_space = SEARCH_SPACES.get(paradigm, {})
        self._tried = set()  # (param, value) pairs already tried

    def _get_current_value(self, param: str):
        """Get current value of a parameter from seed config."""
        training = self.seed_config.get("training", {})
        translator = self.seed_config.get("translator", {})

        if param in training:
            return training[param]
        if param in translator:
            return translator[param]
        return None

    def propose(self) -> Optional[tuple[dict, str, str]]:
        """Propose a new config by mutating one parameter.

        Returns (config_dict, param_name, param_value_str) or None if exhausted.
        """
        # Build candidate mutations: (param, value) where value != current
        candidates = []
        for param, values in self.search_space.items():
            current = self._get_current_value(param)
            for v in values:
                if v == current:
                    continue
                key = (param, str(v))
                if key not in self._tried:
                    candidates.append((param, v))

        if not candidates:
            return None

        # Random selection (could be smarter with surrogate model later)
        param, value = self.rng.choice(candidates)
        self._tried.add((param, str(value)))

        # Create mutated config
        config = copy.deepcopy(self.seed_config)
        if param in TRAINING_KEYS:
            config["training"][param] = value
        else:
            config["translator"][param] = value

        value_str = str(value).replace(".", "p") if value is not None else "none"
        mutation_name = f"{param}_{value_str}"

        return config, param, mutation_name

    def mark_tried(self, param: str, value):
        """Mark a (param, value) as already tried."""
        self._tried.add((param, str(value)))

    @property
    def remaining(self) -> int:
        """Number of untried mutations."""
        count = 0
        for param, values in self.search_space.items():
            current = self._get_current_value(param)
            for v in values:
                if v != current and (param, str(v)) not in self._tried:
                    count += 1
        return count


class ResultTracker:
    """Track autoresearch iterations in a JSON log."""

    def __init__(self, task: str, paradigm: str):
        self.task = task
        self.paradigm = paradigm
        AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = AUTORESEARCH_DIR / f"history_{task}_{paradigm}.json"
        self.history = self._load()

    def _load(self) -> list:
        if self.log_path.exists():
            try:
                return json.loads(self.log_path.read_text())
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self):
        self.log_path.write_text(json.dumps(self.history, indent=2))

    def add(self, entry: dict):
        self.history.append(entry)
        self._save()

    def get_tried_mutations(self) -> set:
        """Return set of (param, value_str) that have been tried."""
        tried = set()
        for h in self.history:
            param = h.get("param")
            value = h.get("mutation_name", "")
            if param:
                tried.add((param, str(h.get("param_value", ""))))
        return tried

    def summary(self) -> str:
        if not self.history:
            return "No iterations yet."
        n = len(self.history)
        accepted = sum(1 for h in self.history if h.get("recommendation") == "ACCEPT")
        rejected = sum(1 for h in self.history if h.get("recommendation") == "REJECT")
        uncertain = sum(1 for h in self.history if h.get("recommendation") == "UNCERTAIN")
        return (f"{n} iterations: {accepted} ACCEPT, {uncertain} UNCERTAIN, "
                f"{rejected} REJECT")


def _find_best_config(task: str, paradigm: str) -> Optional[str]:
    """Find the config of the best-performing experiment for task+paradigm."""
    if not QUEUE_PATH.exists():
        return None

    queue = yaml.safe_load(QUEUE_PATH.read_text())
    best_config = None
    best_metric = -999

    for exp in queue.get("experiments", []):
        if exp.get("status") not in ("done",):
            continue
        results = exp.get("results", {})
        if not results:
            continue

        name = exp["name"]
        if name.startswith("screen_") or name.startswith("cal_"):
            continue

        exp_task = infer_task(name)
        if exp_task != task:
            continue

        config_path = exp.get("config", "")
        if not config_path or not (REPO / config_path).exists():
            continue

        try:
            config = json.loads((REPO / config_path).read_text())
        except (json.JSONDecodeError, OSError):
            continue

        ttype = config.get("translator", {}).get("type", "")
        exp_paradigm = infer_paradigm(ttype, name)
        if exp_paradigm != paradigm:
            continue

        metric = results.get("AUCROC", results.get("MAE", None))
        if metric is None:
            continue
        # For MAE, negate so higher = better
        if task in ("los", "kf"):
            metric = -metric

        if metric > best_metric:
            best_metric = metric
            best_config = config_path

    return best_config


def parse_budget(budget_str: str) -> float:
    """Parse budget string like '12h', '6h', '30m' into seconds."""
    budget_str = budget_str.strip().lower()
    if budget_str.endswith("h"):
        return float(budget_str[:-1]) * 3600
    if budget_str.endswith("m"):
        return float(budget_str[:-1]) * 60
    return float(budget_str)


def run_autoresearch(task: str, paradigm: str, budget_seconds: float,
                     server: str = "local", seed_config: str = None,
                     screening_epochs: int = None, random_seed: int = 42):
    """Main autoresearch loop."""
    # Find seed config
    if seed_config is None:
        seed_config = _find_best_config(task, paradigm)
        if seed_config is None:
            log.error(f"No completed experiments found for {task}/{paradigm}")
            sys.exit(1)
    log.info(f"Seed config: {seed_config}")

    # Determine screening epochs
    if screening_epochs is None:
        screening_epochs = DEFAULT_SCREENING_EPOCHS.get((task, paradigm), 5)

    rng = random.Random(random_seed)
    generator = ConfigGenerator(seed_config, paradigm, rng)
    tracker = ResultTracker(task, paradigm)

    # Pre-load tried mutations from history
    for param, value_str in tracker.get_tried_mutations():
        generator.mark_tried(param, value_str)

    start_time = time.time()
    deadline = start_time + budget_seconds
    iteration = len(tracker.history)

    print(f"\n{'='*60}")
    print(f"  AutoResearch: {task}/{paradigm}")
    print(f"  Seed: {seed_config}")
    print(f"  Budget: {budget_seconds/3600:.1f}h")
    print(f"  Screening: {screening_epochs} epochs")
    print(f"  Server: {server}")
    print(f"  Remaining mutations: {generator.remaining}")
    print(f"  History: {tracker.summary()}")
    print(f"{'='*60}\n")

    while time.time() < deadline and generator.remaining > 0:
        iteration += 1
        elapsed = (time.time() - start_time) / 3600
        remaining_h = (deadline - time.time()) / 3600

        log.info(f"--- Iteration {iteration} ({elapsed:.1f}h elapsed, "
                 f"{remaining_h:.1f}h remaining, {generator.remaining} mutations left) ---")

        # Propose mutation
        proposal = generator.propose()
        if proposal is None:
            log.info("All mutations exhausted")
            break

        config, param, mutation_name = proposal

        # Create config file
        config_name = f"auto_{task}_{paradigm}_{mutation_name}"
        config["output"]["run_dir"] = f"runs/{config_name}"
        config["output"]["log_file"] = f"runs/{config_name}/run.log"

        AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        config_path = AUTORESEARCH_DIR / f"{config_name}.json"
        config_path.write_text(json.dumps(config, indent=2))

        log.info(f"Mutation: {param} -> {config['training'].get(param, config['translator'].get(param))}")

        # Submit screening
        screen_name = submit_screening(
            str(config_path.relative_to(REPO)),
            screening_epochs, server, force=True
        )
        if screen_name is None:
            log.warning(f"Failed to submit screening for {config_name}")
            tracker.add({
                "iteration": iteration,
                "config_name": config_name,
                "param": param,
                "mutation_name": mutation_name,
                "param_value": config["training"].get(param, config["translator"].get(param)),
                "status": "submit_failed",
                "timestamp": datetime.now().isoformat(),
            })
            continue

        # Wait for completion (with remaining budget as timeout)
        remaining_seconds = max(deadline - time.time(), 300)
        success = wait_for_completion(
            screen_name, poll_interval=30,
            timeout=int(remaining_seconds)
        )

        if not success:
            log.warning(f"Screening {screen_name} did not complete in time")
            tracker.add({
                "iteration": iteration,
                "config_name": config_name,
                "param": param,
                "mutation_name": mutation_name,
                "param_value": config["training"].get(param, config["translator"].get(param)),
                "screen_name": screen_name,
                "status": "timeout",
                "timestamp": datetime.now().isoformat(),
            })
            continue

        # Collect result
        result = collect_screening_result(screen_name)
        if result is None:
            log.warning(f"Failed to collect results for {screen_name}")
            tracker.add({
                "iteration": iteration,
                "config_name": config_name,
                "param": param,
                "mutation_name": mutation_name,
                "param_value": config["training"].get(param, config["translator"].get(param)),
                "screen_name": screen_name,
                "status": "collect_failed",
                "timestamp": datetime.now().isoformat(),
            })
            continue

        print_result(result)

        # Record result
        entry = {
            "iteration": iteration,
            "config_name": config_name,
            "config_path": str(config_path.relative_to(REPO)),
            "param": param,
            "mutation_name": mutation_name,
            "param_value": config["training"].get(param, config["translator"].get(param)),
            "screen_name": screen_name,
            "val_task_final": result.get("val_task_final"),
            "percentile": result.get("reference_percentile"),
            "recommendation": result.get("recommendation"),
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
        tracker.add(entry)

        rec = result.get("recommendation", "")
        if rec == "ACCEPT":
            log.info(f"ACCEPT: {config_name} — consider queueing full run")
            # Could auto-promote here in future
        elif rec == "REJECT":
            log.info(f"REJECT: {config_name}")
        else:
            log.info(f"UNCERTAIN: {config_name} — needs full run to confirm")

    # Final summary
    elapsed_h = (time.time() - start_time) / 3600
    print(f"\n{'='*60}")
    print(f"  AutoResearch Complete")
    print(f"  Duration: {elapsed_h:.1f}h")
    print(f"  {tracker.summary()}")
    print(f"  Mutations remaining: {generator.remaining}")
    print(f"{'='*60}\n")

    # Print history
    print_history(task, paradigm)


def print_history(task: str, paradigm: str):
    """Print iteration history for a task+paradigm."""
    tracker = ResultTracker(task, paradigm)
    if not tracker.history:
        print(f"No autoresearch history for {task}/{paradigm}")
        return

    print(f"\n  {'#':<4} {'Mutation':<35} {'val_task':>10} {'Pctl':>6} {'Rec':>12} {'Status':>12}")
    print(f"  {'-'*82}")

    for h in tracker.history:
        it = h.get("iteration", "?")
        mutation = h.get("mutation_name", "?")
        param = h.get("param", "")
        vt = h.get("val_task_final")
        vt_str = f"{vt:.4f}" if vt is not None else "-"
        pctl = h.get("percentile")
        pctl_str = f"p{pctl:.0f}" if pctl is not None else "-"
        rec = h.get("recommendation", "-")
        status = h.get("status", "?")

        print(f"  {it:<4} {param}={mutation:<25} {vt_str:>10} {pctl_str:>6} {rec:>12} {status:>12}")

    print(f"\n  {tracker.summary()}")
    print()


def show_status():
    """Show status of all autoresearch sessions."""
    AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    found = False
    for f in sorted(AUTORESEARCH_DIR.glob("history_*.json")):
        parts = f.stem.split("_", 1)[1]  # Remove "history_" prefix
        # Parse task_paradigm from filename
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not data:
            continue
        found = True
        n = len(data)
        accepted = sum(1 for h in data if h.get("recommendation") == "ACCEPT")
        rejected = sum(1 for h in data if h.get("recommendation") == "REJECT")
        uncertain = sum(1 for h in data if h.get("recommendation") == "UNCERTAIN")
        last_ts = data[-1].get("timestamp", "?")
        print(f"  {parts}: {n} iterations ({accepted}A/{uncertain}U/{rejected}R) "
              f"last: {last_ts}")

    if not found:
        print("  No autoresearch sessions found.")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous experiment iteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", type=str,
                        choices=["aki", "sepsis", "mortality", "los", "kf"],
                        help="Task to optimize")
    parser.add_argument("--paradigm", type=str,
                        choices=["retrieval", "sl", "delta"],
                        help="Paradigm to optimize")
    parser.add_argument("--budget", type=str, default="6h",
                        help="Time budget (e.g., '12h', '6h', '30m')")
    parser.add_argument("--server", type=str, default="local",
                        help="Server to run screening on")
    parser.add_argument("--seed-config", type=str,
                        help="Override seed config (default: best for task+paradigm)")
    parser.add_argument("--epochs", type=int,
                        help="Override screening epochs")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for mutation selection")
    parser.add_argument("--status", action="store_true",
                        help="Show status of all autoresearch sessions")
    parser.add_argument("--history", action="store_true",
                        help="Show iteration history for task+paradigm")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.history:
        if not args.task or not args.paradigm:
            parser.error("--history requires --task and --paradigm")
        print_history(args.task, args.paradigm)
        return

    if not args.task or not args.paradigm:
        parser.print_help()
        sys.exit(1)

    budget_seconds = parse_budget(args.budget)
    run_autoresearch(
        args.task, args.paradigm, budget_seconds,
        server=args.server,
        seed_config=args.seed_config,
        screening_epochs=args.epochs,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
