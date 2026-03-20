#!/usr/bin/env python3
"""Pretrain checkpoint manager — index, match, and copy Phase 1 checkpoints.

Phase 1 (autoencoder pretrain on MIMIC) is reusable across experiments when:
  - Same task (same target data)
  - Same architecture (d_latent, d_model, n_enc_layers, n_dec_layers)
  - Same pretrain_epochs
  - Same seed

Usage:
    python scripts/manage_pretrain.py --list
    python scripts/manage_pretrain.py --find-match configs/new_experiment.json
    python scripts/manage_pretrain.py --auto-copy configs/new_experiment.json
"""

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "runs"
QUEUE_PATH = REPO / "experiments" / "queue.yaml"

TASKS = ("mortality", "aki", "sepsis", "los", "kf", "kidney_function")


@dataclass(frozen=True)
class PretrainFingerprint:
    """Uniquely identifies a Phase 1 pretrain checkpoint.

    Note: n_cross_layers is included because the pretrain checkpoint saves the
    full translator state_dict (including cross-attention blocks), so loading
    a checkpoint from n_cross_layers=2 into a model with n_cross_layers=3 fails.
    """
    task: str
    d_latent: int
    d_model: int
    n_enc_layers: int
    n_dec_layers: int
    n_cross_layers: int
    pretrain_epochs: int
    seed: int
    phase1_self_retrieval: bool = False

    def __str__(self):
        sr_str = " SR" if self.phase1_self_retrieval else ""
        return (f"task={self.task} d_latent={self.d_latent} d_model={self.d_model} "
                f"enc={self.n_enc_layers} dec={self.n_dec_layers} "
                f"cross={self.n_cross_layers} "
                f"pretrain_ep={self.pretrain_epochs} seed={self.seed}{sr_str}")


def infer_task(name: str) -> str:
    """Infer task from experiment/config name."""
    lower = name.lower()
    for task in TASKS:
        if f"_{task}" in lower or lower.startswith(task):
            return task
    if "mortality" in lower:
        return "mortality"
    if "aki" in lower:
        return "aki"
    if "sepsis" in lower:
        return "sepsis"
    if "los" in lower:
        return "los"
    if "kf" in lower or "kidney" in lower:
        return "kf"
    return "unknown"


def fingerprint_from_config(config: dict, config_name: str = "") -> Optional[PretrainFingerprint]:
    """Extract pretrain fingerprint from a config dict."""
    translator = config.get("translator", {})
    training = config.get("training", {})
    ttype = translator.get("type", "")

    # Only SL and retrieval have pretrain phases
    if ttype not in ("shared_latent", "retrieval"):
        return None

    pretrain_epochs = training.get("pretrain_epochs", 10)
    if pretrain_epochs <= 0:
        return None

    # Infer task from config path or data_dir
    task = infer_task(config_name)
    if task == "unknown":
        data_dir = config.get("data_dir", "")
        task = infer_task(data_dir)

    # n_cross_layers can be in training or translator section
    n_cross_layers = training.get("n_cross_layers",
                                   translator.get("n_cross_layers", 2))

    phase1_self_retrieval = training.get("phase1_self_retrieval", False)

    return PretrainFingerprint(
        task=task,
        d_latent=translator.get("d_latent", 128),
        d_model=translator.get("d_model", 128),
        n_enc_layers=translator.get("n_enc_layers", 4),
        n_dec_layers=translator.get("n_dec_layers", 2),
        n_cross_layers=n_cross_layers,
        pretrain_epochs=pretrain_epochs,
        seed=config.get("seed", 2222),
        phase1_self_retrieval=phase1_self_retrieval,
    )


def _build_run_to_config_map() -> dict[str, str]:
    """Map run directory names to config file paths using the queue."""
    mapping = {}
    if QUEUE_PATH.exists():
        queue = yaml.safe_load(QUEUE_PATH.read_text())
        for exp in queue.get("experiments", []):
            config_path = exp.get("config", "")
            # Derive run_dir from output path or name
            output = exp.get("output", "")
            if output:
                # e.g., "runs/aki_v5_cross3/eval.parquet" -> "aki_v5_cross3"
                parts = Path(output).parts
                if len(parts) >= 2 and parts[0] == "runs":
                    mapping[parts[1]] = config_path
            # Also map by experiment name directly
            name = exp.get("name", "")
            if name and name not in mapping:
                mapping[name] = config_path
    return mapping


def index_checkpoints() -> list[tuple[Path, PretrainFingerprint]]:
    """Scan all runs for pretrain_checkpoint.pt and compute fingerprints."""
    run_config_map = _build_run_to_config_map()
    results = []

    for ckpt_path in sorted(RUNS_DIR.glob("*/pretrain_checkpoint.pt")):
        run_name = ckpt_path.parent.name

        # Find the config for this run
        config_rel = run_config_map.get(run_name)
        if not config_rel:
            log.debug(f"No config mapping for {run_name}, skipping")
            continue

        config_abs = REPO / config_rel
        if not config_abs.exists():
            log.debug(f"Config not found: {config_abs}")
            continue

        try:
            config = json.loads(config_abs.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.debug(f"Failed to read config {config_abs}: {e}")
            continue

        fp = fingerprint_from_config(config, config_rel)
        if fp is None:
            continue

        results.append((ckpt_path, fp))

    return results


def find_match(target_config_path: str) -> Optional[Path]:
    """Find a pretrain checkpoint matching the given config."""
    config_path = Path(target_config_path)
    if not config_path.is_absolute():
        config_path = REPO / config_path

    config = json.loads(config_path.read_text())
    target_fp = fingerprint_from_config(config, config_path.name)
    if target_fp is None:
        log.warning("Config does not use SL/retrieval or has no pretrain phase")
        return None

    log.info(f"Looking for match: {target_fp}")

    indexed = index_checkpoints()
    for ckpt_path, fp in indexed:
        if fp == target_fp:
            log.info(f"Match found: {ckpt_path}")
            return ckpt_path

    log.info("No matching pretrain checkpoint found")
    return None


def auto_copy(target_config_path: str) -> Optional[Path]:
    """Find matching checkpoint and copy into the config's run_dir."""
    config_path = Path(target_config_path)
    if not config_path.is_absolute():
        config_path = REPO / config_path

    config = json.loads(config_path.read_text())
    run_dir = Path(config.get("output", {}).get("run_dir", ""))
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir

    if not run_dir.name:
        log.error("Config has no output.run_dir")
        return None

    dest = run_dir / "pretrain_checkpoint.pt"
    if dest.exists():
        log.info(f"Pretrain checkpoint already exists at {dest}")
        return dest

    source = find_match(target_config_path)
    if source is None:
        return None

    # Don't copy onto self
    if source.resolve() == dest.resolve():
        log.info("Source and destination are the same file")
        return dest

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    log.info(f"Copied {source} -> {dest}")
    return dest


def list_checkpoints():
    """Print all pretrain checkpoints with their fingerprints."""
    indexed = index_checkpoints()
    if not indexed:
        print("No pretrain checkpoints found.")
        return

    # Group by fingerprint for deduplication awareness
    by_fp: dict[PretrainFingerprint, list[Path]] = {}
    for path, fp in indexed:
        by_fp.setdefault(fp, []).append(path)

    print(f"\n{'='*80}")
    print(f"  Pretrain Checkpoints ({len(indexed)} total, {len(by_fp)} unique fingerprints)")
    print(f"{'='*80}\n")

    for fp, paths in sorted(by_fp.items(), key=lambda x: (x[0].task, x[0].d_latent)):
        print(f"  [{fp}]")
        for p in paths:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"    {p.parent.name}/pretrain_checkpoint.pt  ({size_mb:.1f} MB)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain checkpoint manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true",
                        help="List all pretrain checkpoints with fingerprints")
    parser.add_argument("--find-match", type=str, metavar="CONFIG",
                        help="Find a matching checkpoint for a config")
    parser.add_argument("--auto-copy", type=str, metavar="CONFIG",
                        help="Find match and copy into the config's run_dir")

    args = parser.parse_args()

    if args.list:
        list_checkpoints()
    elif args.find_match:
        match = find_match(args.find_match)
        if match:
            print(match)
        else:
            sys.exit(1)
    elif args.auto_copy:
        dest = auto_copy(args.auto_copy)
        if dest:
            print(dest)
        else:
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
