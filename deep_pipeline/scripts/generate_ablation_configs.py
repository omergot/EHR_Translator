#!/usr/bin/env python3
"""Generate ablation configs for NeurIPS 2026 paper.

Creates 10 ablation configs × 5 tasks = 50 configs in configs/ablation/.
Each config modifies exactly one variable from the V5 cross3 control.
"""

import json
import copy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
ABLATION_DIR = CONFIG_DIR / "ablation"
ABLATION_DIR.mkdir(exist_ok=True)

# Base configs for each task
BASE_CONFIGS = {
    "mortality": CONFIG_DIR / "mortality_retr_v5_cross3.json",
    "aki": CONFIG_DIR / "aki_v5_cross3.json",
    "sepsis": CONFIG_DIR / "sepsis_retr_v5_cross3.json",
    "los": CONFIG_DIR / "los_retr_v5_cross3.json",
    "kf": CONFIG_DIR / "kf_retr_v5_cross3.json",
}

# Ablation epochs per task (reduced for efficiency)
ABLATION_EPOCHS = {
    "mortality": 30,
    "aki": 35,
    "sepsis": 30,
    "los": 35,
    "kf": 35,
}

# Ablation definitions: (name, changes_to_training, changes_to_translator)
ABLATIONS = [
    # C0: control (just reduced epochs)
    ("C0_control", {}, {}),
    # C1: No retrieval (n_cross_layers=0)
    ("C1_no_retrieval", {"n_cross_layers": 0}, {}),
    # C2: No feature gate
    ("C2_no_feature_gate", {"feature_gate": False}, {}),
    # C3: No MMD alignment
    ("C3_no_mmd", {"lambda_align": 0.0}, {}),
    # C4: No target task loss
    ("C4_no_target_task", {"lambda_target_task": 0.0}, {}),
    # C5: No fidelity loss
    ("C5_no_fidelity", {"lambda_recon": 0.0}, {}),
    # C6: No Phase 1 pretrain
    ("C6_no_pretrain", {"pretrain_epochs": 0}, {}),
    # C7: No target normalization
    ("C7_no_target_norm", {"use_target_normalization": False}, {}),
    # C8: Residual output mode
    ("C8_residual", {"output_mode": "residual"}, {}),
    # C9: No triplet time delta
    ("C9_no_time_delta", {"disable_triplet_time_delta": True}, {}),
]


def generate_configs():
    """Generate all ablation configs."""
    created = []

    for task, base_path in BASE_CONFIGS.items():
        with open(base_path) as f:
            base = json.load(f)

        for abl_name, train_changes, trans_changes in ABLATIONS:
            config = copy.deepcopy(base)

            # Set ablation epochs
            config["training"]["epochs"] = ABLATION_EPOCHS[task]

            # Apply training changes
            for k, v in train_changes.items():
                config["training"][k] = v

            # Apply translator changes
            for k, v in trans_changes.items():
                config["translator"][k] = v

            # Set output paths
            run_name = f"{task}_{abl_name}"
            config["output"]["run_dir"] = f"runs/ablation/{run_name}"
            config["output"]["log_file"] = f"runs/ablation/{run_name}/run.log"

            # Write config
            out_path = ABLATION_DIR / f"{run_name}.json"
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)
                f.write("\n")

            created.append(out_path.name)

    return created


if __name__ == "__main__":
    configs = generate_configs()
    print(f"Created {len(configs)} ablation configs in {ABLATION_DIR}/:")
    for c in sorted(configs):
        print(f"  {c}")
