#!/usr/bin/env python3
"""Generate experiment configs for A/B/C recommendation experiments.

Creates 26 configs (13 experiments x 2 tasks) from base templates.
Output: experiments/configs/<id>_<task>_debug.json
"""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SEPSIS_BASE = REPO / "configs" / "exp_sepsis_oversample20_graddiag_debug.json"
MORTALITY_BASE = REPO / "configs" / "exp_mortality24_full_30ep.json"

TARGET_DATA = {
    "sepsis": "/bigdata/omerg/Thesis/cohort_data/sepsis/miiv",
    "mortality": "/bigdata/omerg/Thesis/cohort_data/mortality24/miiv",
}

EXPERIMENTS = {
    "c1_focal": {
        "training": {"focal_gamma": 2.0, "focal_alpha": 0.75},
    },
    "c3_cosine_fid": {
        "training": {"cosine_fidelity": True},
    },
    "a3_padding_fid": {
        "training": {"padding_aware_fidelity": True, "fidelity_proximity_alpha": 1.0},
    },
    "a1_var_batching": {
        "training": {"variable_length_batching": True},
    },
    "a4_truncate": {
        "training": {"max_seq_len": 72},  # median + 1std for sepsis; mortality is 24 anyway
    },
    "c2_gradnorm": {
        "training": {"use_gradnorm": True, "gradnorm_alpha": 0.3},
    },
    "a2_chunking": {
        "training": {"chunk_size": 30, "chunk_overlap": 5},
    },
    "b1_hidden_mmd": {
        "training": {"lambda_hidden_mmd": 0.2},
        "needs_target": True,
    },
    "b3_knn": {
        "training": {"lambda_knn": 0.1, "knn_k": 5, "knn_temperature": 0.1},
        "needs_target": True,
    },
    "b5_ot": {
        "training": {"lambda_ot": 0.2, "ot_reg": 0.1},
        "needs_target": True,
    },
    "b6_dann": {
        "training": {"lambda_adversarial": 0.2},
        "needs_target": True,
    },
    "b4_contrastive": {
        "training": {"lambda_contrastive": 0.1, "contrastive_temperature": 0.07},
        "needs_target": True,
    },
    "b2_shared_enc": {
        "training": {"lambda_shared_encoder": 0.1},
        "needs_target": True,
    },
}


def make_config(base_path: Path, exp_id: str, task: str, overrides: dict) -> dict:
    with open(base_path) as f:
        cfg = json.load(f)

    # Force debug mode and 20 epochs
    cfg["debug"] = True
    cfg.setdefault("training", {})
    cfg["training"]["epochs"] = 20

    # Set experiment-specific output paths
    cfg["output"] = cfg.get("output", {})
    cfg["output"]["run_dir"] = f"runs/exp_{exp_id}_{task}"
    cfg["output"]["log_file"] = str(REPO / "experiments" / "logs" / f"{exp_id}_{task}.log")

    # Apply training overrides
    for key, val in overrides.get("training", {}).items():
        cfg["training"][key] = val

    # Set target_data_dir if needed
    if overrides.get("needs_target"):
        cfg["target_data_dir"] = TARGET_DATA[task]

    return cfg


def main():
    out_dir = REPO / "experiments" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for exp_id, overrides in EXPERIMENTS.items():
        for task, base_path in [("sepsis", SEPSIS_BASE), ("mortality", MORTALITY_BASE)]:
            cfg = make_config(base_path, exp_id, task, overrides)
            out_path = out_dir / f"{exp_id}_{task}_debug.json"
            with open(out_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"  {out_path.name}")

    print(f"\nGenerated {len(EXPERIMENTS) * 2} configs in {out_dir}")


if __name__ == "__main__":
    main()
