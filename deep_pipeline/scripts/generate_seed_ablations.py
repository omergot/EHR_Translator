#!/usr/bin/env python3
"""Generate multi-seed ablation configs for NeurIPS stability verification.

Creates configs/seeds/<group>/<ablation>_s<seed>.json for each group × ablation × seed.
Also generates queue entries to append to experiments/queue.yaml.
"""
import json
import copy
import sys
from pathlib import Path

# Ablation definitions: (code, name, changes_dict)
# changes_dict maps "section.key" -> new_value
ABLATIONS = [
    ("C0", "control", {}),
    ("C1", "no_retrieval", {"training.n_cross_layers": 0}),
    ("C2", "no_feature_gate", {"training.feature_gate": False}),
    ("C3", "no_mmd", {"training.lambda_align": 0.0}),
    ("C4", "no_target_task", {"training.lambda_target_task": 0.0}),
    ("C5", "no_fidelity", {"training.lambda_recon": 0.0}),
    ("C6", "no_pretrain", {"training.pretrain_epochs": 0}),
    ("C7", "no_target_norm", {"training.use_target_normalization": False}),
    ("C8", "residual", {"training.output_mode": "residual"}),
    ("C9", "no_time_delta", {"training.disable_triplet_time_delta": True}),
]

# Group definitions
GROUPS = {
    "mort_c2": {
        "base_config": "configs/ablation/mort_c2_C0_control.json",
        "seeds": [7777, 42],  # 2222 already exists
        "epochs": 30,
        "server": "a6000",
        "eval_name": "mortality",
        "ablations": "all",  # C0-C9
    },
    "sepsis_v5_cross3": {
        "base_config": "configs/seeds/sepsis_v5_cross3_s2222.json",  # Use as template
        "seeds": [2222, 7777, 42],  # None exist
        "epochs": 30,
        "server_rotation": ["local", "3090", "local"],  # Alternate
        "eval_name": "sepsis",
        "ablations": "all",
    },
    "sepsis_v5_cross2": {
        "base_config": "configs/seeds/sepsis_v5_cross2_s2222.json",
        "seeds": [2222, 7777, 42],
        "epochs": 30,
        "server_rotation": ["3090", "local", "3090"],
        "eval_name": "sepsis",
        "ablations": "all",
    },
    "aki_v5_cross3": {
        "base_config": "configs/ablation/aki_C0_control.json",
        "seeds": [7777, 42],
        "epochs": 35,
        "server": "a6000",
        "eval_name": "aki",
        "ablations": "all",
    },
    "kf_v5_cross3": {
        "base_config": "configs/ablation/kf_C0_control.json",
        "seeds": [7777, 42],
        "epochs": 35,
        "server": "athena",
        "athena_partition": "a100-public",
        "eval_name": "kf",
        "ablations": "all",
    },
    "kf_nfnm": {
        "base_config": "configs/ablation/kf_nfnm_C0_control.json",
        "seeds": [7777, 42],
        "epochs": 35,
        "server": "athena",
        "athena_partition": "a100-public",
        "eval_name": "kf",
        # kf_nfnm has C0-C4, C6-C9 (no C3=no_mmd or C5=no_fidelity since base already has both disabled)
        "ablations": ["C0", "C1", "C2", "C4", "C6", "C7", "C8", "C9"],
    },
    "los_v5_cross3": {
        "base_config": "configs/ablation/los_C0_control.json",
        "seeds": [7777, 42],
        "epochs": 35,
        "server": "athena",
        "athena_partition": "l40s-shared",
        "eval_name": "los",
        "ablations": "all",
    },
}


def apply_ablation(config, abl_code, abl_name, changes):
    """Apply ablation changes to a config dict."""
    cfg = copy.deepcopy(config)
    for key, value in changes.items():
        section, param = key.split(".", 1)
        cfg[section][param] = value
    return cfg


def generate_configs():
    """Generate all configs and queue entries."""
    output_dir = Path("configs/seeds")
    output_dir.mkdir(exist_ok=True)

    configs_created = []
    queue_entries = []

    for group_name, group_def in GROUPS.items():
        # Load base config
        base_path = group_def["base_config"]
        with open(base_path) as f:
            base_config = json.load(f)

        # Determine which ablations to run
        if group_def["ablations"] == "all":
            abl_list = ABLATIONS
        else:
            abl_list = [a for a in ABLATIONS if a[0] in group_def["ablations"]]

        # Override epochs if specified
        target_epochs = group_def.get("epochs")

        for seed in group_def["seeds"]:
            for abl_code, abl_name, changes in abl_list:
                # Create config
                cfg = apply_ablation(base_config, abl_code, abl_name, changes)
                cfg["seed"] = seed

                if target_epochs:
                    cfg["training"]["epochs"] = target_epochs

                # Set run_dir and log_file
                run_name = f"{group_name}_{abl_code}_{abl_name}_s{seed}"
                cfg["output"]["run_dir"] = f"runs/seeds/{run_name}"
                cfg["output"]["log_file"] = f"runs/seeds/{run_name}/run.log"

                # Write config
                config_path = output_dir / f"{run_name}.json"
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                    f.write("\n")

                configs_created.append(str(config_path))

                # Create queue entry
                eval_name = group_def["eval_name"]
                server = group_def.get("server", "local")

                # Handle server rotation for sepsis
                if "server_rotation" in group_def:
                    seed_idx = group_def["seeds"].index(seed)
                    server = group_def["server_rotation"][seed_idx]

                status = "athena_pending" if server == "athena" else "pending"

                entry = {
                    "name": run_name,
                    "config": str(config_path),
                    "output": f"runs/seeds/{run_name}/eval_{eval_name}.parquet",
                    "status": status,
                    "server": server,
                    "branch": "master",
                    "notes": f"{group_name} {abl_code} ({abl_name}) seed={seed}",
                }
                queue_entries.append(entry)

    return configs_created, queue_entries


def main():
    configs, entries = generate_configs()

    # Print summary
    print(f"Created {len(configs)} config files")
    print(f"Generated {len(entries)} queue entries")

    # Group summary
    from collections import Counter
    groups = Counter()
    servers = Counter()
    for e in entries:
        group = e["name"].rsplit("_s", 1)[0].rsplit("_", 2)[0]  # rough group extraction
        groups[e["notes"].split()[0]] += 1
        servers[e["server"]] += 1

    print("\nBy group:")
    for g, c in sorted(groups.items()):
        print(f"  {g}: {c}")

    print("\nBy server:")
    for s, c in sorted(servers.items()):
        print(f"  {s}: {c}")

    # Write queue entries to a temporary file for review
    import yaml
    with open("/tmp/seed_ablation_queue_entries.yaml", "w") as f:
        yaml.dump(entries, f, default_flow_style=False)
    print(f"\nQueue entries written to /tmp/seed_ablation_queue_entries.yaml")
    print("Review and append to experiments/queue.yaml")


if __name__ == "__main__":
    main()
