"""Generate 15 HiRID best-ablation configs from eICU seed templates.

Per-seed diff: only training.training_seed, output.run_dir, output.log_file.
Port diff: data_dir, paths.bounds_csv, paths.static_recipe, split_seed+seed -> 2222.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEEDS_DIR = REPO / "configs" / "seeds"
ABLATION_DIR = REPO / "configs" / "ablation"
OUT_DIR = REPO / "configs" / "hirid" / "ablation"
HIRID_REF_DIR = REPO / "configs" / "hirid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# task -> (eICU template stem, HiRID sibling for path lookup, output stem)
TASKS = {
    "mort_c2":          ("mort_c2_C5_no_fidelity_s42",               "mortality_hirid_sr",  "mort_c2_hirid_best"),
    "aki_v5_cross3":    ("aki_v5_cross3_C1_no_retrieval_s42",        "aki_hirid_sr",        "aki_v5_cross3_hirid_best"),
    "sepsis_v5_cross3": ("sepsis_v5_cross3_C1_no_retrieval_s42",     "sepsis_hirid_sr",     "sepsis_v5_cross3_hirid_best"),
    "los_v5_cross3":    ("los_v5_cross3_C3_no_mmd_s42",              "los_hirid_sr",        "los_v5_cross3_hirid_best"),
    "kf_v5_cross3":     ("kf_v5_cross3_C3_no_mmd_s42",               "kf_hirid_sr",         "kf_v5_cross3_hirid_best"),
}

SEEDS = [2222, 42, 7777]

written = []
for task, (eicu_stem, hirid_ref, out_stem) in TASKS.items():
    eicu_path = SEEDS_DIR / f"{eicu_stem}.json"
    hirid_path = HIRID_REF_DIR / f"{hirid_ref}.json"
    template = json.loads(eicu_path.read_text())
    hirid_ref_cfg = json.loads(hirid_path.read_text())

    # Port paths from eICU -> HiRID
    template["data_dir"] = hirid_ref_cfg["data_dir"]
    paths = template.setdefault("paths", {})
    paths["bounds_csv"] = hirid_ref_cfg["paths"]["bounds_csv"]
    paths["static_recipe"] = hirid_ref_cfg["paths"]["static_recipe"]

    # Fix split seed at 2222 (canonical fold)
    template["seed"] = 2222
    template["split_seed"] = 2222

    for seed in SEEDS:
        cfg = json.loads(json.dumps(template))  # deep copy
        # Only training_seed + output vary across the 3 variants
        cfg.setdefault("training", {})["training_seed"] = seed
        # Drop legacy training.seed if present (training_seed takes precedence anyway)
        cfg["training"].pop("seed", None)
        run_name = f"{out_stem}_s{seed}"
        cfg["output"] = {
            "run_dir": f"runs/hirid_best/{run_name}",
            "log_file": f"runs/hirid_best/{run_name}/run.log",
        }
        out_path = OUT_DIR / f"{run_name}.json"
        out_path.write_text(json.dumps(cfg, indent=2))
        written.append((str(out_path.relative_to(REPO)), seed))

print(f"Wrote {len(written)} configs:")
for p, s in written:
    print(f"  s{s:<5}  {p}")
