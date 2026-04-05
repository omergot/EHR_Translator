#!/usr/bin/env python3
"""Verify the E2EDataset fix gives AUROC 0.808 for mortality."""

import sys
import os
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.WARNING)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = Path("/bigdata/omerg/Thesis/EHR_Translator/deep_pipeline/experiments/.worktree_configs/e2e_cluda_mortality_v2.json")
with open(config_path) as f:
    config = json.load(f)
config["_config_path"] = str(config_path)

sys.path.insert(0, str(Path("/bigdata/omerg/Thesis/EHR_Translator/deep_pipeline")))
from src.adapters.yaib import YAIBRuntime
from src.baselines.end_to_end.data_adapter import YAIBToE2EAdapter, E2EDataset
from icu_benchmarks.data.constants import DataSplit

# ============================================================
# Method 1: YAIB reference
# ============================================================
print("Loading YAIB reference...")
rt = YAIBRuntime(
    data_dir=Path(config["data_dir"]),
    baseline_model_dir=Path(config["baseline_model_dir"]),
    task_config=Path(config["task_config"]),
    model_config=Path(config["model_config"]) if config.get("model_config") else None,
    model_name=config["model_name"],
    vars=copy.deepcopy(config["vars"]),
    file_names=copy.deepcopy(config["file_names"]),
    seed=config.get("seed", 2222),
    batch_size=64,
)
rt.load_data()
rt.load_baseline_model()
model = rt._model.to(device)
model.eval()

ds = rt.create_dataset("test", ram_cache=True)
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, drop_last=False)

probs_yaib, labels_yaib = [], []
with torch.no_grad():
    for data, labels, mask in loader:
        data = data.float().to(device)
        mask = mask.to(device).bool()
        logits = model(data)
        probs = torch.softmax(logits, dim=-1)[:, :, 1]
        probs_yaib.append(torch.masked_select(probs, mask).cpu().numpy())
        labels_yaib.append(torch.masked_select(labels.to(device), mask).cpu().numpy())
probs_yaib = np.concatenate(probs_yaib)
labels_yaib = np.concatenate(labels_yaib)
auroc_yaib = roc_auc_score(labels_yaib, probs_yaib)
print(f"YAIB reference: AUROC={auroc_yaib:.4f}")

# ============================================================
# Method 2: Fixed E2E adapter
# ============================================================
print("\nLoading FIXED E2E adapter...")
adapter = YAIBToE2EAdapter(config)
source_train, source_val, source_test, target_train = adapter.get_loaders()

# Verify fix: check sample 0
x0, y0, s0, vm0 = source_test.dataset[0]
print(f"E2E sample 0: x shape={x0.shape}, vmask valid={vm0.sum().item()}")
print(f"  x range: [{x0.min():.4f}, {x0.max():.4f}]")

# Reload model
rt2 = YAIBRuntime(
    data_dir=Path(config["data_dir"]),
    baseline_model_dir=Path(config["baseline_model_dir"]),
    task_config=Path(config["task_config"]),
    model_config=Path(config["model_config"]) if config.get("model_config") else None,
    model_name=config["model_name"],
    vars=copy.deepcopy(config["vars"]),
    file_names=copy.deepcopy(config["file_names"]),
    seed=config.get("seed", 2222),
    batch_size=64,
)
rt2.load_data()
rt2.load_baseline_model()
model2 = rt2._model.to(device)
model2.eval()

# Exact reproduction of cli.py E2E baseline eval
probs_e2e, labels_e2e = [], []
with torch.no_grad():
    for batch in source_test:
        x, labels, static, vmask = batch
        x_lstm = x.transpose(1, 2).to(device)
        labels = labels.to(device)
        vmask = vmask.to(device)

        logits = model2(x_lstm)
        if logits.dim() == 3:
            probs = torch.softmax(logits, dim=-1)[:, :, 1]
        else:
            probs = torch.softmax(logits, dim=-1)[:, 1]

        if labels.dim() == 1:
            valid = labels >= 0
            if valid.sum() > 0 and probs.dim() == 2:
                last_valid_idx = vmask.long().cumsum(dim=1).argmax(dim=1)
                per_stay_probs = probs[torch.arange(probs.size(0), device=device), last_valid_idx]
                probs_e2e.append(per_stay_probs[valid].cpu().numpy())
                labels_e2e.append(labels[valid].cpu().numpy())
            elif valid.sum() > 0:
                probs_e2e.append(probs[valid].cpu().numpy())
                labels_e2e.append(labels[valid].cpu().numpy())

probs_e2e = np.concatenate(probs_e2e)
labels_e2e = np.concatenate(labels_e2e)
auroc_e2e = roc_auc_score(labels_e2e, probs_e2e)
print(f"\nFixed E2E: AUROC={auroc_e2e:.4f}")
print(f"YAIB ref:  AUROC={auroc_yaib:.4f}")
print(f"Gap: {auroc_e2e - auroc_yaib:+.4f}")

if abs(auroc_e2e - auroc_yaib) < 0.001:
    print("\nSUCCESS: Gap closed! E2E now matches YAIB baseline.")
else:
    print(f"\nWARNING: Gap still {auroc_e2e - auroc_yaib:+.4f}")
