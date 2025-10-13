#!/usr/bin/env python3
"""
Debug the actual ascontiguousarray behavior on our real data.
"""

import torch
import numpy as np
import pandas as pd
import json
import yaml
import sys

sys.path.insert(0, 'src')
from model import CycleVAE

# Load
with open('feature_spec.json') as f:
    feature_spec = json.load(f)
with open('config_used.yml') as f:
    config = yaml.safe_load(f)

checkpoint = torch.load('checkpoints/final_model.ckpt', map_location='cpu', weights_only=False)
model = CycleVAE(config, feature_spec)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Load data
test_data = pd.read_csv('data/test_mimic_preprocessed.csv')
n = len(test_data)
labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
rng = np.random.RandomState(42)
rng.shuffle(labels)
eicu_df = test_data[labels == 0].reset_index(drop=True)
mimic_df = test_data[labels == 1].reset_index(drop=True)

numeric_features = feature_spec['numeric_features']
missing_features = feature_spec['missing_features']
eicu_numeric = torch.FloatTensor(eicu_df[[f for f in numeric_features if f in eicu_df.columns]].values)
eicu_missing = torch.FloatTensor(eicu_df[[f for f in missing_features if f in eicu_df.columns]].values)
mimic_numeric = torch.FloatTensor(mimic_df[[f for f in numeric_features if f in mimic_df.columns]].values)
mimic_missing = torch.FloatTensor(mimic_df[[f for f in missing_features if f in mimic_df.columns]].values)
x_eicu = torch.cat([eicu_numeric, eicu_missing], dim=1)
x_mimic = torch.cat([mimic_numeric, mimic_missing], dim=1)

# Compute roundtrips
device = next(model.parameters()).device
x_eicu = x_eicu.to(device)
x_mimic = x_mimic.to(device)
numeric_dim = model.numeric_dim
all_numeric = torch.cat([x_eicu[:, :numeric_dim], x_mimic[:, :numeric_dim]], dim=0)
model.feature_iqr = model.compute_feature_iqr(all_numeric)

with torch.no_grad():
    x_eicu_to_mimic = model.translate_eicu_to_mimic_deterministic(x_eicu)
    x_mimic_to_eicu = model.translate_mimic_to_eicu_deterministic(x_mimic)
    x_eicu_roundtrip = model.translate_mimic_to_eicu_deterministic(x_eicu_to_mimic)
    x_mimic_roundtrip = model.translate_eicu_to_mimic_deterministic(x_mimic_to_eicu)

x_mimic_np = x_mimic.detach().cpu().numpy()
x_mimic_roundtrip_np = x_mimic_roundtrip.detach().cpu().numpy()

# Get indices
all_features = numeric_features + missing_features
demographic_features = feature_spec.get('demographic_features', ['Age', 'Gender'])
clinical_only_features = [f for f in numeric_features if f not in demographic_features]
clinical_indices = [i for i, f in enumerate(all_features) if f in clinical_only_features]

# Extract clinical features
x_mimic_clinical = x_mimic_np[:, clinical_indices]
x_mimic_roundtrip_clinical = x_mimic_roundtrip_np[:, clinical_indices]

# Get column 0
col_orig = x_mimic_clinical[:, 0]
col_rt = x_mimic_roundtrip_clinical[:, 0]

print("Original columns:")
print(f"col_orig[:5]: {col_orig[:5]}")
print(f"col_rt[:5]: {col_rt[:5]}")
print(f"Are equal? {np.array_equal(col_orig, col_rt)}")
print()

# Convert to contiguous
col_orig_cont = np.ascontiguousarray(col_orig)
col_rt_cont = np.ascontiguousarray(col_rt)

print("After ascontiguousarray:")
print(f"col_orig_cont[:5]: {col_orig_cont[:5]}")
print(f"col_rt_cont[:5]: {col_rt_cont[:5]}")
print(f"Are equal? {np.array_equal(col_orig_cont, col_rt_cont)}")
print(f"Same object? {col_orig_cont is col_rt_cont}")
print(f"Share memory? {np.shares_memory(col_orig_cont, col_rt_cont)}")
print()

# Check if ascontiguousarray returned the same object
print(f"col_orig_cont is col_orig? {col_orig_cont is col_orig}")
print(f"col_rt_cont is col_rt? {col_rt_cont is col_rt}")
print(f"col_orig_cont is col_rt? {col_orig_cont is col_rt}")
print(f"col_rt_cont is col_orig? {col_rt_cont is col_orig}")
print()

# Compute correlations
corr_orig = np.corrcoef(col_orig, col_rt)[0, 1]
corr_cont = np.corrcoef(col_orig_cont, col_rt_cont)[0, 1]

print(f"Correlation (original): {corr_orig:.10f}")
print(f"Correlation (contiguous): {corr_cont:.10f}")
print()

# Try passing the arrays directly to corrcoef
print("Testing direct corrcoef call:")
result1 = np.corrcoef(col_orig, col_rt)
result2 = np.corrcoef(np.ascontiguousarray(col_orig), np.ascontiguousarray(col_rt))

print(f"result1 (original):\n{result1}")
print(f"result2 (contiguous):\n{result2}")

