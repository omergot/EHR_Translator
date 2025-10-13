#!/usr/bin/env python3
"""
Check if arrays are views vs copies and if that affects np.corrcoef.
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

x_eicu_np = x_eicu.detach().cpu().numpy()
x_mimic_np = x_mimic.detach().cpu().numpy()
x_eicu_roundtrip_np = x_eicu_roundtrip.detach().cpu().numpy()
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

print("=" * 80)
print("Array Properties")
print("=" * 80)
print(f"col_orig base: {col_orig.base is not None} (is a view: {col_orig.base is x_mimic_clinical or col_orig.base is x_mimic_np})")
print(f"col_rt base: {col_rt.base is not None} (is a view: {col_rt.base is x_mimic_roundtrip_clinical or col_rt.base is x_mimic_roundtrip_np})")
print()

print(f"col_orig flags:")
print(f"  C_CONTIGUOUS: {col_orig.flags['C_CONTIGUOUS']}")
print(f"  F_CONTIGUOUS: {col_orig.flags['F_CONTIGUOUS']}")
print(f"  OWNDATA: {col_orig.flags['OWNDATA']}")
print()

print(f"col_rt flags:")
print(f"  C_CONTIGUOUS: {col_rt.flags['C_CONTIGUOUS']}")
print(f"  F_CONTIGUOUS: {col_rt.flags['F_CONTIGUOUS']}")
print(f"  OWNDATA: {col_rt.flags['OWNDATA']}")
print()

# Check if they're accidentally the same object
print(f"Same object? {col_orig is col_rt}")
print(f"Same base? {col_orig.base is col_rt.base if col_orig.base is not None and col_rt.base is not None else False}")
print(f"Share memory? {np.shares_memory(col_orig, col_rt)}")
print()

# Print actual values
print(f"col_orig[:10]: {col_orig[:10]}")
print(f"col_rt[:10]: {col_rt[:10]}")
print(f"Are values equal? {np.array_equal(col_orig, col_rt)}")
print()

# Try different ways to compute correlation
print("=" * 80)
print("Testing np.corrcoef with different methods")
print("=" * 80)

# Method 1: Direct
corr1 = np.corrcoef(col_orig, col_rt)[0, 1]
print(f"Method 1 (direct): {corr1:.10f}")

# Method 2: With explicit copies
corr2 = np.corrcoef(col_orig.copy(), col_rt.copy())[0, 1]
print(f"Method 2 (copies): {corr2:.10f}")

# Method 3: Flatten
corr3 = np.corrcoef(col_orig.ravel(), col_rt.ravel())[0, 1]
print(f"Method 3 (ravel): {corr3:.10f}")

# Method 4: As contiguous
corr4 = np.corrcoef(np.ascontiguousarray(col_orig), np.ascontiguousarray(col_rt))[0, 1]
print(f"Method 4 (contiguous): {corr4:.10f}")

# Method 5: Manual correlation
mean_orig = col_orig.mean()
mean_rt = col_rt.mean()
std_orig = col_orig.std()
std_rt = col_rt.std()
cov = ((col_orig - mean_orig) * (col_rt - mean_rt)).mean()
corr_manual = cov / (std_orig * std_rt)
print(f"Method 5 (manual): {corr_manual:.10f}")

