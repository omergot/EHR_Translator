#!/usr/bin/env python3
"""
Definitive test to find the exact bug.
"""

import torch
import numpy as np
import pandas as pd
import json
import yaml
import sys

sys.path.insert(0, 'src')
from model import CycleVAE
from comprehensive_evaluator import ComprehensiveEvaluator

# Load model and data
with open('feature_spec.json') as f:
    feature_spec = json.load(f)
with open('config_used.yml') as f:
    config = yaml.safe_load(f)

checkpoint = torch.load('checkpoints/final_model.ckpt', map_location='cpu', weights_only=False)
model = CycleVAE(config, feature_spec)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

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

# Save FULL arrays before
x_mimic_np_before = x_mimic_np.copy()
x_mimic_roundtrip_np_before = x_mimic_roundtrip_np.copy()

evaluator = ComprehensiveEvaluator(model, feature_spec, 'evaluation')

# Get indices
all_features = numeric_features + missing_features
demographic_features = feature_spec.get('demographic_features', ['Age', 'Gender'])
clinical_only_features = [f for f in numeric_features if f not in demographic_features]
clinical_indices = [i for i, f in enumerate(all_features) if f in clinical_only_features]

# Extract and compute correlation 3 times BEFORE calling evaluator
x_mc = x_mimic_np[:, clinical_indices]
x_mrc = x_mimic_roundtrip_np[:, clinical_indices]

corr_before_1 = np.corrcoef(x_mc[:, 0], x_mrc[:, 0])[0, 1]
corr_before_2 = np.corrcoef(x_mc[:, 0], x_mrc[:, 0])[0, 1]
corr_before_3 = np.corrcoef(x_mc[:, 0], x_mrc[:, 0])[0, 1]

print("BEFORE calling evaluator - computing correlation 3 times:")
print(f"  1: {corr_before_1:.10f}")
print(f"  2: {corr_before_2:.10f}")
print(f"  3: {corr_before_3:.10f}")
print(f"  All identical? {corr_before_1 == corr_before_2 == corr_before_3}")
print()

# Now call evaluator
result = evaluator._compute_correlation_metrics(
    x_eicu_np, x_eicu_roundtrip_np, x_mimic_np, x_mimic_roundtrip_np
)

df_corr = result['summary_df'].iloc[0]['mimic_correlation']
print(f"Evaluator returned: {df_corr:.10f}")
print()

# Compute correlation 3 times AFTER calling evaluator
corr_after_1 = np.corrcoef(x_mc[:, 0], x_mrc[:, 0])[0, 1]
corr_after_2 = np.corrcoef(x_mc[:, 0], x_mrc[:, 0])[0, 1]
corr_after_3 = np.corrcoef(x_mc[:, 0], x_mrc[:, 0])[0, 1]

print("AFTER calling evaluator - computing correlation 3 times:")
print(f"  1: {corr_after_1:.10f}")
print(f"  2: {corr_after_2:.10f}")
print(f"  3: {corr_after_3:.10f}")
print(f"  All identical? {corr_after_1 == corr_after_2 == corr_after_3}")
print()

# Check if arrays changed
arrays_changed = not np.array_equal(x_mimic_np, x_mimic_np_before) or not np.array_equal(x_mimic_roundtrip_np, x_mimic_roundtrip_np_before)
print(f"Did arrays change? {arrays_changed}")
if arrays_changed:
    diff1 = (x_mimic_np != x_mimic_np_before).sum()
    diff2 = (x_mimic_roundtrip_np != x_mimic_roundtrip_np_before).sum()
    print(f"  x_mimic_np: {diff1} elements changed")
    print(f"  x_mimic_roundtrip_np: {diff2} elements changed")
print()

print("SUMMARY:")
print(f"  Correlation BEFORE evaluator: {corr_before_1:.10f}")
print(f"  Correlation from evaluator:   {df_corr:.10f}")
print(f"  Correlation AFTER evaluator:  {corr_after_1:.10f}")
print(f"  Match? BEFORE==DF: {abs(corr_before_1 - df_corr) < 1e-8}, DF==AFTER: {abs(df_corr - corr_after_1) < 1e-8}")

