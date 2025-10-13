#!/usr/bin/env python3
"""
Check if the full arrays are identical, not just the first few elements.
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

# Create copies BEFORE calling evaluator
x_mimic_np_copy = x_mimic_np.copy()
x_mimic_roundtrip_np_copy = x_mimic_roundtrip_np.copy()

print("BEFORE calling evaluator:")
print(f"x_mimic_np checksum: {x_mimic_np.sum():.10f}")
print(f"x_mimic_roundtrip_np checksum: {x_mimic_roundtrip_np.sum():.10f}")
print()

# Create evaluator
evaluator = ComprehensiveEvaluator(model, feature_spec, 'evaluation')

# Get indices
all_features = numeric_features + missing_features
demographic_features = feature_spec.get('demographic_features', ['Age', 'Gender'])
clinical_only_features = [f for f in numeric_features if f not in demographic_features]
clinical_indices = [i for i, f in enumerate(all_features) if f in clinical_only_features]

# Extract BEFORE
x_mimic_clinical_before = x_mimic_np[:, clinical_indices]
x_mimic_roundtrip_clinical_before = x_mimic_roundtrip_np[:, clinical_indices]

print("BEFORE evaluator - extracted clinical features:")
print(f"Column 0 checksum: {x_mimic_clinical_before[:, 0].sum():.10f}")
print(f"Column 0 RT checksum: {x_mimic_roundtrip_clinical_before[:, 0].sum():.10f}")
print()

# Call evaluator
result = evaluator._compute_correlation_metrics(
    x_eicu_np, x_eicu_roundtrip_np, x_mimic_np, x_mimic_roundtrip_np
)

print("AFTER calling evaluator:")
print(f"x_mimic_np checksum: {x_mimic_np.sum():.10f}")
print(f"x_mimic_roundtrip_np checksum: {x_mimic_roundtrip_np.sum():.10f}")
print()

# Check if arrays were modified
print("Checking if arrays were modified:")
if np.array_equal(x_mimic_np, x_mimic_np_copy):
    print("  ✓ x_mimic_np: NOT modified")
else:
    diff_count = (x_mimic_np != x_mimic_np_copy).sum()
    print(f"  ✗ x_mimic_np: MODIFIED ({diff_count} elements changed)")
    
if np.array_equal(x_mimic_roundtrip_np, x_mimic_roundtrip_np_copy):
    print("  ✓ x_mimic_roundtrip_np: NOT modified")
else:
    diff_count = (x_mimic_roundtrip_np != x_mimic_roundtrip_np_copy).sum()
    print(f"  ✗ x_mimic_roundtrip_np: MODIFIED ({diff_count} elements changed)")
print()

# Extract AFTER
x_mimic_clinical_after = x_mimic_np[:, clinical_indices]
x_mimic_roundtrip_clinical_after = x_mimic_roundtrip_np[:, clinical_indices]

print("AFTER evaluator - extracted clinical features:")
print(f"Column 0 checksum: {x_mimic_clinical_after[:, 0].sum():.10f}")
print(f"Column 0 RT checksum: {x_mimic_roundtrip_clinical_after[:, 0].sum():.10f}")
print()

# Check if extracted arrays are identical
if np.array_equal(x_mimic_clinical_before, x_mimic_clinical_after):
    print("✓ Extracted clinical arrays are IDENTICAL")
else:
    print("✗ Extracted clinical arrays DIFFER")
    diff_count = (x_mimic_clinical_before != x_mimic_clinical_after).sum()
    print(f"  {diff_count} elements differ")
print()

# Now compute correlations
corr_before = np.corrcoef(x_mimic_clinical_before[:, 0], x_mimic_roundtrip_clinical_before[:, 0])[0, 1]
corr_after = np.corrcoef(x_mimic_clinical_after[:, 0], x_mimic_roundtrip_clinical_after[:, 0])[0, 1]

print("Correlation comparison:")
print(f"  Before: {corr_before:.10f}")
print(f"  After:  {corr_after:.10f}")
print(f"  DataFrame: {result['summary_df'].iloc[0]['mimic_correlation']:.10f}")

