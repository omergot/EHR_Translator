#!/usr/bin/env python3
"""
Trace the exact bug by comparing arrays at each step.
"""

import torch
import numpy as np
import pandas as pd
import json
import yaml
import sys
from sklearn.metrics import r2_score

sys.path.insert(0, 'src')
from model import CycleVAE
from comprehensive_evaluator import ComprehensiveEvaluator

# Load everything
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

# Convert to tensors
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

# Convert to numpy - SAVE THE ORIGINAL REFERENCES
x_eicu_np = x_eicu.detach().cpu().numpy()
x_mimic_np = x_mimic.detach().cpu().numpy()
x_eicu_roundtrip_np = x_eicu_roundtrip.detach().cpu().numpy()
x_mimic_roundtrip_np = x_mimic_roundtrip.detach().cpu().numpy()

print("=" * 80)
print("STEP 1: Check input arrays")
print("=" * 80)
print(f"x_mimic_np[0:3, 0]: {x_mimic_np[0:3, 0]}")
print(f"x_mimic_roundtrip_np[0:3, 0]: {x_mimic_roundtrip_np[0:3, 0]}")
print()

# Create evaluator
evaluator = ComprehensiveEvaluator(model, feature_spec, 'evaluation')

print("=" * 80)
print("STEP 2: Manual extraction and correlation (BEFORE calling evaluator)")
print("=" * 80)

# Get the indices
all_features = numeric_features + missing_features
demographic_features = feature_spec.get('demographic_features', ['Age', 'Gender'])
clinical_only_features = [f for f in numeric_features if f not in demographic_features]
clinical_indices = [i for i, f in enumerate(all_features) if f in clinical_only_features]

print(f"clinical_indices: {clinical_indices}")
print(f"clinical_only_features[0]: {clinical_only_features[0]}")
print()

# Manual extraction
x_mimic_clinical_manual = x_mimic_np[:, clinical_indices]
x_mimic_roundtrip_clinical_manual = x_mimic_roundtrip_np[:, clinical_indices]

print(f"Manual extraction - column 0:")
print(f"  orig[0:3, 0]: {x_mimic_clinical_manual[0:3, 0]}")
print(f"  rt[0:3, 0]: {x_mimic_roundtrip_clinical_manual[0:3, 0]}")
print()

# Manual correlation
corr_manual = np.corrcoef(x_mimic_clinical_manual[:, 0], x_mimic_roundtrip_clinical_manual[:, 0])
print(f"Manual correlation matrix:")
print(corr_manual)
print(f"Manual correlation: {corr_manual[0, 1]:.10f}")
print()

print("=" * 80)
print("STEP 3: Call evaluator._compute_correlation_metrics")
print("=" * 80)

# Now call the evaluator
result = evaluator._compute_correlation_metrics(
    x_eicu_np, x_eicu_roundtrip_np, x_mimic_np, x_mimic_roundtrip_np
)

df = result['summary_df']
print(f"DataFrame correlation for {df.iloc[0]['feature_name']}: {df.iloc[0]['mimic_correlation']:.10f}")
print()

print("=" * 80)
print("STEP 4: Manual extraction and correlation (AFTER calling evaluator)")
print("=" * 80)

# Extract again after calling evaluator
x_mimic_clinical_after = x_mimic_np[:, clinical_indices]
x_mimic_roundtrip_clinical_after = x_mimic_roundtrip_np[:, clinical_indices]

print(f"After evaluator - column 0:")
print(f"  orig[0:3, 0]: {x_mimic_clinical_after[0:3, 0]}")
print(f"  rt[0:3, 0]: {x_mimic_roundtrip_clinical_after[0:3, 0]}")
print()

# Check if arrays changed
if np.array_equal(x_mimic_clinical_manual, x_mimic_clinical_after):
    print("✓ Arrays are IDENTICAL before and after")
else:
    print("✗ Arrays CHANGED!")
    
corr_after = np.corrcoef(x_mimic_clinical_after[:, 0], x_mimic_roundtrip_clinical_after[:, 0])
print(f"After correlation: {corr_after[0, 1]:.10f}")
print()

print("=" * 80)
print("STEP 5: Check what evaluator actually used")
print("=" * 80)
print(f"evaluator.clinical_indices: {evaluator.clinical_indices}")
print(f"evaluator.clinical_only_features[0]: {evaluator.clinical_only_features[0]}")
print()

# Verify indices match
if evaluator.clinical_indices == clinical_indices:
    print("✓ Indices MATCH")
else:
    print("✗ Indices DIFFER!")
    print(f"  Manual: {clinical_indices}")
    print(f"  Evaluator: {evaluator.clinical_indices}")

print()
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Manual correlation (before): {corr_manual[0, 1]:.10f}")
print(f"DataFrame correlation:       {df.iloc[0]['mimic_correlation']:.10f}")
print(f"Manual correlation (after):  {corr_after[0, 1]:.10f}")

