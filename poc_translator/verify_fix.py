#!/usr/bin/env python3
"""
Verify that the correlation bug fix works.
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

print("=" * 80)
print("VERIFYING CORRELATION BUG FIX")
print("=" * 80)
print()

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
model.feature_iqr = model.compute_feature_iqr(torch.cat([x_eicu[:, :model.numeric_dim], x_mimic[:, :model.numeric_dim]], dim=0))

# Compute translations
with torch.no_grad():
    x_eicu_to_mimic = model.translate_eicu_to_mimic_deterministic(x_eicu)
    x_mimic_to_eicu = model.translate_mimic_to_eicu_deterministic(x_mimic)
    x_eicu_roundtrip = model.translate_mimic_to_eicu_deterministic(x_eicu_to_mimic)
    x_mimic_roundtrip = model.translate_eicu_to_mimic_deterministic(x_mimic_to_eicu)

x_eicu_np = x_eicu.detach().cpu().numpy()
x_mimic_np = x_mimic.detach().cpu().numpy()
x_eicu_roundtrip_np = x_eicu_roundtrip.detach().cpu().numpy()
x_mimic_roundtrip_np = x_mimic_roundtrip.detach().cpu().numpy()

# Create evaluator
print("Creating evaluator (with single-threaded BLAS fix)...")
evaluator = ComprehensiveEvaluator(model, feature_spec, 'evaluation')

# Test 1: Check if correlation is deterministic
print("\n" + "=" * 80)
print("TEST 1: Determinism Check")
print("=" * 80)
print("Computing correlation 5 times on the same arrays...")

all_features = numeric_features + missing_features
demographic_features = feature_spec.get('demographic_features', ['Age', 'Gender'])
clinical_only_features = [f for f in numeric_features if f not in demographic_features]
clinical_indices = [i for i, f in enumerate(all_features) if f in clinical_only_features]

x_mc = x_mimic_np[:, clinical_indices]
x_mrc = x_mimic_roundtrip_np[:, clinical_indices]

corrs = []
for i in range(5):
    corr = np.corrcoef(x_mc[:, 0], x_mrc[:, 0])[0, 1]
    corrs.append(corr)
    print(f"  Run {i+1}: {corr:.10f}")

if len(set(corrs)) == 1:
    print("✅ PASS: All correlations are identical (deterministic)")
else:
    print("❌ FAIL: Correlations differ (still non-deterministic)")
    print(f"   Unique values: {set(corrs)}")

# Test 2: Check if evaluator gives correct results
print("\n" + "=" * 80)
print("TEST 2: Evaluator Correctness")
print("=" * 80)
print("Calling evaluator._compute_correlation_metrics...")

result = evaluator._compute_correlation_metrics(
    x_eicu_np, x_eicu_roundtrip_np, x_mimic_np, x_mimic_roundtrip_np
)

df = result['summary_df']
eval_corr = df.iloc[0]['mimic_correlation']
manual_corr = corrs[0]

print(f"Manual correlation: {manual_corr:.10f}")
print(f"Evaluator correlation: {eval_corr:.10f}")
print(f"Difference: {abs(manual_corr - eval_corr):.10e}")

if abs(manual_corr - eval_corr) < 1e-8:
    print("✅ PASS: Evaluator matches manual computation")
else:
    print("❌ FAIL: Evaluator differs from manual computation")

# Test 3: Check overall metrics quality
print("\n" + "=" * 80)
print("TEST 3: Overall Metrics Quality")
print("=" * 80)
print(f"Features with good quality: {df['mimic_good_quality'].sum()}/{len(df)}")
print(f"Mean R²: {df['mimic_r2'].mean():.6f}")
print(f"Mean Correlation: {df['mimic_correlation'].mean():.6f}")
print()
print("First 5 features:")
print(df[['feature_name', 'mimic_r2', 'mimic_correlation']].head().to_string(index=False))

low_corr = df[df['mimic_correlation'] < 0.95]
if len(low_corr) == 0:
    print("\n✅ PASS: All features have correlation >= 0.95")
else:
    print(f"\n⚠  {len(low_corr)} features with correlation < 0.95:")
    print(low_corr[['feature_name', 'mimic_r2', 'mimic_correlation']].to_string(index=False))

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)

