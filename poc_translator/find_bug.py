#!/usr/bin/env python3
"""
Find the exact bug by printing array IDs and checking if they're the same objects.
"""

import torch
import numpy as np
import pandas as pd
import json
import yaml
import sys
from sklearn.metrics import r2_score

# Add src to path
sys.path.insert(0, 'src')
from model import CycleVAE
from comprehensive_evaluator import ComprehensiveEvaluator

# Load everything
with open('feature_spec.json', 'r') as f:
    feature_spec = json.load(f)

with open('config_used.yml', 'r') as f:
    config = yaml.safe_load(f)

checkpoint = torch.load('checkpoints/final_model.ckpt', map_location='cpu', weights_only=False)
model = CycleVAE(config, feature_spec)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Load and split data
test_mimic_data = pd.read_csv('data/test_mimic_preprocessed.csv')
n = len(test_mimic_data)
labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
rng = np.random.RandomState(42)
rng.shuffle(labels)
eicu_df = test_mimic_data[labels == 0].reset_index(drop=True)
mimic_df = test_mimic_data[labels == 1].reset_index(drop=True)

# Get feature columns
numeric_features = feature_spec['numeric_features']
missing_features = feature_spec['missing_features']

eicu_numeric_cols = [f for f in numeric_features if f in eicu_df.columns]
eicu_missing_cols = [f for f in missing_features if f in eicu_df.columns]
mimic_numeric_cols = [f for f in numeric_features if f in mimic_df.columns]
mimic_missing_cols = [f for f in missing_features if f in mimic_df.columns]

# Convert to tensors
eicu_numeric = torch.FloatTensor(eicu_df[eicu_numeric_cols].values)
eicu_missing = torch.FloatTensor(eicu_df[eicu_missing_cols].values)
mimic_numeric = torch.FloatTensor(mimic_df[mimic_numeric_cols].values)
mimic_missing = torch.FloatTensor(mimic_df[mimic_missing_cols].values)

x_eicu = torch.cat([eicu_numeric, eicu_missing], dim=1)
x_mimic = torch.cat([mimic_numeric, mimic_missing], dim=1)

# Move to device and compute IQR
device = next(model.parameters()).device
x_eicu = x_eicu.to(device)
x_mimic = x_mimic.to(device)
model.eval()
numeric_dim = model.numeric_dim
all_numeric = torch.cat([x_eicu[:, :numeric_dim], x_mimic[:, :numeric_dim]], dim=0)
model.feature_iqr = model.compute_feature_iqr(all_numeric)

# Translations (deterministic)
with torch.no_grad():
    x_eicu_to_mimic = model.translate_eicu_to_mimic_deterministic(x_eicu)
    x_mimic_to_eicu = model.translate_mimic_to_eicu_deterministic(x_mimic)
    x_eicu_roundtrip = model.translate_mimic_to_eicu_deterministic(x_eicu_to_mimic)
    x_mimic_roundtrip = model.translate_eicu_to_mimic_deterministic(x_mimic_to_eicu)

# Convert to numpy
x_eicu_np = x_eicu.detach().cpu().numpy()
x_mimic_np = x_mimic.detach().cpu().numpy()
x_eicu_roundtrip_np = x_eicu_roundtrip.detach().cpu().numpy()
x_mimic_roundtrip_np = x_mimic_roundtrip.detach().cpu().numpy()

print('=== Array IDs ===')
print(f'x_eicu_np: id={id(x_eicu_np)}, shape={x_eicu_np.shape}')
print(f'x_mimic_np: id={id(x_mimic_np)}, shape={x_mimic_np.shape}')
print(f'x_eicu_roundtrip_np: id={id(x_eicu_roundtrip_np)}, shape={x_eicu_roundtrip_np.shape}')
print(f'x_mimic_roundtrip_np: id={id(x_mimic_roundtrip_np)}, shape={x_mimic_roundtrip_np.shape}')
print()

# Create evaluator
evaluator = ComprehensiveEvaluator(model, feature_spec, 'evaluation')

print('=== Calling _compute_correlation_metrics ===')
print(f'Passing arrays with IDs:')
print(f'  x_eicu_np: {id(x_eicu_np)}')
print(f'  x_eicu_roundtrip_np: {id(x_eicu_roundtrip_np)}')
print(f'  x_mimic_np: {id(x_mimic_np)}')
print(f'  x_mimic_roundtrip_np: {id(x_mimic_roundtrip_np)}')
print()

# Instrument the method to see what it receives
original_method = evaluator._compute_correlation_metrics

def debug_method(x_eicu, x_eicu_roundtrip, x_mimic, x_mimic_roundtrip):
    print('Inside _compute_correlation_metrics:')
    print(f'  Received x_mimic: id={id(x_mimic)}, shape={x_mimic.shape}')
    print(f'  Received x_mimic_roundtrip: id={id(x_mimic_roundtrip)}, shape={x_mimic_roundtrip.shape}')
    
    # Extract clinical features
    x_mimic_clinical = x_mimic[:, evaluator.clinical_indices]
    x_mimic_roundtrip_clinical = x_mimic_roundtrip[:, evaluator.clinical_indices]
    
    print(f'  After extraction:')
    print(f'    x_mimic_clinical: id={id(x_mimic_clinical)}, shape={x_mimic_clinical.shape}')
    print(f'    x_mimic_roundtrip_clinical: id={id(x_mimic_roundtrip_clinical)}, shape={x_mimic_roundtrip_clinical.shape}')
    
    # Check first feature
    orig_0 = x_mimic_clinical[:, 0]
    rt_0 = x_mimic_roundtrip_clinical[:, 0]
    
    print(f'  Feature 0 (HR_min):')
    print(f'    orig first 5: {orig_0[:5]}')
    print(f'    rt first 5: {rt_0[:5]}')
    print(f'    Are they same array? {np.shares_memory(orig_0, rt_0)}')
    
    corr_0 = np.corrcoef(orig_0, rt_0)[0, 1]
    r2_0 = r2_score(orig_0, rt_0)
    print(f'    R²={r2_0:.6f}, Corr={corr_0:.6f}')
    print()
    
    # Call original
    result = original_method(x_eicu, x_eicu_roundtrip, x_mimic, x_mimic_roundtrip)
    
    print('After original method:')
    df = result['summary_df']
    hr_min = df[df['feature_name'] == 'HR_min'].iloc[0]
    print(f'  HR_min in DataFrame: R²={hr_min["mimic_r2"]:.6f}, Corr={hr_min["mimic_correlation"]:.6f}')
    print()
    
    return result

evaluator._compute_correlation_metrics = debug_method

# Call the method
results = evaluator._compute_correlation_metrics(
    x_eicu_np, x_eicu_roundtrip_np, x_mimic_np, x_mimic_roundtrip_np
)

df = results['summary_df']
print('=== Final DataFrame (first 5 rows) ===')
print(df[['feature_name', 'mimic_r2', 'mimic_correlation']].head())
print()

# Now manually compute to compare
print('=== Manual Computation Outside Evaluator ===')
x_mimic_clinical_manual = x_mimic_np[:, evaluator.clinical_indices]
x_mimic_roundtrip_clinical_manual = x_mimic_roundtrip_np[:, evaluator.clinical_indices]

orig_manual = x_mimic_clinical_manual[:, 0]
rt_manual = x_mimic_roundtrip_clinical_manual[:, 0]

print(f'Manual - orig first 5: {orig_manual[:5]}')
print(f'Manual - rt first 5: {rt_manual[:5]}')

corr_manual = np.corrcoef(orig_manual, rt_manual)[0, 1]
r2_manual = r2_score(orig_manual, rt_manual)

print(f'Manual - R²={r2_manual:.6f}, Corr={corr_manual:.6f}')
print()

# Compare
hr_min_df = df[df['feature_name'] == 'HR_min'].iloc[0]
print('=== Comparison ===')
print(f'Manual computation:  R²={r2_manual:.6f}, Corr={corr_manual:.6f}')
print(f'DataFrame result:    R²={hr_min_df["mimic_r2"]:.6f}, Corr={hr_min_df["mimic_correlation"]:.6f}')
print(f'CSV file:            R²=0.998299, Corr=0.618714')


