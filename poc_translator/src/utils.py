#!/usr/bin/env python3
"""
Utility Functions for Cycle-VAE
Helper functions for MMD, KS tests, and other utilities.
"""

import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel
import pickle
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) using RBF kernel
    
    Args:
        X: First sample array [n_samples, n_features]
        Y: Second sample array [m_samples, n_features]
        sigma: RBF kernel bandwidth. If None, computed as median pairwise distance
        
    Returns:
        MMD value
    """
    if sigma is None:
        # Compute sigma as median pairwise distance
        distances = []
        for i in range(min(1000, len(X))):  # Sample for efficiency
            for j in range(min(1000, len(Y))):
                distances.append(np.linalg.norm(X[i] - Y[j]))
        sigma = np.median(distances)
    
    # Compute kernel matrices
    XX = rbf_kernel(X, X, gamma=1.0/(2*sigma**2))
    YY = rbf_kernel(Y, Y, gamma=1.0/(2*sigma**2))
    XY = rbf_kernel(X, Y, gamma=1.0/(2*sigma**2))
    
    # Compute MMD
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    
    return mmd

def ks_test_featurewise(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Kolmogorov-Smirnov test for each feature
    
    Args:
        X: First sample array [n_samples, n_features]
        Y: Second sample array [m_samples, n_features]
        
    Returns:
        Tuple of (ks_statistics, p_values)
    """
    n_features = X.shape[1]
    ks_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)
    
    for i in range(n_features):
        ks_stats[i], p_values[i] = stats.ks_2samp(X[:, i], Y[:, i])
    
    return ks_stats, p_values

def save_pickle(obj, filepath: str):
    """Save object to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {filepath}")

def load_pickle(filepath: str):
    """Load object from pickle file"""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded object from {filepath}")
    return obj

def save_json(obj, filepath: str):
    """Save object to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=2)
    logger.info(f"Saved object to {filepath}")

def load_json(filepath: str):
    """Load object from JSON file"""
    with open(filepath, 'r') as f:
        obj = json.load(f)
    logger.info(f"Loaded object from {filepath}")
    return obj

def map_feature_names(eicu_features: pd.DataFrame, mimic_features: pd.DataFrame) -> dict:
    """
    Create mapping between eICU and MIMIC feature names
    
    Args:
        eicu_features: DataFrame with eICU feature names
        mimic_features: DataFrame with MIMIC feature names
        
    Returns:
        Dictionary mapping eICU names to MIMIC names
    """
    mapping = {}
    
    # Assuming both DataFrames have 'Feature_Name' and 'Index' columns
    for _, eicu_row in eicu_features.iterrows():
        eicu_name = eicu_row['Feature_Name']
        eicu_idx = eicu_row['Index']
        
        # Find corresponding MIMIC feature by index
        mimic_row = mimic_features[mimic_features['Index'] == eicu_idx]
        if not mimic_row.empty:
            mimic_name = mimic_row.iloc[0]['Feature_Name']
            mapping[eicu_name] = mimic_name
    
    return mapping

def compute_feature_statistics(data: pd.DataFrame, feature_cols: list) -> dict:
    """
    Compute basic statistics for features
    
    Args:
        data: DataFrame with feature data
        feature_cols: List of feature column names
        
    Returns:
        Dictionary with statistics
    """
    stats_dict = {}
    
    for col in feature_cols:
        stats_dict[col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'missing_pct': (data[col].isnull().sum() / len(data)) * 100
        }
    
    return stats_dict

def clip_outliers(data: pd.DataFrame, feature_cols: list, n_std: float = 3) -> pd.DataFrame:
    """
    Clip outliers using n standard deviations from mean
    
    Args:
        data: DataFrame with feature data
        feature_cols: List of feature column names
        n_std: Number of standard deviations for clipping
        
    Returns:
        DataFrame with clipped values
    """
    data_clipped = data.copy()
    
    for col in feature_cols:
        mean_val = data[col].mean()
        std_val = data[col].std()
        
        lower_bound = mean_val - n_std * std_val
        upper_bound = mean_val + n_std * std_val
        
        data_clipped[col] = data_clipped[col].clip(lower=lower_bound, upper=upper_bound)
    
    return data_clipped

def create_feature_groups(feature_names: list) -> dict:
    """
    Group features by type (vitals, labs, etc.)
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Dictionary with feature groups
    """
    groups = {
        'vitals': [],
        'labs': [],
        'gcs': [],
        'other': []
    }
    
    vital_keywords = ['heart rate', 'respiratory', 'temperature', 'blood pressure', 'o2', 'fio2', 'sao2', 'pao2', 'po2', 'pco2', 'ph']
    lab_keywords = ['creatinine', 'sodium', 'potassium', 'magnesium', 'hemoglobin', 'hgb', 'hematocrit', 'hct', 'wbc', 'rbc', 'rdw', 'mchc', 'mcv', 'lymphocytes', 'lymphs', 'bun', 'urea nitrogen', 'albumin', 'bilirubin', 'ast', 'sgot', 'alt', 'sgpt', 'alkaline', 'alkaline phos', 'lactate', 'pt', 'inr', 'crp', 'c-reactive protein', 'ldh', 'lactate dehydrogenase']
    gcs_keywords = ['gcs', 'eye', 'motor', 'verbal', 'gcseyes', 'gcsmotor', 'gcsverbal']
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        if any(keyword in feature_lower for keyword in vital_keywords):
            groups['vitals'].append(feature)
        elif any(keyword in feature_lower for keyword in lab_keywords):
            groups['labs'].append(feature)
        elif any(keyword in feature_lower for keyword in gcs_keywords):
            groups['gcs'].append(feature)
        else:
            groups['other'].append(feature)
    
    return groups

def validate_data_quality(data: pd.DataFrame, feature_cols: list) -> dict:
    """
    Validate data quality and return quality metrics
    
    Args:
        data: DataFrame with feature data
        feature_cols: List of feature column names
        
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {
        'total_samples': len(data),
        'total_features': len(feature_cols),
        'missing_data': {},
        'outliers': {},
        'zero_variance': [],
        'correlations': {}
    }
    
    # Check missing data
    for col in feature_cols:
        missing_pct = (data[col].isnull().sum() / len(data)) * 100
        quality_metrics['missing_data'][col] = missing_pct
    
    # Check for zero variance features
    for col in feature_cols:
        if data[col].std() == 0:
            quality_metrics['zero_variance'].append(col)
    
    # Check for outliers (beyond 3 std)
    for col in feature_cols:
        mean_val = data[col].mean()
        std_val = data[col].std()
        outliers = ((data[col] < mean_val - 3*std_val) | (data[col] > mean_val + 3*std_val)).sum()
        quality_metrics['outliers'][col] = outliers
    
    # Compute feature correlations
    if len(feature_cols) > 1:
        corr_matrix = data[feature_cols].corr()
        high_corr_pairs = []
        
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.9:  # High correlation threshold
                    high_corr_pairs.append((feature_cols[i], feature_cols[j], corr_val))
        
        quality_metrics['correlations']['high_corr_pairs'] = high_corr_pairs
    
    return quality_metrics

def create_summary_report(data: pd.DataFrame, feature_cols: list, output_path: str):
    """
    Create a comprehensive data summary report
    
    Args:
        data: DataFrame with feature data
        feature_cols: List of feature column names
        output_path: Path to save the report
    """
    report = {
        'dataset_info': {
            'total_samples': len(data),
            'total_features': len(feature_cols),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
        },
        'feature_statistics': compute_feature_statistics(data, feature_cols),
        'quality_metrics': validate_data_quality(data, feature_cols),
        'feature_groups': create_feature_groups(feature_cols)
    }
    
    # Save report
    save_json(report, output_path)
    
    # Print summary
    logger.info("=== DATA SUMMARY REPORT ===")
    logger.info(f"Total samples: {report['dataset_info']['total_samples']}")
    logger.info(f"Total features: {report['dataset_info']['total_features']}")
    logger.info(f"Memory usage: {report['dataset_info']['memory_usage_mb']:.2f} MB")
    
    # Missing data summary
    missing_summary = report['quality_metrics']['missing_data']
    high_missing = {k: v for k, v in missing_summary.items() if v > 50}
    logger.info(f"Features with >50% missing data: {len(high_missing)}")
    
    # Zero variance features
    logger.info(f"Zero variance features: {len(report['quality_metrics']['zero_variance'])}")
    
    logger.info(f"Full report saved to {output_path}")

def test_utils():
    """Test utility functions"""
    logger.info("Testing utility functions...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 10))
    Y = np.random.normal(0.5, 1, (100, 10))
    
    # Test MMD
    mmd_val = mmd_rbf(X, Y)
    logger.info(f"MMD value: {mmd_val:.4f}")
    
    # Test KS test
    ks_stats, p_values = ks_test_featurewise(X, Y)
    logger.info(f"KS statistics mean: {ks_stats.mean():.4f}")
    logger.info(f"Significant features (p<0.05): {(p_values < 0.05).sum()}")
    
    # Test feature grouping
    feature_names = [
        'Heart Rate', 'Creatinine', 'GCS - Eye Opening', 'Temperature',
        'Sodium', 'Blood Pressure', 'Hemoglobin', 'Unknown Feature'
    ]
    groups = create_feature_groups(feature_names)
    logger.info(f"Feature groups: {groups}")
    
    logger.info("Utility functions test completed successfully!")

if __name__ == "__main__":
    test_utils()
