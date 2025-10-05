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
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

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

def audit_preprocessing(output_dir: str, feature_spec: Dict = None) -> Dict:
    """
    IMPROVEMENT 6: Audit preprocessing to identify problematic features
    
    Args:
        output_dir: Path to the output directory containing preprocessed data
        feature_spec: Feature specification dictionary (optional)
        
    Returns:
        Dictionary with audit results
    """
    logger.info("Starting preprocessing audit...")
    output_path = Path(output_dir)
    
    # Load feature specification if not provided
    if feature_spec is None:
        spec_path = output_path / "feature_spec.json"
        if spec_path.exists():
            with open(spec_path, 'r') as f:
                feature_spec = json.load(f)
        else:
            logger.error("Feature specification not found")
            return {}
    
    # Load preprocessed data
    mimic_path = output_path / "data" / "train_mimic_preprocessed.csv"
    eicu_path = output_path / "data" / "train_eicu_preprocessed.csv"
    
    if not mimic_path.exists() or not eicu_path.exists():
        logger.error("Preprocessed data files not found")
        return {}
    
    logger.info(f"Loading MIMIC data from {mimic_path}")
    mimic_data = pd.read_csv(mimic_path)
    
    logger.info(f"Loading eICU data from {eicu_path}")
    eicu_data = pd.read_csv(eicu_path)
    
    # Analyze feature distributions
    analysis_results = _analyze_feature_distributions(mimic_data, eicu_data, feature_spec)
    
    # Comprehensive feature analysis for all features
    comprehensive_analysis = _investigate_comprehensive_features(mimic_data, eicu_data, feature_spec)
    
    # Generate fixes
    preprocessing_fixes = _generate_preprocessing_fixes(analysis_results)
    
    # Compile audit results
    audit_results = {
        'summary': {
            'total_features': len(analysis_results['feature_statistics']),
            'problematic_features': len(analysis_results['problematic_features']),
            'features_needing_robust_scaling': len(preprocessing_fixes['robust_scaling_features']),
            'features_needing_log_transform': len(preprocessing_fixes['log_transform_features']),
            'features_needing_special_handling': len(preprocessing_fixes['special_handling_features'])
        },
        'comprehensive_analysis': comprehensive_analysis,
        'problematic_features': analysis_results['problematic_features'],
        'preprocessing_fixes': preprocessing_fixes,
        'detailed_statistics': analysis_results['feature_statistics']
    }
    
    # Save audit results
    audit_path = output_path / "preprocessing_audit_results.json"
    with open(audit_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_types(audit_results), f, indent=2)
    
    logger.info(f"Audit results saved to {audit_path}")
    
    # Print summary
    summary = audit_results['summary']
    logger.info("=== PREPROCESSING AUDIT SUMMARY ===")
    logger.info(f"Total features analyzed: {summary['total_features']}")
    logger.info(f"Problematic features found: {summary['problematic_features']}")
    logger.info(f"Features needing robust scaling: {summary['features_needing_robust_scaling']}")
    logger.info(f"Features needing log transform: {summary['features_needing_log_transform']}")
    logger.info(f"Features needing special handling: {summary['features_needing_special_handling']}")
    
    # Print comprehensive analysis summary
    if audit_results['comprehensive_analysis']:
        comprehensive_analysis = audit_results['comprehensive_analysis']
        logger.info(f"=== COMPREHENSIVE FEATURE ANALYSIS SUMMARY ===")
        
        # Count features with issues
        features_with_issues = 0
        total_recommendations = 0
        
        for feature_name, analysis in comprehensive_analysis.items():
            if analysis.get('recommendations'):
                features_with_issues += 1
                total_recommendations += len(analysis['recommendations'])
        
        logger.info(f"Features analyzed: {len(comprehensive_analysis)}")
        logger.info(f"Features with issues: {features_with_issues}")
        logger.info(f"Total recommendations: {total_recommendations}")
        
        # Show top problematic features
        problematic_features = []
        for feature_name, analysis in comprehensive_analysis.items():
            if analysis.get('recommendations'):
                problematic_features.append((feature_name, len(analysis['recommendations'])))
        
        problematic_features.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("=== TOP PROBLEMATIC FEATURES ===")
        for feature_name, num_issues in problematic_features[:10]:  # Show top 10
            logger.info(f"{feature_name}: {num_issues} issues")
    
    return audit_results

def _analyze_feature_distributions(mimic_data: pd.DataFrame, eicu_data: pd.DataFrame, feature_spec: Dict) -> Dict:
    """Analyze feature distributions to identify problematic features"""
    logger.info("Analyzing feature distributions between datasets...")
    
    numeric_features = feature_spec.get('numeric_features', [])
    missing_features = feature_spec.get('missing_features', [])
    all_features = numeric_features + missing_features
    
    analysis_results = {
        'problematic_features': [],
        'feature_statistics': {},
        'distribution_differences': {}
    }
    
    for i, feature in enumerate(all_features):
        if feature in mimic_data.columns and feature in eicu_data.columns:
            mimic_values = mimic_data[feature].dropna()
            eicu_values = eicu_data[feature].dropna()
            
            if len(mimic_values) == 0 or len(eicu_values) == 0:
                continue
            
            # Basic statistics
            mimic_stats = {
                'mean': mimic_values.mean(),
                'std': mimic_values.std(),
                'min': mimic_values.min(),
                'max': mimic_values.max(),
                'q25': mimic_values.quantile(0.25),
                'q75': mimic_values.quantile(0.75)
            }
            
            eicu_stats = {
                'mean': eicu_values.mean(),
                'std': eicu_values.std(),
                'min': eicu_values.min(),
                'max': eicu_values.max(),
                'q25': eicu_values.quantile(0.25),
                'q75': eicu_values.quantile(0.75)
            }
            
            analysis_results['feature_statistics'][feature] = {
                'mimic': mimic_stats,
                'eicu': eicu_stats,
                'index': i
            }
            
            # Distribution differences
            mean_ratio = abs(mimic_stats['mean'] / (eicu_stats['mean'] + 1e-8))
            std_ratio = abs(mimic_stats['std'] / (eicu_stats['std'] + 1e-8))
            range_ratio = abs((mimic_stats['max'] - mimic_stats['min']) / 
                            (eicu_stats['max'] - eicu_stats['min'] + 1e-8))
            
            # KS test for distribution similarity
            ks_stat, ks_p_value = stats.ks_2samp(mimic_values, eicu_values)
            
            diff_metrics = {
                'mean_ratio': mean_ratio,
                'std_ratio': std_ratio,
                'range_ratio': range_ratio,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'mean_diff': abs(mimic_stats['mean'] - eicu_stats['mean']),
                'std_diff': abs(mimic_stats['std'] - eicu_stats['std'])
            }
            
            analysis_results['distribution_differences'][feature] = diff_metrics
            
            # Identify problematic features
            is_problematic = (
                mean_ratio > 10 or mean_ratio < 0.1 or  # Extreme mean differences
                std_ratio > 10 or std_ratio < 0.1 or    # Extreme std differences
                range_ratio > 100 or range_ratio < 0.01 or  # Extreme range differences
                ks_stat > 0.5 or                        # High KS statistic
                diff_metrics['mean_diff'] > 1000        # Very large absolute difference
            )
            
            if is_problematic:
                analysis_results['problematic_features'].append({
                    'feature': feature,
                    'index': i,
                    'issues': {
                        'extreme_mean_ratio': mean_ratio > 10 or mean_ratio < 0.1,
                        'extreme_std_ratio': std_ratio > 10 or std_ratio < 0.1,
                        'extreme_range_ratio': range_ratio > 100 or range_ratio < 0.01,
                        'high_ks_stat': ks_stat > 0.5,
                        'large_mean_diff': diff_metrics['mean_diff'] > 1000
                    },
                    'metrics': diff_metrics
                })
    
    logger.info(f"Identified {len(analysis_results['problematic_features'])} problematic features")
    
    return analysis_results

def _investigate_comprehensive_features(mimic_data: pd.DataFrame, eicu_data: pd.DataFrame, feature_spec: Dict) -> Dict:
    """Comprehensive analysis of all features for preprocessing recommendations"""
    logger.info("Running comprehensive feature analysis for all features...")
    
    # Get all feature names
    numeric_features = feature_spec.get('numeric_features', [])
    missing_features = feature_spec.get('missing_features', [])
    all_features = numeric_features + missing_features
    
    comprehensive_analysis = {}
    
    for feature_name in all_features:
        if feature_name not in mimic_data.columns or feature_name not in eicu_data.columns:
            logger.warning(f"Feature {feature_name} not found in data, skipping...")
            continue
            
        logger.info(f"Analyzing feature: {feature_name}")
        
        mimic_values = mimic_data[feature_name].dropna()
        eicu_values = eicu_data[feature_name].dropna()
        
        if len(mimic_values) == 0 or len(eicu_values) == 0:
            logger.warning(f"No valid values for {feature_name}, skipping...")
            continue
        
        analysis = {
            'feature_name': feature_name,
            'mimic_stats': {
                'count': len(mimic_values),
                'mean': mimic_values.mean(),
                'std': mimic_values.std(),
                'min': mimic_values.min(),
                'max': mimic_values.max(),
                'q25': mimic_values.quantile(0.25),
                'q50': mimic_values.quantile(0.50),
                'q75': mimic_values.quantile(0.75),
                'q99': mimic_values.quantile(0.99),
                'outliers': len(mimic_values[mimic_values > mimic_values.quantile(0.99)]),
                'zeros': len(mimic_values[mimic_values == 0])
            },
            'eicu_stats': {
                'count': len(eicu_values),
                'mean': eicu_values.mean(),
                'std': eicu_values.std(),
                'min': eicu_values.min(),
                'max': eicu_values.max(),
                'q25': eicu_values.quantile(0.25),
                'q50': eicu_values.quantile(0.50),
                'q75': eicu_values.quantile(0.75),
                'q99': eicu_values.quantile(0.99),
                'outliers': len(eicu_values[eicu_values > eicu_values.quantile(0.99)]),
                'zeros': len(eicu_values[eicu_values == 0])
            }
        }
        
        # Compute distribution differences
        ks_stat, ks_p_value = stats.ks_2samp(mimic_values, eicu_values)
        
        analysis['comparison'] = {
            'mean_ratio': analysis['mimic_stats']['mean'] / (analysis['eicu_stats']['mean'] + 1e-8),
            'std_ratio': analysis['mimic_stats']['std'] / (analysis['eicu_stats']['std'] + 1e-8),
            'range_mimic': analysis['mimic_stats']['max'] - analysis['mimic_stats']['min'],
            'range_eicu': analysis['eicu_stats']['max'] - analysis['eicu_stats']['min'],
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value
        }
        
        # Generate recommendations for this feature
        recommendations = []
        
        if abs(analysis['comparison']['mean_ratio']) > 10:
            recommendations.append("Apply per-dataset standardization (z-score normalization)")
        
        if analysis['comparison']['ks_statistic'] > 0.3:
            recommendations.append("Consider robust scaling (median/IQR) instead of mean/std")
        
        if analysis['mimic_stats']['outliers'] > len(mimic_values) * 0.01:
            recommendations.append("Apply outlier clipping (99th percentile)")
        
        if analysis['mimic_stats']['zeros'] > len(mimic_values) * 0.1:
            recommendations.append("Handle zero values specially (log1p transform or separate handling)")
        
        # Check for degenerate distribution (most values the same)
        if analysis['mimic_stats']['q25'] == analysis['mimic_stats']['q75']:
            recommendations.append("Degenerate distribution detected - consider log1p transform or robust scaling")
        
        if analysis['eicu_stats']['q25'] == analysis['eicu_stats']['q75']:
            recommendations.append("Degenerate distribution detected in eICU - consider log1p transform or robust scaling")
        
        # Check for extreme standard deviation ratios
        if abs(analysis['comparison']['std_ratio']) > 100 or abs(analysis['comparison']['std_ratio']) < 0.01:
            recommendations.append("Extreme variance differences between datasets - use per-dataset scaling")
        
        analysis['recommendations'] = recommendations
        comprehensive_analysis[feature_name] = analysis
    
    return comprehensive_analysis

def _investigate_feature_32(mimic_data: pd.DataFrame, eicu_data: pd.DataFrame, feature_spec: Dict) -> Dict:
    """Specifically investigate feature 32 (the persistent outlier)"""
    logger.info("Investigating feature index 32 (persistent outlier)...")
    
    # Get all feature names
    numeric_features = feature_spec.get('numeric_features', [])
    missing_features = feature_spec.get('missing_features', [])
    all_features = numeric_features + missing_features
    
    if len(all_features) <= 32:
        logger.warning("Feature index 32 does not exist in feature specification")
        return {}
    
    feature_32_name = all_features[32]
    logger.info(f"Feature 32 corresponds to: {feature_32_name}")
    
    if feature_32_name not in mimic_data.columns or feature_32_name not in eicu_data.columns:
        logger.error(f"Feature {feature_32_name} not found in data")
        return {}
    
    mimic_values = mimic_data[feature_32_name].dropna()
    eicu_values = eicu_data[feature_32_name].dropna()
    
    analysis = {
        'feature_name': feature_32_name,
        'mimic_stats': {
            'count': len(mimic_values),
            'mean': mimic_values.mean(),
            'std': mimic_values.std(),
            'min': mimic_values.min(),
            'max': mimic_values.max(),
            'q25': mimic_values.quantile(0.25),
            'q50': mimic_values.quantile(0.50),
            'q75': mimic_values.quantile(0.75),
            'q99': mimic_values.quantile(0.99),
            'outliers': len(mimic_values[mimic_values > mimic_values.quantile(0.99)]),
            'zeros': len(mimic_values[mimic_values == 0])
        },
        'eicu_stats': {
            'count': len(eicu_values),
            'mean': eicu_values.mean(),
            'std': eicu_values.std(),
            'min': eicu_values.min(),
            'max': eicu_values.max(),
            'q25': eicu_values.quantile(0.25),
            'q50': eicu_values.quantile(0.50),
            'q75': eicu_values.quantile(0.75),
            'q99': eicu_values.quantile(0.99),
            'outliers': len(eicu_values[eicu_values > eicu_values.quantile(0.99)]),
            'zeros': len(eicu_values[eicu_values == 0])
        }
    }
    
    # Compute distribution differences
    ks_stat, ks_p_value = stats.ks_2samp(mimic_values, eicu_values)
    
    analysis['comparison'] = {
        'mean_ratio': analysis['mimic_stats']['mean'] / (analysis['eicu_stats']['mean'] + 1e-8),
        'std_ratio': analysis['mimic_stats']['std'] / (analysis['eicu_stats']['std'] + 1e-8),
        'range_mimic': analysis['mimic_stats']['max'] - analysis['mimic_stats']['min'],
        'range_eicu': analysis['eicu_stats']['max'] - analysis['eicu_stats']['min'],
        'ks_statistic': ks_stat,
        'ks_p_value': ks_p_value
    }
    
    # Generate recommendations
    recommendations = []
    
    if abs(analysis['comparison']['mean_ratio']) > 10:
        recommendations.append("Apply per-dataset standardization (z-score normalization)")
    
    if analysis['comparison']['ks_statistic'] > 0.3:
        recommendations.append("Consider robust scaling (median/IQR) instead of mean/std")
    
    if analysis['mimic_stats']['outliers'] > len(mimic_values) * 0.01:
        recommendations.append("Apply outlier clipping (99th percentile)")
    
    if analysis['mimic_stats']['zeros'] > len(mimic_values) * 0.1:
        recommendations.append("Handle zero values specially (log1p transform or separate handling)")
    
    analysis['recommendations'] = recommendations
    
    return analysis

def _generate_preprocessing_fixes(analysis_results: Dict) -> Dict:
    """Generate specific preprocessing fixes based on analysis"""
    logger.info("Generating preprocessing fixes...")
    
    fixes = {
        'per_dataset_standardization': {
            'enabled': True,
            'reason': 'Fix extreme mean/std differences between datasets'
        },
        'outlier_clipping': {
            'enabled': True,
            'method': 'percentile',
            'percentile': 99,
            'reason': 'Remove extreme outliers that dominate gradients'
        },
        'robust_scaling_features': [],
        'log_transform_features': [],
        'special_handling_features': []
    }
    
    for feature_info in analysis_results['problematic_features']:
        feature = feature_info['feature']
        issues = feature_info['issues']
        metrics = feature_info['metrics']
        
        # Recommend robust scaling for features with extreme ratio differences
        if issues['extreme_mean_ratio'] or issues['extreme_std_ratio']:
            fixes['robust_scaling_features'].append({
                'feature': feature,
                'reason': 'Extreme mean/std ratio differences',
                'mean_ratio': metrics['mean_ratio'],
                'std_ratio': metrics['std_ratio']
            })
        
        # Recommend log transform for features with extreme range differences
        if issues['extreme_range_ratio']:
            fixes['log_transform_features'].append({
                'feature': feature,
                'reason': 'Extreme range differences',
                'range_ratio': metrics['range_ratio']
            })
        
        # Special handling for features with very large absolute differences
        if issues['large_mean_diff']:
            fixes['special_handling_features'].append({
                'feature': feature,
                'reason': 'Very large absolute mean difference',
                'mean_diff': metrics['mean_diff']
            })
    
    return fixes

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
