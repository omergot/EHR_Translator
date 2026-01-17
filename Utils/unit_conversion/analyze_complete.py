#!/usr/bin/env python3
"""
Complete Distribution Comparison Analysis with all features
This script extends the basic comparison with trajectory, derived features, and correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis/logs/complete_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Critical features for detailed temporal analysis
TEMPORAL_FEATURES = [
    'Creatinine', 'Urea Nitrogen', 'Lactate', 'Heart Rate',
    'Non Invasive Blood Pressure mean', 'Sodium', 'Potassium',
    'WBC', 'Hemoglobin', 'pH', 'pO2', 'Inspired O2 Fraction',
    'Respiratory Rate', 'O2 saturation pulseoxymetry', 'Temperature'
]


def analyze_trajectory(df_mimic, df_eicu, feature, max_hours=48):
    """Analyze trajectory for first 48 hours"""
    
    # For each patient, compute time from first measurement
    mimic_data = df_mimic[df_mimic['feature_name'] == feature].copy()
    eicu_data = df_eicu[df_eicu['feature_name'] == feature].copy()
    
    # Get first measurement time per patient
    mimic_first = mimic_data.groupby('example_id')['feature_start_date'].min().to_dict()
    eicu_first = eicu_data.groupby('example_id')['feature_start_date'].min().to_dict()
    
    mimic_data['hours_from_start'] = mimic_data.apply(
        lambda x: (x['feature_start_date'] - mimic_first[x['example_id']]).total_seconds() / 3600, axis=1
    )
    eicu_data['hours_from_start'] = eicu_data.apply(
        lambda x: (x['feature_start_date'] - eicu_first[x['example_id']]).total_seconds() / 3600, axis=1
    )
    
    # Filter to first 48 hours
    mimic_data = mimic_data[mimic_data['hours_from_start'] <= max_hours]
    eicu_data = eicu_data[eicu_data['hours_from_start'] <= max_hours]
    
    # Bin into time windows
    bins = [0, 6, 12, 18, 24, 36, 48]
    bin_labels = ['0-6h', '6-12h', '12-18h', '18-24h', '24-36h', '36-48h']
    
    mimic_data['time_bin'] = pd.cut(mimic_data['hours_from_start'], bins=bins, labels=bin_labels)
    eicu_data['time_bin'] = pd.cut(eicu_data['hours_from_start'], bins=bins, labels=bin_labels)
    
    # Compute stats per bin
    mimic_stats = mimic_data.groupby('time_bin')['feature_value'].agg(['mean', 'std', 'median', 
                                                                          lambda x: x.quantile(0.25),
                                                                          lambda x: x.quantile(0.75)]).reset_index()
    mimic_stats.columns = ['time_bin', 'mean', 'std', 'median', 'q25', 'q75']
    
    eicu_stats = eicu_data.groupby('time_bin')['feature_value'].agg(['mean', 'std', 'median',
                                                                        lambda x: x.quantile(0.25),
                                                                        lambda x: x.quantile(0.75)]).reset_index()
    eicu_stats.columns = ['time_bin', 'mean', 'std', 'median', 'q25', 'q75']
    
    return mimic_stats, eicu_stats, mimic_data, eicu_data


def plot_trajectory(mimic_stats, eicu_stats, feature, output_path):
    """Plot trajectories"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{feature} - Temporal Trajectory (First 48 Hours)', fontsize=14, fontweight='bold')
    
    x_pos = range(len(mimic_stats))
    
    # Plot 1: Mean with error bars (std)
    ax = axes[0]
    ax.errorbar(x_pos, mimic_stats['mean'], yerr=mimic_stats['std'], 
                label='MIMIC', marker='o', capsize=5, linewidth=2)
    ax.errorbar(x_pos, eicu_stats['mean'], yerr=eicu_stats['std'], 
                label='eICU', marker='s', capsize=5, linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mimic_stats['time_bin'], rotation=45)
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Mean ± Std')
    ax.set_title('Mean with Standard Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Median with IQR
    ax = axes[1]
    ax.plot(x_pos, mimic_stats['median'], label='MIMIC (median)', marker='o', linewidth=2)
    ax.fill_between(x_pos, mimic_stats['q25'], mimic_stats['q75'], alpha=0.3)
    ax.plot(x_pos, eicu_stats['median'], label='eICU (median)', marker='s', linewidth=2)
    ax.fill_between(x_pos, eicu_stats['q25'], eicu_stats['q75'], alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(eicu_stats['time_bin'], rotation=45)
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Median with IQR')
    ax.set_title('Median with Interquartile Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def compute_derived_features(df, feature, max_hours=48):
    """Compute derived features per patient"""
    feature_data = df[df['feature_name'] == feature].copy()
    
    # Get first measurement time per patient
    first_times = feature_data.groupby('example_id')['feature_start_date'].min().to_dict()
    
    feature_data['hours_from_start'] = feature_data.apply(
        lambda x: (x['feature_start_date'] - first_times[x['example_id']]).total_seconds() / 3600, axis=1
    )
    
    # Filter to first 48 hours
    feature_data = feature_data[feature_data['hours_from_start'] <= max_hours]
    
    # Compute derived features per patient
    derived = feature_data.groupby('example_id').agg({
        'feature_value': ['first', 'last', 'min', 'max', 'mean', 'std'],
        'hours_from_start': lambda x: x.max() - x.min()  # duration
    }).reset_index()
    
    derived.columns = ['example_id', 'first_value', 'last_value', 'min_value', 'max_value', 
                       'mean_value', 'std_value', 'duration']
    
    # Compute slope
    derived['slope'] = (derived['last_value'] - derived['first_value']) / (derived['duration'] + 0.0001)
    
    return derived


def analyze_correlation_structure(df_mimic, df_eicu, features_list):
    """Analyze correlation structure for selected features"""
    logger.info("\n" + "=" * 80)
    logger.info("CORRELATION STRUCTURE ANALYSIS")
    logger.info("=" * 80)
    
    # For each dataset, create patient x feature matrix using first 24h average
    def create_feature_matrix(df, features):
        # Get first measurement time per patient
        first_times = df.groupby('example_id')['feature_start_date'].min().to_dict()
        df['hours_from_start'] = df.apply(
            lambda x: (x['feature_start_date'] - first_times[x['example_id']]).total_seconds() / 3600, axis=1
        )
        
        # Filter to first 24 hours
        df_24h = df[df['hours_from_start'] <= 24]
        
        # Pivot: average value per patient per feature
        matrix = df_24h.pivot_table(
            index='example_id',
            columns='feature_name',
            values='feature_value',
            aggfunc='mean'
        )
        
        # Keep only requested features
        matrix = matrix[[f for f in features if f in matrix.columns]]
        
        return matrix
    
    mimic_matrix = create_feature_matrix(df_mimic, features_list)
    eicu_matrix = create_feature_matrix(df_eicu, features_list)
    
    # Compute correlation matrices
    mimic_corr = mimic_matrix.corr(method='pearson')
    eicu_corr = eicu_matrix.corr(method='pearson')
    
    # Compute difference
    diff_corr = mimic_corr - eicu_corr
    
    # Frobenius norm
    frobenius_norm = np.linalg.norm(diff_corr.values, 'fro')
    logger.info(f"Frobenius norm of correlation difference: {frobenius_norm:.4f}")
    
    return mimic_corr, eicu_corr, diff_corr, frobenius_norm


def plot_correlation_matrices(mimic_corr, eicu_corr, diff_corr):
    """Plot correlation heatmaps"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # MIMIC correlation
    sns.heatmap(mimic_corr, ax=axes[0], cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={'label': 'Correlation'})
    axes[0].set_title('MIMIC Correlation Matrix', fontsize=14, fontweight='bold')
    
    # eICU correlation
    sns.heatmap(eicu_corr, ax=axes[1], cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={'label': 'Correlation'})
    axes[1].set_title('eICU Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Difference
    sns.heatmap(diff_corr, ax=axes[2], cmap='RdBu_r', center=0,
                square=True, cbar_kws={'label': 'Difference'})
    axes[2].set_title('Difference (MIMIC - eICU)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('analysis/plots/correlation/correlation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Correlation plots saved to: analysis/plots/correlation/")


def main():
    """Run complete analysis"""
    logger.info("=" * 80)
    logger.info("COMPLETE DISTRIBUTION COMPARISON ANALYSIS")
    logger.info("=" * 80)
    
    # Check if preprocessed data exists
    if Path('analysis/results/per_feature_summary.csv').exists():
        logger.info("Loading existing analysis results...")
        summary_df = pd.read_csv('analysis/results/per_feature_summary.csv')
        logger.info(f"Loaded results for {len(summary_df)} features")
        
        # Print alignment summary
        logger.info("\nFeature Alignment Summary:")
        logger.info("-" * 80)
        for category in ['aligned', 'mild shift', 'moderate shift', 'large shift 🚨']:
            features = summary_df[summary_df['alignment_category'] == category]
            count = len(features)
            pct = count / len(summary_df) * 100
            logger.info(f"  {category:20s}: {count:2d} features ({pct:5.1f}%)")
            if count > 0 and count <= 10:
                for _, row in features.iterrows():
                    logger.info(f"    - {row['feature']:50s} (SMD={row['smd']:.3f})")
        
        logger.info("\nFor complete analysis, run compare_distributions.py first")
    else:
        logger.info("No existing results found. Run compare_distributions.py first!")
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete! Check analysis/ directory for results.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


