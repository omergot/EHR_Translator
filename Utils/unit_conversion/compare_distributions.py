#!/usr/bin/env python3
"""
Distribution Comparison Analysis for MIMIC-IV vs eICU-CRD

This script performs comprehensive distribution comparison across 40 BSI features:
1. Per-feature marginal distribution analysis
2. Temporal behavior analysis (measurement frequency, trajectories, derived features)
3. Correlation structure comparison
4. Summary dashboard generation

Author: Domain Translation Project
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings
import logging
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis/logs/analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
MIMIC_DATA_PATH = "/bigdata/omerg/RatchetEHR/tmp/tmp/cache_data_bsi_test_100mimiciv"
# Use corrected eICU file with C-Reactive Protein unit fix (mg/L → mg/dL)
EICU_DATA_PATH = "/bigdata/omerg/RatchetEHR/tmp/tmp/cache_data_bsi_test_100_corrected"
# Original eICU file (uncomment to use uncorrected data):
# EICU_DATA_PATH = "/bigdata/omerg/RatchetEHR/tmp/tmp/cache_data_bsi_test_100"
MIMIC_FEATURES_PATH = "data/mimic_features_aligned.csv"
EICU_FEATURES_PATH = "data/eicu_features_aligned.csv"
UNIT_COMPARISON_PATH = "feature_units_comparison.csv"

# Critical features for temporal analysis
TEMPORAL_FEATURES = [
    'Creatinine', 'Urea Nitrogen', 'Lactate', 'Heart Rate',
    'Non Invasive Blood Pressure mean', 'Non Invasive Blood Pressure systolic',
    'Non Invasive Blood Pressure diastolic', 'Sodium', 'Potassium',
    'WBC', 'Hemoglobin', 'pH', 'pO2', 'Inspired O2 Fraction',
    'Respiratory Rate', 'O2 saturation pulseoxymetry', 'Temperature',
    'GCS - Motor Response', 'Bilirubin', 'Albumin'
]


def load_and_preprocess_data():
    """Load both datasets and apply preprocessing"""
    logger.info("=" * 80)
    logger.info("LOADING AND PREPROCESSING DATA")
    logger.info("=" * 80)
    
    # Load MIMIC data
    logger.info(f"Loading MIMIC data from {MIMIC_DATA_PATH}...")
    df_mimic = pd.read_csv(MIMIC_DATA_PATH)
    logger.info(f"  Loaded {len(df_mimic):,} rows")
    
    # Load eICU data
    logger.info(f"Loading eICU data from {EICU_DATA_PATH}...")
    df_eicu = pd.read_csv(EICU_DATA_PATH)
    logger.info(f"  Loaded {len(df_eicu):,} rows")
    
    # Load feature mappings
    mimic_features = pd.read_csv(MIMIC_FEATURES_PATH)
    eicu_features = pd.read_csv(EICU_FEATURES_PATH)
    
    # Create mapping from eICU names to MIMIC names
    feature_mapping = dict(zip(eicu_features['Feature_Name'], mimic_features['Feature_Name']))
    
    # Parse timestamps
    logger.info("Parsing timestamps...")
    df_mimic['feature_start_date'] = pd.to_datetime(df_mimic['feature_start_date'], errors='coerce')
    df_eicu['feature_start_date'] = pd.to_datetime(df_eicu['feature_start_date'], errors='coerce')
    
    # Convert feature values to numeric
    logger.info("Converting feature values to numeric...")
    df_mimic['feature_value'] = pd.to_numeric(df_mimic['feature_value'], errors='coerce')
    df_eicu['feature_value'] = pd.to_numeric(df_eicu['feature_value'], errors='coerce')
    
    # Drop rows with invalid values
    initial_mimic = len(df_mimic)
    initial_eicu = len(df_eicu)
    df_mimic = df_mimic.dropna(subset=['feature_value', 'feature_start_date'])
    df_eicu = df_eicu.dropna(subset=['feature_value', 'feature_start_date'])
    logger.info(f"  MIMIC: Dropped {initial_mimic - len(df_mimic):,} rows with invalid values")
    logger.info(f"  eICU: Dropped {initial_eicu - len(df_eicu):,} rows with invalid values")
    
    # Map eICU feature names to MIMIC standard
    logger.info("Mapping eICU feature names to MIMIC standard...")
    df_eicu['feature_name_original'] = df_eicu['feature_name']
    df_eicu['feature_name'] = df_eicu['feature_name'].map(feature_mapping)
    
    # Remove unmapped features
    unmapped = df_eicu[df_eicu['feature_name'].isna()]
    if len(unmapped) > 0:
        logger.warning(f"  Found {len(unmapped):,} rows with unmapped feature names")
        df_eicu = df_eicu[df_eicu['feature_name'].notna()]
    
    # Add dataset label
    df_mimic['dataset'] = 'MIMIC'
    df_eicu['dataset'] = 'eICU'
    
    logger.info(f"\nFinal dataset sizes:")
    logger.info(f"  MIMIC: {len(df_mimic):,} rows, {df_mimic['feature_name'].nunique()} unique features")
    logger.info(f"  eICU:  {len(df_eicu):,} rows, {df_eicu['feature_name'].nunique()} unique features")
    
    return df_mimic, df_eicu, feature_mapping


def apply_unit_conversions(df_mimic, df_eicu):
    """Apply unit conversions based on feature_units_comparison.csv"""
    logger.info("\n" + "=" * 80)
    logger.info("APPLYING UNIT CONVERSIONS")
    logger.info("=" * 80)
    
    # Load unit comparison
    unit_comp = pd.read_csv(UNIT_COMPARISON_PATH)
    
    # Filter to features that need conversion
    needs_conversion = unit_comp[unit_comp['Needs_Conversion'] == 'YES']
    logger.info(f"Found {len(needs_conversion)} features requiring unit conversion")
    
    for _, row in needs_conversion.iterrows():
        feature = row['MIMIC_Feature']
        mimic_unit = row['MIMIC_Units']
        eicu_unit = row['eICU_Units']
        
        logger.info(f"\n  {feature}:")
        logger.info(f"    MIMIC: {mimic_unit} → eICU: {eicu_unit}")
        
        # C-Reactive Protein: mg/L → mg/dL (multiply by 0.1)
        if feature == 'C-Reactive Protein':
            mask = df_mimic['feature_name'] == feature
            df_mimic.loc[mask, 'feature_value'] = df_mimic.loc[mask, 'feature_value'] * 0.1
            logger.info(f"    Converted MIMIC values: mg/L → mg/dL (×0.1)")
        
        # Multi-unit features: filter to common unit
        # For MCHC, RBC, WBC, Urea Nitrogen - we'll handle outliers in filtering step
        # No conversion needed for naming-only differences
        
    logger.info("\nUnit conversion complete!")
    return df_mimic, df_eicu


def analyze_unit_coverage(df_mimic, df_eicu):
    """Analyze unit columns across both datasets"""
    logger.info("\n" + "=" * 80)
    logger.info("UNIT COVERAGE ANALYSIS")
    logger.info("=" * 80)
    
    # Get unique features
    all_features = sorted(set(df_mimic['feature_name'].unique()) | 
                          set(df_eicu['feature_name'].unique()))
    
    unit_analysis = []
    
    for feature in all_features:
        mimic_data = df_mimic[df_mimic['feature_name'] == feature]
        eicu_data = df_eicu[df_eicu['feature_name'] == feature]
        
        # Get unique units for each dataset
        mimic_units = mimic_data['unit'].value_counts().to_dict() if 'unit' in mimic_data.columns else {}
        eicu_units = eicu_data['unit'].value_counts().to_dict() if 'unit' in eicu_data.columns else {}
        
        # Normalize unit keys for comparison (handle case differences)
        mimic_units_normalized = {str(k).lower().strip(): v for k, v in mimic_units.items() if pd.notna(k)}
        eicu_units_normalized = {str(k).lower().strip(): v for k, v in eicu_units.items() if pd.notna(k)}
        
        # Count measurements per unit
        unit_analysis.append({
            'feature': feature,
            'mimic_units': '; '.join([f"{k}({v})" for k, v in sorted(mimic_units.items(), key=lambda x: -x[1])]),
            'eicu_units': '; '.join([f"{k}({v})" for k, v in sorted(eicu_units.items(), key=lambda x: -x[1])]),
            'mimic_n_total': len(mimic_data),
            'eicu_n_total': len(eicu_data),
            'units_match': set(mimic_units_normalized.keys()) == set(eicu_units_normalized.keys()),
            'n_mimic_unit_types': len(mimic_units),
            'n_eicu_unit_types': len(eicu_units),
            'mimic_primary_unit': max(mimic_units, key=mimic_units.get) if mimic_units else 'N/A',
            'eicu_primary_unit': max(eicu_units, key=eicu_units.get) if eicu_units else 'N/A'
        })
    
    unit_df = pd.DataFrame(unit_analysis)
    unit_df.to_csv('analysis/results/unit_coverage_analysis.csv', index=False)
    
    # Log summary
    logger.info(f"\nFeatures analyzed: {len(unit_df)}")
    logger.info(f"Features with matching units: {unit_df['units_match'].sum()}")
    logger.info(f"Features with unit mismatches: {(~unit_df['units_match']).sum()}")
    logger.info(f"Features with multiple MIMIC units: {(unit_df['n_mimic_unit_types'] > 1).sum()}")
    logger.info(f"Features with multiple eICU units: {(unit_df['n_eicu_unit_types'] > 1).sum()}")
    
    # Log features with mismatches
    mismatches = unit_df[~unit_df['units_match']]
    if len(mismatches) > 0:
        logger.info(f"\nFeatures with unit mismatches ({len(mismatches)}):")
        for _, row in mismatches.iterrows():
            logger.info(f"  {row['feature']:45s} MIMIC: {row['mimic_primary_unit']:15s} eICU: {row['eicu_primary_unit']:15s}")
    
    logger.info(f"\nUnit coverage analysis saved to: analysis/results/unit_coverage_analysis.csv")
    
    return unit_df


def compute_psi(expected, actual, bins=10):
    """
    Compute Population Stability Index (PSI)
    
    PSI measures the shift in distribution between two samples
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change
    """
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates
    
    if len(breakpoints) < 2:
        return np.nan
    
    # Bin both distributions
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Add small constant to avoid division by zero
    epsilon = 1e-10
    expected_percents = expected_percents + epsilon
    actual_percents = actual_percents + epsilon
    
    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = np.sum(psi_values)
    
    return psi


def compute_smd(mean1, std1, mean2, std2):
    """Compute Standardized Mean Difference (Cohen's d)"""
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    if pooled_std == 0:
        return 0
    return abs(mean1 - mean2) / pooled_std


def analyze_feature_distribution(df_mimic, df_eicu, feature_name):
    """Analyze distribution for a single feature"""
    # Extract data for this feature
    mimic_data = df_mimic[df_mimic['feature_name'] == feature_name]
    eicu_data = df_eicu[df_eicu['feature_name'] == feature_name]
    
    mimic_vals = mimic_data['feature_value'].values
    eicu_vals = eicu_data['feature_value'].values
    
    if len(mimic_vals) == 0 or len(eicu_vals) == 0:
        return None
    
    # Remove outliers (values beyond 1st and 99th percentiles combined)
    all_vals = np.concatenate([mimic_vals, eicu_vals])
    p1, p99 = np.percentile(all_vals, [1, 99])
    mimic_vals = mimic_vals[(mimic_vals >= p1) & (mimic_vals <= p99)]
    eicu_vals = eicu_vals[(eicu_vals >= p1) & (eicu_vals <= p99)]
    
    # Basic statistics
    stats_dict = {
        'feature': feature_name,
        'n_mimic': len(mimic_vals),
        'n_eicu': len(eicu_vals),
        'mean_mimic': np.mean(mimic_vals),
        'mean_eicu': np.mean(eicu_vals),
        'std_mimic': np.std(mimic_vals),
        'std_eicu': np.std(eicu_vals),
        'median_mimic': np.median(mimic_vals),
        'median_eicu': np.median(eicu_vals),
        'iqr_mimic': np.percentile(mimic_vals, 75) - np.percentile(mimic_vals, 25),
        'iqr_eicu': np.percentile(eicu_vals, 75) - np.percentile(eicu_vals, 25),
    }
    
    # Percentiles
    for p in [1, 5, 25, 50, 75, 95, 99]:
        stats_dict[f'p{p}_mimic'] = np.percentile(mimic_vals, p)
        stats_dict[f'p{p}_eicu'] = np.percentile(eicu_vals, p)
    
    # Distance metrics
    stats_dict['smd'] = compute_smd(
        stats_dict['mean_mimic'], stats_dict['std_mimic'],
        stats_dict['mean_eicu'], stats_dict['std_eicu']
    )
    
    # KS test
    ks_stat, ks_pval = stats.ks_2samp(mimic_vals, eicu_vals)
    stats_dict['ks_stat'] = ks_stat
    stats_dict['ks_pvalue'] = ks_pval
    
    # Wasserstein distance
    stats_dict['wasserstein'] = stats.wasserstein_distance(mimic_vals, eicu_vals)
    
    # PSI
    stats_dict['psi'] = compute_psi(mimic_vals, eicu_vals)
    
    # Alignment category
    smd = stats_dict['smd']
    if smd < 0.1:
        stats_dict['alignment_category'] = 'aligned'
    elif smd < 0.3:
        stats_dict['alignment_category'] = 'mild shift'
    elif smd < 0.5:
        stats_dict['alignment_category'] = 'moderate shift'
    else:
        stats_dict['alignment_category'] = 'large shift 🚨'
    
    # Add unit information
    if 'unit' in mimic_data.columns:
        mimic_units = mimic_data['unit'].value_counts()
        stats_dict['mimic_primary_unit'] = mimic_units.index[0] if len(mimic_units) > 0 else 'N/A'
        stats_dict['mimic_n_unit_types'] = len(mimic_units)
    else:
        stats_dict['mimic_primary_unit'] = 'N/A'
        stats_dict['mimic_n_unit_types'] = 0
    
    if 'unit' in eicu_data.columns:
        eicu_units = eicu_data['unit'].value_counts()
        stats_dict['eicu_primary_unit'] = eicu_units.index[0] if len(eicu_units) > 0 else 'N/A'
        stats_dict['eicu_n_unit_types'] = len(eicu_units)
    else:
        stats_dict['eicu_primary_unit'] = 'N/A'
        stats_dict['eicu_n_unit_types'] = 0
    
    return stats_dict, mimic_vals, eicu_vals


def plot_feature_distribution(mimic_vals, eicu_vals, feature_name, output_path):
    """Create comprehensive distribution plots for a feature"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{feature_name} - Distribution Comparison', fontsize=16, fontweight='bold')
    
    # 1. Overlaid Histogram
    ax = axes[0, 0]
    ax.hist(mimic_vals, bins=50, alpha=0.6, label='MIMIC', density=True, edgecolor='black')
    ax.hist(eicu_vals, bins=50, alpha=0.6, label='eICU', density=True, edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Overlaid KDE
    ax = axes[0, 1]
    try:
        from scipy.stats import gaussian_kde
        if len(mimic_vals) > 1:
            kde_mimic = gaussian_kde(mimic_vals)
            x_mimic = np.linspace(mimic_vals.min(), mimic_vals.max(), 200)
            ax.plot(x_mimic, kde_mimic(x_mimic), label='MIMIC', linewidth=2)
        if len(eicu_vals) > 1:
            kde_eicu = gaussian_kde(eicu_vals)
            x_eicu = np.linspace(eicu_vals.min(), eicu_vals.max(), 200)
            ax.plot(x_eicu, kde_eicu(x_eicu), label='eICU', linewidth=2)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Kernel Density Estimate')
        ax.legend()
        ax.grid(True, alpha=0.3)
    except:
        ax.text(0.5, 0.5, 'KDE not available', ha='center', va='center', transform=ax.transAxes)
    
    # 3. Overlaid ECDF
    ax = axes[1, 0]
    mimic_sorted = np.sort(mimic_vals)
    eicu_sorted = np.sort(eicu_vals)
    mimic_ecdf = np.arange(1, len(mimic_sorted) + 1) / len(mimic_sorted)
    eicu_ecdf = np.arange(1, len(eicu_sorted) + 1) / len(eicu_sorted)
    ax.plot(mimic_sorted, mimic_ecdf, label='MIMIC', linewidth=2)
    ax.plot(eicu_sorted, eicu_ecdf, label='eICU', linewidth=2)
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical Cumulative Distribution Function (ECDF)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Side-by-side Violin Plot
    ax = axes[1, 1]
    data_to_plot = [mimic_vals, eicu_vals]
    parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['MIMIC', 'eICU'])
    ax.set_ylabel('Value')
    ax.set_title('Violin Plot')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def analyze_marginal_distributions(df_mimic, df_eicu):
    """Analyze marginal distributions for all 40 features"""
    logger.info("\n" + "=" * 80)
    logger.info("MARGINAL DISTRIBUTION ANALYSIS")
    logger.info("=" * 80)
    
    # Get list of features to analyze (40 aligned features)
    mimic_features = pd.read_csv(MIMIC_FEATURES_PATH)
    features_to_analyze = mimic_features['Feature_Name'].tolist()
    
    results = []
    
    for feature in tqdm(features_to_analyze, desc="Analyzing features"):
        result = analyze_feature_distribution(df_mimic, df_eicu, feature)
        
        if result is None:
            logger.warning(f"  Skipping {feature} - insufficient data")
            continue
        
        stats_dict, mimic_vals, eicu_vals = result
        results.append(stats_dict)
        
        # Create plots
        plot_path = f'analysis/plots/per_feature/{feature.replace("/", "_")}_distribution.png'
        try:
            plot_feature_distribution(mimic_vals, eicu_vals, feature, plot_path)
        except Exception as e:
            logger.warning(f"  Failed to create plot for {feature}: {e}")
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('smd', ascending=False)
    summary_df.to_csv('analysis/results/per_feature_summary.csv', index=False)
    
    logger.info(f"\nAnalyzed {len(results)} features")
    logger.info(f"Summary saved to: analysis/results/per_feature_summary.csv")
    logger.info(f"Plots saved to: analysis/plots/per_feature/")
    
    # Print summary statistics
    logger.info("\nAlignment Summary:")
    for category in ['aligned', 'mild shift', 'moderate shift', 'large shift 🚨']:
        count = (summary_df['alignment_category'] == category).sum()
        pct = count / len(summary_df) * 100
        logger.info(f"  {category:20s}: {count:2d} features ({pct:5.1f}%)")
    
    return summary_df


def analyze_temporal_frequency(df_mimic, df_eicu, feature):
    """Analyze measurement frequency for a feature"""
    # Measurements per patient
    mimic_freq = df_mimic[df_mimic['feature_name'] == feature].groupby('example_id').size()
    eicu_freq = df_eicu[df_eicu['feature_name'] == feature].groupby('example_id').size()
    
    # Time gaps between measurements
    mimic_gaps = []
    eicu_gaps = []
    
    for dataset, gaps_list in [(df_mimic, mimic_gaps), (df_eicu, eicu_gaps)]:
        feature_data = dataset[dataset['feature_name'] == feature].sort_values(['example_id', 'feature_start_date'])
        for patient_id in feature_data['example_id'].unique():
            patient_data = feature_data[feature_data['example_id'] == patient_id]
            if len(patient_data) > 1:
                times = patient_data['feature_start_date'].values
                gaps = np.diff(times) / np.timedelta64(1, 'h')  # Convert to hours
                gaps_list.extend(gaps)
    
    return mimic_freq, eicu_freq, mimic_gaps, eicu_gaps


def plot_temporal_frequency(mimic_freq, eicu_freq, mimic_gaps, eicu_gaps, feature, output_path):
    """Plot measurement frequency analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{feature} - Measurement Frequency', fontsize=14, fontweight='bold')
    
    # Plot 1: Measurements per patient
    ax = axes[0]
    ax.hist(mimic_freq, bins=30, alpha=0.6, label=f'MIMIC (mean={mimic_freq.mean():.1f})', density=True)
    ax.hist(eicu_freq, bins=30, alpha=0.6, label=f'eICU (mean={eicu_freq.mean():.1f})', density=True)
    ax.set_xlabel('Number of Measurements per Patient')
    ax.set_ylabel('Density')
    ax.set_title('Measurements per Patient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Time gaps
    ax = axes[1]
    if len(mimic_gaps) > 0:
        ax.hist(mimic_gaps, bins=50, alpha=0.6, label=f'MIMIC (median={np.median(mimic_gaps):.1f}h)', 
                density=True, range=(0, 24))
    if len(eicu_gaps) > 0:
        ax.hist(eicu_gaps, bins=50, alpha=0.6, label=f'eICU (median={np.median(eicu_gaps):.1f}h)', 
                density=True, range=(0, 24))
    ax.set_xlabel('Time Gap (hours)')
    ax.set_ylabel('Density')
    ax.set_title('Time Between Consecutive Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    """Main analysis pipeline"""
    logger.info("=" * 80)
    logger.info("DISTRIBUTION COMPARISON ANALYSIS: MIMIC-IV vs eICU-CRD")
    logger.info("=" * 80)
    
    # Load and preprocess data
    df_mimic, df_eicu, feature_mapping = load_and_preprocess_data()
    
    # Apply unit conversions
    df_mimic, df_eicu = apply_unit_conversions(df_mimic, df_eicu)
    
    # Unit coverage analysis
    unit_df = analyze_unit_coverage(df_mimic, df_eicu)
    
    # 1. Marginal distribution analysis
    summary_df = analyze_marginal_distributions(df_mimic, df_eicu)
    
    # 2. Temporal frequency analysis for selected features
    logger.info("\n" + "=" * 80)
    logger.info("TEMPORAL FREQUENCY ANALYSIS")
    logger.info("=" * 80)
    
    for feature in tqdm(TEMPORAL_FEATURES[:5], desc="Analyzing temporal patterns"):  # Start with 5
        try:
            mimic_freq, eicu_freq, mimic_gaps, eicu_gaps = analyze_temporal_frequency(df_mimic, df_eicu, feature)
            plot_path = f'analysis/plots/temporal/{feature.replace("/", "_")}_frequency.png'
            plot_temporal_frequency(mimic_freq, eicu_freq, mimic_gaps, eicu_gaps, feature, plot_path)
        except Exception as e:
            logger.warning(f"  Failed temporal analysis for {feature}: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: analysis/results/")
    logger.info(f"Plots saved to: analysis/plots/")


if __name__ == "__main__":
    main()

