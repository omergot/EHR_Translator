#!/usr/bin/env python3
"""
Preprocessing Module for MIMIC and eICU Data
Handles data cleaning, scaling, and train/val/test splitting.
"""

import pandas as pd
import numpy as np
import yaml
import argparse
import sys
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import audit_preprocessing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Preprocessor:
    """Preprocessor class for handling MIMIC and eICU data"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['paths']['output_dir'])
        self.scalers_dir = self.output_dir / "scalers"
        self.scalers_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scalers
        self.mimic_scaler = None
        self.eicu_scaler = None
        
        # Define feature columns based on POC features CSV structure
        self.feature_names = ['HR', 'RR', 'SpO2', 'Temp', 'MAP', 'WBC', 'Na', 'Creat']
        self.demographic_names = ['Age', 'Gender']
        
        # Get max missing percentage from config (default 0.5 = 50%)
        self.max_missing_pct = config.get('preprocessing', {}).get('max_missing_pct', 0.5)
        self.removed_features = []  # Track removed features
        
        # Clinical ranges for clipping (feature_name -> (min, max))
        self.clinical_ranges = {
            # POC Features
            'HR': (30, 250),          # Heart Rate
            'RR': (5, 60),            # Respiratory Rate
            'SpO2': (50, 100),        # Oxygen Saturation
            'Temp': (30, 45),         # Temperature (Celsius)
            'MAP': (30, 250),         # Mean Arterial Pressure
            'WBC': (0.1, 100),        # White Blood Cell count
            'Na': (110, 180),         # Sodium
            'Creat': (0.1, 20),       # Creatinine
            'Age': (0, 120),          # Age
            'Gender': (0, 1),         # Gender (0 or 1)
        }
    
    def load_raw_data(self):
        """Load raw data from POC features CSV files"""
        logger.info("Loading POC features data...")
        
        mimic_path = Path(self.config['paths']['mimic_poc_csv'])
        eicu_path = Path(self.config['paths']['eicu_poc_csv'])
        
        if not mimic_path.exists():
            raise FileNotFoundError(f"MIMIC POC features data not found: {mimic_path}")
        if not eicu_path.exists():
            raise FileNotFoundError(f"eICU POC features data not found: {eicu_path}")
        
        mimic_data = pd.read_csv(mimic_path)
        eicu_data = pd.read_csv(eicu_path)
        
        logger.info(f"Loaded MIMIC POC features: {mimic_data.shape}")
        logger.info(f"Loaded eICU POC features: {eicu_data.shape}")
        
        return mimic_data, eicu_data
    
    def filter_high_missingness_features(self, mimic_data, eicu_data):
        """Remove features with high missingness from both datasets"""
        logger.info(f"Filtering features with >{self.max_missing_pct*100:.0f}% missing values...")
        
        # Calculate missingness for each clinical feature across both datasets
        feature_missingness = {}
        
        for feature in self.feature_names:
            for suffix in ['_min', '_max', '_mean', '_std']:
                col_name = f"{feature}{suffix}"
                if col_name in mimic_data.columns and col_name in eicu_data.columns:
                    # Calculate missingness in both datasets
                    mimic_missing_pct = mimic_data[col_name].isna().sum() / len(mimic_data)
                    eicu_missing_pct = eicu_data[col_name].isna().sum() / len(eicu_data)
                    
                    # Use the maximum missingness across both datasets
                    max_missing_pct = max(mimic_missing_pct, eicu_missing_pct)
                    feature_missingness[col_name] = {
                        'mimic_pct': mimic_missing_pct,
                        'eicu_pct': eicu_missing_pct,
                        'max_pct': max_missing_pct
                    }
        
        # Identify features to remove
        features_to_remove = []
        for col_name, stats in feature_missingness.items():
            if stats['max_pct'] > self.max_missing_pct:
                features_to_remove.append(col_name)
                logger.info(f"  Removing {col_name}: MIMIC {stats['mimic_pct']*100:.1f}%, "
                           f"eICU {stats['eicu_pct']*100:.1f}% missing")
        
        # Also remove the base feature name if ALL its suffixes are removed
        features_to_remove_base = set()
        for feature in self.feature_names:
            feature_cols = [f"{feature}{s}" for s in ['_min', '_max', '_mean', '_std']]
            if all(col in features_to_remove for col in feature_cols if col in mimic_data.columns):
                features_to_remove_base.add(feature)
                logger.info(f"  Removing entire feature: {feature} (all statistics have high missingness)")
        
        # Update feature_names to exclude removed features
        self.feature_names = [f for f in self.feature_names if f not in features_to_remove_base]
        self.removed_features = list(features_to_remove_base)
        
        # Drop columns from both datasets
        if features_to_remove:
            mimic_data = mimic_data.drop(columns=features_to_remove, errors='ignore')
            eicu_data = eicu_data.drop(columns=features_to_remove, errors='ignore')
            logger.info(f"Removed {len(features_to_remove)} feature columns due to high missingness")
        else:
            logger.info("No features removed - all have acceptable missingness levels")
        
        logger.info(f"Remaining features: {self.feature_names}")
        logger.info(f"Final shapes - MIMIC: {mimic_data.shape}, eICU: {eicu_data.shape}")
        
        return mimic_data, eicu_data
    
    def get_feature_columns(self, data):
        """Get feature columns from POC features data"""
        feature_cols = []
        
        # Add all clinical features with their suffixes
        # IMPORTANT: Order must match the CSV column order (min, max, mean, std)
        for feature in self.feature_names:
            for suffix in ['_min', '_max', '_mean', '_std']:
                col_name = f"{feature}{suffix}"
                if col_name in data.columns:
                    feature_cols.append(col_name)
        
        # Add demographic features
        for demo in self.demographic_names:
            if demo in data.columns:
                feature_cols.append(demo)
        
        return feature_cols
    
    def _monotonic_violation_mask(self, data, tolerance=0.0):
        """Return boolean mask for rows violating min <= mean <= max for any base feature.
        
        Args:
            data: DataFrame to check
            tolerance: Allow violations up to this amount (for floating point precision)
        """
        if data is None or len(data) == 0:
            return np.zeros(len(data), dtype=bool)
        violation = np.zeros(len(data), dtype=bool)
        for feature in self.feature_names:
            mn, md, mx = f"{feature}_min", f"{feature}_mean", f"{feature}_max"
            if mn in data.columns and md in data.columns and mx in data.columns:
                vals = data[[mn, md, mx]].values
                # Skip rows with NaN values (they'll be handled elsewhere)
                valid_mask = ~(np.isnan(vals).any(axis=1))
                # violation if mean less than min or mean greater than max or min greater than max
                # Add tolerance for floating point precision issues after scaling
                v = np.zeros(len(vals), dtype=bool)
                v[valid_mask] = (
                    (vals[valid_mask, 1] < vals[valid_mask, 0] - tolerance) |  # mean < min
                    (vals[valid_mask, 1] > vals[valid_mask, 2] + tolerance) |  # mean > max
                    (vals[valid_mask, 0] > vals[valid_mask, 2] + tolerance)    # min > max
                )
                violation |= v
        return violation
    
    def drop_monotonicity_violations(self, data, domain_label="raw"):
        """Drop rows that violate min<=mean<=max and log how many were removed (for RAW stage)."""
        mask = self._monotonic_violation_mask(data)
        num_bad = int(mask.sum())
        if num_bad > 0:
            logger.warning(f"[{domain_label}] Dropping {num_bad} rows with min/mean/max monotonicity violations")
            data = data.loc[~mask].reset_index(drop=True)
        else:
            logger.info(f"[{domain_label}] No min/mean/max monotonicity violations found")
        return data, num_bad
    
    def plot_feature_distributions(self, mimic_data, eicu_data):
        """
        Plot feature distributions for MIMIC vs EICU by gender and age groups.
        Creates one PNG per feature variant (e.g., wbc_min, wbc_max, etc.) with:
        - 4 columns: MIMIC Gender-0, MIMIC Gender-1, EICU Gender-0, EICU Gender-1
        - Multiple rows: one per age group (10-year increments)
        """
        logger.info("=" * 80)
        logger.info("STEP 0.8: Plotting feature distributions by dataset, gender, and age group...")
        logger.info("=" * 80)
        
        # Create directory for distribution plots
        plots_dir = self.output_dir / "distribution_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Suppress matplotlib warnings
        warnings.filterwarnings('ignore')
        
        # Get all feature variants (excluding _missing flags and Age/Gender)
        feature_variants = []
        for feature in self.feature_names:
            for suffix in ['_min', '_max', '_mean', '_std']:
                col_name = f"{feature}{suffix}"
                if col_name in mimic_data.columns and col_name in eicu_data.columns:
                    feature_variants.append(col_name)
        
        logger.info(f"Plotting distributions for {len(feature_variants)} features...")
        
        # Define age groups (10-year increments)
        age_groups = [
            (18, 28), (28, 38), (38, 48), (48, 58), 
            (58, 68), (68, 78), (78, 88), (88, 98)
        ]
        
        # Add dataset label columns
        mimic_data = mimic_data.copy()
        eicu_data = eicu_data.copy()
        mimic_data['dataset'] = 'MIMIC'
        eicu_data['dataset'] = 'EICU'
        
        # Ensure Gender and Age columns exist
        if 'Gender' not in mimic_data.columns or 'Age' not in mimic_data.columns:
            logger.warning("Gender or Age columns missing in MIMIC data - skipping distribution plots")
            return
        if 'Gender' not in eicu_data.columns or 'Age' not in eicu_data.columns:
            logger.warning("Gender or Age columns missing in EICU data - skipping distribution plots")
            return
        
        # Plot each feature variant
        for feature_col in feature_variants:
            logger.info(f"  Plotting {feature_col}...")
            
            # Create figure with subplots: rows = age groups, cols = 4 (2 datasets × 2 genders)
            n_age_groups = len(age_groups)
            fig, axes = plt.subplots(n_age_groups, 4, figsize=(20, 4 * n_age_groups))
            fig.suptitle(f'Distribution of {feature_col} by Dataset, Gender, and Age Group', 
                        fontsize=16, y=0.995)
            
            # Ensure axes is 2D array even with single row
            if n_age_groups == 1:
                axes = axes.reshape(1, -1)
            
            # Plot each age group (rows)
            for row_idx, (age_min, age_max) in enumerate(age_groups):
                # Filter data for this age group
                mimic_age_group = mimic_data[
                    (mimic_data['Age'] >= age_min) & (mimic_data['Age'] < age_max)
                ]
                eicu_age_group = eicu_data[
                    (eicu_data['Age'] >= age_min) & (eicu_data['Age'] < age_max)
                ]
                
                # Plot for each combination of dataset and gender (columns)
                column_configs = [
                    ('MIMIC', 0, mimic_age_group),
                    ('MIMIC', 1, mimic_age_group),
                    ('EICU', 0, eicu_age_group),
                    ('EICU', 1, eicu_age_group)
                ]
                
                for col_idx, (dataset_name, gender, data_subset) in enumerate(column_configs):
                    ax = axes[row_idx, col_idx]
                    
                    # Filter by gender
                    gender_data = data_subset[data_subset['Gender'] == gender]
                    
                    # Get feature values (drop NaN)
                    values = gender_data[feature_col].dropna()
                    
                    if len(values) > 0:
                        # Plot histogram with KDE
                        ax.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
                        
                        # Add KDE if enough data points
                        if len(values) > 10:
                            try:
                                sns.kdeplot(values, ax=ax, color='red', linewidth=2)
                            except:
                                pass  # KDE may fail for some distributions
                        
                        # Add statistics text
                        mean_val = values.mean()
                        median_val = values.median()
                        std_val = values.std()
                        n_samples = len(values)
                        
                        stats_text = f'n={n_samples}\nμ={mean_val:.2f}\nσ={std_val:.2f}\nmed={median_val:.2f}'
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                               verticalalignment='top', fontsize=8, 
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    else:
                        # No data available
                        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=12, color='red')
                    
                    # Set titles and labels
                    if row_idx == 0:
                        gender_label = 'Male' if gender == 1 else 'Female'
                        ax.set_title(f'{dataset_name} - {gender_label}', fontsize=10, fontweight='bold')
                    
                    if col_idx == 0:
                        ax.set_ylabel(f'Age {age_min}-{age_max}\nDensity', fontsize=9)
                    else:
                        ax.set_ylabel('Density', fontsize=9)
                    
                    if row_idx == n_age_groups - 1:
                        ax.set_xlabel(feature_col, fontsize=9)
                    
                    ax.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            # Save plot
            plot_path = plots_dir / f"{feature_col}_distribution.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"    Saved: {plot_path}")
        
        logger.info(f"Distribution plots saved to {plots_dir}")
        logger.info("=" * 80)
    
    def clip_to_clinical_ranges(self, data, feature_cols):
        """Clip feature values to clinically plausible ranges"""
        logger.info("Clipping features to clinical ranges...")
        
        data_clipped = data.copy()
        
        for col in feature_cols:
            # CRITICAL: Don't clip _std columns - they have different ranges than base features!
            if '_std' in col:
                continue
            
            # Extract feature name from column (e.g., 'HR_mean' -> 'HR')
            if '_' in col:
                feature_name = col.split('_')[0]
            else:
                feature_name = col
            
            if feature_name in self.clinical_ranges:
                min_val, max_val = self.clinical_ranges[feature_name]
                data_clipped[col] = data_clipped[col].clip(min_val, max_val)
                logger.debug(f"Clipped {col} to range [{min_val}, {max_val}]")
        
        return data_clipped
    
    def create_missing_flags(self, data, feature_cols):
        """Create missing flags for features"""
        logger.info("Creating missing flags...")
        
        data_with_flags = data.copy()
        
        # Create missing flags for each feature based on mean column
        for feature in self.feature_names:
            mean_col = f"{feature}_mean"
            missing_col = f"{feature}_missing"
            
            if mean_col in data.columns and missing_col not in data.columns:
                # Create missing flag based on null values in mean column
                data_with_flags[missing_col] = data_with_flags[mean_col].isnull().astype(int)
        
        return data_with_flags
    
    def analyze_feature_characteristics(self, data, feature_cols):
        """Analyze feature characteristics to determine appropriate preprocessing"""
        logger.info("Analyzing feature characteristics for adaptive preprocessing...")
        
        feature_analysis = {
            "degenerate": [],        # Features with q25 == q75 (most values identical)
            "high_outliers": [],     # Features with outliers using IQR method (Q3 + 1.5*IQR)
            "high_zeros": [],        # Features with >10% zero values
            "missing_flags": [],     # Binary missing indicator flags
            "std_features": [],      # Standard deviation features (often problematic)
            "demographic": []        # Demographic features
        }
        
        for col in feature_cols:
            if col in data.columns:
                values = data[col].dropna()
                
                if len(values) == 0 or col == "Gender":  # Skip Gender from analysis
                    continue
                
                # Check for degenerate distribution (q25 == q75)
                q25 = values.quantile(0.25)
                q75 = values.quantile(0.75)
                if abs(q25 - q75) < 1e-8:  # Essentially the same
                    feature_analysis["degenerate"].append(col)
                
                # Check for outliers using IQR method
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = len(values[(values < lower_bound) | (values > upper_bound)])
                if outliers > 0:  # Any outliers detected by IQR method
                    feature_analysis["high_outliers"].append(col)
                
                # Check for high zero percentage
                zeros = len(values[values == 0])
                if zeros > len(values) * 0.1:  # >10% zeros
                    feature_analysis["high_zeros"].append(col)
                
                # Identify missing flags
                if col.endswith("_missing"):
                    feature_analysis["missing_flags"].append(col)
                
                # Identify std features
                if "_std" in col:
                    feature_analysis["std_features"].append(col)
                
                # Identify demographic features
                if col in self.demographic_names:
                    feature_analysis["demographic"].append(col)
        
        logger.info(f"Feature analysis complete:")
        logger.info(f"  Degenerate distributions: {len(feature_analysis['degenerate'])}")
        logger.info(f"  High outliers (IQR method): {len(feature_analysis['high_outliers'])}")
        logger.info(f"  High zeros: {len(feature_analysis['high_zeros'])}")
        logger.info(f"  Missing flags: {len(feature_analysis['missing_flags'])}")
        logger.info(f"  Std features: {len(feature_analysis['std_features'])}")
        
        return feature_analysis
    
    def clip_outliers_iqr(self, data, feature_cols, multiplier=3.0):
        """
        CRITICAL FIX: Clip outliers using IQR method for features that need it
        
        Args:
            data: DataFrame with features
            feature_cols: List of columns to check for outliers
            multiplier: IQR multiplier (3.0 is more conservative than 1.5, catching only extreme outliers)
        
        Returns:
            DataFrame with outliers clipped
        """
        logger.info(f"Clipping outliers using IQR method (multiplier={multiplier})...")
        
        data_clipped = data.copy()
        n_total_clipped = 0
        
        for col in feature_cols:
            if col not in data.columns or col.endswith("_missing") or col == "Gender":
                continue
            
            values = data[col].dropna()
            if len(values) == 0:
                continue
            
            # Calculate IQR bounds
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            # Use more conservative multiplier (3.0 instead of 1.5) for extreme outliers only
            # This prevents clipping valid extreme values while catching data errors
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            # Count outliers before clipping
            n_outliers_before = len(values[(values < lower_bound) | (values > upper_bound)])
            
            if n_outliers_before > 0:
                # Clip outliers
                data_clipped[col] = data_clipped[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Count outliers after clipping (should be 0)
                values_after = data_clipped[col].dropna()
                n_outliers_after = len(values_after[(values_after < lower_bound) | (values_after > upper_bound)])
                
                n_total_clipped += n_outliers_before
                
                # Log clipping details
                orig_min, orig_max = values.min(), values.max()
                new_min, new_max = values_after.min(), values_after.max()
                
                logger.warning(f"  {col}: Clipped {n_outliers_before} outliers using IQR")
                logger.warning(f"    IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                logger.warning(f"    Original range: [{orig_min:.2f}, {orig_max:.2f}]")
                logger.warning(f"    Clipped range: [{new_min:.2f}, {new_max:.2f}]")
        
        if n_total_clipped > 0:
            logger.info(f"Total outliers clipped: {n_total_clipped}")
        else:
            logger.info("No outliers detected (IQR method)")
        
        return data_clipped
    
    def apply_feature_specific_transforms(self, data, feature_analysis):
        """Apply log1p transformation to std columns before scaling"""
        logger.info("Applying feature-specific transformations...")
        
        data_transformed = data.copy()
        
        # Apply log1p transformation ONLY to std features (reduces skew)
        # std is always non-negative, so no shift needed
        log1p_features = set(feature_analysis["std_features"])
        
        for col in log1p_features:
            if col in data_transformed.columns and col not in feature_analysis["missing_flags"] and col != "Gender":
                # Apply log1p transformation (NaN stays NaN)
                # std is always >= 0, so no negative shift needed
                data_transformed[col] = np.log1p(data_transformed[col])
                logger.debug(f"Applied log1p transform to {col} (std feature)")
        
        logger.info(f"Applied log1p to {len(log1p_features)} std features")
        return data_transformed


    def prepare_features(self, data, domain="mimic"):
        """Prepare features for preprocessing with advanced feature-specific handling"""
        logger.info(f"Preparing features for {domain}...")
        
        # Get feature columns
        feature_cols = self.get_feature_columns(data)
        
        # Separate numeric and demographic columns
        clinical_numeric_cols = [col for col in feature_cols if "_mean" in col or "_min" in col or "_max" in col or "_std" in col]
        demographic_cols = [col for col in feature_cols if col in self.demographic_names]
        
        # Create missing flags for clinical features (not demographics)
        data = self.create_missing_flags(data, feature_cols)
        missing_cols = [col for col in data.columns if "_missing" in col]
        
        # All available feature columns (including missing flags)
        all_available_cols = clinical_numeric_cols + demographic_cols + missing_cols
        
        # Analyze feature characteristics for adaptive preprocessing
        feature_analysis = self.analyze_feature_characteristics(data, all_available_cols)
        
        # Determine which columns should be scaled vs kept as-is
        # EXCLUDE missing flags and Gender from scaling (keep them as binary indicators)
        excluded_from_scaling = feature_analysis["missing_flags"] + ["Gender"]
        scalable_cols = [col for col in all_available_cols if col not in excluded_from_scaling]
        
        logger.info(f"Total features: {len(all_available_cols)}, Scalable: {len(scalable_cols)}, Missing flags: {len(missing_cols)}, Gender: excluded")
        
        
        # IMPORTANT: DON'T fill missing values yet! 
        # Filling with 0 before transformation breaks RobustScaler when most values are missing
        # We'll fill AFTER transformations and scaling
        
        # Fill missing values for demographics with appropriate defaults
        for col in demographic_cols:
            if col == "Age":
                data[col] = data[col].fillna(data[col].median())  # Use median age
            elif col == "Gender":
                data[col] = data[col].fillna(0)  # Default to 0 for missing gender
                # Ensure Gender remains binary (0 or 1) - no scaling applied
                data[col] = data[col].astype(int).clip(0, 1)
        
        # Ensure missing flags are binary
        for col in missing_cols:
            data[col] = data[col].astype(int)
        
        # Apply feature-specific transformations to scalable features (only log1p on std columns)
        # NaN values will remain NaN through transformations (log1p(NaN) = NaN)
        # Monotonicity is preserved because we only transform std columns (no min/mean/max ordering constraint)
        data = self.apply_feature_specific_transforms(data, feature_analysis)
        
        return data, scalable_cols, missing_cols, feature_analysis
    
    def fit_scalers(self, mimic_train_data, eicu_train_data):
        """Fit group-based RobustScalers that preserve min<=mean<=max ordering"""
        logger.info("Fitting group-based RobustScalers (same params for min/mean/max)...")
        
        # Prepare features with advanced analysis for TRAINING data only
        mimic_prepared, mimic_scalable, mimic_missing, mimic_analysis = self.prepare_features(mimic_train_data, "mimic")
        eicu_prepared, eicu_scalable, eicu_missing, eicu_analysis = self.prepare_features(eicu_train_data, "eicu")
        
        # Define feature groups: {min, mean, max} share same scaler; std separate
        feature_groups = {}
        for col in mimic_scalable:
            if '_' in col:
                base_name = col.rsplit('_', 1)[0]
                suffix = col.rsplit('_', 1)[1]
                if suffix in ['min', 'max', 'mean', 'std']:
                    if base_name not in feature_groups:
                        feature_groups[base_name] = {'min': [], 'max': [], 'mean': [], 'std': []}
                    feature_groups[base_name][suffix].append(col)
        
        logger.info(f"Identified {len(feature_groups)} feature groups")
        
        # Store unified scaling parameters
        self.unified_scaling_params = {
            'mimic': {},
            'eicu': {}
        }
        
        # For each feature group, compute UNIFIED RobustScaler params from pooled {min, mean, max}
        # Apply SAME scaler to all three → preserves ordering automatically
        for base_name, group_cols in feature_groups.items():
            # Pool min, mean, max values to compute unified median/IQR
            mimic_values = []
            eicu_values = []
            
            for suffix in ['min', 'mean', 'max']:
                if group_cols[suffix]:
                    col = group_cols[suffix][0]
                    if col in mimic_prepared.columns:
                        mimic_values.append(mimic_prepared[col].dropna().values)
                    if col in eicu_prepared.columns:
                        eicu_values.append(eicu_prepared[col].dropna().values)
            
            # Compute unified RobustScaler params from pooled data
            if mimic_values and len(mimic_values) > 0:
                mimic_pooled = np.concatenate([v for v in mimic_values if len(v) > 0])
                if len(mimic_pooled) > 0:
                    mimic_median = float(np.median(mimic_pooled))
                    mimic_q25 = float(np.percentile(mimic_pooled, 25))
                    mimic_q75 = float(np.percentile(mimic_pooled, 75))
                    mimic_iqr = mimic_q75 - mimic_q25
                    
                    self.unified_scaling_params['mimic'][base_name] = {
                        'median': mimic_median,
                        'iqr': mimic_iqr if mimic_iqr > 1e-6 else 1.0
                    }
                    logger.debug(f"MIMIC {base_name}: median={mimic_median:.2f}, IQR={mimic_iqr:.2f}")
            
            if eicu_values and len(eicu_values) > 0:
                eicu_pooled = np.concatenate([v for v in eicu_values if len(v) > 0])
                if len(eicu_pooled) > 0:
                    eicu_median = float(np.median(eicu_pooled))
                    eicu_q25 = float(np.percentile(eicu_pooled, 25))
                    eicu_q75 = float(np.percentile(eicu_pooled, 75))
                    eicu_iqr = eicu_q75 - eicu_q25
                    
                    self.unified_scaling_params['eicu'][base_name] = {
                        'median': eicu_median,
                        'iqr': eicu_iqr if eicu_iqr > 1e-6 else 1.0
                    }
                    logger.debug(f"eICU {base_name}: median={eicu_median:.2f}, IQR={eicu_iqr:.2f}")
            
            # Handle std columns separately (after log1p transform)
            for suffix in ['std']:
                if group_cols[suffix]:
                    col = group_cols[suffix][0]
                    
                    # MIMIC std scaling (on log1p-transformed values)
                    if col in mimic_prepared.columns:
                        mimic_std_vals = mimic_prepared[col].dropna().values
                        if len(mimic_std_vals) > 0:
                            mimic_std_median = float(np.median(mimic_std_vals))
                            mimic_std_q25 = float(np.percentile(mimic_std_vals, 25))
                            mimic_std_q75 = float(np.percentile(mimic_std_vals, 75))
                            mimic_std_iqr = mimic_std_q75 - mimic_std_q25
                            
                            self.unified_scaling_params['mimic'][f"{base_name}_std"] = {
                                'median': mimic_std_median,
                                'iqr': mimic_std_iqr if mimic_std_iqr > 1e-6 else 1.0
                            }
                    
                    # eICU std scaling
                    if col in eicu_prepared.columns:
                        eicu_std_vals = eicu_prepared[col].dropna().values
                        if len(eicu_std_vals) > 0:
                            eicu_std_median = float(np.median(eicu_std_vals))
                            eicu_std_q25 = float(np.percentile(eicu_std_vals, 25))
                            eicu_std_q75 = float(np.percentile(eicu_std_vals, 75))
                            eicu_std_iqr = eicu_std_q75 - eicu_std_q25
                            
                            self.unified_scaling_params['eicu'][f"{base_name}_std"] = {
                                'median': eicu_std_median,
                                'iqr': eicu_std_iqr if eicu_std_iqr > 1e-6 else 1.0
                            }
        
        # Store feature groups for later use
        self.feature_groups = feature_groups
        self.scalable_features = mimic_scalable
        
        # Add Age scaling using RobustScaler per domain
        for domain_name, prepared_df in [("mimic", mimic_prepared), ("eicu", eicu_prepared)]:
            if "Age" in prepared_df.columns:
                age_vals = prepared_df["Age"].dropna().values
                if len(age_vals) > 0:
                    age_median = float(np.median(age_vals))
                    age_q25 = float(np.percentile(age_vals, 25))
                    age_q75 = float(np.percentile(age_vals, 75))
                    age_iqr = age_q75 - age_q25
                    
                    self.unified_scaling_params[domain_name]["Age"] = {
                        "median": age_median,
                        "iqr": age_iqr if age_iqr > 1e-6 else 1.0
                    }
        
        # Save scaling parameters
        with open(self.scalers_dir / "unified_scaling_params.pkl", "wb") as f:
            pickle.dump(self.unified_scaling_params, f)
        with open(self.scalers_dir / "feature_groups.pkl", "wb") as f:
            pickle.dump(feature_groups, f)
        with open(self.scalers_dir / "scaler_info.json", "w") as f:
            scaler_info_json = {
                "feature_groups": {k: {k2: v2 for k2, v2 in v.items()} for k, v in feature_groups.items()},
                "base_features": list(feature_groups.keys()),
                "scaling_method": "group_based_robust",  # Same RobustScaler params per group
                "description": "Each {min,mean,max} group shares identical (median,IQR); std separate"
            }
            json.dump(scaler_info_json, f, indent=2)
        
        logger.info("✓ Group-based RobustScalers fitted and saved")
        logger.info(f"  Feature groups ({len(feature_groups)}): {list(feature_groups.keys())}")
        logger.info(f"  Ordering preserved: same (median,IQR) applied to all {'{min,mean,max}'} in each group")
    def transform_data(self, data, domain="mimic"):
        """Transform data using unified RobustScaler that preserves min<=mean<=max ordering"""
        logger.info(f"Transforming {domain} data with unified RobustScaler...")
        
        # Prepare features with advanced analysis
        data_prepared, scalable_cols, missing_cols, feature_analysis = self.prepare_features(data, domain)
        
        # Apply unified RobustScaler - preserves order because (x - median) / IQR maintains relative ordering
        data_scaled = data_prepared.copy()
        
        # Get scaling parameters for this domain
        scaling_params = self.unified_scaling_params[domain]
        
        # Apply unified RobustScaler to each feature group
        for base_name, group_cols in self.feature_groups.items():
            if base_name not in scaling_params:
                continue
            
            params = scaling_params[base_name]
            
            # Scale min, max, mean with unified RobustScaler parameters
            # RobustScaler: (x - median) / IQR preserves order (monotonic transformation)
            for suffix in ['min', 'max', 'mean']:
                if group_cols[suffix]:
                    col = group_cols[suffix][0]
                    if col in data_scaled.columns:
                        # Apply: (x - median) / IQR
                        data_scaled[col] = (data_scaled[col] - params['median']) / params['iqr']
            
            # Scale std columns with separate RobustScaler
            if group_cols['std']:
                col = group_cols['std'][0]
                std_key = f"{base_name}_std"
                if col in data_scaled.columns and std_key in scaling_params:
                    std_params = scaling_params[std_key]
                    # Apply: (x - median) / IQR
                    data_scaled[col] = (data_scaled[col] - std_params['median']) / std_params['iqr']
        
        logger.info(f"✓ Applied group-based RobustScaler to {len(self.feature_groups)} feature groups")
        logger.info("  Monotonicity preserved: same (median,IQR) → if a≤b then scale(a)≤scale(b)")
        
        # Scale Age using RobustScaler
        if "Age" in data_scaled.columns and "Age" in scaling_params:
            age_p = scaling_params["Age"]
            data_scaled["Age"] = (data_scaled["Age"] - age_p["median"]) / age_p["iqr"]
            logger.info("Applied Age RobustScaler for domain: %s" % domain)
        
        # Missing flags remain unscaled (binary indicators)
        logger.info(f"Kept {len(missing_cols)} missing flags unscaled as binary indicators")
        
        # NOW fill NaN values with 0 AFTER all scaling is complete
        # This preserves monotonicity because 0 is filled after transformation
        clinical_feature_cols = [col for col in data_scaled.columns 
                                 if ('_min' in col or '_max' in col or '_mean' in col or '_std' in col)
                                 and not col.endswith('_missing')]
        for col in clinical_feature_cols:
            data_scaled[col] = data_scaled[col].fillna(0)
        
        # Fill demographic NaN (should be rare)
        if 'Age' in data_scaled.columns:
            data_scaled['Age'] = data_scaled['Age'].fillna(0)  # 0 is near median after RobustScaler
        if 'Gender' in data_scaled.columns:
            data_scaled['Gender'] = data_scaled['Gender'].fillna(0)
        
        logger.info(f"Filled NaN values with 0 AFTER scaling (preserves monotonicity)")
        
        return data_scaled, scalable_cols, missing_cols
    
    def split_data(self, data, domain='mimic'):
        """Split data into train/val/test sets"""
        logger.info(f"Splitting {domain} data...")
        
        # Use patient-level split (icu_stay_id)
        patient_ids = data['icu_stay_id'].unique()
        
        # SIMPLIFIED: Split patient IDs into train/test only
        train_ids, test_ids = train_test_split(
            patient_ids, 
            test_size=self.config['data']['test_split'],
            random_state=self.config['data']['random_seed']
        )
        
        # SIMPLIFIED: Create train/test splits only
        train_data = data[data['icu_stay_id'].isin(train_ids)]
        test_data = data[data['icu_stay_id'].isin(test_ids)]
        
        logger.info(f"{domain} splits - Train: {len(train_data)}, Test: {len(test_data)}")
        
        return train_data, test_data
    
    def save_splits(self, mimic_splits, eicu_splits):
        """SIMPLIFIED: Save train/test splits only"""
        logger.info("Saving data splits...")
        
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save MIMIC splits
        train_mimic, test_mimic = mimic_splits
        train_mimic.to_csv(data_dir / "train_mimic_preprocessed.csv", index=False)
        test_mimic.to_csv(data_dir / "test_mimic_preprocessed.csv", index=False)
        
        # Save eICU splits
        train_eicu, test_eicu = eicu_splits
        train_eicu.to_csv(data_dir / "train_eicu_preprocessed.csv", index=False)
        test_eicu.to_csv(data_dir / "test_eicu_preprocessed.csv", index=False)
        
        # SIMPLIFIED: Save split information (train/test only)
        split_info = {
            'mimic': {
                'train_size': len(train_mimic),
                'test_size': len(test_mimic),
                'train_patients': len(train_mimic['icu_stay_id'].unique()),
                'test_patients': len(test_mimic['icu_stay_id'].unique())
            },
            'eicu': {
                'train_size': len(train_eicu),
                'test_size': len(test_eicu),
                'train_patients': len(train_eicu['icu_stay_id'].unique()),
                'test_patients': len(test_eicu['icu_stay_id'].unique())
            }
        }
        
        with open(data_dir / "split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info("Data splits saved")
    
    def create_feature_spec(self, mimic_data, eicu_data):
        """Create feature specification for the model"""
        logger.info("Creating feature specification...")
        
        # Get feature columns
        mimic_features = self.get_feature_columns(mimic_data)
        eicu_features = self.get_feature_columns(eicu_data)
        
        # Create feature spec
        feature_spec = {
            'n_clinical_features': len(self.feature_names),
            'n_demographic_features': len(self.demographic_names),
            'clinical_features': self.feature_names,
            'demographic_features': self.demographic_names,
            'numeric_features': [],
            'missing_features': [],
            'all_features': mimic_features
        }
        
        # Add numeric feature columns
        for feature in self.feature_names:
            # IMPORTANT: Order must match the CSV column order (min, max, mean, std)
            for suffix in ['_min', '_max', '_mean', '_std']:
                col_name = f"{feature}{suffix}"
                if col_name in mimic_data.columns:
                    feature_spec['numeric_features'].append(col_name)
            
            # Add missing feature columns
            missing_col = f"{feature}_missing"
            feature_spec['missing_features'].append(missing_col)
        
        # Add demographic features to numeric features (they don't have missing flags)
        feature_spec['numeric_features'].extend(self.demographic_names)
        
        # Add information about removed features
        feature_spec['removed_features'] = self.removed_features
        feature_spec['max_missing_pct_threshold'] = self.max_missing_pct
        
        # Save feature spec
        spec_path = self.output_dir / "feature_spec.json"
        with open(spec_path, 'w') as f:
            json.dump(feature_spec, f, indent=2)
        
        logger.info(f"Feature specification saved to {spec_path}")
        if self.removed_features:
            logger.info(f"Removed features due to high missingness: {self.removed_features}")
        return feature_spec
    
    def preprocess(self, plot_distributions=False):
        """Main preprocessing pipeline - FIXED for no data leakage"""
        logger.info("Starting preprocessing pipeline...")
        
        # Load raw data
        mimic_data, eicu_data = self.load_raw_data()
        
        # STEP 0: Filter out features with high missingness
        mimic_data, eicu_data = self.filter_high_missingness_features(mimic_data, eicu_data)
        
        # STEP 0.25: CRITICAL FIX - Clip outliers using IQR method BEFORE any other processing
        # This removes extreme outliers that cause training explosions (e.g., SpO2_max = 1,627,555)
        # Uses data-driven IQR bounds rather than hard-coded clinical ranges
        logger.info("=" * 80)
        logger.info("STEP 0.25: Clipping outliers using IQR method...")
        logger.info("=" * 80)
        
        # Get all numeric feature columns (min, max, mean, std for each feature)
        mimic_numeric_cols = [col for col in mimic_data.columns 
                              if '_min' in col or '_max' in col or '_mean' in col or '_std' in col 
                              or col in ['Age', 'Gender']]
        eicu_numeric_cols = [col for col in eicu_data.columns 
                             if '_min' in col or '_max' in col or '_mean' in col or '_std' in col
                             or col in ['Age', 'Gender']]
        
        # Clip MIMIC data using IQR method (3.0 * IQR for extreme outliers only)
        logger.info(f"Clipping MIMIC data outliers (IQR multiplier=3.0)...")
        mimic_data = self.clip_outliers_iqr(mimic_data, mimic_numeric_cols, multiplier=3.0)
        
        # Clip eICU data using IQR method
        logger.info(f"Clipping eICU data outliers (IQR multiplier=3.0)...")
        eicu_data = self.clip_outliers_iqr(eicu_data, eicu_numeric_cols, multiplier=3.0)
        
        logger.info("IQR-based outlier clipping completed!")
        logger.info("=" * 80)
        
        # STEP 0.5: Validate and drop raw rows with min/mean/max violations BEFORE splitting
        mimic_data, m_bad = self.drop_monotonicity_violations(mimic_data, domain_label="RAW-MIMIC")
        eicu_data, e_bad = self.drop_monotonicity_violations(eicu_data, domain_label="RAW-eICU")
        
        # STEP 0.8: Plot feature distributions by dataset, gender, and age group (optional)
        if plot_distributions:
            self.plot_feature_distributions(mimic_data, eicu_data)
        
        # STEP 1: Split data FIRST (before any preprocessing)
        logger.info("Splitting data into train/test sets...")
        mimic_splits = self.split_data(mimic_data, 'mimic')
        eicu_splits = self.split_data(eicu_data, 'eicu')
        
        mimic_train_raw, mimic_test_raw = mimic_splits
        eicu_train_raw, eicu_test_raw = eicu_splits
        
        # STEP 2: Fit scalers ONLY on training data
        self.fit_scalers(mimic_train_raw, eicu_train_raw)
        
        # STEP 3: Transform both train and test data using fitted scalers
        logger.info("Transforming training data...")
        mimic_train_transformed, mimic_scalable, mimic_missing = self.transform_data(mimic_train_raw, "mimic")
        eicu_train_transformed, eicu_scalable, eicu_missing = self.transform_data(eicu_train_raw, "eicu")
        
        logger.info("Transforming test data...")
        mimic_test_transformed, _, _ = self.transform_data(mimic_test_raw, "mimic")
        eicu_test_transformed, _, _ = self.transform_data(eicu_test_raw, "eicu")
        
        # STEP 3.5: Verify monotonicity is preserved after group-based scaling
        # Group-based scaling with same (median,IQR) per {min,mean,max} MUST preserve ordering
        for label, df in [("MIMIC-TRAIN", mimic_train_transformed), ("MIMIC-TEST", mimic_test_transformed),
                          ("eICU-TRAIN", eicu_train_transformed), ("eICU-TEST", eicu_test_transformed)]:
            post_mask = self._monotonic_violation_mask(df, tolerance=1e-6)
            violations = int(post_mask.sum())
            if violations > 0:
                logger.error(f"✗ Found {violations} monotonicity violations in {label}")
                logger.error("  This should NOT happen with group-based scaling (same params per group)")
                # Log first few violations for debugging
                if violations > 0:
                    bad_idx = np.where(post_mask)[0][:5]
                    for idx in bad_idx:
                        for feat in self.feature_names:
                            mn, md, mx = f"{feat}_min", f"{feat}_mean", f"{feat}_max"
                            if all(c in df.columns for c in [mn, md, mx]):
                                vals = df.loc[idx, [mn, md, mx]].values
                                if (vals[1] < vals[0] - 1e-6) or (vals[1] > vals[2] + 1e-6) or (vals[0] > vals[2] + 1e-6):
                                    logger.error(f"    Row {idx} {feat}: min={vals[0]:.6f}, mean={vals[1]:.6f}, max={vals[2]:.6f}")
                raise RuntimeError(f"Monotonicity violated after group-based scaling in {label}")
            else:
                logger.info(f"✓ Monotonicity verified for {label} ({len(df)} rows, min≤mean≤max preserved)")
        
        # STEP 4: Create final splits (already split, just reorganize)
        mimic_final_splits = (mimic_train_transformed, mimic_test_transformed)
        eicu_final_splits = (eicu_train_transformed, eicu_test_transformed)
        
        # Save splits
        self.save_splits(mimic_final_splits, eicu_final_splits)
        
        # Create feature specification
        feature_spec = self.create_feature_spec(mimic_train_transformed, eicu_train_transformed)
        
        logger.info("Preprocessing completed successfully!")
        return feature_spec

def main():
    """Main function for preprocessing"""
    parser = argparse.ArgumentParser(description='Preprocess MIMIC and eICU data')
    parser.add_argument('--config', type=str, default='conf/config.yml', help='Path to config file')
    parser.add_argument('--fit', action='store_true', help='Fit scalers and preprocess data')
    parser.add_argument('--audit', action='store_true', help='Run preprocessing audit to identify problematic features')
    parser.add_argument('--plot-distributions', action='store_true', 
                       help='Plot feature distributions by dataset, gender, and age group (creates PNG files)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create preprocessor
    preprocessor = Preprocessor(config)
    
    if args.fit:
        # Run preprocessing
        feature_spec = preprocessor.preprocess(plot_distributions=args.plot_distributions)
        logger.info("Preprocessing completed!")
        
        # Run audit if requested
        if args.audit:
            logger.info("Running preprocessing audit...")
            audit_results = audit_preprocessing(config['paths']['output_dir'], feature_spec)
            if audit_results:
                logger.info("Preprocessing audit completed - check preprocessing_audit_results.json for details")
    elif args.audit:
        # Run audit only (on existing preprocessed data)
        logger.info("Running preprocessing audit on existing data...")
        audit_results = audit_preprocessing(config['paths']['output_dir'])
        if audit_results:
            logger.info("Preprocessing audit completed - check preprocessing_audit_results.json for details")
        else:
            logger.error("Audit failed - check that preprocessed data exists")
    else:
        logger.info("Use --fit flag to run preprocessing or --audit flag to run audit on existing data")

if __name__ == "__main__":
    main()
