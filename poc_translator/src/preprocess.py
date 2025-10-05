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
    
    def apply_feature_specific_transforms(self, data, feature_analysis):
        """Apply feature-specific transformations before scaling"""
        logger.info("Applying feature-specific transformations...")
        
        data_transformed = data.copy()
        
        # FIRST: Clip outliers for high-outlier features using IQR method (before transformation)
        # Skip degenerate features (IQR=0) to avoid collapsing all values to a single point
        for col in feature_analysis["high_outliers"]:
            if col in data_transformed.columns and col not in feature_analysis["missing_flags"] and col != "Gender":
                # Skip if feature is degenerate (already identified)
                if col in feature_analysis["degenerate"]:
                    logger.debug(f"Skipping IQR clipping for {col} (degenerate distribution)")
                    continue
                
                values = data_transformed[col].dropna()
                if len(values) > 0:
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    iqr = q3 - q1
                    
                    # Additional check: skip if IQR is too small (would collapse values)
                    if iqr < 0.01:
                        logger.debug(f"Skipping IQR clipping for {col} (IQR={iqr:.4f} too small)")
                        continue
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    original_min = data_transformed[col].min()
                    original_max = data_transformed[col].max()
                    data_transformed[col] = data_transformed[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.debug(f"Clipped outliers for {col} using IQR: [{original_min:.3f}, {original_max:.3f}] -> [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        # THEN: Apply log1p transformation to degenerate and std features
        log1p_features = set(feature_analysis["degenerate"] + feature_analysis["std_features"])
        
        for col in log1p_features:
            if col in data_transformed.columns and col not in feature_analysis["missing_flags"] and col != "Gender":  # Skip Gender and missing flags from transformations
                # Ensure no negative values before log1p (check only non-NaN values)
                min_val = data_transformed[col].min(skipna=True)
                if not pd.isna(min_val) and min_val < 0:
                    # Shift to make all values positive
                    data_transformed[col] = data_transformed[col] - min_val
                
                # Apply log1p transformation (NaN stays NaN)
                data_transformed[col] = np.log1p(data_transformed[col])
                logger.debug(f"Applied log1p transform to {col}")
        
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
        
        # Apply feature-specific transformations to scalable features
        # NaN values will remain NaN through transformations (log1p(NaN) = NaN)
        data = self.apply_feature_specific_transforms(data, feature_analysis)
        
        return data, scalable_cols, missing_cols, feature_analysis
    
    def fit_scalers(self, mimic_train_data, eicu_train_data):
        """Fit unified scalers that preserve min/max/mean relationships within each feature"""
        logger.info("Fitting unified scalers on training data...")
        
        # Prepare features with advanced analysis for TRAINING data only
        mimic_prepared, mimic_scalable, mimic_missing, mimic_analysis = self.prepare_features(mimic_train_data, "mimic")
        eicu_prepared, eicu_scalable, eicu_missing, eicu_analysis = self.prepare_features(eicu_train_data, "eicu")
        
        # Group features by base name (HR, RR, SpO2, etc.)
        feature_groups = {}
        for col in mimic_scalable:
            # Extract base feature name (e.g., "HR" from "HR_min")
            if '_' in col:
                base_name = col.rsplit('_', 1)[0]  # Get everything before last underscore
                suffix = col.rsplit('_', 1)[1]
                if suffix in ['min', 'max', 'mean', 'std']:
                    if base_name not in feature_groups:
                        feature_groups[base_name] = {'min': [], 'max': [], 'mean': [], 'std': []}
                    feature_groups[base_name][suffix].append(col)
        
        logger.info(f"Identified {len(feature_groups)} feature groups for unified scaling")
        
        # Store unified scaling parameters for each feature group
        self.unified_scaling_params = {
            'mimic': {},
            'eicu': {}
        }
        
        # For each feature group, compute unified scaling parameters
        for base_name, group_cols in feature_groups.items():
            # Pool min, max, mean values together to compute unified parameters
            mimic_values = []
            eicu_values = []
            
            for suffix in ['min', 'max', 'mean']:
                if group_cols[suffix]:
                    col = group_cols[suffix][0]
                    mimic_values.append(mimic_prepared[col].dropna().values)
                    eicu_values.append(eicu_prepared[col].dropna().values)
            
            # Compute unified parameters from pooled data
            if mimic_values:
                mimic_pooled = np.concatenate(mimic_values)
                mimic_mean = np.mean(mimic_pooled)
                mimic_std = np.std(mimic_pooled)
                mimic_median = np.median(mimic_pooled)
                mimic_q25 = np.percentile(mimic_pooled, 25)
                mimic_q75 = np.percentile(mimic_pooled, 75)
                mimic_iqr = mimic_q75 - mimic_q25
                
                self.unified_scaling_params['mimic'][base_name] = {
                    'mean': mimic_mean,
                    'std': mimic_std if mimic_std > 0 else 1.0,
                    'median': mimic_median,
                    'iqr': mimic_iqr if mimic_iqr > 0 else 1.0
                }
            
            if eicu_values:
                eicu_pooled = np.concatenate(eicu_values)
                eicu_mean = np.mean(eicu_pooled)
                eicu_std = np.std(eicu_pooled)
                eicu_median = np.median(eicu_pooled)
                eicu_q25 = np.percentile(eicu_pooled, 25)
                eicu_q75 = np.percentile(eicu_pooled, 75)
                eicu_iqr = eicu_q75 - eicu_q25
                
                self.unified_scaling_params['eicu'][base_name] = {
                    'mean': eicu_mean,
                    'std': eicu_std if eicu_std > 0 else 1.0,
                    'median': eicu_median,
                    'iqr': eicu_iqr if eicu_iqr > 0 else 1.0
                }
            
            # For _std columns, use separate RobustScaler
            for suffix in ['std']:
                if group_cols[suffix]:
                    col = group_cols[suffix][0]
                    mimic_std_vals = mimic_prepared[col].dropna().values
                    eicu_std_vals = eicu_prepared[col].dropna().values
                    
                    if len(mimic_std_vals) > 0:
                        mimic_std_median = np.median(mimic_std_vals)
                        mimic_std_q25 = np.percentile(mimic_std_vals, 25)
                        mimic_std_q75 = np.percentile(mimic_std_vals, 75)
                        mimic_std_iqr = mimic_std_q75 - mimic_std_q25
                        
                        self.unified_scaling_params['mimic'][f"{base_name}_std"] = {
                            'median': mimic_std_median,
                            'iqr': mimic_std_iqr if mimic_std_iqr > 0 else 1.0
                        }
                    
                    if len(eicu_std_vals) > 0:
                        eicu_std_median = np.median(eicu_std_vals)
                        eicu_std_q25 = np.percentile(eicu_std_vals, 25)
                        eicu_std_q75 = np.percentile(eicu_std_vals, 75)
                        eicu_std_iqr = eicu_std_q75 - eicu_std_q25
                        
                        self.unified_scaling_params['eicu'][f"{base_name}_std"] = {
                            'median': eicu_std_median,
                            'iqr': eicu_std_iqr if eicu_std_iqr > 0 else 1.0
                        }
        
        # Store feature groups for later use
        self.feature_groups = feature_groups
        self.scalable_features = mimic_scalable
        
        # Save scaling parameters
        scaler_info = {
            "unified_scaling_params": self.unified_scaling_params,
            "feature_groups": {k: {k2: v2 for k2, v2 in v.items()} for k, v in feature_groups.items()},
            "mimic_analysis": mimic_analysis,
            "eicu_analysis": eicu_analysis
        }
        
        with open(self.scalers_dir / "unified_scaling_params.pkl", "wb") as f:
            pickle.dump(self.unified_scaling_params, f)
        with open(self.scalers_dir / "feature_groups.pkl", "wb") as f:
            pickle.dump(feature_groups, f)
        with open(self.scalers_dir / "scaler_info.json", "w") as f:
            # Convert for JSON serialization
            scaler_info_json = {
                "feature_groups": {k: {k2: v2 for k2, v2 in v.items()} for k, v in feature_groups.items()},
                "base_features": list(feature_groups.keys())
            }
            json.dump(scaler_info_json, f, indent=2)
        
        logger.info("Unified scalers fitted and saved")
        logger.info(f"Feature groups: {list(feature_groups.keys())}")
    def transform_data(self, data, domain="mimic"):
        """Transform data using unified scaling that preserves min/max/mean relationships"""
        logger.info(f"Transforming {domain} data with unified scaling...")
        
        # Prepare features with advanced analysis
        data_prepared, scalable_cols, missing_cols, feature_analysis = self.prepare_features(data, domain)
        
        # Apply unified scaling
        data_scaled = data_prepared.copy()
        
        # Get scaling parameters for this domain
        scaling_params = self.unified_scaling_params[domain]
        
        # Apply unified scaling to each feature group
        for base_name, group_cols in self.feature_groups.items():
            if base_name not in scaling_params:
                continue
            
            params = scaling_params[base_name]
            
            # Scale min, max, mean with unified parameters (StandardScaler-like)
            for suffix in ['min', 'max', 'mean']:
                if group_cols[suffix]:
                    col = group_cols[suffix][0]
                    if col in data_scaled.columns:
                        # Apply: (x - mean) / std
                        data_scaled[col] = (data_scaled[col] - params['mean']) / params['std']
            
            # Scale std columns with RobustScaler-like approach
            if group_cols['std']:
                col = group_cols['std'][0]
                std_key = f"{base_name}_std"
                if col in data_scaled.columns and std_key in scaling_params:
                    std_params = scaling_params[std_key]
                    # Apply: (x - median) / IQR
                    data_scaled[col] = (data_scaled[col] - std_params['median']) / std_params['iqr']
        
        logger.info(f"Applied unified scaling to {len(self.feature_groups)} feature groups")
        
        # Missing flags remain unscaled (binary indicators)
        logger.info(f"Kept {len(missing_cols)} missing flags unscaled as binary indicators")
        
        # NOW fill missing values with 0 for clinical features (after scaling)
        # This prevents breaking the scaler when most values are missing
        clinical_feature_cols = [col for col in data_scaled.columns 
                                 if ('_min' in col or '_max' in col or '_mean' in col or '_std' in col)
                                 and not col.endswith('_missing')]
        for col in clinical_feature_cols:
            data_scaled[col] = data_scaled[col].fillna(0)
        
        logger.info(f"Filled remaining NaN values with 0 after scaling")
        
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
    
    def preprocess(self):
        """Main preprocessing pipeline - FIXED for no data leakage"""
        logger.info("Starting preprocessing pipeline...")
        
        # Load raw data
        mimic_data, eicu_data = self.load_raw_data()
        
        # STEP 0: Filter out features with high missingness
        mimic_data, eicu_data = self.filter_high_missingness_features(mimic_data, eicu_data)
        
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
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create preprocessor
    preprocessor = Preprocessor(config)
    
    if args.fit:
        # Run preprocessing
        feature_spec = preprocessor.preprocess()
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
