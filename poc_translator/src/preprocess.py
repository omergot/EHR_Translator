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
    
    def get_feature_columns(self, data):
        """Get feature columns from POC features data"""
        feature_cols = []
        
        # Add all clinical features with their suffixes
        for feature in self.feature_names:
            for suffix in ['_mean', '_min', '_max', '_std']:
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
            "high_outliers": [],     # Features with >1% outliers beyond 99th percentile
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
                
                # Check for high outlier percentage
                q99 = values.quantile(0.99)
                outliers = len(values[values > q99])
                if outliers > len(values) * 0.01:  # >1% outliers
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
        logger.info("Feature analysis complete:")
        logger.info(f"  Degenerate distributions: {len(feature_analysis['degenerate'])}")
        logger.info(f"  High outliers: {len(feature_analysis['high_outliers'])}")
        logger.info(f"  High zeros: {len(feature_analysis['high_zeros'])}")
        logger.info(f"  Missing flags: {len(feature_analysis['missing_flags'])}")
        logger.info(f"  Std features: {len(feature_analysis['std_features'])}")
        
        return feature_analysis
    
    def apply_feature_specific_transforms(self, data, feature_analysis):
        """Apply feature-specific transformations before scaling"""
        logger.info("Applying feature-specific transformations...")
        
        data_transformed = data.copy()
        
        # Apply log1p transformation to degenerate and std features
        log1p_features = set(feature_analysis["degenerate"] + feature_analysis["std_features"])
        
        for col in log1p_features:
            if col in data_transformed.columns and col != "Gender":  # Skip Gender from transformations
                # Ensure no negative values before log1p
                min_val = data_transformed[col].min()
                if min_val < 0:
                    # Shift to make all values positive
                    data_transformed[col] = data_transformed[col] - min_val
                
                # Apply log1p transformation
                data_transformed[col] = np.log1p(data_transformed[col])
                logger.debug(f"Applied log1p transform to {col}")
        
        # Clip outliers for high-outlier features
        for col in feature_analysis["high_outliers"]:
            if col in data_transformed.columns and col not in feature_analysis["missing_flags"] and col != "Gender":
                q99 = data_transformed[col].quantile(0.99)
                original_max = data_transformed[col].max()
                data_transformed[col] = data_transformed[col].clip(upper=q99)
                logger.debug(f"Clipped outliers for {col}: max {original_max:.3f} -> {q99:.3f}")
        
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
        
        # Clip to clinical ranges (only for clinical and demographic features)
        clinical_and_demo_cols = clinical_numeric_cols + demographic_cols
        data = self.clip_to_clinical_ranges(data, clinical_and_demo_cols)
        
        # Fill missing values with 0 for clinical numeric features only
        for col in clinical_numeric_cols:
            data[col] = data[col].fillna(0)
        
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
        data = self.apply_feature_specific_transforms(data, feature_analysis)
        
        return data, scalable_cols, missing_cols, feature_analysis
    
    def fit_scalers(self, mimic_data, eicu_data):
        """Fit adaptive scalers on training data"""
        logger.info("Fitting adaptive scalers on training data...")
        
        # Prepare features with advanced analysis
        mimic_prepared, mimic_scalable, mimic_missing, mimic_analysis = self.prepare_features(mimic_data, "mimic")
        eicu_prepared, eicu_scalable, eicu_missing, eicu_analysis = self.prepare_features(eicu_data, "eicu")
        
        # Split data for scaler fitting
        mimic_train, _ = train_test_split(
            mimic_prepared, 
            test_size=0.3, 
            random_state=self.config["data"]["random_seed"]
        )
        eicu_train, _ = train_test_split(
            eicu_prepared, 
            test_size=0.3, 
            random_state=self.config["data"]["random_seed"]
        )
        
        # Determine which scaler to use based on feature analysis
        # Use RobustScaler for problematic features, StandardScaler for well-behaved ones
        problematic_features = set(
            mimic_analysis["degenerate"] + mimic_analysis["std_features"] + 
            mimic_analysis["high_outliers"] + eicu_analysis["degenerate"] + 
            eicu_analysis["std_features"] + eicu_analysis["high_outliers"]
        )
        
        robust_features = [col for col in mimic_scalable if col in problematic_features]
        standard_features = [col for col in mimic_scalable if col not in problematic_features]
        
        logger.info(f"Using RobustScaler for {len(robust_features)} problematic features")
        logger.info(f"Using StandardScaler for {len(standard_features)} well-behaved features")
        
        # Create scalers
        self.mimic_robust_scaler = RobustScaler() if robust_features else None
        self.mimic_standard_scaler = StandardScaler() if standard_features else None
        self.eicu_robust_scaler = RobustScaler() if robust_features else None
        self.eicu_standard_scaler = StandardScaler() if standard_features else None
        
        # Fit scalers
        if self.mimic_robust_scaler and robust_features:
            self.mimic_robust_scaler.fit(mimic_train[robust_features])
        if self.mimic_standard_scaler and standard_features:
            self.mimic_standard_scaler.fit(mimic_train[standard_features])
        if self.eicu_robust_scaler and robust_features:
            self.eicu_robust_scaler.fit(eicu_train[robust_features])
        if self.eicu_standard_scaler and standard_features:
            self.eicu_standard_scaler.fit(eicu_train[standard_features])
        
        # Store feature lists for later use
        self.robust_features = robust_features
        self.standard_features = standard_features
        
        # Save scalers and feature info
        scaler_info = {
            "robust_features": robust_features,
            "standard_features": standard_features,
            "mimic_analysis": mimic_analysis,
            "eicu_analysis": eicu_analysis
        }
        
        with open(self.scalers_dir / "mimic_robust_scaler.pkl", "wb") as f:
            pickle.dump(self.mimic_robust_scaler, f)
        with open(self.scalers_dir / "mimic_standard_scaler.pkl", "wb") as f:
            pickle.dump(self.mimic_standard_scaler, f)
        with open(self.scalers_dir / "eicu_robust_scaler.pkl", "wb") as f:
            pickle.dump(self.eicu_robust_scaler, f)
        with open(self.scalers_dir / "eicu_standard_scaler.pkl", "wb") as f:
            pickle.dump(self.eicu_standard_scaler, f)
        with open(self.scalers_dir / "scaler_info.json", "w") as f:
            # Convert sets to lists for JSON serialization
            scaler_info_json = {
                "robust_features": list(scaler_info["robust_features"]),
                "standard_features": list(scaler_info["standard_features"])
            }
            json.dump(scaler_info_json, f, indent=2)
        
        logger.info("Adaptive scalers fitted and saved")
    def transform_data(self, data, domain="mimic"):
        """Transform data using adaptive fitted scalers"""
        logger.info(f"Transforming {domain} data with adaptive scaling...")
        
        # Prepare features with advanced analysis
        data_prepared, scalable_cols, missing_cols, feature_analysis = self.prepare_features(data, domain)
        
        # Apply appropriate scalers
        data_scaled = data_prepared.copy()
        
        # Use robust scalers for problematic features
        robust_scaler = self.mimic_robust_scaler if domain == "mimic" else self.eicu_robust_scaler
        standard_scaler = self.mimic_standard_scaler if domain == "mimic" else self.eicu_standard_scaler
        
        # Apply robust scaling to problematic features
        if robust_scaler and self.robust_features:
            robust_cols_available = [col for col in self.robust_features if col in data_scaled.columns]
            if robust_cols_available:
                data_scaled[robust_cols_available] = robust_scaler.transform(data_scaled[robust_cols_available])
                logger.debug(f"Applied robust scaling to {len(robust_cols_available)} features")
        
        # Apply standard scaling to well-behaved features
        if standard_scaler and self.standard_features:
            standard_cols_available = [col for col in self.standard_features if col in data_scaled.columns]
            if standard_cols_available:
                data_scaled[standard_cols_available] = standard_scaler.transform(data_scaled[standard_cols_available])
                logger.debug(f"Applied standard scaling to {len(standard_cols_available)} features")
        
        # Missing flags remain unscaled (binary indicators)
        logger.info(f"Kept {len(missing_cols)} missing flags unscaled as binary indicators")
        
        return data_scaled, scalable_cols, missing_cols
        data_scaled = data_prepared.copy()
        data_scaled[numeric_cols] = scaler.transform(data_prepared[numeric_cols])
        
        return data_scaled, numeric_cols, missing_cols
    
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
            for suffix in ['_mean', '_min', '_max', '_std']:
                col_name = f"{feature}{suffix}"
                if col_name in mimic_data.columns:
                    feature_spec['numeric_features'].append(col_name)
            
            # Add missing feature columns
            missing_col = f"{feature}_missing"
            feature_spec['missing_features'].append(missing_col)
        
        # Add demographic features to numeric features (they don't have missing flags)
        feature_spec['numeric_features'].extend(self.demographic_names)
        
        # Save feature spec
        spec_path = self.output_dir / "feature_spec.json"
        with open(spec_path, 'w') as f:
            json.dump(feature_spec, f, indent=2)
        
        logger.info(f"Feature specification saved to {spec_path}")
        return feature_spec
    
    def preprocess(self):
        """Main preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Load raw data
        mimic_data, eicu_data = self.load_raw_data()
        
        # Fit scalers
        self.fit_scalers(mimic_data, eicu_data)
        
        # Transform data
        # Transform data with adaptive scaling
        mimic_transformed, mimic_scalable, mimic_missing = self.transform_data(mimic_data, "mimic")
        eicu_transformed, eicu_scalable, eicu_missing = self.transform_data(eicu_data, "eicu")
        
        # Split data
        mimic_splits = self.split_data(mimic_transformed, 'mimic')
        eicu_splits = self.split_data(eicu_transformed, 'eicu')
        
        # Save splits
        self.save_splits(mimic_splits, eicu_splits)
        
        # Create feature specification
        feature_spec = self.create_feature_spec(mimic_data, eicu_data)
        
        logger.info("Preprocessing completed successfully!")
        return feature_spec

def main():
    """Main function for preprocessing"""
    parser = argparse.ArgumentParser(description='Preprocess MIMIC and eICU data')
    parser.add_argument('--config', type=str, default='conf/config.yml', help='Path to config file')
    parser.add_argument('--fit', action='store_true', help='Fit scalers and preprocess data')
    
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
    else:
        logger.info("Use --fit flag to run preprocessing")

if __name__ == "__main__":
    main()
