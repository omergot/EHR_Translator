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
from sklearn.preprocessing import StandardScaler
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
    
    def prepare_features(self, data, domain='mimic'):
        """Prepare features for preprocessing"""
        logger.info(f"Preparing features for {domain}...")
        
        # Get feature columns
        feature_cols = self.get_feature_columns(data)
        
        # Separate numeric and demographic columns
        numeric_cols = [col for col in feature_cols if '_mean' in col or '_min' in col or '_max' in col or '_std' in col]
        demographic_cols = [col for col in feature_cols if col in self.demographic_names]
        
        # Combine all columns that should be processed
        all_processing_cols = numeric_cols + demographic_cols
        
        # Create missing flags for clinical features (not demographics)
        data = self.create_missing_flags(data, feature_cols)
        missing_cols = [col for col in data.columns if '_missing' in col]
        
        # Clip to clinical ranges
        data = self.clip_to_clinical_ranges(data, all_processing_cols)
        
        # Fill missing values with 0 for numeric features
        for col in numeric_cols:
            data[col] = data[col].fillna(0)
        
        # Fill missing values for demographics with appropriate defaults
        for col in demographic_cols:
            if col == 'Age':
                data[col] = data[col].fillna(data[col].median())  # Use median age
            elif col == 'Gender':
                data[col] = data[col].fillna(0)  # Default to 0 for missing gender
        
        # Ensure missing flags are binary
        for col in missing_cols:
            data[col] = data[col].astype(int)
        
        return data, numeric_cols, missing_cols
    
    def fit_scalers(self, mimic_data, eicu_data):
        """Fit scalers on training data only"""
        logger.info("Fitting scalers on training data...")
        
        # Prepare features
        mimic_prepared, mimic_numeric, mimic_missing = self.prepare_features(mimic_data, 'mimic')
        eicu_prepared, eicu_numeric, eicu_missing = self.prepare_features(eicu_data, 'eicu')
        
        # Split data for scaler fitting
        mimic_train, _ = train_test_split(
            mimic_prepared, 
            test_size=0.3, 
            random_state=self.config['data']['random_seed']
        )
        eicu_train, _ = train_test_split(
            eicu_prepared, 
            test_size=0.3, 
            random_state=self.config['data']['random_seed']
        )
        
        # Fit MIMIC scaler
        self.mimic_scaler = StandardScaler()
        self.mimic_scaler.fit(mimic_train[mimic_numeric])
        
        # Fit eICU scaler
        self.eicu_scaler = StandardScaler()
        self.eicu_scaler.fit(eicu_train[eicu_numeric])
        
        # Save scalers
        with open(self.scalers_dir / "mimic_scaler.pkl", 'wb') as f:
            pickle.dump(self.mimic_scaler, f)
        
        with open(self.scalers_dir / "eicu_scaler.pkl", 'wb') as f:
            pickle.dump(self.eicu_scaler, f)
        
        logger.info("Scalers fitted and saved")
    
    def transform_data(self, data, domain='mimic'):
        """Transform data using fitted scalers"""
        logger.info(f"Transforming {domain} data...")
        
        # Prepare features
        data_prepared, numeric_cols, missing_cols = self.prepare_features(data, domain)
        
        # Get appropriate scaler
        scaler = self.mimic_scaler if domain == 'mimic' else self.eicu_scaler
        
        # Scale numeric features
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
        mimic_transformed, mimic_numeric, mimic_missing = self.transform_data(mimic_data, 'mimic')
        eicu_transformed, eicu_numeric, eicu_missing = self.transform_data(eicu_data, 'eicu')
        
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
