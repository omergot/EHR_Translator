#!/usr/bin/env python3
"""
POC Feature Extractor for MIMIC-IV and eICU-CRD Domain Translation Project

This script extracts aligned patient data from MIMIC-IV and eICU-CRD for domain translation.
Focuses on 10 specific aligned features: Heart Rate, Respiratory Rate, SpO₂, Temperature, 
MAP, WBC, Sodium, Creatinine, Age, Gender.

Author: POC Translation Project
Date: January 2025
"""

import pandas as pd
import numpy as np
import yaml
import argparse
import sys
import os
from pathlib import Path
import logging
from tqdm import tqdm
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the 10 aligned features for this POC
POC_FEATURES = {
    'Heart Rate': 0,
    'Respiratory Rate': 1, 
    'SpO₂': 2,
    'Temperature': 3,
    'MAP': 4,  # Mean Arterial Pressure
    'WBC': 5,  # White Blood Cell Count
    'Sodium': 6,
    'Creatinine': 7,
    'Age': 8,
    'Gender': 9
}

# MIMIC-IV ItemID mappings for the 10 POC features
MIMIC_ITEMID_MAP = {
    'Heart Rate': [220045],  # Heart Rate
    'Respiratory Rate': [220210, 224690],  # Respiratory Rate, Respiratory Rate (Total)
    'SpO₂': [220277],  # SpO2
    'Temperature': [223761, 223762],  # Temperature Celsius, Temperature Fahrenheit
    'MAP': [220181, 220052],  # Non Invasive Blood Pressure mean, Arterial Blood Pressure mean
    'WBC': [51300, 51301],  # WBC count
    'Sodium': [50983],  # Sodium
    'Creatinine': [50912],  # Creatinine
    'Age': None,  # Calculated from DOB
    'Gender': None  # From patients table
}

# eICU-CRD column mappings for the 10 POC features
EICU_COLUMN_MAP = {
    'Heart Rate': 'heartrate',
    'Respiratory Rate': 'respiration', 
    'SpO₂': 'sao2',
    'Temperature': 'temperature',
    'MAP': 'systemicmean',  # Mean arterial pressure
    'WBC': None,  # From lab table
    'Sodium': None,  # From lab table  
    'Creatinine': None,  # From lab table
    'Age': None,  # Calculated from patient table
    'Gender': None  # From patient table
}

def load_config():
    """Load configuration from config.yml"""
    config_path = Path(__file__).parent.parent / "conf" / "config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_db_connection(connection_string):
    """Create database connection using SQLAlchemy"""
    try:
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        logger.error(f"Failed to create database connection: {e}")
        raise

def get_mimic_demographic_data(engine, config):
    """Extract demographic data (age, gender) from MIMIC-IV using cohort table"""
    cohort_table = config['db']['cohort_tables']['mimic']
    
    query = f"""
    SELECT 
        i.stay_id as icustay_id,
        i.subject_id,
        p.gender,
        p.anchor_age as age
    FROM mimiciv_icu.icustays i
    JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
    JOIN {cohort_table} c ON i.stay_id = c.example_id
    WHERE p.anchor_age IS NOT NULL
    """
    
    logger.info(f"Extracting MIMIC demographic data using cohort: {cohort_table}...")
    return pd.read_sql(text(query), engine)

def get_eicu_demographic_data(engine, config):
    """Extract demographic data (age, gender) from eICU-CRD using cohort table"""
    cohort_table = config['db']['cohort_tables']['eicu']
    
    query = f"""
    SELECT 
        p.patientunitstayid as icustay_id,
        CASE 
            WHEN p.age = '> 89' THEN 91.0
            WHEN p.age ~ '^[0-9]+$' THEN p.age::float
            ELSE NULL 
        END as age,
        p.gender
    FROM eicu_crd.patient p
    JOIN {cohort_table} c ON p.patientunitstayid = c.example_id
    WHERE p.patientunitstayid IS NOT NULL
    """
    
    logger.info(f"Extracting eICU demographic data using cohort: {cohort_table}...")
    return pd.read_sql(text(query), engine)

def extract_mimic_data(engine, config):
    """Extract MIMIC-IV data for the 10 POC features"""
    logger.info("Starting MIMIC-IV data extraction...")
    
    # Get demographic data first
    demo_data = get_mimic_demographic_data(engine, config)
    logger.info(f"Found {len(demo_data)} MIMIC ICU stays with demographic data")
    
    # Initialize result DataFrame
    result_data = []
    
    # Process each ICU stay
    icustay_ids = demo_data['icustay_id'].unique()
    
    with tqdm(total=len(icustay_ids), desc="Processing MIMIC ICU stays") as pbar:
        for icustay_id in icustay_ids:
            try:
                stay_data = demo_data[demo_data['icustay_id'] == icustay_id].iloc[0]
                
                # Get vital signs and lab values for this stay
                vitals_query = """
                SELECT 
                    ce.itemid,
                    ce.valuenum,
                    ce.charttime
                FROM mimiciv_icu.chartevents ce
                JOIN mimiciv_icu.icustays i ON ce.stay_id = i.stay_id
                WHERE ce.stay_id = :icustay_id
                    AND ce.charttime >= i.intime
                    AND ce.charttime <= i.intime + INTERVAL '24 hours'
                    AND ce.itemid = ANY(:itemids)
                    AND ce.valuenum IS NOT NULL
                    AND ce.valuenum > 0
                """
                
                labs_query = """
                SELECT 
                    le.itemid,
                    le.valuenum,
                    le.charttime
                FROM mimiciv_hosp.labevents le
                JOIN mimiciv_icu.icustays i ON le.subject_id = i.subject_id AND le.hadm_id = i.hadm_id
                WHERE i.stay_id = :icustay_id
                    AND le.charttime >= i.intime
                    AND le.charttime <= i.intime + INTERVAL '24 hours'
                    AND le.itemid = ANY(:itemids)
                    AND le.valuenum IS NOT NULL
                    AND le.valuenum > 0
                """
                
                # Collect all itemids for vitals and labs
                vital_itemids = []
                lab_itemids = []
                
                for feature, itemids in MIMIC_ITEMID_MAP.items():
                    if itemids and feature not in ['Age', 'Gender']:
                        if feature in ['Heart Rate', 'Respiratory Rate', 'SpO₂', 'Temperature', 'MAP']:
                            vital_itemids.extend(itemids)
                        else:
                            lab_itemids.extend(itemids)
                
                # Get vital signs data
                vitals_df = pd.DataFrame()
                if vital_itemids:
                    vitals_df = pd.read_sql(text(vitals_query), engine, params={
                        'icustay_id': int(icustay_id),
                        'itemids': vital_itemids
                    })
                
                # Get lab data  
                labs_df = pd.DataFrame()
                if lab_itemids:
                    labs_df = pd.read_sql(text(labs_query), engine, params={
                        'icustay_id': int(icustay_id),
                        'itemids': lab_itemids
                    })
                
                # Combine all measurements
                all_measurements = pd.concat([vitals_df, labs_df], ignore_index=True)
                
                # Build feature record
                feature_record = {
                    'patient_id': stay_data['subject_id'],
                    'icu_stay_id': icustay_id
                }
                
                # Process each POC feature
                for feature_name, feature_idx in POC_FEATURES.items():
                    if feature_name == 'Age':
                        feature_record[f'{feature_name}'] = stay_data['age']
                    elif feature_name == 'Gender':
                        # Convert gender to numeric (0=F, 1=M)
                        feature_record[f'{feature_name}'] = 1 if stay_data['gender'] == 'M' else 0
                    else:
                        # Get measurements for this feature
                        itemids = MIMIC_ITEMID_MAP[feature_name]
                        if itemids:
                            feature_measurements = all_measurements[all_measurements['itemid'].isin(itemids)]
                            
                            if not feature_measurements.empty:
                                values = feature_measurements['valuenum'].dropna()
                                if len(values) > 0:
                                    feature_record[f'{feature_name}_min'] = values.min()
                                    feature_record[f'{feature_name}_max'] = values.max() 
                                    feature_record[f'{feature_name}_mean'] = values.mean()
                                    feature_record[f'{feature_name}_std'] = values.std() if len(values) > 1 else 0.0
                                else:
                                    feature_record[f'{feature_name}_min'] = np.nan
                                    feature_record[f'{feature_name}_max'] = np.nan
                                    feature_record[f'{feature_name}_mean'] = np.nan
                                    feature_record[f'{feature_name}_std'] = np.nan
                            else:
                                feature_record[f'{feature_name}_min'] = np.nan
                                feature_record[f'{feature_name}_max'] = np.nan
                                feature_record[f'{feature_name}_mean'] = np.nan
                                feature_record[f'{feature_name}_std'] = np.nan
                        else:
                            feature_record[f'{feature_name}_min'] = np.nan
                            feature_record[f'{feature_name}_max'] = np.nan
                            feature_record[f'{feature_name}_mean'] = np.nan
                            feature_record[f'{feature_name}_std'] = np.nan
                
                result_data.append(feature_record)
                
            except Exception as e:
                logger.warning(f"Error processing MIMIC ICU stay {icustay_id}: {e}")
                continue
            
            pbar.update(1)
    
    result_df = pd.DataFrame(result_data)
    logger.info(f"Extracted MIMIC data for {len(result_df)} ICU stays")
    return result_df

def extract_eicu_data(engine, config):
    """Extract eICU-CRD data for the 10 POC features"""
    logger.info("Starting eICU-CRD data extraction...")
    
    # Get demographic data first
    demo_data = get_eicu_demographic_data(engine, config)
    logger.info(f"Found {len(demo_data)} eICU ICU stays with demographic data")
    
    # Initialize result DataFrame
    result_data = []
    
    # Process each ICU stay
    icustay_ids = demo_data['icustay_id'].unique()
    
    with tqdm(total=len(icustay_ids), desc="Processing eICU ICU stays") as pbar:
        for icustay_id in icustay_ids:
            try:
                stay_data = demo_data[demo_data['icustay_id'] == icustay_id].iloc[0]
                
                # Get vital signs data (first 24 hours)
                vitals_query = """
                SELECT *
                FROM eicu_crd.vitalperiodic
                WHERE patientunitstayid = :icustay_id
                    AND observationoffset >= 0
                    AND observationoffset <= 1440  -- 24 hours in minutes
                """
                
                vitals_df = pd.read_sql(text(vitals_query), engine, params={
                    'icustay_id': int(icustay_id)
                })
                
                # Get lab data (first 24 hours)
                lab_query = """
                SELECT labname, labresult, labresultoffset
                FROM eicu_crd.lab
                WHERE patientunitstayid = :icustay_id
                    AND labresultoffset >= 0
                    AND labresultoffset <= 1440  -- 24 hours in minutes
                    AND labname IN ('WBC x 1000', 'sodium', 'creatinine', 'Sodium', 'Creatinine')
                    AND labresult IS NOT NULL
                """
                
                lab_df = pd.read_sql(text(lab_query), engine, params={
                    'icustay_id': int(icustay_id)
                })
                
                # Build feature record
                feature_record = {
                    'patient_id': icustay_id,  # eICU doesn't have separate patient ID
                    'icu_stay_id': icustay_id
                }
                
                # Process each POC feature
                for feature_name, feature_idx in POC_FEATURES.items():
                    if feature_name == 'Age':
                        feature_record[f'{feature_name}'] = stay_data['age']
                    elif feature_name == 'Gender':
                        # Convert gender to numeric (0=F, 1=M)
                        gender_val = stay_data['gender']
                        if gender_val == 'Male':
                            feature_record[f'{feature_name}'] = 1
                        elif gender_val == 'Female':
                            feature_record[f'{feature_name}'] = 0
                        else:
                            feature_record[f'{feature_name}'] = np.nan
                    elif feature_name in ['WBC', 'Sodium', 'Creatinine']:
                        # Lab values
                        lab_names = {
                            'WBC': ['WBC x 1000'],
                            'Sodium': ['sodium', 'Sodium'],
                            'Creatinine': ['creatinine', 'Creatinine']
                        }
                        
                        feature_labs = lab_df[lab_df['labname'].isin(lab_names[feature_name])]
                        
                        if not feature_labs.empty:
                            try:
                                values = pd.to_numeric(feature_labs['labresult'], errors='coerce').dropna()
                                if len(values) > 0:
                                    feature_record[f'{feature_name}_min'] = values.min()
                                    feature_record[f'{feature_name}_max'] = values.max()
                                    feature_record[f'{feature_name}_mean'] = values.mean()
                                    feature_record[f'{feature_name}_std'] = values.std() if len(values) > 1 else 0.0
                                else:
                                    feature_record[f'{feature_name}_min'] = np.nan
                                    feature_record[f'{feature_name}_max'] = np.nan
                                    feature_record[f'{feature_name}_mean'] = np.nan
                                    feature_record[f'{feature_name}_std'] = np.nan
                            except:
                                feature_record[f'{feature_name}_min'] = np.nan
                                feature_record[f'{feature_name}_max'] = np.nan
                                feature_record[f'{feature_name}_mean'] = np.nan
                                feature_record[f'{feature_name}_std'] = np.nan
                        else:
                            feature_record[f'{feature_name}_min'] = np.nan
                            feature_record[f'{feature_name}_max'] = np.nan
                            feature_record[f'{feature_name}_mean'] = np.nan
                            feature_record[f'{feature_name}_std'] = np.nan
                    else:
                        # Vital signs
                        column_name = EICU_COLUMN_MAP[feature_name]
                        if column_name and column_name in vitals_df.columns:
                            values = pd.to_numeric(vitals_df[column_name], errors='coerce').dropna()
                            
                            # Filter out unrealistic values
                            if feature_name == 'Heart Rate':
                                values = values[(values >= 20) & (values <= 300)]
                            elif feature_name == 'Respiratory Rate':
                                values = values[(values >= 5) & (values <= 60)]
                            elif feature_name == 'SpO₂':
                                values = values[(values >= 50) & (values <= 100)]
                            elif feature_name == 'Temperature':
                                values = values[(values >= 25) & (values <= 45)]
                            elif feature_name == 'MAP':
                                values = values[(values >= 30) & (values <= 200)]
                            
                            if len(values) > 0:
                                feature_record[f'{feature_name}_min'] = values.min()
                                feature_record[f'{feature_name}_max'] = values.max()
                                feature_record[f'{feature_name}_mean'] = values.mean()
                                feature_record[f'{feature_name}_std'] = values.std() if len(values) > 1 else 0.0
                            else:
                                feature_record[f'{feature_name}_min'] = np.nan
                                feature_record[f'{feature_name}_max'] = np.nan
                                feature_record[f'{feature_name}_mean'] = np.nan
                                feature_record[f'{feature_name}_std'] = np.nan
                        else:
                            feature_record[f'{feature_name}_min'] = np.nan
                            feature_record[f'{feature_name}_max'] = np.nan
                            feature_record[f'{feature_name}_mean'] = np.nan
                            feature_record[f'{feature_name}_std'] = np.nan
                
                result_data.append(feature_record)
                
            except Exception as e:
                logger.warning(f"Error processing eICU ICU stay {icustay_id}: {e}")
                continue
            
            pbar.update(1)
    
    result_df = pd.DataFrame(result_data)
    logger.info(f"Extracted eICU data for {len(result_df)} ICU stays")
    return result_df

def standardize_column_names(df, dataset_name):
    """Standardize column names to match the required schema"""
    logger.info(f"Standardizing column names for {dataset_name}...")
    
    # Create mapping for consistent naming
    column_mapping = {}
    
    # Map existing columns to standardized names
    for feature_name in POC_FEATURES.keys():
        if feature_name in ['Age', 'Gender']:
            continue  # These don't have _min, _max, etc.
        
        # Create short names for the schema
        short_names = {
            'Heart Rate': 'HR',
            'Respiratory Rate': 'RR', 
            'SpO₂': 'SpO2',
            'Temperature': 'Temp',
            'MAP': 'MAP',
            'WBC': 'WBC',
            'Sodium': 'Na',
            'Creatinine': 'Creat'
        }
        
        short_name = short_names.get(feature_name, feature_name.replace(' ', ''))
        
        for suffix in ['_min', '_max', '_mean', '_std']:
            old_col = f'{feature_name}{suffix}'
            new_col = f'{short_name}{suffix}'
            if old_col in df.columns:
                column_mapping[old_col] = new_col
    
    # Rename columns
    df_renamed = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_columns = ['patient_id', 'icu_stay_id']
    for col in required_columns:
        if col not in df_renamed.columns:
            logger.warning(f"Missing required column: {col}")
    
    return df_renamed

def validate_output_data(df, dataset_name):
    """Validate the extracted data"""
    logger.info(f"Validating {dataset_name} data...")
    
    # Basic checks
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['patient_id', 'icu_stay_id', 'Age', 'Gender']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
    
    # Check feature columns
    expected_features = ['HR', 'RR', 'SpO2', 'Temp', 'MAP', 'WBC', 'Na', 'Creat']
    feature_cols = []
    for feature in expected_features:
        for suffix in ['_min', '_max', '_mean', '_std']:
            col = f'{feature}{suffix}'
            if col in df.columns:
                feature_cols.append(col)
    
    logger.info(f"Found {len(feature_cols)} feature columns")
    
    # Check missing values
    missing_counts = df.isnull().sum()
    logger.info("Missing values per column:")
    for col, count in missing_counts.items():
        if count > 0:
            pct = count / len(df) * 100
            logger.info(f"  {col}: {count} ({pct:.1f}%)")
    
    # Check value ranges for key features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logger.info("Value ranges for numeric columns:")
    for col in numeric_cols:
        if not df[col].isna().all():
            logger.info(f"  {col}: {df[col].min():.2f} - {df[col].max():.2f}")
    
    return True

def create_sample_data():
    """Create sample POC data for testing when database is not available"""
    logger.info("Creating sample POC data for testing...")
    
    import numpy as np
    np.random.seed(42)  # For reproducible results
    
    # Create sample MIMIC data
    n_samples = 100
    feature_names = ['HR', 'RR', 'SpO2', 'Temp', 'MAP', 'WBC', 'Na', 'Creat']
    
    mimic_data = {
        'patient_id': np.random.randint(1, 50000, n_samples),
        'icu_stay_id': range(1, n_samples + 1),
        'Age': np.random.normal(65, 15, n_samples).clip(18, 89),
        'Gender': np.random.choice([0, 1], n_samples)  # 0=F, 1=M
    }
    
    # Add feature columns with realistic ranges
    feature_ranges = {
        'HR': (40, 150, 80, 20),      # Heart Rate: min, max, mean, std
        'RR': (8, 40, 18, 6),         # Respiratory Rate
        'SpO2': (85, 100, 96, 3),     # SpO2
        'Temp': (35, 40, 37, 1),      # Temperature
        'MAP': (50, 120, 80, 15),     # Mean Arterial Pressure
        'WBC': (2, 25, 9, 4),         # White Blood Cell count
        'Na': (125, 155, 140, 5),     # Sodium
        'Creat': (0.5, 8.0, 1.2, 1.5) # Creatinine
    }
    
    for feature in feature_names:
        min_val, max_val, mean_val, std_val = feature_ranges[feature]
        
        # Generate realistic values for min, max, mean, std
        for suffix in ['_min', '_max', '_mean', '_std']:
            if suffix == '_min':
                mimic_data[f'{feature}{suffix}'] = np.random.normal(min_val + 5, std_val/2, n_samples).clip(min_val, max_val)
            elif suffix == '_max':
                mimic_data[f'{feature}{suffix}'] = np.random.normal(max_val - 10, std_val/2, n_samples).clip(min_val, max_val)
            elif suffix == '_mean':
                mimic_data[f'{feature}{suffix}'] = np.random.normal(mean_val, std_val, n_samples).clip(min_val, max_val)
            else:  # _std
                mimic_data[f'{feature}{suffix}'] = np.random.exponential(std_val/2, n_samples).clip(0, std_val*2)
    
    mimic_df = pd.DataFrame(mimic_data)
    
    # Create sample eICU data with different patient IDs
    eicu_data = {
        'patient_id': np.random.randint(50001, 100000, n_samples),
        'icu_stay_id': range(n_samples + 1, 2 * n_samples + 1),
        'Age': np.random.normal(62, 18, n_samples).clip(18, 89),
        'Gender': np.random.choice([0, 1], n_samples)
    }
    
    # Add same feature columns
    for feature in feature_names:
        min_val, max_val, mean_val, std_val = feature_ranges[feature]
        
        for suffix in ['_min', '_max', '_mean', '_std']:
            if suffix == '_min':
                eicu_data[f'{feature}{suffix}'] = np.random.normal(min_val + 5, std_val/2, n_samples).clip(min_val, max_val)
            elif suffix == '_max':
                eicu_data[f'{feature}{suffix}'] = np.random.normal(max_val - 10, std_val/2, n_samples).clip(min_val, max_val)
            elif suffix == '_mean':
                eicu_data[f'{feature}{suffix}'] = np.random.normal(mean_val, std_val, n_samples).clip(min_val, max_val)
            else:  # _std
                eicu_data[f'{feature}{suffix}'] = np.random.exponential(std_val/2, n_samples).clip(0, std_val*2)
    
    eicu_df = pd.DataFrame(eicu_data)
    
    return mimic_df, eicu_df

def main():
    """Main function to extract POC features"""
    parser = argparse.ArgumentParser(description='Extract POC features from MIMIC-IV and eICU-CRD')
    parser.add_argument('--mimic-only', action='store_true', help='Extract only MIMIC data')
    parser.add_argument('--eicu-only', action='store_true', help='Extract only eICU data')
    parser.add_argument('--validate', action='store_true', help='Validate extracted data', default=True)
    parser.add_argument('--sample-size', type=int, help='Limit number of ICU stays to process (for testing)')
    parser.add_argument('--sample', action='store_true', help='Create sample data for testing (no database required)')
    
    args = parser.parse_args()
    
    # Load configuration (skip if using sample data)
    if not args.sample:
        try:
            config = load_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return
    
    logger.info("=== POC Feature Extraction for Domain Translation ===")
    logger.info(f"Target features: {list(POC_FEATURES.keys())}")
    
    # Handle sample data creation
    if args.sample:
        logger.info("Creating sample data for testing...")
        mimic_data, eicu_data = create_sample_data()
        
        # Save sample data (use default paths when no config available)
        mimic_output_path = "/bigdata/omerg/Thesis/mimic_poc_features.csv"
        eicu_output_path = "/bigdata/omerg/Thesis/eicu_poc_features.csv"
        
        mimic_data.to_csv(mimic_output_path, index=False)
        eicu_data.to_csv(eicu_output_path, index=False)
        
        logger.info(f"Sample MIMIC data saved to: {mimic_output_path}")
        logger.info(f"Sample eICU data saved to: {eicu_output_path}")
        
        if args.validate:
            validate_output_data(mimic_data, "MIMIC Sample")
            validate_output_data(eicu_data, "eICU Sample")
        
        logger.info("=== Sample POC Feature Generation Complete ===")
        return
    
    try:
        # Extract MIMIC data
        if not args.eicu_only:
            logger.info("Connecting to MIMIC-IV database...")
            mimic_engine = create_db_connection(config['db']['mimic_conn'])
            
            mimic_data = extract_mimic_data(mimic_engine, config)
            
            if args.sample_size:
                mimic_data = mimic_data.head(args.sample_size)
                logger.info(f"Limited MIMIC data to {len(mimic_data)} samples")
            
            # Standardize column names
            mimic_data = standardize_column_names(mimic_data, "MIMIC")
            
            # Save MIMIC data
            mimic_output_path = os.path.join(config['paths']['output_dir'], "mimic_poc_features.csv")
            mimic_data.to_csv(mimic_output_path, index=False)
            logger.info(f"MIMIC data saved to: {mimic_output_path}")
            
            if args.validate:
                validate_output_data(mimic_data, "MIMIC")
        
        # Extract eICU data
        if not args.mimic_only:
            logger.info("Connecting to eICU-CRD database...")
            eicu_engine = create_db_connection(config['db']['eicu_conn'])
            
            eicu_data = extract_eicu_data(eicu_engine, config)
            
            if args.sample_size:
                eicu_data = eicu_data.head(args.sample_size)
                logger.info(f"Limited eICU data to {len(eicu_data)} samples")
            
            # Standardize column names  
            eicu_data = standardize_column_names(eicu_data, "eICU")
            
            # Save eICU data
            eicu_output_path = os.path.join(config['paths']['output_dir'], "eicu_poc_features.csv")
            eicu_data.to_csv(eicu_output_path, index=False)
            logger.info(f"eICU data saved to: {eicu_output_path}")
            
            if args.validate:
                validate_output_data(eicu_data, "eICU")
        
        logger.info("=== POC Feature Extraction Complete ===")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()
