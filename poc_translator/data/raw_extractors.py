#!/usr/bin/env python3
"""
Raw Data Extractor for MIMIC and eICU
Executes SQL queries and saves raw CSV files with 24-hour aggregated features.
"""

import pandas as pd
import yaml
import argparse
import sys
from pathlib import Path
import psycopg2
from sqlalchemy import create_engine, text
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def execute_query(engine, sql_file_path):
    """Execute SQL query and return results as DataFrame"""
    try:
        with open(sql_file_path, 'r') as f:
            sql_query = f.read()
        
        logger.info(f"Executing query from {sql_file_path}")
        logger.info(f"Query length: {len(sql_query)} characters")
        
        # Execute query in chunks to handle large datasets
        chunk_size = 10000
        chunks = []
        
        for chunk in pd.read_sql(text(sql_query), engine, chunksize=chunk_size):
            chunks.append(chunk)
            logger.info(f"Processed chunk with {len(chunk)} rows")
        
        if chunks:
            result = pd.concat(chunks, ignore_index=True)
            logger.info(f"Total rows extracted: {len(result)}")
            return result
        else:
            logger.warning("No data returned from query")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        raise

def extract_mimic_data(config):
    """Extract MIMIC data using generated SQL query"""
    logger.info("Extracting MIMIC data...")
    
    # Create database connection
    engine = create_db_connection(config['db']['mimic_conn'])
    
    # Execute query
    sql_path = Path(config['paths']['tmp_sql_dir']) / "mimic_query.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"MIMIC SQL query not found: {sql_path}")
    
    data = execute_query(engine, sql_path)
    
    # Save to CSV
    output_path = Path(config['paths']['output_dir']) / "data" / "mimic_raw_24h.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(output_path, index=False)
    logger.info(f"MIMIC data saved to {output_path}")
    
    return data

def extract_eicu_data(config):
    """Extract eICU data using generated SQL query"""
    logger.info("Extracting eICU data...")
    
    # Create database connection
    engine = create_db_connection(config['db']['eicu_conn'])
    
    # Execute query
    sql_path = Path(config['paths']['tmp_sql_dir']) / "eicu_query.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"eICU SQL query not found: {sql_path}")
    
    data = execute_query(engine, sql_path)
    
    # Save to CSV
    output_path = Path(config['paths']['output_dir']) / "data" / "eicu_raw_24h.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(output_path, index=False)
    logger.info(f"eICU data saved to {output_path}")
    
    return data

def validate_data(data, dataset_name):
    """Validate extracted data"""
    logger.info(f"Validating {dataset_name} data...")
    
    # Check basic structure
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Columns: {list(data.columns)}")
    
    # Check for required columns
    required_cols = ['icustay_id']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
    
    # Check for feature columns
    feature_cols = [col for col in data.columns if col.startswith('feature_')]
    logger.info(f"Found {len(feature_cols)} feature columns")
    
    # Check for missing values
    missing_counts = data.isnull().sum()
    logger.info(f"Missing values per column:")
    for col, count in missing_counts.items():
        if count > 0:
            logger.info(f"  {col}: {count} ({count/len(data)*100:.1f}%)")
    
    # Check data types
    logger.info(f"Data types:")
    for col, dtype in data.dtypes.items():
        logger.info(f"  {col}: {dtype}")
    
    return True

def create_sample_data():
    """Create sample data for testing when database is not available"""
    logger.info("Creating sample data for testing...")
    
    import numpy as np
    
    # Create sample MIMIC data
    n_samples = 1000
    n_features = 40
    
    mimic_data = {
        'icustay_id': range(1, n_samples + 1),
        'subject_id': np.random.randint(1, 10000, n_samples),
        'hadm_id': np.random.randint(1, 5000, n_samples)
    }
    
    # Add feature columns
    for i in range(n_features):
        mimic_data[f'feature_{i}_mean'] = np.random.normal(0, 1, n_samples)
        mimic_data[f'feature_{i}_min'] = np.random.normal(-1, 0.5, n_samples)
        mimic_data[f'feature_{i}_max'] = np.random.normal(1, 0.5, n_samples)
        mimic_data[f'feature_{i}_last'] = np.random.normal(0, 1, n_samples)
        mimic_data[f'feature_{i}_count'] = np.random.randint(1, 50, n_samples)
        mimic_data[f'feature_{i}_missing'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    mimic_df = pd.DataFrame(mimic_data)
    
    # Create sample eICU data
    eicu_data = {
        'icustay_id': range(1, n_samples + 1)
    }
    
    # Add feature columns
    for i in range(n_features):
        eicu_data[f'feature_{i}_mean'] = np.random.normal(0, 1, n_samples)
        eicu_data[f'feature_{i}_min'] = np.random.normal(-1, 0.5, n_samples)
        eicu_data[f'feature_{i}_max'] = np.random.normal(1, 0.5, n_samples)
        eicu_data[f'feature_{i}_last'] = np.random.normal(0, 1, n_samples)
        eicu_data[f'feature_{i}_count'] = np.random.randint(1, 50, n_samples)
        eicu_data[f'feature_{i}_missing'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    eicu_df = pd.DataFrame(eicu_data)
    
    return mimic_df, eicu_df

def main():
    """Main function to extract raw data"""
    parser = argparse.ArgumentParser(description='Extract raw data from MIMIC and eICU databases')
    parser.add_argument('--sample', action='store_true', help='Create sample data for testing')
    parser.add_argument('--mimic-only', action='store_true', help='Extract only MIMIC data')
    parser.add_argument('--eicu-only', action='store_true', help='Extract only eICU data')
    parser.add_argument('--validate', action='store_true', help='Validate extracted data')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    if args.sample:
        logger.info("Creating sample data...")
        mimic_data, eicu_data = create_sample_data()
        
        # Save sample data
        output_dir = Path(config['paths']['output_dir']) / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mimic_path = output_dir / "mimic_raw_24h.csv"
        eicu_path = output_dir / "eicu_raw_24h.csv"
        
        mimic_data.to_csv(mimic_path, index=False)
        eicu_data.to_csv(eicu_path, index=False)
        
        logger.info(f"Sample data saved to {mimic_path} and {eicu_path}")
        
        if args.validate:
            validate_data(mimic_data, "MIMIC")
            validate_data(eicu_data, "eICU")
        
        return
    
    try:
        # Extract MIMIC data
        if not args.eicu_only:
            mimic_data = extract_mimic_data(config)
            if args.validate:
                validate_data(mimic_data, "MIMIC")
        
        # Extract eICU data
        if not args.mimic_only:
            eicu_data = extract_eicu_data(config)
            if args.validate:
                validate_data(eicu_data, "eICU")
        
        logger.info("Data extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Data extraction failed: {e}")
        logger.info("You can use --sample flag to create sample data for testing")
        raise

if __name__ == "__main__":
    main()
