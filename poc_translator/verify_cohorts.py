#!/usr/bin/env python3
"""
Verify Cohort Tables Script
Checks access to BSI cohort tables and shows basic statistics.
"""

import yaml
import sys
from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_cohorts():
    """Verify cohort table access and show statistics"""
    try:
        # Load configuration
        config_path = Path(__file__).parent / "conf" / "config.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("=== COHORT TABLE VERIFICATION ===")
        
        # Test MIMIC cohort
        logger.info("Testing MIMIC cohort table...")
        mimic_engine = create_engine(config['db']['mimic_conn'])
        mimic_cohort = config['db']['cohort_tables']['mimic']
        
        with mimic_engine.connect() as conn:
            # Get basic stats
            result = conn.execute(f"SELECT COUNT(*) as total FROM {mimic_cohort}")
            total_count = result.fetchone()[0]
            
            # Get sample data structure
            result = conn.execute(f"SELECT * FROM {mimic_cohort} LIMIT 5")
            sample_data = result.fetchall()
            columns = result.keys()
            
            logger.info(f"✓ MIMIC cohort: {total_count} records")
            logger.info(f"  Columns: {list(columns)}")
            logger.info(f"  Sample data: {sample_data[:2]}")  # Show first 2 rows
        
        # Test eICU cohort
        logger.info("Testing eICU cohort table...")
        eicu_engine = create_engine(config['db']['eicu_conn'])
        eicu_cohort = config['db']['cohort_tables']['eicu']
        
        with eicu_engine.connect() as conn:
            # Get basic stats
            result = conn.execute(f"SELECT COUNT(*) as total FROM {eicu_cohort}")
            total_count = result.fetchone()[0]
            
            # Get sample data structure
            result = conn.execute(f"SELECT * FROM {eicu_cohort} LIMIT 5")
            sample_data = result.fetchall()
            columns = result.keys()
            
            logger.info(f"✓ eICU cohort: {total_count} records")
            logger.info(f"  Columns: {list(columns)}")
            logger.info(f"  Sample data: {sample_data[:2]}")  # Show first 2 rows
        
        # Test OMOP schema access for these patients
        logger.info("Testing OMOP data access for cohort patients...")
        with mimic_engine.connect() as conn:
            # Check if we can find measurements for cohort patients
            result = conn.execute(f"""
                SELECT COUNT(DISTINCT m.person_id) as patients_with_measurements
                FROM {config['db']['omop_schema']}.measurement m
                JOIN {mimic_cohort} c ON m.person_id = c.person_id
                LIMIT 1
            """)
            patients_with_data = result.fetchone()[0]
            logger.info(f"✓ MIMIC patients with OMOP measurements: {patients_with_data}")
        
        logger.info("🎉 Cohort verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cohort verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main verification function"""
    success = verify_cohorts()
    
    if success:
        logger.info("✅ Cohort tables are ready for use!")
        logger.info("\nNext steps:")
        logger.info("1. Run: python sql/make_queries.py")
        logger.info("2. Run: python data/raw_extractors.py")
        logger.info("3. Run: python src/preprocess.py --fit")
    else:
        logger.error("❌ Cohort verification failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

