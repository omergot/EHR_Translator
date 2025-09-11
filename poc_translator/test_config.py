#!/usr/bin/env python3
"""
Test Configuration Script
Tests database connections and configuration loading.
"""

import yaml
import sys
from pathlib import Path
from sqlalchemy import create_engine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config():
    """Test configuration loading and database connections"""
    try:
        # Load configuration
        config_path = Path(__file__).parent / "conf" / "config.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("✓ Configuration loaded successfully")
        logger.info(f"OMOP Schema: {config['db']['omop_schema']}")
        
        # Test MIMIC connection
        logger.info("Testing MIMIC database connection...")
        mimic_engine = create_engine(config['db']['mimic_conn'])
        
        # Test simple query
        with mimic_engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            logger.info("✓ MIMIC database connection successful")
        
        # Test eICU connection (if available)
        logger.info("Testing eICU database connection...")
        eicu_engine = create_engine(config['db']['eicu_conn'])
        
        with eicu_engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            logger.info("✓ eICU database connection successful")
        
        # Test OMOP schema access
        logger.info(f"Testing OMOP schema access: {config['db']['omop_schema']}")
        with mimic_engine.connect() as conn:
            # Test if OMOP tables exist
            result = conn.execute(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{config['db']['omop_schema']}'
                AND table_name IN ('visit_occurrence', 'measurement', 'observation', 'concept')
                LIMIT 5
            """)
            tables = [row[0] for row in result]
            logger.info(f"✓ Found OMOP tables: {tables}")
        
        # Test cohort table access
        logger.info("Testing cohort table access...")
        with mimic_engine.connect() as conn:
            # Test MIMIC cohort table
            mimic_cohort = config['db']['cohort_tables']['mimic']
            result = conn.execute(f"SELECT COUNT(*) as count FROM {mimic_cohort}")
            count = result.fetchone()[0]
            logger.info(f"✓ MIMIC cohort table accessible: {count} records")
        
        with eicu_engine.connect() as conn:
            # Test eICU cohort table
            eicu_cohort = config['db']['cohort_tables']['eicu']
            result = conn.execute(f"SELECT COUNT(*) as count FROM {eicu_cohort}")
            count = result.fetchone()[0]
            logger.info(f"✓ eICU cohort table accessible: {count} records")
        
        logger.info("🎉 All configuration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("=== CONFIGURATION TEST ===")
    
    success = test_config()
    
    if success:
        logger.info("✅ Configuration is ready for use!")
        logger.info("\nNext steps:")
        logger.info("1. Run: python sql/make_queries.py")
        logger.info("2. Run: python data/raw_extractors.py")
        logger.info("3. Run: python src/preprocess.py --fit")
    else:
        logger.error("❌ Configuration needs to be fixed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
