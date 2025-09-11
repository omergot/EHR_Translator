#!/usr/bin/env python3
"""
Feature Extraction Script for eICU and MIMIC-IV
Extracts numerical measurements from both databases with configurable sampling strategies.
Supports balanced sampling for faster execution and better dataset balance.
"""

import pandas as pd
import psycopg2
import yaml
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def load_config():
    """Load configuration from config.yml"""
    config_path = Path(__file__).parent.parent / "conf" / "config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_db_connection(conn_string):
    """Get database connection"""
    try:
        conn = psycopg2.connect(conn_string)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def create_mimic_sampling_cohort_sql(sample_size=10000, min_stay_hours=6):
    """Create MIMIC-IV sampling cohort with quality filters"""
    return f"""
    CREATE TABLE IF NOT EXISTS {{schema_name}}.{{cohort_table_name}} AS
    SELECT 
        stay_id as example_id,
        subject_id as person_id,
        intime as start_datetime,
        intime::date as start_date,
        outtime as end_datetime,
        outtime::date as end_date,
        0 as y
    FROM mimiciv_icu.icustays i
    WHERE outtime > intime
        AND intime IS NOT NULL
        AND outtime IS NOT NULL
        -- Quality filters
        AND EXTRACT(EPOCH FROM (outtime - intime))/3600 >= {min_stay_hours}  -- Minimum stay duration
        -- Ensure some vital signs data exists
        AND EXISTS (
            SELECT 1 FROM mimiciv_icu.chartevents ce 
            WHERE ce.stay_id = i.stay_id 
            AND ce.itemid IN (220045, 220210, 220277)  -- HR, RR, SpO2
            AND ce.valuenum IS NOT NULL
            LIMIT 1
        )
        -- Ensure demographic data exists
        AND EXISTS (
            SELECT 1 FROM mimiciv_hosp.patients p 
            WHERE p.subject_id = i.subject_id 
            AND p.anchor_age IS NOT NULL
            AND p.gender IS NOT NULL
        )
    ORDER BY RANDOM()
    LIMIT {sample_size};
    """

def create_eicu_sampling_cohort_sql(sample_size=10000, min_stay_hours=6):
    """Create eICU sampling cohort with quality filters"""
    return f"""
    CREATE TABLE IF NOT EXISTS {{schema_name}}.{{cohort_table_name}} AS
    SELECT 
        patientunitstayid as example_id,
        patienthealthsystemstayid as person_id,
        (date '2000-1-1' + (hospitaladmitoffset * interval '1 minutes')) as start_datetime,
        (date '2000-1-1' + (hospitaladmitoffset * interval '1 minutes'))::date as start_date,
        (date '2000-1-1' + (unitDischargeOffset * interval '1 minutes')) as end_datetime,
        (date '2000-1-1' + (unitDischargeOffset * interval '1 minutes'))::date as end_date,
        0 as y
    FROM eicu_crd.patient p
    WHERE unitDischargeOffset > hospitaladmitoffset
        AND hospitaladmitoffset IS NOT NULL
        AND unitDischargeOffset IS NOT NULL
        -- Quality filters
        AND (unitDischargeOffset - hospitaladmitoffset) >= {min_stay_hours * 60}  -- Minimum stay duration in minutes
        -- Ensure some vital signs data exists
        AND EXISTS (
            SELECT 1 FROM eicu_crd.vitalperiodic v 
            WHERE v.patientunitstayid = p.patientunitstayid 
            AND (v.heartrate IS NOT NULL OR v.respiration IS NOT NULL OR v.sao2 IS NOT NULL)
            LIMIT 1
        )
        -- Ensure demographic data exists
        AND p.age IS NOT NULL
        AND p.age != '> 89'
        AND p.gender IS NOT NULL
    ORDER BY RANDOM()
    LIMIT {sample_size};
    """

def create_simplified_eicu_features_sql():
    """Create simplified eICU features SQL without BSI constraints"""
    return """
    WITH lab_res AS (
        SELECT
            b.example_id,
            b.person_id,
            labname as feature_name,
            labresult::TEXT as feature_value,
            (date '2000-1-1' + (labresultoffset * interval '1 minutes')) as feature_start_date,
            labMeasureNameSystem as unit
        FROM eicu_crd.lab as a
        JOIN {cohort_table} as b
        ON (b.example_id = a.patientunitstayid)
        WHERE labname IS NOT NULL
            AND labresultoffset IS NOT NULL
            AND labresult IS NOT NULL
    ),
    nurse_res AS (
        SELECT
            b.example_id,
            b.person_id,
            nursingchartcelltypevalname as feature_name,
            CASE WHEN nursingchartvalue~E'^[0-9]+\\.?[0-9]*$' THEN nursingchartvalue ELSE NULL END as feature_value,
            (date '2000-1-1' + (nursingChartOffset * interval '1 minutes')) as feature_start_date,
            '' as unit
        FROM eicu_crd.nursecharting as a
        JOIN {cohort_table} as b
        ON (b.example_id = a.patientunitstayid)
        WHERE nursingchartcelltypevalname IS NOT NULL
            AND nursingChartOffset IS NOT NULL
            AND nursingchartvalue IS NOT NULL
            AND nursingchartvalue~E'^[0-9]+\\.?[0-9]*$'
    ),
    resp_res AS (
        SELECT
            b.example_id,
            b.person_id,
            respchartvaluelabel as feature_name,
            CASE WHEN respchartvalue~E'^[0-9]+\\.?[0-9]*$' THEN respchartvalue ELSE NULL END as feature_value,
            (date '2000-1-1' + (respChartOffset * interval '1 minutes')) as feature_start_date,
            '' as unit
        FROM eicu_crd.respiratorycharting as a
        JOIN {cohort_table} as b
        ON (b.example_id = a.patientunitstayid)
        WHERE respchartvaluelabel IS NOT NULL
            AND respChartOffset IS NOT NULL
            AND respchartvalue IS NOT NULL
            AND respchartvalue~E'^[0-9]+\\.?[0-9]*$'
    ),
    labothername_res AS (
        SELECT
            b.example_id,
            b.person_id,
            labothername as feature_name,
            CASE WHEN labotherresult~E'^[0-9]+\\.?[0-9]*$' THEN labotherresult ELSE NULL END as feature_value,
            (date '2000-1-1' + (labotheroffset * interval '1 minutes')) as feature_start_date,
            '' as unit
        FROM eicu_crd.customlab as a
        JOIN {cohort_table} as b
        ON (b.example_id = a.patientunitstayid)
        WHERE labothername IS NOT NULL
            AND labotheroffset IS NOT NULL
            AND labotherresult IS NOT NULL
            AND labotherresult~E'^[0-9]+\\.?[0-9]*$'
    ),
    person_demographics AS (
        SELECT 
            b.example_id,
            a.patienthealthsystemstayid as person_id, 
            'age' as feature_name,
            age::text as feature_value,
            NULL::timestamp as feature_start_date,
            '' as unit
        FROM eicu_crd.patient a
        JOIN {cohort_table} b
        ON (b.example_id = a.patientunitstayid)
        WHERE age IS NOT NULL
            AND age != '> 89'  -- Exclude non-numeric age values
    ),
    measurements AS (
        SELECT * FROM lab_res WHERE feature_value IS NOT NULL
        UNION ALL
        SELECT * FROM nurse_res WHERE feature_value IS NOT NULL
        UNION ALL
        SELECT * FROM resp_res WHERE feature_value IS NOT NULL
        UNION ALL
        SELECT * FROM labothername_res WHERE feature_value IS NOT NULL
        UNION ALL
        SELECT * FROM person_demographics
    )
    SELECT 
        example_id,
        person_id,
        feature_name,
        feature_value,
        feature_start_date
    FROM measurements
    -- No date filtering needed since we're joining with cohort table
    ORDER BY example_id, feature_start_date
    """

def create_simplified_mimic_features_sql():
    """Create simplified MIMIC-IV features SQL using existing cohort table"""
    return """
    WITH measurements_raw AS (
        SELECT
            b.example_id, 
            b.person_id,
            label as feature_name,
            valuenum::text as feature_value,
            charttime::timestamp without time zone as feature_start_date
        FROM mimiciv_hosp.labevents le
        JOIN mimiciv_icu.icustays i
        USING (subject_id, hadm_id)
        JOIN {cohort_table} b
        ON (b.example_id = i.stay_id)
        JOIN mimiciv_hosp.d_labitems
        USING (itemid)
        WHERE charttime IS NOT NULL
            AND valuenum IS NOT NULL

        UNION ALL

        SELECT
            b.example_id, 
            b.person_id,
            result_name as feature_name,
            result_value::text as feature_value,
            chartdate::timestamp without time zone as feature_start_date
        FROM mimiciv_hosp.omr o
        JOIN {cohort_table} b
        ON (o.subject_id = b.person_id)
        WHERE chartdate IS NOT NULL
            AND result_value IS NOT NULL
            AND result_name IN ('Weight (Lbs)', 'Height (Inches)')

        UNION ALL

        SELECT
            b.example_id, 
            b.person_id,
            label as feature_name,
            valuenum::text as feature_value,
            charttime::timestamp without time zone as feature_start_date
        FROM mimiciv_icu.chartevents ce
        JOIN {cohort_table} b
        ON (b.example_id = ce.stay_id)
        JOIN mimiciv_icu.d_items
        USING (itemid)
        WHERE charttime IS NOT NULL
            AND valuenum IS NOT NULL
    ), 
    person_demographics AS (
        SELECT 
            b.example_id,
            b.person_id, 
            'age' as feature_name,
            anchor_age::text as feature_value,
            NULL::timestamp as feature_start_date
        FROM mimiciv_hosp.patients a
        JOIN {cohort_table} b
        ON (a.subject_id = b.person_id)
    )
    
    SELECT 
        example_id,
        person_id,
        feature_name,
        feature_value,
        feature_start_date
    FROM measurements_raw
    WHERE feature_value IS NOT NULL
        AND feature_value != ''
        AND feature_value !~ '[a-zA-Z]'  -- Exclude values with letters
    
    UNION ALL
    
    SELECT 
        example_id,
        person_id,
        feature_name,
        feature_value::text,
        feature_start_date
    FROM person_demographics
    
    ORDER BY example_id, feature_start_date
    """

def create_sampling_cohorts(config, sample_size, min_stay_hours):
    """Create balanced sampling cohorts if they don't exist"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Schema and table names for sampling cohorts
    sampling_schemas = {
        'mimic': f"mimic_generic_sample_{sample_size}_{timestamp}",
        'eicu': f"eicu_generic_sample_{sample_size}_{timestamp}"
    }
    
    cohort_tables = {
        'mimic': f"{sampling_schemas['mimic']}.generic_cohort",
        'eicu': f"{sampling_schemas['eicu']}.generic_cohort"
    }
    
    logger.info(f"Creating sampling cohorts with {sample_size} stays each...")
    
    # Create MIMIC sampling cohort
    mimic_conn = get_db_connection(config['db']['mimic_conn'])
    if mimic_conn:
        try:
            with mimic_conn.cursor() as cur:
                # Create schema
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {sampling_schemas['mimic']}")
                mimic_conn.commit()
                
                # Create cohort table
                mimic_cohort_sql = create_mimic_sampling_cohort_sql(sample_size, min_stay_hours).format(
                    schema_name=sampling_schemas['mimic'],
                    cohort_table_name="generic_cohort"
                )
                cur.execute(mimic_cohort_sql)
                mimic_conn.commit()
                
                # Check created size
                cur.execute(f"SELECT COUNT(*) FROM {cohort_tables['mimic']}")
                mimic_count = cur.fetchone()[0]
                logger.info(f"Created MIMIC sampling cohort: {mimic_count} stays")
                
        except Exception as e:
            logger.error(f"Error creating MIMIC sampling cohort: {e}")
            raise
        finally:
            mimic_conn.close()
    
    # Create eICU sampling cohort
    eicu_conn = get_db_connection(config['db']['eicu_conn'])
    if eicu_conn:
        try:
            with eicu_conn.cursor() as cur:
                # Create schema
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {sampling_schemas['eicu']}")
                eicu_conn.commit()
                
                # Create cohort table
                eicu_cohort_sql = create_eicu_sampling_cohort_sql(sample_size, min_stay_hours).format(
                    schema_name=sampling_schemas['eicu'],
                    cohort_table_name="generic_cohort"
                )
                cur.execute(eicu_cohort_sql)
                eicu_conn.commit()
                
                # Check created size
                cur.execute(f"SELECT COUNT(*) FROM {cohort_tables['eicu']}")
                eicu_count = cur.fetchone()[0]
                logger.info(f"Created eICU sampling cohort: {eicu_count} stays")
                
        except Exception as e:
            logger.error(f"Error creating eICU sampling cohort: {e}")
            raise
        finally:
            eicu_conn.close()
    
    return cohort_tables, sampling_schemas

def execute_sql_queries(config, use_sampling=False, sample_size=10000, min_stay_hours=6):
    """Execute SQL queries for both databases with optional sampling"""
    
    # Database connections
    mimic_conn = get_db_connection(config['db']['mimic_conn'])
    eicu_conn = get_db_connection(config['db']['eicu_conn'])
    
    if not mimic_conn or not eicu_conn:
        print("Failed to establish database connections")
        return
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if use_sampling:
            # Create new sampling cohorts
            cohort_tables, sampling_schemas = create_sampling_cohorts(config, sample_size, min_stay_hours)
            eicu_cohort_table = cohort_tables['eicu']
            mimic_cohort_table = cohort_tables['mimic']
            
            logger.info(f"Using sampling approach:")
            logger.info(f"  Sample size: {sample_size} stays per database")
            logger.info(f"  Minimum stay duration: {min_stay_hours} hours")
            logger.info(f"  eICU cohort table: {eicu_cohort_table}")
            logger.info(f"  MIMIC cohort table: {mimic_cohort_table}")
        else:
            # Use existing cohort tables from config
            eicu_cohort_table = config['db']['cohort_tables']['eicu']
            mimic_cohort_table = config['db']['cohort_tables']['mimic']
            sampling_schemas = None
            
            logger.info(f"Using existing cohort tables:")
            logger.info(f"  eICU cohort table: {eicu_cohort_table}")
            logger.info(f"  MIMIC cohort table: {mimic_cohort_table}")
        
        # Execute eICU queries
        logger.info("Extracting eICU features...")
        eicu_features_sql = create_simplified_eicu_features_sql().format(
            cohort_table=eicu_cohort_table
        )
        
        eicu_features_df = pd.read_sql_query(eicu_features_sql, eicu_conn)
        logger.info(f"Extracted {len(eicu_features_df)} eICU feature records")
        
        # Execute MIMIC queries
        logger.info("Extracting MIMIC features...")
        mimic_features_sql = create_simplified_mimic_features_sql().format(
            cohort_table=mimic_cohort_table
        )
        
        mimic_features_df = pd.read_sql_query(mimic_features_sql, mimic_conn)
        logger.info(f"Extracted {len(mimic_features_df)} MIMIC feature records")
        
        # Save results
        output_dir = Path(config['paths']['output_dir']) / "data"
        output_dir.mkdir(exist_ok=True)
        
        eicu_output = output_dir / f"eicu_all_features_{timestamp}.csv"
        mimic_output = output_dir / f"mimic_all_features_{timestamp}.csv"
        
        eicu_features_df.to_csv(eicu_output, index=False)
        mimic_features_df.to_csv(mimic_output, index=False)
        
        logger.info(f"Saved eICU features to: {eicu_output}")
        logger.info(f"Saved MIMIC features to: {mimic_output}")
        
        # Print feature summaries
        logger.info("\neICU Feature Summary:")
        logger.info(f"Unique features: {eicu_features_df['feature_name'].nunique()}")
        logger.info(f"Unique patients: {eicu_features_df['example_id'].nunique()}")
        logger.info("Top 10 features by count:")
        for feature, count in eicu_features_df['feature_name'].value_counts().head(10).items():
            logger.info(f"  {feature}: {count}")
        
        logger.info("\nMIMIC Feature Summary:")
        logger.info(f"Unique features: {mimic_features_df['feature_name'].nunique()}")
        logger.info(f"Unique patients: {mimic_features_df['example_id'].nunique()}")
        logger.info("Top 10 features by count:")
        for feature, count in mimic_features_df['feature_name'].value_counts().head(10).items():
            logger.info(f"  {feature}: {count}")
        
        # Cleanup sampling schemas if created
        if use_sampling and sampling_schemas:
            logger.info("\nCleaning up sampling schemas...")
            
            # Cleanup MIMIC sampling schema
            mimic_conn = get_db_connection(config['db']['mimic_conn'])
            if mimic_conn:
                try:
                    with mimic_conn.cursor() as cur:
                        cur.execute(f"DROP SCHEMA {sampling_schemas['mimic']} CASCADE")
                        mimic_conn.commit()
                except Exception as e:
                    logger.warning(f"Error cleaning up MIMIC sampling schema: {e}")
                finally:
                    mimic_conn.close()
            
            # Cleanup eICU sampling schema
            eicu_conn = get_db_connection(config['db']['eicu_conn'])
            if eicu_conn:
                try:
                    with eicu_conn.cursor() as cur:
                        cur.execute(f"DROP SCHEMA {sampling_schemas['eicu']} CASCADE")
                        eicu_conn.commit()
                except Exception as e:
                    logger.warning(f"Error cleaning up eICU sampling schema: {e}")
                finally:
                    eicu_conn.close()
            
            logger.info("Sampling schemas cleaned up successfully")
        
        logger.info("\nFeature extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        if eicu_conn:
            eicu_conn.close()
        if mimic_conn:
            mimic_conn.close()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Extract features from MIMIC-IV and eICU databases')
    parser.add_argument('--use-sampling', action='store_true', 
                       help='Use balanced sampling instead of existing cohort tables')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Number of ICU stays to sample from each database (default: 10000)')
    parser.add_argument('--min-stay-hours', type=int, default=6,
                       help='Minimum ICU stay duration in hours (default: 6)')
    parser.add_argument('--use-existing', action='store_true',
                       help='Use existing BSI cohort tables from config (default behavior)')
    
    args = parser.parse_args()
    
    # Default to existing cohorts unless sampling is explicitly requested
    use_sampling = args.use_sampling
    
    if use_sampling:
        logger.info(f"Starting balanced sampling feature extraction...")
        logger.info(f"Configuration: {args.sample_size} stays per DB, {args.min_stay_hours}h minimum stay")
    else:
        logger.info("Starting feature extraction using existing BSI cohort tables...")
    
    # Load configuration
    config = load_config()
    
    # Execute SQL queries
    execute_sql_queries(config, use_sampling, args.sample_size, args.min_stay_hours)
    
    logger.info("Feature extraction completed successfully!")

if __name__ == "__main__":
    main()
