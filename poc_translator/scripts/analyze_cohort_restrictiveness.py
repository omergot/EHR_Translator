#!/usr/bin/env python3
"""
Cohort Restrictiveness Analysis Script

This script analyzes the impact of using strict cohort tables versus public tables
for MIMIC-IV and eICU-CRD databases. It measures how many ICU stays are included
in the strict cohorts vs. the general population and identifies what's being excluded.
"""

import pandas as pd
import yaml
import sys
import psycopg2
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def load_config():
    """Load configuration from config.yml"""
    config_path = Path(__file__).parent.parent / "conf" / "config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_db_connection(config):
    """Get database connection using connection string"""
    # Use mimic connection string (both databases seem to be in same instance based on config)
    mimic_conn_str = config['db']['mimic_conn']
    conn = psycopg2.connect(mimic_conn_str)
    return conn

def run_query(conn, query, description):
    """Run a query and return results with description"""
    print(f"\n{description}")
    print("-" * 60)
    print(f"Query: {query}")
    
    try:
        df = pd.read_sql_query(query, conn)
        print(f"Results:\n{df}")
        return df
    except Exception as e:
        print(f"Error running query: {e}")
        return None

def analyze_mimic_cohorts(conn):
    """Analyze MIMIC-IV cohort restrictiveness"""
    print("\n" + "="*80)
    print("MIMIC-IV COHORT ANALYSIS")
    print("="*80)
    
    queries = [
        {
            'query': "SELECT COUNT(*) as strict_cohort_count FROM mimiciv_bsi_100_2h_test.__mimiciv_bsi_100_2h_cohort;",
            'description': "1. Strict BSI Cohort Count"
        },
        {
            'query': """SELECT COUNT(*) as adult_icu_stays_4h_plus 
                        FROM mimiciv_icu.icustays 
                        WHERE EXTRACT(epoch FROM (outtime - intime))/3600 >= 4 
                        AND hadm_id IS NOT NULL;""",
            'description': "2. All Adult ICU Stays (≥4h duration, with hospital admission)"
        },
        {
            'query': """SELECT COUNT(*) as all_icu_stays 
                        FROM mimiciv_icu.icustays;""",
            'description': "3. All ICU Stays (no filters)"
        },
        {
            'query': """SELECT COUNT(*) as stays_with_chartevents
                        FROM mimiciv_icu.icustays i
                        WHERE EXISTS (
                            SELECT 1 FROM mimiciv_icu.chartevents c 
                            WHERE c.stay_id = i.stay_id
                        );""",
            'description': "4. ICU Stays with Chart Events Data"
        },
        {
            'query': """SELECT COUNT(*) as stays_with_labevents
                        FROM mimiciv_icu.icustays i
                        WHERE EXISTS (
                            SELECT 1 FROM mimiciv_hosp.labevents l 
                            WHERE l.hadm_id = i.hadm_id
                        );""",
            'description': "5. ICU Stays with Lab Events Data"
        },
        {
            'query': """SELECT 
                            COUNT(*) as stays_24h_data,
                            COUNT(DISTINCT i.subject_id) as unique_patients
                        FROM mimiciv_icu.icustays i
                        WHERE EXTRACT(epoch FROM (outtime - intime))/3600 >= 24 
                        AND hadm_id IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM mimiciv_icu.chartevents c 
                            WHERE c.stay_id = i.stay_id
                            AND c.charttime >= i.intime
                            AND c.charttime < i.intime + INTERVAL '24 hours'
                        );""",
            'description': "6. ICU Stays ≥24h with Chart Data in First 24h"
        }
    ]
    
    results = {}
    for q in queries:
        result = run_query(conn, q['query'], q['description'])
        if result is not None:
            # Store the first column value
            key = q['description'].split('.')[1].strip() if '.' in q['description'] else q['description']
            results[key] = result.iloc[0, 0]
    
    return results

def analyze_eicu_cohorts(conn):
    """Analyze eICU-CRD cohort restrictiveness"""
    print("\n" + "="*80)
    print("eICU-CRD COHORT ANALYSIS")
    print("="*80)
    
    queries = [
        {
            'query': "SELECT COUNT(*) as strict_cohort_count FROM eicu_bsi_100_2h_test.__eicu_bsi_100_2h_cohort;",
            'description': "1. Strict BSI Cohort Count"
        },
        {
            'query': """SELECT COUNT(*) as all_patient_units 
                        FROM eicu_crd.patientunitstay 
                        WHERE patientunitstayid IS NOT NULL;""",
            'description': "2. All Patient Unit Stays"
        },
        {
            'query': """SELECT COUNT(*) as stays_with_vitals
                        FROM eicu_crd.patientunitstay p
                        WHERE EXISTS (
                            SELECT 1 FROM eicu_crd.vitalPeriodic v 
                            WHERE v.patientunitstayid = p.patientunitstayid
                        );""",
            'description': "3. Patient Stays with Vital Signs Data"
        },
        {
            'query': """SELECT COUNT(*) as stays_with_labs
                        FROM eicu_crd.patientunitstay p
                        WHERE EXISTS (
                            SELECT 1 FROM eicu_crd.lab l 
                            WHERE l.patientunitstayid = p.patientunitstayid
                        );""",
            'description': "4. Patient Stays with Lab Data"
        },
        {
            'query': """SELECT 
                            COUNT(*) as stays_24h_data,
                            COUNT(DISTINCT p.patientid) as unique_patients
                        FROM eicu_crd.patientunitstay p
                        WHERE p.patientunitstayid IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM eicu_crd.vitalPeriodic v 
                            WHERE v.patientunitstayid = p.patientunitstayid
                            AND v.observationoffset >= 0
                            AND v.observationoffset < 24 * 60
                        );""",
            'description': "5. Patient Stays with Vital Data in First 24h"
        }
    ]
    
    results = {}
    for q in queries:
        result = run_query(conn, q['query'], q['description'])
        if result is not None:
            # Store the first column value
            key = q['description'].split('.')[1].strip() if '.' in q['description'] else q['description']
            results[key] = result.iloc[0, 0]
    
    return results

def analyze_public_table_options(conn):
    """Analyze what public tables are available and their data coverage"""
    print("\n" + "="*80)
    print("PUBLIC TABLE ANALYSIS")
    print("="*80)
    
    print("\nMIMIC-IV Public Tables Analysis:")
    print("-" * 40)
    
    mimic_queries = [
        {
            'query': """SELECT 
                        COUNT(DISTINCT stay_id) as total_stays,
                        COUNT(DISTINCT subject_id) as total_patients,
                        MIN(intime) as earliest_admission,
                        MAX(intime) as latest_admission
                        FROM mimiciv_icu.icustays;""",
            'description': "mimiciv_icu.icustays Overview"
        },
        {
            'query': """SELECT 
                        COUNT(DISTINCT itemid) as unique_items,
                        COUNT(*) as total_measurements,
                        MIN(charttime) as earliest_chart,
                        MAX(charttime) as latest_chart
                        FROM mimiciv_icu.chartevents 
                        WHERE valuenum IS NOT NULL;""",
            'description': "mimiciv_icu.chartevents Coverage"
        },
        {
            'query': """SELECT 
                        COUNT(DISTINCT itemid) as unique_lab_items,
                        COUNT(*) as total_lab_results,
                        MIN(charttime) as earliest_lab,
                        MAX(charttime) as latest_lab
                        FROM mimiciv_hosp.labevents 
                        WHERE valuenum IS NOT NULL;""",
            'description': "mimiciv_hosp.labevents Coverage"
        }
    ]
    
    for q in mimic_queries:
        run_query(conn, q['query'], q['description'])
    
    print("\neICU-CRD Public Tables Analysis:")
    print("-" * 40)
    
    eicu_queries = [
        {
            'query': """SELECT 
                        COUNT(DISTINCT patientunitstayid) as total_stays,
                        COUNT(DISTINCT patientid) as total_patients,
                        MIN(unitdischargeoffset) as min_discharge_offset,
                        MAX(unitdischargeoffset) as max_discharge_offset
                        FROM eicu_crd.patientunitstay;""",
            'description': "eicu_crd.patientunitstay Overview"
        },
        {
            'query': """SELECT 
                        COUNT(DISTINCT patientunitstayid) as stays_with_vitals,
                        COUNT(*) as total_vital_measurements,
                        MIN(observationoffset) as earliest_offset,
                        MAX(observationoffset) as latest_offset
                        FROM eicu_crd.vitalPeriodic;""",
            'description': "eicu_crd.vitalPeriodic Coverage"
        },
        {
            'query': """SELECT 
                        COUNT(DISTINCT patientunitstayid) as stays_with_labs,
                        COUNT(*) as total_lab_results,
                        MIN(labresultoffset) as earliest_lab_offset,
                        MAX(labresultoffset) as latest_lab_offset
                        FROM eicu_crd.lab;""",
            'description': "eicu_crd.lab Coverage"
        }
    ]
    
    for q in eicu_queries:
        run_query(conn, q['query'], q['description'])

def generate_less_restrictive_options():
    """Generate suggestions for less restrictive cohort options"""
    print("\n" + "="*80)
    print("LESS RESTRICTIVE COHORT OPTIONS")
    print("="*80)
    
    options = [
        {
            'name': 'Option 1: Extended Time Window (48h)',
            'mimic_query': '''SELECT COUNT(*) 
                            FROM mimiciv_icu.icustays 
                            WHERE EXTRACT(epoch FROM (outtime - intime))/3600 >= 48 
                            AND hadm_id IS NOT NULL;''',
            'eicu_query': '''SELECT COUNT(*) 
                           FROM eicu_crd.patientunitstay 
                           WHERE unitdischargeoffset >= 48 * 60;''',
            'description': 'ICU stays with at least 48 hours of data'
        },
        {
            'name': 'Option 2: Any ICU Stay with Data',
            'mimic_query': '''SELECT COUNT(DISTINCT i.stay_id) 
                            FROM mimiciv_icu.icustays i
                            WHERE EXISTS (
                                SELECT 1 FROM mimiciv_icu.chartevents c 
                                WHERE c.stay_id = i.stay_id
                            ) OR EXISTS (
                                SELECT 1 FROM mimiciv_hosp.labevents l 
                                WHERE l.hadm_id = i.hadm_id
                            );''',
            'eicu_query': '''SELECT COUNT(DISTINCT p.patientunitstayid) 
                           FROM eicu_crd.patientunitstay p
                           WHERE EXISTS (
                               SELECT 1 FROM eicu_crd.vitalPeriodic v 
                               WHERE v.patientunitstayid = p.patientunitstayid
                           ) OR EXISTS (
                               SELECT 1 FROM eicu_crd.lab l 
                               WHERE l.patientunitstayid = p.patientunitstayid
                           );''',
            'description': 'Any ICU stay with either vital signs or lab data'
        },
        {
            'name': 'Option 3: Adult ICU Stays (≥18 years)',
            'mimic_query': '''SELECT COUNT(DISTINCT i.stay_id)
                            FROM mimiciv_icu.icustays i
                            JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
                            WHERE EXTRACT(YEAR FROM i.intime) - EXTRACT(YEAR FROM p.dob) >= 18;''',
            'eicu_query': '''SELECT COUNT(DISTINCT p.patientunitstayid)
                           FROM eicu_crd.patientunitstay p
                           JOIN eicu_crd.patient pt ON p.patientunitstayid = pt.patientunitstayid
                           WHERE pt.age != '' AND CAST(pt.age AS INTEGER) >= 18;''',
            'description': 'Adult ICU stays (age ≥ 18 years)'
        }
    ]
    
    for option in options:
        print(f"\n{option['name']}:")
        print(f"Description: {option['description']}")
        print(f"MIMIC Query: {option['mimic_query']}")
        print(f"eICU Query: {option['eicu_query']}")
        print("-" * 60)

def main():
    """Main analysis function"""
    print("Analyzing Cohort Restrictiveness")
    print("="*80)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load configuration
        config = load_config()
        print(f"Loaded configuration from: {Path(__file__).parent.parent / 'conf' / 'config.yml'}")
        
        # Get database connection
        conn = get_db_connection(config)
        print("Connected to database successfully")
        
        # Analyze MIMIC cohorts
        mimic_results = analyze_mimic_cohorts(conn)
        
        # Analyze eICU cohorts  
        eicu_results = analyze_eicu_cohorts(conn)
        
        # Analyze public table options
        analyze_public_table_options(conn)
        
        # Generate less restrictive options
        generate_less_restrictive_options()
        
        # Summary comparison
        print("\n" + "="*80)
        print("COHORT RESTRICTIVENESS SUMMARY")
        print("="*80)
        
        if mimic_results:
            strict_mimic = mimic_results.get('Strict BSI Cohort Count', 0)
            all_mimic = mimic_results.get('All Adult ICU Stays (≥4h duration, with hospital admission)', 0)
            if all_mimic > 0:
                mimic_retention = (strict_mimic / all_mimic) * 100
                print(f"\nMIMIC-IV:")
                print(f"  Strict cohort: {strict_mimic:,} stays")
                print(f"  All adult ICU stays (≥4h): {all_mimic:,} stays")
                print(f"  Retention rate: {mimic_retention:.1f}%")
                print(f"  Excluded: {all_mimic - strict_mimic:,} stays ({100-mimic_retention:.1f}%)")
        
        if eicu_results:
            strict_eicu = eicu_results.get('Strict BSI Cohort Count', 0)
            all_eicu = eicu_results.get('All Patient Unit Stays', 0)
            if all_eicu > 0:
                eicu_retention = (strict_eicu / all_eicu) * 100
                print(f"\neICU-CRD:")
                print(f"  Strict cohort: {strict_eicu:,} stays")
                print(f"  All patient unit stays: {all_eicu:,} stays")
                print(f"  Retention rate: {eicu_retention:.1f}%")
                print(f"  Excluded: {all_eicu - strict_eicu:,} stays ({100-eicu_retention:.1f}%)")
        
        print(f"\nRecommendations:")
        print(f"1. If retention rates are very low (<10%), consider using public tables")
        print(f"2. Use mimiciv_icu.icustays and eicu_crd.patientunitstay as base tables")
        print(f"3. Apply minimal filters (e.g., age ≥18, stay duration ≥24h)")
        print(f"4. Consider extending time windows to 48h for more data")
        
        conn.close()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
