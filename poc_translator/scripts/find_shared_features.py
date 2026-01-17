#!/usr/bin/env python3
"""
Shared Feature Discovery Script for MIMIC-IV and eICU-CRD

This script identifies all shared/overlapping clinical features between MIMIC-IV 
and eICU-CRD databases by analyzing their measurement catalogs and finding matches.

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
from sqlalchemy import create_engine, text
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

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


def get_mimic_features(engine):
    """Extract all available features/measurements from MIMIC-IV"""
    logger.info("Extracting MIMIC-IV feature catalog...")
    
    # Get chartevents items (vitals and other charted values)
    chartevents_query = """
    SELECT DISTINCT
        di.itemid,
        di.label,
        di.abbreviation,
        di.category,
        di.unitname,
        'chartevents' as source_table,
        COUNT(*) as num_records
    FROM mimiciv_icu.d_items di
    LEFT JOIN mimiciv_icu.chartevents ce ON di.itemid = ce.itemid
    WHERE di.label IS NOT NULL
    GROUP BY di.itemid, di.label, di.abbreviation, di.category, di.unitname
    HAVING COUNT(*) > 100  -- Only features with meaningful data
    ORDER BY num_records DESC
    """
    
    # Get labevents items (laboratory values)
    labevents_query = """
    SELECT DISTINCT
        di.itemid,
        di.label,
        di.fluid,
        di.category,
        'labevents' as source_table,
        COUNT(*) as num_records
    FROM mimiciv_hosp.d_labitems di
    LEFT JOIN mimiciv_hosp.labevents le ON di.itemid = le.itemid
    WHERE di.label IS NOT NULL
    GROUP BY di.itemid, di.label, di.fluid, di.category
    HAVING COUNT(*) > 100
    ORDER BY num_records DESC
    """
    
    # Get procedureevents items
    procedureevents_query = """
    SELECT DISTINCT
        di.itemid,
        di.label,
        di.abbreviation,
        di.category,
        'procedureevents' as source_table,
        COUNT(*) as num_records
    FROM mimiciv_icu.d_items di
    LEFT JOIN mimiciv_icu.procedureevents pe ON di.itemid = pe.itemid
    WHERE di.label IS NOT NULL
    GROUP BY di.itemid, di.label, di.abbreviation, di.category
    HAVING COUNT(*) > 100
    ORDER BY num_records DESC
    """
    
    try:
        chart_df = pd.read_sql(text(chartevents_query), engine)
        logger.info(f"Found {len(chart_df)} chartevents features")
        
        lab_df = pd.read_sql(text(labevents_query), engine)
        logger.info(f"Found {len(lab_df)} labevents features")
        
        # Combine all MIMIC features
        mimic_features = pd.concat([chart_df, lab_df], ignore_index=True)
        
        # Standardize columns
        mimic_features = mimic_features.rename(columns={
            'itemid': 'feature_id',
            'label': 'feature_name',
            'unitname': 'unit'
        })
        
        # Add normalized name for matching
        mimic_features['normalized_name'] = mimic_features['feature_name'].str.lower().str.strip()
        mimic_features['normalized_name'] = mimic_features['normalized_name'].str.replace(r'[^a-z0-9\s]', '', regex=True)
        
        logger.info(f"Total MIMIC features extracted: {len(mimic_features)}")
        return mimic_features
        
    except Exception as e:
        logger.error(f"Error extracting MIMIC features: {e}")
        raise


def get_eicu_features(engine):
    """Extract all available features/measurements from eICU-CRD"""
    logger.info("Extracting eICU-CRD feature catalog...")
    
    # Get vitalperiodic columns (vital signs)
    vitals_query = """
    SELECT 
        'vitalperiodic' as source_table,
        'heartrate' as feature_name,
        'Heart Rate' as display_name,
        'bpm' as unit,
        COUNT(*) as num_records
    FROM eicu_crd.vitalperiodic
    WHERE heartrate IS NOT NULL
    UNION ALL
    SELECT 
        'vitalperiodic',
        'respiration',
        'Respiratory Rate',
        'breaths/min',
        COUNT(*)
    FROM eicu_crd.vitalperiodic
    WHERE respiration IS NOT NULL
    UNION ALL
    SELECT 
        'vitalperiodic',
        'sao2',
        'SpO2',
        '%',
        COUNT(*)
    FROM eicu_crd.vitalperiodic
    WHERE sao2 IS NOT NULL
    UNION ALL
    SELECT 
        'vitalperiodic',
        'temperature',
        'Temperature',
        'Celsius',
        COUNT(*)
    FROM eicu_crd.vitalperiodic
    WHERE temperature IS NOT NULL
    UNION ALL
    SELECT 
        'vitalperiodic',
        'systemicsystolic',
        'Systolic BP',
        'mmHg',
        COUNT(*)
    FROM eicu_crd.vitalperiodic
    WHERE systemicsystolic IS NOT NULL
    UNION ALL
    SELECT 
        'vitalperiodic',
        'systemicdiastolic',
        'Diastolic BP',
        'mmHg',
        COUNT(*)
    FROM eicu_crd.vitalperiodic
    WHERE systemicdiastolic IS NOT NULL
    UNION ALL
    SELECT 
        'vitalperiodic',
        'systemicmean',
        'Mean Arterial Pressure',
        'mmHg',
        COUNT(*)
    FROM eicu_crd.vitalperiodic
    WHERE systemicmean IS NOT NULL
    UNION ALL
    SELECT 
        'vitalperiodic',
        'cvp',
        'Central Venous Pressure',
        'mmHg',
        COUNT(*)
    FROM eicu_crd.vitalperiodic
    WHERE cvp IS NOT NULL
    """
    
    # Get lab values (most common labs)
    lab_query = """
    SELECT 
        'lab' as source_table,
        labname as feature_name,
        labname as display_name,
        labmeasurenamesystem as unit,
        COUNT(*) as num_records
    FROM eicu_crd.lab
    WHERE labname IS NOT NULL
    GROUP BY labname, labmeasurenamesystem
    HAVING COUNT(*) > 100
    ORDER BY num_records DESC
    """
    
    # Get nurseCharting items (additional vitals and assessments)
    nurse_query = """
    SELECT 
        'nurseCharting' as source_table,
        nursingchartcelltypevallabel as feature_name,
        nursingchartcelltypevallabel as display_name,
        NULL as unit,
        COUNT(*) as num_records
    FROM eicu_crd.nursecharting
    WHERE nursingchartcelltypevallabel IS NOT NULL
    GROUP BY nursingchartcelltypevallabel
    HAVING COUNT(*) > 100
    ORDER BY num_records DESC
    """
    
    try:
        vitals_df = pd.read_sql(text(vitals_query), engine)
        logger.info(f"Found {len(vitals_df)} vital sign features")
        
        lab_df = pd.read_sql(text(lab_query), engine)
        logger.info(f"Found {len(lab_df)} lab features")
        
        nurse_df = pd.read_sql(text(nurse_query), engine)
        logger.info(f"Found {len(nurse_df)} nurse charting features")
        
        # Combine all eICU features
        eicu_features = pd.concat([vitals_df, lab_df, nurse_df], ignore_index=True)
        
        # Add normalized name for matching
        eicu_features['normalized_name'] = eicu_features['feature_name'].str.lower().str.strip()
        eicu_features['normalized_name'] = eicu_features['normalized_name'].str.replace(r'[^a-z0-9\s]', '', regex=True)
        
        logger.info(f"Total eICU features extracted: {len(eicu_features)}")
        return eicu_features
        
    except Exception as e:
        logger.error(f"Error extracting eICU features: {e}")
        raise


def calculate_similarity(str1, str2):
    """Calculate string similarity ratio using SequenceMatcher"""
    return SequenceMatcher(None, str1, str2).ratio()


def find_matches(mimic_features, eicu_features, similarity_threshold=0.8):
    """Find matching features between MIMIC and eICU based on name similarity"""
    logger.info(f"Finding matches with similarity threshold: {similarity_threshold}")
    
    matches = []
    
    # Try exact matches first
    for idx_e, eicu_row in eicu_features.iterrows():
        eicu_name = eicu_row['normalized_name']
        
        # Look for exact match
        exact_matches = mimic_features[mimic_features['normalized_name'] == eicu_name]
        
        if len(exact_matches) > 0:
            for idx_m, mimic_row in exact_matches.iterrows():
                matches.append({
                    'match_type': 'exact',
                    'similarity': 1.0,
                    'mimic_feature_id': mimic_row['feature_id'],
                    'mimic_feature_name': mimic_row['feature_name'],
                    'mimic_source_table': mimic_row['source_table'],
                    'mimic_category': mimic_row.get('category', ''),
                    'mimic_unit': mimic_row.get('unit', ''),
                    'mimic_num_records': mimic_row['num_records'],
                    'eicu_feature_name': eicu_row['feature_name'],
                    'eicu_display_name': eicu_row['display_name'],
                    'eicu_source_table': eicu_row['source_table'],
                    'eicu_unit': eicu_row.get('unit', ''),
                    'eicu_num_records': eicu_row['num_records']
                })
    
    logger.info(f"Found {len(matches)} exact matches")
    
    # Now try fuzzy matching for unmatched features
    matched_mimic_ids = set([m['mimic_feature_id'] for m in matches])
    matched_eicu_names = set([m['eicu_feature_name'] for m in matches])
    
    unmatched_mimic = mimic_features[~mimic_features['feature_id'].isin(matched_mimic_ids)]
    unmatched_eicu = eicu_features[~eicu_features['feature_name'].isin(matched_eicu_names)]
    
    logger.info(f"Performing fuzzy matching on {len(unmatched_eicu)} eICU features vs {len(unmatched_mimic)} MIMIC features...")
    
    for idx_e, eicu_row in unmatched_eicu.iterrows():
        eicu_name = eicu_row['normalized_name']
        
        best_match = None
        best_similarity = 0.0
        
        for idx_m, mimic_row in unmatched_mimic.iterrows():
            mimic_name = mimic_row['normalized_name']
            
            # Calculate similarity
            similarity = calculate_similarity(eicu_name, mimic_name)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = mimic_row
        
        if best_match is not None:
            matches.append({
                'match_type': 'fuzzy',
                'similarity': best_similarity,
                'mimic_feature_id': best_match['feature_id'],
                'mimic_feature_name': best_match['feature_name'],
                'mimic_source_table': best_match['source_table'],
                'mimic_category': best_match.get('category', ''),
                'mimic_unit': best_match.get('unit', ''),
                'mimic_num_records': best_match['num_records'],
                'eicu_feature_name': eicu_row['feature_name'],
                'eicu_display_name': eicu_row['display_name'],
                'eicu_source_table': eicu_row['source_table'],
                'eicu_unit': eicu_row.get('unit', ''),
                'eicu_num_records': eicu_row['num_records']
            })
    
    logger.info(f"Found {len([m for m in matches if m['match_type'] == 'fuzzy'])} fuzzy matches")
    
    return pd.DataFrame(matches)


def add_manual_mappings(matches_df):
    """Add known manual mappings for features that don't match by name"""
    logger.info("Adding manual feature mappings...")
    
    # Define known mappings that might not match by name similarity
    manual_mappings = [
        {
            'match_type': 'manual',
            'similarity': 1.0,
            'mimic_feature_name': 'Heart Rate',
            'mimic_source_table': 'chartevents',
            'eicu_feature_name': 'heartrate',
            'eicu_display_name': 'Heart Rate',
            'eicu_source_table': 'vitalperiodic',
            'notes': 'Core vital sign'
        },
        {
            'match_type': 'manual',
            'similarity': 1.0,
            'mimic_feature_name': 'Respiratory Rate',
            'mimic_source_table': 'chartevents',
            'eicu_feature_name': 'respiration',
            'eicu_display_name': 'Respiratory Rate',
            'eicu_source_table': 'vitalperiodic',
            'notes': 'Core vital sign'
        },
        {
            'match_type': 'manual',
            'similarity': 1.0,
            'mimic_feature_name': 'SpO2',
            'mimic_source_table': 'chartevents',
            'eicu_feature_name': 'sao2',
            'eicu_display_name': 'SpO2',
            'eicu_source_table': 'vitalperiodic',
            'notes': 'Oxygen saturation'
        },
        {
            'match_type': 'manual',
            'similarity': 1.0,
            'mimic_feature_name': 'Temperature',
            'mimic_source_table': 'chartevents',
            'eicu_feature_name': 'temperature',
            'eicu_display_name': 'Temperature',
            'eicu_source_table': 'vitalperiodic',
            'notes': 'Core vital sign'
        },
        {
            'match_type': 'manual',
            'similarity': 1.0,
            'mimic_feature_name': 'Arterial Blood Pressure mean',
            'mimic_source_table': 'chartevents',
            'eicu_feature_name': 'systemicmean',
            'eicu_display_name': 'Mean Arterial Pressure',
            'eicu_source_table': 'vitalperiodic',
            'notes': 'Mean arterial pressure'
        }
    ]
    
    # Check if manual mappings already exist in matches
    for mapping in manual_mappings:
        exists = False
        if not matches_df.empty:
            exists = (
                (matches_df['mimic_feature_name'] == mapping['mimic_feature_name']) &
                (matches_df['eicu_feature_name'] == mapping['eicu_feature_name'])
            ).any()
        
        if not exists:
            matches_df = pd.concat([matches_df, pd.DataFrame([mapping])], ignore_index=True)
    
    return matches_df


def categorize_matches(matches_df):
    """Categorize matched features into clinical categories"""
    logger.info("Categorizing matched features...")
    
    # Define categories based on feature names and types
    def assign_category(row):
        name = row['mimic_feature_name'].lower()
        
        if any(term in name for term in ['heart rate', 'pulse', 'hr ']):
            return 'Vital Signs - Cardiovascular'
        elif any(term in name for term in ['blood pressure', 'bp', 'arterial', 'systolic', 'diastolic', 'map', 'mean']):
            return 'Vital Signs - Blood Pressure'
        elif any(term in name for term in ['respiratory', 'respiration', 'breathing', 'rr ']):
            return 'Vital Signs - Respiratory'
        elif any(term in name for term in ['spo2', 'oxygen', 'o2', 'saturation']):
            return 'Vital Signs - Oxygenation'
        elif any(term in name for term in ['temperature', 'temp']):
            return 'Vital Signs - Temperature'
        elif any(term in name for term in ['glucose', 'sugar']):
            return 'Laboratory - Glucose'
        elif any(term in name for term in ['sodium', 'potassium', 'chloride', 'calcium', 'magnesium']):
            return 'Laboratory - Electrolytes'
        elif any(term in name for term in ['creatinine', 'bun', 'urea', 'kidney', 'renal']):
            return 'Laboratory - Renal'
        elif any(term in name for term in ['wbc', 'white blood', 'hemoglobin', 'hematocrit', 'platelet', 'rbc']):
            return 'Laboratory - Hematology'
        elif any(term in name for term in ['bilirubin', 'alt', 'ast', 'liver', 'hepatic']):
            return 'Laboratory - Hepatic'
        elif any(term in name for term in ['lactate', 'ph', 'pco2', 'po2', 'bicarbonate']):
            return 'Laboratory - Blood Gas'
        elif any(term in name for term in ['weight', 'height', 'bmi']):
            return 'Anthropometric'
        elif any(term in name for term in ['gcs', 'glasgow', 'consciousness']):
            return 'Neurological'
        else:
            return 'Other'
    
    matches_df['category'] = matches_df.apply(assign_category, axis=1)
    
    return matches_df


def generate_report(matches_df, output_dir):
    """Generate summary report of shared features"""
    logger.info("Generating feature matching report...")
    
    report_path = os.path.join(output_dir, "shared_features_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MIMIC-IV and eICU-CRD Shared Features Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Matches Found: {len(matches_df)}\n")
        f.write(f"  - Exact Matches: {len(matches_df[matches_df['match_type'] == 'exact'])}\n")
        f.write(f"  - Fuzzy Matches: {len(matches_df[matches_df['match_type'] == 'fuzzy'])}\n")
        f.write(f"  - Manual Mappings: {len(matches_df[matches_df['match_type'] == 'manual'])}\n\n")
        
        # Summary by category
        f.write("-" * 80 + "\n")
        f.write("Matches by Category:\n")
        f.write("-" * 80 + "\n")
        category_counts = matches_df['category'].value_counts()
        for category, count in category_counts.items():
            f.write(f"  {category}: {count}\n")
        f.write("\n")
        
        # Top matches by data availability
        f.write("-" * 80 + "\n")
        f.write("Top 20 Matches by Data Availability (MIMIC records):\n")
        f.write("-" * 80 + "\n")
        top_matches = matches_df.nlargest(20, 'mimic_num_records')
        for idx, row in top_matches.iterrows():
            f.write(f"\n{idx+1}. {row['mimic_feature_name']} <-> {row['eicu_display_name']}\n")
            f.write(f"   Category: {row['category']}\n")
            f.write(f"   Match Type: {row['match_type']} (similarity: {row['similarity']:.2f})\n")
            f.write(f"   MIMIC: {row['mimic_source_table']} ({row['mimic_num_records']:,} records)\n")
            f.write(f"   eICU: {row['eicu_source_table']} ({row['eicu_num_records']:,} records)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"Report saved to: {report_path}")


def main():
    """Main function to find shared features"""
    parser = argparse.ArgumentParser(description='Find shared features between MIMIC-IV and eICU-CRD')
    parser.add_argument('--similarity-threshold', type=float, default=0.75,
                        help='Similarity threshold for fuzzy matching (0.0-1.0)')
    parser.add_argument('--output-dir', type=str, 
                        help='Output directory for results (default from config)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config()
        
        output_dir = args.output_dir or config['paths']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("=== Shared Feature Discovery ===")
        logger.info(f"Similarity threshold: {args.similarity_threshold}")
        
        # Connect to databases
        logger.info("Connecting to MIMIC-IV database...")
        mimic_engine = create_db_connection(config['db']['mimic_conn'])
        
        logger.info("Connecting to eICU-CRD database...")
        eicu_engine = create_db_connection(config['db']['eicu_conn'])
        
        # Extract features from both databases
        mimic_features = get_mimic_features(mimic_engine)
        eicu_features = get_eicu_features(eicu_engine)
        
        # Save raw feature catalogs
        mimic_catalog_path = os.path.join(output_dir, "mimic_feature_catalog.csv")
        eicu_catalog_path = os.path.join(output_dir, "eicu_feature_catalog.csv")
        
        mimic_features.to_csv(mimic_catalog_path, index=False)
        eicu_features.to_csv(eicu_catalog_path, index=False)
        
        logger.info(f"MIMIC feature catalog saved to: {mimic_catalog_path}")
        logger.info(f"eICU feature catalog saved to: {eicu_catalog_path}")
        
        # Find matches
        matches_df = find_matches(mimic_features, eicu_features, args.similarity_threshold)
        
        # Add manual mappings
        matches_df = add_manual_mappings(matches_df)
        
        # Categorize matches
        matches_df = categorize_matches(matches_df)
        
        # Sort by similarity and data availability
        matches_df = matches_df.sort_values(['similarity', 'mimic_num_records'], 
                                           ascending=[False, False])
        
        # Save matches
        matches_path = os.path.join(output_dir, "shared_features.csv")
        matches_df.to_csv(matches_path, index=False)
        logger.info(f"Shared features saved to: {matches_path}")
        
        # Generate report
        generate_report(matches_df, output_dir)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total MIMIC features: {len(mimic_features)}")
        logger.info(f"Total eICU features: {len(eicu_features)}")
        logger.info(f"Total shared features found: {len(matches_df)}")
        logger.info(f"  - Exact matches: {len(matches_df[matches_df['match_type'] == 'exact'])}")
        logger.info(f"  - Fuzzy matches: {len(matches_df[matches_df['match_type'] == 'fuzzy'])}")
        logger.info(f"  - Manual mappings: {len(matches_df[matches_df['match_type'] == 'manual'])}")
        logger.info("=" * 80)
        
        logger.info("\n=== Feature Discovery Complete ===")
        
    except Exception as e:
        logger.error(f"Feature discovery failed: {e}")
        raise


if __name__ == "__main__":
    main()





