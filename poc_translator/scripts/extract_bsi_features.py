#!/usr/bin/env python3
"""
BSI Feature Extractor for MIMIC-IV and eICU-CRD Domain Translation Project

This script extracts raw time-series patient data from MIMIC-IV and eICU-CRD for domain translation.
Focuses on 40 specific aligned features for BSI (Bloodstream Infection) prediction.
Unlike the POC script, this extracts RAW measurements (not aggregated statistics).

Author: BSI Translation Project
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

# Define the 40 aligned features for BSI prediction
BSI_FEATURES = {
    'Alanine Aminotransferase (ALT)': 1,
    'Albumin': 2,
    'Alkaline Phosphatase': 3,
    'Asparate Aminotransferase (AST)': 4,
    'Bilirubin': 5,
    'C-Reactive Protein': 6,
    'Creatinine': 7,
    'GCS - Eye Opening': 8,
    'GCS - Motor Response': 9,
    'GCS - Verbal Response': 10,
    'Head of Bed': 11,
    'Heart Rate': 12,
    'Hematocrit': 13,
    'Hemoglobin': 14,
    'INR(PT)': 15,
    'Inspired O2 Fraction': 16,
    'Lactate': 17,
    'Lymphocytes': 18,
    'MCHC': 19,
    'MCV': 20,
    'Magnesium': 21,
    'Mean Airway Pressure': 22,
    'Non Invasive Blood Pressure diastolic': 23,
    'Non Invasive Blood Pressure mean': 24,
    'Non Invasive Blood Pressure systolic': 25,
    'O2 saturation pulseoxymetry': 26,
    'PT': 27,
    'Potassium': 28,
    'RBC': 29,
    'RDW': 30,
    'Respiratory Rate': 31,
    'Sodium': 32,
    'Temperature Fahrenheit': 33,
    'Temperature': 34,
    'Tidal Volume (observed)': 35,
    'Urea Nitrogen': 36,
    'WBC': 37,
    'pH': 38,
    'pO2': 39,
    'Lactate Dehydrogenase (LD)': 40
}

# MIMIC-IV feature name mappings
# Maps standardized feature names to MIMIC label names
MIMIC_FEATURE_NAMES = {
    'Alanine Aminotransferase (ALT)': ['Alanine Aminotransferase (ALT)', 'ALT'],
    'Albumin': ['Albumin'],
    'Alkaline Phosphatase': ['Alkaline Phosphatase'],
    'Asparate Aminotransferase (AST)': ['Asparate Aminotransferase (AST)'],
    'Bilirubin': ['Bilirubin, Total'],
    'C-Reactive Protein': ['C-Reactive Protein', 'C Reactive Protein (CRP)'],
    'Creatinine': ['Creatinine', 'Creatinine (0-1.3)'],
    'GCS - Eye Opening': ['Eye Opening'],
    'GCS - Motor Response': ['Motor Response'],
    'GCS - Verbal Response': ['Verbal Response'],
    'Head of Bed': ['Head of Bed'],
    'Heart Rate': ['Heart Rate'],
    'Hematocrit': ['Hematocrit', 'Hematocrit, Calculated'],
    'Hemoglobin': ['Hemoglobin'],
    'INR(PT)': ['INR(PT)', 'INR', 'INR (2-4 ref. range)'],
    'Inspired O2 Fraction': ['Inspired O2 Fraction'],
    'Lactate': ['Lactate'],
    'Lymphocytes': ['Lymphocytes'],
    'MCHC': ['MCHC'],
    'MCV': ['MCV'],
    'Magnesium': ['Magnesium', 'Magnesium (1.6-2.6)'],
    'Mean Airway Pressure': ['Mean Airway Pressure'],
    'Non Invasive Blood Pressure diastolic': ['NBP [Diastolic]', 'Non Invasive Blood Pressure diastolic'],
    'Non Invasive Blood Pressure mean': ['NBP Mean', 'Non Invasive Blood Pressure mean'],
    'Non Invasive Blood Pressure systolic': ['NBP [Systolic]', 'Non Invasive Blood Pressure systolic'],
    'O2 saturation pulseoxymetry': ['Oxygen Saturation', 'SpO2'],
    'PT': ['PT', 'Prothrombin time', 'PT(11-13.5)'],
    'Potassium': ['Potassium', 'Potassium (3.5-5.3)', 'Potassium, Whole Blood'],
    'RBC': ['RBC', 'Red Blood Cells'],
    'RDW': ['RDW'],
    'Respiratory Rate': ['Respiratory Rate', 'Respiratory Rate (spontaneous)'],
    'Sodium': ['Sodium'],
    'Temperature Fahrenheit': ['Temperature F', 'Temperature Fahrenheit'],
    'Temperature': ['Temperature', 'Temperature C (calc)'],
    'Tidal Volume (observed)': ['Tidal Volume (observed)'],
    'Urea Nitrogen': ['Urea Nitrogen', 'BUN', 'BUN (6-20)'],
    'WBC': ['WBC', 'White Blood Cells', 'WBC (4-11,000)', 'WBC   (4-11,000)'],
    'pH': ['pH'],
    'pO2': ['pO2'],
    'Lactate Dehydrogenase (LD)': ['Lactate Dehydrogenase (LD)']
}

# eICU-CRD feature name mappings
# Maps standardized feature names to eICU table + column combinations
EICU_FEATURE_NAMES = {
    'Alanine Aminotransferase (ALT)': [('lab', 'ALT (SGPT)', 'labname'), ('lab', 'ALT (SGPT) ', 'labname')],
    'Albumin': [('lab', 'albumin', 'labname')],
    'Alkaline Phosphatase': [('lab', 'alkaline phos.', 'labname')],
    'Asparate Aminotransferase (AST)': [('lab', 'AST (SGOT)', 'labname')],
    'Bilirubin': [('lab', 'total bilirubin', 'labname')],
    'C-Reactive Protein': [('lab', 'CRP', 'labname')],
    'Creatinine': [('lab', 'creatinine', 'labname')],
    'GCS - Eye Opening': [('nursecharting', 'Eyes', 'nursingchartcelltypevalname')],
    'GCS - Motor Response': [('nursecharting', 'Motor', 'nursingchartcelltypevalname')],
    'GCS - Verbal Response': [('nursecharting', 'Verbal', 'nursingchartcelltypevalname')],
    'Head of Bed': [('nursecharting', 'Head of Bed Elevation', 'nursingchartcelltypevalname')],
    'Heart Rate': [('vitalperiodic', 'heartrate', 'column')],
    'Hematocrit': [('lab', 'Hct', 'labname')],
    'Hemoglobin': [('lab', 'Hgb', 'labname')],
    'INR(PT)': [('lab', 'PT - INR', 'labname')],
    'Inspired O2 Fraction': [('respiratorycharting', 'FiO2', 'respchartvaluelabel')],
    'Lactate': [('lab', 'lactate', 'labname')],
    'Lymphocytes': [('lab', '-lymphs', 'labname')],
    'MCHC': [('lab', 'MCHC', 'labname')],
    'MCV': [('lab', 'MCV', 'labname')],
    'Magnesium': [('lab', 'magnesium', 'labname')],
    'Mean Airway Pressure': [('respiratorycharting', 'Mean Airway Pressure', 'respchartvaluelabel')],
    'Non Invasive Blood Pressure diastolic': [('nursecharting', 'Non-Invasive BP Diastolic', 'nursingchartcelltypevalname')],
    'Non Invasive Blood Pressure mean': [('nursecharting', 'Non-Invasive BP Mean', 'nursingchartcelltypevalname')],
    'Non Invasive Blood Pressure systolic': [('nursecharting', 'Non-Invasive BP Systolic', 'nursingchartcelltypevalname')],
    'O2 saturation pulseoxymetry': [('nursecharting', 'O2 Sat (%)', 'nursingchartcelltypevalname'), ('nursecharting', 'O2 Saturation', 'nursingchartcelltypevalname')],
    'PT': [('lab', 'PT', 'labname')],
    'Potassium': [('lab', 'potassium', 'labname')],
    'RBC': [('lab', 'RBC', 'labname')],
    'RDW': [('lab', 'RDW', 'labname')],
    'Respiratory Rate': [('vitalperiodic', 'respiration', 'column')],
    'Sodium': [('lab', 'sodium', 'labname')],
    'Temperature Fahrenheit': [('nursecharting', 'Temperature (F)', 'nursingchartcelltypevalname')],
    'Temperature': [('vitalperiodic', 'temperature', 'column'), ('nursecharting', 'Temperature (C)', 'nursingchartcelltypevalname')],
    'Tidal Volume (observed)': [('respiratorycharting', 'Exhaled TV (patient)', 'respchartvaluelabel')],
    'Urea Nitrogen': [('lab', 'BUN', 'labname'), ('lab', '24 h urine urea nitrogen', 'labname')],
    'WBC': [('lab', 'WBC x 1000', 'labname'), ('lab', 'WBC', 'labname')],
    'pH': [('lab', 'pH', 'labname')],
    'pO2': [('lab', 'paO2', 'labname')],
    'Lactate Dehydrogenase (LD)': [('lab', 'LDH', 'labname')]
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

def get_mimic_demographic_data(engine):
    """Extract demographic data (age, gender) from MIMIC-IV using public tables"""
    
    query = """
    SELECT 
        i.stay_id as icustay_id,
        i.subject_id,
        i.hadm_id,
        p.gender,
        p.anchor_age as age,
        i.intime,
        i.outtime
    FROM mimiciv_icu.icustays i
    JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
    WHERE p.anchor_age IS NOT NULL
        AND p.anchor_age >= 18  -- Adult patients only
        AND i.hadm_id IS NOT NULL  -- Must have hospital admission
        AND EXTRACT(epoch FROM (i.outtime - i.intime))/3600 >= 72  -- At least 72h stay
        AND i.intime IS NOT NULL
        AND i.outtime IS NOT NULL
    """
    
    logger.info("Extracting MIMIC demographic data using public tables (adults, >=72h stays)...")
    return pd.read_sql(text(query), engine)

def get_eicu_demographic_data(engine):
    """Extract demographic data (age, gender) from eICU-CRD using public tables"""
    
    query = """
    SELECT 
        p.patientunitstayid as icustay_id,
        p.patientunitstayid as subject_id,  -- eICU uses same ID for both
        CASE 
            WHEN p.age = '> 89' THEN 91.0
            WHEN p.age ~ '^[0-9]+$' THEN p.age::float
            ELSE NULL 
        END as age,
        p.gender
    FROM eicu_crd.patient p
    WHERE p.patientunitstayid IS NOT NULL
        AND p.unitdischargeoffset >= 72 * 60  -- At least 72h stay (in minutes)
        AND p.age != '' AND p.age != 'Unknown' AND p.age IS NOT NULL
        AND CASE 
            WHEN p.age = '> 89' THEN 90 
            ELSE CAST(p.age AS INTEGER) 
        END >= 18  -- Adult patients
    """
    
    logger.info("Extracting eICU demographic data using public tables (adults, >=72h stays)...")
    return pd.read_sql(text(query), engine)

def extract_mimic_units_metadata(engine):
    """Extract units metadata for MIMIC-IV features"""
    logger.info("Extracting MIMIC-IV units metadata...")
    
    # Collect all feature label names for query
    all_label_names = []
    for feature_name, label_list in MIMIC_FEATURE_NAMES.items():
        all_label_names.extend(label_list)
    
    # Query chartevents for units
    chartevents_units_query = """
    SELECT DISTINCT
        d.label as feature_name,
        'chartevents' as source_table,
        ce.valueuom as unit
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.d_items d ON ce.itemid = d.itemid
    WHERE d.label = ANY(:label_names)
        AND ce.valueuom IS NOT NULL
        AND ce.valueuom != ''
    """
    
    chartevents_units = pd.read_sql(text(chartevents_units_query), engine, params={
        'label_names': all_label_names
    })
    logger.info(f"Found {len(chartevents_units)} chartevents unit mappings")
    
    # Query labevents for units
    labevents_units_query = """
    SELECT DISTINCT
        d.label as feature_name,
        'labevents' as source_table,
        le.valueuom as unit
    FROM mimiciv_hosp.labevents le
    JOIN mimiciv_hosp.d_labitems d ON le.itemid = d.itemid
    WHERE d.label = ANY(:label_names)
        AND le.valueuom IS NOT NULL
        AND le.valueuom != ''
    """
    
    labevents_units = pd.read_sql(text(labevents_units_query), engine, params={
        'label_names': all_label_names
    })
    logger.info(f"Found {len(labevents_units)} labevents unit mappings")
    
    # Combine both sources
    all_units = pd.concat([chartevents_units, labevents_units], ignore_index=True)
    
    # Map to standardized feature names
    feature_name_map = {}
    for std_name, label_list in MIMIC_FEATURE_NAMES.items():
        for label in label_list:
            feature_name_map[label] = std_name
    
    all_units['feature_name'] = all_units['feature_name'].map(feature_name_map)
    
    # Add hardcoded units for features with NULL units in database
    hardcoded_units = []
    
    # Features that have NULL units in database but known clinical units
    mimic_hardcoded_map = {
        ('GCS - Eye Opening', 'chartevents'): 'score',
        ('GCS - Motor Response', 'chartevents'): 'score',
        ('GCS - Verbal Response', 'chartevents'): 'score',
        ('Head of Bed', 'chartevents'): 'degrees',
        ('INR(PT)', 'chartevents'): 'ratio',
        ('INR(PT)', 'labevents'): 'ratio',
        ('Inspired O2 Fraction', 'chartevents'): 'fraction',
        ('Temperature', 'labevents'): 'C',
    }
    
    for (feature, source), unit in mimic_hardcoded_map.items():
        hardcoded_units.append({
            'feature_name': feature,
            'source_table': source,
            'unit': unit
        })
    
    if hardcoded_units:
        hardcoded_df = pd.DataFrame(hardcoded_units)
        all_units = pd.concat([all_units, hardcoded_df], ignore_index=True)
        logger.info(f"Added {len(hardcoded_units)} hardcoded unit mappings for NULL units")
    
    # Group by feature and source to get all units
    units_summary = all_units.groupby(['feature_name', 'source_table'])['unit'].apply(
        lambda x: ';'.join(sorted(set(x)))
    ).reset_index()
    units_summary.columns = ['feature_name', 'source_table', 'units_found']
    
    logger.info(f"Extracted units metadata for {units_summary['feature_name'].nunique()} MIMIC features")
    return units_summary

def extract_eicu_units_metadata(engine):
    """Extract units metadata for eICU-CRD features"""
    logger.info("Extracting eICU-CRD units metadata...")
    
    all_units = []
    
    # Extract from lab table (has explicit units)
    lab_names = []
    for feature_name, sources in EICU_FEATURE_NAMES.items():
        for table, name, col_type in sources:
            if table == 'lab' and col_type == 'labname':
                lab_names.append((feature_name, name))
    
    if lab_names:
        lab_name_list = [name for _, name in lab_names]
        lab_units_query = """
        SELECT DISTINCT
            labname as source_name,
            'lab' as source_table,
            labmeasurenamesystem as unit
        FROM eicu_crd.lab
        WHERE labname = ANY(:lab_names)
            AND labmeasurenamesystem IS NOT NULL
            AND labmeasurenamesystem != ''
        """
        
        lab_units = pd.read_sql(text(lab_units_query), engine, params={
            'lab_names': lab_name_list
        })
        
        # Map lab names to feature names
        lab_name_map = {name: feature for feature, name in lab_names}
        lab_units['feature_name'] = lab_units['source_name'].map(lab_name_map)
        lab_units = lab_units.drop(columns=['source_name'])
        
        all_units.append(lab_units)
        logger.info(f"Found {len(lab_units)} lab unit mappings")
    
    # Add hardcoded units for tables without explicit unit columns
    hardcoded_units = []
    
    # Nursecharting features - units inferred from clinical knowledge
    nurse_unit_map = {
        'GCS - Eye Opening': 'score',
        'GCS - Motor Response': 'score',
        'GCS - Verbal Response': 'score',
        'Head of Bed': 'degrees',
        'Non Invasive Blood Pressure diastolic': 'mmHg',
        'Non Invasive Blood Pressure mean': 'mmHg',
        'Non Invasive Blood Pressure systolic': 'mmHg',
        'O2 saturation pulseoxymetry': '%',
        'Temperature Fahrenheit': 'F',
        'Temperature': 'C'
    }
    
    for feature, unit in nurse_unit_map.items():
        if feature in EICU_FEATURE_NAMES:
            for table, name, col_type in EICU_FEATURE_NAMES[feature]:
                if table == 'nursecharting':
                    hardcoded_units.append({
                        'feature_name': feature,
                        'source_table': 'nursecharting',
                        'unit': unit
                    })
                    break
    
    # Respiratory charting features
    resp_unit_map = {
        'Inspired O2 Fraction': 'fraction',
        'Mean Airway Pressure': 'cmH2O',
        'Tidal Volume (observed)': 'mL'
    }
    
    for feature, unit in resp_unit_map.items():
        if feature in EICU_FEATURE_NAMES:
            for table, name, col_type in EICU_FEATURE_NAMES[feature]:
                if table == 'respiratorycharting':
                    hardcoded_units.append({
                        'feature_name': feature,
                        'source_table': 'respiratorycharting',
                        'unit': unit
                    })
                    break
    
    # Vital periodic features
    vital_unit_map = {
        'Heart Rate': 'bpm',
        'Respiratory Rate': 'breaths/min',
        'Temperature': 'C'
    }
    
    for feature, unit in vital_unit_map.items():
        if feature in EICU_FEATURE_NAMES:
            for table, col_name, col_type in EICU_FEATURE_NAMES[feature]:
                if table == 'vitalperiodic':
                    hardcoded_units.append({
                        'feature_name': feature,
                        'source_table': 'vitalperiodic',
                        'unit': unit
                    })
                    break
    
    # Lab features with NULL units in database
    lab_null_unit_map = {
        'pH': 'units'
    }
    
    for feature, unit in lab_null_unit_map.items():
        if feature in EICU_FEATURE_NAMES:
            hardcoded_units.append({
                'feature_name': feature,
                'source_table': 'lab',
                'unit': unit
            })
    
    if hardcoded_units:
        hardcoded_df = pd.DataFrame(hardcoded_units)
        all_units.append(hardcoded_df)
        logger.info(f"Added {len(hardcoded_units)} hardcoded unit mappings")
    
    # Combine all sources
    if not all_units:
        logger.warning("No units metadata found")
        return pd.DataFrame(columns=['feature_name', 'source_table', 'units_found'])
    
    combined_units = pd.concat(all_units, ignore_index=True)
    
    # Group by feature and source to get all units
    units_summary = combined_units.groupby(['feature_name', 'source_table'])['unit'].apply(
        lambda x: ';'.join(sorted(set(x))) if isinstance(x, (list, pd.Series)) else str(x)
    ).reset_index()
    units_summary.columns = ['feature_name', 'source_table', 'units_found']
    
    logger.info(f"Extracted units metadata for {units_summary['feature_name'].nunique()} eICU features")
    return units_summary

def extract_mimic_data(engine):
    """Extract MIMIC-IV raw time-series data for the 40 BSI features"""
    logger.info("Starting MIMIC-IV raw data extraction...")
    
    # Get demographic data first
    demo_data = get_mimic_demographic_data(engine)
    logger.info(f"Found {len(demo_data)} MIMIC ICU stays with demographic data")
    
    if demo_data.empty:
        logger.warning("No demographic data found")
        return pd.DataFrame()
    
    # Get all ICU stay IDs
    icustay_ids = demo_data['icustay_id'].unique().tolist()
    
    # Collect all feature label names for query
    all_label_names = []
    for feature_name, label_list in MIMIC_FEATURE_NAMES.items():
        all_label_names.extend(label_list)
    
    logger.info(f"Extracting measurements for {len(BSI_FEATURES)} features from {len(icustay_ids)} stays...")
    
    # BATCH QUERY 1: Get ALL chart events data in one query
    chartevents_query = """
    SELECT 
        ce.stay_id as icustay_id,
        d.label as feature_name,
        ce.valuenum as feature_value,
        ce.charttime,
        i.intime
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays i ON ce.stay_id = i.stay_id
    JOIN mimiciv_icu.d_items d ON ce.itemid = d.itemid
    WHERE ce.stay_id = ANY(:icustay_ids)
        AND ce.charttime >= i.intime + INTERVAL '24 hours'
        AND ce.charttime <= i.intime + INTERVAL '72 hours'
        AND d.label = ANY(:label_names)
        AND ce.valuenum IS NOT NULL
    """
    
    chartevents_df = pd.read_sql(text(chartevents_query), engine, params={
        'icustay_ids': icustay_ids,
        'label_names': all_label_names
    })
    logger.info(f"Retrieved {len(chartevents_df)} chart event measurements")
    
    # BATCH QUERY 2: Get ALL lab events data in one query
    labevents_query = """
    SELECT 
        i.stay_id as icustay_id,
        d.label as feature_name,
        le.valuenum as feature_value,
        le.charttime,
        i.intime
    FROM mimiciv_hosp.labevents le
    JOIN mimiciv_icu.icustays i ON le.subject_id = i.subject_id AND le.hadm_id = i.hadm_id
    JOIN mimiciv_hosp.d_labitems d ON le.itemid = d.itemid
    WHERE i.stay_id = ANY(:icustay_ids)
        AND le.charttime >= i.intime + INTERVAL '24 hours'
        AND le.charttime <= i.intime + INTERVAL '72 hours'
        AND d.label = ANY(:label_names)
        AND le.valuenum IS NOT NULL
    """
    
    labevents_df = pd.read_sql(text(labevents_query), engine, params={
        'icustay_ids': icustay_ids,
        'label_names': all_label_names
    })
    logger.info(f"Retrieved {len(labevents_df)} lab event measurements")
    
    # Combine all measurements
    all_measurements = pd.concat([chartevents_df, labevents_df], ignore_index=True)
    
    if all_measurements.empty:
        logger.warning("No measurements found")
        return pd.DataFrame()
    
    # Calculate time offset in minutes from ICU admission
    all_measurements['time_offset'] = (
        (all_measurements['charttime'] - all_measurements['intime']).dt.total_seconds() / 60
    ).astype(int)
    
    # Drop charttime and intime columns
    all_measurements = all_measurements.drop(columns=['charttime', 'intime'])
    
    # Map feature names to standardized names
    feature_name_map = {}
    for std_name, label_list in MIMIC_FEATURE_NAMES.items():
        for label in label_list:
            feature_name_map[label] = std_name
    
    all_measurements['feature_name'] = all_measurements['feature_name'].map(feature_name_map)
    
    # Merge with demographic data
    all_measurements = all_measurements.merge(
        demo_data[['icustay_id', 'subject_id', 'age', 'gender']],
        on='icustay_id',
        how='left'
    )
    
    # Convert gender to numeric (0=F, 1=M)
    all_measurements['gender'] = all_measurements['gender'].map({'F': 0, 'M': 1})
    
    # Rename columns to match output schema
    all_measurements = all_measurements.rename(columns={
        'subject_id': 'patient_id',
        'icustay_id': 'icu_stay_id'
    })
    
    # Pivot data to have one row per timepoint with all features as columns
    logger.info("Pivoting data to wide format...")
    pivoted_data = all_measurements.pivot_table(
        index=['patient_id', 'icu_stay_id', 'time_offset', 'age', 'gender'],
        columns='feature_name',
        values='feature_value',
        aggfunc='mean'  # If multiple measurements at same time, take mean
    ).reset_index()
    
    logger.info(f"Extracted MIMIC data: {len(pivoted_data)} timepoints across {pivoted_data['icu_stay_id'].nunique()} ICU stays")
    return pivoted_data

def extract_eicu_data(engine):
    """Extract eICU-CRD raw time-series data for the 40 BSI features"""
    logger.info("Starting eICU-CRD raw data extraction...")
    
    # Get demographic data first
    demo_data = get_eicu_demographic_data(engine)
    logger.info(f"Found {len(demo_data)} eICU ICU stays with demographic data")
    
    if demo_data.empty:
        logger.warning("No demographic data found")
        return pd.DataFrame()
    
    # Get all ICU stay IDs
    icustay_ids = demo_data['icustay_id'].unique().tolist()
    
    logger.info(f"Extracting measurements for {len(BSI_FEATURES)} features from {len(icustay_ids)} stays...")
    
    all_measurements = []
    
    # Extract from lab table
    lab_names = []
    for feature_name, sources in EICU_FEATURE_NAMES.items():
        for table, name, col_type in sources:
            if table == 'lab' and col_type == 'labname':
                lab_names.append((feature_name, name))
    
    if lab_names:
        lab_name_list = [name for _, name in lab_names]
        lab_query = """
        SELECT 
            patientunitstayid as icustay_id,
            labname as source_name,
            labresult as feature_value,
            labresultoffset as time_offset
        FROM eicu_crd.lab
        WHERE patientunitstayid = ANY(:icustay_ids)
            AND labresultoffset >= 1440  -- 24 hours in minutes
            AND labresultoffset <= 4320  -- 72 hours in minutes
            AND labname = ANY(:lab_names)
            AND labresult IS NOT NULL
        """
        
        lab_df = pd.read_sql(text(lab_query), engine, params={
            'icustay_ids': icustay_ids,
            'lab_names': lab_name_list
        })
        
        # Map lab names to feature names
        lab_name_map = {name: feature for feature, name in lab_names}
        lab_df['feature_name'] = lab_df['source_name'].map(lab_name_map)
        lab_df = lab_df.drop(columns=['source_name'])
        
        # Convert feature_value to numeric
        lab_df['feature_value'] = pd.to_numeric(lab_df['feature_value'], errors='coerce')
        
        all_measurements.append(lab_df)
        logger.info(f"Retrieved {len(lab_df)} lab measurements")
    
    # Extract from nursecharting table
    nurse_names = []
    for feature_name, sources in EICU_FEATURE_NAMES.items():
        for table, name, col_type in sources:
            if table == 'nursecharting' and col_type == 'nursingchartcelltypevalname':
                nurse_names.append((feature_name, name))
    
    if nurse_names:
        nurse_name_list = [name for _, name in nurse_names]
        nurse_query = """
        SELECT 
            patientunitstayid as icustay_id,
            nursingchartcelltypevalname as source_name,
            CASE 
                WHEN nursingchartvalue ~ '^[0-9]+\.?[0-9]*$' 
                THEN nursingchartvalue::float 
                ELSE NULL 
            END as feature_value,
            nursingchartoffset as time_offset
        FROM eicu_crd.nursecharting
        WHERE patientunitstayid = ANY(:icustay_ids)
            AND nursingchartoffset >= 1440  -- 24 hours in minutes
            AND nursingchartoffset <= 4320  -- 72 hours in minutes
            AND nursingchartcelltypevalname = ANY(:nurse_names)
            AND nursingchartvalue IS NOT NULL
        """
        
        nurse_df = pd.read_sql(text(nurse_query), engine, params={
            'icustay_ids': icustay_ids,
            'nurse_names': nurse_name_list
        })
        
        # Map nurse names to feature names
        nurse_name_map = {name: feature for feature, name in nurse_names}
        nurse_df['feature_name'] = nurse_df['source_name'].map(nurse_name_map)
        nurse_df = nurse_df.drop(columns=['source_name'])
        
        all_measurements.append(nurse_df)
        logger.info(f"Retrieved {len(nurse_df)} nurse charting measurements")
    
    # Extract from respiratorycharting table
    resp_names = []
    for feature_name, sources in EICU_FEATURE_NAMES.items():
        for table, name, col_type in sources:
            if table == 'respiratorycharting' and col_type == 'respchartvaluelabel':
                resp_names.append((feature_name, name))
    
    if resp_names:
        resp_name_list = [name for _, name in resp_names]
        resp_query = """
        SELECT 
            patientunitstayid as icustay_id,
            respchartvaluelabel as source_name,
            CASE 
                WHEN respchartvalue ~ '^[0-9]+\.?[0-9]*$' 
                THEN respchartvalue::float 
                ELSE NULL 
            END as feature_value,
            respchartoffset as time_offset
        FROM eicu_crd.respiratorycharting
        WHERE patientunitstayid = ANY(:icustay_ids)
            AND respchartoffset >= 1440  -- 24 hours in minutes
            AND respchartoffset <= 4320  -- 72 hours in minutes
            AND respchartvaluelabel = ANY(:resp_names)
            AND respchartvalue IS NOT NULL
        """
        
        resp_df = pd.read_sql(text(resp_query), engine, params={
            'icustay_ids': icustay_ids,
            'resp_names': resp_name_list
        })
        
        # Map resp names to feature names
        resp_name_map = {name: feature for feature, name in resp_names}
        resp_df['feature_name'] = resp_df['source_name'].map(resp_name_map)
        resp_df = resp_df.drop(columns=['source_name'])
        
        all_measurements.append(resp_df)
        logger.info(f"Retrieved {len(resp_df)} respiratory charting measurements")
    
    # Extract from vitalperiodic table
    vital_columns = []
    for feature_name, sources in EICU_FEATURE_NAMES.items():
        for table, col_name, col_type in sources:
            if table == 'vitalperiodic' and col_type == 'column':
                vital_columns.append((feature_name, col_name))
    
    if vital_columns:
        col_names = [col for _, col in vital_columns]
        vital_query = f"""
        SELECT 
            patientunitstayid as icustay_id,
            {', '.join(col_names)},
            observationoffset as time_offset
        FROM eicu_crd.vitalperiodic
        WHERE patientunitstayid = ANY(:icustay_ids)
            AND observationoffset >= 1440  -- 24 hours in minutes
            AND observationoffset <= 4320  -- 72 hours in minutes
        """
        
        vital_df = pd.read_sql(text(vital_query), engine, params={
            'icustay_ids': icustay_ids
        })
        
        # Melt vital signs to long format
        vital_df = vital_df.melt(
            id_vars=['icustay_id', 'time_offset'],
            value_vars=col_names,
            var_name='source_name',
            value_name='feature_value'
        )
        
        # Map column names to feature names
        vital_name_map = {col: feature for feature, col in vital_columns}
        vital_df['feature_name'] = vital_df['source_name'].map(vital_name_map)
        vital_df = vital_df.drop(columns=['source_name'])
        
        # Convert to numeric
        vital_df['feature_value'] = pd.to_numeric(vital_df['feature_value'], errors='coerce')
        
        all_measurements.append(vital_df)
        logger.info(f"Retrieved {len(vital_df)} vital sign measurements")
    
    # Combine all measurements
    if not all_measurements:
        logger.warning("No measurements found")
        return pd.DataFrame()
    
    combined_measurements = pd.concat(all_measurements, ignore_index=True)
    
    # Remove rows with null values
    combined_measurements = combined_measurements.dropna(subset=['feature_value'])
    
    # Merge with demographic data
    combined_measurements = combined_measurements.merge(
        demo_data[['icustay_id', 'subject_id', 'age', 'gender']],
        on='icustay_id',
        how='left'
    )
    
    # Convert gender to numeric (0=Female, 1=Male)
    combined_measurements['gender'] = combined_measurements['gender'].map({'Female': 0, 'Male': 1})
    
    # Rename columns to match output schema
    combined_measurements = combined_measurements.rename(columns={
        'subject_id': 'patient_id',
        'icustay_id': 'icu_stay_id'
    })
    
    # Pivot data to have one row per timepoint with all features as columns
    logger.info("Pivoting data to wide format...")
    pivoted_data = combined_measurements.pivot_table(
        index=['patient_id', 'icu_stay_id', 'time_offset', 'age', 'gender'],
        columns='feature_name',
        values='feature_value',
        aggfunc='mean'  # If multiple measurements at same time, take mean
    ).reset_index()
    
    logger.info(f"Extracted eICU data: {len(pivoted_data)} timepoints across {pivoted_data['icu_stay_id'].nunique()} ICU stays")
    return pivoted_data

def validate_output_data(df, dataset_name):
    """Validate the extracted data"""
    logger.info(f"Validating {dataset_name} data...")
    
    # Basic checks
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Number of unique ICU stays: {df['icu_stay_id'].nunique()}")
    logger.info(f"Total timepoints: {len(df)}")
    
    # Check for required columns
    required_cols = ['patient_id', 'icu_stay_id', 'time_offset', 'age', 'gender']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
    
    # Check feature columns
    feature_cols = [col for col in df.columns if col not in required_cols]
    logger.info(f"Found {len(feature_cols)} feature columns")
    logger.info(f"Feature columns: {feature_cols}")
    
    # Check missing values
    logger.info("Missing value percentages for features:")
    for col in feature_cols:
        if col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            logger.info(f"  {col}: {missing_pct:.1f}%")
    
    # Check time offset range
    if 'time_offset' in df.columns:
        logger.info(f"Time offset range: {df['time_offset'].min()} - {df['time_offset'].max()} minutes")
    
    # Check age and gender distributions
    if 'age' in df.columns:
        logger.info(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f}")
    
    if 'gender' in df.columns:
        gender_counts = df.groupby('icu_stay_id')['gender'].first().value_counts()
        logger.info(f"Gender distribution (unique stays): {dict(gender_counts)}")
    
    return True

def create_sample_data():
    """Create sample BSI data for testing when database is not available"""
    logger.info("Creating sample BSI data for testing...")
    
    np.random.seed(42)
    
    # Create sample data with multiple timepoints per patient
    n_patients = 50
    timepoints_per_patient = 20
    
    patient_ids_mimic = np.random.randint(1, 50000, n_patients)
    patient_ids_eicu = np.random.randint(50001, 100000, n_patients)
    
    def generate_patient_data(patient_ids, dataset_name):
        all_rows = []
        
        for i, pid in enumerate(patient_ids):
            icu_stay_id = i + 1 if dataset_name == 'MIMIC' else i + 1001
            age = np.random.normal(65, 15)
            age = np.clip(age, 18, 89)
            gender = np.random.choice([0, 1])
            
            # Generate timepoints (1440 to 4320 minutes = 24-72 hours)
            time_offsets = sorted(np.random.choice(range(1440, 4321, 30), timepoints_per_patient, replace=False))
            
            for time_offset in time_offsets:
                row = {
                    'patient_id': pid,
                    'icu_stay_id': icu_stay_id,
                    'time_offset': time_offset,
                    'age': age,
                    'gender': gender
                }
                
                # Add feature values (with some missing values)
                feature_ranges = {
                    'Heart Rate': (40, 150),
                    'Respiratory Rate': (8, 40),
                    'O2 saturation pulseoxymetry': (85, 100),
                    'Temperature': (35, 40),
                    'WBC': (2, 25),
                    'Sodium': (125, 155),
                    'Creatinine': (0.5, 8.0),
                    'Hematocrit': (20, 50),
                    'Hemoglobin': (7, 17),
                    'Potassium': (2.5, 6.0)
                }
                
                for feature, (min_val, max_val) in feature_ranges.items():
                    # 70% chance of having a value
                    if np.random.random() < 0.7:
                        row[feature] = np.random.uniform(min_val, max_val)
                    else:
                        row[feature] = np.nan
                
                all_rows.append(row)
        
        return pd.DataFrame(all_rows)
    
    mimic_df = generate_patient_data(patient_ids_mimic, 'MIMIC')
    eicu_df = generate_patient_data(patient_ids_eicu, 'eICU')
    
    return mimic_df, eicu_df

def main():
    """Main function to extract BSI features"""
    parser = argparse.ArgumentParser(description='Extract BSI features from MIMIC-IV and eICU-CRD')
    parser.add_argument('--mimic-only', action='store_true', help='Extract only MIMIC data')
    parser.add_argument('--eicu-only', action='store_true', help='Extract only eICU data')
    parser.add_argument('--validate', action='store_true', help='Validate extracted data', default=True)
    parser.add_argument('--sample-size', type=int, help='Limit number of ICU stays to process (for testing)')
    parser.add_argument('--sample', action='store_true', help='Create sample data for testing (no database required)')
    parser.add_argument('--metadata-only', action='store_true', help='Extract only units metadata (no full data extraction)')
    
    args = parser.parse_args()
    
    # Load configuration (skip if using sample data)
    if not args.sample:
        try:
            config = load_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return
    
    logger.info("=== BSI Feature Extraction for Domain Translation ===")
    logger.info(f"Target features: {len(BSI_FEATURES)}")
    logger.info("Output format: Raw time-series measurements (not aggregated)")
    logger.info("Time window: 24-72 hours after ICU admission")
    
    # Handle sample data creation
    if args.sample:
        logger.info("Creating sample data for testing...")
        mimic_data, eicu_data = create_sample_data()
        
        # Save sample data
        mimic_output_path = "/bigdata/omerg/Thesis/EHR_Translator/poc_translator/mimic_bsi_features.csv"
        eicu_output_path = "/bigdata/omerg/Thesis/EHR_Translator/poc_translator/eicu_bsi_features.csv"
        
        mimic_data.to_csv(mimic_output_path, index=False)
        eicu_data.to_csv(eicu_output_path, index=False)
        
        logger.info(f"Sample MIMIC data saved to: {mimic_output_path}")
        logger.info(f"Sample eICU data saved to: {eicu_output_path}")
        
        if args.validate:
            validate_output_data(mimic_data, "MIMIC Sample")
            validate_output_data(eicu_data, "eICU Sample")
        
        logger.info("=== Sample BSI Feature Generation Complete ===")
        return
    
    # Handle metadata-only extraction
    if args.metadata_only:
        logger.info("=== Extracting Units Metadata Only ===")
        
        try:
            # Extract MIMIC metadata
            if not args.eicu_only:
                logger.info("Connecting to MIMIC-IV database...")
                mimic_engine = create_db_connection(config['db']['mimic_conn'])
                
                mimic_metadata = extract_mimic_units_metadata(mimic_engine)
                
                if not mimic_metadata.empty:
                    # Save MIMIC metadata
                    mimic_metadata_path = os.path.join(config['paths']['output_dir'], "mimic_feature_units.csv")
                    mimic_metadata.to_csv(mimic_metadata_path, index=False)
                    logger.info(f"MIMIC units metadata saved to: {mimic_metadata_path}")
                    logger.info(f"Extracted units for {mimic_metadata['feature_name'].nunique()} features")
                else:
                    logger.warning("No MIMIC metadata extracted")
            
            # Extract eICU metadata
            if not args.mimic_only:
                logger.info("Connecting to eICU-CRD database...")
                eicu_engine = create_db_connection(config['db']['eicu_conn'])
                
                eicu_metadata = extract_eicu_units_metadata(eicu_engine)
                
                if not eicu_metadata.empty:
                    # Save eICU metadata
                    eicu_metadata_path = os.path.join(config['paths']['output_dir'], "eicu_feature_units.csv")
                    eicu_metadata.to_csv(eicu_metadata_path, index=False)
                    logger.info(f"eICU units metadata saved to: {eicu_metadata_path}")
                    logger.info(f"Extracted units for {eicu_metadata['feature_name'].nunique()} features")
                else:
                    logger.warning("No eICU metadata extracted")
            
            logger.info("=== Units Metadata Extraction Complete ===")
            return
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            raise
    
    try:
        # Extract MIMIC data
        if not args.eicu_only:
            logger.info("Connecting to MIMIC-IV database...")
            mimic_engine = create_db_connection(config['db']['mimic_conn'])
            
            mimic_data = extract_mimic_data(mimic_engine)
            
            if not mimic_data.empty:
                if args.sample_size:
                    # Limit by number of ICU stays
                    unique_stays = mimic_data['icu_stay_id'].unique()[:args.sample_size]
                    mimic_data = mimic_data[mimic_data['icu_stay_id'].isin(unique_stays)]
                    logger.info(f"Limited MIMIC data to {len(unique_stays)} ICU stays")
                
                # Save MIMIC data
                mimic_output_path = os.path.join(config['paths']['output_dir'], "mimic_bsi_features.csv")
                mimic_data.to_csv(mimic_output_path, index=False)
                logger.info(f"MIMIC data saved to: {mimic_output_path}")
                
                if args.validate:
                    validate_output_data(mimic_data, "MIMIC")
        
        # Extract eICU data
        if not args.mimic_only:
            logger.info("Connecting to eICU-CRD database...")
            eicu_engine = create_db_connection(config['db']['eicu_conn'])
            
            eicu_data = extract_eicu_data(eicu_engine)
            
            if not eicu_data.empty:
                if args.sample_size:
                    # Limit by number of ICU stays
                    unique_stays = eicu_data['icu_stay_id'].unique()[:args.sample_size]
                    eicu_data = eicu_data[eicu_data['icu_stay_id'].isin(unique_stays)]
                    logger.info(f"Limited eICU data to {len(unique_stays)} ICU stays")
                
                # Save eICU data
                eicu_output_path = os.path.join(config['paths']['output_dir'], "eicu_bsi_features.csv")
                eicu_data.to_csv(eicu_output_path, index=False)
                logger.info(f"eICU data saved to: {eicu_output_path}")
                
                if args.validate:
                    validate_output_data(eicu_data, "eICU")
        
        logger.info("=== BSI Feature Extraction Complete ===")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()

