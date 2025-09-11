#!/usr/bin/env python3
"""
SQL Query Builder for MIMIC and eICU Feature Extraction
Reads aligned feature CSVs and generates SQL queries for 24-hour aggregation.
"""

import pandas as pd
import yaml
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def load_config():
    """Load configuration from config.yml"""
    config_path = Path(__file__).parent.parent / "conf" / "config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_aligned_features(config):
    """Load aligned feature CSVs"""
    eicu_features = pd.read_csv(config['paths']['aligned_eicu_csv'])
    mimic_features = pd.read_csv(config['paths']['aligned_mimic_csv'])
    
    # Create feature mapping
    feature_mapping = {}
    for _, row in eicu_features.iterrows():
        eicu_name = row['Feature_Name']
        index = row['Index']
        mimic_name = mimic_features.iloc[index]['Feature_Name']
        feature_mapping[eicu_name] = mimic_name
    
    return eicu_features, mimic_features, feature_mapping

def get_feature_mappings():
    """Try to get feature mappings from analyze_features.py"""
    try:
        # This would need to be adapted based on actual analyze_features.py structure
        # For now, we'll create placeholder mappings
        return {
            'MIMIC_ITEMID_MAP': {},
            'EICU_COLUMN_MAP': {}
        }
    except ImportError:
        print("Warning: Could not import analyze_features.py")
        return {
            'MIMIC_ITEMID_MAP': {},
            'EICU_COLUMN_MAP': {}
        }

def generate_mimic_sql(features_df, itemid_map=None, omop_schema="omop", cohort_table=None):
    """Generate MIMIC-IV SQL query for feature extraction using native MIMIC-IV tables and cohort table"""
    
    # Default itemid mappings (user should update these)
    default_itemids = {
        # Vital signs
        'Heart Rate': 220045,
        'Respiratory Rate': 220210,
        'Temperature': 223761,
        'Temperature (F)': 223761,  # Same as Temperature
        'Temperature Fahrenheit': 223761,  # Same as Temperature
        
        # Blood pressure
        'Non-Invasive BP Systolic': 220179,  # Using systolic BP itemid
        'Non Invasive Blood Pressure systolic': 220179,
        'Non-Invasive BP Diastolic': 220180,  # Using diastolic BP itemid
        'Non Invasive Blood Pressure diastolic': 220180,
        'Non-Invasive BP Mean': 220181,  # Using mean BP itemid
        'Non Invasive Blood Pressure mean': 220181,
        
        # Oxygenation
        'O2 Sat (%)': 220277,  # Using O2 saturation itemid
        'O2 saturation pulseoxymetry': 220277,
        'FiO2': 223835,
        'Inspired O2 Fraction': 223835,
        'paO2': 50821,  # Using PO2 itemid
        'pO2': 50821,
        
        # Blood gases
        'pH': 220274,
        'PCO2': 50818,
        
        # Electrolytes
        'sodium': 50983,
        'Sodium': 50983,
        'potassium': 50971,
        'Potassium': 50971,
        'magnesium': 50960,  # Magnesium itemid
        'Magnesium': 50960,
        
        # Kidney function
        'creatinine': 50912,
        'Creatinine': 50912,
        'BUN': 51066,
        'Urea Nitrogen': 51066,
        
        # Liver function
        'albumin': 50862,
        'Albumin': 50862,
        'total bilirubin': 50861,
        'Bilirubin': 50861,
        'AST (SGOT)': 50876,
        'Asparate Aminotransferase (AST)': 50876,
        'ALT (SGPT)': 50861,  # Using ALT itemid
        'Alanine Aminotransferase (ALT)': 50861,
        'alkaline phos.': 50863,
        'Alkaline Phosphatase': 50863,
        
        # Blood counts
        'Hgb': 50360,
        'Hemoglobin': 50360,
        'Hct': 50370,
        'Hematocrit': 50370,
        'WBC x 1000': 51300,
        'WBC': 51300,
        'RBC': 50310,  # RBC count itemid
        'RDW': 50320,  # RDW itemid
        'MCHC': 50340,  # MCHC itemid
        'MCV': 50330,  # MCV itemid
        '-lymphs': 51301,  # Lymphocytes itemid
        'Lymphocytes': 51301,
        
        # Coagulation
        'PT': 51274,
        'PT - INR': 51237,
        'INR(PT)': 51237,
        
        # Other lab values
        'lactate': 50813,
        'Lactate': 50813,
        'CRP': 50867,  # C-reactive protein itemid
        'C-Reactive Protein': 50867,
        'LDH': 50954,  # LDH itemid
        'Lactate Dehydrogenase (LD)': 50954,
        
        # Glasgow Coma Scale components
        'Eyes': 220739,
        'GCS - Eye Opening': 220739,
        'Motor': 223901,
        'GCS - Motor Response': 223901,
        'Verbal': 223900,
        'GCS - Verbal Response': 223900,
        
        # Ventilation
        'Mean Airway Pressure': 220774,  # Airway pressure itemid
        'Exhaled TV (patient)': 224688,  # Tidal volume itemid
        'Tidal Volume (observed)': 224688,
        
        # Position
        'Head of Bed Elevation': 228096,  # Head of bed itemid
        'Head of Bed': 228096,
    }
    
    if itemid_map:
        default_itemids.update(itemid_map)
    
    # Generate SQL
    sql_parts = []
    
    # Header
    sql_parts.append(f"""
-- MIMIC-IV Feature Extraction Query using native MIMIC-IV tables and cohort table
-- Extracts 24-hour aggregated features for each ICU stay in the BSI cohort

WITH cohort_patients AS (
  SELECT DISTINCT person_id, example_id
  FROM {cohort_table if cohort_table else 'mimiciv_bsi_100_2h_test.__mimiciv_bsi_100_2h_cohort'}
),
icu_window AS (
  SELECT 
    i.stay_id as icustay_id, 
    i.subject_id as subject_id, 
    i.hadm_id as hadm_id, 
    i.intime,
    i.intime + INTERVAL '24 hours' as intime_plus_24h
  FROM mimiciv_icu.icustays i
  JOIN cohort_patients c ON i.stay_id = c.example_id
  WHERE i.intime IS NOT NULL
)
SELECT""")
    
    # Generate feature aggregations
    for _, row in features_df.iterrows():
        feature_name = row['Feature_Name']
        feature_idx = row['Index']
        
        # Map feature name to itemid
        itemid = default_itemids.get(feature_name, f"ITEMID_{feature_idx}")
        
        # Generate aggregation for this feature
        feature_sql = f"""
  -- {feature_name} (Index: {feature_idx})
  AVG(CASE WHEN m.itemid = {itemid} THEN m.valuenum END) AS feature_{feature_idx}_mean,
  MIN(CASE WHEN m.itemid = {itemid} THEN m.valuenum END) AS feature_{feature_idx}_min,
  MAX(CASE WHEN m.itemid = {itemid} THEN m.valuenum END) AS feature_{feature_idx}_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = {itemid} AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = {itemid} AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_{feature_idx}_last,
  COUNT(CASE WHEN m.itemid = {itemid} THEN 1 END) AS feature_{feature_idx}_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = {itemid} THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_{feature_idx}_missing,"""
        
        sql_parts.append(feature_sql)
    
    # Final query structure
    sql_parts.append(f"""
  -- Additional metadata
  i.icustay_id,
  i.subject_id,
  i.hadm_id
FROM icu_window i
LEFT JOIN (
  SELECT subject_id, hadm_id, itemid, valuenum, charttime FROM mimiciv_hosp.labevents 
  WHERE valuenum IS NOT NULL
  UNION ALL
  SELECT subject_id, hadm_id, itemid, valuenum, charttime FROM mimiciv_icu.chartevents
  WHERE valuenum IS NOT NULL
) m ON (m.subject_id = i.subject_id AND m.hadm_id = i.hadm_id)
  AND m.charttime >= i.intime 
  AND m.charttime < i.intime_plus_24h
GROUP BY i.icustay_id, i.subject_id, i.hadm_id, i.intime, i.intime_plus_24h
ORDER BY i.icustay_id;
""")
    
    return "\n".join(sql_parts)

def generate_eicu_sql(features_df, column_map=None, cohort_table=None):
    """Generate eICU SQL query for feature extraction using cohort table"""
    
    # Default column mappings (user should update these)
    default_columns = {
        # Vital signs
        'Heart Rate': 'heartrate',
        'Respiratory Rate': 'respiration',
        'Temperature': 'temperature',
        'Temperature (F)': 'temperature',  # Same as Temperature
        'Temperature Fahrenheit': 'temperature',  # Same as Temperature
        
        # Blood pressure
        'Non-Invasive BP Systolic': 'systemicsystolic',
        'Non Invasive Blood Pressure systolic': 'systemicsystolic',
        'Non-Invasive BP Diastolic': 'systemicdiastolic',
        'Non Invasive Blood Pressure diastolic': 'systemicdiastolic',
        'Non-Invasive BP Mean': 'systemicmean',
        'Non Invasive Blood Pressure mean': 'systemicmean',
        
        # Oxygenation
        'O2 Sat (%)': 'sao2',
        'O2 saturation pulseoxymetry': 'sao2',
        'FiO2': 'fio2',
        'Inspired O2 Fraction': 'fio2',
        'paO2': 'po2',  # Using PO2 column
        'pO2': 'po2',
        
        # Blood gases
        'pH': 'ph',
        'PCO2': 'pco2',
        
        # Electrolytes
        'sodium': 'sodium',
        'Sodium': 'sodium',
        'potassium': 'potassium',
        'Potassium': 'potassium',
        'magnesium': 'magnesium',
        'Magnesium': 'magnesium',
        
        # Kidney function
        'creatinine': 'creatinine',
        'Creatinine': 'creatinine',
        'BUN': 'bun',
        'Urea Nitrogen': 'bun',
        
        # Liver function
        'albumin': 'albumin',
        'Albumin': 'albumin',
        'total bilirubin': 'bilirubin',
        'Bilirubin': 'bilirubin',
        'AST (SGOT)': 'ast',
        'Asparate Aminotransferase (AST)': 'ast',
        'ALT (SGPT)': 'alt',
        'Alanine Aminotransferase (ALT)': 'alt',
        'alkaline phos.': 'alkalinephos',
        'Alkaline Phosphatase': 'alkalinephos',
        
        # Blood counts
        'Hgb': 'hemoglobin',
        'Hemoglobin': 'hemoglobin',
        'Hct': 'hematocrit',
        'Hematocrit': 'hematocrit',
        'WBC x 1000': 'wbc',
        'WBC': 'wbc',
        'RBC': 'rbc',
        'RDW': 'rdw',
        'MCHC': 'mchc',
        'MCV': 'mcv',
        '-lymphs': 'lymphocytes',
        'Lymphocytes': 'lymphocytes',
        
        # Coagulation
        'PT': 'pt',
        'PT - INR': 'inr',
        'INR(PT)': 'inr',
        
        # Other lab values
        'lactate': 'lactate',
        'Lactate': 'lactate',
        'CRP': 'crp',
        'C-Reactive Protein': 'crp',
        'LDH': 'ldh',
        'Lactate Dehydrogenase (LD)': 'ldh',
        
        # Glasgow Coma Scale components
        'Eyes': 'gcseyes',
        'GCS - Eye Opening': 'gcseyes',
        'Motor': 'gcsmotor',
        'GCS - Motor Response': 'gcsmotor',
        'Verbal': 'gcsverbal',
        'GCS - Verbal Response': 'gcsverbal',
        
        # Ventilation
        'Mean Airway Pressure': 'meanairwaypressure',
        'Exhaled TV (patient)': 'tidalvolume',
        'Tidal Volume (observed)': 'tidalvolume',
        
        # Position
        'Head of Bed Elevation': 'headofbed',
        'Head of Bed': 'headofbed',
    }
    
    if column_map:
        default_columns.update(column_map)
    
    # Generate SQL
    sql_parts = []
    
    # Header
    sql_parts.append(f"""
-- eICU Feature Extraction Query using cohort table
-- Extracts 24-hour aggregated features for each ICU stay in the BSI cohort

WITH cohort_patients AS (
  SELECT DISTINCT example_id as patientunitstayid
  FROM {cohort_table if cohort_table else 'eicu_bsi_100_2h_test.__eicu_bsi_100_2h_cohort'}
),
stays AS (
  SELECT 
    p.patientunitstayid
  FROM eicu_crd.patient p
  JOIN cohort_patients c ON p.patientunitstayid = c.patientunitstayid
  WHERE p.patientunitstayid IS NOT NULL
)
SELECT""")
    
    # Generate feature aggregations
    for _, row in features_df.iterrows():
        feature_name = row['Feature_Name']
        feature_idx = row['Index']
        
        # Map feature name to column
        column_name = default_columns.get(feature_name, None)
        
        # Available columns in eicu_crd.vitalperiodic
        available_vitalperiodic_cols = [
            'heartrate', 'respiration', 'temperature', 'sao2',
            'systemicsystolic', 'systemicdiastolic', 'systemicmean',
            'pasystolic', 'padiastolic', 'pamean', 'cvp', 'icp', 'etco2'
        ]
        
        if column_name and column_name in available_vitalperiodic_cols:
            # Feature is available in vitalperiodic - use real data
            feature_sql = f"""
  -- {feature_name} (Index: {feature_idx}) - Available in vitalperiodic
  AVG(CASE WHEN v.{column_name} IS NOT NULL THEN v.{column_name} END) AS feature_{feature_idx}_mean,
  MIN(CASE WHEN v.{column_name} IS NOT NULL THEN v.{column_name} END) AS feature_{feature_idx}_min,
  MAX(CASE WHEN v.{column_name} IS NOT NULL THEN v.{column_name} END) AS feature_{feature_idx}_max,
  (SELECT v2.{column_name}
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.{column_name} IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_{feature_idx}_last,
  COUNT(CASE WHEN v.{column_name} IS NOT NULL THEN 1 END) AS feature_{feature_idx}_count,
  CASE WHEN COUNT(CASE WHEN v.{column_name} IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_{feature_idx}_missing,"""
        else:
            # Feature not available in vitalperiodic - use NULL placeholders
            feature_sql = f"""
  -- {feature_name} (Index: {feature_idx}) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_{feature_idx}_mean,
  NULL AS feature_{feature_idx}_min,
  NULL AS feature_{feature_idx}_max,
  NULL AS feature_{feature_idx}_last,
  0 AS feature_{feature_idx}_count,
  1 AS feature_{feature_idx}_missing,"""
        
        sql_parts.append(feature_sql)
    
    # Final query structure
    sql_parts.append("""
  -- Additional metadata
  s.patientunitstayid as icustay_id
FROM stays s
LEFT JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = s.patientunitstayid
  AND v.observationoffset >= 0
  AND v.observationoffset < 24 * 60
GROUP BY s.patientunitstayid
ORDER BY s.patientunitstayid;
""")
    
    return "\n".join(sql_parts)

def create_placeholders_json(features_df, output_dir):
    """Create placeholders JSON for user to fill in itemids/columns"""
    placeholders = {
        'mimic_itemids': {},
        'eicu_columns': {},
        'notes': 'Fill in the appropriate itemids/columns for your database schema'
    }
    
    for _, row in features_df.iterrows():
        feature_name = row['Feature_Name']
        feature_idx = row['Index']
        
        placeholders['mimic_itemids'][feature_name] = {
            'itemid': f"ITEMID_{feature_idx}",
            'description': f"ItemID for {feature_name} in MIMIC chartevents table"
        }
        
        placeholders['eicu_columns'][feature_name] = {
            'column': f"COLUMN_{feature_idx}",
            'description': f"Column name for {feature_name} in eICU vitalperiodic table"
        }
    
    placeholder_path = Path(output_dir) / "placeholders.json"
    with open(placeholder_path, 'w') as f:
        json.dump(placeholders, f, indent=2)
    
    return placeholder_path

def main():
    """Main function to generate SQL queries"""
    print("Generating SQL queries for MIMIC and eICU feature extraction...")
    
    # Load configuration
    config = load_config()
    
    # Load aligned features
    eicu_features, mimic_features, feature_mapping = load_aligned_features(config)
    
    print(f"Loaded {len(eicu_features)} aligned features")
    print(f"EICU features: {len(eicu_features)}")
    print(f"MIMIC features: {len(mimic_features)}")
    
    # Try to get feature mappings
    mappings = get_feature_mappings()
    
    # Generate SQL queries
    print("\nGenerating MIMIC SQL query...")
    omop_schema = config.get('db', {}).get('omop_schema', 'omop')
    mimic_cohort_table = config.get('db', {}).get('cohort_tables', {}).get('mimic')
    mimic_sql = generate_mimic_sql(mimic_features, mappings.get('MIMIC_ITEMID_MAP'), omop_schema, mimic_cohort_table)
    
    print("Generating eICU SQL query...")
    eicu_cohort_table = config.get('db', {}).get('cohort_tables', {}).get('eicu')
    eicu_sql = generate_eicu_sql(eicu_features, mappings.get('EICU_COLUMN_MAP'), eicu_cohort_table)
    
    # Save SQL files
    output_dir = Path(config['paths']['tmp_sql_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mimic_sql_path = output_dir / "mimic_query.sql"
    eicu_sql_path = output_dir / "eicu_query.sql"
    
    with open(mimic_sql_path, 'w') as f:
        f.write(mimic_sql)
    
    with open(eicu_sql_path, 'w') as f:
        f.write(eicu_sql)
    
    # Create placeholders JSON
    placeholder_path = create_placeholders_json(mimic_features, output_dir)
    
    print(f"\nSQL queries generated successfully!")
    print(f"MIMIC query: {mimic_sql_path}")
    print(f"eICU query: {eicu_sql_path}")
    print(f"Placeholders: {placeholder_path}")
    print(f"\nNext steps:")
    print(f"1. Review and update placeholders.json with correct itemids/columns")
    print(f"2. Update SQL queries if needed for your specific schema")
    print(f"3. Run data/raw_extractors.py to execute the queries")

if __name__ == "__main__":
    main()
