# Shared Features Discovery Tool

## Overview

The `find_shared_features.py` script discovers and maps all shared clinical features between MIMIC-IV and eICU-CRD databases. This tool goes beyond the predefined 10 POC features to identify all potential overlapping measurements.

## Features

1. **Comprehensive Feature Extraction**
   - Extracts all chartevents, labevents, and procedureevents from MIMIC-IV
   - Extracts all vitals, labs, and nurse charting data from eICU-CRD

2. **Intelligent Matching**
   - **Exact matching**: Finds features with identical normalized names
   - **Fuzzy matching**: Uses string similarity (configurable threshold) to find similar features
   - **Manual mappings**: Includes known clinical equivalences that may differ in naming

3. **Categorization**
   - Automatically categorizes matched features into clinical domains:
     - Vital Signs (Cardiovascular, Blood Pressure, Respiratory, Oxygenation, Temperature)
     - Laboratory (Glucose, Electrolytes, Renal, Hematology, Hepatic, Blood Gas)
     - Anthropometric
     - Neurological
     - Other

4. **Rich Output**
   - Shared features CSV with match details and similarity scores
   - Complete feature catalogs for both databases
   - Detailed text report with summaries and top matches

## Usage

### Basic Usage

```bash
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python scripts/find_shared_features.py
```

### Advanced Options

```bash
# Adjust similarity threshold for fuzzy matching (default: 0.75)
python scripts/find_shared_features.py --similarity-threshold 0.8

# Specify custom output directory
python scripts/find_shared_features.py --output-dir /path/to/output
```

### Parameters

- `--similarity-threshold`: Float between 0.0-1.0 for fuzzy matching
  - Lower values (0.6-0.7): More matches, but may include false positives
  - Higher values (0.8-0.9): Fewer but more precise matches
  - Default: 0.75

- `--output-dir`: Custom output directory (defaults to config.yml setting)

## Output Files

The script generates three main outputs in the configured output directory:

### 1. `shared_features.csv`
Complete list of matched features with columns:
- `match_type`: exact, fuzzy, or manual
- `similarity`: Match confidence score (0.0-1.0)
- `mimic_feature_id`: MIMIC itemid
- `mimic_feature_name`: MIMIC feature name
- `mimic_source_table`: chartevents, labevents, etc.
- `mimic_category`: MIMIC category
- `mimic_unit`: Unit of measurement
- `mimic_num_records`: Number of records in MIMIC
- `eicu_feature_name`: eICU column/feature name
- `eicu_display_name`: eICU display name
- `eicu_source_table`: vitalperiodic, lab, nursecharting, etc.
- `eicu_unit`: Unit of measurement
- `eicu_num_records`: Number of records in eICU
- `category`: Clinical category assigned by the tool

### 2. `mimic_feature_catalog.csv` & `eicu_feature_catalog.csv`
Complete catalogs of all available features in each database.

### 3. `shared_features_report.txt`
Human-readable summary report including:
- Match statistics by type
- Breakdown by clinical category
- Top 20 matches by data availability
- Match details for each feature

## Example Workflow

```bash
# 1. Discover all shared features
python scripts/find_shared_features.py

# 2. Review the shared_features_report.txt for summary
less data/output/shared_features_report.txt

# 3. Examine the CSV for detailed analysis
python -c "import pandas as pd; df = pd.read_csv('data/output/shared_features.csv'); print(df[df['category'] == 'Laboratory - Renal'])"

# 4. Use findings to expand POC features in extract_poc_features.py
```

## Integration with POC Feature Extraction

After identifying shared features, you can:

1. Update `POC_FEATURES` dict in `extract_poc_features.py`
2. Add corresponding mappings to `MIMIC_ITEMID_MAP` and `EICU_COLUMN_MAP`
3. Re-run feature extraction with expanded feature set

Example:
```python
# Based on shared_features.csv findings, add new features:
POC_FEATURES = {
    # ... existing features ...
    'Glucose': 10,
    'Potassium': 11,
    # ... etc
}

MIMIC_ITEMID_MAP = {
    # ... existing mappings ...
    'Glucose': [50809, 50931],  # From feature catalog
    'Potassium': [50822, 50971],
}

EICU_COLUMN_MAP = {
    # ... existing mappings ...
    'Glucose': None,  # From lab table
    'Potassium': None,
}
```

## Notes

- The script requires access to both MIMIC-IV and eICU-CRD databases
- Database connection details are read from `conf/config.yml`
- Processing may take several minutes depending on database size
- Only features with >100 records are included to ensure data quality

## Troubleshooting

### "Failed to create database connection"
- Check that database connection strings in `conf/config.yml` are correct
- Verify you have necessary database access permissions

### "No features found"
- Ensure the database schemas exist (mimiciv_icu, mimiciv_hosp, eicu_crd)
- Check that tables are populated with data

### Too many/few matches
- Adjust `--similarity-threshold` parameter
- Review manual_mappings in the script for known equivalences





