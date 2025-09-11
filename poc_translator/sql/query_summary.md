# Unified Feature Extraction Queries

## Overview

These queries extract 40 clinical features from ALL patients (not limited to ICU or cohorts) in both MIMIC-IV and eICU databases.

## Files

- `mimic_unified_query.sql`: Extracts features from all MIMIC-IV hospital admissions
- `eicu_unified_query.sql`: Extracts features from all eICU patient unit stays

## Key Features

- **Comprehensive Coverage**: Includes ALL patients, not just ICU stays
- **40 Clinical Features**: Complete extraction of laboratory values, vital signs, and clinical assessments
- **24-hour Observation Window**: Uses first 24 hours after admission for each patient
- **Multiple Data Sources**: Accesses lab, chart, nursing, and respiratory data tables
- **Optimized Performance**: Uses CTEs and window functions for efficiency
- **Data Quality Filters**: Removes invalid and negative values

## Data Quality Notes

Some features may have low coverage in certain patient populations:
- Ventilator parameters (only available for mechanically ventilated patients)
- Some specialized lab values may not be routinely ordered
- GCS components may be missing for non-neurological patients

## Usage

Run these queries against your MIMIC-IV and eICU databases respectively. The output includes:
- Mean, min, max values for each feature in the 24-hour window
- Last recorded value
- Count of measurements
- Missing data indicator

## Database Requirements

- MIMIC-IV: Requires access to `mimiciv_hosp` and `mimiciv_icu` schemas
- eICU: Requires access to `eicu_crd` schema

The queries are designed to work with standard MIMIC-IV and eICU database installations.
