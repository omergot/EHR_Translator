# Database Analysis and Feature Extraction Scripts

## explore_databases.py

This script provides comprehensive exploration and analysis of both eICU and MIMIC-IV database structures to help you understand what data and features are available.

### What it does:

1. **Database Structure Analysis**:
   - Lists all accessible schemas and tables in both databases
   - Analyzes column names, data types, and constraints
   - Counts rows in each table (where accessible)
   - Samples data to understand content structure

2. **Feature Potential Assessment**:
   - Categorizes columns as numerical, temporal, or categorical
   - Identifies high/medium/low potential features based on column names and types
   - Highlights measurement-related columns (values, results, rates, etc.)
   - Suggests which tables are most valuable for feature extraction

3. **Comprehensive Reporting**:
   - Generates detailed Markdown report with database structure summary
   - Creates JSON file with raw analysis data
   - Provides feature extraction recommendations
   - Shows table statistics and potential feature counts

### Usage:

```bash
cd /bigdata/omerg/Thesis/poc_translator/scripts
python3 explore_databases.py
```

### Output Files:

The script creates timestamped files in `database_analysis/` directory:
- `database_structure_report_YYYYMMDD_HHMMSS.md` - Human-readable summary report
- `database_structure_data_YYYYMMDD_HHMMSS.json` - Raw analysis data

### Report Contents:

**Executive Summary**: Overall statistics across both databases

**Per Database Analysis**:
- Schema and table inventories
- Column counts and data types
- Row counts and accessibility status
- Feature potential assessment

**Recommendations**: 
- Top tables for feature extraction ranked by potential feature count
- Specific column recommendations for each database

**Example Output**:
```
eICU Database Analysis
├── eicu_crd schema
│   ├── patient (73 columns, 200,859 rows) - 8 high potential features
│   ├── lab (7 columns, 27,872,575 rows) - 3 high potential features  
│   ├── nursecharting (8 columns, 74,170,943 rows) - 2 high potential features
│   └── ... more tables
└── other schemas...
```

## extract_all_features.py

This script extracts all numerical measurements from both eICU and MIMIC-IV databases without BSI-specific constraints.

### What it does:

1. **Removes BSI-specific constraints** that were in the original queries:
   - No minimum ICU stay duration requirements (48 hours)
   - No blood culture site filtering (`culturesite = 'Blood, Central Line' or culturesite = 'Blood, Venipuncture'`)
   - No culture date filtering relative to measurements
   - No unit conversions specific to BSI analysis
   - No BSI organism classification
   - No hospital filtering based on minimum case counts
   - No culture offset requirements

2. **Extracts all numerical measurements** from:
   - **eICU**: Lab results, nursing charts, respiratory charts, custom lab results, and patient demographics
   - **MIMIC-IV**: Lab events, chart events, OMR results (height/weight), and patient demographics

3. **Creates simplified cohort tables** that include all ICU stays without BSI filtering

### Usage:

```bash
cd /bigdata/omerg/Thesis/poc_translator/scripts
python3 extract_all_features.py
```

### Requirements:

- Python 3.7+
- PostgreSQL database connections configured in `conf/config.yml`
- Required packages: pandas, psycopg2, yaml

### Output:

The script creates two timestamped CSV files in the `data/` directory:
- `eicu_all_features_YYYYMMDD_HHMMSS.csv`
- `mimic_all_features_YYYYMMDD_HHMMSS.csv`

Each file contains columns:
- `example_id`: Patient stay identifier
- `person_id`: Patient identifier
- `feature_name`: Name of the measurement/feature
- `feature_value`: Numerical value
- `feature_start_date`: Timestamp of the measurement

### Features extracted:

The script extracts all available numerical measurements, including but not limited to:
- Laboratory values (complete blood count, chemistry panels, etc.)
- Vital signs (heart rate, blood pressure, temperature, etc.)
- Ventilator settings and respiratory measurements
- Patient demographics (age, height, weight)
- Other clinical measurements available in the respective databases

### Notes:

- Uses temporary schemas to avoid conflicts with existing tables
- Automatically cleans up temporary tables after execution
- Only extracts numerical values (filters out text/categorical data)
- No time window restrictions - extracts all measurements during ICU stays
- Includes comprehensive error handling and progress reporting

## compare_distributions.py

This script analyzes the quality of distribution matching between trained and untrained models - the real test of translation performance.

### What it does:

1. **Loads trained and untrained models**:
   - Compares a trained Cycle Encoder-Decoder model with an untrained baseline
   - Tests whether training actually improves distribution matching

2. **Computes distribution matching metrics**:
   - **KS (Kolmogorov-Smirnov) statistics**: Measures distribution similarity (KS < 0.1 = good match)
   - **Wasserstein distance**: Measures distributional distance between translated and target data
   - Compares both metrics for: translated vs target, and source vs target (baseline)
   - Identifies which features are well-matched vs poorly-matched

3. **Per-feature analysis**:
   - Evaluates each clinical feature separately
   - Provides detailed statistics on translation quality
   - Identifies improvement over baseline (source→target)

### Usage:

```bash
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python scripts/compare_distributions.py
```

### Output:

- **Console output**: Per-feature KS and Wasserstein statistics
- **`dist.txt`**: Detailed distribution analysis results
- Summary statistics showing:
  - Number of features with KS < 0.1 (well-matched)
  - Average improvement over baseline
  - Worst-performing features that need attention

### When to Use:

- After training to validate that the model learned meaningful translations
- To compare different model configurations or training runs
- To identify which features need additional training focus
- To verify that conditional Wasserstein loss is effective

**Note**: Reconstruction quality alone doesn't indicate good translation. This script tests the critical property: does translated data match the target distribution?

## compare_trained_vs_untrained.py

This script provides a direct comparison between trained and untrained models to measure the impact of training on translation quality.

### What it does:

1. **Side-by-side model comparison**:
   - Loads both trained and untrained (random initialization) models
   - Translates the same data using both models
   - Compares translation quality metrics

2. **Distribution matching analysis**:
   - KS statistics for both models
   - Wasserstein distances for both models
   - Improvement metrics (trained vs untrained)

3. **Statistical significance testing**:
   - Tests whether improvements are statistically significant
   - Identifies features where training made the most impact

### Usage:

```bash
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python scripts/compare_trained_vs_untrained.py
```

### Output:

Console output showing:
- Per-feature comparison of KS and Wasserstein metrics
- Percentage improvement from training
- Features with significant improvement
- Features that still need work

### When to Use:

- To validate that training is effective
- To quantify the benefit of your training approach
- To identify which features benefit most from training
- For ablation studies and model comparisons

## Script Comparison

| Script | Purpose | Output | When to Use |
|--------|---------|--------|-------------|
| `explore_databases.py` | **Database Structure Analysis** | Markdown report + JSON data | **First step** - understand what's available in the databases |
| `demo_exploration.py` | **Analysis of Exploration Results** | Console output with recommendations | **Second step** - get actionable insights from exploration |
| `extract_all_features.py` | **Feature Data Extraction** | CSV files with actual data | **Third step** - extract the actual feature data for analysis |
| `extract_poc_features.py` | **POC Feature Extraction** | CSV files with aligned POC features | **Main pipeline** - extract point-of-care features for training |
| `compare_distributions.py` | **Distribution Matching Analysis** | dist.txt with detailed statistics | **After training** - validate translation quality |
| `compare_trained_vs_untrained.py` | **Training Impact Analysis** | Console output with comparison | **After training** - measure training effectiveness |

## demo_exploration.py

A demonstration script that shows how to programmatically analyze the JSON output from `explore_databases.py`.

### What it does:

- Loads the most recent database exploration results
- Analyzes and ranks tables by feature potential
- Shows feature distribution statistics across schemas
- Generates specific recommendations for feature extraction
- Provides actionable insights for your data pipeline

### Usage:

```bash
cd /bigdata/omerg/Thesis/poc_translator/scripts
# First run the exploration (if not done already)
python3 explore_databases.py
# Then analyze the results
python3 demo_exploration.py
```

### Recommended Workflow:

1. **First**, run `explore_databases.py` to understand the database structure and identify the most valuable tables for your research
2. **Then**, run `demo_exploration.py` to get programmatic analysis and specific recommendations  
3. **Next**, run `extract_all_features.py` to extract the actual numerical features for your machine learning pipeline
4. **Finally**, use the exploration reports to guide any additional targeted feature extraction if needed
