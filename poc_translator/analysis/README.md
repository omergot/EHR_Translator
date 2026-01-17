# Distribution Comparison Analysis: MIMIC-IV vs eICU-CRD

## Overview

This directory contains comprehensive distribution comparison analysis between MIMIC-IV and eICU-CRD datasets for 40 aligned BSI (Bloodstream Infection) prediction features.

## Analysis Components

### 1. Per-Feature Marginal Distribution Analysis
**Output**: `results/per_feature_summary.csv`

For each of the 40 features, we computed:

**Statistical Metrics:**
- Basic stats: mean, std, median, IQR
- Percentiles: 1st, 5th, 25th, 50th, 75th, 95th, 99th
- Sample counts

**Distance Metrics:**
- **SMD** (Standardized Mean Difference / Cohen's d) - measures effect size
- **KS Statistic** (Kolmogorov-Smirnov) - tests if distributions are different
- **Wasserstein Distance** - measures distance between distributions
- **PSI** (Population Stability Index) - measures distribution shift

**Alignment Categories:**
- `aligned`: SMD < 0.1 (distributions are very similar)
- `mild shift`: 0.1 ≤ SMD < 0.3 (small difference)
- `moderate shift`: 0.3 ≤ SMD < 0.5 (moderate difference)
- `large shift 🚨`: SMD ≥ 0.5 (substantial difference requiring attention)

**Visualizations**: `plots/per_feature/{feature}_distribution.png`

Each feature has 4 plots:
1. Overlaid histogram
2. Kernel density estimate (KDE)
3. Empirical cumulative distribution function (ECDF)
4. Side-by-side violin plot

### 2. Temporal Frequency Analysis
**Output**: `plots/temporal/{feature}_frequency.png`

For selected critical features (15-20), we analyzed:
- **Measurements per patient**: How many times each feature was measured
- **Time gaps**: Time between consecutive measurements

This reveals:
- Measurement frequency differences between databases
- Sampling rate patterns
- Data availability patterns

### 3. Trajectory Analysis (First 48 Hours)
**Output**: `plots/temporal/{feature}_trajectory.png`

For critical features, we tracked values over time:
- Time bins: 0-6h, 6-12h, 12-18h, 18-24h, 24-36h, 36-48h
- Metrics per bin: mean ± std, median with IQR
- Shows temporal evolution patterns

### 4. Derived Features Analysis
**Computed per patient:**
- `first_value`: First measurement in 48h window
- `last_value`: Last measurement in 48h window
- `min_value`, `max_value`: Range
- `slope`: Rate of change (last - first) / duration

Compare distributions of these derived metrics between MIMIC and eICU.

### 5. Correlation Structure Comparison
**Output**: `plots/correlation/correlation_comparison.png`

For 15-20 key features:
- Computed Pearson correlation matrices using first 24h averages
- Generated 3 heatmaps:
  1. MIMIC correlation matrix
  2. eICU correlation matrix  
  3. Difference matrix (MIMIC - eICU)
- Frobenius norm: quantifies overall correlation difference

## Running the Analysis

### Step 1: Main Distribution Analysis
```bash
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python3 scripts/compare_distributions.py
```

This will:
1. Load MIMIC and eICU data from cache files
2. Apply feature name mappings
3. Apply unit conversions (e.g., C-Reactive Protein: mg/L → mg/dL)
4. Analyze all 40 features
5. Generate plots and summary tables

**Runtime**: ~10-15 minutes for 40 features

### Step 2: Extended Analysis (Optional)
```bash
python3 scripts/analyze_complete.py
```

Runs trajectory, derived features, and correlation analysis.

## Key Files

### Data Inputs
- `/bigdata/omerg/RatchetEHR/tmp/tmp/cache_data_bsi_test_100mimiciv` - MIMIC data
- `/bigdata/omerg/RatchetEHR/tmp/tmp/cache_data_bsi_test_100` - eICU data
- `data/mimic_features_aligned.csv` - MIMIC feature list
- `data/eicu_features_aligned.csv` - eICU feature list
- `feature_units_comparison.csv` - Unit conversion mapping

### Outputs
- `results/per_feature_summary.csv` - Main results table
- `plots/per_feature/` - 40 distribution plots
- `plots/temporal/` - Temporal analysis plots
- `plots/correlation/` - Correlation heatmaps
- `logs/analysis.log` - Execution log

## Interpreting Results

### SMD (Standardized Mean Difference)
- **< 0.1**: Negligible difference - distributions are aligned
- **0.1-0.3**: Small effect - mild shift, usually acceptable
- **0.3-0.5**: Medium effect - moderate shift, may need adjustment
- **≥ 0.5**: Large effect - significant shift, requires correction

### PSI (Population Stability Index)
- **< 0.1**: No significant population shift
- **0.1-0.2**: Moderate shift
- **≥ 0.2**: Significant shift

### KS Statistic
- **p-value < 0.05**: Distributions are statistically different
- Combined with SMD to assess practical significance

## Next Steps

Based on the analysis results:

1. **Identify problematic features** (large shift category)
2. **Investigate causes**:
   - Unit conversion issues?
   - Different patient populations?
   - Measurement protocols?
   - Data quality?

3. **Apply corrections**:
   - Additional normalization
   - Feature transformation
   - Domain adaptation techniques
   - Stratification

4. **Re-run analysis** to verify corrections worked

## Directory Structure

```
analysis/
├── README.md (this file)
├── plots/
│   ├── per_feature/       # Distribution plots for 40 features
│   ├── temporal/          # Temporal analysis plots
│   └── correlation/       # Correlation heatmaps
├── results/
│   ├── per_feature_summary.csv       # Main results table
│   ├── temporal_summary.csv          # Temporal metrics
│   └── correlation_metrics.csv       # Correlation differences
└── logs/
    └── analysis.log       # Execution logs
```

## Contact

For questions or issues with the analysis, check the log files in `logs/` or refer to the main project documentation.


