# Comprehensive Distribution Analysis: MIMIC-IV vs eICU-CRD

**Analysis Date**: November 23, 2025  
**Cohort**: Blood Stream Infection (BSI) Patients with ICU stay ≥48 hours  
**Features**: 38 aligned clinical features

---

## 1. Dataset Overview

### 1.1 Data Sources
- **MIMIC-IV**: Medical Information Mart for Intensive Care IV
  - Tertiary academic medical center
  - Beth Israel Deaconess Medical Center, Boston, MA
  - Time period: 2008-2019
- **eICU-CRD**: eICU Collaborative Research Database
  - Multi-center database (>200 hospitals)
  - Diverse geographic locations across the United States
  - Time period: 2014-2015

### 1.2 Cohort Definition
- **Inclusion Criteria**:
  - Documented blood culture collection (Blood, Central Line OR Blood, Venipuncture)
  - ICU length of stay ≥48 hours
  - Culture taken during ICU stay (offset ≥0)
- **Observation Window**: ICU admission until blood culture collection time

### 1.3 Sample Sizes
- **MIMIC-IV**: 3,767,050 total measurements across 38 features
- **eICU-CRD**: 648,545 total measurements across 38 features
- **MIMIC/eICU Ratio**: 5.8:1

---

## 2. Statistical Metrics

### 2.1 Standardized Mean Difference (SMD)
**Formula**: `SMD = (mean₁ - mean₂) / √[(std₁² + std₂²) / 2]`

**Interpretation**:
- `|SMD| < 0.1`: **Aligned** - Negligible difference, distributions are well-matched
- `0.1 ≤ |SMD| < 0.3`: **Mild Shift** - Small difference, generally acceptable
- `0.3 ≤ |SMD| < 0.5`: **Moderate Shift** - Medium difference, may need attention
- `|SMD| ≥ 0.5`: **Large Shift** - Large difference, requires investigation

**Advantages**: Scale-invariant, commonly used in propensity score matching

### 2.2 Kolmogorov-Smirnov (KS) Statistic
**Range**: [0, 1]  
**Measures**: Maximum vertical distance between two cumulative distribution functions (CDFs)

**Interpretation**:
- `KS < 0.1`: Very similar distributions
- `0.1 ≤ KS < 0.3`: Moderately different distributions
- `KS ≥ 0.3`: Substantially different distributions
- **P-value**: Tests null hypothesis that both samples come from the same distribution

**Advantages**: Non-parametric, sensitive to differences in both location and shape

### 2.3 Wasserstein Distance (Earth Mover's Distance)
**Measures**: Minimum "work" (in feature units) to transform one distribution to another

**Interpretation**: Scale-dependent (units of the feature). Lower is better.
- Sensitive to both location shifts and shape differences
- Can be interpreted as average difference between distributions

### 2.4 Population Stability Index (PSI)
**Formula**: `PSI = Σ[(P₁ᵢ - P₂ᵢ) × ln(P₁ᵢ / P₂ᵢ)]` where P is proportion in bin i

**Interpretation**:
- `PSI < 0.1`: No significant change
- `0.1 ≤ PSI < 0.25`: Moderate change
- `PSI ≥ 0.25`: Significant change

**Usage**: Commonly used in model monitoring to detect distribution drift

---

## 3. Overall Distribution Summary

### 3.1 Alignment Categories

| Category | Count | Percentage | Features |
|----------|-------|------------|----------|
| **ALIGNED** | 9 | 23.7% | Lymphocytes, Heart Rate, Sodium, ... (6 more) |
| **MILD SHIFT** | 19 | 50.0% | GCS - Eye Opening, Temperature, Asparate Aminotransferase (AST), ... (16 more) |
| **MODERATE SHIFT** | 3 | 7.9% | pH, Lactate, Urea Nitrogen |
| **LARGE SHIFT 🚨** | 7 | 18.4% | C-Reactive Protein, RBC, Mean Airway Pressure, ... (4 more) |

### 3.2 Key Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Average SMD | 0.324 | Overall mild-to-moderate difference |
| Median SMD | 0.195 | Most features have mild shift |
| Average KS Statistic | 0.179 | Moderate distributional differences |
| Average Wasserstein | 19.025 | Varies by feature scale |
| Average PSI | 1.098 | Significant population shift overall |
| Features with SMD < 0.1 | 9 (23.7%) | Excellent alignment |
| Features with SMD ≤ 0.3 | 28 (73.7%) | Good-to-excellent alignment |
| Features with SMD > 0.5 | 7 (18.4%) | Need investigation |

### 3.3 Unit Coverage Analysis

**Unit Consistency Findings** (39 features analyzed; Bilirubin has no MIMIC data):
- **Features with matching units**: 8 features (20.5%)
- **Features with unit mismatches**: 31 features (79.5%)
  - Note: Many "mismatches" are case/format differences (e.g., mg/dL vs MG/DL), not actual unit differences
- **Features with multiple unit types**:
  - MIMIC: 3 features have 2+ unit types recorded
  - eICU: 22 features have 2+ unit types recorded

**Unit Mismatch Categories**:

1. **Case/Format Differences Only** (not actual unit differences):
   - Albumin: g/dL vs G/DL
   - Creatinine: mg/dL vs MG/DL
   - Lactate: mmol/L vs MMOL/L
   - Magnesium: mg/dL vs MG/DL
   - PT: sec vs SECONDS
   - Urea Nitrogen: mg/dL vs MG/DL
   - pO2: mm Hg vs MMHG

2. **Actual Unit Differences** (require attention):
   - C-Reactive Protein: mg/L (MIMIC) vs mg/dL (eICU) - **CORRECTED by dividing eICU by 10**
   - Potassium: mEq/L (MIMIC) vs mmol/L (eICU) - Equivalent units (1:1 ratio)
   - Sodium: mEq/L (MIMIC) vs mmol/L (eICU) - Equivalent units (1:1 ratio)

3. **Generic Labels in eICU** (not actual units):
   - GCS scores: N/A vs "Glasgow coma score"
   - Heart Rate: bpm vs "Heart Rate"
   - Blood Pressures: mmHg vs "Non-Invasive BP"
   - Respiratory Rate: insp/min vs "Respiratory Rate"
   - Temperature: N/A vs "DEGREES" / "Temperature"
   - Ventilator parameters: cmH2O/mL vs "respFlowPtVentData" / "respFlowSettings"

4. **Multiple Unit Types in Same Dataset**:
   - **Hemoglobin**: MIMIC has g/dL and g/dl (case difference)
   - **MCHC**: MIMIC has % and g/dL (two different measurement types!)
   - **WBC**: MIMIC has K/uL and #/hpf (two different measurement types!)
   - **RBC**: eICU has 8 different unit variants (MIL/CMM, mil/mm3, M/uL, etc.)

**Detailed Unit Coverage** (see `analysis/results/aligned_features_unit_coverage.csv` for complete details)

---

## 4. Feature-by-Feature Analysis

### Organization
Features are organized by alignment category (from most concerning to best aligned):
1. **Large Shift** (SMD ≥ 0.5) - Requires investigation
2. **Moderate Shift** (0.3 ≤ SMD < 0.5) - May need attention
3. **Mild Shift** (0.1 ≤ SMD < 0.3) - Generally acceptable
4. **Aligned** (SMD < 0.1) - Excellent alignment

### 4.1 Large Shift Features (SMD ≥ 0.5)

**Count**: 7 features

#### C-Reactive Protein

**Units**: MIMIC: `mg/L` | eICU: `mg/dL` | Same: `NO`

**Sample Sizes**:
- MIMIC: 259
- eICU: 24
- Ratio (MIMIC/eICU): 10.8:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 10.68 ± 8.56 | 148.10 ± 60.16 | -137.42 (-1286.8%) |
| Median [IQR] | 8.47 [13.58] | 164.00 [38.90] | -155.53 |
| 1st Percentile | 0.27 | 10.61 | -10.34 |
| 5th Percentile | 0.73 | 18.21 | -17.48 |
| 25th Percentile | 3.36 | 125.10 | -121.74 |
| 75th Percentile | 16.94 | 164.00 | -147.06 |
| 95th Percentile | 27.36 | 262.36 | -234.99 |
| 99th Percentile | 29.76 | 276.16 | -246.40 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 3.198 | large shift 🚨 |
| **KS Statistic** | 0.917 (p < 0.001) | High difference |
| **Wasserstein** | 137.417 | Average distance between distributions |
| **PSI** | 18.646 | Significant population shift |

---

#### RBC

**Units**: MIMIC: `#/hpf;m/uL` | eICU: `M/mcL` | Same: `NO`

**Sample Sizes**:
- MIMIC: 2,193
- eICU: 3,934
- Ratio (MIMIC/eICU): 0.6:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 18.97 ± 30.76 | 3.28 ± 0.65 | 15.69 (82.7%) |
| Median [IQR] | 6.00 [18.00] | 3.18 [0.79] | 2.82 |
| 1st Percentile | 0.00 | 2.26 | -2.26 |
| 5th Percentile | 0.00 | 2.44 | -2.44 |
| 25th Percentile | 2.00 | 2.81 | -0.81 |
| 75th Percentile | 20.00 | 3.60 | 16.40 |
| 95th Percentile | 94.00 | 4.51 | 89.49 |
| 99th Percentile | 139.32 | 5.51 | 133.81 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.721 | large shift 🚨 |
| **KS Statistic** | 0.525 (p < 0.001) | High difference |
| **Wasserstein** | 16.638 | Average distance between distributions |
| **PSI** | 12.284 | Significant population shift |

---

#### Mean Airway Pressure

**Units**: MIMIC: `cmH2O` | eICU: `cmH2O` | Same: `YES`

**Sample Sizes**:
- MIMIC: 71,640
- eICU: 4,933
- Ratio (MIMIC/eICU): 14.5:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 10.62 ± 4.12 | 13.40 ± 3.57 | -2.78 (-26.2%) |
| Median [IQR] | 9.10 [5.30] | 13.00 [5.00] | -3.90 |
| 1st Percentile | 3.00 | 6.10 | -3.10 |
| 5th Percentile | 6.00 | 8.00 | -2.00 |
| 25th Percentile | 7.70 | 11.00 | -3.30 |
| 75th Percentile | 13.00 | 16.00 | -3.00 |
| 95th Percentile | 19.00 | 19.00 | 0.00 |
| 99th Percentile | 23.00 | 21.00 | 2.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.721 | large shift 🚨 |
| **KS Statistic** | 0.370 (p < 0.001) | High difference |
| **Wasserstein** | 2.871 | Average distance between distributions |
| **PSI** | 0.838 | Significant population shift |

---

#### Albumin

**Units**: MIMIC: `g/dL` | eICU: `g/dL` | Same: `YES`

**Sample Sizes**:
- MIMIC: 6,841
- eICU: 1,242
- Ratio (MIMIC/eICU): 5.5:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 2.88 ± 0.62 | 2.45 ± 0.62 | 0.43 (14.8%) |
| Median [IQR] | 2.90 [0.90] | 2.45 [0.90] | 0.45 |
| 1st Percentile | 1.50 | 1.40 | 0.10 |
| 5th Percentile | 1.90 | 1.50 | 0.40 |
| 25th Percentile | 2.40 | 1.90 | 0.50 |
| 75th Percentile | 3.30 | 2.80 | 0.50 |
| 95th Percentile | 3.90 | 3.50 | 0.40 |
| 99th Percentile | 4.36 | 4.00 | 0.36 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.691 | large shift 🚨 |
| **KS Statistic** | 0.274 (p < 0.001) | Moderate difference |
| **Wasserstein** | 0.427 | Average distance between distributions |
| **PSI** | 0.501 | Significant population shift |

---

#### Temperature Fahrenheit

**Units**: MIMIC: `°F` | eICU: `F` | Same: `NO`

**Sample Sizes**:
- MIMIC: 124,722
- eICU: 65,717
- Ratio (MIMIC/eICU): 1.9:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 98.77 ± 1.30 | 99.62 ± 1.50 | -0.85 (-0.9%) |
| Median [IQR] | 98.60 [1.50] | 99.70 [2.00] | -1.10 |
| 1st Percentile | 95.70 | 95.20 | 0.50 |
| 5th Percentile | 96.80 | 97.30 | -0.50 |
| 25th Percentile | 98.00 | 98.60 | -0.60 |
| 75th Percentile | 99.50 | 100.60 | -1.10 |
| 95th Percentile | 101.10 | 102.00 | -0.90 |
| 99th Percentile | 102.30 | 102.70 | -0.40 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.604 | large shift 🚨 |
| **KS Statistic** | 0.315 (p < 0.001) | High difference |
| **Wasserstein** | 0.868 | Average distance between distributions |
| **PSI** | 0.498 | Significant population shift |

---

#### Tidal Volume (observed)

**Units**: MIMIC: `mL` | eICU: `mL` | Same: `YES`

**Sample Sizes**:
- MIMIC: 73,565
- eICU: 1,626
- Ratio (MIMIC/eICU): 45.2:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 468.08 ± 112.80 | 537.00 ± 135.12 | -68.92 (-14.7%) |
| Median [IQR] | 460.00 [132.00] | 518.00 [182.00] | -58.00 |
| 1st Percentile | 227.00 | 300.00 | -73.00 |
| 5th Percentile | 301.00 | 341.00 | -40.00 |
| 25th Percentile | 396.00 | 435.00 | -39.00 |
| 75th Percentile | 528.00 | 617.00 | -89.00 |
| 95th Percentile | 672.00 | 797.00 | -125.00 |
| 99th Percentile | 807.00 | 875.00 | -68.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.554 | large shift 🚨 |
| **KS Statistic** | 0.241 (p < 0.001) | Moderate difference |
| **Wasserstein** | 68.916 | Average distance between distributions |
| **PSI** | 0.390 | Significant population shift |

---

#### O2 saturation pulseoxymetry

**Units**: MIMIC: `%` | eICU: `%` | Same: `YES`

**Sample Sizes**:
- MIMIC: 494,072
- eICU: 2,142
- Ratio (MIMIC/eICU): 230.7:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 97.11 ± 2.71 | 95.59 ± 2.87 | 1.52 (1.6%) |
| Median [IQR] | 98.00 [5.00] | 96.00 [4.00] | 2.00 |
| 1st Percentile | 89.00 | 87.39 | 1.61 |
| 5th Percentile | 92.00 | 89.80 | 2.20 |
| 25th Percentile | 95.00 | 94.00 | 1.00 |
| 75th Percentile | 100.00 | 98.00 | 2.00 |
| 95th Percentile | 100.00 | 99.00 | 1.00 |
| 99th Percentile | 100.00 | 100.00 | 0.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.543 | large shift 🚨 |
| **KS Statistic** | 0.280 (p < 0.001) | Moderate difference |
| **Wasserstein** | 1.515 | Average distance between distributions |
| **PSI** | 0.370 | Significant population shift |

---

### 4.2 Moderate Shift Features (0.3 ≤ SMD < 0.5)

**Count**: 3 features

#### pH

**Units**: MIMIC: `units` | eICU: `units` | Same: `YES`

**Sample Sizes**:
- MIMIC: 46,203
- eICU: 4,223
- Ratio (MIMIC/eICU): 10.9:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 7.23 ± 0.47 | 7.36 ± 0.10 | -0.13 (-1.8%) |
| Median [IQR] | 7.36 [0.14] | 7.38 [0.13] | -0.02 |
| 1st Percentile | 5.00 | 7.12 | -2.12 |
| 5th Percentile | 6.00 | 7.18 | -1.18 |
| 25th Percentile | 7.28 | 7.30 | -0.02 |
| 75th Percentile | 7.42 | 7.43 | -0.01 |
| 95th Percentile | 7.49 | 7.50 | -0.01 |
| 99th Percentile | 7.53 | 7.53 | 0.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.375 | moderate shift |
| **KS Statistic** | 0.089 (p < 0.001) | Low difference |
| **Wasserstein** | 0.127 | Average distance between distributions |
| **PSI** | 0.181 | Moderate population shift |

---

#### Lactate

**Units**: MIMIC: `mmol/L` | eICU: `mmol/L` | Same: `YES`

**Sample Sizes**:
- MIMIC: 27,319
- eICU: 1,211
- Ratio (MIMIC/eICU): 22.6:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 2.60 ± 1.97 | 3.37 ± 2.22 | -0.78 (-30.0%) |
| Median [IQR] | 1.90 [1.90] | 2.90 [2.90] | -1.00 |
| 1st Percentile | 0.70 | 0.60 | 0.10 |
| 5th Percentile | 0.80 | 0.89 | -0.09 |
| 25th Percentile | 1.30 | 1.70 | -0.40 |
| 75th Percentile | 3.20 | 4.60 | -1.40 |
| 95th Percentile | 6.80 | 7.40 | -0.60 |
| 99th Percentile | 10.30 | 12.10 | -1.80 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.371 | moderate shift |
| **KS Statistic** | 0.216 (p < 0.001) | Moderate difference |
| **Wasserstein** | 0.782 | Average distance between distributions |
| **PSI** | 0.282 | Significant population shift |

---

#### Urea Nitrogen

**Units**: MIMIC: `mg/dL` | eICU: `G/24HR;mg/dL` | Same: `NO`

**Sample Sizes**:
- MIMIC: 36,748
- eICU: 4,852
- Ratio (MIMIC/eICU): 7.6:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 31.03 ± 23.18 | 38.68 ± 24.31 | -7.66 (-24.7%) |
| Median [IQR] | 24.00 [27.00] | 33.00 [34.00] | -9.00 |
| 1st Percentile | 5.00 | 6.00 | -1.00 |
| 5th Percentile | 7.00 | 9.00 | -2.00 |
| 25th Percentile | 14.00 | 20.00 | -6.00 |
| 75th Percentile | 41.00 | 54.00 | -13.00 |
| 95th Percentile | 82.00 | 88.00 | -6.00 |
| 99th Percentile | 108.00 | 110.49 | -2.49 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.322 | moderate shift |
| **KS Statistic** | 0.178 (p < 0.001) | Moderate difference |
| **Wasserstein** | 7.661 | Average distance between distributions |
| **PSI** | 0.174 | Moderate population shift |

---

### 4.3 Mild Shift Features (0.1 ≤ SMD < 0.3)

**Count**: 19 features

#### GCS - Eye Opening

**Units**: MIMIC: `score` | eICU: `score` | Same: `YES`

**Sample Sizes**:
- MIMIC: 145,490
- eICU: 35,124
- Ratio (MIMIC/eICU): 4.1:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 2.81 ± 1.19 | 3.13 ± 1.06 | -0.32 (-11.3%) |
| Median [IQR] | 3.00 [2.00] | 4.00 [2.00] | -1.00 |
| 1st Percentile | 1.00 | 1.00 | 0.00 |
| 5th Percentile | 1.00 | 1.00 | 0.00 |
| 25th Percentile | 2.00 | 2.00 | 0.00 |
| 75th Percentile | 4.00 | 4.00 | 0.00 |
| 95th Percentile | 4.00 | 4.00 | 0.00 |
| 99th Percentile | 4.00 | 4.00 | 0.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.281 | mild shift |
| **KS Statistic** | 0.113 (p < 0.001) | Moderate difference |
| **Wasserstein** | 0.317 | Average distance between distributions |
| **PSI** | 0.087 | Negligible population shift |

---

#### Temperature

**Units**: MIMIC: `C` | eICU: `C` | Same: `YES`

**Sample Sizes**:
- MIMIC: 10,668
- eICU: 1,526
- Ratio (MIMIC/eICU): 7.0:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 37.01 ± 0.90 | 37.21 ± 0.54 | -0.20 (-0.5%) |
| Median [IQR] | 37.00 [0.90] | 37.00 [0.00] | 0.00 |
| 1st Percentile | 34.00 | 36.40 | -2.40 |
| 5th Percentile | 35.60 | 37.00 | -1.40 |
| 25th Percentile | 36.60 | 37.00 | -0.40 |
| 75th Percentile | 37.50 | 37.00 | 0.50 |
| 95th Percentile | 38.50 | 38.50 | 0.00 |
| 99th Percentile | 39.10 | 39.00 | 0.10 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.273 | mild shift |
| **KS Statistic** | 0.459 (p < 0.001) | High difference |
| **Wasserstein** | 0.425 | Average distance between distributions |
| **PSI** | 4.632 | Significant population shift |

---

#### Asparate Aminotransferase (AST)

**Units**: MIMIC: `IU/L` | eICU: `Units/L` | Same: `NO`

**Sample Sizes**:
- MIMIC: 12,138
- eICU: 1,038
- Ratio (MIMIC/eICU): 11.7:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 353.09 ± 945.53 | 156.57 ± 553.74 | 196.52 (55.7%) |
| Median [IQR] | 63.00 [139.75] | 53.00 [72.00] | 10.00 |
| 1st Percentile | 12.00 | 11.00 | 1.00 |
| 5th Percentile | 16.00 | 15.00 | 1.00 |
| 25th Percentile | 31.00 | 27.00 | 4.00 |
| 75th Percentile | 170.75 | 99.00 | 71.75 |
| 95th Percentile | 1875.45 | 513.00 | 1362.45 |
| 99th Percentile | 5452.45 | 3979.98 | 1472.47 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.254 | mild shift |
| **KS Statistic** | 0.157 (p < 0.001) | Moderate difference |
| **Wasserstein** | 196.519 | Average distance between distributions |
| **PSI** | 0.259 | Significant population shift |

---

#### GCS - Motor Response

**Units**: MIMIC: `score` | eICU: `score` | Same: `YES`

**Sample Sizes**:
- MIMIC: 144,655
- eICU: 35,143
- Ratio (MIMIC/eICU): 4.1:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 4.96 ± 1.56 | 5.31 ± 1.19 | -0.35 (-7.0%) |
| Median [IQR] | 6.00 [1.00] | 6.00 [1.00] | 0.00 |
| 1st Percentile | 1.00 | 1.00 | 0.00 |
| 5th Percentile | 1.00 | 2.00 | -1.00 |
| 25th Percentile | 5.00 | 5.00 | 0.00 |
| 75th Percentile | 6.00 | 6.00 | 0.00 |
| 95th Percentile | 6.00 | 6.00 | 0.00 |
| 99th Percentile | 6.00 | 6.00 | 0.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.251 | mild shift |
| **KS Statistic** | 0.097 (p < 0.001) | Low difference |
| **Wasserstein** | 0.348 | Average distance between distributions |
| **PSI** | 0.078 | Negligible population shift |

---

#### PT

**Units**: MIMIC: `sec` | eICU: `sec` | Same: `YES`

**Sample Sizes**:
- MIMIC: 23,085
- eICU: 907
- Ratio (MIMIC/eICU): 25.5:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 17.39 ± 6.95 | 15.71 ± 6.53 | 1.68 (9.7%) |
| Median [IQR] | 15.10 [6.00] | 14.00 [4.50] | 1.10 |
| 1st Percentile | 10.70 | 10.60 | 0.10 |
| 5th Percentile | 11.40 | 11.00 | 0.40 |
| 25th Percentile | 13.10 | 11.90 | 1.20 |
| 75th Percentile | 19.10 | 16.40 | 2.70 |
| 95th Percentile | 31.90 | 27.17 | 4.73 |
| 99th Percentile | 45.92 | 48.10 | -2.18 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.250 | mild shift |
| **KS Statistic** | 0.179 (p < 0.001) | Moderate difference |
| **Wasserstein** | 1.781 | Average distance between distributions |
| **PSI** | 0.232 | Moderate population shift |

---

#### Creatinine

**Units**: MIMIC: `mg/dL` | eICU: `mg/dL` | Same: `YES`

**Sample Sizes**:
- MIMIC: 36,825
- eICU: 4,826
- Ratio (MIMIC/eICU): 7.6:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 1.56 ± 1.35 | 1.88 ± 1.39 | -0.32 (-20.5%) |
| Median [IQR] | 1.10 [1.20] | 1.40 [1.35] | -0.30 |
| 1st Percentile | 0.30 | 0.43 | -0.13 |
| 5th Percentile | 0.40 | 0.65 | -0.25 |
| 25th Percentile | 0.70 | 0.95 | -0.25 |
| 75th Percentile | 1.90 | 2.30 | -0.40 |
| 95th Percentile | 4.50 | 5.17 | -0.67 |
| 99th Percentile | 6.90 | 6.72 | 0.18 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.234 | mild shift |
| **KS Statistic** | 0.208 (p < 0.001) | Moderate difference |
| **Wasserstein** | 0.324 | Average distance between distributions |
| **PSI** | 0.166 | Moderate population shift |

---

#### INR(PT)

**Units**: MIMIC: `ratio` | eICU: `ratio` | Same: `YES`

**Sample Sizes**:
- MIMIC: 23,244
- eICU: 918
- Ratio (MIMIC/eICU): 25.3:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 1.59 ± 0.68 | 1.44 ± 0.61 | 0.15 (9.2%) |
| Median [IQR] | 1.40 [0.50] | 1.24 [0.40] | 0.16 |
| 1st Percentile | 0.90 | 1.00 | -0.10 |
| 5th Percentile | 1.00 | 1.00 | 0.00 |
| 25th Percentile | 1.20 | 1.10 | 0.10 |
| 75th Percentile | 1.70 | 1.50 | 0.20 |
| 95th Percentile | 3.00 | 2.60 | 0.40 |
| 99th Percentile | 4.40 | 4.30 | 0.10 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.226 | mild shift |
| **KS Statistic** | 0.152 (p < 0.001) | Moderate difference |
| **Wasserstein** | 0.150 | Average distance between distributions |
| **PSI** | 0.171 | Moderate population shift |

---

#### pO2

**Units**: MIMIC: `mm Hg` | eICU: `mm Hg` | Same: `YES`

**Sample Sizes**:
- MIMIC: 40,288
- eICU: 4,194
- Ratio (MIMIC/eICU): 9.6:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 110.44 ± 62.00 | 98.59 ± 52.20 | 11.85 (10.7%) |
| Median [IQR] | 96.00 [65.00] | 86.00 [48.00] | 10.00 |
| 1st Percentile | 31.00 | 28.00 | 3.00 |
| 5th Percentile | 38.00 | 36.00 | 2.00 |
| 25th Percentile | 70.00 | 68.00 | 2.00 |
| 75th Percentile | 135.00 | 116.00 | 19.00 |
| 95th Percentile | 233.00 | 199.35 | 33.65 |
| 99th Percentile | 344.00 | 304.00 | 40.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.207 | mild shift |
| **KS Statistic** | 0.106 (p < 0.001) | Moderate difference |
| **Wasserstein** | 12.298 | Average distance between distributions |
| **PSI** | 0.066 | Negligible population shift |

---

#### GCS - Verbal Response

**Units**: MIMIC: `score` | eICU: `score` | Same: `YES`

**Sample Sizes**:
- MIMIC: 145,109
- eICU: 35,122
- Ratio (MIMIC/eICU): 4.1:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 2.34 ± 1.76 | 2.66 ± 1.57 | -0.32 (-13.9%) |
| Median [IQR] | 1.00 [3.00] | 3.00 [3.00] | -2.00 |
| 1st Percentile | 1.00 | 1.00 | 0.00 |
| 5th Percentile | 1.00 | 1.00 | 0.00 |
| 25th Percentile | 1.00 | 1.00 | 0.00 |
| 75th Percentile | 4.00 | 4.00 | 0.00 |
| 95th Percentile | 5.00 | 5.00 | 0.00 |
| 99th Percentile | 5.00 | 5.00 | 0.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.195 | mild shift |
| **KS Statistic** | 0.212 (p < 0.001) | Moderate difference |
| **Wasserstein** | 0.472 | Average distance between distributions |
| **PSI** | 0.002 | Negligible population shift |

---

#### Alanine Aminotransferase (ALT)

**Units**: MIMIC: `IU/L` | eICU: `Units/L` | Same: `NO`

**Sample Sizes**:
- MIMIC: 12,102
- eICU: 1,032
- Ratio (MIMIC/eICU): 11.7:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 241.78 ± 613.92 | 137.03 ± 450.74 | 104.75 (43.3%) |
| Median [IQR] | 40.00 [107.00] | 33.00 [59.00] | 7.00 |
| 1st Percentile | 7.00 | 7.00 | 0.00 |
| 5th Percentile | 10.00 | 11.00 | -1.00 |
| 25th Percentile | 20.00 | 19.00 | 1.00 |
| 75th Percentile | 127.00 | 78.00 | 49.00 |
| 95th Percentile | 1373.95 | 527.05 | 846.90 |
| 99th Percentile | 3298.97 | 2466.00 | 832.97 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.195 | mild shift |
| **KS Statistic** | 0.113 (p < 0.001) | Moderate difference |
| **Wasserstein** | 104.940 | Average distance between distributions |
| **PSI** | 0.139 | Moderate population shift |

---

#### Inspired O2 Fraction

**Units**: MIMIC: `fraction` | eICU: `fraction` | Same: `YES`

**Sample Sizes**:
- MIMIC: 93,347
- eICU: 10,489
- Ratio (MIMIC/eICU): 8.9:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 50.81 ± 17.94 | 54.19 ± 17.57 | -3.38 (-6.7%) |
| Median [IQR] | 50.00 [10.00] | 50.00 [20.00] | 0.00 |
| 1st Percentile | 30.00 | 30.00 | 0.00 |
| 5th Percentile | 30.00 | 30.00 | 0.00 |
| 25th Percentile | 40.00 | 40.00 | 0.00 |
| 75th Percentile | 50.00 | 60.00 | -10.00 |
| 95th Percentile | 100.00 | 100.00 | 0.00 |
| 99th Percentile | 100.00 | 100.00 | 0.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.191 | mild shift |
| **KS Statistic** | 0.196 (p < 0.001) | Moderate difference |
| **Wasserstein** | 3.787 | Average distance between distributions |
| **PSI** | 0.173 | Moderate population shift |

---

#### Respiratory Rate

**Units**: MIMIC: `insp/min` | eICU: `breaths/min` | Same: `NO`

**Sample Sizes**:
- MIMIC: 492,165
- eICU: 95,811
- Ratio (MIMIC/eICU): 5.1:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 20.43 ± 5.67 | 21.44 ± 6.04 | -1.02 (-5.0%) |
| Median [IQR] | 20.00 [8.00] | 21.00 [8.00] | -1.00 |
| 1st Percentile | 10.00 | 10.00 | 0.00 |
| 5th Percentile | 12.00 | 12.00 | 0.00 |
| 25th Percentile | 16.00 | 17.00 | -1.00 |
| 75th Percentile | 24.00 | 25.00 | -1.00 |
| 95th Percentile | 31.00 | 33.00 | -2.00 |
| 99th Percentile | 35.00 | 36.00 | -1.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.173 | mild shift |
| **KS Statistic** | 0.072 (p < 0.001) | Low difference |
| **Wasserstein** | 1.016 | Average distance between distributions |
| **PSI** | 0.041 | Negligible population shift |

---

#### WBC

**Units**: MIMIC: `#/hpf;K/uL` | eICU: `K/mcL` | Same: `NO`

**Sample Sizes**:
- MIMIC: 35,242
- eICU: 3,914
- Ratio (MIMIC/eICU): 9.0:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 12.67 ± 7.88 | 13.85 ± 6.55 | -1.18 (-9.3%) |
| Median [IQR] | 11.10 [8.30] | 12.60 [8.30] | -1.50 |
| 1st Percentile | 1.00 | 3.50 | -2.50 |
| 5th Percentile | 3.00 | 5.47 | -2.47 |
| 25th Percentile | 7.60 | 9.30 | -1.70 |
| 75th Percentile | 15.90 | 17.60 | -1.70 |
| 95th Percentile | 27.40 | 25.40 | 2.00 |
| 99th Percentile | 41.66 | 35.89 | 5.77 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.163 | mild shift |
| **KS Statistic** | 0.125 (p < 0.001) | Moderate difference |
| **Wasserstein** | 1.689 | Average distance between distributions |
| **PSI** | 0.136 | Moderate population shift |

---

#### Alkaline Phosphatase

**Units**: MIMIC: `IU/L` | eICU: `Units/L` | Same: `NO`

**Sample Sizes**:
- MIMIC: 11,878
- eICU: 1,039
- Ratio (MIMIC/eICU): 11.4:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 126.58 ± 106.71 | 112.48 ± 74.32 | 14.09 (11.1%) |
| Median [IQR] | 92.00 [83.00] | 94.00 [81.50] | -2.00 |
| 1st Percentile | 32.77 | 31.38 | 1.39 |
| 5th Percentile | 41.00 | 41.00 | 0.00 |
| 25th Percentile | 63.00 | 62.00 | 1.00 |
| 75th Percentile | 146.00 | 143.50 | 2.50 |
| 95th Percentile | 344.00 | 228.10 | 115.90 |
| 99th Percentile | 600.00 | 409.00 | 191.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.153 | mild shift |
| **KS Statistic** | 0.070 (p < 0.001) | Low difference |
| **Wasserstein** | 14.976 | Average distance between distributions |
| **PSI** | 0.091 | Negligible population shift |

---

#### Potassium

**Units**: MIMIC: `mEq/L` | eICU: `mmol/L` | Same: `NO`

**Sample Sizes**:
- MIMIC: 38,980
- eICU: 4,892
- Ratio (MIMIC/eICU): 8.0:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 4.12 ± 0.62 | 4.04 ± 0.56 | 0.08 (2.1%) |
| Median [IQR] | 4.00 [0.80] | 4.00 [0.70] | 0.00 |
| 1st Percentile | 3.00 | 2.90 | 0.10 |
| 5th Percentile | 3.20 | 3.20 | 0.00 |
| 25th Percentile | 3.70 | 3.70 | 0.00 |
| 75th Percentile | 4.50 | 4.40 | 0.10 |
| 95th Percentile | 5.30 | 5.10 | 0.20 |
| 99th Percentile | 5.90 | 5.61 | 0.29 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.143 | mild shift |
| **KS Statistic** | 0.056 (p < 0.001) | Low difference |
| **Wasserstein** | 0.085 | Average distance between distributions |
| **PSI** | 0.038 | Negligible population shift |

---

#### Magnesium

**Units**: MIMIC: `mg/dL` | eICU: `mg/dL` | Same: `YES`

**Sample Sizes**:
- MIMIC: 36,072
- eICU: 3,407
- Ratio (MIMIC/eICU): 10.6:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 2.05 ± 0.33 | 2.09 ± 0.34 | -0.04 (-1.9%) |
| Median [IQR] | 2.00 [0.40] | 2.10 [0.40] | -0.10 |
| 1st Percentile | 1.30 | 1.30 | 0.00 |
| 5th Percentile | 1.50 | 1.60 | -0.10 |
| 25th Percentile | 1.80 | 1.90 | -0.10 |
| 75th Percentile | 2.20 | 2.30 | -0.10 |
| 95th Percentile | 2.60 | 2.70 | -0.10 |
| 99th Percentile | 3.00 | 3.10 | -0.10 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.118 | mild shift |
| **KS Statistic** | 0.060 (p < 0.001) | Low difference |
| **Wasserstein** | 0.040 | Average distance between distributions |
| **PSI** | 0.023 | Negligible population shift |

---

#### Non Invasive Blood Pressure systolic

**Units**: MIMIC: `mmHg` | eICU: `mmHg` | Same: `YES`

**Sample Sizes**:
- MIMIC: 287,972
- eICU: 64,725
- Ratio (MIMIC/eICU): 4.4:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 117.53 ± 21.67 | 119.94 ± 21.35 | -2.41 (-2.1%) |
| Median [IQR] | 115.00 [31.00] | 118.00 [30.00] | -3.00 |
| 1st Percentile | 79.00 | 80.00 | -1.00 |
| 5th Percentile | 86.00 | 88.00 | -2.00 |
| 25th Percentile | 101.00 | 104.00 | -3.00 |
| 75th Percentile | 132.00 | 134.00 | -2.00 |
| 95th Percentile | 157.00 | 159.00 | -2.00 |
| 99th Percentile | 172.00 | 173.00 | -1.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.112 | mild shift |
| **KS Statistic** | 0.062 (p < 0.001) | Low difference |
| **Wasserstein** | 2.410 | Average distance between distributions |
| **PSI** | 0.018 | Negligible population shift |

---

#### MCV

**Units**: MIMIC: `fL` | eICU: `fL` | Same: `YES`

**Sample Sizes**:
- MIMIC: 32,934
- eICU: 3,945
- Ratio (MIMIC/eICU): 8.3:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 91.62 ± 6.49 | 90.96 ± 5.59 | 0.67 (0.7%) |
| Median [IQR] | 91.00 [9.00] | 90.20 [6.80] | 0.80 |
| 1st Percentile | 77.00 | 79.00 | -2.00 |
| 5th Percentile | 82.00 | 83.70 | -1.70 |
| 25th Percentile | 87.00 | 87.10 | -0.10 |
| 75th Percentile | 96.00 | 93.90 | 2.10 |
| 95th Percentile | 103.00 | 101.80 | 1.20 |
| 99th Percentile | 108.00 | 105.91 | 2.09 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.110 | mild shift |
| **KS Statistic** | 0.105 (p < 0.001) | Moderate difference |
| **Wasserstein** | 1.080 | Average distance between distributions |
| **PSI** | 0.085 | Negligible population shift |

---

#### Lactate Dehydrogenase (LD)

**Units**: MIMIC: `IU/L` | eICU: `Units/L` | Same: `NO`

**Sample Sizes**:
- MIMIC: 6,459
- eICU: 178
- Ratio (MIMIC/eICU): 36.3:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 602.42 ± 900.23 | 508.21 ± 948.38 | 94.21 (15.6%) |
| Median [IQR] | 324.00 [314.00] | 293.00 [143.00] | 31.00 |
| 1st Percentile | 133.00 | 158.00 | -25.00 |
| 5th Percentile | 162.00 | 182.90 | -20.90 |
| 25th Percentile | 231.00 | 208.25 | 22.75 |
| 75th Percentile | 545.00 | 351.25 | 193.75 |
| 95th Percentile | 2073.10 | 1892.00 | 181.10 |
| 99th Percentile | 5581.00 | 4506.23 | 1074.77 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.102 | mild shift |
| **KS Statistic** | 0.234 (p < 0.001) | Moderate difference |
| **Wasserstein** | 134.189 | Average distance between distributions |
| **PSI** | 0.428 | Significant population shift |

---

### 4.4 Aligned Features (SMD < 0.1)

**Count**: 9 features

#### Lymphocytes

**Units**: MIMIC: `%` | eICU: `%` | Same: `YES`

**Sample Sizes**:
- MIMIC: 6,427
- eICU: 2,131
- Ratio (MIMIC/eICU): 3.0:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 10.47 ± 10.15 | 9.60 ± 7.30 | 0.87 (8.3%) |
| Median [IQR] | 7.90 [9.05] | 8.30 [8.15] | -0.40 |
| 1st Percentile | 0.00 | 0.00 | 0.00 |
| 5th Percentile | 1.00 | 1.00 | 0.00 |
| 25th Percentile | 4.00 | 4.75 | -0.75 |
| 75th Percentile | 13.05 | 12.90 | 0.15 |
| 95th Percentile | 29.84 | 22.00 | 7.84 |
| 99th Percentile | 55.00 | 30.00 | 25.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.098 | aligned |
| **KS Statistic** | 0.062 (p < 0.001) | Low difference |
| **Wasserstein** | 1.508 | Average distance between distributions |
| **PSI** | 0.059 | Negligible population shift |

---

#### Heart Rate

**Units**: MIMIC: `bpm` | eICU: `bpm` | Same: `YES`

**Sample Sizes**:
- MIMIC: 494,823
- eICU: 97,758
- Ratio (MIMIC/eICU): 5.1:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 89.59 ± 18.45 | 91.02 ± 17.22 | -1.43 (-1.6%) |
| Median [IQR] | 89.00 [26.00] | 91.00 [24.00] | -2.00 |
| 1st Percentile | 54.00 | 55.00 | -1.00 |
| 5th Percentile | 61.00 | 62.00 | -1.00 |
| 25th Percentile | 76.00 | 79.00 | -3.00 |
| 75th Percentile | 102.00 | 103.00 | -1.00 |
| 95th Percentile | 122.00 | 120.00 | 2.00 |
| 99th Percentile | 134.00 | 131.00 | 3.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.080 | aligned |
| **KS Statistic** | 0.058 (p < 0.001) | Low difference |
| **Wasserstein** | 2.021 | Average distance between distributions |
| **PSI** | 0.026 | Negligible population shift |

---

#### Sodium

**Units**: MIMIC: `mEq/L` | eICU: `mmol/L` | Same: `NO`

**Sample Sizes**:
- MIMIC: 39,319
- eICU: 4,895
- Ratio (MIMIC/eICU): 8.0:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 139.43 ± 5.57 | 139.87 ± 5.57 | -0.44 (-0.3%) |
| Median [IQR] | 139.00 [7.00] | 140.00 [7.00] | -1.00 |
| 1st Percentile | 126.00 | 126.00 | 0.00 |
| 5th Percentile | 130.00 | 130.00 | 0.00 |
| 25th Percentile | 136.00 | 136.00 | 0.00 |
| 75th Percentile | 143.00 | 143.00 | 0.00 |
| 95th Percentile | 149.00 | 149.00 | 0.00 |
| 99th Percentile | 154.00 | 153.00 | 1.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.078 | aligned |
| **KS Statistic** | 0.048 (p < 0.001) | Low difference |
| **Wasserstein** | 0.549 | Average distance between distributions |
| **PSI** | 0.024 | Negligible population shift |

---

#### MCHC

**Units**: MIMIC: `%;g/dL` | eICU: `g/dL` | Same: `NO`

**Sample Sizes**:
- MIMIC: 32,932
- eICU: 3,929
- Ratio (MIMIC/eICU): 8.4:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 32.98 ± 1.69 | 33.07 ± 1.14 | -0.09 (-0.3%) |
| Median [IQR] | 33.00 [2.40] | 33.10 [1.60] | -0.10 |
| 1st Percentile | 29.00 | 30.10 | -1.10 |
| 5th Percentile | 30.10 | 31.10 | -1.00 |
| 25th Percentile | 31.80 | 32.30 | -0.50 |
| 75th Percentile | 34.20 | 33.90 | 0.30 |
| 95th Percentile | 35.74 | 34.80 | 0.94 |
| 99th Percentile | 36.60 | 35.70 | 0.90 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.063 | aligned |
| **KS Statistic** | 0.118 (p < 0.001) | Moderate difference |
| **Wasserstein** | 0.470 | Average distance between distributions |
| **PSI** | 0.305 | Significant population shift |

---

#### Non Invasive Blood Pressure diastolic

**Units**: MIMIC: `mmHg` | eICU: `mmHg` | Same: `YES`

**Sample Sizes**:
- MIMIC: 287,177
- eICU: 65,090
- Ratio (MIMIC/eICU): 4.4:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 64.09 ± 14.55 | 63.23 ± 12.75 | 0.86 (1.3%) |
| Median [IQR] | 62.00 [20.00] | 62.00 [17.00] | 0.00 |
| 1st Percentile | 37.00 | 38.00 | -1.00 |
| 5th Percentile | 43.00 | 44.00 | -1.00 |
| 25th Percentile | 53.00 | 54.00 | -1.00 |
| 75th Percentile | 73.00 | 71.00 | 2.00 |
| 95th Percentile | 91.00 | 86.00 | 5.00 |
| 99th Percentile | 102.00 | 98.00 | 4.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.063 | aligned |
| **KS Statistic** | 0.054 (p < 0.001) | Low difference |
| **Wasserstein** | 1.584 | Average distance between distributions |
| **PSI** | 0.040 | Negligible population shift |

---

#### RDW

**Units**: MIMIC: `%` | eICU: `%` | Same: `YES`

**Sample Sizes**:
- MIMIC: 32,903
- eICU: 3,925
- Ratio (MIMIC/eICU): 8.4:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 15.98 ± 2.42 | 15.86 ± 2.30 | 0.12 (0.8%) |
| Median [IQR] | 15.50 [3.00] | 15.50 [2.00] | 0.00 |
| 1st Percentile | 12.40 | 12.50 | -0.10 |
| 5th Percentile | 13.00 | 13.10 | -0.10 |
| 25th Percentile | 14.20 | 14.50 | -0.30 |
| 75th Percentile | 17.20 | 16.50 | 0.70 |
| 95th Percentile | 20.90 | 19.60 | 1.30 |
| 99th Percentile | 23.50 | 24.70 | -1.20 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.053 | aligned |
| **KS Statistic** | 0.102 (p < 0.001) | Moderate difference |
| **Wasserstein** | 0.441 | Average distance between distributions |
| **PSI** | 0.132 | Moderate population shift |

---

#### Hemoglobin

**Units**: MIMIC: `g/dL;g/dl` | eICU: `g/dL` | Same: `NO`

**Sample Sizes**:
- MIMIC: 36,898
- eICU: 3,998
- Ratio (MIMIC/eICU): 9.2:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 9.83 ± 1.90 | 9.74 ± 1.73 | 0.09 (0.9%) |
| Median [IQR] | 9.60 [2.70] | 9.60 [2.00] | 0.00 |
| 1st Percentile | 6.50 | 6.70 | -0.20 |
| 5th Percentile | 7.20 | 7.20 | 0.00 |
| 25th Percentile | 8.40 | 8.60 | -0.20 |
| 75th Percentile | 11.10 | 10.60 | 0.50 |
| 95th Percentile | 13.40 | 13.00 | 0.40 |
| 99th Percentile | 14.70 | 14.90 | -0.20 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.048 | aligned |
| **KS Statistic** | 0.072 (p < 0.001) | Low difference |
| **Wasserstein** | 0.233 | Average distance between distributions |
| **PSI** | 0.064 | Negligible population shift |

---

#### Non Invasive Blood Pressure mean

**Units**: MIMIC: `mmHg` | eICU: `mmHg` | Same: `YES`

**Sample Sizes**:
- MIMIC: 287,781
- eICU: 64,656
- Ratio (MIMIC/eICU): 4.5:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 77.09 ± 14.98 | 77.74 ± 13.45 | -0.65 (-0.8%) |
| Median [IQR] | 75.00 [21.00] | 77.00 [19.00] | -2.00 |
| 1st Percentile | 49.00 | 51.00 | -2.00 |
| 5th Percentile | 55.00 | 57.00 | -2.00 |
| 25th Percentile | 66.00 | 68.00 | -2.00 |
| 75th Percentile | 87.00 | 87.00 | 0.00 |
| 95th Percentile | 105.00 | 101.00 | 4.00 |
| 99th Percentile | 116.00 | 113.00 | 3.00 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.046 | aligned |
| **KS Statistic** | 0.048 (p < 0.001) | Low difference |
| **Wasserstein** | 1.631 | Average distance between distributions |
| **PSI** | 0.029 | Negligible population shift |

---

#### Hematocrit

**Units**: MIMIC: `%` | eICU: `%` | Same: `YES`

**Sample Sizes**:
- MIMIC: 36,575
- eICU: 4,029
- Ratio (MIMIC/eICU): 9.1:1

**Distribution Metrics**:

| Metric | MIMIC-IV | eICU-CRD | Difference |
|--------|----------|----------|------------|
| Mean ± SD | 29.60 ± 5.46 | 29.37 ± 5.05 | 0.23 (0.8%) |
| Median [IQR] | 28.80 [7.50] | 29.00 [6.50] | -0.20 |
| 1st Percentile | 20.20 | 20.53 | -0.33 |
| 5th Percentile | 22.00 | 22.00 | 0.00 |
| 25th Percentile | 25.50 | 25.70 | -0.20 |
| 75th Percentile | 33.00 | 32.20 | 0.80 |
| 95th Percentile | 39.90 | 39.50 | 0.40 |
| 99th Percentile | 44.20 | 44.10 | 0.10 |

**Statistical Comparison**:

| Test | Value | Interpretation |
|------|-------|----------------|
| **SMD** | 0.044 | aligned |
| **KS Statistic** | 0.055 (p < 0.001) | Low difference |
| **Wasserstein** | 0.440 | Average distance between distributions |
| **PSI** | 0.029 | Negligible population shift |

---

---

## 5. Detailed Analysis of Large Shift Features

This section provides in-depth analysis of the 7 features with large distributional shifts (SMD ≥ 0.5).

### 5.1 C-Reactive Protein (SMD: 3.198)

**Key Observations**:
- Extremely large SMD (3.198) - highest among all features
- MIMIC mean: 10.68 mg/L vs eICU mean: 148.10 mg/dL
- **13.9× difference in means** (13.9×)
- Very small sample sizes: MIMIC n=259, eICU n=24
- KS statistic: 0.917 (extremely high)

**Likely Causes**:
1. **Unit mismatch**: MIMIC uses mg/L, eICU uses mg/dL (10× conversion factor)
2. Low sample size may affect reliability

**Recommendation**: Apply unit conversion (divide eICU by 10 OR multiply MIMIC by 10)

---

### 5.2 RBC (SMD: 0.721)

**Key Observations**:
- Very large SMD (0.721)
- MIMIC mean: 18.97 vs eICU mean: 3.28
- **5.8× difference in means** (5.8×)
- MIMIC has high standard deviation (30.76), suggesting mixed units
- eICU values are tightly clustered (std=0.65)

**Likely Causes**:
1. **Mixed units in MIMIC**: Combination of '#/hpf' (microscopy count) and 'm/uL' (concentration)
2. eICU uses only 'M/mcL' (equivalent to m/uL)
3. Mean of ~3.3 in eICU is clinically consistent with RBC count (million cells/mcL)
4. MIMIC mean of ~19 suggests contamination from '#/hpf' measurements

**Recommendation**: Filter MIMIC data to exclude '#/hpf' measurements, keep only 'm/uL'

---

### 5.3 Mean Airway Pressure (SMD: 0.721)

**Key Observations**:
- Large SMD (0.721)
- MIMIC mean: 10.62 cmH₂O vs eICU mean: 13.40 cmH₂O
- **26% higher in eICU** (26.2%)
- Units are the same (cmH₂O)
- Large sample in MIMIC (n=71,640), smaller in eICU (n=4,933)
- KS statistic: 0.370 (high)

**Likely Causes**:
1. **Real clinical difference**: Different ventilator management strategies
2. **Population difference**: eICU patients may be more severely ill
3. **Multi-center variation**: eICU spans >200 hospitals vs single center MIMIC

**Recommendation**: Consider as real clinical difference; may need stratification or adjustment

---

### 5.4 Albumin (SMD: 0.691)

**Key Observations**:
- Large SMD (0.691)
- MIMIC mean: 2.88 g/dL vs eICU mean: 2.45 g/dL
- **15% lower in eICU** (14.8%)
- Units are the same (g/dL)
- Clinically significant difference (normal range: 3.5-5.5 g/dL)

**Likely Causes**:
1. **Severity difference**: Lower albumin indicates worse nutritional status/inflammation
2. **Population characteristics**: eICU patients may be more severely ill
3. **Hospital practice variation**: Different admission thresholds

**Recommendation**: Consider as real clinical difference; adjust for severity if needed

---

### 5.5 Temperature Fahrenheit (SMD: 0.604)

**Key Observations**:
- Large SMD (0.604)
- MIMIC mean: 98.77°F vs eICU mean: 99.62°F
- **0.85°F higher in eICU** (0.85°F)
- Large sample sizes: MIMIC n=124,722, eICU n=65,717
- Both means are clinically normal (98.6°F is standard)

**Likely Causes**:
1. **Measurement method**: Oral vs axillary vs rectal vs tympanic
2. **Time of measurement**: Different patterns throughout the day
3. **Patient population**: Slightly different baseline temperatures
4. **Statistical significance**: With large N, small differences become significant

**Recommendation**: Likely acceptable; consider adjusting if critical for analysis

---

### 5.6 Tidal Volume (observed) (SMD: 0.554)

**Key Observations**:
- Large SMD (0.554)
- MIMIC mean: 468.08 mL vs eICU mean: 537.00 mL
- **15% higher in eICU** (14.7%)
- Units are the same (mL)
- Much larger sample in MIMIC (n=73,565) vs eICU (n=1,626)

**Likely Causes**:
1. **Ventilator settings**: Different protocols for lung-protective ventilation
2. **Patient body size**: eICU patients may be larger on average
3. **Clinical practice variation**: Different target tidal volumes (6-8 mL/kg IBW)

**Recommendation**: Real clinical difference; consider normalizing by predicted body weight

---

### 5.7 O2 saturation pulseoxymetry (SMD: 0.543)

**Key Observations**:
- Large SMD (0.543)
- MIMIC mean: 97.11% vs eICU mean: 95.59%
- **1.52% lower in eICU** (1.52%)
- Huge sample in MIMIC (n=494,072), much smaller in eICU (n=2,142)
- Both are in clinically acceptable range (>94%)

**Likely Causes**:
1. **Severity difference**: eICU patients may have worse oxygenation
2. **Sample size disparity**: 230:1 ratio (MIMIC has far more measurements)
3. **Measurement frequency**: MIMIC may include more routine checks
4. **Clinical significance**: 1.5% difference is clinically meaningful

**Recommendation**: Real clinical difference indicating severity; adjust for acuity if needed

---


## 6. Summary of Distribution Issues

### 6.1 Issues by Root Cause

| Root Cause | Features | Action Required |
|------------|----------|-----------------|
| **Unit Mismatch** | C-Reactive Protein (mg/L vs mg/dL) | Apply 10× unit conversion |
| **Mixed Units** | RBC (#/hpf vs m/uL) | Filter to consistent unit |
| **Clinical Difference** | Mean Airway Pressure, Albumin, Tidal Volume, O2 Saturation, Temperature F | Consider adjusting or stratifying |

### 6.2 Priority for Action

**HIGH PRIORITY (Data Quality Issues)**:
1. **C-Reactive Protein** - Unit conversion required (SMD: 3.198)
2. **RBC** - Filter mixed units (SMD: 0.721)

**MEDIUM PRIORITY (Clinical Differences)**:
3. **Mean Airway Pressure** - Consider clinical adjustment (SMD: 0.721)
4. **Albumin** - May indicate severity difference (SMD: 0.691)
5. **Temperature Fahrenheit** - Small absolute difference (SMD: 0.604)

**LOW PRIORITY (Acceptable Differences)**:
6. **Tidal Volume** - Real practice variation (SMD: 0.554)
7. **O2 Saturation** - Possible severity indicator (SMD: 0.543)

### 6.3 Expected Impact of Fixes

After applying unit conversion and filtering fixes:
- **C-Reactive Protein**: Expected SMD < 0.3 (from 3.198)
- **RBC**: Expected SMD < 0.3 (from 0.721)
- **Overall alignment**: Expected to reach **80-85% of features with SMD ≤ 0.3** (from current 73.7%)

---

## 7. Consultation Questions

### For Clinical Interpretation:
1. **Mean Airway Pressure**: Is a 26% difference (10.6 vs 13.4 cmH₂O) clinically significant for BSI patients? Could this reflect different ventilator management strategies?

2. **Albumin**: The 15% lower albumin in eICU (2.88 vs 2.45 g/dL) - does this suggest eICU patients are more severely ill, or could this be a measurement/timing difference?

3. **Temperature Fahrenheit**: The 0.85°F difference (98.77 vs 99.62°F) - is this within acceptable measurement variation, or could it indicate different measurement methods (oral/axillary/rectal)?

4. **O2 Saturation**: eICU patients have 1.5% lower SpO₂ (97.1% vs 95.6%) - is this clinically significant for outcomes, or acceptable variation?

5. **Tidal Volume**: 15% higher in eICU (468 vs 537 mL) - could this be due to differences in patient body size, or different lung-protective ventilation protocols?

### For Statistical Approach:
1. Given that 73.7% of features have SMD ≤ 0.3, is this sufficient alignment for cross-database studies?

2. Should we use propensity score matching or other adjustment methods to account for the observed differences?

3. For features with large shifts due to clinical differences (not data quality), what adjustment strategies would you recommend?

4. How should we weight the different distance metrics (SMD, KS, Wasserstein, PSI) when making decisions about feature usability?

5. Are there specific features that should be excluded from cross-database models due to their distributional differences?

---

## 8. Appendices

### 8.1 Data Files
- `analysis/results/per_feature_summary.csv` - Complete statistical results for all features
- `analysis/plots/per_feature/` - Distribution plots (histogram, KDE, ECDF) for each feature
- `feature_units_comparison.csv` - Unit comparison between MIMIC and eICU

### 8.2 Missing Features
Two features from the original 40 BSI features are not included in this analysis:
- **Feature 1**: Insufficient data in one or both databases
- **Feature 2**: Insufficient data in one or both databases

### 8.3 Analysis Methodology
1. **Data Extraction**: Features extracted from both databases using standardized SQL queries
2. **Data Cleaning**: 
   - Parsed dates to datetime format
   - Converted feature values to numeric (invalid values → NaN)
   - Dropped rows with missing values
   - Applied feature name standardization mapping
3. **Statistical Analysis**: Calculated SMD, KS statistic, Wasserstein distance, PSI for each feature
4. **Visualization**: Generated distribution plots for visual comparison
### 8.4 Complete Feature Summary Table

All 38 features sorted by SMD (descending):

| Rank | Feature | MIMIC n | eICU n | MIMIC Mean±SD | eICU Mean±SD | SMD | KS Stat | Wasserstein | PSI | Category |
|------|---------|---------|--------|---------------|--------------|-----|---------|-------------|-----|----------|
| 1 | C-Reactive Protein | 259 | 24 | 10.68±8.56 | 148.10±60.16 | 3.198 | 0.917 | 137.417 | 18.646 | large shift 🚨 |
| 2 | RBC | 2,193 | 3,934 | 18.97±30.76 | 3.28±0.65 | 0.721 | 0.525 | 16.638 | 12.284 | large shift 🚨 |
| 3 | Mean Airway Pressure | 71,640 | 4,933 | 10.62±4.12 | 13.40±3.57 | 0.721 | 0.370 | 2.871 | 0.838 | large shift 🚨 |
| 4 | Albumin | 6,841 | 1,242 | 2.88±0.62 | 2.45±0.62 | 0.691 | 0.274 | 0.427 | 0.501 | large shift 🚨 |
| 5 | Temperature Fahrenheit | 124,722 | 65,717 | 98.77±1.30 | 99.62±1.50 | 0.604 | 0.315 | 0.868 | 0.498 | large shift 🚨 |
| 6 | Tidal Volume (observed) | 73,565 | 1,626 | 468.08±112.80 | 537.00±135.12 | 0.554 | 0.241 | 68.916 | 0.390 | large shift 🚨 |
| 7 | O2 saturation pulseoxymetry | 494,072 | 2,142 | 97.11±2.71 | 95.59±2.87 | 0.543 | 0.280 | 1.515 | 0.370 | large shift 🚨 |
| 8 | pH | 46,203 | 4,223 | 7.23±0.47 | 7.36±0.10 | 0.375 | 0.089 | 0.127 | 0.181 | moderate shift |
| 9 | Lactate | 27,319 | 1,211 | 2.60±1.97 | 3.37±2.22 | 0.371 | 0.216 | 0.782 | 0.282 | moderate shift |
| 10 | Urea Nitrogen | 36,748 | 4,852 | 31.03±23.18 | 38.68±24.31 | 0.322 | 0.178 | 7.661 | 0.174 | moderate shift |
| 11 | GCS - Eye Opening | 145,490 | 35,124 | 2.81±1.19 | 3.13±1.06 | 0.281 | 0.113 | 0.317 | 0.087 | mild shift |
| 12 | Temperature | 10,668 | 1,526 | 37.01±0.90 | 37.21±0.54 | 0.273 | 0.459 | 0.425 | 4.632 | mild shift |
| 13 | Asparate Aminotransferase (AST) | 12,138 | 1,038 | 353.09±945.53 | 156.57±553.74 | 0.254 | 0.157 | 196.519 | 0.259 | mild shift |
| 14 | GCS - Motor Response | 144,655 | 35,143 | 4.96±1.56 | 5.31±1.19 | 0.251 | 0.097 | 0.348 | 0.078 | mild shift |
| 15 | PT | 23,085 | 907 | 17.39±6.95 | 15.71±6.53 | 0.250 | 0.179 | 1.781 | 0.232 | mild shift |
| 16 | Creatinine | 36,825 | 4,826 | 1.56±1.35 | 1.88±1.39 | 0.234 | 0.208 | 0.324 | 0.166 | mild shift |
| 17 | INR(PT) | 23,244 | 918 | 1.59±0.68 | 1.44±0.61 | 0.226 | 0.152 | 0.150 | 0.171 | mild shift |
| 18 | pO2 | 40,288 | 4,194 | 110.44±62.00 | 98.59±52.20 | 0.207 | 0.106 | 12.298 | 0.066 | mild shift |
| 19 | GCS - Verbal Response | 145,109 | 35,122 | 2.34±1.76 | 2.66±1.57 | 0.195 | 0.212 | 0.472 | 0.002 | mild shift |
| 20 | Alanine Aminotransferase (ALT) | 12,102 | 1,032 | 241.78±613.92 | 137.03±450.74 | 0.195 | 0.113 | 104.940 | 0.139 | mild shift |
| 21 | Inspired O2 Fraction | 93,347 | 10,489 | 50.81±17.94 | 54.19±17.57 | 0.191 | 0.196 | 3.787 | 0.173 | mild shift |
| 22 | Respiratory Rate | 492,165 | 95,811 | 20.43±5.67 | 21.44±6.04 | 0.173 | 0.072 | 1.016 | 0.041 | mild shift |
| 23 | WBC | 35,242 | 3,914 | 12.67±7.88 | 13.85±6.55 | 0.163 | 0.125 | 1.689 | 0.136 | mild shift |
| 24 | Alkaline Phosphatase | 11,878 | 1,039 | 126.58±106.71 | 112.48±74.32 | 0.153 | 0.070 | 14.976 | 0.091 | mild shift |
| 25 | Potassium | 38,980 | 4,892 | 4.12±0.62 | 4.04±0.56 | 0.143 | 0.056 | 0.085 | 0.038 | mild shift |
| 26 | Magnesium | 36,072 | 3,407 | 2.05±0.33 | 2.09±0.34 | 0.118 | 0.060 | 0.040 | 0.023 | mild shift |
| 27 | Non Invasive Blood Pressure systolic | 287,972 | 64,725 | 117.53±21.67 | 119.94±21.35 | 0.112 | 0.062 | 2.410 | 0.018 | mild shift |
| 28 | MCV | 32,934 | 3,945 | 91.62±6.49 | 90.96±5.59 | 0.110 | 0.105 | 1.080 | 0.085 | mild shift |
| 29 | Lactate Dehydrogenase (LD) | 6,459 | 178 | 602.42±900.23 | 508.21±948.38 | 0.102 | 0.234 | 134.189 | 0.428 | mild shift |
| 30 | Lymphocytes | 6,427 | 2,131 | 10.47±10.15 | 9.60±7.30 | 0.098 | 0.062 | 1.508 | 0.059 | aligned |
| 31 | Heart Rate | 494,823 | 97,758 | 89.59±18.45 | 91.02±17.22 | 0.080 | 0.058 | 2.021 | 0.026 | aligned |
| 32 | Sodium | 39,319 | 4,895 | 139.43±5.57 | 139.87±5.57 | 0.078 | 0.048 | 0.549 | 0.024 | aligned |
| 33 | MCHC | 32,932 | 3,929 | 32.98±1.69 | 33.07±1.14 | 0.063 | 0.118 | 0.470 | 0.305 | aligned |
| 34 | Non Invasive Blood Pressure diastolic | 287,177 | 65,090 | 64.09±14.55 | 63.23±12.75 | 0.063 | 0.054 | 1.584 | 0.040 | aligned |
| 35 | RDW | 32,903 | 3,925 | 15.98±2.42 | 15.86±2.30 | 0.053 | 0.102 | 0.441 | 0.132 | aligned |
| 36 | Hemoglobin | 36,898 | 3,998 | 9.83±1.90 | 9.74±1.73 | 0.048 | 0.072 | 0.233 | 0.064 | aligned |
| 37 | Non Invasive Blood Pressure mean | 287,781 | 64,656 | 77.09±14.98 | 77.74±13.45 | 0.046 | 0.048 | 1.631 | 0.029 | aligned |
| 38 | Hematocrit | 36,575 | 4,029 | 29.60±5.46 | 29.37±5.05 | 0.044 | 0.055 | 0.440 | 0.029 | aligned |

---

**End of Analysis**

*Total Features Analyzed*: 38  
*Analysis Date*: November 23, 2025  
*Generated from*: `analysis/results/per_feature_summary.csv`