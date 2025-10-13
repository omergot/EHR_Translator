# Comprehensive Evaluation Report (Simplified Model)

*Generated for simplified CycleVAE with 3 losses: reconstruction, cycle, conditional Wasserstein*

*Note: Missing flags, Age, and Gender are excluded from evaluation (input-only)*

## Executive Summary

### 📊 Reconstruction Quality (A→A')

*Note: Data is normalized - MAE in standard deviation units, use IQR metrics*

**eICU Reconstruction:**
- MAE: 0.0195 (std dev units)
- % within 0.5 IQR: 100.0%
- % within 1.0 IQR: 100.0%

**MIMIC Reconstruction:**
- MAE: 0.0248 (std dev units)
- % within 0.5 IQR: 100.0%
- % within 1.0 IQR: 100.0%

### 🔄 Cycle Consistency (A→B'→A')

*Note: Data is normalized - use IQR metrics for meaningful percentages*

**eICU Cycle:**
- MAE: 0.0557 (std dev units)
- % within 0.5 IQR: 99.9%
- % within 1.0 IQR: 100.0%

**MIMIC Cycle:**
- MAE: 0.0554 (std dev units)
- % within 0.5 IQR: 100.0%
- % within 1.0 IQR: 100.0%

### 🧠 Latent Space Analysis

**Original Domains (eICU vs MIMIC):**
- Euclidean Distance: 0.1733
- Cosine Similarity: 0.9973
- KL Divergence: 0.0802

**After Translation (eICU→MIMIC vs real MIMIC):**
- Euclidean Distance: 0.4967
- Cosine Similarity: 0.9775
- KL Divergence: 0.6196

### 📈 Distribution Matching

- Mean Wasserstein Distance: 0.1475
- Mean KS Statistic: 0.6884

---

### Legacy Metrics (Clinical Features Only)

*Note: R² and Correlation below are computed on **roundtrip/cycle** data (A→B'→A'), not on direct reconstruction*

**Translation Quality Metrics:**
- **Clinical Features Evaluated**: 24

**R² (Variance Explained) - Target: > 0.5:**
- **eICU**: 24/24 features (100.0%), Mean R²: 0.991
- **MIMIC**: 24/24 features (100.0%), Mean R²: 0.989

**Correlation (Linear Relationship) - Target: > 0.7:**
- **eICU**: 24/24 features (100.0%), Mean corr: 0.997
- **MIMIC**: 24/24 features (100.0%), Mean corr: 0.997

**Combined (R² > 0.5 AND correlation > 0.7):**
- **eICU**: 24/24 features (100.0%)
- **MIMIC**: 24/24 features (100.0%)

**Distribution Matching (KS statistic - effect size):**
*Note: p-values not used (uninformative with large N=24)*

**eICU→MIMIC Translation:**
- Excellent (KS<0.1): 16/24 (66.7%)
- Good (KS<0.2): 19/24 (79.2%)
- Acceptable (KS<0.3): 20/24 (83.3%)
- Mean KS: 0.163

**MIMIC→eICU Translation:**
- Excellent (KS<0.1): 13/24 (54.2%)
- Good (KS<0.2): 20/24 (83.3%)
- Acceptable (KS<0.3): 20/24 (83.3%)
- Mean KS: 0.172

### Overall Assessment

**EXCELLENT** - Model shows strong translation quality



## Feature Quality Analysis

### Best Performing Features (eICU Round-trip)

| Feature | R² | Correlation |
|---------|----|-------------|
| SpO2_max | 1.000 | 1.000 |
| WBC_min | 0.999 | 0.999 |
| HR_std | 0.997 | 0.999 |
| WBC_max | 0.997 | 0.998 |
| SpO2_mean | 0.997 | 0.999 |

### Best Performing Features (MIMIC Round-trip)

| Feature | R² | Correlation |
|---------|----|-------------|
| SpO2_max | 0.998 | 0.999 |
| WBC_min | 0.998 | 0.999 |
| WBC_max | 0.998 | 0.999 |
| SpO2_mean | 0.997 | 0.999 |
| HR_std | 0.997 | 0.999 |

### Worst Performing Features (eICU Round-trip)

| Feature | R² | Correlation |
|---------|----|-------------|
| RR_min | 0.966 | 0.996 |
| HR_max | 0.979 | 0.997 |
| Creat_min | 0.984 | 0.992 |
| HR_min | 0.985 | 0.995 |
| RR_max | 0.986 | 0.998 |

### Worst Performing Features (MIMIC Round-trip)

| Feature | R² | Correlation |
|---------|----|-------------|
| RR_min | 0.953 | 0.995 |
| RR_max | 0.974 | 0.997 |
| SpO2_min | 0.977 | 0.997 |
| HR_max | 0.981 | 0.997 |
| HR_min | 0.986 | 0.996 |


## Per-Feature IQR Analysis

*Detailed breakdown of IQR-normalized errors for each clinical feature*

### Reconstruction (A→A')

| Feature | eICU % in 0.5 IQR | eICU % in 1.0 IQR | MIMIC % in 0.5 IQR | MIMIC % in 1.0 IQR | eICU % < 0.05 abs | MIMIC % < 0.05 abs |
|---------|-------------------|-------------------|--------------------|--------------------|-----------------|------------------|
| HR_min | 100.0% | 100.0% | 100.0% | 100.0% | 93.8% | 95.6% |
| HR_max | 100.0% | 100.0% | 100.0% | 100.0% | 78.7% | 75.0% |
| HR_mean | 100.0% | 100.0% | 100.0% | 100.0% | 97.8% | 96.8% |
| HR_std | 100.0% | 100.0% | 100.0% | 100.0% | 93.6% | 89.2% |
| RR_min | 100.0% | 100.0% | 100.0% | 100.0% | 91.6% | 91.1% |
| RR_max | 100.0% | 100.0% | 100.0% | 100.0% | 83.3% | 78.1% |
| RR_mean | 100.0% | 100.0% | 100.0% | 100.0% | 99.6% | 99.8% |
| RR_std | 100.0% | 100.0% | 100.0% | 100.0% | 94.7% | 89.3% |
| SpO2_min | 100.0% | 100.0% | 100.0% | 100.0% | 56.3% | 79.0% |
| SpO2_max | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| SpO2_mean | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 99.3% |
| SpO2_std | 100.0% | 100.0% | 100.0% | 100.0% | 95.9% | 90.6% |
| WBC_min | 100.0% | 100.0% | 100.0% | 100.0% | 96.7% | 92.5% |
| WBC_max | 100.0% | 100.0% | 100.0% | 100.0% | 94.8% | 88.6% |
| WBC_mean | 100.0% | 100.0% | 100.0% | 100.0% | 94.2% | 91.2% |
| WBC_std | 100.0% | 100.0% | 100.0% | 100.0% | 95.9% | 95.9% |
| Na_min | 100.0% | 100.0% | 100.0% | 100.0% | 87.5% | 85.6% |
| Na_max | 100.0% | 100.0% | 100.0% | 100.0% | 88.8% | 83.1% |
| Na_mean | 100.0% | 100.0% | 100.0% | 100.0% | 91.7% | 77.0% |
| Na_std | 100.0% | 100.0% | 100.0% | 100.0% | 93.4% | 86.9% |
| Creat_min | 100.0% | 100.0% | 99.9% | 100.0% | 82.8% | 83.7% |
| Creat_max | 100.0% | 100.0% | 100.0% | 100.0% | 93.4% | 63.7% |
| Creat_mean | 100.0% | 100.0% | 100.0% | 100.0% | 89.1% | 68.2% |
| Creat_std | 100.0% | 100.0% | 100.0% | 100.0% | 92.5% | 79.4% |

**Best Performing Features (Reconstruction, % within 0.5 IQR):**

eICU:
- HR_min: 100.0%
- HR_max: 100.0%
- HR_mean: 100.0%
- HR_std: 100.0%
- RR_min: 100.0%

MIMIC:
- HR_min: 100.0%
- HR_max: 100.0%
- HR_mean: 100.0%
- HR_std: 100.0%
- RR_min: 100.0%

**Worst Performing Features (Reconstruction, % within 0.5 IQR):**

eICU:
- Creat_std: 100.0%
- SpO2_min: 100.0%
- Na_min: 100.0%
- Creat_mean: 100.0%
- Creat_max: 100.0%

MIMIC:
- Creat_min: 99.9%
- Creat_mean: 100.0%
- Creat_std: 100.0%
- Creat_max: 100.0%
- Na_std: 100.0%

### Cycle Consistency (A→B'→A')

| Feature | eICU % in 0.5 IQR | eICU % in 1.0 IQR | MIMIC % in 0.5 IQR | MIMIC % in 1.0 IQR | eICU % < 0.05 abs | MIMIC % < 0.05 abs |
|---------|-------------------|-------------------|--------------------|--------------------|-----------------|------------------|
| HR_min | 100.0% | 100.0% | 100.0% | 100.0% | 48.6% | 50.2% |
| HR_max | 100.0% | 100.0% | 100.0% | 100.0% | 25.1% | 22.8% |
| HR_mean | 100.0% | 100.0% | 100.0% | 100.0% | 71.7% | 73.2% |
| HR_std | 100.0% | 100.0% | 100.0% | 100.0% | 78.4% | 76.4% |
| RR_min | 100.0% | 100.0% | 100.0% | 100.0% | 41.9% | 33.0% |
| RR_max | 100.0% | 100.0% | 100.0% | 100.0% | 33.1% | 22.5% |
| RR_mean | 100.0% | 100.0% | 100.0% | 100.0% | 95.5% | 97.1% |
| RR_std | 100.0% | 100.0% | 100.0% | 100.0% | 52.5% | 53.0% |
| SpO2_min | 100.0% | 100.0% | 100.0% | 100.0% | 27.9% | 19.8% |
| SpO2_max | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| SpO2_mean | 100.0% | 100.0% | 100.0% | 100.0% | 99.3% | 98.5% |
| SpO2_std | 100.0% | 100.0% | 100.0% | 100.0% | 58.7% | 55.5% |
| WBC_min | 100.0% | 100.0% | 100.0% | 100.0% | 93.5% | 89.7% |
| WBC_max | 100.0% | 100.0% | 100.0% | 100.0% | 64.4% | 68.9% |
| WBC_mean | 100.0% | 100.0% | 100.0% | 100.0% | 63.2% | 62.5% |
| WBC_std | 100.0% | 100.0% | 100.0% | 100.0% | 62.0% | 67.1% |
| Na_min | 100.0% | 100.0% | 100.0% | 100.0% | 39.3% | 43.2% |
| Na_max | 100.0% | 100.0% | 100.0% | 100.0% | 50.1% | 49.9% |
| Na_mean | 100.0% | 100.0% | 100.0% | 100.0% | 62.7% | 64.4% |
| Na_std | 100.0% | 100.0% | 100.0% | 100.0% | 60.4% | 64.0% |
| Creat_min | 98.0% | 100.0% | 99.0% | 100.0% | 39.8% | 38.4% |
| Creat_max | 100.0% | 100.0% | 100.0% | 100.0% | 28.1% | 29.0% |
| Creat_mean | 100.0% | 100.0% | 100.0% | 100.0% | 17.8% | 20.6% |
| Creat_std | 100.0% | 100.0% | 100.0% | 100.0% | 27.2% | 35.6% |

**Best Performing Features (Cycle, % within 0.5 IQR):**

eICU:
- HR_min: 100.0%
- HR_max: 100.0%
- HR_mean: 100.0%
- HR_std: 100.0%
- RR_min: 100.0%

MIMIC:
- HR_min: 100.0%
- HR_max: 100.0%
- HR_mean: 100.0%
- HR_std: 100.0%
- RR_min: 100.0%

**Worst Performing Features (Cycle, % within 0.5 IQR):**

eICU:
- Creat_min: 98.0%
- Creat_std: 100.0%
- Creat_mean: 100.0%
- Creat_max: 100.0%
- Na_std: 100.0%

MIMIC:
- Creat_min: 99.0%
- Creat_std: 100.0%
- Creat_mean: 100.0%
- Creat_max: 100.0%
- Na_std: 100.0%



## Distribution Analysis

*KS statistic thresholds: <0.1=excellent, <0.2=good, <0.3=acceptable*

### Features with Good Distribution Matching (KS < 0.2)

- **eICU→MIMIC**: 19 features
  - HR_min, HR_max, HR_mean, HR_std, RR_min, RR_max, RR_mean, RR_std, SpO2_mean, SpO2_std (and 9 more)
- **MIMIC→eICU**: 20 features
  - HR_min, HR_max, HR_mean, HR_std, RR_min, RR_max, RR_mean, RR_std, SpO2_min, SpO2_mean (and 10 more)

### Features with Poor Distribution Matching (KS ≥ 0.3)

- **eICU→MIMIC**: 4 features
  - SpO2_max, WBC_std, Na_std, Creat_std
- **MIMIC→eICU**: 4 features
  - SpO2_max, WBC_std, Na_std, Creat_std


## Missingness Analysis

### Performance by Feature Count Buckets

| Bucket | eICU Samples | MIMIC Samples | eICU MSE | MIMIC MSE |
|--------|--------------|---------------|----------|----------|
| very_low | 99 | 63 | 0.0422 | 0.0462 |
| low | 351 | 57 | 0.0546 | 0.0402 |
| medium | 1595 | 19 | 0.0465 | 0.0424 |
| high | 3548 | 93 | 0.0629 | 0.0636 |
| very_high | 14188 | 8365 | 0.0651 | 0.0622 |


## Demographic Analysis

### Age-based Performance

| Age Group | Samples | MSE | Mean Age |
|-----------|---------|-----|----------|
| young | 1120 | 0.0890 | -1.8 |
| adult | 2084 | 0.0646 | -1.1 |
| middle | 4860 | 0.0544 | -0.5 |
| senior | 6756 | 0.0571 | 0.1 |
| elderly | 4961 | 0.0725 | 0.8 |

### Gender-based Performance

| Gender | Samples | MSE |
|--------|---------|-----|
| gender_0 | 9081 | 0.0660 |
| gender_1 | 10700 | 0.0603 |


## Recommendations

### Distribution Matching Improvements

Features with poor distribution matching: SpO2_max, WBC_std, Na_std, Creat_std

**Suggested actions**:
- Increase MMD loss weight for problematic features
- Consider per-feature MMD loss
- Review feature scaling and normalization

### General Recommendations

1. **Monitor training stability**: Ensure loss components are balanced
2. **Validate on held-out data**: Test translation quality on unseen patients
3. **Consider ensemble methods**: Combine multiple models for better robustness
4. **Domain adaptation**: Fine-tune on target domain data if available
5. **Clinical validation**: Validate translations with domain experts
