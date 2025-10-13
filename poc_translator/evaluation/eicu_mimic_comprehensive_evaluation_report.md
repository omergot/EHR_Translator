# Comprehensive Evaluation Report (Simplified Model)

*Generated for simplified CycleVAE with 3 losses: reconstruction, cycle, conditional Wasserstein*

*Note: Missing flags, Age, and Gender are excluded from evaluation (input-only)*

## Executive Summary

### 📊 Reconstruction Quality (A→A')

*Note: Data is normalized - MAE in standard deviation units, use IQR metrics*

**eICU Reconstruction:**
- MAE: 0.1557 (std dev units)
- % within 0.5 IQR: 87.3%
- % within 1.0 IQR: 97.2%

**MIMIC Reconstruction:**
- MAE: 0.1509 (std dev units)
- % within 0.5 IQR: 88.5%
- % within 1.0 IQR: 97.8%

### 🔄 Cycle Consistency (A→B'→A')

*Note: Data is normalized - use IQR metrics for meaningful percentages*

**eICU Cycle:**
- MAE: 0.1754 (std dev units)
- % within 0.5 IQR: 85.7%
- % within 1.0 IQR: 96.9%

**MIMIC Cycle:**
- MAE: 0.1793 (std dev units)
- % within 0.5 IQR: 86.6%
- % within 1.0 IQR: 97.5%

### 🧠 Latent Space Analysis

**Original Domains (eICU vs MIMIC):**
- Euclidean Distance: 5.8138
- Cosine Similarity: 0.9475
- KL Divergence: 7.4535

**After Translation (eICU→MIMIC vs real MIMIC):**
- Euclidean Distance: 6.1568
- Cosine Similarity: 0.9221
- KL Divergence: 8.4804

### 📈 Distribution Matching

- Mean Wasserstein Distance: 0.1167
- Mean KS Statistic: 0.7124

---

### Legacy Metrics (Clinical Features Only)

**Translation Quality Metrics:**
- **Clinical Features Evaluated**: 24

**R² (Variance Explained) - Target: > 0.5:**
- **eICU**: 20/24 features (83.3%), Mean R²: 0.771
- **MIMIC**: 20/24 features (83.3%), Mean R²: 0.707

**Correlation (Linear Relationship) - Target: > 0.7:**
- **eICU**: 3/24 features (12.5%), Mean corr: 0.548
- **MIMIC**: 3/24 features (12.5%), Mean corr: 0.536

**Combined (R² > 0.5 AND correlation > 0.7):**
- **eICU**: 3/24 features (12.5%)
- **MIMIC**: 3/24 features (12.5%)

**Distribution Matching (KS < 0.3, p > 0.05):**
- **eICU→MIMIC Translation**: 0/24 features (0.0%)
- **MIMIC→eICU Translation**: 0/24 features (0.0%)

### Overall Assessment

**POOR** - Model shows weak translation quality requiring significant improvements



## Feature Quality Analysis

### Best Performing Features (eICU Round-trip)

| Feature | R² | Correlation |
|---------|----|-------------|
| Creat_max | 0.972 | 0.556 |
| Na_mean | 0.971 | 0.545 |
| Creat_mean | 0.968 | 0.538 |
| WBC_mean | 0.966 | 0.527 |
| WBC_max | 0.965 | 0.986 |

### Best Performing Features (MIMIC Round-trip)

| Feature | R² | Correlation |
|---------|----|-------------|
| WBC_mean | 0.973 | 0.669 |
| Creat_max | 0.971 | 0.517 |
| Na_mean | 0.971 | 0.630 |
| Creat_mean | 0.970 | 0.539 |
| WBC_max | 0.968 | 0.794 |

### Worst Performing Features (eICU Round-trip)

| Feature | R² | Correlation |
|---------|----|-------------|
| SpO2_max | -0.122 | 0.111 |
| RR_min | 0.134 | 0.206 |
| SpO2_mean | 0.288 | 0.299 |
| RR_mean | 0.389 | 0.614 |
| Na_std | 0.735 | 0.458 |

### Worst Performing Features (MIMIC Round-trip)

| Feature | R² | Correlation |
|---------|----|-------------|
| SpO2_max | -1.664 | 0.004 |
| RR_min | 0.144 | 0.345 |
| RR_mean | 0.418 | 0.679 |
| SpO2_mean | 0.444 | 0.363 |
| Na_std | 0.579 | 0.421 |


## Per-Feature IQR Analysis

*Detailed breakdown of IQR-normalized errors for each clinical feature*

### Reconstruction (A→A')

| Feature | eICU % in 0.5 IQR | eICU % in 1.0 IQR | MIMIC % in 0.5 IQR | MIMIC % in 1.0 IQR |
|---------|-------------------|-------------------|--------------------|--------------------|--|
| HR_min | 89.8% | 98.7% | 92.4% | 99.2% |
| HR_max | 96.0% | 99.5% | 98.5% | 99.9% |
| HR_mean | 92.9% | 99.5% | 95.6% | 99.7% |
| HR_std | 91.8% | 99.3% | 94.5% | 99.6% |
| RR_min | 51.3% | 87.4% | 61.6% | 89.8% |
| RR_max | 84.6% | 99.2% | 92.0% | 99.8% |
| RR_mean | 61.4% | 88.8% | 58.0% | 88.5% |
| RR_std | 89.8% | 99.3% | 92.3% | 99.6% |
| SpO2_min | 95.9% | 99.9% | 99.1% | 100.0% |
| SpO2_max | 67.1% | 93.7% | 89.9% | 99.4% |
| SpO2_mean | 65.6% | 92.6% | 63.4% | 90.0% |
| SpO2_std | 91.5% | 99.7% | 95.8% | 99.9% |
| WBC_min | 95.2% | 99.5% | 95.4% | 99.5% |
| WBC_max | 97.8% | 99.8% | 98.9% | 99.9% |
| WBC_mean | 98.4% | 100.0% | 99.7% | 100.0% |
| WBC_std | 78.2% | 89.2% | 66.6% | 92.9% |
| Na_min | 97.5% | 99.7% | 97.4% | 99.6% |
| Na_max | 97.5% | 99.7% | 97.5% | 99.6% |
| Na_mean | 99.5% | 100.0% | 99.5% | 100.0% |
| Na_std | 79.6% | 93.1% | 68.8% | 93.7% |
| Creat_min | 93.8% | 99.9% | 97.7% | 99.8% |
| Creat_max | 98.4% | 99.8% | 99.5% | 100.0% |
| Creat_mean | 98.2% | 100.0% | 99.4% | 100.0% |
| Creat_std | 84.3% | 95.7% | 69.8% | 95.6% |

**Best Performing Features (Reconstruction, % within 0.5 IQR):**

eICU:
- Na_mean: 99.5%
- Creat_max: 98.4%
- WBC_mean: 98.4%
- Creat_mean: 98.2%
- WBC_max: 97.8%

MIMIC:
- WBC_mean: 99.7%
- Na_mean: 99.5%
- Creat_max: 99.5%
- Creat_mean: 99.4%
- SpO2_min: 99.1%

**Worst Performing Features (Reconstruction, % within 0.5 IQR):**

eICU:
- RR_min: 51.3%
- RR_mean: 61.4%
- SpO2_mean: 65.6%
- SpO2_max: 67.1%
- WBC_std: 78.2%

MIMIC:
- RR_mean: 58.0%
- RR_min: 61.6%
- SpO2_mean: 63.4%
- WBC_std: 66.6%
- Na_std: 68.8%

### Cycle Consistency (A→B'→A')

| Feature | eICU % in 0.5 IQR | eICU % in 1.0 IQR | MIMIC % in 0.5 IQR | MIMIC % in 1.0 IQR |
|---------|-------------------|-------------------|--------------------|--------------------|--|
| HR_min | 86.7% | 97.9% | 86.9% | 98.4% |
| HR_max | 93.8% | 99.3% | 96.3% | 99.6% |
| HR_mean | 89.7% | 99.2% | 91.1% | 99.5% |
| HR_std | 88.0% | 98.8% | 90.3% | 99.3% |
| RR_min | 50.5% | 87.1% | 61.2% | 89.7% |
| RR_max | 81.6% | 98.8% | 89.3% | 99.3% |
| RR_mean | 60.9% | 88.5% | 57.5% | 88.3% |
| RR_std | 86.8% | 99.0% | 88.7% | 99.2% |
| SpO2_min | 91.9% | 99.8% | 97.3% | 100.0% |
| SpO2_max | 65.3% | 93.3% | 92.4% | 99.6% |
| SpO2_mean | 64.1% | 92.6% | 61.3% | 90.0% |
| SpO2_std | 87.4% | 99.4% | 92.4% | 99.5% |
| WBC_min | 93.9% | 99.3% | 93.7% | 99.4% |
| WBC_max | 97.7% | 99.8% | 97.6% | 99.8% |
| WBC_mean | 97.8% | 99.9% | 98.7% | 100.0% |
| WBC_std | 76.9% | 88.3% | 63.2% | 91.5% |
| Na_min | 96.3% | 99.5% | 95.9% | 99.4% |
| Na_max | 96.8% | 99.6% | 96.0% | 99.4% |
| Na_mean | 99.0% | 99.9% | 98.8% | 99.9% |
| Na_std | 79.3% | 92.5% | 67.6% | 93.2% |
| Creat_min | 92.9% | 99.8% | 94.8% | 99.8% |
| Creat_max | 98.2% | 99.9% | 98.3% | 99.9% |
| Creat_mean | 97.9% | 100.0% | 98.5% | 100.0% |
| Creat_std | 83.7% | 93.8% | 69.7% | 95.5% |

**Best Performing Features (Cycle, % within 0.5 IQR):**

eICU:
- Na_mean: 99.0%
- Creat_max: 98.2%
- Creat_mean: 97.9%
- WBC_mean: 97.8%
- WBC_max: 97.7%

MIMIC:
- Na_mean: 98.8%
- WBC_mean: 98.7%
- Creat_mean: 98.5%
- Creat_max: 98.3%
- WBC_max: 97.6%

**Worst Performing Features (Cycle, % within 0.5 IQR):**

eICU:
- RR_min: 50.5%
- RR_mean: 60.9%
- SpO2_mean: 64.1%
- SpO2_max: 65.3%
- WBC_std: 76.9%

MIMIC:
- RR_mean: 57.5%
- RR_min: 61.2%
- SpO2_mean: 61.3%
- WBC_std: 63.2%
- Na_std: 67.6%



## Distribution Analysis

### Features with Good Distribution Matching

- **eICU→MIMIC**: 0 features
- **MIMIC→eICU**: 0 features

### Features with Poor Distribution Matching

- **eICU→MIMIC**: 24 features
  - HR_min, HR_max, HR_mean, HR_std, RR_min, RR_max, RR_mean, RR_std, SpO2_min, SpO2_max (and 14 more)
- **MIMIC→eICU**: 24 features
  - HR_min, HR_max, HR_mean, HR_std, RR_min, RR_max, RR_mean, RR_std, SpO2_min, SpO2_max (and 14 more)


## Missingness Analysis

### Performance by Feature Count Buckets

| Bucket | eICU Samples | MIMIC Samples | eICU MSE | MIMIC MSE |
|--------|--------------|---------------|----------|----------|
| very_low | 129 | 89 | 0.1783 | 0.1493 |
| low | 462 | 69 | 0.1782 | 0.1433 |
| medium | 2078 | 30 | 0.1547 | 0.1424 |
| high | 4706 | 347 | 0.1694 | 0.1728 |
| very_high | 19000 | 10927 | 0.1822 | 0.1931 |


## Demographic Analysis

### Age-based Performance

| Age Group | Samples | MSE | Mean Age |
|-----------|---------|-----|----------|
| young | 1466 | 0.2168 | -1.8 |
| adult | 2773 | 0.1759 | -1.1 |
| middle | 6487 | 0.1564 | -0.5 |
| senior | 9053 | 0.1660 | 0.1 |
| elderly | 6596 | 0.2066 | 0.8 |

### Gender-based Performance

| Gender | Samples | MSE |
|--------|---------|-----|
| gender_0 | 12081 | 0.1808 |
| gender_1 | 14294 | 0.1750 |


## Recommendations

### Feature-specific Improvements

The following features show poor round-trip consistency and may need special attention:

- **eICU round-trip issues**: HR_max, HR_mean, HR_std, RR_min, RR_mean (and 16 more)
- **MIMIC round-trip issues**: HR_min, HR_max, HR_mean, HR_std, RR_min (and 16 more)

**Suggested actions**:
- Review feature preprocessing and normalization
- Consider feature-specific loss weighting
- Investigate domain-specific feature distributions

### Distribution Matching Improvements

Features with poor distribution matching: HR_min, HR_max, HR_mean, HR_std, RR_min (and 19 more)

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
