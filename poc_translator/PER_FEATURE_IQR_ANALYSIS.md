# Per-Feature IQR Analysis

## Overview

Added comprehensive per-feature IQR analysis to the evaluation report, providing detailed breakdown of model performance for each clinical feature.

## What Was Added

### New Evaluation Report Section

**Location**: Added as Section 3 in the comprehensive evaluation report (between "Feature Quality Analysis" and "Distribution Analysis")

**Content**:
1. **Per-feature tables** showing IQR-normalized error percentages for both reconstruction and cycle consistency
2. **Best/worst performing features** ranked by % within 0.5 IQR
3. **Separate analysis** for both eICU and MIMIC domains

## Tables Included

### Reconstruction (A→A')

| Feature | eICU % in 0.5 IQR | eICU % in 1.0 IQR | MIMIC % in 0.5 IQR | MIMIC % in 1.0 IQR |
|---------|-------------------|-------------------|--------------------|--------------------|
| HR_mean | 75.3% | 92.1% | 78.6% | 94.3% |
| ... | ... | ... | ... | ... |

### Cycle Consistency (A→B'→A')

| Feature | eICU % in 0.5 IQR | eICU % in 1.0 IQR | MIMIC % in 0.5 IQR | MIMIC % in 1.0 IQR |
|---------|-------------------|-------------------|--------------------|--------------------|
| HR_mean | 68.1% | 89.7% | 72.4% | 91.2% |
| ... | ... | ... | ... | ... |

## Best/Worst Performers

For both reconstruction and cycle, the report now shows:

**Best Performing Features**:
- Top 5 features for eICU (highest % within 0.5 IQR)
- Top 5 features for MIMIC

**Worst Performing Features**:
- Bottom 5 features for eICU (lowest % within 0.5 IQR)
- Bottom 5 features for MIMIC

## Interpretation Guide

### % within 0.5 IQR (Tight Tolerance)
- **>80%**: Excellent - predictions very close to true values
- **70-80%**: Good - predictions reasonably accurate
- **60-70%**: Fair - predictions acceptable but room for improvement
- **<60%**: Poor - feature needs attention

### % within 1.0 IQR (Loose Tolerance)
- **>95%**: Excellent - almost all predictions reasonable
- **90-95%**: Good - most predictions reasonable
- **80-90%**: Fair - many predictions reasonable
- **<80%**: Poor - many predictions outside acceptable range

## Why This Is Important

### 1. **Feature-Specific Insights**
Instead of only seeing overall averages, you can now identify:
- Which specific features (e.g., `HR_mean`, `SpO2_std`) are well-reconstructed
- Which features need improvement

### 2. **Targeted Model Improvements**
- If certain features consistently underperform, you can:
  - Adjust feature weights in the loss
  - Investigate if those features need different preprocessing
  - Check if the worst features are being properly tracked for Wasserstein loss

### 3. **Clinical Validation**
- Identify if clinically important features (e.g., vital signs) are being accurately preserved
- Ensure critical features meet acceptable accuracy thresholds

### 4. **Domain Comparison**
- See if certain features are harder to reconstruct in eICU vs MIMIC
- Identify domain-specific challenges

## Example Use Cases

### Finding Problematic Features
```markdown
**Worst Performing Features (Reconstruction, % within 0.5 IQR):**

eICU:
- Temp_std: 42.3%  ← Needs attention!
- Creat_max: 48.1%
- BUN_std: 51.2%
```

This tells you that temperature variability (`Temp_std`) is poorly reconstructed and might need:
- More training emphasis (add to worst-K for Wasserstein)
- Better preprocessing (check for outliers)
- Different feature engineering

### Confirming Strong Features
```markdown
**Best Performing Features (Reconstruction, % within 0.5 IQR):**

eICU:
- HR_mean: 85.7%  ← Excellent!
- SpO2_mean: 83.2%  ← Excellent!
- RR_mean: 81.4%  ← Excellent!
```

This confirms that core vital signs are being accurately preserved.

## Implementation Details

### Code Location
- **Function**: `_generate_per_feature_iqr_analysis()` in `src/evaluate.py` (lines 837-958)
- **Called from**: `generate_comprehensive_report()` (line 677)

### Data Source
- Uses the `comprehensive` results from `comprehensive_results.json`
- Specifically the `eicu_reconstruction_errors`, `mimic_reconstruction_errors`, `eicu_cycle_errors`, and `mimic_cycle_errors` dictionaries
- Feature names from `correlation_metrics` CSV

### Report Position
Section 3 in the markdown report, between:
1. Executive Summary
2. Feature Quality Analysis (R², correlation)
3. **→ Per-Feature IQR Analysis** ← NEW
4. Distribution Analysis (KS statistics)
5. Missingness Analysis
6. Demographic Analysis
7. Recommendations

## Benefits

### ✅ Granular Performance Tracking
- Track each feature's accuracy individually
- Monitor improvement over training epochs

### ✅ Normalized Data Compatible
- IQR-normalized metrics work correctly with standardized data
- Avoids misleading percentage errors from small true values

### ✅ Actionable Insights
- Directly identifies which features to prioritize
- Guides hyperparameter tuning (e.g., which features for worst-K)

### ✅ Clinical Relevance
- Maps directly to clinical features (vitals, labs)
- Easy to interpret for clinical collaborators

## Next Steps

When you re-run evaluation with `--comprehensive`, the report will automatically include this new section:

```bash
python src/evaluate.py \
    --config conf/config.yml \
    --model checkpoints/final_model.ckpt \
    --comprehensive
```

The comprehensive evaluation report will now show:
1. Overall metrics (executive summary)
2. Legacy R² and correlation metrics
3. **New: Detailed per-feature IQR tables and rankings** ⭐
4. Distribution analysis
5. Missingness and demographic analysis
6. Recommendations

---

**Status**: ✅ Complete - No linting errors
**Files Modified**: `src/evaluate.py` (added 122 lines)
**Report Impact**: Adds detailed per-feature breakdown section to markdown report


