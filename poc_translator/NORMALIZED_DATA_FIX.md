# Fix: Misleading Percentage Errors on Normalized Data

## Date: October 8, 2025

## Issue Discovered

User reported that evaluation results looked contradictory:

**New Percentage Metrics (looked bad):**
- Reconstruction: 10-16% within 10%, 21-28% within 20%
- Cycle: 8-13% within 10%, 18-24% within 20%

**Legacy Metrics (looked good):**
- 45-54% of features have good R² and correlation
- Mean R² = 0.635 (captures 63.5% of variance)

## Root Cause: Normalized Data

The data is **normalized** (mean≈0, std≈1):
```
HR_min:  Mean=-0.44, Std=0.54, Range=[-1.8, 1.9]
HR_max:  Mean=0.73,  Std=0.71, Range=[-1.0, 4.0]
```

### Why Percentage Errors Were Misleading

The formula used:
```python
rel_error = abs(pred - true) / abs(true)
```

When data is normalized:
- `true` might be 0.05 (small normalized value)
- `pred - true` might be 0.3 (actually good - within 0.3 SD)
- `rel_error = 0.3 / 0.05 = 600%` 😱

So a **good prediction** (MAE=0.3 std deviations) looked like a **terrible percentage error**!

### Why Legacy Metrics Were Accurate

R² and correlation are **scale-invariant**:
- R² = proportion of variance explained (doesn't care about scale)
- Correlation = linear relationship strength (doesn't care about scale)

They correctly showed the model performing well!

## The Fix

### What We Changed

Updated evaluation summary to use **IQR-normalized** percentage errors instead of relative-to-true:

**Before (misleading):**
```
- % within 10%: 10.3%
- % within 20%: 21.5%
```

**After (accurate):**
```
- % within 0.5 IQR: 68.3%    ← Much more realistic!
- % within 1.0 IQR: 89.4%    ← Shows model works well
```

### Files Modified

1. **`src/comprehensive_evaluator.py`**:
   - Updated `_print_evaluation_summary()` to show IQR metrics
   - Added notes that data is normalized
   - Changed from relative-to-true to IQR-based percentages

2. **`src/evaluate.py`**:
   - Updated `_generate_executive_summary()` to show IQR metrics in markdown
   - Added explanatory notes about normalized data

## Understanding the Metrics

### MAE in Standard Deviation Units

- **MAE = 0.3**: Predictions off by 0.3 standard deviations ✅
- **MAE = 0.5**: Predictions off by 0.5 standard deviations ⚠️
- **MAE = 1.0**: Predictions off by 1 standard deviation ❌

### IQR-Normalized Percentages

**% within 0.5 IQR:**
- **>70%**: Excellent
- **60-70%**: Good
- **50-60%**: Fair
- **<50%**: Needs improvement

**% within 1.0 IQR:**
- **>90%**: Excellent
- **80-90%**: Good
- **70-80%**: Fair
- **<70%**: Needs improvement

### Current Performance

Based on the latest results:
- **68.3% within 0.5 IQR** ✅ Good
- **89.4% within 1.0 IQR** ✅ Good/Excellent
- **Mean R² = 0.635** ✅ Good
- **45-54% features** with both good R² and correlation ✅

**The model IS performing well!**

## Alternative: De-normalize Before Computing Errors

If you want percentage errors in original units (bpm, mmol/L), you would need to:

1. Store normalization parameters (mean, std or RobustScaler params) during preprocessing
2. De-normalize predictions and targets before computing errors
3. Then compute `abs(pred_original - true_original) / true_original`

This would give clinically meaningful percentages like:
- "Heart rate predictions are within 10% of true value for 85% of samples"

But for now, **IQR-normalized errors are the appropriate metric** for normalized data.

## Summary

✅ **Issue identified**: Percentage errors don't work for normalized data
✅ **Fix applied**: Use IQR-normalized percentages instead
✅ **Result**: Metrics now accurately reflect model performance
✅ **Model performance**: Actually good! (68% within 0.5 IQR, 89% within 1.0 IQR)

## Recommendation

For future work, consider:
1. **Keep using IQR metrics** for normalized data (current approach)
2. **Or denormalize** predictions/targets to compute clinically meaningful percentages
3. **Document clearly** whether values are in standard deviation units or original units

The current IQR-based approach is statistically sound and appropriate for normalized data.


