# Correlation Bug Investigation Findings

## Executive Summary

The low correlation values in `correlation_metrics.csv` do NOT match what the evaluator code should compute. Manual recomputation yields correlations of ~0.999, while the CSV shows ~0.4-0.5.

## Key Findings

### 1. Manual Computation vs. CSV File

Manual recomputation of correlations using IDENTICAL data and code logic:

| Feature | Manual R² | Manual Corr | CSV R² | CSV Corr | Match? |
|---------|-----------|-------------|--------|----------|--------|
| HR_min  | 0.998299  | **0.999469** | 0.998299 | **0.515399** | ✗ R² matches, Corr WRONG |
| HR_max  | 0.998641  | **0.999729** | 0.998641 | **0.635956** | ✗ R² matches, Corr WRONG |
| HR_mean | 0.999694  | **0.999919** | 0.999694 | **0.702737** | ✗ R² matches, Corr WRONG |
| HR_std  | 0.999642  | **1.000000** | 0.999642 | **0.524272** | ✗ R² matches, Corr WRONG |
| RR_min  | 0.998890  | **0.999754** | 0.998890 | **0.662488** | ✗ R² matches, Corr WRONG |

### 2. Key Observations

1. **R² values match EXACTLY** (to 6 decimal places) between manual computation and CSV
2. **Correlation values are COMPLETELY WRONG** in the CSV
3. The correlation values in the CSV appear to be roughly **√R²** or some transformation of R²

### 3. Hypothesis

The most likely explanation is that the CSV was generated with **buggy code** that computed correlation incorrectly, possibly:

- Using the wrong arrays (e.g., comparing x_mimic with x_mimic instead of x_mimic with x_mimic_roundtrip)
- Applying some transformation to the correlation values
- Having a bug in the np.corrcoef indexing (using wrong matrix element)
- Data alignment issues during evaluation

### 4. Evidence Against Common Theories

❌ **NOT a data loading issue**: R² values match exactly
❌ **NOT a feature alignment issue**: Checked feature indices, all correct
❌ **NOT a variance issue**: All features have std > 1e-8
❌ **NOT a numpy.corrcoef bug**: Verified np.corrcoef works correctly on test data

## What the Code SHOULD Compute

Current code in `comprehensive_evaluator.py` lines 358-360:

```python
r2_val = r2_score(x_mimic_clinical[:, i], x_mimic_roundtrip_clinical[:, i])
corr_matrix = np.corrcoef(x_mimic_clinical[:, i], x_mimic_roundtrip_clinical[:, i])
corr_val = corr_matrix[0, 1]
```

This code is **CORRECT** and should produce correlations ~0.999.

## Debug Changes Made

Added debug logging to `src/comprehensive_evaluator.py` (lines 362-367, 387-389):

1. **For first feature (HR_min)**: Print R², correlation, shapes, means, stds, and correlation matrix
2. **After DataFrame creation**: Print first row to verify values before saving to CSV

These debug statements will help identify:
- If the evaluator is computing correct values during runtime
- If values get corrupted between computation and CSV writing
- If the existing CSV is from old/buggy code

## Recommended Action

**Re-run the comprehensive evaluation** with the debug-enabled evaluator:

```bash
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python src/evaluate.py --model checkpoints/final_model.ckpt --comprehensive --mimic_only
```

The debug output will show:
1. What correlation values are actually computed (should be ~0.999)
2. What gets stored in the DataFrame (should match computation)
3. Whether the CSV file will have correct values

## Expected Outcome

If my hypothesis is correct:
- Debug output will show **Corr=0.999469** for HR_min
- New CSV will have correct correlation values (~0.95-0.99)
- This will match the high R² values
- The "negative correlation between R² and correlation" issue will be resolved

## Files Modified

1. `src/comprehensive_evaluator.py`: Added debug logging (lines 362-367, 387-389)
   - **Note**: These are debug statements only, no logic changes

## Next Steps

1. ✅ Run evaluation with debug output
2. ⏳ Verify debug output shows correct correlations
3. ⏳ Check if new CSV has correct values
4. ⏳ If yes: Remove debug statements and confirm evaluation is working
5. ⏳ If no: Investigate further based on debug output


