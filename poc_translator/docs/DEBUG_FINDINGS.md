# Debug Findings: Evaluation Metrics Sanity Checks

## Summary

Running the debug checks on the mimic-only comprehensive evaluation revealed **critical issues** with the R² vs Pearson correlation calculations, confirming the user's suspicion of bugs in the metric computation.

---

## 1. R² vs Pearson Correlation Discrepancy ✓ **[NOT A BUG - EXPLAINED]**

### Expected Behavior
For simple linear regression with perfect predictions, we expect:
```
R² ≈ correlation²
```

But R² **CAN** differ from correlation² when:
1. Predictions are systematically biased
2. Non-linear transformations occur
3. Model performance is worse than predicting the mean

### Actual Results (After Re-run)

**eICU Roundtrip:**
- Mean |R² - corr²|: **0.160**
- Max |R² - corr²|: **1.362**
- Features with large diff (>0.05): **13/24** (54%)

**MIMIC Roundtrip:**
- Mean |R² - corr²|: **0.179**
- Max |R² - corr²|: **2.227**
- Features with large diff (>0.05): **11/24** (46%)

### "Worst Offenders" (Actually Problematic Features)

**SpO2_max:**
- **eICU**: R² = -1.36, correlation ≈ 0, MSE = 0.0008
- **MIMIC**: R² = -2.23, correlation ≈ 0, MSE = 0.0007
- **Interpretation**: Negative R² means roundtrip is **worse than predicting the mean**
- The model **completely fails** to reconstruct SpO2_max through cycle

**RR_min:**
- **eICU**: R² = -0.03, MSE = 0.109
- **MIMIC**: R² = -0.01, MSE = 0.109
- **Interpretation**: Barely worse than mean prediction

**WBC_max, Na_max, Creat features:**
- Large |R² - corr²| suggests **systematic bias** in roundtrip predictions
- Model learns but with distortion

### Root Cause: THIS IS NOT A BUG! ✓

**Negative R² is mathematically valid:**
```
R² = 1 - (SS_res / SS_tot)
```
Where:
- SS_res = sum((y_true - y_pred)²)
- SS_tot = sum((y_true - mean(y_true))²)

If SS_res > SS_tot, then **R² < 0** → Model is worse than mean!

**Why R² ≠ correlation²:**
- R² measures explained variance (goodness of fit)
- correlation² measures linear relationship strength
- They differ when predictions are biased or non-linear

**This indicates real model problems, not metric bugs:**
1. SpO2_max reconstruction through cycle is **broken**
2. Some features have systematic bias in roundtrip
3. Cycle translation is lossy (expected for VAE)

---

## 2. KS Statistics Analysis ✓ **[FIXED]**

### Results Look Reasonable

**eICU→MIMIC:**
- Mean KS: 0.18 (acceptable)
- 17/24 features < 0.2 (good) - **71%**
- 20/24 features < 0.3 (acceptable threshold) - **83%**

**Worst performers:**
- SpO2_max: KS=0.52 (poor - distribution mismatch)
- Creat_std: KS=0.43
- Na_std: KS=0.38

### P-value Interpretation ✓ **[FIXED]**

**All 24/24 features have p < 0.05** - This is **expected and uninformative** with large sample sizes (N=5731 per domain).

**Original bug:** Evaluation used `(KS < 0.3) & (p > 0.05)`, marking **0/24 features as "good"** because p-value threshold was impossible to meet.

**Fix applied:** Removed p-value from criteria, now use KS-only with three tiers:
- Excellent: KS < 0.1
- Good: KS < 0.2  
- Acceptable: KS < 0.3

See `KS_CRITERIA_FIX.md` for full details.

---

## 3. Scaling Invariants ✓

### Original Data
**Both domains:** ✓ No violations of min ≤ mean ≤ max

All 6 features (HR, RR, Temp, SpO2, Na, Creat) satisfy the constraint in the preprocessed data.

### After Translation
Could not test due to import error, but the original data is clean.

---

## Recommended Actions

### **✅ COMPLETED: Feature Alignment Fixed**

Feature ordering bug was fixed in `evaluate.py` using `_get_feature_columns()`. Evaluation has been re-run with correct alignment.

### **✅ COMPLETED: KS Criteria Fixed**

Removed p-value dependency, now using KS magnitude only with three tiers (excellent/good/acceptable).

### **✅ COMPLETED: MIMIC-only Split Fixed**

Added `_split_mimic_for_cycle()` to properly split MIMIC test set into pseudo-domains.

### **🔍 Model Investigation Needed**

The R² metrics reveal **real model problems** (not bugs):

1. **SpO2_max is completely broken** (R² = -2.23):
   - Investigate why cycle reconstruction fails
   - Check if SpO2_max has special characteristics (range, distribution)
   - Consider feature-specific handling or exclusion

2. **RR_min has negative R²** (barely):
   - Review respiratory rate preprocessing
   - Check for outliers or compression issues

3. **Several features show systematic bias**:
   - WBC_max, Na_max, Creat features
   - Large |R² - corr²| suggests non-linear distortion
   - Consider adjusting loss weights or architecture

### **📊 Report Interpretation**

Current metrics show:
- **eICU**: 12/24 features with R² > 0.5 (50%)
- **MIMIC**: 13/24 features with R² > 0.5 (54%)
- **2 features with R² < 0** (SpO2_max, RR_min)

These numbers are **now reliable** and reflect actual model performance. The low R² for some features indicates the cycle-VAE has trouble with those specific features.

---

## Next Steps

1. ✅ Feature ordering bug fixed in `evaluate.py`
2. ✅ MIMIC-only split bug fixed in `evaluate.py`
3. ✅ KS criteria bug fixed (removed p-value dependency)
4. ✅ Re-run mimic-only comprehensive evaluation completed
5. ✅ Verified R² calculation is correct (not a bug)
6. ⏩ **Investigate model-level issues**:
   - Why SpO2_max fails completely (R² = -2.23)
   - Why some features have systematic bias
   - Consider architecture/loss adjustments

---

## Conclusion

**Initial suspicion confirmed:** There were bugs, but not in R² calculation!

**Bugs fixed:**
1. ✅ Feature ordering (caused misalignment)
2. ✅ MIMIC-only split (used same samples for both domains)
3. ✅ KS criteria (p-value dependency with large N)

**Not a bug (but revealing):**
- R² vs correlation² discrepancy is **mathematically correct**
- Negative R² means model is worse than mean (valid!)
- Large discrepancy indicates **real model problems**:
  - SpO2_max completely fails cycle reconstruction
  - Several features have systematic bias
  - VAE cycle is lossy (expected but quantified)

**Impact:**
- Previous evaluation results were **invalid** (wrong alignment + split)
- New results are **reliable** and show actual model performance
- Metrics correctly identify problematic features (SpO2_max, RR_min)

