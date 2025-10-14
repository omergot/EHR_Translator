# Evaluation Pipeline Fixes Summary

## Issues Fixed

### 1. ✅ Feature Ordering Bug (CRITICAL)
**Problem:** Feature columns not following `feature_spec` order, causing misalignment between original and translated data.

**Fix:** Added `_get_feature_columns()` method in `evaluate.py` to ensure consistent feature ordering from `feature_spec`.

**Impact:** This could have caused the massive R² vs Pearson discrepancy (e.g., R²=-2.22 for SpO2_max).

**Files modified:**
- `src/evaluate.py`: Added `_get_feature_columns()` and applied throughout

---

### 2. ✅ MIMIC-only Split Bug (CRITICAL)
**Problem:** In mimic-only mode, evaluation used the same MIMIC samples for both "eICU" and "MIMIC" domains, making metrics artificially identical.

**Fix:** Added `_split_mimic_for_cycle()` method to split MIMIC test data into two pseudo-domains (50/50 with shuffled labels), mirroring the training behavior.

**Impact:** Now mimic-only evaluation properly tests cycle-consistency on separate subsets.

**Files modified:**
- `src/evaluate.py`: Added `_split_mimic_for_cycle()` and applied to all evaluation methods

---

### 3. ✅ KS Test P-value Dependency (MAJOR)
**Problem:** Distribution matching criteria used `(KS < 0.3) & (p > 0.05)`, resulting in **0/24 "good" features** because all p-values < 0.05 with large N=5731.

**Why this was wrong:**
- P-values test statistical significance, not effect size
- With large N, even tiny KS differences → p < 0.05
- P-value tells you "distributions differ" but not "how much"

**Fix:** Removed p-value from quality criteria, now use KS magnitude only with three tiers:
- **Excellent**: KS < 0.1 (distributions nearly identical)
- **Good**: KS < 0.2 (distributions very similar)
- **Acceptable**: KS < 0.3 (noticeable but acceptable difference)

**Impact:** Now shows realistic distribution matching: 17/24 (71%) have KS < 0.2 instead of misleading 0/24.

**Files modified:**
- `src/comprehensive_evaluator.py`: Lines 434-441, 1066-1092
- `src/evaluate.py`: Lines 840-866, 1077-1103, 1195-1198

**See:** `KS_CRITERIA_FIX.md` for detailed explanation

---

## Remaining Issues to Investigate

### ⚠️ R² vs Pearson Correlation Discrepancy (CRITICAL)
**Status:** Partially fixed by feature ordering, needs verification

**Evidence of bug:**
- SpO2_max: R²=-2.22, correlation²=0.00 (impossible for simple regression!)
- Creat_max: R²=0.91 vs correlation²=0.08 (massive mismatch)
- Mean |R² - corr²|: 0.22 (should be ~0)

**Expected:** R² ≈ correlation² for simple linear regression (roundtrip)

**Next steps:**
1. Re-run comprehensive evaluation with fixed feature ordering
2. Verify if R² now matches correlation²
3. If still broken, investigate `comprehensive_evaluator.py` lines 347-357

**See:** `DEBUG_FINDINGS.md` for full analysis

---

## Files Modified

### Core Fixes
- `src/evaluate.py` - Feature ordering, mimic-only split, KS reporting
- `src/comprehensive_evaluator.py` - KS criteria, reporting

### Documentation
- `DEBUG_FINDINGS.md` - Debug analysis results
- `KS_CRITERIA_FIX.md` - KS criteria fix explanation
- `EVALUATION_FIXES_SUMMARY.md` - This file

### Debug Tools
- `debug_evaluation_metrics.py` - Sanity check script (R², KS, scaling)

---

## Testing

### To verify all fixes work:

```bash
# Re-run mimic-only comprehensive evaluation
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python src/evaluate.py --config conf/config.yml \
  --model output/model.ckpt \
  --output-dir output \
  --comprehensive --mimic-only

# Run sanity checks
python debug_evaluation_metrics.py
```

### Expected outcomes:
1. ✅ Feature ordering: All arrays aligned
2. ✅ MIMIC-only: Separate pseudo-domains (not identical samples)
3. ✅ KS criteria: Meaningful percentages (not 0/24)
4. ⏳ R² check: |R² - corr²| < 0.05 for most features

---

## Impact on Previous Results

### ❌ Previous mimic-only evaluation is INVALID
- Used same samples for both domains
- Feature misalignment may have corrupted metrics
- KS criteria showed misleading 0/24

### ✅ Need to regenerate:
- All comprehensive evaluation reports
- All mimic-only evaluation reports
- Feature quality analysis

---

## Summary

**Fixed:**
- Feature ordering (critical alignment bug)
- MIMIC-only split (critical evaluation bug)
- KS criteria (major interpretation bug)

**To investigate:**
- R² calculation (potential critical bug)

**Next action:**
Re-run comprehensive evaluation and verify R² matches correlation².

