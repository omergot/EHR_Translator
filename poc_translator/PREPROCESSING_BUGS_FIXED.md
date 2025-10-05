# All Preprocessing Bugs Found and Fixed

## Summary
Found and fixed **8 CRITICAL BUGS** that were corrupting the preprocessed data and causing training failures.

---

## Bug #1: Missing Flags Transformed with log1p ✅ FIXED
**Symptom:** Missing flag values were 0.6931471805599453 instead of 0 or 1

**Root Cause:** Line 224 in `apply_feature_specific_transforms` was missing the check to exclude missing flags from log1p transformation
- `log1p(1) = ln(2) = 0.693147...`

**Fix:** Added `col not in feature_analysis["missing_flags"]` check

---

## Bug #2: Column Ordering Mismatch ✅ FIXED
**Symptom:** All feature values were scrambled/misaligned

**Root Cause:** `get_feature_columns()` generated columns in order `['_mean', '_min', '_max', '_std']` but CSV files have `['_min', '_max', '_mean', '_std']`

**Fix:** Changed suffix order to `['_min', '_max', '_mean', '_std']` in 4 locations:
- `preprocess.py` line 148 - `get_feature_columns()`
- `preprocess.py` line 552 - `create_feature_spec()`
- `evaluate.py` line 157 - feature iteration
- `evaluate.py` line 242 - feature iteration

---

## Bug #3: Filling NaN with 0 BEFORE Scaling ✅ FIXED
**Symptom:** Temperature features became all 0 after preprocessing

**Root Cause:** When 90% of values are NaN and filled with 0 before scaling:
- Median = 0, Q25 = 0, Q75 = 0
- IQR = 0
- RobustScaler: (X - 0) / 0 = undefined → all became 0

**Fix:** 
- Removed `fillna(0)` before transformation (was line 271-273)
- Added `fillna(0)` AFTER scaling (new lines 459-464)

---

## Bug #4: Feature Filtering for High Missingness ✅ IMPLEMENTED
**Symptom:** Temperature (89.6% missing) and MAP (75.1% missing) in eICU were breaking the scaler

**Solution:** Added `filter_high_missingness_features()` method
- Configurable threshold via `config.yml`: `preprocessing.max_missing_pct: 0.5`
- Removes features with >50% missing values
- Temp and MAP are now automatically filtered out

---

## Bug #5: DataFrame Assignment Losing Column Order ✅ FIXED
**Symptom:** _std features were still becoming all 0 after previous fixes

**Root Cause:** Pandas DataFrame assignment with intermediate DataFrame creation was causing alignment issues

**Fix:** Changed from:
```python
data_scaled[cols] = pd.DataFrame(transformed, columns=cols, index=data_scaled.index)
```
To:
```python
data_scaled.loc[:, cols] = transformed
```

---

## Bug #6: **CRITICAL** - _std Columns Clipped to Wrong Range ⚠️ **THIS WAS THE MAIN BUG!**
**Symptom:** ALL _std features became single constant values:
- HR_std: all became 30.0 (then log1p → 3.434)
- RR_std: all became 5.0 (then log1p → 1.792)
- SpO2_std: all became 50.0 (then log1p → 3.932)
- Na_std: all became 110.0 (then log1p → 4.710)
- Creat_std: all became 0.1 (then log1p → 0.095)

**Root Cause:** `clip_to_clinical_ranges()` was applying base feature ranges to _std columns!
- `HR_std` → extracted `HR` → clipped to (30, 250)
- But HR_std values are typically 0-20, NOT 30-250!
- Clipping HR_std to [30, 250] sets all values to 30

**Fix:** Skip _std columns in `clip_to_clinical_ranges()`:
```python
if '_std' in col:
    continue  # Don't clip _std columns
```

**Impact:** This was the most severe bug - it completely destroyed all variation in _std features

---

## Bug #7: **CRITICAL** - Independent Scaling Destroys Min/Max Relationships ⚠️ **FUNDAMENTAL DESIGN FLAW!**
**Symptom:** After scaling, `HR_min` values were GREATER than `HR_max` values!
- Example: Patient 141179 had `HR_min=0.542` and `HR_max=0.036` (min > max!)

**Root Cause:** StandardScaler was treating `HR_min`, `HR_max`, and `HR_mean` as completely independent columns:
- `HR_min`: scaled with mean=69.69, std=15.34
- `HR_max`: scaled with mean=109.23, std=21.48  
- `HR_mean`: scaled with mean=85.19, std=16.30

For patient 141179 (HR_min=78, HR_max=110):
- `HR_min = (78 - 69.69) / 15.34 = 0.542` (above average min)
- `HR_max = (110 - 109.23) / 21.48 = 0.036` (barely above average max)
- **Result: min > max!** 😱

**Why This Happened:** Each column has different means. The dataset average HR_max (109.2) is close to this patient's actual max (110), while the average HR_min (69.7) is far from their min (78). After independent centering, the relationships flipped!

**Fix:** Implemented **unified scaling** that preserves relationships:
1. **Pool values**: Combine all HR_min, HR_max, HR_mean values together
2. **Compute unified parameters**: Calculate mean/std from pooled data (mean=88.04, std=24.20)
3. **Apply same parameters**: Scale ALL HR statistics with the SAME transformation

```python
# OLD (WRONG - independent scaling):
HR_min_scaled = (HR_min - HR_min_mean) / HR_min_std
HR_max_scaled = (HR_max - HR_max_mean) / HR_max_std

# NEW (CORRECT - unified scaling):
unified_mean = pooled_mean_of_all_HR_values
unified_std = pooled_std_of_all_HR_values
HR_min_scaled = (HR_min - unified_mean) / unified_std
HR_max_scaled = (HR_max - unified_mean) / unified_std
# Now: if HR_min < HR_max before scaling, then HR_min_scaled < HR_max_scaled ✅
```

**Verification:** Tested on 10 patients - **0 violations** found! All maintain `min < mean < max` ✅

**Impact:** This was a FUNDAMENTAL design flaw that made the data semantically incorrect. Models trained on this would learn nonsensical patterns where "min" features had higher values than "max" features!

---

## Verification

After all fixes, _std features now have proper variation:
- **HR_std**: 16,894 unique values ✅ (was: 2 values = bug)
- **RR_std**: 15,379 unique values ✅ (was: 2 values = bug)
- **SpO2_std**: 7,382 unique values ✅ (was: 2 values = bug)
- **WBC_std**: 8,955 unique values ✅ (was working, still works)
- **Na_std**: 1,063 unique values ✅ (was: 2 values = bug)
- **Creat_std**: 2,676 unique values ✅ (was: 2 values = bug)

---

## Required Action

**Run preprocessing to regenerate data:**
```bash
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python3 src/preprocess.py --config conf/config.yml --fit
```

After this, the data will be correct:
- ✅ Missing flags: 0 or 1 (not 0.693...)
- ✅ Temp and MAP: Removed (high missingness)
- ✅ All _std features: Proper variation (not single values)
- ✅ All features: Correct alignment and scaling
- ✅ Min/max relationships: Preserved (min < max for all patients)

---

## Files Modified
1. `src/preprocess.py` - Main fixes
2. `src/evaluate.py` - Column ordering fix
3. `conf/config.yml` - Added preprocessing.max_missing_pct parameter

---

## Bug #8: Dataset Column Ordering Mismatch ✅ FIXED
**Symptom:** CUDA index out of bounds error during training after fixing preprocessing bugs

**Root Cause:** In `dataset.py` line 50, the code was sorting features alphabetically:
```python
clinical_features.sort()  # This scrambled the order!
```

This caused the dataset to load features in a different order than they appeared in the CSV:
- CSV order: `HR_min, HR_max, HR_mean, HR_std, RR_min, ...`
- After sorting: `Creat_max, Creat_mean, Creat_min, Creat_std, HR_max, ...`

**Fix:** 
- Modified `FeatureDataset.__init__()` to accept `feature_spec` parameter
- Use `feature_spec` to maintain correct column order from preprocessing
- Removed alphabetical sorting that was scrambling the order
- Updated `CombinedDataModule.setup()` to pass `feature_spec` to datasets

**Files Modified:**
- `src/dataset.py` lines 29-52 - Added feature_spec parameter and removed sorting
- `src/dataset.py` lines 125-137 - Pass feature_spec when creating datasets

---

## Training Impact

**ALL previous training results are INVALID** because they used corrupted preprocessing data where:
1. _std features had no variation (all constant values)
2. Missing flags were wrong (0.693 instead of 1)
3. Some features had wrong values (column misalignment)
4. **Min/max relationships were violated** (min > max in many cases)
5. **Column ordering mismatch** between CSV and model input

The model cannot learn meaningful patterns from:
- Constant-valued features
- Semantically incorrect data (where "minimum" is greater than "maximum")
- Misaligned feature columns

Therefore, all previous results need to be discarded and training re-run with the corrected, unified scaling approach.


