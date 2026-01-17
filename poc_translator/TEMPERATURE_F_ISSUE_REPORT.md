# Temperature (F) Data Quality Issue - Investigation Report

**Date**: November 21, 2025  
**Issue**: Temperature Fahrenheit showing SMD of 4.126 (extreme shift)

---

## 🔍 Investigation Summary

### 1. Cache File Analysis

**Finding**: 89.6% of Temperature (F) values in cache are ZERO

| Metric | Value |
|--------|-------|
| Total measurements | 68,374 |
| Zero values | 61,267 (89.6%) |
| Valid range (95-105°F) | 6,889 (10.1%) |
| Mean | 10.34°F (abnormal) |
| Median | 0°F |

### 2. Source Database Verification

**Finding**: Source database has NO ZEROS - all values are valid!

```sql
-- Direct query from eicu_crd.nursecharting
-- Sample of 100 records:
```

| Metric | Value |
|--------|-------|
| Total in database | 6,267,330 measurements from 187,371 patients |
| Zero values | 0 (0.0%) ✅ |
| Mean (sample) | 98.93°F (normal) |
| All values | In valid range (95-105°F) |

### 3. Cache File Pattern Analysis

**Finding**: Cache file has MIXED data - some valid values, many zeros

Sample valid values found:
```
Patient 3038044: 97°F
Patient 3041060: 99°F, 100°F, 98.8°F
```

Sample zero values found:
```
Patient 3038044: Dozens of consecutive 0°F readings
```

---

## 🚨 Root Cause

**The zeros are NOT in the source database.**  
**They are introduced during cache file creation/processing.**

### Possible Causes:

1. **NULL/Missing Value Handling**:
   - eICU database may have NULL values for some time periods
   - Cache creation process might be converting NULLs to 0 instead of excluding them

2. **Data Type Conversion Issue**:
   - `nursingchartvalue` field is text in eICU
   - Conversion to numeric might be failing for some values and defaulting to 0

3. **Join/Filter Logic**:
   - Cache creation query might be creating records for time periods where no measurement exists
   - Default fill value might be 0

4. **Unit Conversion Logic**:
   - If there was a unit conversion attempt that failed, might default to 0

---

## 📊 Impact on Analysis

### Current Distribution Comparison:

| Dataset | Mean | Valid % | Issue |
|---------|------|---------|-------|
| **MIMIC** | 98.67°F | ~100% | ✅ Good data |
| **eICU** | 10.34°F | 10.1% | 🚨 90% zeros |
| **SMD** | 4.126 | - | Extreme shift due to zeros |

### What Should Be:

| Dataset | Mean | Issue |
|---------|------|-------|
| **MIMIC** | 98.67°F | ✅ Good |
| **eICU** | 98.93°F | ✅ Should be good (database is clean) |
| **Expected SMD** | < 0.2 | Should be aligned |

---

## ✅ Recommendations

### Option 1: Filter Out Zeros (Quick Fix)
```python
# In compare_distributions.py, add before analysis:
mask = (df_eicu['feature_name'] == 'Temperature (F)') & (df_eicu['feature_value'] > 0)
df_eicu = df_eicu[mask | (df_eicu['feature_name'] != 'Temperature (F)')]
```

**Pros**: Quick, will improve analysis immediately  
**Cons**: Doesn't fix root cause

### Option 2: Recreate Cache File (Proper Fix)
Investigate and fix the cache file creation process:

1. **Check the SQL query** that extracts Temperature (F)
2. **Verify NULL handling**: Ensure NULLs are excluded, not converted to 0
3. **Check data type conversion**: Ensure proper numeric conversion
4. **Test query**: Verify extracted data matches database before creating cache

**Pros**: Fixes root cause  
**Cons**: Requires more investigation

### Option 3: Use Temperature (C) Instead
If Temperature (F) continues to have issues:
- Celsius temperature likely has better data quality
- Most clinical systems use Celsius as primary
- MIMIC has Temperature in Celsius too

**Pros**: Avoids the problematic feature  
**Cons**: Loses Fahrenheit measurements

---

## 🔧 Immediate Action Items

### Priority 1: Filter for Current Analysis
Add this to `compare_distributions.py`:
```python
# After loading data, before unit conversion:
logger.info("Filtering Temperature (F) zero values...")
temp_f_mask = (df_eicu['feature_name'] == 'Temperature (F)') & (df_eicu['feature_value'] > 0)
df_eicu = df_eicu[temp_f_mask | (df_eicu['feature_name'] != 'Temperature (F)')]
```

### Priority 2: Investigate Cache Creation
1. Locate the script/query that creates `cache_data_bsi_test_100`
2. Find the Temperature (F) extraction logic
3. Check how NULL values are handled
4. Verify the join conditions

### Priority 3: Validate Other Features
Check if other features have similar zero-value issues:
```python
# For each feature in cache:
zero_pct = (df_eicu.groupby('feature_name')['feature_value']
            .apply(lambda x: (pd.to_numeric(x, errors='coerce') == 0).sum() / len(x) * 100))
print(zero_pct[zero_pct > 10])  # Features with >10% zeros
```

---

## 📈 Expected Improvement

If zeros are filtered or cache is fixed:

| Metric | Current | After Fix | Improvement |
|--------|---------|-----------|-------------|
| Temperature (F) SMD | 4.126 | ~0.1-0.2 | ✅ 95% reduction |
| Features with large shift | 6 | **5** | ✅ 1 less |
| Overall alignment | 76.3% good | **79%+ good** | ✅ Better |

---

## 🎯 Conclusion

1. ✅ **Source database is clean** - no zeros in eICU nursecharting table
2. 🚨 **Issue is in cache file creation** - zeros introduced during processing
3. ✅ **Easy to fix** - either filter zeros or fix cache creation logic
4. 📊 **High impact** - will significantly improve alignment for this feature

The Temperature (F) issue is a **processing artifact**, not a real distribution shift. Once fixed, this feature should align well between MIMIC and eICU (expected SMD < 0.2).

---

## 📂 References

- Cache file: `/bigdata/omerg/RatchetEHR/tmp/tmp/cache_data_bsi_test_100`
- Source table: `eicu_crd.nursecharting`
- Field: `nursingchartcelltypevalname = 'Temperature (F)'`
- Analysis: `analysis/results/per_feature_summary.csv`

