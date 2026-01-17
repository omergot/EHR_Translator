# eICU Query Issues - Complete Analysis

## 🚨 Critical Issues Found

### Issue 1: ❌ **resp_res** - Same Regex Bug (CRITICAL)

```sql
CASE WHEN respchartvalue~E'^\\d+$' THEN respchartvalue::float ELSE 0 end as feature_value,
```

**Problem**: Only matches integers, rejects decimals
**Affected Features**:
- **Inspired O2 Fraction (FiO2)** - Often recorded as 0.21, 0.4, 0.5, etc. ← Would ALL become 0!
- **Mean Airway Pressure** - Can have decimals (12.5 cmH2O)
- **Tidal Volume (observed)** - Can have decimals (450.5 mL)

**Evidence from Analysis**:
- Tidal Volume: SMD = 0.555 (large shift) ← Likely affected
- Mean Airway Pressure: SMD = 0.200 (mild shift) ← Might be affected

**Fix**:
```sql
CASE WHEN respchartvalue~E'^[0-9]+(\\.[0-9]+)?$' THEN respchartvalue::float ELSE NULL end as feature_value,
```

---

### Issue 2: ❌ **labothername_res** - Same Regex Bug

```sql
CASE WHEN labotherresult~E'^\\d+$' THEN labotherresult::float ELSE 0 end as feature_value,
```

**Problem**: Only matches integers
**Affected**: Custom lab values that may have decimals

**Fix**:
```sql
CASE WHEN labotherresult~E'^[0-9]+(\\.[0-9]+)?$' THEN labotherresult::float ELSE NULL end as feature_value,
```

---

### Issue 3: ⚠️ **Using 0 Instead of NULL** (All CTEs)

**Problem**: Invalid values are converted to 0 instead of being excluded

```sql
ELSE 0 end as feature_value,  -- ← Creates fake data points!
```

**Impact**:
- Adds physiologically impossible 0 values
- Pollutes the dataset with fake measurements
- Causes distribution shifts

**Fix**: Use `ELSE NULL` in ALL CTEs:
```sql
ELSE NULL end as feature_value,
```

Then add at the end:
```sql
WHERE feature_value IS NOT NULL
```

---

### Issue 4: ⚠️ **lab_res** - No Numeric Validation

```sql
labresult as feature_value,  -- ← Taken as-is, no validation!
```

**Problem**: 
- `labresult` is TEXT field in eICU
- Can contain non-numeric values ("pending", "cancelled", ">1000", etc.)
- No validation or conversion

**Evidence**: The analysis script does `pd.to_numeric(..., errors='coerce')` to handle this, but database should filter

**Fix**:
```sql
CASE WHEN labresult~E'^[0-9]+(\\.[0-9]+)?$' THEN labresult::float ELSE NULL end as feature_value,
```

---

## ✅ What's Working

### 1. **person_with_date** - Correct Regex
```sql
age ~ '^[0-9\.]+$'  -- ✓ Handles decimals correctly
```

### 2. **Join Logic** - Correct
- Proper joins on patient ID
- Correct filtering for blood culture time (before culture date)

### 3. **Time Offset Calculation** - Correct
```sql
(date '2000-1-1' + (offset * interval '1 minutes'))
```

### 4. **Unit Preservation** - Partially Correct
- `lab_res` keeps `labMeasureNameSystem` ✓
- Others use empty string (acceptable since units are inferred)

---

## 🔍 Analysis Results Correlation

### Features Likely Affected by Regex Bug:

| Feature | Current SMD | Likely Issue | Expected After Fix |
|---------|-------------|--------------|-------------------|
| **Temperature (F)** | 4.126 | ✅ Fixed in nurse_res | Will improve |
| **Tidal Volume** | 0.555 | 🚨 resp_res bug | Should improve significantly |
| **O2 saturation** | 0.543 | Possibly nurse_res (was 0 issue) | May improve |
| **Mean Airway Pressure** | Mild (analysis) | 🚨 resp_res bug | May improve |

### FiO2 Investigation Needed:

FiO2 (Inspired O2 Fraction) is in `respiratorycharting` - needs special attention:
- Often recorded as decimals: 0.21 (21%), 0.4 (40%), etc.
- Current regex would convert ALL to 0
- This could explain any distribution issues with this feature

---

## 📋 Recommended Fixes - Complete SQL

### Fixed nurse_res:
```sql
nurse_res as (
    SELECT
        b.example_id,
        b.person_id,
        nursingchartcelltypevalname as feature_name,
        CASE WHEN nursingchartvalue~E'^[0-9]+(\\.[0-9]+)?$' 
             THEN nursingchartvalue::float 
             ELSE NULL 
        end as feature_value,
        (date '2000-1-1' + (nursingChartOffset * interval '1 minutes')) as feature_start_date,
        '' as unit
    FROM eicu_crd.nursecharting as a
    JOIN {cohort_table} as b ON (example_id = patientunitstayid)
    JOIN blood_cultures bc ON (bc.patientunitstayid = b.example_id)
    WHERE nursingchartcelltypevalname IS NOT NULL
        AND nursingChartOffset >= 0
        AND (date '2000-1-1' + (nursingChartOffset * interval '1 minutes')) < bc.culture_date
        AND nursingchartvalue~E'^[0-9]+(\\.[0-9]+)?$'  -- Pre-filter invalid values
),
```

### Fixed resp_res:
```sql
resp_res as (
    SELECT
        b.example_id,
        b.person_id,
        respchartvaluelabel as feature_name,
        CASE WHEN respchartvalue~E'^[0-9]+(\\.[0-9]+)?$' 
             THEN respchartvalue::float 
             ELSE NULL 
        end as feature_value,
        (date '2000-1-1' + (respChartOffset * interval '1 minutes')) as feature_start_date,
        '' as unit
    FROM eicu_crd.respiratorycharting as a
    JOIN {cohort_table} as b ON (example_id = patientunitstayid)
    JOIN blood_cultures bc ON (bc.patientunitstayid = b.example_id)
    WHERE respchartvaluelabel IS NOT NULL
        AND respChartOffset >= 0
        AND (date '2000-1-1' + (respChartOffset * interval '1 minutes')) < bc.culture_date
        AND respchartvalue~E'^[0-9]+(\\.[0-9]+)?$'  -- Pre-filter invalid values
),
```

### Fixed labothername_res:
```sql
labothername_res as (
    SELECT
        b.example_id,
        b.person_id,
        labothername as feature_name,
        CASE WHEN labotherresult~E'^[0-9]+(\\.[0-9]+)?$' 
             THEN labotherresult::float 
             ELSE NULL 
        end as feature_value,
        (date '2000-1-1' + (labotheroffset * interval '1 minutes')) as feature_start_date,
        '' as unit
    FROM eicu_crd.customlab as a
    JOIN {cohort_table} as b ON (example_id = patientunitstayid)
    JOIN blood_cultures bc ON (bc.patientunitstayid = b.example_id)
    WHERE labothername IS NOT NULL
        AND labotheroffset >= 0
        AND (date '2000-1-1' + (labotheroffset * interval '1 minutes')) < bc.culture_date
        AND labotherresult~E'^[0-9]+(\\.[0-9]+)?$'  -- Pre-filter invalid values
),
```

### Fixed lab_res:
```sql
lab_res as (
    SELECT
        b.example_id,
        b.person_id,
        labname as feature_name,
        CASE WHEN labresult~E'^[0-9]+(\\.[0-9]+)?$' 
             THEN labresult::float 
             ELSE NULL 
        end as feature_value,
        (date '2000-1-1' + (labresultoffset * interval '1 minutes')) as feature_start_date,
        labMeasureNameSystem as unit
    FROM eicu_crd.lab as a
    JOIN {cohort_table} as b ON (example_id = patientunitstayid)
    JOIN blood_cultures bc ON (bc.patientunitstayid = b.example_id)
    WHERE labname IS NOT NULL
        AND labresultoffset >= 0
        AND (date '2000-1-1' + (labresultoffset * interval '1 minutes')) < bc.culture_date
        AND labresult~E'^[0-9]+(\\.[0-9]+)?$'  -- Pre-filter invalid values
),
```

### Final SELECT - Add NULL Filter:
```sql
select 
    example_id,
    person_id,
    feature_name,
    feature_value::TEXT as feature_value,
    feature_start_date
FROM converted
WHERE feature_value IS NOT NULL  -- ← Add this!
```

---

## 📊 Expected Impact After Fixes

### Features That Should Improve:

1. **Temperature (F)**: 4.126 → <0.2 (already fixed in nurse_res)
2. **Tidal Volume**: 0.555 → <0.3 (resp_res fix)
3. **Inspired O2 Fraction**: Check current SMD, likely affected
4. **Mean Airway Pressure**: May improve

### Overall Improvement:

- **Current**: 6 features with large shift (15.8%)
- **After fixes**: Estimated **3-4 features** with large shift (~10%)
- **Alignment**: From 76.3% → **~82-85%** features well-aligned

---

## ✅ Action Items

1. **Immediate**: Update all 4 CTEs with decimal-matching regex
2. **Immediate**: Change `ELSE 0` to `ELSE NULL` everywhere
3. **Immediate**: Add `WHERE feature_value IS NOT NULL` at end
4. **Testing**: Regenerate cache file with fixed query
5. **Validation**: Re-run distribution analysis
6. **Verification**: Check specific features (FiO2, Tidal Volume, Temperature F)

---

## 🎯 Summary

**Total Issues Found**: 4 critical regex bugs  
**Affected CTEs**: nurse_res (fixed), resp_res, labothername_res, lab_res  
**Estimated Impact**: 3-4 features with large distribution shifts  
**Root Cause**: Using `^\\d+$` (integers only) instead of `^[0-9]+(\\.[0-9]+)?$` (decimals)  
**Additional Issue**: Using `ELSE 0` creates fake zero values instead of excluding invalid data

All issues are fixable with simple regex updates and NULL handling!

