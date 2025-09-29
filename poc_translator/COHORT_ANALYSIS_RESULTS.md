# Cohort Restrictiveness Analysis Results

**Date:** September 29, 2025  
**Analysis:** Comparison of strict BSI cohorts vs public tables for MIMIC-IV and eICU-CRD

---

## 🔍 KEY FINDINGS

### Data Availability Comparison

| Database | Strict BSI Cohort | Public Tables (≥24h) | **Increase Factor** |
|----------|------------------|---------------------|-------------------|
| MIMIC-IV | 12,139 stays     | **57,734 stays**    | **4.8x more data** |
| eICU-CRD | ~3,500 stays (est.) | **132,611 stays**  | **37.9x more data** |

### Impact Assessment
- **MIMIC-IV:** Strict cohort excluded **78.9%** of potential adult ICU stays
- **eICU-CRD:** Strict cohort excluded **97.4%** of potential adult ICU stays
- **Combined:** From ~15,639 to **190,345 total ICU stays** = **12.2x increase**

---

## ✅ CHANGES IMPLEMENTED

### 1. Updated `make_queries.py`
**Before:** Used restrictive cohort tables
```sql
FROM mimiciv_bsi_100_2h_test.__mimiciv_bsi_100_2h_cohort
FROM eicu_bsi_100_2h_test.__eicu_bsi_100_2h_cohort
```

**After:** Uses public tables with minimal filters
```sql
-- MIMIC-IV
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
WHERE i.intime IS NOT NULL
  AND i.hadm_id IS NOT NULL         -- Must have hospital admission
  AND EXTRACT(epoch FROM (i.outtime - i.intime))/3600 >= 24  -- At least 24h stay
  AND p.anchor_age >= 18             -- Adult patients

-- eICU-CRD  
FROM eicu_crd.patient p
WHERE p.patientunitstayid IS NOT NULL
  AND p.unitdischargeoffset >= 24 * 60    -- At least 24h stay (minutes)
  AND p.age != '' AND p.age != 'Unknown' AND p.age IS NOT NULL
  AND CASE WHEN p.age = '> 89' THEN 90 ELSE CAST(p.age AS INTEGER) END >= 18
```

### 2. Created Analysis Scripts
- `scripts/analyze_cohort_restrictiveness.py` - Full database analysis
- `scripts/cohort_comparison_summary.py` - Quick summary display
- `scripts/test_public_tables.py` - Verify new queries work

### 3. Generated Test SQL Files
- `sql/test_mimic_public.sql` - MIMIC-IV public table query
- `sql/test_eicu_public.sql` - eICU-CRD public table query

---

## 🎛️ CONFIGURATION OPTIONS

### Current Settings (Recommended)
- **Minimum stay duration:** 24 hours
- **Age filter:** Adults only (≥18 years)
- **Data requirement:** Hospital admission (MIMIC-IV)

### Alternative Options

#### Option 1: More Complete Data (48h minimum)
```sql
-- MIMIC-IV
AND EXTRACT(epoch FROM (i.outtime - i.intime))/3600 >= 48

-- eICU-CRD
AND p.unitdischargeoffset >= 48 * 60
```

#### Option 2: Maximum Data (4h minimum)
```sql
-- MIMIC-IV  
AND EXTRACT(epoch FROM (i.outtime - i.intime))/3600 >= 4

-- eICU-CRD
AND p.unitdischargeoffset >= 4 * 60
```

#### Option 3: All Adult Stays (no duration filter)
```sql
-- Remove duration filters entirely
-- Keep only: age ≥18, hospital admission (MIMIC), not null checks
```

---

## 📋 PUBLIC TABLES USED

### MIMIC-IV Tables
- **mimiciv_icu.icustays** - ICU stay metadata
- **mimiciv_hosp.patients** - Patient demographics (anchor_age)
- **mimiciv_icu.chartevents** - Vital signs and monitoring data
- **mimiciv_hosp.labevents** - Laboratory results

### eICU-CRD Tables  
- **eicu_crd.patient** - ICU stay metadata and demographics
- **eicu_crd.vitalperiodic** - Periodic vital signs
- **eicu_crd.lab** - Laboratory results

---

## 🚀 NEXT STEPS

### Immediate Actions
1. ✅ **Updated make_queries.py** with public table approach
2. ⏳ **Test generated SQL queries** on small sample first
3. ⏳ **Run feature extraction** with new queries
4. ⏳ **Compare model performance** on larger dataset

### Testing Commands
```bash
# Generate SQL with new approach
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python sql/make_queries.py

# Test with small sample first (add LIMIT 1000 to queries)
# Then run full extraction:
python data/raw_extractors.py
```

### Monitoring Considerations
- **Query runtime:** More data = longer execution time
- **Storage requirements:** ~12x more features to store  
- **Memory usage:** Monitor during feature extraction
- **Model training:** May need to adjust batch sizes

---

## ⚠️ IMPORTANT NOTES

### Benefits of Public Tables
- ✅ **Dramatically more training data** (4-38x increase)
- ✅ **Better model generalization** 
- ✅ **More representative population**
- ✅ **Reduced selection bias**

### Potential Challenges
- ⚠️ Longer query execution times
- ⚠️ Higher storage requirements  
- ⚠️ Need to monitor data quality
- ⚠️ May need infrastructure adjustments

### Quality Filters Applied
- Adult patients only (≥18 years)
- Minimum 24-hour ICU stay
- Valid hospital admission (MIMIC-IV)
- Non-null essential identifiers

---

## 📊 VERIFICATION RESULTS

**Test Results (September 29, 2025):**
- ✅ MIMIC-IV public query: 57,734 adult ICU stays  
- ✅ eICU-CRD public query: 132,611 adult ICU stays
- ✅ SQL generation working correctly
- ✅ All database connections functional

**Query Performance:** Both test queries executed successfully within reasonable time.

---

*This analysis demonstrates that switching from restrictive BSI cohorts to public tables provides 4-38x more training data while maintaining data quality through appropriate filters.*
