-- OPTIMIZED MIMIC-IV Feature Extraction Query
-- Fixes performance issues and expands data coverage
-- Uses window functions and CTEs for better efficiency

WITH cohort_patients AS (
  SELECT DISTINCT person_id, example_id
  FROM mimiciv_bsi_100_2h_test.__mimiciv_bsi_100_2h_cohort
),

icu_window AS (
  SELECT 
    i.stay_id as icustay_id, 
    i.subject_id as subject_id, 
    i.hadm_id as hadm_id, 
    i.intime,
    i.intime + INTERVAL '24 hours' as intime_plus_24h,
    -- Extended time window for more comprehensive data
    GREATEST(i.intime - INTERVAL '6 hours', a.admittime) as extended_start,
    i.intime + INTERVAL '48 hours' as extended_end
  FROM mimiciv_icu.icustays i
  JOIN cohort_patients c ON i.stay_id = c.example_id
  JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
  WHERE i.intime IS NOT NULL
),

-- Consolidated data extraction with all measurements
all_measurements AS (
  SELECT 
    i.icustay_id, i.subject_id, i.hadm_id,
    i.intime, i.intime_plus_24h,
    l.itemid, l.valuenum, l.charttime, 'lab' as source_type
  FROM icu_window i
  JOIN mimiciv_hosp.labevents l ON l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id
  WHERE l.valuenum IS NOT NULL 
    AND l.charttime >= i.extended_start
    AND l.charttime < i.extended_end
  
  UNION ALL
  
  SELECT 
    i.icustay_id, i.subject_id, i.hadm_id,
    i.intime, i.intime_plus_24h,
    c.itemid, c.valuenum, c.charttime, 'chart' as source_type
  FROM icu_window i
  JOIN mimiciv_icu.chartevents c ON c.stay_id = i.icustay_id
  WHERE c.valuenum IS NOT NULL
    AND c.charttime >= i.extended_start
    AND c.charttime < i.extended_end
),

-- Calculate last values efficiently using window functions
last_values AS (
  SELECT 
    icustay_id, itemid,
    FIRST_VALUE(valuenum) OVER (
      PARTITION BY icustay_id, itemid 
      ORDER BY charttime DESC 
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as last_value,
    ROW_NUMBER() OVER (PARTITION BY icustay_id, itemid ORDER BY charttime DESC) as rn
  FROM all_measurements
  WHERE charttime >= intime AND charttime < intime_plus_24h
)

SELECT
  i.icustay_id,
  i.subject_id,
  i.hadm_id,

  
  -- Feature 0: Alanine Aminotransferase (ALT)
  AVG(CASE WHEN m.itemid IN (50861, 50862) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_0_mean,
  MIN(CASE WHEN m.itemid IN (50861, 50862) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_0_min,
  MAX(CASE WHEN m.itemid IN (50861, 50862) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_0_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50861, 50862) AND lv.rn = 1 LIMIT 1) AS feature_0_last,
  COUNT(CASE WHEN m.itemid IN (50861, 50862) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_0_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50861, 50862) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_0_missing,

  
  -- Feature 1: Albumin
  AVG(CASE WHEN m.itemid IN (50862) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_1_mean,
  MIN(CASE WHEN m.itemid IN (50862) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_1_min,
  MAX(CASE WHEN m.itemid IN (50862) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_1_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50862) AND lv.rn = 1 LIMIT 1) AS feature_1_last,
  COUNT(CASE WHEN m.itemid IN (50862) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_1_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50862) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_1_missing,

  
  -- Feature 2: Alkaline Phosphatase
  AVG(CASE WHEN m.itemid IN (50863) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_2_mean,
  MIN(CASE WHEN m.itemid IN (50863) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_2_min,
  MAX(CASE WHEN m.itemid IN (50863) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_2_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50863) AND lv.rn = 1 LIMIT 1) AS feature_2_last,
  COUNT(CASE WHEN m.itemid IN (50863) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_2_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50863) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_2_missing,

  
  -- Feature 3: Aspartate Aminotransferase (AST)
  AVG(CASE WHEN m.itemid IN (50876) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_3_mean,
  MIN(CASE WHEN m.itemid IN (50876) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_3_min,
  MAX(CASE WHEN m.itemid IN (50876) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_3_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50876) AND lv.rn = 1 LIMIT 1) AS feature_3_last,
  COUNT(CASE WHEN m.itemid IN (50876) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_3_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50876) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_3_missing,

  
  -- Feature 4: Bilirubin
  AVG(CASE WHEN m.itemid IN (50885, 50884, 50883) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_4_mean,
  MIN(CASE WHEN m.itemid IN (50885, 50884, 50883) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_4_min,
  MAX(CASE WHEN m.itemid IN (50885, 50884, 50883) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_4_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50885, 50884, 50883) AND lv.rn = 1 LIMIT 1) AS feature_4_last,
  COUNT(CASE WHEN m.itemid IN (50885, 50884, 50883) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_4_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50885, 50884, 50883) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_4_missing,

  
  -- Feature 5: C-Reactive Protein
  AVG(CASE WHEN m.itemid IN (50867) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_5_mean,
  MIN(CASE WHEN m.itemid IN (50867) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_5_min,
  MAX(CASE WHEN m.itemid IN (50867) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_5_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50867) AND lv.rn = 1 LIMIT 1) AS feature_5_last,
  COUNT(CASE WHEN m.itemid IN (50867) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_5_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50867) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_5_missing,

  
  -- Feature 6: Creatinine
  AVG(CASE WHEN m.itemid IN (50912) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_6_mean,
  MIN(CASE WHEN m.itemid IN (50912) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_6_min,
  MAX(CASE WHEN m.itemid IN (50912) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_6_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50912) AND lv.rn = 1 LIMIT 1) AS feature_6_last,
  COUNT(CASE WHEN m.itemid IN (50912) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_6_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50912) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_6_missing,

  
  -- Feature 7: GCS - Eye Opening
  AVG(CASE WHEN m.itemid IN (220739, 184) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_7_mean,
  MIN(CASE WHEN m.itemid IN (220739, 184) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_7_min,
  MAX(CASE WHEN m.itemid IN (220739, 184) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_7_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (220739, 184) AND lv.rn = 1 LIMIT 1) AS feature_7_last,
  COUNT(CASE WHEN m.itemid IN (220739, 184) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_7_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (220739, 184) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_7_missing,

  
  -- Feature 8: GCS - Motor Response
  AVG(CASE WHEN m.itemid IN (223901, 454) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_8_mean,
  MIN(CASE WHEN m.itemid IN (223901, 454) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_8_min,
  MAX(CASE WHEN m.itemid IN (223901, 454) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_8_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (223901, 454) AND lv.rn = 1 LIMIT 1) AS feature_8_last,
  COUNT(CASE WHEN m.itemid IN (223901, 454) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_8_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (223901, 454) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_8_missing,

  
  -- Feature 9: GCS - Verbal Response
  AVG(CASE WHEN m.itemid IN (223900, 723) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_9_mean,
  MIN(CASE WHEN m.itemid IN (223900, 723) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_9_min,
  MAX(CASE WHEN m.itemid IN (223900, 723) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_9_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (223900, 723) AND lv.rn = 1 LIMIT 1) AS feature_9_last,
  COUNT(CASE WHEN m.itemid IN (223900, 723) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_9_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (223900, 723) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_9_missing,

  
  -- Feature 10: Head of Bed
  AVG(CASE WHEN m.itemid IN (228096, 224363) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_10_mean,
  MIN(CASE WHEN m.itemid IN (228096, 224363) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_10_min,
  MAX(CASE WHEN m.itemid IN (228096, 224363) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_10_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (228096, 224363) AND lv.rn = 1 LIMIT 1) AS feature_10_last,
  COUNT(CASE WHEN m.itemid IN (228096, 224363) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_10_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (228096, 224363) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_10_missing,

  
  -- Feature 11: Heart Rate
  AVG(CASE WHEN m.itemid IN (220045, 211) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_11_mean,
  MIN(CASE WHEN m.itemid IN (220045, 211) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_11_min,
  MAX(CASE WHEN m.itemid IN (220045, 211) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_11_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (220045, 211) AND lv.rn = 1 LIMIT 1) AS feature_11_last,
  COUNT(CASE WHEN m.itemid IN (220045, 211) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_11_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (220045, 211) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_11_missing,

  
  -- Feature 12: Hematocrit
  AVG(CASE WHEN m.itemid IN (50370, 51221) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_12_mean,
  MIN(CASE WHEN m.itemid IN (50370, 51221) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_12_min,
  MAX(CASE WHEN m.itemid IN (50370, 51221) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_12_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50370, 51221) AND lv.rn = 1 LIMIT 1) AS feature_12_last,
  COUNT(CASE WHEN m.itemid IN (50370, 51221) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_12_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50370, 51221) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_12_missing,

  
  -- Feature 13: Hemoglobin
  AVG(CASE WHEN m.itemid IN (50360, 51222) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_13_mean,
  MIN(CASE WHEN m.itemid IN (50360, 51222) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_13_min,
  MAX(CASE WHEN m.itemid IN (50360, 51222) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_13_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50360, 51222) AND lv.rn = 1 LIMIT 1) AS feature_13_last,
  COUNT(CASE WHEN m.itemid IN (50360, 51222) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_13_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50360, 51222) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_13_missing,

  
  -- Feature 14: INR(PT)
  AVG(CASE WHEN m.itemid IN (51237, 51274) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_14_mean,
  MIN(CASE WHEN m.itemid IN (51237, 51274) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_14_min,
  MAX(CASE WHEN m.itemid IN (51237, 51274) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_14_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (51237, 51274) AND lv.rn = 1 LIMIT 1) AS feature_14_last,
  COUNT(CASE WHEN m.itemid IN (51237, 51274) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_14_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (51237, 51274) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_14_missing,

  
  -- Feature 15: Inspired O2 Fraction
  AVG(CASE WHEN m.itemid IN (223835, 190, 3420) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_15_mean,
  MIN(CASE WHEN m.itemid IN (223835, 190, 3420) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_15_min,
  MAX(CASE WHEN m.itemid IN (223835, 190, 3420) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_15_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (223835, 190, 3420) AND lv.rn = 1 LIMIT 1) AS feature_15_last,
  COUNT(CASE WHEN m.itemid IN (223835, 190, 3420) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_15_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (223835, 190, 3420) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_15_missing,

  
  -- Feature 16: Lactate
  AVG(CASE WHEN m.itemid IN (50813) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_16_mean,
  MIN(CASE WHEN m.itemid IN (50813) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_16_min,
  MAX(CASE WHEN m.itemid IN (50813) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_16_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50813) AND lv.rn = 1 LIMIT 1) AS feature_16_last,
  COUNT(CASE WHEN m.itemid IN (50813) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_16_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50813) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_16_missing,

  
  -- Feature 17: Lymphocytes
  AVG(CASE WHEN m.itemid IN (51301, 51244) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_17_mean,
  MIN(CASE WHEN m.itemid IN (51301, 51244) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_17_min,
  MAX(CASE WHEN m.itemid IN (51301, 51244) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_17_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (51301, 51244) AND lv.rn = 1 LIMIT 1) AS feature_17_last,
  COUNT(CASE WHEN m.itemid IN (51301, 51244) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_17_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (51301, 51244) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_17_missing,

  
  -- Feature 18: MCHC
  AVG(CASE WHEN m.itemid IN (50340, 51250) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_18_mean,
  MIN(CASE WHEN m.itemid IN (50340, 51250) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_18_min,
  MAX(CASE WHEN m.itemid IN (50340, 51250) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_18_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50340, 51250) AND lv.rn = 1 LIMIT 1) AS feature_18_last,
  COUNT(CASE WHEN m.itemid IN (50340, 51250) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_18_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50340, 51250) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_18_missing,

  
  -- Feature 19: MCV
  AVG(CASE WHEN m.itemid IN (50330, 51249) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_19_mean,
  MIN(CASE WHEN m.itemid IN (50330, 51249) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_19_min,
  MAX(CASE WHEN m.itemid IN (50330, 51249) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_19_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50330, 51249) AND lv.rn = 1 LIMIT 1) AS feature_19_last,
  COUNT(CASE WHEN m.itemid IN (50330, 51249) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_19_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50330, 51249) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_19_missing,

  
  -- Feature 20: Magnesium
  AVG(CASE WHEN m.itemid IN (50960) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_20_mean,
  MIN(CASE WHEN m.itemid IN (50960) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_20_min,
  MAX(CASE WHEN m.itemid IN (50960) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_20_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50960) AND lv.rn = 1 LIMIT 1) AS feature_20_last,
  COUNT(CASE WHEN m.itemid IN (50960) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_20_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50960) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_20_missing,

  
  -- Feature 21: Mean Airway Pressure
  AVG(CASE WHEN m.itemid IN (220774, 221, 543) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_21_mean,
  MIN(CASE WHEN m.itemid IN (220774, 221, 543) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_21_min,
  MAX(CASE WHEN m.itemid IN (220774, 221, 543) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_21_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (220774, 221, 543) AND lv.rn = 1 LIMIT 1) AS feature_21_last,
  COUNT(CASE WHEN m.itemid IN (220774, 221, 543) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_21_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (220774, 221, 543) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_21_missing,

  
  -- Feature 22: Non Invasive Blood Pressure diastolic
  AVG(CASE WHEN m.itemid IN (220180, 8364, 8441) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_22_mean,
  MIN(CASE WHEN m.itemid IN (220180, 8364, 8441) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_22_min,
  MAX(CASE WHEN m.itemid IN (220180, 8364, 8441) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_22_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (220180, 8364, 8441) AND lv.rn = 1 LIMIT 1) AS feature_22_last,
  COUNT(CASE WHEN m.itemid IN (220180, 8364, 8441) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_22_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (220180, 8364, 8441) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_22_missing,

  
  -- Feature 23: Non Invasive Blood Pressure mean
  AVG(CASE WHEN m.itemid IN (220181, 456, 52) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_23_mean,
  MIN(CASE WHEN m.itemid IN (220181, 456, 52) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_23_min,
  MAX(CASE WHEN m.itemid IN (220181, 456, 52) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_23_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (220181, 456, 52) AND lv.rn = 1 LIMIT 1) AS feature_23_last,
  COUNT(CASE WHEN m.itemid IN (220181, 456, 52) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_23_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (220181, 456, 52) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_23_missing,

  
  -- Feature 24: Non Invasive Blood Pressure systolic
  AVG(CASE WHEN m.itemid IN (220179, 455, 51) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_24_mean,
  MIN(CASE WHEN m.itemid IN (220179, 455, 51) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_24_min,
  MAX(CASE WHEN m.itemid IN (220179, 455, 51) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_24_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (220179, 455, 51) AND lv.rn = 1 LIMIT 1) AS feature_24_last,
  COUNT(CASE WHEN m.itemid IN (220179, 455, 51) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_24_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (220179, 455, 51) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_24_missing,

  
  -- Feature 25: O2 saturation pulseoxymetry
  AVG(CASE WHEN m.itemid IN (220277, 646, 834) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_25_mean,
  MIN(CASE WHEN m.itemid IN (220277, 646, 834) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_25_min,
  MAX(CASE WHEN m.itemid IN (220277, 646, 834) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_25_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (220277, 646, 834) AND lv.rn = 1 LIMIT 1) AS feature_25_last,
  COUNT(CASE WHEN m.itemid IN (220277, 646, 834) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_25_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (220277, 646, 834) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_25_missing,

  
  -- Feature 26: PT
  AVG(CASE WHEN m.itemid IN (51274, 51237) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_26_mean,
  MIN(CASE WHEN m.itemid IN (51274, 51237) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_26_min,
  MAX(CASE WHEN m.itemid IN (51274, 51237) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_26_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (51274, 51237) AND lv.rn = 1 LIMIT 1) AS feature_26_last,
  COUNT(CASE WHEN m.itemid IN (51274, 51237) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_26_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (51274, 51237) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_26_missing,

  
  -- Feature 27: Potassium
  AVG(CASE WHEN m.itemid IN (50971, 50822) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_27_mean,
  MIN(CASE WHEN m.itemid IN (50971, 50822) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_27_min,
  MAX(CASE WHEN m.itemid IN (50971, 50822) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_27_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50971, 50822) AND lv.rn = 1 LIMIT 1) AS feature_27_last,
  COUNT(CASE WHEN m.itemid IN (50971, 50822) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_27_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50971, 50822) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_27_missing,

  
  -- Feature 28: RBC
  AVG(CASE WHEN m.itemid IN (50310, 51279) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_28_mean,
  MIN(CASE WHEN m.itemid IN (50310, 51279) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_28_min,
  MAX(CASE WHEN m.itemid IN (50310, 51279) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_28_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50310, 51279) AND lv.rn = 1 LIMIT 1) AS feature_28_last,
  COUNT(CASE WHEN m.itemid IN (50310, 51279) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_28_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50310, 51279) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_28_missing,

  
  -- Feature 29: RDW
  AVG(CASE WHEN m.itemid IN (50320, 51277) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_29_mean,
  MIN(CASE WHEN m.itemid IN (50320, 51277) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_29_min,
  MAX(CASE WHEN m.itemid IN (50320, 51277) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_29_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50320, 51277) AND lv.rn = 1 LIMIT 1) AS feature_29_last,
  COUNT(CASE WHEN m.itemid IN (50320, 51277) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_29_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50320, 51277) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_29_missing,

  
  -- Feature 30: Respiratory Rate
  AVG(CASE WHEN m.itemid IN (220210, 615, 618) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_30_mean,
  MIN(CASE WHEN m.itemid IN (220210, 615, 618) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_30_min,
  MAX(CASE WHEN m.itemid IN (220210, 615, 618) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_30_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (220210, 615, 618) AND lv.rn = 1 LIMIT 1) AS feature_30_last,
  COUNT(CASE WHEN m.itemid IN (220210, 615, 618) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_30_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (220210, 615, 618) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_30_missing,

  
  -- Feature 31: Sodium
  AVG(CASE WHEN m.itemid IN (50983, 50824) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_31_mean,
  MIN(CASE WHEN m.itemid IN (50983, 50824) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_31_min,
  MAX(CASE WHEN m.itemid IN (50983, 50824) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_31_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50983, 50824) AND lv.rn = 1 LIMIT 1) AS feature_31_last,
  COUNT(CASE WHEN m.itemid IN (50983, 50824) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_31_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50983, 50824) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_31_missing,

  
  -- Feature 32: Temperature Fahrenheit
  AVG(CASE WHEN m.itemid IN (223761, 678) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_32_mean,
  MIN(CASE WHEN m.itemid IN (223761, 678) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_32_min,
  MAX(CASE WHEN m.itemid IN (223761, 678) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_32_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (223761, 678) AND lv.rn = 1 LIMIT 1) AS feature_32_last,
  COUNT(CASE WHEN m.itemid IN (223761, 678) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_32_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (223761, 678) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_32_missing,

  
  -- Feature 33: Temperature
  AVG(CASE WHEN m.itemid IN (223762, 676) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_33_mean,
  MIN(CASE WHEN m.itemid IN (223762, 676) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_33_min,
  MAX(CASE WHEN m.itemid IN (223762, 676) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_33_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (223762, 676) AND lv.rn = 1 LIMIT 1) AS feature_33_last,
  COUNT(CASE WHEN m.itemid IN (223762, 676) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_33_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (223762, 676) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_33_missing,

  
  -- Feature 34: Tidal Volume (observed)
  AVG(CASE WHEN m.itemid IN (224688, 682) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_34_mean,
  MIN(CASE WHEN m.itemid IN (224688, 682) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_34_min,
  MAX(CASE WHEN m.itemid IN (224688, 682) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_34_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (224688, 682) AND lv.rn = 1 LIMIT 1) AS feature_34_last,
  COUNT(CASE WHEN m.itemid IN (224688, 682) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_34_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (224688, 682) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_34_missing,

  
  -- Feature 35: Urea Nitrogen
  AVG(CASE WHEN m.itemid IN (51066) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_35_mean,
  MIN(CASE WHEN m.itemid IN (51066) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_35_min,
  MAX(CASE WHEN m.itemid IN (51066) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_35_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (51066) AND lv.rn = 1 LIMIT 1) AS feature_35_last,
  COUNT(CASE WHEN m.itemid IN (51066) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_35_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (51066) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_35_missing,

  
  -- Feature 36: WBC
  AVG(CASE WHEN m.itemid IN (51300, 51301) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_36_mean,
  MIN(CASE WHEN m.itemid IN (51300, 51301) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_36_min,
  MAX(CASE WHEN m.itemid IN (51300, 51301) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_36_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (51300, 51301) AND lv.rn = 1 LIMIT 1) AS feature_36_last,
  COUNT(CASE WHEN m.itemid IN (51300, 51301) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_36_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (51300, 51301) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_36_missing,

  
  -- Feature 37: pH
  AVG(CASE WHEN m.itemid IN (220274, 780, 860) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_37_mean,
  MIN(CASE WHEN m.itemid IN (220274, 780, 860) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_37_min,
  MAX(CASE WHEN m.itemid IN (220274, 780, 860) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_37_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (220274, 780, 860) AND lv.rn = 1 LIMIT 1) AS feature_37_last,
  COUNT(CASE WHEN m.itemid IN (220274, 780, 860) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_37_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (220274, 780, 860) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_37_missing,

  
  -- Feature 38: pO2
  AVG(CASE WHEN m.itemid IN (50821, 490) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_38_mean,
  MIN(CASE WHEN m.itemid IN (50821, 490) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_38_min,
  MAX(CASE WHEN m.itemid IN (50821, 490) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_38_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50821, 490) AND lv.rn = 1 LIMIT 1) AS feature_38_last,
  COUNT(CASE WHEN m.itemid IN (50821, 490) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_38_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50821, 490) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_38_missing,

  
  -- Feature 39: Lactate Dehydrogenase (LD)
  AVG(CASE WHEN m.itemid IN (50954) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_39_mean,
  MIN(CASE WHEN m.itemid IN (50954) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_39_min,
  MAX(CASE WHEN m.itemid IN (50954) 
           AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
           THEN m.valuenum END) AS feature_39_max,
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.icustay_id = i.icustay_id AND lv.itemid IN (50954) AND lv.rn = 1 LIMIT 1) AS feature_39_last,
  COUNT(CASE WHEN m.itemid IN (50954) 
             AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
             THEN 1 END) AS feature_39_count,
  CASE WHEN COUNT(CASE WHEN m.itemid IN (50954) 
                       AND m.charttime >= i.intime AND m.charttime < i.intime_plus_24h 
                       THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_39_missing


FROM icu_window i
LEFT JOIN all_measurements m ON m.icustay_id = i.icustay_id
GROUP BY i.icustay_id, i.subject_id, i.hadm_id, i.intime, i.intime_plus_24h
ORDER BY i.icustay_id;