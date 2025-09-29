
-- eICU Feature Extraction Query using public tables (non-restrictive)
-- Extracts 24-hour aggregated features for adult ICU stays (>=24h duration) 
-- CHANGE: Switched from restrictive BSI cohort to public tables for more data

-- Using public tables instead of restrictive cohort  
WITH stays AS (
  SELECT 
    p.patientunitstayid
  FROM eicu_crd.patient p
  WHERE p.patientunitstayid IS NOT NULL
    AND p.unitdischargeoffset >= 24 * 60  -- At least 24h stay (in minutes)
    AND p.age != '' AND p.age != 'Unknown' AND p.age IS NOT NULL
    AND CASE 
        WHEN p.age = '> 89' THEN 90 
        ELSE CAST(p.age AS INTEGER) 
        END >= 18  -- Adult patients
)
SELECT

  -- Heart Rate (Index: 0) - Available in vitalperiodic
  AVG(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_0_mean,
  MIN(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_0_min,
  MAX(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_0_max,
  (SELECT v2.heartrate
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.heartrate IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_0_last,
  COUNT(CASE WHEN v.heartrate IS NOT NULL THEN 1 END) AS feature_0_count,
  CASE WHEN COUNT(CASE WHEN v.heartrate IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_0_missing,

  -- Heart Rate (Index: 1) - Available in vitalperiodic
  AVG(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_1_mean,
  MIN(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_1_min,
  MAX(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_1_max,
  (SELECT v2.heartrate
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.heartrate IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_1_last,
  COUNT(CASE WHEN v.heartrate IS NOT NULL THEN 1 END) AS feature_1_count,
  CASE WHEN COUNT(CASE WHEN v.heartrate IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_1_missing,

  -- Heart Rate (Index: 2) - Available in vitalperiodic
  AVG(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_2_mean,
  MIN(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_2_min,
  MAX(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_2_max,
  (SELECT v2.heartrate
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.heartrate IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_2_last,
  COUNT(CASE WHEN v.heartrate IS NOT NULL THEN 1 END) AS feature_2_count,
  CASE WHEN COUNT(CASE WHEN v.heartrate IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_2_missing,

  -- Heart Rate (Index: 3) - Available in vitalperiodic
  AVG(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_3_mean,
  MIN(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_3_min,
  MAX(CASE WHEN v.heartrate IS NOT NULL THEN v.heartrate END) AS feature_3_max,
  (SELECT v2.heartrate
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.heartrate IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_3_last,
  COUNT(CASE WHEN v.heartrate IS NOT NULL THEN 1 END) AS feature_3_count,
  CASE WHEN COUNT(CASE WHEN v.heartrate IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_3_missing,

  -- Respiratory Rate (Index: 4) - Available in vitalperiodic
  AVG(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_4_mean,
  MIN(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_4_min,
  MAX(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_4_max,
  (SELECT v2.respiration
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.respiration IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_4_last,
  COUNT(CASE WHEN v.respiration IS NOT NULL THEN 1 END) AS feature_4_count,
  CASE WHEN COUNT(CASE WHEN v.respiration IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_4_missing,

  -- Respiratory Rate (Index: 5) - Available in vitalperiodic
  AVG(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_5_mean,
  MIN(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_5_min,
  MAX(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_5_max,
  (SELECT v2.respiration
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.respiration IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_5_last,
  COUNT(CASE WHEN v.respiration IS NOT NULL THEN 1 END) AS feature_5_count,
  CASE WHEN COUNT(CASE WHEN v.respiration IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_5_missing,

  -- Respiratory Rate (Index: 6) - Available in vitalperiodic
  AVG(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_6_mean,
  MIN(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_6_min,
  MAX(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_6_max,
  (SELECT v2.respiration
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.respiration IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_6_last,
  COUNT(CASE WHEN v.respiration IS NOT NULL THEN 1 END) AS feature_6_count,
  CASE WHEN COUNT(CASE WHEN v.respiration IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_6_missing,

  -- Respiratory Rate (Index: 7) - Available in vitalperiodic
  AVG(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_7_mean,
  MIN(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_7_min,
  MAX(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_7_max,
  (SELECT v2.respiration
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.respiration IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_7_last,
  COUNT(CASE WHEN v.respiration IS NOT NULL THEN 1 END) AS feature_7_count,
  CASE WHEN COUNT(CASE WHEN v.respiration IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_7_missing,

  -- O2 Sat (%) (Index: 8) - Available in vitalperiodic
  AVG(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_8_mean,
  MIN(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_8_min,
  MAX(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_8_max,
  (SELECT v2.sao2
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.sao2 IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_8_last,
  COUNT(CASE WHEN v.sao2 IS NOT NULL THEN 1 END) AS feature_8_count,
  CASE WHEN COUNT(CASE WHEN v.sao2 IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_8_missing,

  -- O2 Sat (%) (Index: 9) - Available in vitalperiodic
  AVG(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_9_mean,
  MIN(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_9_min,
  MAX(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_9_max,
  (SELECT v2.sao2
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.sao2 IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_9_last,
  COUNT(CASE WHEN v.sao2 IS NOT NULL THEN 1 END) AS feature_9_count,
  CASE WHEN COUNT(CASE WHEN v.sao2 IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_9_missing,

  -- O2 Sat (%) (Index: 10) - Available in vitalperiodic
  AVG(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_10_mean,
  MIN(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_10_min,
  MAX(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_10_max,
  (SELECT v2.sao2
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.sao2 IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_10_last,
  COUNT(CASE WHEN v.sao2 IS NOT NULL THEN 1 END) AS feature_10_count,
  CASE WHEN COUNT(CASE WHEN v.sao2 IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_10_missing,

  -- O2 Sat (%) (Index: 11) - Available in vitalperiodic
  AVG(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_11_mean,
  MIN(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_11_min,
  MAX(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_11_max,
  (SELECT v2.sao2
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.sao2 IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_11_last,
  COUNT(CASE WHEN v.sao2 IS NOT NULL THEN 1 END) AS feature_11_count,
  CASE WHEN COUNT(CASE WHEN v.sao2 IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_11_missing,

  -- Temperature (Index: 12) - Available in vitalperiodic
  AVG(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_12_mean,
  MIN(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_12_min,
  MAX(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_12_max,
  (SELECT v2.temperature
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.temperature IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_12_last,
  COUNT(CASE WHEN v.temperature IS NOT NULL THEN 1 END) AS feature_12_count,
  CASE WHEN COUNT(CASE WHEN v.temperature IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_12_missing,

  -- Temperature (Index: 13) - Available in vitalperiodic
  AVG(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_13_mean,
  MIN(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_13_min,
  MAX(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_13_max,
  (SELECT v2.temperature
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.temperature IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_13_last,
  COUNT(CASE WHEN v.temperature IS NOT NULL THEN 1 END) AS feature_13_count,
  CASE WHEN COUNT(CASE WHEN v.temperature IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_13_missing,

  -- Temperature (Index: 14) - Available in vitalperiodic
  AVG(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_14_mean,
  MIN(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_14_min,
  MAX(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_14_max,
  (SELECT v2.temperature
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.temperature IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_14_last,
  COUNT(CASE WHEN v.temperature IS NOT NULL THEN 1 END) AS feature_14_count,
  CASE WHEN COUNT(CASE WHEN v.temperature IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_14_missing,

  -- Temperature (Index: 15) - Available in vitalperiodic
  AVG(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_15_mean,
  MIN(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_15_min,
  MAX(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_15_max,
  (SELECT v2.temperature
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.temperature IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_15_last,
  COUNT(CASE WHEN v.temperature IS NOT NULL THEN 1 END) AS feature_15_count,
  CASE WHEN COUNT(CASE WHEN v.temperature IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_15_missing,

  -- Non-Invasive BP Mean (Index: 16) - Available in vitalperiodic
  AVG(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_16_mean,
  MIN(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_16_min,
  MAX(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_16_max,
  (SELECT v2.systemicmean
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.systemicmean IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_16_last,
  COUNT(CASE WHEN v.systemicmean IS NOT NULL THEN 1 END) AS feature_16_count,
  CASE WHEN COUNT(CASE WHEN v.systemicmean IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_16_missing,

  -- Non-Invasive BP Mean (Index: 17) - Available in vitalperiodic
  AVG(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_17_mean,
  MIN(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_17_min,
  MAX(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_17_max,
  (SELECT v2.systemicmean
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.systemicmean IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_17_last,
  COUNT(CASE WHEN v.systemicmean IS NOT NULL THEN 1 END) AS feature_17_count,
  CASE WHEN COUNT(CASE WHEN v.systemicmean IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_17_missing,

  -- Non-Invasive BP Mean (Index: 18) - Available in vitalperiodic
  AVG(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_18_mean,
  MIN(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_18_min,
  MAX(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_18_max,
  (SELECT v2.systemicmean
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.systemicmean IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_18_last,
  COUNT(CASE WHEN v.systemicmean IS NOT NULL THEN 1 END) AS feature_18_count,
  CASE WHEN COUNT(CASE WHEN v.systemicmean IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_18_missing,

  -- Non-Invasive BP Mean (Index: 19) - Available in vitalperiodic
  AVG(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_19_mean,
  MIN(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_19_min,
  MAX(CASE WHEN v.systemicmean IS NOT NULL THEN v.systemicmean END) AS feature_19_max,
  (SELECT v2.systemicmean
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.systemicmean IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_19_last,
  COUNT(CASE WHEN v.systemicmean IS NOT NULL THEN 1 END) AS feature_19_count,
  CASE WHEN COUNT(CASE WHEN v.systemicmean IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_19_missing,

  -- WBC (Index: 20) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_20_mean,
  NULL AS feature_20_min,
  NULL AS feature_20_max,
  NULL AS feature_20_last,
  0 AS feature_20_count,
  1 AS feature_20_missing,

  -- WBC (Index: 21) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_21_mean,
  NULL AS feature_21_min,
  NULL AS feature_21_max,
  NULL AS feature_21_last,
  0 AS feature_21_count,
  1 AS feature_21_missing,

  -- WBC (Index: 22) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_22_mean,
  NULL AS feature_22_min,
  NULL AS feature_22_max,
  NULL AS feature_22_last,
  0 AS feature_22_count,
  1 AS feature_22_missing,

  -- WBC (Index: 23) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_23_mean,
  NULL AS feature_23_min,
  NULL AS feature_23_max,
  NULL AS feature_23_last,
  0 AS feature_23_count,
  1 AS feature_23_missing,

  -- sodium (Index: 24) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_24_mean,
  NULL AS feature_24_min,
  NULL AS feature_24_max,
  NULL AS feature_24_last,
  0 AS feature_24_count,
  1 AS feature_24_missing,

  -- sodium (Index: 25) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_25_mean,
  NULL AS feature_25_min,
  NULL AS feature_25_max,
  NULL AS feature_25_last,
  0 AS feature_25_count,
  1 AS feature_25_missing,

  -- sodium (Index: 26) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_26_mean,
  NULL AS feature_26_min,
  NULL AS feature_26_max,
  NULL AS feature_26_last,
  0 AS feature_26_count,
  1 AS feature_26_missing,

  -- sodium (Index: 27) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_27_mean,
  NULL AS feature_27_min,
  NULL AS feature_27_max,
  NULL AS feature_27_last,
  0 AS feature_27_count,
  1 AS feature_27_missing,

  -- creatinine (Index: 28) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_28_mean,
  NULL AS feature_28_min,
  NULL AS feature_28_max,
  NULL AS feature_28_last,
  0 AS feature_28_count,
  1 AS feature_28_missing,

  -- creatinine (Index: 29) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_29_mean,
  NULL AS feature_29_min,
  NULL AS feature_29_max,
  NULL AS feature_29_last,
  0 AS feature_29_count,
  1 AS feature_29_missing,

  -- creatinine (Index: 30) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_30_mean,
  NULL AS feature_30_min,
  NULL AS feature_30_max,
  NULL AS feature_30_last,
  0 AS feature_30_count,
  1 AS feature_30_missing,

  -- creatinine (Index: 31) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_31_mean,
  NULL AS feature_31_min,
  NULL AS feature_31_max,
  NULL AS feature_31_last,
  0 AS feature_31_count,
  1 AS feature_31_missing,

  -- Age (Index: 32) - From patient table
  CASE WHEN p.age = '> 89' THEN 90 ELSE CAST(p.age AS INTEGER) END AS feature_32_mean,
  CASE WHEN p.age = '> 89' THEN 90 ELSE CAST(p.age AS INTEGER) END AS feature_32_min,
  CASE WHEN p.age = '> 89' THEN 90 ELSE CAST(p.age AS INTEGER) END AS feature_32_max,
  CASE WHEN p.age = '> 89' THEN 90 ELSE CAST(p.age AS INTEGER) END AS feature_32_last,
  1 AS feature_32_count,
  CASE WHEN p.age IS NULL OR p.age = '' THEN 1 ELSE 0 END AS feature_32_missing,

  -- Gender (Index: 33) - From patient table
  CASE WHEN p.gender = 'Male' THEN 1 ELSE 0 END AS feature_33_mean,
  CASE WHEN p.gender = 'Male' THEN 1 ELSE 0 END AS feature_33_min,
  CASE WHEN p.gender = 'Male' THEN 1 ELSE 0 END AS feature_33_max,
  CASE WHEN p.gender = 'Male' THEN 1 ELSE 0 END AS feature_33_last,
  1 AS feature_33_count,
  CASE WHEN p.gender IS NULL OR p.gender = '' THEN 1 ELSE 0 END AS feature_33_missing,

  -- Additional metadata
  s.patientunitstayid as icustay_id
FROM stays s
JOIN eicu_crd.patient p ON p.patientunitstayid = s.patientunitstayid
LEFT JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = s.patientunitstayid
  AND v.observationoffset >= 0
  AND v.observationoffset < 24 * 60
GROUP BY s.patientunitstayid, p.age, p.gender
ORDER BY s.patientunitstayid;
