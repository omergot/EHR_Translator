
-- eICU Feature Extraction Query using public tables (non-restrictive)
-- Extracts 24-hour aggregated features for adult ICU stays (>=24h duration) 
-- CHANGE: Switched from restrictive BSI cohort to public tables for more data

-- Using public tables instead of restrictive cohort  
stays AS (
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

  -- Temperature (Index: 1) - Available in vitalperiodic
  AVG(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_1_mean,
  MIN(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_1_min,
  MAX(CASE WHEN v.temperature IS NOT NULL THEN v.temperature END) AS feature_1_max,
  (SELECT v2.temperature
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.temperature IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_1_last,
  COUNT(CASE WHEN v.temperature IS NOT NULL THEN 1 END) AS feature_1_count,
  CASE WHEN COUNT(CASE WHEN v.temperature IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_1_missing,

  -- Respiratory Rate (Index: 2) - Available in vitalperiodic
  AVG(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_2_mean,
  MIN(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_2_min,
  MAX(CASE WHEN v.respiration IS NOT NULL THEN v.respiration END) AS feature_2_max,
  (SELECT v2.respiration
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.respiration IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_2_last,
  COUNT(CASE WHEN v.respiration IS NOT NULL THEN 1 END) AS feature_2_count,
  CASE WHEN COUNT(CASE WHEN v.respiration IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_2_missing,

  -- O2 Sat (%) (Index: 3) - Available in vitalperiodic
  AVG(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_3_mean,
  MIN(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_3_min,
  MAX(CASE WHEN v.sao2 IS NOT NULL THEN v.sao2 END) AS feature_3_max,
  (SELECT v2.sao2
   FROM eicu_crd.vitalperiodic v2
   WHERE v2.patientunitstayid = s.patientunitstayid
     AND v2.observationoffset >= 0
     AND v2.observationoffset < 24 * 60
     AND v2.sao2 IS NOT NULL
   ORDER BY v2.observationoffset DESC
   LIMIT 1) AS feature_3_last,
  COUNT(CASE WHEN v.sao2 IS NOT NULL THEN 1 END) AS feature_3_count,
  CASE WHEN COUNT(CASE WHEN v.sao2 IS NOT NULL THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_3_missing,

  -- sodium (Index: 4) - Not available in vitalperiodic (would need lab/nursing tables)
  NULL AS feature_4_mean,
  NULL AS feature_4_min,
  NULL AS feature_4_max,
  NULL AS feature_4_last,
  0 AS feature_4_count,
  1 AS feature_4_missing,

  -- Additional metadata
  s.patientunitstayid as icustay_id
FROM stays s
LEFT JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = s.patientunitstayid
  AND v.observationoffset >= 0
  AND v.observationoffset < 24 * 60
GROUP BY s.patientunitstayid
ORDER BY s.patientunitstayid;
