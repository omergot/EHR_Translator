
-- MIMIC-IV Feature Extraction Query using public tables (non-restrictive)  
-- Extracts 24-hour aggregated features for adult ICU stays (>=24h duration)
-- CHANGE: Switched from restrictive BSI cohort to public tables for more data

-- Using public tables instead of restrictive cohort
icu_window AS (
  SELECT 
    i.stay_id as icustay_id, 
    i.subject_id as subject_id, 
    i.hadm_id as hadm_id, 
    i.intime,
    i.intime + INTERVAL '24 hours' as intime_plus_24h
  FROM mimiciv_icu.icustays i
  JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
  WHERE i.intime IS NOT NULL
    AND i.hadm_id IS NOT NULL  -- Must have hospital admission
    AND EXTRACT(epoch FROM (i.outtime - i.intime))/3600 >= 24  -- At least 24h stay
    AND p.anchor_age >= 18  -- Adult patients
)
SELECT

  -- Heart Rate (Index: 0)
  AVG(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_0_mean,
  MIN(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_0_min,
  MAX(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_0_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220045 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220045 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_0_last,
  COUNT(CASE WHEN m.itemid = 220045 THEN 1 END) AS feature_0_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220045 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_0_missing,

  -- Temperature (Index: 1)
  AVG(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_1_mean,
  MIN(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_1_min,
  MAX(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_1_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 223761 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 223761 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_1_last,
  COUNT(CASE WHEN m.itemid = 223761 THEN 1 END) AS feature_1_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 223761 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_1_missing,

  -- Respiratory Rate (Index: 2)
  AVG(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_2_mean,
  MIN(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_2_min,
  MAX(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_2_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220210 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220210 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_2_last,
  COUNT(CASE WHEN m.itemid = 220210 THEN 1 END) AS feature_2_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220210 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_2_missing,

  -- O2 Sat (%) (Index: 3)
  AVG(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_3_mean,
  MIN(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_3_min,
  MAX(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_3_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220277 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220277 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_3_last,
  COUNT(CASE WHEN m.itemid = 220277 THEN 1 END) AS feature_3_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220277 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_3_missing,

  -- sodium (Index: 4)
  AVG(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_4_mean,
  MIN(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_4_min,
  MAX(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_4_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 50983 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 50983 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_4_last,
  COUNT(CASE WHEN m.itemid = 50983 THEN 1 END) AS feature_4_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 50983 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_4_missing,

  -- Additional metadata
  i.icustay_id,
  i.subject_id,
  i.hadm_id
FROM icu_window i
LEFT JOIN (
  SELECT subject_id, hadm_id, itemid, valuenum, charttime FROM mimiciv_hosp.labevents 
  WHERE valuenum IS NOT NULL
  UNION ALL
  SELECT subject_id, hadm_id, itemid, valuenum, charttime FROM mimiciv_icu.chartevents
  WHERE valuenum IS NOT NULL
) m ON (m.subject_id = i.subject_id AND m.hadm_id = i.hadm_id)
  AND m.charttime >= i.intime 
  AND m.charttime < i.intime_plus_24h
GROUP BY i.icustay_id, i.subject_id, i.hadm_id, i.intime, i.intime_plus_24h
ORDER BY i.icustay_id;
