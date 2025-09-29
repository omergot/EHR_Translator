
-- MIMIC-IV Feature Extraction Query using public tables (non-restrictive)  
-- Extracts 24-hour aggregated features for adult ICU stays (>=24h duration)
-- CHANGE: Switched from restrictive BSI cohort to public tables for more data

-- Using public tables instead of restrictive cohort
WITH icu_window AS (
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

  -- Heart Rate (Index: 1)
  AVG(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_1_mean,
  MIN(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_1_min,
  MAX(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_1_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220045 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220045 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_1_last,
  COUNT(CASE WHEN m.itemid = 220045 THEN 1 END) AS feature_1_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220045 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_1_missing,

  -- Heart Rate (Index: 2)
  AVG(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_2_mean,
  MIN(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_2_min,
  MAX(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_2_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220045 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220045 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_2_last,
  COUNT(CASE WHEN m.itemid = 220045 THEN 1 END) AS feature_2_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220045 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_2_missing,

  -- Heart Rate (Index: 3)
  AVG(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_3_mean,
  MIN(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_3_min,
  MAX(CASE WHEN m.itemid = 220045 THEN m.valuenum END) AS feature_3_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220045 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220045 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_3_last,
  COUNT(CASE WHEN m.itemid = 220045 THEN 1 END) AS feature_3_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220045 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_3_missing,

  -- Respiratory Rate (Index: 4)
  AVG(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_4_mean,
  MIN(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_4_min,
  MAX(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_4_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220210 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220210 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_4_last,
  COUNT(CASE WHEN m.itemid = 220210 THEN 1 END) AS feature_4_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220210 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_4_missing,

  -- Respiratory Rate (Index: 5)
  AVG(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_5_mean,
  MIN(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_5_min,
  MAX(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_5_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220210 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220210 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_5_last,
  COUNT(CASE WHEN m.itemid = 220210 THEN 1 END) AS feature_5_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220210 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_5_missing,

  -- Respiratory Rate (Index: 6)
  AVG(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_6_mean,
  MIN(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_6_min,
  MAX(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_6_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220210 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220210 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_6_last,
  COUNT(CASE WHEN m.itemid = 220210 THEN 1 END) AS feature_6_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220210 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_6_missing,

  -- Respiratory Rate (Index: 7)
  AVG(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_7_mean,
  MIN(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_7_min,
  MAX(CASE WHEN m.itemid = 220210 THEN m.valuenum END) AS feature_7_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220210 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220210 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_7_last,
  COUNT(CASE WHEN m.itemid = 220210 THEN 1 END) AS feature_7_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220210 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_7_missing,

  -- O2 Sat (%) (Index: 8)
  AVG(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_8_mean,
  MIN(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_8_min,
  MAX(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_8_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220277 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220277 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_8_last,
  COUNT(CASE WHEN m.itemid = 220277 THEN 1 END) AS feature_8_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220277 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_8_missing,

  -- O2 Sat (%) (Index: 9)
  AVG(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_9_mean,
  MIN(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_9_min,
  MAX(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_9_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220277 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220277 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_9_last,
  COUNT(CASE WHEN m.itemid = 220277 THEN 1 END) AS feature_9_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220277 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_9_missing,

  -- O2 Sat (%) (Index: 10)
  AVG(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_10_mean,
  MIN(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_10_min,
  MAX(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_10_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220277 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220277 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_10_last,
  COUNT(CASE WHEN m.itemid = 220277 THEN 1 END) AS feature_10_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220277 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_10_missing,

  -- O2 Sat (%) (Index: 11)
  AVG(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_11_mean,
  MIN(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_11_min,
  MAX(CASE WHEN m.itemid = 220277 THEN m.valuenum END) AS feature_11_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220277 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220277 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_11_last,
  COUNT(CASE WHEN m.itemid = 220277 THEN 1 END) AS feature_11_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220277 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_11_missing,

  -- Temperature (Index: 12)
  AVG(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_12_mean,
  MIN(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_12_min,
  MAX(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_12_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 223761 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 223761 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_12_last,
  COUNT(CASE WHEN m.itemid = 223761 THEN 1 END) AS feature_12_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 223761 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_12_missing,

  -- Temperature (Index: 13)
  AVG(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_13_mean,
  MIN(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_13_min,
  MAX(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_13_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 223761 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 223761 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_13_last,
  COUNT(CASE WHEN m.itemid = 223761 THEN 1 END) AS feature_13_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 223761 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_13_missing,

  -- Temperature (Index: 14)
  AVG(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_14_mean,
  MIN(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_14_min,
  MAX(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_14_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 223761 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 223761 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_14_last,
  COUNT(CASE WHEN m.itemid = 223761 THEN 1 END) AS feature_14_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 223761 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_14_missing,

  -- Temperature (Index: 15)
  AVG(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_15_mean,
  MIN(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_15_min,
  MAX(CASE WHEN m.itemid = 223761 THEN m.valuenum END) AS feature_15_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 223761 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 223761 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_15_last,
  COUNT(CASE WHEN m.itemid = 223761 THEN 1 END) AS feature_15_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 223761 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_15_missing,

  -- Non-Invasive BP Mean (Index: 16)
  AVG(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_16_mean,
  MIN(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_16_min,
  MAX(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_16_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220181 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220181 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_16_last,
  COUNT(CASE WHEN m.itemid = 220181 THEN 1 END) AS feature_16_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220181 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_16_missing,

  -- Non-Invasive BP Mean (Index: 17)
  AVG(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_17_mean,
  MIN(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_17_min,
  MAX(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_17_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220181 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220181 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_17_last,
  COUNT(CASE WHEN m.itemid = 220181 THEN 1 END) AS feature_17_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220181 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_17_missing,

  -- Non-Invasive BP Mean (Index: 18)
  AVG(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_18_mean,
  MIN(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_18_min,
  MAX(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_18_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220181 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220181 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_18_last,
  COUNT(CASE WHEN m.itemid = 220181 THEN 1 END) AS feature_18_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220181 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_18_missing,

  -- Non-Invasive BP Mean (Index: 19)
  AVG(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_19_mean,
  MIN(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_19_min,
  MAX(CASE WHEN m.itemid = 220181 THEN m.valuenum END) AS feature_19_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 220181 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 220181 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_19_last,
  COUNT(CASE WHEN m.itemid = 220181 THEN 1 END) AS feature_19_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 220181 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_19_missing,

  -- WBC (Index: 20)
  AVG(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_20_mean,
  MIN(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_20_min,
  MAX(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_20_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 51300 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 51300 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_20_last,
  COUNT(CASE WHEN m.itemid = 51300 THEN 1 END) AS feature_20_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 51300 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_20_missing,

  -- WBC (Index: 21)
  AVG(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_21_mean,
  MIN(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_21_min,
  MAX(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_21_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 51300 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 51300 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_21_last,
  COUNT(CASE WHEN m.itemid = 51300 THEN 1 END) AS feature_21_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 51300 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_21_missing,

  -- WBC (Index: 22)
  AVG(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_22_mean,
  MIN(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_22_min,
  MAX(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_22_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 51300 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 51300 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_22_last,
  COUNT(CASE WHEN m.itemid = 51300 THEN 1 END) AS feature_22_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 51300 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_22_missing,

  -- WBC (Index: 23)
  AVG(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_23_mean,
  MIN(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_23_min,
  MAX(CASE WHEN m.itemid = 51300 THEN m.valuenum END) AS feature_23_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 51300 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 51300 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_23_last,
  COUNT(CASE WHEN m.itemid = 51300 THEN 1 END) AS feature_23_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 51300 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_23_missing,

  -- sodium (Index: 24)
  AVG(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_24_mean,
  MIN(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_24_min,
  MAX(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_24_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 50983 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 50983 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_24_last,
  COUNT(CASE WHEN m.itemid = 50983 THEN 1 END) AS feature_24_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 50983 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_24_missing,

  -- sodium (Index: 25)
  AVG(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_25_mean,
  MIN(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_25_min,
  MAX(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_25_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 50983 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 50983 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_25_last,
  COUNT(CASE WHEN m.itemid = 50983 THEN 1 END) AS feature_25_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 50983 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_25_missing,

  -- sodium (Index: 26)
  AVG(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_26_mean,
  MIN(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_26_min,
  MAX(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_26_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 50983 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 50983 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_26_last,
  COUNT(CASE WHEN m.itemid = 50983 THEN 1 END) AS feature_26_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 50983 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_26_missing,

  -- sodium (Index: 27)
  AVG(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_27_mean,
  MIN(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_27_min,
  MAX(CASE WHEN m.itemid = 50983 THEN m.valuenum END) AS feature_27_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 50983 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 50983 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_27_last,
  COUNT(CASE WHEN m.itemid = 50983 THEN 1 END) AS feature_27_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 50983 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_27_missing,

  -- creatinine (Index: 28)
  AVG(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_28_mean,
  MIN(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_28_min,
  MAX(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_28_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 50912 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 50912 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_28_last,
  COUNT(CASE WHEN m.itemid = 50912 THEN 1 END) AS feature_28_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 50912 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_28_missing,

  -- creatinine (Index: 29)
  AVG(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_29_mean,
  MIN(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_29_min,
  MAX(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_29_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 50912 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 50912 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_29_last,
  COUNT(CASE WHEN m.itemid = 50912 THEN 1 END) AS feature_29_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 50912 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_29_missing,

  -- creatinine (Index: 30)
  AVG(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_30_mean,
  MIN(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_30_min,
  MAX(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_30_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 50912 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 50912 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_30_last,
  COUNT(CASE WHEN m.itemid = 50912 THEN 1 END) AS feature_30_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 50912 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_30_missing,

  -- creatinine (Index: 31)
  AVG(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_31_mean,
  MIN(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_31_min,
  MAX(CASE WHEN m.itemid = 50912 THEN m.valuenum END) AS feature_31_max,
  (SELECT l2.valuenum
   FROM (
     SELECT l.valuenum, l.charttime FROM mimiciv_hosp.labevents l 
     WHERE l.subject_id = i.subject_id AND l.hadm_id = i.hadm_id AND l.itemid = 50912 AND l.charttime >= i.intime AND l.charttime < i.intime_plus_24h AND l.valuenum IS NOT NULL
     UNION ALL
     SELECT c.valuenum, c.charttime FROM mimiciv_icu.chartevents c
     WHERE c.stay_id = i.icustay_id AND c.itemid = 50912 AND c.charttime >= i.intime AND c.charttime < i.intime_plus_24h AND c.valuenum IS NOT NULL
   ) l2
   ORDER BY l2.charttime DESC LIMIT 1) AS feature_31_last,
  COUNT(CASE WHEN m.itemid = 50912 THEN 1 END) AS feature_31_count,
  CASE WHEN COUNT(CASE WHEN m.itemid = 50912 THEN 1 END) = 0 THEN 1 ELSE 0 END AS feature_31_missing,

  -- Age (Index: 32) - From patients table
  p.anchor_age AS feature_32_mean,
  p.anchor_age AS feature_32_min,
  p.anchor_age AS feature_32_max,
  p.anchor_age AS feature_32_last,
  1 AS feature_32_count,
  CASE WHEN p.anchor_age IS NULL THEN 1 ELSE 0 END AS feature_32_missing,

  -- Gender (Index: 33) - From patients table
  CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS feature_33_mean,
  CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS feature_33_min,
  CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS feature_33_max,
  CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS feature_33_last,
  1 AS feature_33_count,
  CASE WHEN p.gender IS NULL THEN 1 ELSE 0 END AS feature_33_missing,

  -- Additional metadata
  i.icustay_id,
  i.subject_id,
  i.hadm_id
FROM icu_window i
LEFT JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
LEFT JOIN (
  SELECT subject_id, hadm_id, itemid, valuenum, charttime FROM mimiciv_hosp.labevents 
  WHERE valuenum IS NOT NULL
  UNION ALL
  SELECT subject_id, hadm_id, itemid, valuenum, charttime FROM mimiciv_icu.chartevents
  WHERE valuenum IS NOT NULL
) m ON (m.subject_id = i.subject_id AND m.hadm_id = i.hadm_id)
  AND m.charttime >= i.intime 
  AND m.charttime < i.intime_plus_24h
GROUP BY i.icustay_id, i.subject_id, i.hadm_id, i.intime, i.intime_plus_24h, p.anchor_age, p.gender
ORDER BY i.icustay_id;
