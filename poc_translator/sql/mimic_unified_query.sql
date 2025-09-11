
-- UNIFIED MIMIC-IV Feature Extraction Query for ALL Patients
-- Extracts 40 clinical features from all hospital admissions (not limited to ICU)
-- Uses first 24 hours after admission as observation window
-- Prioritizes data accuracy and completeness over query speed

WITH patient_admissions AS (
  -- Get all hospital admissions with extended time windows for comprehensive data
  SELECT 
    a.hadm_id,
    a.subject_id,
    a.admittime,
    a.admittime + INTERVAL '24 hours' as obs_end_time,
    -- Extended window for better data capture
    a.admittime - INTERVAL '6 hours' as extended_start,
    a.admittime + INTERVAL '48 hours' as extended_end,
    -- Assign ICU stay ID if patient has ICU stay
    COALESCE(
      (SELECT i.stay_id FROM mimiciv_icu.icustays i 
       WHERE i.hadm_id = a.hadm_id 
       AND i.intime <= a.admittime + INTERVAL '24 hours'
       ORDER BY i.intime LIMIT 1), 
      a.hadm_id  -- Use hadm_id as identifier if no ICU stay
    ) as stay_identifier
  FROM mimiciv_hosp.admissions a
  WHERE a.admittime IS NOT NULL
),

-- Consolidated measurements from all relevant MIMIC tables
all_measurements AS (
  -- Hospital lab events (most comprehensive lab data)
  SELECT 
    pa.hadm_id, pa.subject_id, pa.stay_identifier,
    pa.admittime, pa.obs_end_time,
    l.itemid, l.valuenum, l.charttime,
    'lab' as data_source
  FROM patient_admissions pa
  JOIN mimiciv_hosp.labevents l ON l.hadm_id = pa.hadm_id
  WHERE l.valuenum IS NOT NULL 
    AND l.charttime >= pa.extended_start
    AND l.charttime < pa.extended_end
    AND l.valuenum > 0  -- Filter out negative/zero values that are likely errors
  
  UNION ALL
  
  -- ICU chart events (vital signs, ventilator settings, etc.)
  SELECT 
    pa.hadm_id, pa.subject_id, pa.stay_identifier,
    pa.admittime, pa.obs_end_time,
    c.itemid, c.valuenum, c.charttime,
    'chart' as data_source
  FROM patient_admissions pa
  JOIN mimiciv_icu.icustays i ON i.hadm_id = pa.hadm_id
  JOIN mimiciv_icu.chartevents c ON c.stay_id = i.stay_id
  WHERE c.valuenum IS NOT NULL
    AND c.charttime >= pa.extended_start
    AND c.charttime < pa.extended_end
    AND c.valuenum > 0
  
  -- Note: Procedure events excluded for reliability - lab and chart events provide comprehensive coverage
),

-- Pre-compute last values for efficiency
last_values AS (
  SELECT 
    hadm_id, itemid,
    FIRST_VALUE(valuenum) OVER (
      PARTITION BY hadm_id, itemid 
      ORDER BY charttime DESC 
      ROWS UNBOUNDED PRECEDING
    ) as last_value
  FROM all_measurements
  WHERE charttime >= admittime AND charttime < obs_end_time
),

-- Feature-specific itemid mappings
feature_itemids AS (
    SELECT 0 as feature_idx, 'Alanine Aminotransferase (ALT)' as feature_name, ARRAY[50861, 50862] as itemids
  UNION ALL
    SELECT 1 as feature_idx, 'Albumin' as feature_name, ARRAY[50862] as itemids
  UNION ALL
    SELECT 2 as feature_idx, 'Alkaline Phosphatase' as feature_name, ARRAY[50863] as itemids
  UNION ALL
    SELECT 3 as feature_idx, 'Asparate Aminotransferase (AST)' as feature_name, ARRAY[50876] as itemids
  UNION ALL
    SELECT 4 as feature_idx, 'Bilirubin' as feature_name, ARRAY[50885, 50884, 50883] as itemids
  UNION ALL
    SELECT 5 as feature_idx, 'C-Reactive Protein' as feature_name, ARRAY[50867] as itemids
  UNION ALL
    SELECT 6 as feature_idx, 'Creatinine' as feature_name, ARRAY[50912] as itemids
  UNION ALL
    SELECT 7 as feature_idx, 'GCS - Eye Opening' as feature_name, ARRAY[220739, 184] as itemids
  UNION ALL
    SELECT 8 as feature_idx, 'GCS - Motor Response' as feature_name, ARRAY[223901, 454] as itemids
  UNION ALL
    SELECT 9 as feature_idx, 'GCS - Verbal Response' as feature_name, ARRAY[223900, 723] as itemids
  UNION ALL
    SELECT 10 as feature_idx, 'Head of Bed' as feature_name, ARRAY[228096, 224363] as itemids
  UNION ALL
    SELECT 11 as feature_idx, 'Heart Rate' as feature_name, ARRAY[220045, 211] as itemids
  UNION ALL
    SELECT 12 as feature_idx, 'Hematocrit' as feature_name, ARRAY[50370, 51221] as itemids
  UNION ALL
    SELECT 13 as feature_idx, 'Hemoglobin' as feature_name, ARRAY[50360, 51222] as itemids
  UNION ALL
    SELECT 14 as feature_idx, 'INR(PT)' as feature_name, ARRAY[51237, 51274] as itemids
  UNION ALL
    SELECT 15 as feature_idx, 'Inspired O2 Fraction' as feature_name, ARRAY[223835, 190, 3420] as itemids
  UNION ALL
    SELECT 16 as feature_idx, 'Lactate' as feature_name, ARRAY[50813] as itemids
  UNION ALL
    SELECT 17 as feature_idx, 'Lymphocytes' as feature_name, ARRAY[51301, 51244] as itemids
  UNION ALL
    SELECT 18 as feature_idx, 'MCHC' as feature_name, ARRAY[50340, 51250] as itemids
  UNION ALL
    SELECT 19 as feature_idx, 'MCV' as feature_name, ARRAY[50330, 51249] as itemids
  UNION ALL
    SELECT 20 as feature_idx, 'Magnesium' as feature_name, ARRAY[50960] as itemids
  UNION ALL
    SELECT 21 as feature_idx, 'Mean Airway Pressure' as feature_name, ARRAY[220774, 221, 543] as itemids
  UNION ALL
    SELECT 22 as feature_idx, 'Non Invasive Blood Pressure diastolic' as feature_name, ARRAY[220180, 8364, 8441] as itemids
  UNION ALL
    SELECT 23 as feature_idx, 'Non Invasive Blood Pressure mean' as feature_name, ARRAY[220181, 456, 52] as itemids
  UNION ALL
    SELECT 24 as feature_idx, 'Non Invasive Blood Pressure systolic' as feature_name, ARRAY[220179, 455, 51] as itemids
  UNION ALL
    SELECT 25 as feature_idx, 'O2 saturation pulseoxymetry' as feature_name, ARRAY[220277, 646, 834] as itemids
  UNION ALL
    SELECT 26 as feature_idx, 'PT' as feature_name, ARRAY[51274, 51237] as itemids
  UNION ALL
    SELECT 27 as feature_idx, 'Potassium' as feature_name, ARRAY[50971, 50822] as itemids
  UNION ALL
    SELECT 28 as feature_idx, 'RBC' as feature_name, ARRAY[50310, 51279] as itemids
  UNION ALL
    SELECT 29 as feature_idx, 'RDW' as feature_name, ARRAY[50320, 51277] as itemids
  UNION ALL
    SELECT 30 as feature_idx, 'Respiratory Rate' as feature_name, ARRAY[220210, 615, 618] as itemids
  UNION ALL
    SELECT 31 as feature_idx, 'Sodium' as feature_name, ARRAY[50983, 50824] as itemids
  UNION ALL
    SELECT 32 as feature_idx, 'Temperature Fahrenheit' as feature_name, ARRAY[223761, 678] as itemids
  UNION ALL
    SELECT 33 as feature_idx, 'Temperature' as feature_name, ARRAY[223762, 676] as itemids
  UNION ALL
    SELECT 34 as feature_idx, 'Tidal Volume (observed)' as feature_name, ARRAY[224688, 682] as itemids
  UNION ALL
    SELECT 35 as feature_idx, 'Urea Nitrogen' as feature_name, ARRAY[51066] as itemids
  UNION ALL
    SELECT 36 as feature_idx, 'WBC' as feature_name, ARRAY[51300, 51301] as itemids
  UNION ALL
    SELECT 37 as feature_idx, 'pH' as feature_name, ARRAY[220274, 780, 860] as itemids
  UNION ALL
    SELECT 38 as feature_idx, 'pO2' as feature_name, ARRAY[50821, 490] as itemids
  UNION ALL
    SELECT 39 as feature_idx, 'Lactate Dehydrogenase (LD)' as feature_name, ARRAY[50954] as itemids
)

-- Main feature extraction with comprehensive statistics
SELECT
  pa.hadm_id,
  pa.subject_id,
  pa.stay_identifier,
  pa.admittime,
  -- Feature 0: Alanine Aminotransferase (ALT)
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 0)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_0_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 0)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_0_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 0)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_0_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 0)) 
   LIMIT 1) AS feature_0_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 0)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_0_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 0)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_0_missing,

  -- Feature 1: Albumin
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 1)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_1_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 1)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_1_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 1)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_1_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 1)) 
   LIMIT 1) AS feature_1_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 1)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_1_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 1)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_1_missing,

  -- Feature 2: Alkaline Phosphatase
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 2)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_2_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 2)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_2_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 2)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_2_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 2)) 
   LIMIT 1) AS feature_2_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 2)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_2_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 2)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_2_missing,

  -- Feature 3: Asparate Aminotransferase (AST)
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 3)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_3_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 3)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_3_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 3)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_3_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 3)) 
   LIMIT 1) AS feature_3_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 3)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_3_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 3)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_3_missing,

  -- Feature 4: Bilirubin
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 4)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_4_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 4)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_4_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 4)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_4_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 4)) 
   LIMIT 1) AS feature_4_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 4)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_4_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 4)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_4_missing,

  -- Feature 5: C-Reactive Protein
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 5)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_5_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 5)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_5_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 5)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_5_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 5)) 
   LIMIT 1) AS feature_5_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 5)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_5_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 5)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_5_missing,

  -- Feature 6: Creatinine
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 6)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_6_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 6)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_6_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 6)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_6_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 6)) 
   LIMIT 1) AS feature_6_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 6)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_6_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 6)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_6_missing,

  -- Feature 7: GCS - Eye Opening
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 7)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_7_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 7)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_7_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 7)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_7_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 7)) 
   LIMIT 1) AS feature_7_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 7)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_7_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 7)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_7_missing,

  -- Feature 8: GCS - Motor Response
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 8)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_8_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 8)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_8_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 8)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_8_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 8)) 
   LIMIT 1) AS feature_8_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 8)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_8_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 8)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_8_missing,

  -- Feature 9: GCS - Verbal Response
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 9)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_9_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 9)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_9_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 9)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_9_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 9)) 
   LIMIT 1) AS feature_9_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 9)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_9_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 9)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_9_missing,

  -- Feature 10: Head of Bed
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 10)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_10_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 10)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_10_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 10)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_10_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 10)) 
   LIMIT 1) AS feature_10_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 10)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_10_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 10)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_10_missing,

  -- Feature 11: Heart Rate
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 11)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_11_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 11)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_11_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 11)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_11_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 11)) 
   LIMIT 1) AS feature_11_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 11)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_11_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 11)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_11_missing,

  -- Feature 12: Hematocrit
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 12)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_12_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 12)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_12_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 12)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_12_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 12)) 
   LIMIT 1) AS feature_12_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 12)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_12_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 12)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_12_missing,

  -- Feature 13: Hemoglobin
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 13)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_13_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 13)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_13_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 13)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_13_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 13)) 
   LIMIT 1) AS feature_13_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 13)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_13_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 13)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_13_missing,

  -- Feature 14: INR(PT)
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 14)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_14_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 14)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_14_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 14)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_14_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 14)) 
   LIMIT 1) AS feature_14_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 14)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_14_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 14)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_14_missing,

  -- Feature 15: Inspired O2 Fraction
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 15)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_15_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 15)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_15_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 15)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_15_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 15)) 
   LIMIT 1) AS feature_15_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 15)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_15_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 15)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_15_missing,

  -- Feature 16: Lactate
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 16)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_16_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 16)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_16_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 16)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_16_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 16)) 
   LIMIT 1) AS feature_16_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 16)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_16_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 16)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_16_missing,

  -- Feature 17: Lymphocytes
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 17)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_17_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 17)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_17_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 17)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_17_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 17)) 
   LIMIT 1) AS feature_17_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 17)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_17_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 17)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_17_missing,

  -- Feature 18: MCHC
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 18)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_18_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 18)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_18_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 18)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_18_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 18)) 
   LIMIT 1) AS feature_18_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 18)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_18_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 18)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_18_missing,

  -- Feature 19: MCV
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 19)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_19_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 19)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_19_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 19)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_19_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 19)) 
   LIMIT 1) AS feature_19_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 19)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_19_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 19)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_19_missing,

  -- Feature 20: Magnesium
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 20)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_20_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 20)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_20_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 20)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_20_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 20)) 
   LIMIT 1) AS feature_20_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 20)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_20_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 20)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_20_missing,

  -- Feature 21: Mean Airway Pressure
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 21)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_21_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 21)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_21_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 21)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_21_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 21)) 
   LIMIT 1) AS feature_21_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 21)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_21_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 21)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_21_missing,

  -- Feature 22: Non Invasive Blood Pressure diastolic
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 22)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_22_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 22)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_22_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 22)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_22_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 22)) 
   LIMIT 1) AS feature_22_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 22)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_22_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 22)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_22_missing,

  -- Feature 23: Non Invasive Blood Pressure mean
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 23)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_23_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 23)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_23_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 23)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_23_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 23)) 
   LIMIT 1) AS feature_23_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 23)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_23_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 23)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_23_missing,

  -- Feature 24: Non Invasive Blood Pressure systolic
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 24)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_24_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 24)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_24_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 24)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_24_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 24)) 
   LIMIT 1) AS feature_24_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 24)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_24_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 24)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_24_missing,

  -- Feature 25: O2 saturation pulseoxymetry
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 25)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_25_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 25)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_25_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 25)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_25_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 25)) 
   LIMIT 1) AS feature_25_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 25)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_25_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 25)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_25_missing,

  -- Feature 26: PT
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 26)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_26_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 26)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_26_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 26)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_26_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 26)) 
   LIMIT 1) AS feature_26_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 26)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_26_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 26)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_26_missing,

  -- Feature 27: Potassium
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 27)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_27_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 27)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_27_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 27)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_27_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 27)) 
   LIMIT 1) AS feature_27_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 27)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_27_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 27)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_27_missing,

  -- Feature 28: RBC
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 28)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_28_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 28)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_28_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 28)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_28_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 28)) 
   LIMIT 1) AS feature_28_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 28)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_28_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 28)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_28_missing,

  -- Feature 29: RDW
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 29)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_29_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 29)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_29_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 29)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_29_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 29)) 
   LIMIT 1) AS feature_29_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 29)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_29_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 29)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_29_missing,

  -- Feature 30: Respiratory Rate
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 30)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_30_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 30)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_30_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 30)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_30_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 30)) 
   LIMIT 1) AS feature_30_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 30)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_30_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 30)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_30_missing,

  -- Feature 31: Sodium
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 31)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_31_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 31)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_31_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 31)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_31_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 31)) 
   LIMIT 1) AS feature_31_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 31)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_31_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 31)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_31_missing,

  -- Feature 32: Temperature Fahrenheit
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 32)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_32_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 32)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_32_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 32)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_32_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 32)) 
   LIMIT 1) AS feature_32_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 32)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_32_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 32)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_32_missing,

  -- Feature 33: Temperature
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 33)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_33_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 33)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_33_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 33)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_33_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 33)) 
   LIMIT 1) AS feature_33_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 33)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_33_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 33)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_33_missing,

  -- Feature 34: Tidal Volume (observed)
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 34)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_34_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 34)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_34_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 34)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_34_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 34)) 
   LIMIT 1) AS feature_34_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 34)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_34_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 34)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_34_missing,

  -- Feature 35: Urea Nitrogen
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 35)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_35_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 35)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_35_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 35)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_35_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 35)) 
   LIMIT 1) AS feature_35_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 35)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_35_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 35)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_35_missing,

  -- Feature 36: WBC
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 36)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_36_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 36)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_36_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 36)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_36_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 36)) 
   LIMIT 1) AS feature_36_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 36)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_36_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 36)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_36_missing,

  -- Feature 37: pH
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 37)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_37_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 37)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_37_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 37)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_37_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 37)) 
   LIMIT 1) AS feature_37_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 37)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_37_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 37)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_37_missing,

  -- Feature 38: pO2
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 38)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_38_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 38)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_38_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 38)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_38_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 38)) 
   LIMIT 1) AS feature_38_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 38)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_38_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 38)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_38_missing,

  -- Feature 39: Lactate Dehydrogenase (LD)
  AVG(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 39)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_39_mean,
  
  MIN(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 39)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_39_min,
  
  MAX(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 39)) 
           AND m.charttime >= pa.admittime 
           AND m.charttime < pa.obs_end_time 
           THEN m.valuenum END) AS feature_39_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.hadm_id = pa.hadm_id 
   AND lv.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 39)) 
   LIMIT 1) AS feature_39_last,
  
  COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 39)) 
             AND m.charttime >= pa.admittime 
             AND m.charttime < pa.obs_end_time 
             THEN 1 END) AS feature_39_count,
  
  CASE WHEN COUNT(CASE WHEN m.itemid = ANY((SELECT itemids FROM feature_itemids WHERE feature_idx = 39)) 
                       AND m.charttime >= pa.admittime 
                       AND m.charttime < pa.obs_end_time 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_39_missing

FROM patient_admissions pa
LEFT JOIN all_measurements m ON m.hadm_id = pa.hadm_id
GROUP BY pa.hadm_id, pa.subject_id, pa.stay_identifier, pa.admittime, pa.obs_end_time
ORDER BY pa.hadm_id;
