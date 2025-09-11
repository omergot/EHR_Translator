
-- UNIFIED eICU Feature Extraction Query for ALL Patients
-- Extracts 40 clinical features from all patient unit stays
-- Uses first 24 hours after unit admission as observation window
-- Accesses multiple eICU tables: vitalperiodic, lab, nursecharting, respiratorycharting
-- Prioritizes data accuracy and completeness

WITH patient_stays AS (
  -- Get all patient unit stays with extended time windows
  SELECT 
    p.patientunitstayid,
    p.patienthealthsystemstayid,
    p.unitadmitoffset,
    p.unitdischargeoffset,
    p.hospitaldischargeyear,
    -- Main observation window: first 24 hours
    0 as obs_start_offset,  -- Start at unit admission
    24 * 60 as obs_end_offset,  -- 24 hours after admission
    -- Extended window for comprehensive data capture
    GREATEST(-6 * 60, p.hospitaladmitoffset) as extended_start_offset,
    LEAST(48 * 60, COALESCE(p.unitdischargeoffset, 48 * 60)) as extended_end_offset
  FROM eicu_crd.patient p
  WHERE p.patientunitstayid IS NOT NULL
    AND p.unitadmitoffset IS NOT NULL
),

-- Vital signs from vitalperiodic table
vital_measurements AS (
  SELECT 
    ps.patientunitstayid,
    'heartrate' as measurement_type,
    v.heartrate as value,
    v.observationoffset as measurement_time
  FROM patient_stays ps
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = ps.patientunitstayid
  WHERE v.heartrate IS NOT NULL
    AND v.heartrate > 0
    AND v.observationoffset >= ps.extended_start_offset
    AND v.observationoffset < ps.extended_end_offset
  
  UNION ALL
  
  SELECT ps.patientunitstayid, 'respiration', v.respiration, v.observationoffset
  FROM patient_stays ps
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = ps.patientunitstayid
  WHERE v.respiration IS NOT NULL AND v.respiration > 0
    AND v.observationoffset >= ps.extended_start_offset
    AND v.observationoffset < ps.extended_end_offset
  
  UNION ALL
  
  SELECT ps.patientunitstayid, 'temperature', v.temperature, v.observationoffset
  FROM patient_stays ps
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = ps.patientunitstayid
  WHERE v.temperature IS NOT NULL AND v.temperature > 0
    AND v.observationoffset >= ps.extended_start_offset
    AND v.observationoffset < ps.extended_end_offset
  
  UNION ALL
  
  SELECT ps.patientunitstayid, 'sao2', v.sao2, v.observationoffset
  FROM patient_stays ps
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = ps.patientunitstayid
  WHERE v.sao2 IS NOT NULL AND v.sao2 > 0
    AND v.observationoffset >= ps.extended_start_offset
    AND v.observationoffset < ps.extended_end_offset
  
  UNION ALL
  
  SELECT ps.patientunitstayid, 'systolic_bp', v.systemicsystolic, v.observationoffset
  FROM patient_stays ps
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = ps.patientunitstayid
  WHERE v.systemicsystolic IS NOT NULL AND v.systemicsystolic > 0
    AND v.observationoffset >= ps.extended_start_offset
    AND v.observationoffset < ps.extended_end_offset
  
  UNION ALL
  
  SELECT ps.patientunitstayid, 'diastolic_bp', v.systemicdiastolic, v.observationoffset
  FROM patient_stays ps
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = ps.patientunitstayid
  WHERE v.systemicdiastolic IS NOT NULL AND v.systemicdiastolic > 0
    AND v.observationoffset >= ps.extended_start_offset
    AND v.observationoffset < ps.extended_end_offset
  
  UNION ALL
  
  SELECT ps.patientunitstayid, 'mean_bp', v.systemicmean, v.observationoffset
  FROM patient_stays ps
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = ps.patientunitstayid
  WHERE v.systemicmean IS NOT NULL AND v.systemicmean > 0
    AND v.observationoffset >= ps.extended_start_offset
    AND v.observationoffset < ps.extended_end_offset
),

-- Lab measurements from lab table (most comprehensive lab data)
lab_measurements AS (
  SELECT 
    ps.patientunitstayid,
    LOWER(TRIM(l.labname)) as measurement_type,
    CAST(l.labresult AS NUMERIC) as value,
    l.labresultoffset as measurement_time
  FROM patient_stays ps
  JOIN eicu_crd.lab l ON l.patientunitstayid = ps.patientunitstayid
  WHERE l.labresult IS NOT NULL
    AND l.labresult ~ '^-?[0-9]+(\\.[0-9]+)?$'  -- Only numeric results
    AND CAST(l.labresult AS NUMERIC) > 0  -- Filter out negative/zero values
    AND l.labresultoffset >= ps.extended_start_offset
    AND l.labresultoffset < ps.extended_end_offset
    AND (
      LOWER(l.labname) LIKE ANY(ARRAY[
        '%sodium%', '%potassium%', '%chloride%', '%creatinine%', '%bun%',
        '%glucose%', '%hemoglobin%', '%hematocrit%', '%wbc%', '%white blood%',
        '%rbc%', '%red blood%', '%platelet%', '%albumin%', '%bilirubin%',
        '%alt%', '%ast%', '%alkaline%', '%lactate%', '%ph%', '%pao2%',
        '%pco2%', '%lymph%', '%magnesium%', '%inr%', '%pt %', '%mcv%',
        '%mchc%', '%rdw%', '%ldh%', '%crp%'
      ])
    )
),

-- Nursing measurements (GCS, FiO2, positioning)
nursing_measurements AS (
  SELECT 
    ps.patientunitstayid,
    CASE 
      WHEN LOWER(nc.nursingchartcelltypevalname) LIKE '%glasgow%' 
           AND LOWER(nc.nursingchartcelltypevalname) LIKE '%eye%' THEN 'gcs_eye'
      WHEN LOWER(nc.nursingchartcelltypevalname) LIKE '%glasgow%' 
           AND LOWER(nc.nursingchartcelltypevalname) LIKE '%motor%' THEN 'gcs_motor'
      WHEN LOWER(nc.nursingchartcelltypevalname) LIKE '%glasgow%' 
           AND LOWER(nc.nursingchartcelltypevalname) LIKE '%verbal%' THEN 'gcs_verbal'
      WHEN LOWER(nc.nursingchartcelltypevalname) LIKE '%head%' 
           AND LOWER(nc.nursingchartcelltypevalname) LIKE '%bed%' THEN 'head_of_bed'
      WHEN LOWER(nc.nursingchartcelltypevalname) LIKE '%fio2%' THEN 'fio2'
      ELSE LOWER(TRIM(nc.nursingchartcelltypevalname))
    END as measurement_type,
    CAST(nc.nursingchartvalue AS NUMERIC) as value,
    nc.nursingchartoffset as measurement_time
  FROM patient_stays ps
  JOIN eicu_crd.nursecharting nc ON nc.patientunitstayid = ps.patientunitstayid
  WHERE nc.nursingchartvalue IS NOT NULL
    AND nc.nursingchartvalue ~ '^-?[0-9]+(\\.[0-9]+)?$'
    AND CAST(nc.nursingchartvalue AS NUMERIC) > 0
    AND nc.nursingchartoffset >= ps.extended_start_offset
    AND nc.nursingchartoffset < ps.extended_end_offset
    AND (
      LOWER(nc.nursingchartcelltypevalname) LIKE ANY(ARRAY[
        '%glasgow%', '%head%bed%', '%fio2%', '%oxygen%'
      ])
    )
),

-- Respiratory measurements (ventilator settings)
respiratory_measurements AS (
  SELECT 
    ps.patientunitstayid,
    CASE 
      WHEN LOWER(rc.respchartvaluelabel) LIKE '%fio2%' THEN 'fio2'
      WHEN LOWER(rc.respchartvaluelabel) LIKE '%peep%' THEN 'peep'
      WHEN LOWER(rc.respchartvaluelabel) LIKE '%tidal%' 
           OR LOWER(rc.respchartvaluelabel) LIKE '%vt%' THEN 'tidal_volume'
      WHEN LOWER(rc.respchartvaluelabel) LIKE '%mean%airway%' 
           OR LOWER(rc.respchartvaluelabel) LIKE '%map%' THEN 'mean_airway_pressure'
      ELSE LOWER(TRIM(rc.respchartvaluelabel))
    END as measurement_type,
    CAST(rc.respchartvalue AS NUMERIC) as value,
    rc.respchartoffset as measurement_time
  FROM patient_stays ps
  JOIN eicu_crd.respiratorycharting rc ON rc.patientunitstayid = ps.patientunitstayid
  WHERE rc.respchartvalue IS NOT NULL
    AND rc.respchartvalue ~ '^-?[0-9]+(\\.[0-9]+)?$'
    AND CAST(rc.respchartvalue AS NUMERIC) > 0
    AND rc.respchartoffset >= ps.extended_start_offset
    AND rc.respchartoffset < ps.extended_end_offset
),

-- Combine all measurements
all_measurements AS (
  SELECT * FROM vital_measurements
  UNION ALL SELECT * FROM lab_measurements
  UNION ALL SELECT * FROM nursing_measurements
  UNION ALL SELECT * FROM respiratory_measurements
),

-- Pre-compute last values for efficiency
last_values AS (
  SELECT 
    patientunitstayid, measurement_type,
    FIRST_VALUE(value) OVER (
      PARTITION BY patientunitstayid, measurement_type 
      ORDER BY measurement_time DESC 
      ROWS UNBOUNDED PRECEDING
    ) as last_value
  FROM all_measurements
  WHERE measurement_time >= 0 AND measurement_time < 24 * 60
),

-- Feature mappings
feature_mappings AS (
    SELECT 0 as feature_idx, 'ALT (SGPT)' as feature_name, ARRAY['alt', 'alanine aminotransferase', 'sgpt'] as measurement_types
  UNION ALL
    SELECT 1 as feature_idx, 'albumin' as feature_name, ARRAY['albumin'] as measurement_types
  UNION ALL
    SELECT 2 as feature_idx, 'alkaline phos.' as feature_name, ARRAY['alkaline phosphatase', 'alk phos'] as measurement_types
  UNION ALL
    SELECT 3 as feature_idx, 'AST (SGOT)' as feature_name, ARRAY['ast', 'aspartate aminotransferase', 'sgot'] as measurement_types
  UNION ALL
    SELECT 4 as feature_idx, 'total bilirubin' as feature_name, ARRAY['bilirubin', 'total bilirubin'] as measurement_types
  UNION ALL
    SELECT 5 as feature_idx, 'CRP' as feature_name, ARRAY['crp', 'c-reactive protein'] as measurement_types
  UNION ALL
    SELECT 6 as feature_idx, 'creatinine' as feature_name, ARRAY['creatinine'] as measurement_types
  UNION ALL
    SELECT 7 as feature_idx, 'Eyes' as feature_name, ARRAY['gcs_eye', 'glasgow coma scale eye'] as measurement_types
  UNION ALL
    SELECT 8 as feature_idx, 'Motor' as feature_name, ARRAY['gcs_motor', 'glasgow coma scale motor'] as measurement_types
  UNION ALL
    SELECT 9 as feature_idx, 'Verbal' as feature_name, ARRAY['gcs_verbal', 'glasgow coma scale verbal'] as measurement_types
  UNION ALL
    SELECT 10 as feature_idx, 'Head of Bed Elevation' as feature_name, ARRAY['head_of_bed', 'hob'] as measurement_types
  UNION ALL
    SELECT 11 as feature_idx, 'Heart Rate' as feature_name, ARRAY['heartrate', 'heart rate', 'hr'] as measurement_types
  UNION ALL
    SELECT 12 as feature_idx, 'Hct' as feature_name, ARRAY['hematocrit', 'hct'] as measurement_types
  UNION ALL
    SELECT 13 as feature_idx, 'Hgb' as feature_name, ARRAY['hemoglobin', 'hgb', 'hb'] as measurement_types
  UNION ALL
    SELECT 14 as feature_idx, 'PT - INR' as feature_name, ARRAY['inr', 'pt inr'] as measurement_types
  UNION ALL
    SELECT 15 as feature_idx, 'FiO2' as feature_name, ARRAY['fio2', 'inspired oxygen'] as measurement_types
  UNION ALL
    SELECT 16 as feature_idx, 'lactate' as feature_name, ARRAY['lactate', 'lactic acid'] as measurement_types
  UNION ALL
    SELECT 17 as feature_idx, '-lymphs' as feature_name, ARRAY['lymphocytes', 'lymphs', 'lymph'] as measurement_types
  UNION ALL
    SELECT 18 as feature_idx, 'MCHC' as feature_name, ARRAY['mchc'] as measurement_types
  UNION ALL
    SELECT 19 as feature_idx, 'MCV' as feature_name, ARRAY['mcv'] as measurement_types
  UNION ALL
    SELECT 20 as feature_idx, 'magnesium' as feature_name, ARRAY['magnesium', 'mg'] as measurement_types
  UNION ALL
    SELECT 21 as feature_idx, 'Mean Airway Pressure' as feature_name, ARRAY['mean_airway_pressure', 'map'] as measurement_types
  UNION ALL
    SELECT 22 as feature_idx, 'Non-Invasive BP Diastolic' as feature_name, ARRAY['diastolic_bp', 'systemicdiastolic'] as measurement_types
  UNION ALL
    SELECT 23 as feature_idx, 'Non-Invasive BP Mean' as feature_name, ARRAY['mean_bp', 'systemicmean'] as measurement_types
  UNION ALL
    SELECT 24 as feature_idx, 'Non-Invasive BP Systolic' as feature_name, ARRAY['systolic_bp', 'systemicsystolic'] as measurement_types
  UNION ALL
    SELECT 25 as feature_idx, 'O2 Sat (%)' as feature_name, ARRAY['sao2', 'o2 sat', 'oxygen saturation'] as measurement_types
  UNION ALL
    SELECT 26 as feature_idx, 'PT' as feature_name, ARRAY['pt', 'prothrombin time'] as measurement_types
  UNION ALL
    SELECT 27 as feature_idx, 'potassium' as feature_name, ARRAY['potassium', 'k'] as measurement_types
  UNION ALL
    SELECT 28 as feature_idx, 'RBC' as feature_name, ARRAY['rbc', 'red blood cells'] as measurement_types
  UNION ALL
    SELECT 29 as feature_idx, 'RDW' as feature_name, ARRAY['rdw'] as measurement_types
  UNION ALL
    SELECT 30 as feature_idx, 'Respiratory Rate' as feature_name, ARRAY['respiration', 'respiratory rate', 'rr'] as measurement_types
  UNION ALL
    SELECT 31 as feature_idx, 'sodium' as feature_name, ARRAY['sodium', 'na'] as measurement_types
  UNION ALL
    SELECT 32 as feature_idx, 'Temperature (F)' as feature_name, ARRAY['temperature'] as measurement_types
  UNION ALL
    SELECT 33 as feature_idx, 'Temperature' as feature_name, ARRAY['temperature'] as measurement_types
  UNION ALL
    SELECT 34 as feature_idx, 'Exhaled TV (patient)' as feature_name, ARRAY['tidal_volume', 'vt'] as measurement_types
  UNION ALL
    SELECT 35 as feature_idx, 'BUN' as feature_name, ARRAY['bun', 'blood urea nitrogen', 'urea nitrogen'] as measurement_types
  UNION ALL
    SELECT 36 as feature_idx, 'WBC x 1000' as feature_name, ARRAY['wbc', 'white blood cells'] as measurement_types
  UNION ALL
    SELECT 37 as feature_idx, 'pH' as feature_name, ARRAY['ph'] as measurement_types
  UNION ALL
    SELECT 38 as feature_idx, 'paO2' as feature_name, ARRAY['pao2', 'po2', 'oxygen partial pressure'] as measurement_types
  UNION ALL
    SELECT 39 as feature_idx, 'LDH' as feature_name, ARRAY['ldh', 'lactate dehydrogenase'] as measurement_types
)

-- Main feature extraction with comprehensive statistics
SELECT
  ps.patientunitstayid as icustay_id,
  ps.patienthealthsystemstayid,
  -- Feature 0: ALT (SGPT)
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 0)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_0_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 0)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_0_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 0)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_0_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 0)) 
   LIMIT 1) AS feature_0_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 0)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_0_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 0)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_0_missing,

  -- Feature 1: albumin
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 1)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_1_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 1)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_1_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 1)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_1_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 1)) 
   LIMIT 1) AS feature_1_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 1)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_1_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 1)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_1_missing,

  -- Feature 2: alkaline phos.
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 2)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_2_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 2)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_2_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 2)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_2_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 2)) 
   LIMIT 1) AS feature_2_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 2)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_2_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 2)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_2_missing,

  -- Feature 3: AST (SGOT)
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 3)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_3_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 3)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_3_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 3)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_3_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 3)) 
   LIMIT 1) AS feature_3_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 3)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_3_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 3)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_3_missing,

  -- Feature 4: total bilirubin
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 4)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_4_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 4)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_4_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 4)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_4_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 4)) 
   LIMIT 1) AS feature_4_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 4)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_4_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 4)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_4_missing,

  -- Feature 5: CRP
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 5)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_5_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 5)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_5_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 5)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_5_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 5)) 
   LIMIT 1) AS feature_5_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 5)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_5_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 5)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_5_missing,

  -- Feature 6: creatinine
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 6)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_6_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 6)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_6_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 6)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_6_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 6)) 
   LIMIT 1) AS feature_6_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 6)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_6_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 6)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_6_missing,

  -- Feature 7: Eyes
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 7)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_7_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 7)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_7_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 7)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_7_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 7)) 
   LIMIT 1) AS feature_7_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 7)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_7_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 7)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_7_missing,

  -- Feature 8: Motor
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 8)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_8_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 8)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_8_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 8)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_8_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 8)) 
   LIMIT 1) AS feature_8_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 8)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_8_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 8)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_8_missing,

  -- Feature 9: Verbal
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 9)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_9_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 9)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_9_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 9)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_9_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 9)) 
   LIMIT 1) AS feature_9_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 9)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_9_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 9)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_9_missing,

  -- Feature 10: Head of Bed Elevation
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 10)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_10_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 10)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_10_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 10)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_10_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 10)) 
   LIMIT 1) AS feature_10_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 10)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_10_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 10)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_10_missing,

  -- Feature 11: Heart Rate
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 11)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_11_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 11)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_11_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 11)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_11_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 11)) 
   LIMIT 1) AS feature_11_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 11)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_11_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 11)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_11_missing,

  -- Feature 12: Hct
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 12)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_12_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 12)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_12_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 12)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_12_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 12)) 
   LIMIT 1) AS feature_12_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 12)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_12_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 12)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_12_missing,

  -- Feature 13: Hgb
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 13)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_13_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 13)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_13_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 13)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_13_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 13)) 
   LIMIT 1) AS feature_13_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 13)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_13_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 13)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_13_missing,

  -- Feature 14: PT - INR
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 14)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_14_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 14)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_14_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 14)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_14_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 14)) 
   LIMIT 1) AS feature_14_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 14)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_14_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 14)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_14_missing,

  -- Feature 15: FiO2
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 15)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_15_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 15)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_15_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 15)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_15_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 15)) 
   LIMIT 1) AS feature_15_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 15)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_15_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 15)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_15_missing,

  -- Feature 16: lactate
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 16)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_16_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 16)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_16_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 16)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_16_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 16)) 
   LIMIT 1) AS feature_16_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 16)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_16_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 16)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_16_missing,

  -- Feature 17: -lymphs
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 17)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_17_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 17)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_17_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 17)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_17_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 17)) 
   LIMIT 1) AS feature_17_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 17)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_17_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 17)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_17_missing,

  -- Feature 18: MCHC
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 18)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_18_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 18)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_18_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 18)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_18_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 18)) 
   LIMIT 1) AS feature_18_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 18)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_18_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 18)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_18_missing,

  -- Feature 19: MCV
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 19)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_19_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 19)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_19_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 19)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_19_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 19)) 
   LIMIT 1) AS feature_19_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 19)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_19_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 19)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_19_missing,

  -- Feature 20: magnesium
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 20)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_20_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 20)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_20_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 20)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_20_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 20)) 
   LIMIT 1) AS feature_20_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 20)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_20_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 20)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_20_missing,

  -- Feature 21: Mean Airway Pressure
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 21)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_21_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 21)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_21_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 21)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_21_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 21)) 
   LIMIT 1) AS feature_21_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 21)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_21_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 21)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_21_missing,

  -- Feature 22: Non-Invasive BP Diastolic
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 22)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_22_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 22)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_22_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 22)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_22_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 22)) 
   LIMIT 1) AS feature_22_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 22)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_22_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 22)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_22_missing,

  -- Feature 23: Non-Invasive BP Mean
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 23)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_23_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 23)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_23_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 23)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_23_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 23)) 
   LIMIT 1) AS feature_23_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 23)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_23_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 23)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_23_missing,

  -- Feature 24: Non-Invasive BP Systolic
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 24)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_24_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 24)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_24_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 24)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_24_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 24)) 
   LIMIT 1) AS feature_24_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 24)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_24_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 24)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_24_missing,

  -- Feature 25: O2 Sat (%)
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 25)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_25_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 25)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_25_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 25)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_25_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 25)) 
   LIMIT 1) AS feature_25_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 25)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_25_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 25)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_25_missing,

  -- Feature 26: PT
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 26)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_26_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 26)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_26_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 26)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_26_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 26)) 
   LIMIT 1) AS feature_26_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 26)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_26_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 26)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_26_missing,

  -- Feature 27: potassium
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 27)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_27_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 27)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_27_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 27)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_27_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 27)) 
   LIMIT 1) AS feature_27_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 27)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_27_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 27)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_27_missing,

  -- Feature 28: RBC
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 28)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_28_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 28)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_28_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 28)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_28_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 28)) 
   LIMIT 1) AS feature_28_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 28)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_28_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 28)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_28_missing,

  -- Feature 29: RDW
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 29)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_29_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 29)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_29_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 29)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_29_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 29)) 
   LIMIT 1) AS feature_29_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 29)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_29_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 29)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_29_missing,

  -- Feature 30: Respiratory Rate
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 30)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_30_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 30)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_30_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 30)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_30_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 30)) 
   LIMIT 1) AS feature_30_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 30)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_30_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 30)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_30_missing,

  -- Feature 31: sodium
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 31)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_31_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 31)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_31_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 31)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_31_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 31)) 
   LIMIT 1) AS feature_31_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 31)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_31_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 31)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_31_missing,

  -- Feature 32: Temperature (F)
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 32)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_32_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 32)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_32_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 32)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_32_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 32)) 
   LIMIT 1) AS feature_32_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 32)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_32_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 32)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_32_missing,

  -- Feature 33: Temperature
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 33)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_33_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 33)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_33_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 33)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_33_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 33)) 
   LIMIT 1) AS feature_33_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 33)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_33_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 33)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_33_missing,

  -- Feature 34: Exhaled TV (patient)
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 34)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_34_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 34)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_34_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 34)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_34_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 34)) 
   LIMIT 1) AS feature_34_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 34)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_34_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 34)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_34_missing,

  -- Feature 35: BUN
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 35)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_35_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 35)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_35_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 35)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_35_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 35)) 
   LIMIT 1) AS feature_35_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 35)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_35_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 35)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_35_missing,

  -- Feature 36: WBC x 1000
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 36)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_36_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 36)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_36_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 36)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_36_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 36)) 
   LIMIT 1) AS feature_36_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 36)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_36_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 36)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_36_missing,

  -- Feature 37: pH
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 37)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_37_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 37)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_37_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 37)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_37_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 37)) 
   LIMIT 1) AS feature_37_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 37)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_37_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 37)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_37_missing,

  -- Feature 38: paO2
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 38)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_38_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 38)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_38_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 38)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_38_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 38)) 
   LIMIT 1) AS feature_38_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 38)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_38_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 38)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_38_missing,

  -- Feature 39: LDH
  AVG(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 39)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_39_mean,
  
  MIN(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 39)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_39_min,
  
  MAX(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 39)) 
           AND m.measurement_time >= ps.obs_start_offset 
           AND m.measurement_time < ps.obs_end_offset 
           THEN m.value END) AS feature_39_max,
  
  (SELECT lv.last_value FROM last_values lv 
   WHERE lv.patientunitstayid = ps.patientunitstayid 
   AND lv.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 39)) 
   LIMIT 1) AS feature_39_last,
  
  COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 39)) 
             AND m.measurement_time >= ps.obs_start_offset 
             AND m.measurement_time < ps.obs_end_offset 
             THEN 1 END) AS feature_39_count,
  
  CASE WHEN COUNT(CASE WHEN m.measurement_type = ANY((SELECT measurement_types FROM feature_mappings WHERE feature_idx = 39)) 
                       AND m.measurement_time >= ps.obs_start_offset 
                       AND m.measurement_time < ps.obs_end_offset 
                       THEN 1 END) = 0 
       THEN 1 ELSE 0 END AS feature_39_missing

FROM patient_stays ps
LEFT JOIN all_measurements m ON m.patientunitstayid = ps.patientunitstayid
GROUP BY ps.patientunitstayid, ps.patienthealthsystemstayid, ps.obs_start_offset, ps.obs_end_offset
ORDER BY ps.patientunitstayid;
