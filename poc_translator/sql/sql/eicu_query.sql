-- OPTIMIZED eICU Feature Extraction Query
-- Includes multiple data sources for comprehensive coverage
-- Fixes the missing data issue by accessing lab, nursing, and respiratory tables

WITH cohort_patients AS (
  SELECT DISTINCT example_id as patientunitstayid
  FROM eicu_bsi_100_2h_test.__eicu_bsi_100_2h_cohort
),

patient_stays AS (
  SELECT 
    p.patientunitstayid,
    p.patienthealthsystemstayid,
    p.hospitaladmitoffset,
    p.unitdischargeoffset,
    -- Extended time window for more comprehensive data
    GREATEST(p.hospitaladmitoffset, -6*60) as extended_start,  -- 6 hours before or hospital admit
    LEAST(p.unitdischargeoffset, 48*60) as extended_end        -- 48 hours or discharge
  FROM eicu_crd.patient p
  JOIN cohort_patients c ON p.patientunitstayid = c.patientunitstayid
  WHERE p.patientunitstayid IS NOT NULL
),

-- Comprehensive data extraction from multiple tables
all_measurements AS (
  -- Vital signs from vitalperiodic
  SELECT 
    p.patientunitstayid,
    'heartrate' as measurement_type,
    v.heartrate as value,
    v.observationoffset as measurement_time
  FROM patient_stays p
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = p.patientunitstayid
  WHERE v.heartrate IS NOT NULL
    AND v.observationoffset >= p.extended_start
    AND v.observationoffset < p.extended_end
  
  UNION ALL
  
  SELECT p.patientunitstayid, 'respiration', v.respiration, v.observationoffset
  FROM patient_stays p
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = p.patientunitstayid
  WHERE v.respiration IS NOT NULL
    AND v.observationoffset >= p.extended_start
    AND v.observationoffset < p.extended_end
  
  UNION ALL
  
  SELECT p.patientunitstayid, 'temperature', v.temperature, v.observationoffset
  FROM patient_stays p
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = p.patientunitstayid
  WHERE v.temperature IS NOT NULL
    AND v.observationoffset >= p.extended_start
    AND v.observationoffset < p.extended_end
  
  UNION ALL
  
  SELECT p.patientunitstayid, 'sao2', v.sao2, v.observationoffset
  FROM patient_stays p
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = p.patientunitstayid
  WHERE v.sao2 IS NOT NULL
    AND v.observationoffset >= p.extended_start
    AND v.observationoffset < p.extended_end
  
  UNION ALL
  
  SELECT p.patientunitstayid, 'systolic_bp', v.systemicsystolic, v.observationoffset
  FROM patient_stays p
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = p.patientunitstayid
  WHERE v.systemicsystolic IS NOT NULL
    AND v.observationoffset >= p.extended_start
    AND v.observationoffset < p.extended_end
  
  UNION ALL
  
  SELECT p.patientunitstayid, 'diastolic_bp', v.systemicdiastolic, v.observationoffset
  FROM patient_stays p
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = p.patientunitstayid
  WHERE v.systemicdiastolic IS NOT NULL
    AND v.observationoffset >= p.extended_start
    AND v.observationoffset < p.extended_end
  
  UNION ALL
  
  SELECT p.patientunitstayid, 'mean_bp', v.systemicmean, v.observationoffset
  FROM patient_stays p
  JOIN eicu_crd.vitalperiodic v ON v.patientunitstayid = p.patientunitstayid
  WHERE v.systemicmean IS NOT NULL
    AND v.observationoffset >= p.extended_start
    AND v.observationoffset < p.extended_end
  
  UNION ALL
  
  -- Lab values from lab table
  SELECT 
    p.patientunitstayid,
    LOWER(l.labname) as measurement_type,
    CAST(l.labresult AS NUMERIC) as value,
    l.labresultoffset as measurement_time
  FROM patient_stays p
  JOIN eicu_crd.lab l ON l.patientunitstayid = p.patientunitstayid
  WHERE l.labresult IS NOT NULL
    AND l.labresult ~ '^-?[0-9]+(\.[0-9]+)?$'  -- Only numeric results
    AND l.labresultoffset >= p.extended_start
    AND l.labresultoffset < p.extended_end
  
  UNION ALL
  
  -- Nursing data for GCS and other measurements
  SELECT 
    p.patientunitstayid,
    CASE 
      WHEN LOWER(nc.nursingchartcelltypevalname) LIKE '%glasgow%' AND LOWER(nc.nursingchartcelltypevalname) LIKE '%eye%' THEN 'gcs_eye'
      WHEN LOWER(nc.nursingchartcelltypevalname) LIKE '%glasgow%' AND LOWER(nc.nursingchartcelltypevalname) LIKE '%motor%' THEN 'gcs_motor'
      WHEN LOWER(nc.nursingchartcelltypevalname) LIKE '%glasgow%' AND LOWER(nc.nursingchartcelltypevalname) LIKE '%verbal%' THEN 'gcs_verbal'
      WHEN LOWER(nc.nursingchartcelltypevalname) LIKE '%head%' AND LOWER(nc.nursingchartcelltypevalname) LIKE '%bed%' THEN 'head_of_bed'
      WHEN LOWER(nc.nursingchartcelltypevalname) LIKE '%fio2%' THEN 'fio2'
      ELSE LOWER(nc.nursingchartcelltypevalname)
    END as measurement_type,
    CAST(nc.nursingchartvalue AS NUMERIC) as value,
    nc.nursingchartoffset as measurement_time
  FROM patient_stays p
  JOIN eicu_crd.nursecharting nc ON nc.patientunitstayid = p.patientunitstayid
  WHERE nc.nursingchartvalue IS NOT NULL
    AND nc.nursingchartvalue ~ '^-?[0-9]+(\.[0-9]+)?$'
    AND nc.nursingchartoffset >= p.extended_start
    AND nc.nursingchartoffset < p.extended_end
  
  UNION ALL
  
  -- Respiratory data for ventilation parameters
  SELECT 
    p.patientunitstayid,
    CASE 
      WHEN LOWER(rc.respchartvaluelabel) LIKE '%fio2%' THEN 'fio2'
      WHEN LOWER(rc.respchartvaluelabel) LIKE '%peep%' THEN 'peep'
      WHEN LOWER(rc.respchartvaluelabel) LIKE '%tidal%' OR LOWER(rc.respchartvaluelabel) LIKE '%vt%' THEN 'tidal_volume'
      WHEN LOWER(rc.respchartvaluelabel) LIKE '%mean%airway%' OR LOWER(rc.respchartvaluelabel) LIKE '%map%' THEN 'mean_airway_pressure'
      ELSE LOWER(rc.respchartvaluelabel)
    END as measurement_type,
    CAST(rc.respchartvalue AS NUMERIC) as value,
    rc.respchartoffset as measurement_time
  FROM patient_stays p
  JOIN eicu_crd.respiratorycharting rc ON rc.patientunitstayid = p.patientunitstayid
  WHERE rc.respchartvalue IS NOT NULL
    AND rc.respchartvalue ~ '^-?[0-9]+(\.[0-9]+)?$'
    AND rc.respchartoffset >= p.extended_start
    AND rc.respchartoffset < p.extended_end
),

-- Calculate last values efficiently using window functions
last_values AS (
  SELECT 
    patientunitstayid, measurement_type,
    FIRST_VALUE(value) OVER (
      PARTITION BY patientunitstayid, measurement_type 
      ORDER BY measurement_time DESC 
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as last_value,
    ROW_NUMBER() OVER (PARTITION BY patientunitstayid, measurement_type ORDER BY measurement_time DESC) as rn
  FROM all_measurements
  WHERE measurement_time >= 0 AND measurement_time < 24 * 60  -- 24 hours
)

SELECT
  p.patientunitstayid as icustay_id,

  
  -- Feature 0: ALT
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%alt%' OR m.measurement_type LIKE '%alanine aminotransferase%' OR m.measurement_type LIKE '%sgpt%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_0_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%alt%' OR m.measurement_type LIKE '%alanine aminotransferase%' OR m.measurement_type LIKE '%sgpt%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_0_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%alt%' OR m.measurement_type LIKE '%alanine aminotransferase%' OR m.measurement_type LIKE '%sgpt%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_0_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%alt%' OR lv.measurement_type LIKE '%alanine aminotransferase%' OR lv.measurement_type LIKE '%sgpt%')
   AND lv.rn = 1 LIMIT 1) AS feature_0_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%alt%' OR m.measurement_type LIKE '%alanine aminotransferase%' OR m.measurement_type LIKE '%sgpt%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_0_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%alt%' OR m.measurement_type LIKE '%alanine aminotransferase%' OR m.measurement_type LIKE '%sgpt%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_0_missing,

  
  -- Feature 1: Albumin
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%albumin%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_1_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%albumin%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_1_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%albumin%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_1_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%albumin%')
   AND lv.rn = 1 LIMIT 1) AS feature_1_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%albumin%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_1_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%albumin%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_1_missing,

  
  -- Feature 2: Alkaline Phosphatase
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%alkaline phosphatase%' OR m.measurement_type LIKE '%alk phos%' OR m.measurement_type LIKE '%alkphos%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_2_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%alkaline phosphatase%' OR m.measurement_type LIKE '%alk phos%' OR m.measurement_type LIKE '%alkphos%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_2_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%alkaline phosphatase%' OR m.measurement_type LIKE '%alk phos%' OR m.measurement_type LIKE '%alkphos%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_2_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%alkaline phosphatase%' OR lv.measurement_type LIKE '%alk phos%' OR lv.measurement_type LIKE '%alkphos%')
   AND lv.rn = 1 LIMIT 1) AS feature_2_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%alkaline phosphatase%' OR m.measurement_type LIKE '%alk phos%' OR m.measurement_type LIKE '%alkphos%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_2_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%alkaline phosphatase%' OR m.measurement_type LIKE '%alk phos%' OR m.measurement_type LIKE '%alkphos%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_2_missing,

  
  -- Feature 3: AST
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ast%' OR m.measurement_type LIKE '%aspartate aminotransferase%' OR m.measurement_type LIKE '%sgot%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_3_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ast%' OR m.measurement_type LIKE '%aspartate aminotransferase%' OR m.measurement_type LIKE '%sgot%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_3_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ast%' OR m.measurement_type LIKE '%aspartate aminotransferase%' OR m.measurement_type LIKE '%sgot%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_3_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%ast%' OR lv.measurement_type LIKE '%aspartate aminotransferase%' OR lv.measurement_type LIKE '%sgot%')
   AND lv.rn = 1 LIMIT 1) AS feature_3_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ast%' OR m.measurement_type LIKE '%aspartate aminotransferase%' OR m.measurement_type LIKE '%sgot%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_3_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%ast%' OR m.measurement_type LIKE '%aspartate aminotransferase%' OR m.measurement_type LIKE '%sgot%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_3_missing,

  
  -- Feature 4: Bilirubin
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%bilirubin%' OR m.measurement_type LIKE '%total bilirubin%' OR m.measurement_type LIKE '%bili%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_4_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%bilirubin%' OR m.measurement_type LIKE '%total bilirubin%' OR m.measurement_type LIKE '%bili%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_4_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%bilirubin%' OR m.measurement_type LIKE '%total bilirubin%' OR m.measurement_type LIKE '%bili%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_4_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%bilirubin%' OR lv.measurement_type LIKE '%total bilirubin%' OR lv.measurement_type LIKE '%bili%')
   AND lv.rn = 1 LIMIT 1) AS feature_4_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%bilirubin%' OR m.measurement_type LIKE '%total bilirubin%' OR m.measurement_type LIKE '%bili%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_4_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%bilirubin%' OR m.measurement_type LIKE '%total bilirubin%' OR m.measurement_type LIKE '%bili%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_4_missing,

  
  -- Feature 5: C-Reactive Protein
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%crp%' OR m.measurement_type LIKE '%c-reactive protein%' OR m.measurement_type LIKE '%c reactive protein%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_5_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%crp%' OR m.measurement_type LIKE '%c-reactive protein%' OR m.measurement_type LIKE '%c reactive protein%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_5_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%crp%' OR m.measurement_type LIKE '%c-reactive protein%' OR m.measurement_type LIKE '%c reactive protein%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_5_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%crp%' OR lv.measurement_type LIKE '%c-reactive protein%' OR lv.measurement_type LIKE '%c reactive protein%')
   AND lv.rn = 1 LIMIT 1) AS feature_5_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%crp%' OR m.measurement_type LIKE '%c-reactive protein%' OR m.measurement_type LIKE '%c reactive protein%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_5_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%crp%' OR m.measurement_type LIKE '%c-reactive protein%' OR m.measurement_type LIKE '%c reactive protein%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_5_missing,

  
  -- Feature 6: Creatinine
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%creatinine%' OR m.measurement_type LIKE '%creat%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_6_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%creatinine%' OR m.measurement_type LIKE '%creat%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_6_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%creatinine%' OR m.measurement_type LIKE '%creat%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_6_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%creatinine%' OR lv.measurement_type LIKE '%creat%')
   AND lv.rn = 1 LIMIT 1) AS feature_6_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%creatinine%' OR m.measurement_type LIKE '%creat%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_6_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%creatinine%' OR m.measurement_type LIKE '%creat%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_6_missing,

  
  -- Feature 7: GCS - Eye Opening
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_eye%' OR m.measurement_type LIKE '%glasgow coma scale eye%' OR m.measurement_type LIKE '%glasgow eye%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_7_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_eye%' OR m.measurement_type LIKE '%glasgow coma scale eye%' OR m.measurement_type LIKE '%glasgow eye%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_7_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_eye%' OR m.measurement_type LIKE '%glasgow coma scale eye%' OR m.measurement_type LIKE '%glasgow eye%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_7_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%gcs_eye%' OR lv.measurement_type LIKE '%glasgow coma scale eye%' OR lv.measurement_type LIKE '%glasgow eye%')
   AND lv.rn = 1 LIMIT 1) AS feature_7_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_eye%' OR m.measurement_type LIKE '%glasgow coma scale eye%' OR m.measurement_type LIKE '%glasgow eye%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_7_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%gcs_eye%' OR m.measurement_type LIKE '%glasgow coma scale eye%' OR m.measurement_type LIKE '%glasgow eye%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_7_missing,

  
  -- Feature 8: GCS - Motor Response
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_motor%' OR m.measurement_type LIKE '%glasgow coma scale motor%' OR m.measurement_type LIKE '%glasgow motor%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_8_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_motor%' OR m.measurement_type LIKE '%glasgow coma scale motor%' OR m.measurement_type LIKE '%glasgow motor%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_8_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_motor%' OR m.measurement_type LIKE '%glasgow coma scale motor%' OR m.measurement_type LIKE '%glasgow motor%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_8_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%gcs_motor%' OR lv.measurement_type LIKE '%glasgow coma scale motor%' OR lv.measurement_type LIKE '%glasgow motor%')
   AND lv.rn = 1 LIMIT 1) AS feature_8_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_motor%' OR m.measurement_type LIKE '%glasgow coma scale motor%' OR m.measurement_type LIKE '%glasgow motor%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_8_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%gcs_motor%' OR m.measurement_type LIKE '%glasgow coma scale motor%' OR m.measurement_type LIKE '%glasgow motor%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_8_missing,

  
  -- Feature 9: GCS - Verbal Response
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_verbal%' OR m.measurement_type LIKE '%glasgow coma scale verbal%' OR m.measurement_type LIKE '%glasgow verbal%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_9_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_verbal%' OR m.measurement_type LIKE '%glasgow coma scale verbal%' OR m.measurement_type LIKE '%glasgow verbal%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_9_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_verbal%' OR m.measurement_type LIKE '%glasgow coma scale verbal%' OR m.measurement_type LIKE '%glasgow verbal%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_9_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%gcs_verbal%' OR lv.measurement_type LIKE '%glasgow coma scale verbal%' OR lv.measurement_type LIKE '%glasgow verbal%')
   AND lv.rn = 1 LIMIT 1) AS feature_9_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%gcs_verbal%' OR m.measurement_type LIKE '%glasgow coma scale verbal%' OR m.measurement_type LIKE '%glasgow verbal%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_9_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%gcs_verbal%' OR m.measurement_type LIKE '%glasgow coma scale verbal%' OR m.measurement_type LIKE '%glasgow verbal%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_9_missing,

  
  -- Feature 10: Head of Bed
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%head_of_bed%' OR m.measurement_type LIKE '%hob%' OR m.measurement_type LIKE '%head of bed elevation%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_10_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%head_of_bed%' OR m.measurement_type LIKE '%hob%' OR m.measurement_type LIKE '%head of bed elevation%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_10_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%head_of_bed%' OR m.measurement_type LIKE '%hob%' OR m.measurement_type LIKE '%head of bed elevation%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_10_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%head_of_bed%' OR lv.measurement_type LIKE '%hob%' OR lv.measurement_type LIKE '%head of bed elevation%')
   AND lv.rn = 1 LIMIT 1) AS feature_10_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%head_of_bed%' OR m.measurement_type LIKE '%hob%' OR m.measurement_type LIKE '%head of bed elevation%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_10_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%head_of_bed%' OR m.measurement_type LIKE '%hob%' OR m.measurement_type LIKE '%head of bed elevation%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_10_missing,

  
  -- Feature 11: Heart Rate
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%heartrate%' OR m.measurement_type LIKE '%heart rate%' OR m.measurement_type LIKE '%hr%' OR m.measurement_type LIKE '%pulse%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_11_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%heartrate%' OR m.measurement_type LIKE '%heart rate%' OR m.measurement_type LIKE '%hr%' OR m.measurement_type LIKE '%pulse%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_11_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%heartrate%' OR m.measurement_type LIKE '%heart rate%' OR m.measurement_type LIKE '%hr%' OR m.measurement_type LIKE '%pulse%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_11_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%heartrate%' OR lv.measurement_type LIKE '%heart rate%' OR lv.measurement_type LIKE '%hr%' OR lv.measurement_type LIKE '%pulse%')
   AND lv.rn = 1 LIMIT 1) AS feature_11_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%heartrate%' OR m.measurement_type LIKE '%heart rate%' OR m.measurement_type LIKE '%hr%' OR m.measurement_type LIKE '%pulse%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_11_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%heartrate%' OR m.measurement_type LIKE '%heart rate%' OR m.measurement_type LIKE '%hr%' OR m.measurement_type LIKE '%pulse%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_11_missing,

  
  -- Feature 12: Hematocrit
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%hematocrit%' OR m.measurement_type LIKE '%hct%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_12_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%hematocrit%' OR m.measurement_type LIKE '%hct%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_12_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%hematocrit%' OR m.measurement_type LIKE '%hct%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_12_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%hematocrit%' OR lv.measurement_type LIKE '%hct%')
   AND lv.rn = 1 LIMIT 1) AS feature_12_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%hematocrit%' OR m.measurement_type LIKE '%hct%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_12_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%hematocrit%' OR m.measurement_type LIKE '%hct%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_12_missing,

  
  -- Feature 13: Hemoglobin
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%hemoglobin%' OR m.measurement_type LIKE '%hgb%' OR m.measurement_type LIKE '%hb%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_13_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%hemoglobin%' OR m.measurement_type LIKE '%hgb%' OR m.measurement_type LIKE '%hb%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_13_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%hemoglobin%' OR m.measurement_type LIKE '%hgb%' OR m.measurement_type LIKE '%hb%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_13_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%hemoglobin%' OR lv.measurement_type LIKE '%hgb%' OR lv.measurement_type LIKE '%hb%')
   AND lv.rn = 1 LIMIT 1) AS feature_13_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%hemoglobin%' OR m.measurement_type LIKE '%hgb%' OR m.measurement_type LIKE '%hb%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_13_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%hemoglobin%' OR m.measurement_type LIKE '%hgb%' OR m.measurement_type LIKE '%hb%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_13_missing,

  
  -- Feature 14: INR
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%inr%' OR m.measurement_type LIKE '%pt inr%' OR m.measurement_type LIKE '%pt/inr%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_14_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%inr%' OR m.measurement_type LIKE '%pt inr%' OR m.measurement_type LIKE '%pt/inr%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_14_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%inr%' OR m.measurement_type LIKE '%pt inr%' OR m.measurement_type LIKE '%pt/inr%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_14_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%inr%' OR lv.measurement_type LIKE '%pt inr%' OR lv.measurement_type LIKE '%pt/inr%')
   AND lv.rn = 1 LIMIT 1) AS feature_14_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%inr%' OR m.measurement_type LIKE '%pt inr%' OR m.measurement_type LIKE '%pt/inr%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_14_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%inr%' OR m.measurement_type LIKE '%pt inr%' OR m.measurement_type LIKE '%pt/inr%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_14_missing,

  
  -- Feature 15: FiO2
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%fio2%' OR m.measurement_type LIKE '%inspired oxygen%' OR m.measurement_type LIKE '%fraction inspired oxygen%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_15_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%fio2%' OR m.measurement_type LIKE '%inspired oxygen%' OR m.measurement_type LIKE '%fraction inspired oxygen%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_15_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%fio2%' OR m.measurement_type LIKE '%inspired oxygen%' OR m.measurement_type LIKE '%fraction inspired oxygen%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_15_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%fio2%' OR lv.measurement_type LIKE '%inspired oxygen%' OR lv.measurement_type LIKE '%fraction inspired oxygen%')
   AND lv.rn = 1 LIMIT 1) AS feature_15_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%fio2%' OR m.measurement_type LIKE '%inspired oxygen%' OR m.measurement_type LIKE '%fraction inspired oxygen%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_15_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%fio2%' OR m.measurement_type LIKE '%inspired oxygen%' OR m.measurement_type LIKE '%fraction inspired oxygen%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_15_missing,

  
  -- Feature 16: Lactate
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%lactate%' OR m.measurement_type LIKE '%lactic acid%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_16_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%lactate%' OR m.measurement_type LIKE '%lactic acid%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_16_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%lactate%' OR m.measurement_type LIKE '%lactic acid%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_16_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%lactate%' OR lv.measurement_type LIKE '%lactic acid%')
   AND lv.rn = 1 LIMIT 1) AS feature_16_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%lactate%' OR m.measurement_type LIKE '%lactic acid%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_16_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%lactate%' OR m.measurement_type LIKE '%lactic acid%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_16_missing,

  
  -- Feature 17: Lymphocytes
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%lymphocytes%' OR m.measurement_type LIKE '%lymphs%' OR m.measurement_type LIKE '%lymph%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_17_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%lymphocytes%' OR m.measurement_type LIKE '%lymphs%' OR m.measurement_type LIKE '%lymph%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_17_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%lymphocytes%' OR m.measurement_type LIKE '%lymphs%' OR m.measurement_type LIKE '%lymph%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_17_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%lymphocytes%' OR lv.measurement_type LIKE '%lymphs%' OR lv.measurement_type LIKE '%lymph%')
   AND lv.rn = 1 LIMIT 1) AS feature_17_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%lymphocytes%' OR m.measurement_type LIKE '%lymphs%' OR m.measurement_type LIKE '%lymph%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_17_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%lymphocytes%' OR m.measurement_type LIKE '%lymphs%' OR m.measurement_type LIKE '%lymph%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_17_missing,

  
  -- Feature 18: MCHC
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mchc%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_18_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mchc%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_18_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mchc%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_18_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%mchc%')
   AND lv.rn = 1 LIMIT 1) AS feature_18_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mchc%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_18_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%mchc%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_18_missing,

  
  -- Feature 19: MCV
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mcv%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_19_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mcv%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_19_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mcv%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_19_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%mcv%')
   AND lv.rn = 1 LIMIT 1) AS feature_19_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mcv%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_19_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%mcv%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_19_missing,

  
  -- Feature 20: Magnesium
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%magnesium%' OR m.measurement_type LIKE '%mg%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_20_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%magnesium%' OR m.measurement_type LIKE '%mg%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_20_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%magnesium%' OR m.measurement_type LIKE '%mg%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_20_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%magnesium%' OR lv.measurement_type LIKE '%mg%')
   AND lv.rn = 1 LIMIT 1) AS feature_20_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%magnesium%' OR m.measurement_type LIKE '%mg%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_20_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%magnesium%' OR m.measurement_type LIKE '%mg%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_20_missing,

  
  -- Feature 21: Mean Airway Pressure
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mean_airway_pressure%' OR m.measurement_type LIKE '%map%' OR m.measurement_type LIKE '%mean airway pressure%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_21_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mean_airway_pressure%' OR m.measurement_type LIKE '%map%' OR m.measurement_type LIKE '%mean airway pressure%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_21_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mean_airway_pressure%' OR m.measurement_type LIKE '%map%' OR m.measurement_type LIKE '%mean airway pressure%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_21_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%mean_airway_pressure%' OR lv.measurement_type LIKE '%map%' OR lv.measurement_type LIKE '%mean airway pressure%')
   AND lv.rn = 1 LIMIT 1) AS feature_21_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mean_airway_pressure%' OR m.measurement_type LIKE '%map%' OR m.measurement_type LIKE '%mean airway pressure%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_21_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%mean_airway_pressure%' OR m.measurement_type LIKE '%map%' OR m.measurement_type LIKE '%mean airway pressure%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_21_missing,

  
  -- Feature 22: Diastolic BP
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%diastolic_bp%' OR m.measurement_type LIKE '%systemicdiastolic%' OR m.measurement_type LIKE '%diastolic%' OR m.measurement_type LIKE '%dbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_22_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%diastolic_bp%' OR m.measurement_type LIKE '%systemicdiastolic%' OR m.measurement_type LIKE '%diastolic%' OR m.measurement_type LIKE '%dbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_22_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%diastolic_bp%' OR m.measurement_type LIKE '%systemicdiastolic%' OR m.measurement_type LIKE '%diastolic%' OR m.measurement_type LIKE '%dbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_22_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%diastolic_bp%' OR lv.measurement_type LIKE '%systemicdiastolic%' OR lv.measurement_type LIKE '%diastolic%' OR lv.measurement_type LIKE '%dbp%')
   AND lv.rn = 1 LIMIT 1) AS feature_22_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%diastolic_bp%' OR m.measurement_type LIKE '%systemicdiastolic%' OR m.measurement_type LIKE '%diastolic%' OR m.measurement_type LIKE '%dbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_22_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%diastolic_bp%' OR m.measurement_type LIKE '%systemicdiastolic%' OR m.measurement_type LIKE '%diastolic%' OR m.measurement_type LIKE '%dbp%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_22_missing,

  
  -- Feature 23: Mean BP
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mean_bp%' OR m.measurement_type LIKE '%systemicmean%' OR m.measurement_type LIKE '%mean blood pressure%' OR m.measurement_type LIKE '%mbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_23_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mean_bp%' OR m.measurement_type LIKE '%systemicmean%' OR m.measurement_type LIKE '%mean blood pressure%' OR m.measurement_type LIKE '%mbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_23_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mean_bp%' OR m.measurement_type LIKE '%systemicmean%' OR m.measurement_type LIKE '%mean blood pressure%' OR m.measurement_type LIKE '%mbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_23_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%mean_bp%' OR lv.measurement_type LIKE '%systemicmean%' OR lv.measurement_type LIKE '%mean blood pressure%' OR lv.measurement_type LIKE '%mbp%')
   AND lv.rn = 1 LIMIT 1) AS feature_23_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%mean_bp%' OR m.measurement_type LIKE '%systemicmean%' OR m.measurement_type LIKE '%mean blood pressure%' OR m.measurement_type LIKE '%mbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_23_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%mean_bp%' OR m.measurement_type LIKE '%systemicmean%' OR m.measurement_type LIKE '%mean blood pressure%' OR m.measurement_type LIKE '%mbp%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_23_missing,

  
  -- Feature 24: Systolic BP
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%systolic_bp%' OR m.measurement_type LIKE '%systemicsystolic%' OR m.measurement_type LIKE '%systolic%' OR m.measurement_type LIKE '%sbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_24_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%systolic_bp%' OR m.measurement_type LIKE '%systemicsystolic%' OR m.measurement_type LIKE '%systolic%' OR m.measurement_type LIKE '%sbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_24_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%systolic_bp%' OR m.measurement_type LIKE '%systemicsystolic%' OR m.measurement_type LIKE '%systolic%' OR m.measurement_type LIKE '%sbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_24_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%systolic_bp%' OR lv.measurement_type LIKE '%systemicsystolic%' OR lv.measurement_type LIKE '%systolic%' OR lv.measurement_type LIKE '%sbp%')
   AND lv.rn = 1 LIMIT 1) AS feature_24_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%systolic_bp%' OR m.measurement_type LIKE '%systemicsystolic%' OR m.measurement_type LIKE '%systolic%' OR m.measurement_type LIKE '%sbp%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_24_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%systolic_bp%' OR m.measurement_type LIKE '%systemicsystolic%' OR m.measurement_type LIKE '%systolic%' OR m.measurement_type LIKE '%sbp%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_24_missing,

  
  -- Feature 25: O2 Saturation
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%sao2%' OR m.measurement_type LIKE '%o2 sat%' OR m.measurement_type LIKE '%oxygen saturation%' OR m.measurement_type LIKE '%o2sat%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_25_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%sao2%' OR m.measurement_type LIKE '%o2 sat%' OR m.measurement_type LIKE '%oxygen saturation%' OR m.measurement_type LIKE '%o2sat%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_25_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%sao2%' OR m.measurement_type LIKE '%o2 sat%' OR m.measurement_type LIKE '%oxygen saturation%' OR m.measurement_type LIKE '%o2sat%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_25_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%sao2%' OR lv.measurement_type LIKE '%o2 sat%' OR lv.measurement_type LIKE '%oxygen saturation%' OR lv.measurement_type LIKE '%o2sat%')
   AND lv.rn = 1 LIMIT 1) AS feature_25_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%sao2%' OR m.measurement_type LIKE '%o2 sat%' OR m.measurement_type LIKE '%oxygen saturation%' OR m.measurement_type LIKE '%o2sat%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_25_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%sao2%' OR m.measurement_type LIKE '%o2 sat%' OR m.measurement_type LIKE '%oxygen saturation%' OR m.measurement_type LIKE '%o2sat%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_25_missing,

  
  -- Feature 26: PT
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%pt%' OR m.measurement_type LIKE '%prothrombin time%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_26_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%pt%' OR m.measurement_type LIKE '%prothrombin time%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_26_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%pt%' OR m.measurement_type LIKE '%prothrombin time%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_26_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%pt%' OR lv.measurement_type LIKE '%prothrombin time%')
   AND lv.rn = 1 LIMIT 1) AS feature_26_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%pt%' OR m.measurement_type LIKE '%prothrombin time%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_26_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%pt%' OR m.measurement_type LIKE '%prothrombin time%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_26_missing,

  
  -- Feature 27: Potassium
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%potassium%' OR m.measurement_type LIKE '%k%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_27_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%potassium%' OR m.measurement_type LIKE '%k%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_27_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%potassium%' OR m.measurement_type LIKE '%k%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_27_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%potassium%' OR lv.measurement_type LIKE '%k%')
   AND lv.rn = 1 LIMIT 1) AS feature_27_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%potassium%' OR m.measurement_type LIKE '%k%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_27_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%potassium%' OR m.measurement_type LIKE '%k%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_27_missing,

  
  -- Feature 28: RBC
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%rbc%' OR m.measurement_type LIKE '%red blood cells%' OR m.measurement_type LIKE '%red blood cell count%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_28_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%rbc%' OR m.measurement_type LIKE '%red blood cells%' OR m.measurement_type LIKE '%red blood cell count%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_28_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%rbc%' OR m.measurement_type LIKE '%red blood cells%' OR m.measurement_type LIKE '%red blood cell count%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_28_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%rbc%' OR lv.measurement_type LIKE '%red blood cells%' OR lv.measurement_type LIKE '%red blood cell count%')
   AND lv.rn = 1 LIMIT 1) AS feature_28_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%rbc%' OR m.measurement_type LIKE '%red blood cells%' OR m.measurement_type LIKE '%red blood cell count%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_28_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%rbc%' OR m.measurement_type LIKE '%red blood cells%' OR m.measurement_type LIKE '%red blood cell count%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_28_missing,

  
  -- Feature 29: RDW
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%rdw%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_29_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%rdw%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_29_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%rdw%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_29_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%rdw%')
   AND lv.rn = 1 LIMIT 1) AS feature_29_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%rdw%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_29_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%rdw%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_29_missing,

  
  -- Feature 30: Respiratory Rate
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%respiration%' OR m.measurement_type LIKE '%respiratory rate%' OR m.measurement_type LIKE '%rr%' OR m.measurement_type LIKE '%resp rate%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_30_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%respiration%' OR m.measurement_type LIKE '%respiratory rate%' OR m.measurement_type LIKE '%rr%' OR m.measurement_type LIKE '%resp rate%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_30_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%respiration%' OR m.measurement_type LIKE '%respiratory rate%' OR m.measurement_type LIKE '%rr%' OR m.measurement_type LIKE '%resp rate%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_30_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%respiration%' OR lv.measurement_type LIKE '%respiratory rate%' OR lv.measurement_type LIKE '%rr%' OR lv.measurement_type LIKE '%resp rate%')
   AND lv.rn = 1 LIMIT 1) AS feature_30_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%respiration%' OR m.measurement_type LIKE '%respiratory rate%' OR m.measurement_type LIKE '%rr%' OR m.measurement_type LIKE '%resp rate%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_30_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%respiration%' OR m.measurement_type LIKE '%respiratory rate%' OR m.measurement_type LIKE '%rr%' OR m.measurement_type LIKE '%resp rate%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_30_missing,

  
  -- Feature 31: Sodium
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%sodium%' OR m.measurement_type LIKE '%na%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_31_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%sodium%' OR m.measurement_type LIKE '%na%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_31_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%sodium%' OR m.measurement_type LIKE '%na%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_31_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%sodium%' OR lv.measurement_type LIKE '%na%')
   AND lv.rn = 1 LIMIT 1) AS feature_31_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%sodium%' OR m.measurement_type LIKE '%na%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_31_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%sodium%' OR m.measurement_type LIKE '%na%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_31_missing,

  
  -- Feature 32: Temperature F
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%temperature%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_32_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%temperature%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_32_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%temperature%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_32_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%temperature%')
   AND lv.rn = 1 LIMIT 1) AS feature_32_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%temperature%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_32_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%temperature%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_32_missing,

  
  -- Feature 33: Temperature
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%temperature%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_33_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%temperature%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_33_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%temperature%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_33_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%temperature%')
   AND lv.rn = 1 LIMIT 1) AS feature_33_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%temperature%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_33_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%temperature%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_33_missing,

  
  -- Feature 34: Tidal Volume
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%tidal_volume%' OR m.measurement_type LIKE '%vt%' OR m.measurement_type LIKE '%tidal volume%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_34_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%tidal_volume%' OR m.measurement_type LIKE '%vt%' OR m.measurement_type LIKE '%tidal volume%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_34_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%tidal_volume%' OR m.measurement_type LIKE '%vt%' OR m.measurement_type LIKE '%tidal volume%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_34_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%tidal_volume%' OR lv.measurement_type LIKE '%vt%' OR lv.measurement_type LIKE '%tidal volume%')
   AND lv.rn = 1 LIMIT 1) AS feature_34_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%tidal_volume%' OR m.measurement_type LIKE '%vt%' OR m.measurement_type LIKE '%tidal volume%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_34_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%tidal_volume%' OR m.measurement_type LIKE '%vt%' OR m.measurement_type LIKE '%tidal volume%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_34_missing,

  
  -- Feature 35: BUN
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%bun%' OR m.measurement_type LIKE '%blood urea nitrogen%' OR m.measurement_type LIKE '%urea nitrogen%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_35_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%bun%' OR m.measurement_type LIKE '%blood urea nitrogen%' OR m.measurement_type LIKE '%urea nitrogen%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_35_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%bun%' OR m.measurement_type LIKE '%blood urea nitrogen%' OR m.measurement_type LIKE '%urea nitrogen%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_35_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%bun%' OR lv.measurement_type LIKE '%blood urea nitrogen%' OR lv.measurement_type LIKE '%urea nitrogen%')
   AND lv.rn = 1 LIMIT 1) AS feature_35_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%bun%' OR m.measurement_type LIKE '%blood urea nitrogen%' OR m.measurement_type LIKE '%urea nitrogen%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_35_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%bun%' OR m.measurement_type LIKE '%blood urea nitrogen%' OR m.measurement_type LIKE '%urea nitrogen%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_35_missing,

  
  -- Feature 36: WBC
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%wbc%' OR m.measurement_type LIKE '%white blood cells%' OR m.measurement_type LIKE '%white blood cell count%' OR m.measurement_type LIKE '%leukocyte%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_36_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%wbc%' OR m.measurement_type LIKE '%white blood cells%' OR m.measurement_type LIKE '%white blood cell count%' OR m.measurement_type LIKE '%leukocyte%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_36_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%wbc%' OR m.measurement_type LIKE '%white blood cells%' OR m.measurement_type LIKE '%white blood cell count%' OR m.measurement_type LIKE '%leukocyte%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_36_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%wbc%' OR lv.measurement_type LIKE '%white blood cells%' OR lv.measurement_type LIKE '%white blood cell count%' OR lv.measurement_type LIKE '%leukocyte%')
   AND lv.rn = 1 LIMIT 1) AS feature_36_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%wbc%' OR m.measurement_type LIKE '%white blood cells%' OR m.measurement_type LIKE '%white blood cell count%' OR m.measurement_type LIKE '%leukocyte%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_36_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%wbc%' OR m.measurement_type LIKE '%white blood cells%' OR m.measurement_type LIKE '%white blood cell count%' OR m.measurement_type LIKE '%leukocyte%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_36_missing,

  
  -- Feature 37: pH
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ph%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_37_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ph%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_37_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ph%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_37_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%ph%')
   AND lv.rn = 1 LIMIT 1) AS feature_37_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ph%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_37_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%ph%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_37_missing,

  
  -- Feature 38: pO2
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%pao2%' OR m.measurement_type LIKE '%po2%' OR m.measurement_type LIKE '%oxygen partial pressure%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_38_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%pao2%' OR m.measurement_type LIKE '%po2%' OR m.measurement_type LIKE '%oxygen partial pressure%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_38_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%pao2%' OR m.measurement_type LIKE '%po2%' OR m.measurement_type LIKE '%oxygen partial pressure%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_38_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%pao2%' OR lv.measurement_type LIKE '%po2%' OR lv.measurement_type LIKE '%oxygen partial pressure%')
   AND lv.rn = 1 LIMIT 1) AS feature_38_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%pao2%' OR m.measurement_type LIKE '%po2%' OR m.measurement_type LIKE '%oxygen partial pressure%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_38_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%pao2%' OR m.measurement_type LIKE '%po2%' OR m.measurement_type LIKE '%oxygen partial pressure%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_38_missing,

  
  -- Feature 39: LDH
  (SELECT AVG(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ldh%' OR m.measurement_type LIKE '%lactate dehydrogenase%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_39_mean,
  (SELECT MIN(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ldh%' OR m.measurement_type LIKE '%lactate dehydrogenase%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_39_min,
  (SELECT MAX(m.value) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ldh%' OR m.measurement_type LIKE '%lactate dehydrogenase%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_39_max,
  (SELECT lv.last_value 
   FROM last_values lv 
   WHERE lv.patientunitstayid = p.patientunitstayid 
   AND (lv.measurement_type LIKE '%ldh%' OR lv.measurement_type LIKE '%lactate dehydrogenase%')
   AND lv.rn = 1 LIMIT 1) AS feature_39_last,
  (SELECT COUNT(*) 
   FROM all_measurements m 
   WHERE m.patientunitstayid = p.patientunitstayid 
   AND (m.measurement_type LIKE '%ldh%' OR m.measurement_type LIKE '%lactate dehydrogenase%')
   AND m.measurement_time >= 0 AND m.measurement_time < 24*60) AS feature_39_count,
  CASE WHEN (SELECT COUNT(*) 
             FROM all_measurements m 
             WHERE m.patientunitstayid = p.patientunitstayid 
             AND (m.measurement_type LIKE '%ldh%' OR m.measurement_type LIKE '%lactate dehydrogenase%')
             AND m.measurement_time >= 0 AND m.measurement_time < 24*60) = 0 THEN 1 ELSE 0 END AS feature_39_missing


FROM patient_stays p
ORDER BY p.patientunitstayid;