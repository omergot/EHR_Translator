# Database Structure Analysis Report

Generated on: 2025-09-03 20:20:07

## Executive Summary

- **Total Schemas Analyzed**: 86
- **Total Tables Found**: 882
- **Total Columns Analyzed**: 11142

## eICU Database Analysis

**Schemas**: ['bsi_100_0.1h_test', 'bsi_100_0.2h_test', 'bsi_100_0.5h_test', 'bsi_100_2h_test', 'bsi_100_test', 'bsi_10_test', 'bsi_eicu_100_test', 'bsi_less_features_100_test', 'cohort_analysis', 'eicu_bsi_100_0.05h_test', 'eicu_bsi_100_0.1h_test', 'eicu_bsi_100_0.2h_test', 'eicu_bsi_100_0.5h_external_test', 'eicu_bsi_100_0.5h_test', 'eicu_bsi_100_2h_external_test', 'eicu_bsi_100_2h_test', 'eicu_bsi_100_4h_external_test', 'eicu_bsi_100_4h_test', 'eicu_bsi_100_5h_external_test', 'eicu_bsi_100_5h_test', 'eicu_bsi_100_test', 'eicu_bsi_10_test', 'eicu_crd', 'eicu_mortality_100_2h_test', 'eicu_mortality_100_test', 'mimic_core', 'mimic_hosp', 'mimic_icu', 'mimiciii', 'mimiciv_bsi_100_0.25h_test', 'mimiciv_bsi_100_0.5h_test', 'mimiciv_bsi_100_2h_test', 'mimiciv_bsi_100_4h_test', 'mimiciv_bsi_100_8h_test', 'mimiciv_hosp', 'mimiciv_icu', 'mortality_100_2h_test', 'omop', 'public', 'reconstruction_eicu_100_test', 'reconstruction_less_features_100_test', 'reconstruction_mimiciv_100_test', 'temp_eicu_20250903_200943']

### Schema: `bsi_100_0.1h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_100_0.1h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_100_0.1h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_100_0.2h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_100_0.2h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_100_0.2h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_100_0.5h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_100_0.5h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_100_0.5h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_100_2h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_100_2h_cohort | BASE TABLE | 2197 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_100_2h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_100_cohort | BASE TABLE | 2197 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_100_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_10_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_10_cohort | BASE TABLE | 2197 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_10_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_eicu_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_eicu_100_cohort | BASE TABLE | 1012 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_eicu_100_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `bsi_less_features_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_less_features_100_cohort | BASE TABLE | 2197 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_less_features_100_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `cohort_analysis`

- **Tables**: 1
- **Total Columns**: 13
- **High Potential Features**: 6
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| bsi_eicu_cohort_analysis | BASE TABLE | 954 | 13 | 6 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**bsi_eicu_cohort_analysis**
- High potential: initial_count, positive_count, negative_count, total_count, good_hospitals_count, final_count
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_0.05h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_0.05h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_0.05h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_0.1h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_0.1h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_0.1h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_0.2h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_0.2h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_0.2h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_0.5h_external_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_0.5h_external_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_0.5h_external_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_0.5h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_0.5h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_0.5h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_2h_external_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_2h_external_cohort | BASE TABLE | 58 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_2h_external_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_2h_test`

- **Tables**: 1
- **Total Columns**: 13
- **High Potential Features**: 6
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_2h_cohort | BASE TABLE | 779 | 13 | 6 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**__eicu_bsi_100_2h_cohort**
- High potential: initial_count, positive_count, negative_count, total_count, good_hospitals_count, final_count
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_4h_external_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_4h_external_cohort | BASE TABLE | 58 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_4h_external_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_4h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_4h_cohort | BASE TABLE | 954 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_4h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_5h_external_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_5h_external_cohort | BASE TABLE | 58 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_5h_external_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_5h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_5h_cohort | BASE TABLE | 954 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_5h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_cohort | BASE TABLE | 1012 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_10_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_10_cohort | BASE TABLE | 1012 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_10_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_crd`

- **Tables**: 31
- **Total Columns**: 391
- **High Potential Features**: 21
- **Medium Potential Features**: 264

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| admissiondrug | BASE TABLE | 874920 | 14 | 1 | admissiondrugid, patientunitstayid, drugoffset... |
| admissiondx | BASE TABLE | 626858 | 6 | 0 | admissiondxid, patientunitstayid, admitdxenteredoffset |
| allergy | BASE TABLE | 251949 | 13 | 0 | allergyid, patientunitstayid, allergyoffset... |
| apacheapsvar | BASE TABLE | 171177 | 26 | 3 | apacheapsvarid, patientunitstayid, intubated... |
| apachepatientresult | BASE TABLE | 297064 | 23 | 1 | apachepatientresultsid, patientunitstayid, acutephysiologyscore... |
| apachepredvar | BASE TABLE | 171177 | 51 | 4 | apachepredvarid, patientunitstayid, sicuday... |
| careplancareprovider | BASE TABLE | 502765 | 8 | 0 | cplcareprovderid, patientunitstayid, careprovidersaveoffset |
| careplaneol | BASE TABLE | 1433 | 5 | 0 | cpleolid, patientunitstayid, cpleolsaveoffset... |
| careplangeneral | BASE TABLE | 3115018 | 6 | 0 | cplgeneralid, patientunitstayid, cplitemoffset |
| careplangoal | BASE TABLE | 504139 | 7 | 0 | cplgoalid, patientunitstayid, cplgoaloffset |
| careplaninfectiousdisease | BASE TABLE | 8056 | 8 | 0 | cplinfectid, patientunitstayid, cplinfectdiseaseoffset |
| customlab | BASE TABLE | 1082 | 7 | 0 | customlabid, patientunitstayid, labotheroffset... |
| diagnosis | BASE TABLE | 2710672 | 7 | 0 | diagnosisid, patientunitstayid, diagnosisoffset |
| hospital | BASE TABLE | 208 | 4 | 0 | hospitalid |
| infusiondrug | BASE TABLE | 4803719 | 9 | 0 | infusiondrugid, patientunitstayid, infusionoffset |
| intakeoutput | BASE TABLE | 12030289 | 12 | 1 | intakeoutputid, patientunitstayid, intakeoutputoffset... |
| lab | BASE TABLE | 39132531 | 10 | 3 | labid, patientunitstayid, labresultoffset... |
| medication | BASE TABLE | 7301853 | 15 | 0 | medicationid, patientunitstayid, drugorderoffset... |
| microlab | BASE TABLE | 16996 | 7 | 0 | microlabid, patientunitstayid, culturetakenoffset |
| note | BASE TABLE | 2254179 | 8 | 0 | noteid, patientunitstayid, noteoffset... |
| nurseassessment | BASE TABLE | 15602498 | 8 | 0 | nurseassessid, patientunitstayid, nurseassessoffset... |
| nursecare | BASE TABLE | 8311132 | 8 | 0 | nursecareid, patientunitstayid, nursecareoffset... |
| nursecharting | BASE TABLE | 151604232 | 8 | 0 | nursingchartid, patientunitstayid, nursingchartoffset... |
| pasthistory | BASE TABLE | 1149180 | 8 | 0 | pasthistoryid, patientunitstayid, pasthistoryoffset... |
| patient | BASE TABLE | 200859 | 29 | 3 | patientunitstayid, patienthealthsystemstayid, hospitalid... |
| physicalexam | BASE TABLE | 9212316 | 6 | 0 | physicalexamid, patientunitstayid, physicalexamoffset |
| respiratorycare | BASE TABLE | 865381 | 34 | 3 | respcareid, patientunitstayid, respcarestatusoffset... |
| respiratorycharting | BASE TABLE | 20168176 | 7 | 0 | respchartid, patientunitstayid, respchartoffset... |
| treatment | BASE TABLE | 3688745 | 5 | 0 | treatmentid, patientunitstayid, treatmentoffset |
| vitalaperiodic | BASE TABLE | 25075074 | 13 | 0 | vitalaperiodicid, patientunitstayid, observationoffset... |
| vitalperiodic | BASE TABLE | 146671642 | 19 | 2 | vitalperiodicid, patientunitstayid, observationoffset... |

#### Key Feature-Rich Tables

**admissiondrug**
- High potential: drugdosage
- Medium potential: admissiondrugid, patientunitstayid, drugoffset, drugenteredoffset, drugnotetype, specialtytype, usertype, drugname, drughiclseqno

**allergy**
- Medium potential: allergyid, patientunitstayid, allergyoffset, allergyenteredoffset, allergynotetype, specialtytype, usertype, drugname, allergytype, allergyname

**apacheapsvar**
- High potential: temperature, respiratoryrate, heartrate
- Medium potential: apacheapsvarid, patientunitstayid, intubated, vent, dialysis, eyes, motor, verbal, meds, urine

**apachepatientresult**
- High potential: apachepatientresultsid
- Medium potential: patientunitstayid, acutephysiologyscore, apachescore, predictediculos, actualiculos, predictedhospitallos, actualhospitallos, preopmi, preopcardiaccath, ptcawithin24h

**apachepredvar**
- High potential: bedcount, graftcount, age, managementsystem
- Medium potential: apachepredvarid, patientunitstayid, sicuday, saps3day1, saps3today, saps3yesterday, gender, teachtype, region, admitsource

### Schema: `eicu_mortality_100_2h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_mortality_100_2h_cohort | BASE TABLE | 170115 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_mortality_100_2h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_mortality_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_mortality_100_cohort | BASE TABLE | 170115 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_mortality_100_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `mimic_core`

- **Tables**: 3
- **Total Columns**: 28
- **High Potential Features**: 1
- **Medium Potential Features**: 18

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| admissions | BASE TABLE | 0 | 15 | 0 | subject_id, hadm_id, hospital_expire_flag |
| patients | BASE TABLE | 0 | 6 | 1 | subject_id, anchor_age, anchor_year |
| transfers | BASE TABLE | 0 | 7 | 0 | subject_id, hadm_id, transfer_id |

#### Key Feature-Rich Tables

**admissions**
- Medium potential: subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, edregtime, edouttime, hospital_expire_flag

**transfers**
- Medium potential: subject_id, hadm_id, transfer_id, eventtype, intime, outtime

### Schema: `mimic_hosp`

- **Tables**: 17
- **Total Columns**: 187
- **High Potential Features**: 6
- **Medium Potential Features**: 93

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| d_hcpcs | BASE TABLE | 0 | 4 | 0 | category |
| d_icd_diagnoses | BASE TABLE | 0 | 3 | 0 | icd_version |
| d_icd_procedures | BASE TABLE | 0 | 3 | 0 | icd_version |
| d_labitems | BASE TABLE | 0 | 5 | 0 | itemid |
| diagnoses_icd | BASE TABLE | 0 | 5 | 0 | subject_id, hadm_id, seq_num... |
| drgcodes | BASE TABLE | 0 | 7 | 0 | subject_id, hadm_id, drg_severity... |
| emar | BASE TABLE | 0 | 11 | 0 | subject_id, hadm_id, emar_seq... |
| emar_detail | BASE TABLE | 0 | 33 | 0 | subject_id, emar_seq, parent_field_ordinal... |
| hcpcsevents | BASE TABLE | 0 | 6 | 0 | subject_id, hadm_id, seq_num |
| labevents | BASE TABLE | 0 | 15 | 1 | labevent_id, subject_id, hadm_id... |
| microbiologyevents | BASE TABLE | 0 | 24 | 1 | microevent_id, subject_id, hadm_id... |
| pharmacy | BASE TABLE | 0 | 27 | 3 | subject_id, hadm_id, pharmacy_id... |
| poe | BASE TABLE | 0 | 11 | 0 | poe_seq, subject_id, hadm_id |
| poe_detail | BASE TABLE | 0 | 5 | 0 | poe_seq, subject_id |
| prescriptions | BASE TABLE | 0 | 17 | 1 | subject_id, hadm_id, pharmacy_id... |
| procedures_icd | BASE TABLE | 0 | 6 | 0 | subject_id, hadm_id, seq_num... |
| services | BASE TABLE | 0 | 5 | 0 | subject_id, hadm_id |

#### Key Feature-Rich Tables

**emar**
- Medium potential: subject_id, hadm_id, emar_seq, pharmacy_id, charttime, scheduletime, storetime

**emar_detail**
- Medium potential: subject_id, emar_seq, parent_field_ordinal, administration_type, pharmacy_id, barcode_type

**labevents**
- High potential: valuenum
- Medium potential: labevent_id, subject_id, hadm_id, specimen_id, itemid, charttime, storetime, ref_range_lower, ref_range_upper

**microbiologyevents**
- High potential: dilution_value
- Medium potential: microevent_id, subject_id, hadm_id, micro_specimen_id, chartdate, charttime, spec_itemid, spec_type_desc, test_seq, storedate

**pharmacy**
- High potential: basal_rate, doses_per_24_hrs, expiration_value
- Medium potential: subject_id, hadm_id, pharmacy_id, starttime, stoptime, proc_type, entertime, verifiedtime, infusion_type, duration

### Schema: `mimic_icu`

- **Tables**: 7
- **Total Columns**: 96
- **High Potential Features**: 15
- **Medium Potential Features**: 61

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| chartevents | BASE TABLE | 0 | 10 | 1 | subject_id, hadm_id, stay_id... |
| d_items | BASE TABLE | 0 | 9 | 2 | itemid, lownormalvalue, highnormalvalue |
| datetimeevents | BASE TABLE | 0 | 9 | 0 | subject_id, hadm_id, stay_id... |
| icustays | BASE TABLE | 0 | 8 | 0 | subject_id, hadm_id, stay_id... |
| inputevents | BASE TABLE | 0 | 26 | 6 | subject_id, hadm_id, stay_id... |
| outputevents | BASE TABLE | 0 | 8 | 1 | subject_id, hadm_id, stay_id... |
| procedureevents | BASE TABLE | 0 | 26 | 5 | subject_id, hadm_id, stay_id... |

#### Key Feature-Rich Tables

**chartevents**
- High potential: valuenum
- Medium potential: subject_id, hadm_id, stay_id, charttime, storetime, itemid, warning

**d_items**
- High potential: lownormalvalue, highnormalvalue
- Medium potential: itemid, label, unitname, param_type

**datetimeevents**
- Medium potential: subject_id, hadm_id, stay_id, charttime, storetime, itemid, value, warning

**icustays**
- Medium potential: subject_id, hadm_id, stay_id, intime, outtime, los

**inputevents**
- High potential: amount, rate, patientweight, totalamount, originalamount, originalrate
- Medium potential: subject_id, hadm_id, stay_id, starttime, endtime, storetime, itemid, orderid, linkorderid, ordercategoryname

### Schema: `mimiciii`

- **Tables**: 273
- **Total Columns**: 3825
- **High Potential Features**: 228
- **Medium Potential Features**: 2643

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| admissions | BASE TABLE | 58976 | 20 | 0 | row_id, subject_id, hadm_id... |
| callout | BASE TABLE | 34499 | 25 | 0 | row_id, subject_id, hadm_id... |
| caregivers | BASE TABLE | 7567 | 5 | 0 | row_id, cgid, mimic_id |
| chartevents | BASE TABLE | 330712483 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_1 | BASE TABLE | 22204 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_10 | BASE TABLE | 2072743 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_100 | BASE TABLE | 3561885 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_101 | BASE TABLE | 1174868 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_102 | BASE TABLE | 293325 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_103 | BASE TABLE | 249776 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_104 | BASE TABLE | 3010424 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_105 | BASE TABLE | 679113 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_106 | BASE TABLE | 5208156 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_107 | BASE TABLE | 673719 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_108 | BASE TABLE | 2785057 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_109 | BASE TABLE | 1687886 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_11 | BASE TABLE | 178 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_110 | BASE TABLE | 2445808 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_111 | BASE TABLE | 2433936 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_112 | BASE TABLE | 4449487 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_113 | BASE TABLE | 1676872 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_114 | BASE TABLE | 476670 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_115 | BASE TABLE | 1621393 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_116 | BASE TABLE | 2309226 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_117 | BASE TABLE | 690295 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_118 | BASE TABLE | 1009254 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_119 | BASE TABLE | 803881 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_12 | BASE TABLE | 892239 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_120 | BASE TABLE | 2367096 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_121 | BASE TABLE | 1360432 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_122 | BASE TABLE | 982518 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_123 | BASE TABLE | 655454 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_124 | BASE TABLE | 1807316 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_125 | BASE TABLE | 34909 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_126 | BASE TABLE | 1378959 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_127 | BASE TABLE | 178112 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_128 | BASE TABLE | 1772387 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_129 | BASE TABLE | 1802684 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_13 | BASE TABLE | 1181039 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_130 | BASE TABLE | 1622363 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_131 | BASE TABLE | 43749 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_132 | BASE TABLE | 601818 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_133 | BASE TABLE | 2085994 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_134 | BASE TABLE | 5266438 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_135 | BASE TABLE | 1573583 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_136 | BASE TABLE | 3870155 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_137 | BASE TABLE | 719203 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_138 | BASE TABLE | 1600973 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_139 | BASE TABLE | 1687615 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_14 | BASE TABLE | 1136214 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_140 | BASE TABLE | 1146136 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_141 | BASE TABLE | 1619782 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_142 | BASE TABLE | 204405 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_143 | BASE TABLE | 725866 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_144 | BASE TABLE | 302 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_145 | BASE TABLE | 976252 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_146 | BASE TABLE | 649745 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_147 | BASE TABLE | 1804988 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_148 | BASE TABLE | 33554 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_149 | BASE TABLE | 1375295 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_15 | BASE TABLE | 3418901 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_150 | BASE TABLE | 174222 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_151 | BASE TABLE | 1769925 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_152 | BASE TABLE | 1796313 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_153 | BASE TABLE | 18753 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_154 | BASE TABLE | 0 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_155 | BASE TABLE | 2762225 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_156 | BASE TABLE | 431909 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_157 | BASE TABLE | 2023672 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_158 | BASE TABLE | 0 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_159 | BASE TABLE | 1149788 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_16 | BASE TABLE | 1198681 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_160 | BASE TABLE | 1149537 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_161 | BASE TABLE | 1156173 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_162 | BASE TABLE | 648200 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_163 | BASE TABLE | 526472 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_164 | BASE TABLE | 1290488 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_165 | BASE TABLE | 1289885 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_166 | BASE TABLE | 1292916 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_167 | BASE TABLE | 208 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_168 | BASE TABLE | 2737105 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_169 | BASE TABLE | 466344 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_17 | BASE TABLE | 1111444 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_170 | BASE TABLE | 2671816 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_171 | BASE TABLE | 3262258 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_172 | BASE TABLE | 4068153 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_173 | BASE TABLE | 765274 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_174 | BASE TABLE | 1139355 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_175 | BASE TABLE | 1983602 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_176 | BASE TABLE | 2185541 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_177 | BASE TABLE | 3552998 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_178 | BASE TABLE | 2289753 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_179 | BASE TABLE | 1610057 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_18 | BASE TABLE | 3216866 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_180 | BASE TABLE | 395061 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_181 | BASE TABLE | 4797667 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_182 | BASE TABLE | 3317320 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_183 | BASE TABLE | 2611372 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_184 | BASE TABLE | 4089672 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_185 | BASE TABLE | 1559194 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_186 | BASE TABLE | 2089736 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_187 | BASE TABLE | 1465008 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_188 | BASE TABLE | 717326 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_189 | BASE TABLE | 2933324 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_19 | BASE TABLE | 1669170 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_190 | BASE TABLE | 3110922 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_191 | BASE TABLE | 618565 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_192 | BASE TABLE | 1165 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_193 | BASE TABLE | 1849287 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_194 | BASE TABLE | 4059618 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_195 | BASE TABLE | 3154406 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_196 | BASE TABLE | 2873716 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_197 | BASE TABLE | 2659813 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_198 | BASE TABLE | 3763143 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_199 | BASE TABLE | 1718572 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_2 | BASE TABLE | 737224 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_20 | BASE TABLE | 818852 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_200 | BASE TABLE | 2662741 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_201 | BASE TABLE | 1605091 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_202 | BASE TABLE | 5553957 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_203 | BASE TABLE | 5627006 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_204 | BASE TABLE | 716961 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_205 | BASE TABLE | 816157 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_206 | BASE TABLE | 1862707 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_207 | BASE TABLE | 2313406 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_21 | BASE TABLE | 974476 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_22 | BASE TABLE | 2544519 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_23 | BASE TABLE | 9458 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_24 | BASE TABLE | 3236350 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_25 | BASE TABLE | 1837071 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_26 | BASE TABLE | 954139 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_27 | BASE TABLE | 1456328 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_28 | BASE TABLE | 945638 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_29 | BASE TABLE | 1545176 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_3 | BASE TABLE | 56235 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_30 | BASE TABLE | 955452 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_31 | BASE TABLE | 5180809 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_32 | BASE TABLE | 3303151 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_33 | BASE TABLE | 3676785 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_34 | BASE TABLE | 7811955 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_35 | BASE TABLE | 783762 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_36 | BASE TABLE | 402022 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_37 | BASE TABLE | 786544 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_38 | BASE TABLE | 349997 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_39 | BASE TABLE | 1032728 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_4 | BASE TABLE | 1442406 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_40 | BASE TABLE | 1589945 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_41 | BASE TABLE | 950038 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_42 | BASE TABLE | 1579681 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_43 | BASE TABLE | 1553537 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_44 | BASE TABLE | 392595 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_45 | BASE TABLE | 1155571 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_46 | BASE TABLE | 985720 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_47 | BASE TABLE | 774157 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_48 | BASE TABLE | 917780 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_49 | BASE TABLE | 4595328 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_5 | BASE TABLE | 878442 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_50 | BASE TABLE | 852968 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_51 | BASE TABLE | 883558 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_52 | BASE TABLE | 3205052 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_53 | BASE TABLE | 1022222 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_54 | BASE TABLE | 1641889 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_55 | BASE TABLE | 1654497 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_56 | BASE TABLE | 784361 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_57 | BASE TABLE | 913144 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_58 | BASE TABLE | 787137 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_59 | BASE TABLE | 1251345 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_6 | BASE TABLE | 1659172 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_60 | BASE TABLE | 774906 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_61 | BASE TABLE | 962191 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_62 | BASE TABLE | 3386719 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_63 | BASE TABLE | 580529 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_64 | BASE TABLE | 666496 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_65 | BASE TABLE | 2048955 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_66 | BASE TABLE | 954354 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_67 | BASE TABLE | 437 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_68 | BASE TABLE | 1447426 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_69 | BASE TABLE | 3418917 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_7 | BASE TABLE | 636690 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_70 | BASE TABLE | 2135542 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_71 | BASE TABLE | 774213 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_72 | BASE TABLE | 300570 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_73 | BASE TABLE | 1042512 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_74 | BASE TABLE | 378549 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_75 | BASE TABLE | 772277 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_76 | BASE TABLE | 773891 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_77 | BASE TABLE | 376047 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_78 | BASE TABLE | 740176 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_79 | BASE TABLE | 1099236 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_8 | BASE TABLE | 285028 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_80 | BASE TABLE | 933238 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_81 | BASE TABLE | 20754 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_82 | BASE TABLE | 727719 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_83 | BASE TABLE | 937064 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_84 | BASE TABLE | 1049706 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_85 | BASE TABLE | 952177 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_86 | BASE TABLE | 321012 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_87 | BASE TABLE | 3464326 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_88 | BASE TABLE | 5925297 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_89 | BASE TABLE | 1716561 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_9 | BASE TABLE | 2096678 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_90 | BASE TABLE | 1417336 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_91 | BASE TABLE | 803816 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_92 | BASE TABLE | 423898 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_93 | BASE TABLE | 592344 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_94 | BASE TABLE | 250247 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_95 | BASE TABLE | 2955851 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_96 | BASE TABLE | 808725 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_97 | BASE TABLE | 1083809 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_98 | BASE TABLE | 1281335 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_99 | BASE TABLE | 2261065 | 16 | 1 | row_id, subject_id, hadm_id... |
| cptevents | BASE TABLE | 573146 | 13 | 0 | row_id, subject_id, hadm_id... |
| d_cpt | BASE TABLE | 134 | 10 | 0 | row_id, category, mincodeinsubsection... |
| d_icd_diagnoses | BASE TABLE | 14567 | 5 | 0 | row_id, mimic_id |
| d_icd_procedures | BASE TABLE | 3882 | 5 | 0 | row_id, mimic_id |
| d_items | BASE TABLE | 12487 | 11 | 0 | row_id, itemid, conceptid... |
| d_labitems | BASE TABLE | 753 | 7 | 0 | row_id, itemid, mimic_id |
| datetimeevents | BASE TABLE | 4485937 | 15 | 0 | row_id, subject_id, hadm_id... |
| diagnoses_icd | BASE TABLE | 651047 | 6 | 0 | row_id, subject_id, hadm_id... |
| drgcodes | BASE TABLE | 125557 | 9 | 0 | row_id, subject_id, hadm_id... |
| gcpt_admission_location_to_concept | BASE TABLE | 9 | 3 | 0 | concept_id, mimic_id |
| gcpt_admission_type_to_concept | BASE TABLE | 4 | 3 | 0 | visit_concept_id, mimic_id |
| gcpt_admissions_diagnosis_to_concept | BASE TABLE | 1523 | 4 | 0 | concept_id, mimic_id |
| gcpt_atb_to_concept | BASE TABLE | 30 | 3 | 0 | mimic_id |
| gcpt_care_site | BASE TABLE | 30 | 6 | 0 | place_of_service_concept_id, visit_detail_concept_id, mimic_id |
| gcpt_chart_label_to_concept | BASE TABLE | 319 | 8 | 2 | value_lb, value_ub, measurement_concept_id... |
| gcpt_chart_observation_to_concept | BASE TABLE | 57 | 5 | 0 | concept_id, itemid, mimic_id |
| gcpt_continuous_unit_carevue | BASE TABLE | 12 | 3 | 0 | mimic_id |
| gcpt_cpt4_to_concept | BASE TABLE | 23 | 4 | 0 | procedure_concept_id, omop_mapping_is_sure, mimic_id |
| gcpt_cv_input_label_to_concept | BASE TABLE | 2782 | 5 | 0 | item_id, original_route, concept_id... |
| gcpt_datetimeevents_to_concept | BASE TABLE | 155 | 6 | 0 | observation_concept_id, itemid, observation_source_concept_id... |
| gcpt_derived_to_concept | BASE TABLE | 8 | 4 | 0 | itemid, concept_id, mimic_id |
| gcpt_discharge_location_to_concept | BASE TABLE | 17 | 4 | 0 | concept_id, mimic_id |
| gcpt_drgcode_to_concept | BASE TABLE | 1355 | 6 | 0 | non_standard_concept_id, standard_concept_id, mimic_id |
| gcpt_ethnicity_to_concept | BASE TABLE | 41 | 5 | 0 | race_concept_id, ethnicity_concept_id, priority... |
| gcpt_heart_rhythm_to_concept | BASE TABLE | 47 | 3 | 0 | concept_id, mimic_id |
| gcpt_inputevents_drug_to_concept | BASE TABLE | 108 | 4 | 0 | concept_id, itemid, mimic_id |
| gcpt_insurance_to_concept | BASE TABLE | 5 | 3 | 0 | concept_id, mimic_id |
| gcpt_lab_label_to_concept | BASE TABLE | 118 | 3 | 0 | concept_id, mimic_id |
| gcpt_lab_unit_to_concept | BASE TABLE | 71 | 3 | 0 | concept_id, mimic_id |
| gcpt_lab_value_to_concept | BASE TABLE | 3703 | 4 | 2 | value_as_concept_id, value_as_num, mimic_id |
| gcpt_labs_from_chartevents_to_concept | BASE TABLE | 80 | 4 | 0 | measurement_type_concept_id, mimic_id |
| gcpt_labs_specimen_to_concept | BASE TABLE | 17 | 4 | 0 | specimen_concept_id, mimic_id |
| gcpt_language_to_concept | BASE TABLE | 75 | 3 | 0 | concept_id, mimic_id |
| gcpt_map_route_to_concept | BASE TABLE | 28 | 4 | 0 | concept_id, mimic_id |
| gcpt_marital_status_to_concept | BASE TABLE | 7 | 3 | 0 | concept_id, mimic_id |
| gcpt_microbiology_specimen_to_concept | BASE TABLE | 88 | 4 | 0 | specimen_concept_id, mimic_id |
| gcpt_mv_input_label_to_concept | BASE TABLE | 159 | 5 | 0 | item_id, route, concept_id... |
| gcpt_note_category_to_concept | BASE TABLE | 15 | 4 | 0 | concept_id, category_id, mimic_id |
| gcpt_note_section_to_concept | BASE TABLE | 1113 | 6 | 0 | section_id, category_id, mimic_id |
| gcpt_org_name_to_concept | BASE TABLE | 362 | 3 | 0 | snomed, mimic_id |
| gcpt_output_label_to_concept | BASE TABLE | 1154 | 4 | 0 | item_id, concept_id, mimic_id |
| gcpt_prescriptions_ndcisnullzero_to_concept | BASE TABLE | 1354 | 4 | 0 | concept_id, mimic_id |
| gcpt_procedure_to_concept | BASE TABLE | 113 | 4 | 0 | item_id, snomed, concept_id... |
| gcpt_religion_to_concept | BASE TABLE | 20 | 3 | 0 | concept_id, mimic_id |
| gcpt_resistance_to_concept | BASE TABLE | 4 | 3 | 1 | value_as_concept_id, mimic_id |
| gcpt_route_to_concept | BASE TABLE | 78 | 3 | 0 | concept_id, mimic_id |
| gcpt_seq_num_to_concept | BASE TABLE | 44 | 3 | 0 | seq_num, concept_id, mimic_id |
| gcpt_spec_type_to_concept | BASE TABLE | 87 | 4 | 0 | snomed, mimic_id |
| gcpt_unit_doseera_concept_id | BASE TABLE | 27 | 6 | 1 | unit_concept_id, temporal_unit_concept_id, mimic_id |
| icustays | BASE TABLE | 61532 | 13 | 0 | row_id, subject_id, hadm_id... |
| inputevents_cv | BASE TABLE | 17527935 | 23 | 4 | row_id, subject_id, hadm_id... |
| inputevents_mv | BASE TABLE | 3618991 | 32 | 6 | row_id, subject_id, hadm_id... |
| labevents | BASE TABLE | 27854055 | 10 | 1 | row_id, subject_id, hadm_id... |
| microbiologyevents | BASE TABLE | 631726 | 17 | 1 | row_id, subject_id, hadm_id... |
| noteevents | BASE TABLE | 2083180 | 12 | 0 | row_id, subject_id, hadm_id... |
| outputevents | BASE TABLE | 4349218 | 14 | 1 | row_id, subject_id, hadm_id... |
| patients | BASE TABLE | 46520 | 9 | 0 | row_id, subject_id, expire_flag... |
| prescriptions | BASE TABLE | 4156450 | 20 | 0 | row_id, subject_id, hadm_id... |
| procedureevents_mv | BASE TABLE | 258066 | 26 | 1 | row_id, subject_id, hadm_id... |
| procedures_icd | BASE TABLE | 240095 | 6 | 0 | row_id, subject_id, hadm_id... |
| services | BASE TABLE | 73343 | 7 | 0 | row_id, subject_id, hadm_id... |
| transfers | BASE TABLE | 261897 | 14 | 0 | row_id, subject_id, hadm_id... |

#### Key Feature-Rich Tables

**admissions**
- Medium potential: row_id, subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, edregtime, edouttime, hospital_expire_flag

**callout**
- Medium potential: row_id, subject_id, hadm_id, submit_wardid, curr_wardid, callout_wardid, request_tele, request_resp, request_cdiff, request_mrsa

**chartevents**
- High potential: valuenum
- Medium potential: row_id, subject_id, hadm_id, icustay_id, itemid, charttime, storetime, cgid, warning, error

**chartevents_1**
- High potential: valuenum
- Medium potential: row_id, subject_id, hadm_id, icustay_id, itemid, charttime, storetime, cgid, warning, error

**chartevents_10**
- High potential: valuenum
- Medium potential: row_id, subject_id, hadm_id, icustay_id, itemid, charttime, storetime, cgid, warning, error

### Schema: `mimiciv_bsi_100_0.25h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mimiciv_bsi_100_0.25h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__mimiciv_bsi_100_0.25h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `mimiciv_bsi_100_0.5h_test`

- **Tables**: 1
- **Total Columns**: 8
- **High Potential Features**: 0
- **Medium Potential Features**: 8

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mimiciv_bsi_100_0.5h_cohort | BASE TABLE | Access denied or table empty | 8 | 0 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**__mimiciv_bsi_100_0.5h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y, last_years

### Schema: `mimiciv_bsi_100_2h_test`

- **Tables**: 1
- **Total Columns**: 8
- **High Potential Features**: 0
- **Medium Potential Features**: 8

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mimiciv_bsi_100_2h_cohort | BASE TABLE | 12139 | 8 | 0 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**__mimiciv_bsi_100_2h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y, last_years

### Schema: `mimiciv_bsi_100_4h_test`

- **Tables**: 1
- **Total Columns**: 8
- **High Potential Features**: 0
- **Medium Potential Features**: 8

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mimiciv_bsi_100_4h_cohort | BASE TABLE | 12139 | 8 | 0 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**__mimiciv_bsi_100_4h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y, last_years

### Schema: `mimiciv_bsi_100_8h_test`

- **Tables**: 1
- **Total Columns**: 8
- **High Potential Features**: 0
- **Medium Potential Features**: 8

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mimiciv_bsi_100_8h_cohort | BASE TABLE | 12139 | 8 | 0 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**__mimiciv_bsi_100_8h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y, last_years

### Schema: `mimiciv_hosp`

- **Tables**: 22
- **Total Columns**: 229
- **High Potential Features**: 7
- **Medium Potential Features**: 115

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| admissions | BASE TABLE | 431231 | 16 | 0 | subject_id, hadm_id, hospital_expire_flag |
| d_hcpcs | BASE TABLE | 89200 | 4 | 0 | category |
| d_icd_diagnoses | BASE TABLE | 109775 | 3 | 0 | icd_version |
| d_icd_procedures | BASE TABLE | 85257 | 3 | 0 | icd_version |
| d_labitems | BASE TABLE | 1622 | 4 | 0 | itemid |
| diagnoses_icd | BASE TABLE | 4756326 | 5 | 0 | subject_id, hadm_id, seq_num... |
| drgcodes | BASE TABLE | 604377 | 7 | 0 | subject_id, hadm_id, drg_severity... |
| emar | BASE TABLE | 26850359 | 12 | 0 | subject_id, hadm_id, emar_seq... |
| emar_detail | BASE TABLE | 54744789 | 33 | 0 | subject_id, emar_seq, pharmacy_id |
| hcpcsevents | BASE TABLE | 150771 | 6 | 0 | subject_id, hadm_id, seq_num |
| labevents | BASE TABLE | 118171367 | 16 | 1 | labevent_id, subject_id, hadm_id... |
| microbiologyevents | BASE TABLE | 3228713 | 25 | 1 | microevent_id, subject_id, hadm_id... |
| omr | BASE TABLE | 6439169 | 5 | 0 | subject_id, seq_num |
| patients | BASE TABLE | 299712 | 6 | 1 | subject_id, anchor_age, anchor_year |
| pharmacy | BASE TABLE | 13584514 | 27 | 3 | subject_id, hadm_id, pharmacy_id... |
| poe | BASE TABLE | 39366291 | 12 | 0 | poe_seq, subject_id, hadm_id |
| poe_detail | BASE TABLE | 3879418 | 5 | 0 | poe_seq, subject_id |
| prescriptions | BASE TABLE | 15416708 | 21 | 1 | subject_id, hadm_id, pharmacy_id... |
| procedures_icd | BASE TABLE | 669186 | 6 | 0 | subject_id, hadm_id, seq_num... |
| provider | BASE TABLE | 40508 | 1 | 0 |  |
| services | BASE TABLE | 468029 | 5 | 0 | subject_id, hadm_id |
| transfers | BASE TABLE | 1890972 | 7 | 0 | subject_id, hadm_id, transfer_id |

#### Key Feature-Rich Tables

**admissions**
- Medium potential: subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, edregtime, edouttime, hospital_expire_flag

**emar**
- Medium potential: subject_id, hadm_id, emar_seq, pharmacy_id, charttime, scheduletime, storetime

**labevents**
- High potential: valuenum
- Medium potential: labevent_id, subject_id, hadm_id, specimen_id, itemid, charttime, storetime, ref_range_lower, ref_range_upper

**microbiologyevents**
- High potential: dilution_value
- Medium potential: microevent_id, subject_id, hadm_id, micro_specimen_id, chartdate, charttime, spec_itemid, spec_type_desc, test_seq, storedate

**pharmacy**
- High potential: basal_rate, doses_per_24_hrs, expiration_value
- Medium potential: subject_id, hadm_id, pharmacy_id, starttime, stoptime, proc_type, entertime, verifiedtime, infusion_type, duration

### Schema: `mimiciv_icu`

- **Tables**: 9
- **Total Columns**: 113
- **High Potential Features**: 18
- **Medium Potential Features**: 73

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| caregiver | BASE TABLE | 15468 | 1 | 0 | caregiver_id |
| chartevents | BASE TABLE | 313645063 | 11 | 1 | subject_id, hadm_id, stay_id... |
| d_items | BASE TABLE | 4014 | 9 | 2 | itemid, lownormalvalue, highnormalvalue |
| datetimeevents | BASE TABLE | 7112999 | 10 | 0 | subject_id, hadm_id, stay_id... |
| icustays | BASE TABLE | 73181 | 8 | 0 | subject_id, hadm_id, stay_id... |
| ingredientevents | BASE TABLE | 11627821 | 17 | 4 | subject_id, hadm_id, stay_id... |
| inputevents | BASE TABLE | 8978893 | 26 | 6 | subject_id, hadm_id, stay_id... |
| outputevents | BASE TABLE | 4234967 | 9 | 1 | subject_id, hadm_id, stay_id... |
| procedureevents | BASE TABLE | 696092 | 22 | 4 | subject_id, hadm_id, stay_id... |

#### Key Feature-Rich Tables

**chartevents**
- High potential: valuenum
- Medium potential: subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, warning

**d_items**
- High potential: lownormalvalue, highnormalvalue
- Medium potential: itemid, label, unitname, param_type

**datetimeevents**
- Medium potential: subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, warning

**icustays**
- Medium potential: subject_id, hadm_id, stay_id, intime, outtime, los

**ingredientevents**
- High potential: amount, rate, originalamount, originalrate
- Medium potential: subject_id, hadm_id, stay_id, caregiver_id, starttime, endtime, storetime, itemid, orderid, linkorderid

### Schema: `mortality_100_2h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mortality_100_2h_cohort | BASE TABLE | 28973 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__mortality_100_2h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `omop`

- **Tables**: 40
- **Total Columns**: 429
- **High Potential Features**: 22
- **Medium Potential Features**: 297

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| attribute_definition | BASE TABLE | 0 | 5 | 0 | attribute_definition_id, attribute_type_concept_id |
| care_site | BASE TABLE | 0 | 6 | 0 | care_site_id, place_of_service_concept_id, location_id |
| cdm_source | BASE TABLE | 0 | 10 | 0 |  |
| cohort | BASE TABLE | 0 | 4 | 0 | cohort_definition_id, subject_id |
| cohort_attribute | BASE TABLE | 0 | 7 | 2 | cohort_definition_id, subject_id, attribute_definition_id... |
| cohort_definition | BASE TABLE | 0 | 7 | 0 | cohort_definition_id, definition_type_concept_id, subject_concept_id |
| concept | BASE TABLE | 0 | 10 | 0 | concept_id |
| concept_ancestor | BASE TABLE | 0 | 4 | 2 | ancestor_concept_id, descendant_concept_id, min_levels_of_separation... |
| concept_class | BASE TABLE | 0 | 3 | 0 | concept_class_concept_id |
| concept_relationship | BASE TABLE | 0 | 6 | 0 | concept_id_1, concept_id_2 |
| concept_synonym | BASE TABLE | 0 | 3 | 1 | concept_id, language_concept_id |
| condition_era | BASE TABLE | 0 | 6 | 1 | condition_era_id, person_id, condition_concept_id... |
| condition_occurrence | BASE TABLE | 0 | 16 | 0 | condition_occurrence_id, person_id, condition_concept_id... |
| cost | BASE TABLE | 0 | 22 | 1 | cost_id, cost_event_id, cost_type_concept_id... |
| death | BASE TABLE | 0 | 7 | 0 | person_id, death_type_concept_id, cause_concept_id... |
| device_exposure | BASE TABLE | 0 | 15 | 0 | device_exposure_id, person_id, device_concept_id... |
| domain | BASE TABLE | 0 | 3 | 0 | domain_concept_id |
| dose_era | BASE TABLE | 0 | 9 | 4 | dose_era_id, person_id, drug_concept_id... |
| drug_era | BASE TABLE | 0 | 7 | 1 | drug_era_id, person_id, drug_concept_id... |
| drug_exposure | BASE TABLE | 0 | 24 | 0 | drug_exposure_id, person_id, drug_concept_id... |
| drug_strength | BASE TABLE | 0 | 12 | 4 | drug_concept_id, ingredient_concept_id, amount_value... |
| fact_relationship | BASE TABLE | 0 | 5 | 0 | domain_concept_id_1, fact_id_1, domain_concept_id_2... |
| location | BASE TABLE | 0 | 8 | 0 | location_id |
| measurement | BASE TABLE | 0 | 20 | 2 | measurement_id, person_id, measurement_concept_id... |
| metadata | BASE TABLE | 0 | 7 | 1 | metadata_concept_id, metadata_type_concept_id, value_as_concept_id |
| note | BASE TABLE | 0 | 14 | 1 | note_id, person_id, note_type_concept_id... |
| note_nlp | BASE TABLE | 0 | 17 | 0 | note_nlp_id, note_id, section_concept_id... |
| observation | BASE TABLE | 0 | 18 | 2 | observation_id, person_id, observation_concept_id... |
| observation_period | BASE TABLE | 0 | 7 | 0 | observation_period_id, person_id, period_type_concept_id |
| payer_plan_period | BASE TABLE | 0 | 17 | 0 | payer_plan_period_id, person_id, payer_concept_id... |
| person | BASE TABLE | 0 | 18 | 0 | person_id, gender_concept_id, year_of_birth... |
| procedure_occurrence | BASE TABLE | 0 | 14 | 0 | procedure_occurrence_id, person_id, procedure_concept_id... |
| provider | BASE TABLE | 0 | 13 | 0 | provider_id, specialty_concept_id, care_site_id... |
| relationship | BASE TABLE | 0 | 6 | 0 | relationship_concept_id |
| source_to_concept_map | BASE TABLE | 0 | 9 | 0 | source_concept_id, target_concept_id |
| specimen | BASE TABLE | 0 | 15 | 0 | specimen_id, person_id, specimen_concept_id... |
| visit_detail | BASE TABLE | 0 | 23 | 0 | visit_detail_id, person_id, visit_detail_concept_id... |
| visit_detail_assign | BASE TABLE | 11 | 8 | 0 | visit_detail_id, visit_occurrence_id |
| visit_occurrence | BASE TABLE | 0 | 19 | 0 | visit_occurrence_id, person_id, visit_concept_id... |
| vocabulary | BASE TABLE | 0 | 5 | 0 | vocabulary_concept_id |

#### Key Feature-Rich Tables

**cohort_attribute**
- High potential: value_as_number, value_as_concept_id
- Medium potential: cohort_definition_id, subject_id, cohort_start_date, cohort_end_date, attribute_definition_id

**condition_era**
- High potential: condition_occurrence_count
- Medium potential: condition_era_id, person_id, condition_concept_id, condition_era_start_date, condition_era_end_date

**condition_occurrence**
- Medium potential: condition_occurrence_id, person_id, condition_concept_id, condition_start_date, condition_start_datetime, condition_end_date, condition_end_datetime, condition_type_concept_id, provider_id, visit_occurrence_id

**cost**
- High potential: amount_allowed
- Medium potential: cost_id, cost_event_id, cost_type_concept_id, currency_concept_id, total_charge, total_cost, total_paid, paid_by_payer, paid_by_patient, paid_patient_copay

**death**
- Medium potential: person_id, death_date, death_datetime, death_type_concept_id, cause_concept_id, cause_source_concept_id

### Schema: `public`

- **Tables**: 5
- **Total Columns**: 18
- **High Potential Features**: 0
- **Medium Potential Features**: 4

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| drug_converter | BASE TABLE | 258 | 3 | 0 | index |
| icd10_pheno | BASE TABLE | 1925 | 5 | 0 | index |
| icd9_to_icd10 | BASE TABLE | 23912 | 3 | 0 | index |
| medical_hist_converter | BASE TABLE | 41 | 3 | 0 | index |
| mimic_to_eicu_converter | BASE TABLE | 91 | 4 | 0 |  |

#### Key Feature-Rich Tables

No feature-rich tables found in this schema.

### Schema: `reconstruction_eicu_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __reconstruction_eicu_100_cohort | BASE TABLE | 200859 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__reconstruction_eicu_100_cohort**
- Medium potential: example_id, person_id, start_date, end_date, start_datetime, end_datetime, y

### Schema: `reconstruction_less_features_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __reconstruction_less_features_100_cohort | BASE TABLE | 28973 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__reconstruction_less_features_100_cohort**
- Medium potential: example_id, person_id, start_date, end_date, start_datetime, end_datetime, y

### Schema: `reconstruction_mimiciv_100_test`

- **Tables**: 1
- **Total Columns**: 8
- **High Potential Features**: 0
- **Medium Potential Features**: 8

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __reconstruction_mimiciv_100_cohort | BASE TABLE | 73181 | 8 | 0 | example_id, person_id, last_years... |

#### Key Feature-Rich Tables

**__reconstruction_mimiciv_100_cohort**
- Medium potential: example_id, person_id, start_date, end_date, start_datetime, end_datetime, last_years, y

### Schema: `temp_eicu_20250903_200943`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| simplified_cohort | BASE TABLE | 200450 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**simplified_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

## MIMIC-IV Database Analysis

**Schemas**: ['bsi_100_0.1h_test', 'bsi_100_0.2h_test', 'bsi_100_0.5h_test', 'bsi_100_2h_test', 'bsi_100_test', 'bsi_10_test', 'bsi_eicu_100_test', 'bsi_less_features_100_test', 'cohort_analysis', 'eicu_bsi_100_0.05h_test', 'eicu_bsi_100_0.1h_test', 'eicu_bsi_100_0.2h_test', 'eicu_bsi_100_0.5h_external_test', 'eicu_bsi_100_0.5h_test', 'eicu_bsi_100_2h_external_test', 'eicu_bsi_100_2h_test', 'eicu_bsi_100_4h_external_test', 'eicu_bsi_100_4h_test', 'eicu_bsi_100_5h_external_test', 'eicu_bsi_100_5h_test', 'eicu_bsi_100_test', 'eicu_bsi_10_test', 'eicu_crd', 'eicu_mortality_100_2h_test', 'eicu_mortality_100_test', 'mimic_core', 'mimic_hosp', 'mimic_icu', 'mimiciii', 'mimiciv_bsi_100_0.25h_test', 'mimiciv_bsi_100_0.5h_test', 'mimiciv_bsi_100_2h_test', 'mimiciv_bsi_100_4h_test', 'mimiciv_bsi_100_8h_test', 'mimiciv_hosp', 'mimiciv_icu', 'mortality_100_2h_test', 'omop', 'public', 'reconstruction_eicu_100_test', 'reconstruction_less_features_100_test', 'reconstruction_mimiciv_100_test', 'temp_eicu_20250903_200943']

### Schema: `bsi_100_0.1h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_100_0.1h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_100_0.1h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_100_0.2h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_100_0.2h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_100_0.2h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_100_0.5h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_100_0.5h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_100_0.5h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_100_2h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_100_2h_cohort | BASE TABLE | 2197 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_100_2h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_100_cohort | BASE TABLE | 2197 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_100_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_10_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_10_cohort | BASE TABLE | 2197 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_10_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `bsi_eicu_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_eicu_100_cohort | BASE TABLE | 1012 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_eicu_100_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `bsi_less_features_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __bsi_less_features_100_cohort | BASE TABLE | 2197 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__bsi_less_features_100_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `cohort_analysis`

- **Tables**: 1
- **Total Columns**: 13
- **High Potential Features**: 6
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| bsi_eicu_cohort_analysis | BASE TABLE | 954 | 13 | 6 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**bsi_eicu_cohort_analysis**
- High potential: initial_count, positive_count, negative_count, total_count, good_hospitals_count, final_count
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_0.05h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_0.05h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_0.05h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_0.1h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_0.1h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_0.1h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_0.2h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_0.2h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_0.2h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_0.5h_external_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_0.5h_external_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_0.5h_external_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_0.5h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_0.5h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_0.5h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_2h_external_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_2h_external_cohort | BASE TABLE | 58 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_2h_external_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_2h_test`

- **Tables**: 1
- **Total Columns**: 13
- **High Potential Features**: 6
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_2h_cohort | BASE TABLE | 779 | 13 | 6 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**__eicu_bsi_100_2h_cohort**
- High potential: initial_count, positive_count, negative_count, total_count, good_hospitals_count, final_count
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_4h_external_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_4h_external_cohort | BASE TABLE | 58 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_4h_external_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_4h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_4h_cohort | BASE TABLE | 954 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_4h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_5h_external_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_5h_external_cohort | BASE TABLE | 58 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_5h_external_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_5h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_5h_cohort | BASE TABLE | 954 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_5h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_100_cohort | BASE TABLE | 1012 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_100_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_bsi_10_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_bsi_10_cohort | BASE TABLE | 1012 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_bsi_10_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_crd`

- **Tables**: 31
- **Total Columns**: 391
- **High Potential Features**: 21
- **Medium Potential Features**: 264

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| admissiondrug | BASE TABLE | 874920 | 14 | 1 | admissiondrugid, patientunitstayid, drugoffset... |
| admissiondx | BASE TABLE | 626858 | 6 | 0 | admissiondxid, patientunitstayid, admitdxenteredoffset |
| allergy | BASE TABLE | 251949 | 13 | 0 | allergyid, patientunitstayid, allergyoffset... |
| apacheapsvar | BASE TABLE | 171177 | 26 | 3 | apacheapsvarid, patientunitstayid, intubated... |
| apachepatientresult | BASE TABLE | 297064 | 23 | 1 | apachepatientresultsid, patientunitstayid, acutephysiologyscore... |
| apachepredvar | BASE TABLE | 171177 | 51 | 4 | apachepredvarid, patientunitstayid, sicuday... |
| careplancareprovider | BASE TABLE | 502765 | 8 | 0 | cplcareprovderid, patientunitstayid, careprovidersaveoffset |
| careplaneol | BASE TABLE | 1433 | 5 | 0 | cpleolid, patientunitstayid, cpleolsaveoffset... |
| careplangeneral | BASE TABLE | 3115018 | 6 | 0 | cplgeneralid, patientunitstayid, cplitemoffset |
| careplangoal | BASE TABLE | 504139 | 7 | 0 | cplgoalid, patientunitstayid, cplgoaloffset |
| careplaninfectiousdisease | BASE TABLE | 8056 | 8 | 0 | cplinfectid, patientunitstayid, cplinfectdiseaseoffset |
| customlab | BASE TABLE | 1082 | 7 | 0 | customlabid, patientunitstayid, labotheroffset... |
| diagnosis | BASE TABLE | 2710672 | 7 | 0 | diagnosisid, patientunitstayid, diagnosisoffset |
| hospital | BASE TABLE | 208 | 4 | 0 | hospitalid |
| infusiondrug | BASE TABLE | 4803719 | 9 | 0 | infusiondrugid, patientunitstayid, infusionoffset |
| intakeoutput | BASE TABLE | 12030289 | 12 | 1 | intakeoutputid, patientunitstayid, intakeoutputoffset... |
| lab | BASE TABLE | 39132531 | 10 | 3 | labid, patientunitstayid, labresultoffset... |
| medication | BASE TABLE | 7301853 | 15 | 0 | medicationid, patientunitstayid, drugorderoffset... |
| microlab | BASE TABLE | 16996 | 7 | 0 | microlabid, patientunitstayid, culturetakenoffset |
| note | BASE TABLE | 2254179 | 8 | 0 | noteid, patientunitstayid, noteoffset... |
| nurseassessment | BASE TABLE | 15602498 | 8 | 0 | nurseassessid, patientunitstayid, nurseassessoffset... |
| nursecare | BASE TABLE | 8311132 | 8 | 0 | nursecareid, patientunitstayid, nursecareoffset... |
| nursecharting | BASE TABLE | 151604232 | 8 | 0 | nursingchartid, patientunitstayid, nursingchartoffset... |
| pasthistory | BASE TABLE | 1149180 | 8 | 0 | pasthistoryid, patientunitstayid, pasthistoryoffset... |
| patient | BASE TABLE | 200859 | 29 | 3 | patientunitstayid, patienthealthsystemstayid, hospitalid... |
| physicalexam | BASE TABLE | 9212316 | 6 | 0 | physicalexamid, patientunitstayid, physicalexamoffset |
| respiratorycare | BASE TABLE | 865381 | 34 | 3 | respcareid, patientunitstayid, respcarestatusoffset... |
| respiratorycharting | BASE TABLE | 20168176 | 7 | 0 | respchartid, patientunitstayid, respchartoffset... |
| treatment | BASE TABLE | 3688745 | 5 | 0 | treatmentid, patientunitstayid, treatmentoffset |
| vitalaperiodic | BASE TABLE | 25075074 | 13 | 0 | vitalaperiodicid, patientunitstayid, observationoffset... |
| vitalperiodic | BASE TABLE | 146671642 | 19 | 2 | vitalperiodicid, patientunitstayid, observationoffset... |

#### Key Feature-Rich Tables

**admissiondrug**
- High potential: drugdosage
- Medium potential: admissiondrugid, patientunitstayid, drugoffset, drugenteredoffset, drugnotetype, specialtytype, usertype, drugname, drughiclseqno

**allergy**
- Medium potential: allergyid, patientunitstayid, allergyoffset, allergyenteredoffset, allergynotetype, specialtytype, usertype, drugname, allergytype, allergyname

**apacheapsvar**
- High potential: temperature, respiratoryrate, heartrate
- Medium potential: apacheapsvarid, patientunitstayid, intubated, vent, dialysis, eyes, motor, verbal, meds, urine

**apachepatientresult**
- High potential: apachepatientresultsid
- Medium potential: patientunitstayid, acutephysiologyscore, apachescore, predictediculos, actualiculos, predictedhospitallos, actualhospitallos, preopmi, preopcardiaccath, ptcawithin24h

**apachepredvar**
- High potential: bedcount, graftcount, age, managementsystem
- Medium potential: apachepredvarid, patientunitstayid, sicuday, saps3day1, saps3today, saps3yesterday, gender, teachtype, region, admitsource

### Schema: `eicu_mortality_100_2h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_mortality_100_2h_cohort | BASE TABLE | 170115 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_mortality_100_2h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `eicu_mortality_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __eicu_mortality_100_cohort | BASE TABLE | 170115 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__eicu_mortality_100_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `mimic_core`

- **Tables**: 3
- **Total Columns**: 28
- **High Potential Features**: 1
- **Medium Potential Features**: 18

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| admissions | BASE TABLE | 0 | 15 | 0 | subject_id, hadm_id, hospital_expire_flag |
| patients | BASE TABLE | 0 | 6 | 1 | subject_id, anchor_age, anchor_year |
| transfers | BASE TABLE | 0 | 7 | 0 | subject_id, hadm_id, transfer_id |

#### Key Feature-Rich Tables

**admissions**
- Medium potential: subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, edregtime, edouttime, hospital_expire_flag

**transfers**
- Medium potential: subject_id, hadm_id, transfer_id, eventtype, intime, outtime

### Schema: `mimic_hosp`

- **Tables**: 17
- **Total Columns**: 187
- **High Potential Features**: 6
- **Medium Potential Features**: 93

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| d_hcpcs | BASE TABLE | 0 | 4 | 0 | category |
| d_icd_diagnoses | BASE TABLE | 0 | 3 | 0 | icd_version |
| d_icd_procedures | BASE TABLE | 0 | 3 | 0 | icd_version |
| d_labitems | BASE TABLE | 0 | 5 | 0 | itemid |
| diagnoses_icd | BASE TABLE | 0 | 5 | 0 | subject_id, hadm_id, seq_num... |
| drgcodes | BASE TABLE | 0 | 7 | 0 | subject_id, hadm_id, drg_severity... |
| emar | BASE TABLE | 0 | 11 | 0 | subject_id, hadm_id, emar_seq... |
| emar_detail | BASE TABLE | 0 | 33 | 0 | subject_id, emar_seq, parent_field_ordinal... |
| hcpcsevents | BASE TABLE | 0 | 6 | 0 | subject_id, hadm_id, seq_num |
| labevents | BASE TABLE | 0 | 15 | 1 | labevent_id, subject_id, hadm_id... |
| microbiologyevents | BASE TABLE | 0 | 24 | 1 | microevent_id, subject_id, hadm_id... |
| pharmacy | BASE TABLE | 0 | 27 | 3 | subject_id, hadm_id, pharmacy_id... |
| poe | BASE TABLE | 0 | 11 | 0 | poe_seq, subject_id, hadm_id |
| poe_detail | BASE TABLE | 0 | 5 | 0 | poe_seq, subject_id |
| prescriptions | BASE TABLE | 0 | 17 | 1 | subject_id, hadm_id, pharmacy_id... |
| procedures_icd | BASE TABLE | 0 | 6 | 0 | subject_id, hadm_id, seq_num... |
| services | BASE TABLE | 0 | 5 | 0 | subject_id, hadm_id |

#### Key Feature-Rich Tables

**emar**
- Medium potential: subject_id, hadm_id, emar_seq, pharmacy_id, charttime, scheduletime, storetime

**emar_detail**
- Medium potential: subject_id, emar_seq, parent_field_ordinal, administration_type, pharmacy_id, barcode_type

**labevents**
- High potential: valuenum
- Medium potential: labevent_id, subject_id, hadm_id, specimen_id, itemid, charttime, storetime, ref_range_lower, ref_range_upper

**microbiologyevents**
- High potential: dilution_value
- Medium potential: microevent_id, subject_id, hadm_id, micro_specimen_id, chartdate, charttime, spec_itemid, spec_type_desc, test_seq, storedate

**pharmacy**
- High potential: basal_rate, doses_per_24_hrs, expiration_value
- Medium potential: subject_id, hadm_id, pharmacy_id, starttime, stoptime, proc_type, entertime, verifiedtime, infusion_type, duration

### Schema: `mimic_icu`

- **Tables**: 7
- **Total Columns**: 96
- **High Potential Features**: 15
- **Medium Potential Features**: 61

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| chartevents | BASE TABLE | 0 | 10 | 1 | subject_id, hadm_id, stay_id... |
| d_items | BASE TABLE | 0 | 9 | 2 | itemid, lownormalvalue, highnormalvalue |
| datetimeevents | BASE TABLE | 0 | 9 | 0 | subject_id, hadm_id, stay_id... |
| icustays | BASE TABLE | 0 | 8 | 0 | subject_id, hadm_id, stay_id... |
| inputevents | BASE TABLE | 0 | 26 | 6 | subject_id, hadm_id, stay_id... |
| outputevents | BASE TABLE | 0 | 8 | 1 | subject_id, hadm_id, stay_id... |
| procedureevents | BASE TABLE | 0 | 26 | 5 | subject_id, hadm_id, stay_id... |

#### Key Feature-Rich Tables

**chartevents**
- High potential: valuenum
- Medium potential: subject_id, hadm_id, stay_id, charttime, storetime, itemid, warning

**d_items**
- High potential: lownormalvalue, highnormalvalue
- Medium potential: itemid, label, unitname, param_type

**datetimeevents**
- Medium potential: subject_id, hadm_id, stay_id, charttime, storetime, itemid, value, warning

**icustays**
- Medium potential: subject_id, hadm_id, stay_id, intime, outtime, los

**inputevents**
- High potential: amount, rate, patientweight, totalamount, originalamount, originalrate
- Medium potential: subject_id, hadm_id, stay_id, starttime, endtime, storetime, itemid, orderid, linkorderid, ordercategoryname

### Schema: `mimiciii`

- **Tables**: 273
- **Total Columns**: 3825
- **High Potential Features**: 228
- **Medium Potential Features**: 2643

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| admissions | BASE TABLE | 58976 | 20 | 0 | row_id, subject_id, hadm_id... |
| callout | BASE TABLE | 34499 | 25 | 0 | row_id, subject_id, hadm_id... |
| caregivers | BASE TABLE | 7567 | 5 | 0 | row_id, cgid, mimic_id |
| chartevents | BASE TABLE | 330712483 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_1 | BASE TABLE | 22204 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_10 | BASE TABLE | 2072743 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_100 | BASE TABLE | 3561885 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_101 | BASE TABLE | 1174868 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_102 | BASE TABLE | 293325 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_103 | BASE TABLE | 249776 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_104 | BASE TABLE | 3010424 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_105 | BASE TABLE | 679113 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_106 | BASE TABLE | 5208156 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_107 | BASE TABLE | 673719 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_108 | BASE TABLE | 2785057 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_109 | BASE TABLE | 1687886 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_11 | BASE TABLE | 178 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_110 | BASE TABLE | 2445808 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_111 | BASE TABLE | 2433936 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_112 | BASE TABLE | 4449487 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_113 | BASE TABLE | 1676872 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_114 | BASE TABLE | 476670 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_115 | BASE TABLE | 1621393 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_116 | BASE TABLE | 2309226 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_117 | BASE TABLE | 690295 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_118 | BASE TABLE | 1009254 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_119 | BASE TABLE | 803881 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_12 | BASE TABLE | 892239 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_120 | BASE TABLE | 2367096 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_121 | BASE TABLE | 1360432 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_122 | BASE TABLE | 982518 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_123 | BASE TABLE | 655454 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_124 | BASE TABLE | 1807316 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_125 | BASE TABLE | 34909 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_126 | BASE TABLE | 1378959 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_127 | BASE TABLE | 178112 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_128 | BASE TABLE | 1772387 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_129 | BASE TABLE | 1802684 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_13 | BASE TABLE | 1181039 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_130 | BASE TABLE | 1622363 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_131 | BASE TABLE | 43749 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_132 | BASE TABLE | 601818 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_133 | BASE TABLE | 2085994 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_134 | BASE TABLE | 5266438 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_135 | BASE TABLE | 1573583 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_136 | BASE TABLE | 3870155 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_137 | BASE TABLE | 719203 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_138 | BASE TABLE | 1600973 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_139 | BASE TABLE | 1687615 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_14 | BASE TABLE | 1136214 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_140 | BASE TABLE | 1146136 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_141 | BASE TABLE | 1619782 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_142 | BASE TABLE | 204405 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_143 | BASE TABLE | 725866 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_144 | BASE TABLE | 302 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_145 | BASE TABLE | 976252 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_146 | BASE TABLE | 649745 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_147 | BASE TABLE | 1804988 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_148 | BASE TABLE | 33554 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_149 | BASE TABLE | 1375295 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_15 | BASE TABLE | 3418901 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_150 | BASE TABLE | 174222 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_151 | BASE TABLE | 1769925 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_152 | BASE TABLE | 1796313 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_153 | BASE TABLE | 18753 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_154 | BASE TABLE | 0 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_155 | BASE TABLE | 2762225 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_156 | BASE TABLE | 431909 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_157 | BASE TABLE | 2023672 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_158 | BASE TABLE | 0 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_159 | BASE TABLE | 1149788 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_16 | BASE TABLE | 1198681 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_160 | BASE TABLE | 1149537 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_161 | BASE TABLE | 1156173 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_162 | BASE TABLE | 648200 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_163 | BASE TABLE | 526472 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_164 | BASE TABLE | 1290488 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_165 | BASE TABLE | 1289885 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_166 | BASE TABLE | 1292916 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_167 | BASE TABLE | 208 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_168 | BASE TABLE | 2737105 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_169 | BASE TABLE | 466344 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_17 | BASE TABLE | 1111444 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_170 | BASE TABLE | 2671816 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_171 | BASE TABLE | 3262258 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_172 | BASE TABLE | 4068153 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_173 | BASE TABLE | 765274 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_174 | BASE TABLE | 1139355 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_175 | BASE TABLE | 1983602 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_176 | BASE TABLE | 2185541 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_177 | BASE TABLE | 3552998 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_178 | BASE TABLE | 2289753 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_179 | BASE TABLE | 1610057 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_18 | BASE TABLE | 3216866 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_180 | BASE TABLE | 395061 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_181 | BASE TABLE | 4797667 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_182 | BASE TABLE | 3317320 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_183 | BASE TABLE | 2611372 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_184 | BASE TABLE | 4089672 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_185 | BASE TABLE | 1559194 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_186 | BASE TABLE | 2089736 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_187 | BASE TABLE | 1465008 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_188 | BASE TABLE | 717326 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_189 | BASE TABLE | 2933324 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_19 | BASE TABLE | 1669170 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_190 | BASE TABLE | 3110922 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_191 | BASE TABLE | 618565 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_192 | BASE TABLE | 1165 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_193 | BASE TABLE | 1849287 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_194 | BASE TABLE | 4059618 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_195 | BASE TABLE | 3154406 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_196 | BASE TABLE | 2873716 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_197 | BASE TABLE | 2659813 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_198 | BASE TABLE | 3763143 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_199 | BASE TABLE | 1718572 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_2 | BASE TABLE | 737224 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_20 | BASE TABLE | 818852 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_200 | BASE TABLE | 2662741 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_201 | BASE TABLE | 1605091 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_202 | BASE TABLE | 5553957 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_203 | BASE TABLE | 5627006 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_204 | BASE TABLE | 716961 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_205 | BASE TABLE | 816157 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_206 | BASE TABLE | 1862707 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_207 | BASE TABLE | 2313406 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_21 | BASE TABLE | 974476 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_22 | BASE TABLE | 2544519 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_23 | BASE TABLE | 9458 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_24 | BASE TABLE | 3236350 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_25 | BASE TABLE | 1837071 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_26 | BASE TABLE | 954139 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_27 | BASE TABLE | 1456328 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_28 | BASE TABLE | 945638 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_29 | BASE TABLE | 1545176 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_3 | BASE TABLE | 56235 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_30 | BASE TABLE | 955452 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_31 | BASE TABLE | 5180809 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_32 | BASE TABLE | 3303151 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_33 | BASE TABLE | 3676785 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_34 | BASE TABLE | 7811955 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_35 | BASE TABLE | 783762 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_36 | BASE TABLE | 402022 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_37 | BASE TABLE | 786544 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_38 | BASE TABLE | 349997 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_39 | BASE TABLE | 1032728 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_4 | BASE TABLE | 1442406 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_40 | BASE TABLE | 1589945 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_41 | BASE TABLE | 950038 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_42 | BASE TABLE | 1579681 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_43 | BASE TABLE | 1553537 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_44 | BASE TABLE | 392595 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_45 | BASE TABLE | 1155571 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_46 | BASE TABLE | 985720 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_47 | BASE TABLE | 774157 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_48 | BASE TABLE | 917780 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_49 | BASE TABLE | 4595328 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_5 | BASE TABLE | 878442 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_50 | BASE TABLE | 852968 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_51 | BASE TABLE | 883558 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_52 | BASE TABLE | 3205052 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_53 | BASE TABLE | 1022222 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_54 | BASE TABLE | 1641889 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_55 | BASE TABLE | 1654497 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_56 | BASE TABLE | 784361 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_57 | BASE TABLE | 913144 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_58 | BASE TABLE | 787137 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_59 | BASE TABLE | 1251345 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_6 | BASE TABLE | 1659172 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_60 | BASE TABLE | 774906 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_61 | BASE TABLE | 962191 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_62 | BASE TABLE | 3386719 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_63 | BASE TABLE | 580529 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_64 | BASE TABLE | 666496 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_65 | BASE TABLE | 2048955 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_66 | BASE TABLE | 954354 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_67 | BASE TABLE | 437 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_68 | BASE TABLE | 1447426 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_69 | BASE TABLE | 3418917 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_7 | BASE TABLE | 636690 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_70 | BASE TABLE | 2135542 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_71 | BASE TABLE | 774213 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_72 | BASE TABLE | 300570 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_73 | BASE TABLE | 1042512 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_74 | BASE TABLE | 378549 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_75 | BASE TABLE | 772277 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_76 | BASE TABLE | 773891 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_77 | BASE TABLE | 376047 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_78 | BASE TABLE | 740176 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_79 | BASE TABLE | 1099236 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_8 | BASE TABLE | 285028 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_80 | BASE TABLE | 933238 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_81 | BASE TABLE | 20754 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_82 | BASE TABLE | 727719 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_83 | BASE TABLE | 937064 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_84 | BASE TABLE | 1049706 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_85 | BASE TABLE | 952177 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_86 | BASE TABLE | 321012 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_87 | BASE TABLE | 3464326 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_88 | BASE TABLE | 5925297 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_89 | BASE TABLE | 1716561 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_9 | BASE TABLE | 2096678 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_90 | BASE TABLE | 1417336 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_91 | BASE TABLE | 803816 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_92 | BASE TABLE | 423898 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_93 | BASE TABLE | 592344 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_94 | BASE TABLE | 250247 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_95 | BASE TABLE | 2955851 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_96 | BASE TABLE | 808725 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_97 | BASE TABLE | 1083809 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_98 | BASE TABLE | 1281335 | 16 | 1 | row_id, subject_id, hadm_id... |
| chartevents_99 | BASE TABLE | 2261065 | 16 | 1 | row_id, subject_id, hadm_id... |
| cptevents | BASE TABLE | 573146 | 13 | 0 | row_id, subject_id, hadm_id... |
| d_cpt | BASE TABLE | 134 | 10 | 0 | row_id, category, mincodeinsubsection... |
| d_icd_diagnoses | BASE TABLE | 14567 | 5 | 0 | row_id, mimic_id |
| d_icd_procedures | BASE TABLE | 3882 | 5 | 0 | row_id, mimic_id |
| d_items | BASE TABLE | 12487 | 11 | 0 | row_id, itemid, conceptid... |
| d_labitems | BASE TABLE | 753 | 7 | 0 | row_id, itemid, mimic_id |
| datetimeevents | BASE TABLE | 4485937 | 15 | 0 | row_id, subject_id, hadm_id... |
| diagnoses_icd | BASE TABLE | 651047 | 6 | 0 | row_id, subject_id, hadm_id... |
| drgcodes | BASE TABLE | 125557 | 9 | 0 | row_id, subject_id, hadm_id... |
| gcpt_admission_location_to_concept | BASE TABLE | 9 | 3 | 0 | concept_id, mimic_id |
| gcpt_admission_type_to_concept | BASE TABLE | 4 | 3 | 0 | visit_concept_id, mimic_id |
| gcpt_admissions_diagnosis_to_concept | BASE TABLE | 1523 | 4 | 0 | concept_id, mimic_id |
| gcpt_atb_to_concept | BASE TABLE | 30 | 3 | 0 | mimic_id |
| gcpt_care_site | BASE TABLE | 30 | 6 | 0 | place_of_service_concept_id, visit_detail_concept_id, mimic_id |
| gcpt_chart_label_to_concept | BASE TABLE | 319 | 8 | 2 | value_lb, value_ub, measurement_concept_id... |
| gcpt_chart_observation_to_concept | BASE TABLE | 57 | 5 | 0 | concept_id, itemid, mimic_id |
| gcpt_continuous_unit_carevue | BASE TABLE | 12 | 3 | 0 | mimic_id |
| gcpt_cpt4_to_concept | BASE TABLE | 23 | 4 | 0 | procedure_concept_id, omop_mapping_is_sure, mimic_id |
| gcpt_cv_input_label_to_concept | BASE TABLE | 2782 | 5 | 0 | item_id, original_route, concept_id... |
| gcpt_datetimeevents_to_concept | BASE TABLE | 155 | 6 | 0 | observation_concept_id, itemid, observation_source_concept_id... |
| gcpt_derived_to_concept | BASE TABLE | 8 | 4 | 0 | itemid, concept_id, mimic_id |
| gcpt_discharge_location_to_concept | BASE TABLE | 17 | 4 | 0 | concept_id, mimic_id |
| gcpt_drgcode_to_concept | BASE TABLE | 1355 | 6 | 0 | non_standard_concept_id, standard_concept_id, mimic_id |
| gcpt_ethnicity_to_concept | BASE TABLE | 41 | 5 | 0 | race_concept_id, ethnicity_concept_id, priority... |
| gcpt_heart_rhythm_to_concept | BASE TABLE | 47 | 3 | 0 | concept_id, mimic_id |
| gcpt_inputevents_drug_to_concept | BASE TABLE | 108 | 4 | 0 | concept_id, itemid, mimic_id |
| gcpt_insurance_to_concept | BASE TABLE | 5 | 3 | 0 | concept_id, mimic_id |
| gcpt_lab_label_to_concept | BASE TABLE | 118 | 3 | 0 | concept_id, mimic_id |
| gcpt_lab_unit_to_concept | BASE TABLE | 71 | 3 | 0 | concept_id, mimic_id |
| gcpt_lab_value_to_concept | BASE TABLE | 3703 | 4 | 2 | value_as_concept_id, value_as_num, mimic_id |
| gcpt_labs_from_chartevents_to_concept | BASE TABLE | 80 | 4 | 0 | measurement_type_concept_id, mimic_id |
| gcpt_labs_specimen_to_concept | BASE TABLE | 17 | 4 | 0 | specimen_concept_id, mimic_id |
| gcpt_language_to_concept | BASE TABLE | 75 | 3 | 0 | concept_id, mimic_id |
| gcpt_map_route_to_concept | BASE TABLE | 28 | 4 | 0 | concept_id, mimic_id |
| gcpt_marital_status_to_concept | BASE TABLE | 7 | 3 | 0 | concept_id, mimic_id |
| gcpt_microbiology_specimen_to_concept | BASE TABLE | 88 | 4 | 0 | specimen_concept_id, mimic_id |
| gcpt_mv_input_label_to_concept | BASE TABLE | 159 | 5 | 0 | item_id, route, concept_id... |
| gcpt_note_category_to_concept | BASE TABLE | 15 | 4 | 0 | concept_id, category_id, mimic_id |
| gcpt_note_section_to_concept | BASE TABLE | 1113 | 6 | 0 | section_id, category_id, mimic_id |
| gcpt_org_name_to_concept | BASE TABLE | 362 | 3 | 0 | snomed, mimic_id |
| gcpt_output_label_to_concept | BASE TABLE | 1154 | 4 | 0 | item_id, concept_id, mimic_id |
| gcpt_prescriptions_ndcisnullzero_to_concept | BASE TABLE | 1354 | 4 | 0 | concept_id, mimic_id |
| gcpt_procedure_to_concept | BASE TABLE | 113 | 4 | 0 | item_id, snomed, concept_id... |
| gcpt_religion_to_concept | BASE TABLE | 20 | 3 | 0 | concept_id, mimic_id |
| gcpt_resistance_to_concept | BASE TABLE | 4 | 3 | 1 | value_as_concept_id, mimic_id |
| gcpt_route_to_concept | BASE TABLE | 78 | 3 | 0 | concept_id, mimic_id |
| gcpt_seq_num_to_concept | BASE TABLE | 44 | 3 | 0 | seq_num, concept_id, mimic_id |
| gcpt_spec_type_to_concept | BASE TABLE | 87 | 4 | 0 | snomed, mimic_id |
| gcpt_unit_doseera_concept_id | BASE TABLE | 27 | 6 | 1 | unit_concept_id, temporal_unit_concept_id, mimic_id |
| icustays | BASE TABLE | 61532 | 13 | 0 | row_id, subject_id, hadm_id... |
| inputevents_cv | BASE TABLE | 17527935 | 23 | 4 | row_id, subject_id, hadm_id... |
| inputevents_mv | BASE TABLE | 3618991 | 32 | 6 | row_id, subject_id, hadm_id... |
| labevents | BASE TABLE | 27854055 | 10 | 1 | row_id, subject_id, hadm_id... |
| microbiologyevents | BASE TABLE | 631726 | 17 | 1 | row_id, subject_id, hadm_id... |
| noteevents | BASE TABLE | 2083180 | 12 | 0 | row_id, subject_id, hadm_id... |
| outputevents | BASE TABLE | 4349218 | 14 | 1 | row_id, subject_id, hadm_id... |
| patients | BASE TABLE | 46520 | 9 | 0 | row_id, subject_id, expire_flag... |
| prescriptions | BASE TABLE | 4156450 | 20 | 0 | row_id, subject_id, hadm_id... |
| procedureevents_mv | BASE TABLE | 258066 | 26 | 1 | row_id, subject_id, hadm_id... |
| procedures_icd | BASE TABLE | 240095 | 6 | 0 | row_id, subject_id, hadm_id... |
| services | BASE TABLE | 73343 | 7 | 0 | row_id, subject_id, hadm_id... |
| transfers | BASE TABLE | 261897 | 14 | 0 | row_id, subject_id, hadm_id... |

#### Key Feature-Rich Tables

**admissions**
- Medium potential: row_id, subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, edregtime, edouttime, hospital_expire_flag

**callout**
- Medium potential: row_id, subject_id, hadm_id, submit_wardid, curr_wardid, callout_wardid, request_tele, request_resp, request_cdiff, request_mrsa

**chartevents**
- High potential: valuenum
- Medium potential: row_id, subject_id, hadm_id, icustay_id, itemid, charttime, storetime, cgid, warning, error

**chartevents_1**
- High potential: valuenum
- Medium potential: row_id, subject_id, hadm_id, icustay_id, itemid, charttime, storetime, cgid, warning, error

**chartevents_10**
- High potential: valuenum
- Medium potential: row_id, subject_id, hadm_id, icustay_id, itemid, charttime, storetime, cgid, warning, error

### Schema: `mimiciv_bsi_100_0.25h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mimiciv_bsi_100_0.25h_cohort | BASE TABLE | Access denied or table empty | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__mimiciv_bsi_100_0.25h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y

### Schema: `mimiciv_bsi_100_0.5h_test`

- **Tables**: 1
- **Total Columns**: 8
- **High Potential Features**: 0
- **Medium Potential Features**: 8

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mimiciv_bsi_100_0.5h_cohort | BASE TABLE | Access denied or table empty | 8 | 0 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**__mimiciv_bsi_100_0.5h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y, last_years

### Schema: `mimiciv_bsi_100_2h_test`

- **Tables**: 1
- **Total Columns**: 8
- **High Potential Features**: 0
- **Medium Potential Features**: 8

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mimiciv_bsi_100_2h_cohort | BASE TABLE | 12139 | 8 | 0 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**__mimiciv_bsi_100_2h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y, last_years

### Schema: `mimiciv_bsi_100_4h_test`

- **Tables**: 1
- **Total Columns**: 8
- **High Potential Features**: 0
- **Medium Potential Features**: 8

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mimiciv_bsi_100_4h_cohort | BASE TABLE | 12139 | 8 | 0 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**__mimiciv_bsi_100_4h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y, last_years

### Schema: `mimiciv_bsi_100_8h_test`

- **Tables**: 1
- **Total Columns**: 8
- **High Potential Features**: 0
- **Medium Potential Features**: 8

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mimiciv_bsi_100_8h_cohort | BASE TABLE | 12139 | 8 | 0 | example_id, person_id, y... |

#### Key Feature-Rich Tables

**__mimiciv_bsi_100_8h_cohort**
- Medium potential: example_id, person_id, start_date, start_datetime, end_date, end_datetime, y, last_years

### Schema: `mimiciv_hosp`

- **Tables**: 22
- **Total Columns**: 229
- **High Potential Features**: 7
- **Medium Potential Features**: 115

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| admissions | BASE TABLE | 431231 | 16 | 0 | subject_id, hadm_id, hospital_expire_flag |
| d_hcpcs | BASE TABLE | 89200 | 4 | 0 | category |
| d_icd_diagnoses | BASE TABLE | 109775 | 3 | 0 | icd_version |
| d_icd_procedures | BASE TABLE | 85257 | 3 | 0 | icd_version |
| d_labitems | BASE TABLE | 1622 | 4 | 0 | itemid |
| diagnoses_icd | BASE TABLE | 4756326 | 5 | 0 | subject_id, hadm_id, seq_num... |
| drgcodes | BASE TABLE | 604377 | 7 | 0 | subject_id, hadm_id, drg_severity... |
| emar | BASE TABLE | 26850359 | 12 | 0 | subject_id, hadm_id, emar_seq... |
| emar_detail | BASE TABLE | 54744789 | 33 | 0 | subject_id, emar_seq, pharmacy_id |
| hcpcsevents | BASE TABLE | 150771 | 6 | 0 | subject_id, hadm_id, seq_num |
| labevents | BASE TABLE | 118171367 | 16 | 1 | labevent_id, subject_id, hadm_id... |
| microbiologyevents | BASE TABLE | 3228713 | 25 | 1 | microevent_id, subject_id, hadm_id... |
| omr | BASE TABLE | 6439169 | 5 | 0 | subject_id, seq_num |
| patients | BASE TABLE | 299712 | 6 | 1 | subject_id, anchor_age, anchor_year |
| pharmacy | BASE TABLE | 13584514 | 27 | 3 | subject_id, hadm_id, pharmacy_id... |
| poe | BASE TABLE | 39366291 | 12 | 0 | poe_seq, subject_id, hadm_id |
| poe_detail | BASE TABLE | 3879418 | 5 | 0 | poe_seq, subject_id |
| prescriptions | BASE TABLE | 15416708 | 21 | 1 | subject_id, hadm_id, pharmacy_id... |
| procedures_icd | BASE TABLE | 669186 | 6 | 0 | subject_id, hadm_id, seq_num... |
| provider | BASE TABLE | 40508 | 1 | 0 |  |
| services | BASE TABLE | 468029 | 5 | 0 | subject_id, hadm_id |
| transfers | BASE TABLE | 1890972 | 7 | 0 | subject_id, hadm_id, transfer_id |

#### Key Feature-Rich Tables

**admissions**
- Medium potential: subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, edregtime, edouttime, hospital_expire_flag

**emar**
- Medium potential: subject_id, hadm_id, emar_seq, pharmacy_id, charttime, scheduletime, storetime

**labevents**
- High potential: valuenum
- Medium potential: labevent_id, subject_id, hadm_id, specimen_id, itemid, charttime, storetime, ref_range_lower, ref_range_upper

**microbiologyevents**
- High potential: dilution_value
- Medium potential: microevent_id, subject_id, hadm_id, micro_specimen_id, chartdate, charttime, spec_itemid, spec_type_desc, test_seq, storedate

**pharmacy**
- High potential: basal_rate, doses_per_24_hrs, expiration_value
- Medium potential: subject_id, hadm_id, pharmacy_id, starttime, stoptime, proc_type, entertime, verifiedtime, infusion_type, duration

### Schema: `mimiciv_icu`

- **Tables**: 9
- **Total Columns**: 113
- **High Potential Features**: 18
- **Medium Potential Features**: 73

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| caregiver | BASE TABLE | 15468 | 1 | 0 | caregiver_id |
| chartevents | BASE TABLE | 313645063 | 11 | 1 | subject_id, hadm_id, stay_id... |
| d_items | BASE TABLE | 4014 | 9 | 2 | itemid, lownormalvalue, highnormalvalue |
| datetimeevents | BASE TABLE | 7112999 | 10 | 0 | subject_id, hadm_id, stay_id... |
| icustays | BASE TABLE | 73181 | 8 | 0 | subject_id, hadm_id, stay_id... |
| ingredientevents | BASE TABLE | 11627821 | 17 | 4 | subject_id, hadm_id, stay_id... |
| inputevents | BASE TABLE | 8978893 | 26 | 6 | subject_id, hadm_id, stay_id... |
| outputevents | BASE TABLE | 4234967 | 9 | 1 | subject_id, hadm_id, stay_id... |
| procedureevents | BASE TABLE | 696092 | 22 | 4 | subject_id, hadm_id, stay_id... |

#### Key Feature-Rich Tables

**chartevents**
- High potential: valuenum
- Medium potential: subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, warning

**d_items**
- High potential: lownormalvalue, highnormalvalue
- Medium potential: itemid, label, unitname, param_type

**datetimeevents**
- Medium potential: subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, warning

**icustays**
- Medium potential: subject_id, hadm_id, stay_id, intime, outtime, los

**ingredientevents**
- High potential: amount, rate, originalamount, originalrate
- Medium potential: subject_id, hadm_id, stay_id, caregiver_id, starttime, endtime, storetime, itemid, orderid, linkorderid

### Schema: `mortality_100_2h_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __mortality_100_2h_cohort | BASE TABLE | 28973 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__mortality_100_2h_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

### Schema: `omop`

- **Tables**: 40
- **Total Columns**: 429
- **High Potential Features**: 22
- **Medium Potential Features**: 297

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| attribute_definition | BASE TABLE | 0 | 5 | 0 | attribute_definition_id, attribute_type_concept_id |
| care_site | BASE TABLE | 0 | 6 | 0 | care_site_id, place_of_service_concept_id, location_id |
| cdm_source | BASE TABLE | 0 | 10 | 0 |  |
| cohort | BASE TABLE | 0 | 4 | 0 | cohort_definition_id, subject_id |
| cohort_attribute | BASE TABLE | 0 | 7 | 2 | cohort_definition_id, subject_id, attribute_definition_id... |
| cohort_definition | BASE TABLE | 0 | 7 | 0 | cohort_definition_id, definition_type_concept_id, subject_concept_id |
| concept | BASE TABLE | 0 | 10 | 0 | concept_id |
| concept_ancestor | BASE TABLE | 0 | 4 | 2 | ancestor_concept_id, descendant_concept_id, min_levels_of_separation... |
| concept_class | BASE TABLE | 0 | 3 | 0 | concept_class_concept_id |
| concept_relationship | BASE TABLE | 0 | 6 | 0 | concept_id_1, concept_id_2 |
| concept_synonym | BASE TABLE | 0 | 3 | 1 | concept_id, language_concept_id |
| condition_era | BASE TABLE | 0 | 6 | 1 | condition_era_id, person_id, condition_concept_id... |
| condition_occurrence | BASE TABLE | 0 | 16 | 0 | condition_occurrence_id, person_id, condition_concept_id... |
| cost | BASE TABLE | 0 | 22 | 1 | cost_id, cost_event_id, cost_type_concept_id... |
| death | BASE TABLE | 0 | 7 | 0 | person_id, death_type_concept_id, cause_concept_id... |
| device_exposure | BASE TABLE | 0 | 15 | 0 | device_exposure_id, person_id, device_concept_id... |
| domain | BASE TABLE | 0 | 3 | 0 | domain_concept_id |
| dose_era | BASE TABLE | 0 | 9 | 4 | dose_era_id, person_id, drug_concept_id... |
| drug_era | BASE TABLE | 0 | 7 | 1 | drug_era_id, person_id, drug_concept_id... |
| drug_exposure | BASE TABLE | 0 | 24 | 0 | drug_exposure_id, person_id, drug_concept_id... |
| drug_strength | BASE TABLE | 0 | 12 | 4 | drug_concept_id, ingredient_concept_id, amount_value... |
| fact_relationship | BASE TABLE | 0 | 5 | 0 | domain_concept_id_1, fact_id_1, domain_concept_id_2... |
| location | BASE TABLE | 0 | 8 | 0 | location_id |
| measurement | BASE TABLE | 0 | 20 | 2 | measurement_id, person_id, measurement_concept_id... |
| metadata | BASE TABLE | 0 | 7 | 1 | metadata_concept_id, metadata_type_concept_id, value_as_concept_id |
| note | BASE TABLE | 0 | 14 | 1 | note_id, person_id, note_type_concept_id... |
| note_nlp | BASE TABLE | 0 | 17 | 0 | note_nlp_id, note_id, section_concept_id... |
| observation | BASE TABLE | 0 | 18 | 2 | observation_id, person_id, observation_concept_id... |
| observation_period | BASE TABLE | 0 | 7 | 0 | observation_period_id, person_id, period_type_concept_id |
| payer_plan_period | BASE TABLE | 0 | 17 | 0 | payer_plan_period_id, person_id, payer_concept_id... |
| person | BASE TABLE | 0 | 18 | 0 | person_id, gender_concept_id, year_of_birth... |
| procedure_occurrence | BASE TABLE | 0 | 14 | 0 | procedure_occurrence_id, person_id, procedure_concept_id... |
| provider | BASE TABLE | 0 | 13 | 0 | provider_id, specialty_concept_id, care_site_id... |
| relationship | BASE TABLE | 0 | 6 | 0 | relationship_concept_id |
| source_to_concept_map | BASE TABLE | 0 | 9 | 0 | source_concept_id, target_concept_id |
| specimen | BASE TABLE | 0 | 15 | 0 | specimen_id, person_id, specimen_concept_id... |
| visit_detail | BASE TABLE | 0 | 23 | 0 | visit_detail_id, person_id, visit_detail_concept_id... |
| visit_detail_assign | BASE TABLE | 11 | 8 | 0 | visit_detail_id, visit_occurrence_id |
| visit_occurrence | BASE TABLE | 0 | 19 | 0 | visit_occurrence_id, person_id, visit_concept_id... |
| vocabulary | BASE TABLE | 0 | 5 | 0 | vocabulary_concept_id |

#### Key Feature-Rich Tables

**cohort_attribute**
- High potential: value_as_number, value_as_concept_id
- Medium potential: cohort_definition_id, subject_id, cohort_start_date, cohort_end_date, attribute_definition_id

**condition_era**
- High potential: condition_occurrence_count
- Medium potential: condition_era_id, person_id, condition_concept_id, condition_era_start_date, condition_era_end_date

**condition_occurrence**
- Medium potential: condition_occurrence_id, person_id, condition_concept_id, condition_start_date, condition_start_datetime, condition_end_date, condition_end_datetime, condition_type_concept_id, provider_id, visit_occurrence_id

**cost**
- High potential: amount_allowed
- Medium potential: cost_id, cost_event_id, cost_type_concept_id, currency_concept_id, total_charge, total_cost, total_paid, paid_by_payer, paid_by_patient, paid_patient_copay

**death**
- Medium potential: person_id, death_date, death_datetime, death_type_concept_id, cause_concept_id, cause_source_concept_id

### Schema: `public`

- **Tables**: 5
- **Total Columns**: 18
- **High Potential Features**: 0
- **Medium Potential Features**: 4

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| drug_converter | BASE TABLE | 258 | 3 | 0 | index |
| icd10_pheno | BASE TABLE | 1925 | 5 | 0 | index |
| icd9_to_icd10 | BASE TABLE | 23912 | 3 | 0 | index |
| medical_hist_converter | BASE TABLE | 41 | 3 | 0 | index |
| mimic_to_eicu_converter | BASE TABLE | 91 | 4 | 0 |  |

#### Key Feature-Rich Tables

No feature-rich tables found in this schema.

### Schema: `reconstruction_eicu_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __reconstruction_eicu_100_cohort | BASE TABLE | 200859 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__reconstruction_eicu_100_cohort**
- Medium potential: example_id, person_id, start_date, end_date, start_datetime, end_datetime, y

### Schema: `reconstruction_less_features_100_test`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __reconstruction_less_features_100_cohort | BASE TABLE | 28973 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**__reconstruction_less_features_100_cohort**
- Medium potential: example_id, person_id, start_date, end_date, start_datetime, end_datetime, y

### Schema: `reconstruction_mimiciv_100_test`

- **Tables**: 1
- **Total Columns**: 8
- **High Potential Features**: 0
- **Medium Potential Features**: 8

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| __reconstruction_mimiciv_100_cohort | BASE TABLE | 73181 | 8 | 0 | example_id, person_id, last_years... |

#### Key Feature-Rich Tables

**__reconstruction_mimiciv_100_cohort**
- Medium potential: example_id, person_id, start_date, end_date, start_datetime, end_datetime, last_years, y

### Schema: `temp_eicu_20250903_200943`

- **Tables**: 1
- **Total Columns**: 7
- **High Potential Features**: 0
- **Medium Potential Features**: 7

#### Tables Overview

| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |
|------------|------|------|---------|------------------------|------------------------|
| simplified_cohort | BASE TABLE | 200450 | 7 | 0 | example_id, person_id, y |

#### Key Feature-Rich Tables

**simplified_cohort**
- Medium potential: example_id, person_id, start_datetime, start_date, end_datetime, end_date, y

## Feature Extraction Recommendations

### eICU Recommended Tables for Feature Extraction:
- **eicu_crd.apachepredvar** (50 potential features, 171177 rows)
- **eicu_crd.apacheapsvar** (26 potential features, 171177 rows)
- **mimiciii.inputevents_mv** (25 potential features, 3618991 rows)
- **eicu_crd.respiratorycare** (23 potential features, 865381 rows)
- **mimic_icu.inputevents** (21 potential features, 0 rows)
- **mimiciv_icu.inputevents** (21 potential features, 8978893 rows)
- **mimic_icu.procedureevents** (20 potential features, 0 rows)
- **eicu_crd.vitalperiodic** (19 potential features, 146671642 rows)
- **mimic_hosp.microbiologyevents** (19 potential features, 0 rows)
- **mimiciii.callout** (19 potential features, 34499 rows)

### MIMIC-IV Recommended Tables for Feature Extraction:
- **eicu_crd.apachepredvar** (50 potential features, 171177 rows)
- **eicu_crd.apacheapsvar** (26 potential features, 171177 rows)
- **mimiciii.inputevents_mv** (25 potential features, 3618991 rows)
- **eicu_crd.respiratorycare** (23 potential features, 865381 rows)
- **mimic_icu.inputevents** (21 potential features, 0 rows)
- **mimiciv_icu.inputevents** (21 potential features, 8978893 rows)
- **mimic_icu.procedureevents** (20 potential features, 0 rows)
- **eicu_crd.vitalperiodic** (19 potential features, 146671642 rows)
- **mimic_hosp.microbiologyevents** (19 potential features, 0 rows)
- **mimiciii.callout** (19 potential features, 34499 rows)

---
*Report generated by explore_databases.py*
*Raw data available in: database_structure_data_20250903_202006.json*
