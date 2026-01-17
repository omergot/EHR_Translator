# Unit Conversion Requirements for MIMIC-IV vs eICU-CRD

## Summary

Out of 40 BSI features:
- **26 features (65.0%)** have the same units - ✅ No conversion needed
- **14 features (35.0%)** have different units - ⚠️ **Requires conversion**

---

## ✅ Features with Same Units (26)

No conversion needed for these features:

| Index | Feature | Unit |
|-------|---------|------|
| 1 | Albumin | g/dL |
| 4 | Bilirubin | mg/dL |
| 6 | Creatinine | mg/dL |
| 7 | GCS - Eye Opening | score |
| 8 | GCS - Motor Response | score |
| 9 | GCS - Verbal Response | score |
| 10 | Head of Bed | degrees |
| 11 | Heart Rate | bpm |
| 12 | Hematocrit | % |
| 14 | INR(PT) | ratio |
| 15 | Inspired O2 Fraction | fraction |
| 16 | Lactate | mmol/L |
| 17 | Lymphocytes | % |
| 19 | MCV | fL |
| 20 | Magnesium | mg/dL |
| 21 | Mean Airway Pressure | cmH2O |
| 22 | Non Invasive Blood Pressure diastolic | mmHg |
| 23 | Non Invasive Blood Pressure mean | mmHg |
| 24 | Non Invasive Blood Pressure systolic | mmHg |
| 25 | O2 saturation pulseoxymetry | % |
| 26 | PT | sec |
| 29 | RDW | % |
| 33 | Temperature | C |
| 34 | Tidal Volume (observed) | mL |
| 37 | pH | units |
| 38 | pO2 | mm Hg |

---

## ⚠️ Features Requiring Unit Conversion (14)

### 1. **Alanine Aminotransferase (ALT)**
- MIMIC: `IU/L` → eICU: `Units/L`
- **Conversion**: IU/L and Units/L are equivalent (same thing)
- **Action**: No mathematical conversion needed, just normalize naming

### 2. **Alkaline Phosphatase**
- MIMIC: `IU/L` → eICU: `Units/L`
- **Conversion**: IU/L and Units/L are equivalent
- **Action**: No mathematical conversion needed, just normalize naming

### 3. **Asparate Aminotransferase (AST)**
- MIMIC: `IU/L` → eICU: `Units/L`
- **Conversion**: IU/L and Units/L are equivalent
- **Action**: No mathematical conversion needed, just normalize naming

### 4. **C-Reactive Protein**
- MIMIC: `mg/L` → eICU: `mg/dL`
- **Conversion**: 1 mg/L = 0.1 mg/dL
- **Action**: Multiply MIMIC values by 0.1 OR divide eICU values by 0.1

### 5. **Hemoglobin**
- MIMIC: `g/dL;g/dl` → eICU: `g/dL`
- **Conversion**: Same unit, just case difference
- **Action**: No conversion needed, standardize to g/dL

### 6. **MCHC (Mean Corpuscular Hemoglobin Concentration)**
- MIMIC: `%;g/dL` (multiple units) → eICU: `g/dL`
- **Conversion**: Keep g/dL values, filter out % values
- **Action**: Standardize to g/dL for both datasets

### 7. **Potassium**
- MIMIC: `mEq/L` → eICU: `mmol/L`
- **Conversion**: For potassium (K⁺, valence=1): mEq/L = mmol/L
- **Action**: No mathematical conversion needed (equivalent for monovalent ions)

### 8. **RBC (Red Blood Cell Count)**
- MIMIC: `#/hpf;m/uL` (multiple units) → eICU: `M/mcL`
- **Conversion**: M/mcL (million per microliter) = m/uL
- **Action**: Filter MIMIC to use m/uL values only (remove #/hpf)

### 9. **Respiratory Rate**
- MIMIC: `insp/min` → eICU: `breaths/min`
- **Conversion**: Inspirations per minute = breaths per minute
- **Action**: No mathematical conversion needed (same measurement)

### 10. **Sodium**
- MIMIC: `mEq/L` → eICU: `mmol/L`
- **Conversion**: For sodium (Na⁺, valence=1): mEq/L = mmol/L
- **Action**: No mathematical conversion needed (equivalent for monovalent ions)

### 11. **Temperature Fahrenheit**
- MIMIC: `°F` → eICU: `F`
- **Conversion**: Same unit, just notation difference
- **Action**: No conversion needed, standardize notation

### 12. **Urea Nitrogen (BUN)**
- MIMIC: `mg/dL` → eICU: `G/24HR;mg/dL`
- **Conversion**: Keep mg/dL values, filter out G/24HR (different measurement)
- **Action**: Standardize to mg/dL for both datasets

### 13. **WBC (White Blood Cell Count)**
- MIMIC: `#/hpf;K/uL` (multiple units) → eICU: `K/mcL`
- **Conversion**: K/uL (thousand per microliter) = K/mcL
- **Action**: Filter MIMIC to use K/uL values only (remove #/hpf)

### 14. **Lactate Dehydrogenase (LD)**
- MIMIC: `IU/L` → eICU: `Units/L`
- **Conversion**: IU/L and Units/L are equivalent
- **Action**: No mathematical conversion needed, just normalize naming

---

## Conversion Strategy

### Category 1: Naming Only (8 features)
No mathematical conversion needed, just standardize unit names:
- ALT, Alkaline Phosphatase, AST, LDH: `IU/L` = `Units/L`
- Hemoglobin: `g/dl` → `g/dL`
- Potassium, Sodium: `mEq/L` = `mmol/L` (for monovalent ions)
- Respiratory Rate: `insp/min` = `breaths/min`
- Temperature Fahrenheit: `°F` = `F`

### Category 2: Mathematical Conversion (1 feature)
Requires actual value transformation:
- **C-Reactive Protein**: mg/L → mg/dL (multiply by 0.1)

### Category 3: Multi-unit Filtering (3 features)
Multiple units recorded, need to filter to common unit:
- **MCHC**: Keep `g/dL`, remove `%`
- **RBC**: Keep `m/uL` (= `M/mcL`), remove `#/hpf`
- **WBC**: Keep `K/uL` (= `K/mcL`), remove `#/hpf`
- **Urea Nitrogen**: Keep `mg/dL`, remove `G/24HR`

---

## Implementation Notes

1. **Standardize all unit names** to a consistent format (e.g., always use `IU/L` instead of `Units/L`)
2. **Apply mathematical conversion** only to C-Reactive Protein
3. **Filter multi-unit features** to keep only the common unit between datasets
4. **Verify equivalencies** for monovalent ion measurements (K⁺, Na⁺)

---

## Files Generated

- `mimic_feature_units.csv` - MIMIC-IV units metadata (40 features)
- `eicu_feature_units.csv` - eICU-CRD units metadata (40 features)  
- `feature_units_comparison.csv` - Side-by-side comparison of all 40 features


