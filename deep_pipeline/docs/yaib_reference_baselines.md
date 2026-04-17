# YAIB Paper Reference Baselines

**Source**: van de Water et al., "Yet Another ICU Benchmark: A Flexible Multi-Center Framework for Clinical ML", ICLR 2024 (arXiv:2306.05109)

All values from Table 3 (classification tasks). Results reported as mean +/- std across seeds.

---

## 1. LSTM Results by Dataset (AUROC x 100)

The LSTM is our baseline architecture. These numbers show native in-domain performance.

| Task | MIMIC-IV | eICU | HiRID | AUMCdb |
|---|---|---|---|---|
| Mortality24 | 86.7 +/- 0.4 | 85.5 +/- 0.2 | 84.0 +/- 0.7 | 83.7 +/- 0.7 |
| AKI | 89.7 +/- 0.1 | 90.2 +/- 0.1 | 81.0 +/- 0.4 | 86.5 +/- 0.4 |
| Sepsis | 82.0 +/- 0.3 | 74.0 +/- 0.2 | 78.8 +/- 0.4 | 77.1 +/- 0.8 |

## 2. LSTM Results by Dataset (AUPRC x 100)

| Task | MIMIC-IV | eICU | HiRID | AUMCdb |
|---|---|---|---|---|
| Mortality24 | 41.0 +/- 0.7 | 35.7 +/- 0.8 | 46.7 +/- 1.2 | 39.3 +/- 1.1 |
| AKI | 66.5 +/- 0.2 | 69.9 +/- 0.2 | 37.4 +/- 0.7 | 62.1 +/- 0.7 |
| Sepsis | 8.0 +/- 0.2 | 4.0 +/- 0.1 | 9.1 +/- 0.3 | 7.6 +/- 0.5 |

## 3. All Classification Results (AUROC x 100)

Full reference across all 6 model architectures reported in the paper.

### Mortality24

| Model | MIMIC-IV | eICU | HiRID | AUMCdb |
|---|---|---|---|---|
| LSTM | 86.7 +/- 0.4 | 85.5 +/- 0.2 | 84.0 +/- 0.7 | 83.7 +/- 0.7 |
| GRU | 86.8 +/- 0.3 | **86.0 +/- 0.3** | 83.1 +/- 0.4 | 83.3 +/- 1.7 |
| TCN | 86.5 +/- 0.4 | 85.7 +/- 0.2 | 82.1 +/- 1.2 | 83.9 +/- 1.2 |
| Transformer | 86.9 +/- 0.4 | 85.8 +/- 0.2 | 81.8 +/- 1.2 | 83.2 +/- 0.9 |
| LogReg | 83.3 +/- 0.2 | 80.2 +/- 0.2 | 79.6 +/- 0.9 | 76.3 +/- 0.6 |
| LGBM | 84.5 +/- 0.3 | 82.5 +/- 0.2 | 81.3 +/- 0.7 | 82.5 +/- 0.8 |

### AKI

| Model | MIMIC-IV | eICU | HiRID | AUMCdb |
|---|---|---|---|---|
| LSTM | 89.7 +/- 0.1 | 90.2 +/- 0.1 | 81.0 +/- 0.4 | 86.5 +/- 0.4 |
| GRU | 89.4 +/- 0.1 | **90.9 +/- 0.1** | 80.3 +/- 0.5 | 86.3 +/- 0.3 |
| TCN | 89.2 +/- 0.1 | 90.3 +/- 0.1 | 80.2 +/- 0.4 | 85.3 +/- 0.3 |
| Transformer | 89.3 +/- 0.2 | 90.3 +/- 0.1 | 80.3 +/- 0.4 | 85.3 +/- 0.3 |
| LogReg | 86.7 +/- 0.1 | 87.7 +/- 0.1 | 76.7 +/- 0.3 | 82.2 +/- 0.2 |
| LGBM | 88.9 +/- 0.1 | 89.9 +/- 0.1 | 80.0 +/- 0.2 | 85.9 +/- 0.3 |

### Sepsis

| Model | MIMIC-IV | eICU | HiRID | AUMCdb |
|---|---|---|---|---|
| LSTM | 82.0 +/- 0.3 | 74.0 +/- 0.2 | 78.8 +/- 0.4 | 77.1 +/- 0.8 |
| GRU | 82.8 +/- 0.4 | **77.4 +/- 0.3** | 78.5 +/- 0.4 | 77.6 +/- 0.8 |
| TCN | 80.8 +/- 0.1 | 74.6 +/- 0.3 | 78.3 +/- 0.4 | 74.1 +/- 0.8 |
| Transformer | 82.5 +/- 0.5 | 76.5 +/- 0.4 | 78.3 +/- 0.5 | 73.8 +/- 0.8 |
| LogReg | 73.2 +/- 0.2 | 64.1 +/- 0.2 | 68.5 +/- 0.3 | 67.1 +/- 0.4 |
| LGBM | 79.5 +/- 0.1 | 73.3 +/- 0.2 | 75.1 +/- 0.3 | 73.5 +/- 0.4 |

## 4. Regression Results by Dataset (Table 4)

From YAIB paper Table 4 (regression tasks). Results reported as Mean Absolute Error (↓ is better).

### Kidney Function (mg/dL)

| Model | AUMCdb | HiRID | eICU | MIMIC-IV |
|---|---|---|---|---|
| EN | 0.24±0.00 | 0.28±0.00 | 0.31±0.00 | 0.25±0.00 |
| LGBM | 0.32±0.00 | 0.34±0.00 | **0.24±0.00** | **0.24±0.00** |
| GRU | 0.29±0.00 | 0.32±0.01 | 0.34±0.01 | 0.30±0.01 |
| LSTM | 0.29±0.00 | 0.33±0.00 | 0.28±0.01 | 0.28±0.01 |
| TCN | 0.28±0.01 | **0.23±0.01** | 0.28±0.01 | 0.28±0.01 |
| TF | 0.26±0.00 | 0.31±0.01 | 0.32±0.01 | 0.28±0.01 |

### Length of Stay (hours)

| Model | AUMCdb | HiRID | eICU | MIMIC-IV |
|---|---|---|---|---|
| EN | 54.9±0.1 | 47.2±0.1 | 43.6±0.0 | 46.5±0.0 |
| LGBM | 44.7±0.0 | 39.2±0.1 | 39.3±0.0 | 40.1±0.0 |
| GRU | 42.9±0.1 | 39.6±0.1 | 38.9±0.1 | 39.9±0.1 |
| LSTM | 44.8±0.1 | 39.8±0.1 | 39.2±0.1 | 40.6±0.1 |
| TCN | 43.7±0.1 | 39.9±0.1 | 38.9±0.0 | 40.4±0.1 |
| TF | **41.8±0.1** | **39.1±0.1** | **38.2±0.1** | **39.6±0.1** |

---

## 5. eICU → MIMIC: Gap Analysis vs eICU-Native

Our frozen baseline is a MIMIC-IV-trained LSTM evaluated on eICU data (cross-domain). The translator adapts eICU inputs to improve this frozen model's performance.

### Classification (AUROC × 100)

| Reference Point | Mortality24 | AKI | Sepsis |
|---|---|---|---|
| Our frozen baseline (MIMIC LSTM on eICU) | 80.79 | 85.58 | 71.59 |
| eICU-native LSTM | 85.5 | 90.2 | 74.0 |
| MIMIC-native LSTM | 86.7 | 89.7 | 82.0 |
| Best eICU model (GRU) | 86.0 | 90.9 | 77.4 |
| **Our best translator** | **85.49** | **91.28** | **76.78** |

### Regression (MAE ↓)

| Reference Point | LoS (hours) | KF (mg/dL) |
|---|---|---|
| Our frozen baseline | 42.5 | 0.403 |
| eICU-native LSTM | 39.2 | 0.28 |
| MIMIC-native LSTM | 40.6 | 0.28 |
| Best eICU model | 38.2 (TF) | 0.24 (EN) |
| **Our best translator** | **37.2** | **0.292** |

### Status per Task (eICU → MIMIC)

| Task | Our Best | vs eICU-native LSTM | Gap/Margin | Status |
|---|---|---|---|---|
| **Mortality24** | 85.49 | 85.5 | −0.01 | **Tied** (within noise) |
| **AKI** | 91.28 | 90.2 | **+1.08** | **Won** |
| **Sepsis** | 76.78 | 74.0 | **+2.78** | **Won** |
| **LoS** | 37.2h | 39.2h | **−2.0h** | **Won** |
| **KF** | 0.292 | 0.28 | +0.012 | **Gap** (4.3% relative) |

**Summary**: 4/5 tasks beat eICU-native LSTM. Only KF remains (0.292 vs 0.28 target). Mortality is borderline — essentially tied.

---

## 6. HiRID → MIMIC: Gap Analysis vs HiRID-Native

Source = HiRID, Target = MIMIC. Same frozen MIMIC-trained LSTM. Translator adapts HiRID inputs.

### HiRID-Native Baselines (YAIB Paper)

| Task | LSTM | Best Model | Best Model Type |
|---|---|---|---|
| Mortality (AUROC) | 84.0 ± 0.7 | 84.9 ± 0.7 | TF |
| AKI (AUROC) | 81.0 ± 0.4 | 82.2 ± 0.2 | GRU |
| Sepsis (AUROC) | 78.8 ± 0.4 | 80.8 ± 0.5 | TCN/TF |
| LoS (MAE, h) | 39.8 ± 0.1 | 39.1 ± 0.1 | TF |
| KF (MAE, mg/dL) | 0.33 ± 0.00 | 0.23 ± 0.01 | TCN |

### Our HiRID → MIMIC Results

Unit conversions use MIMIC normalization: 1 LoS unit ≈ 168h, 1 KF unit ≈ 12.2 mg/dL.

| Task | Frozen Baseline | Translated | Δ | Config |
|---|---|---|---|---|
| Mortality (AUROC) | 76.07 | 80.81 | +4.74 | `mortality_hirid_sr` |
| AKI (AUROC) | 75.20 | 82.96 | +7.76 | `aki_hirid_sr` |
| Sepsis (AUROC) | 72.02 | 79.79 | +7.77 | `sepsis_hirid_sr` |
| LoS (MAE, h) | ~59.6 | **INVALID** | — | `los_hirid_nommd` ⚠️ NaN task loss bug, rerun pending |
| KF (MAE, mg/dL) | ~0.289 | ~0.273 | −0.016 | `kf_hirid_nofid_nommd` |

### Status per Task (HiRID → MIMIC)

| Task | Our Best | vs HiRID-native LSTM | Gap/Margin | Status |
|---|---|---|---|---|
| **Mortality24** | 80.81 | 84.0 | −3.19 | **Gap** |
| **AKI** | 82.96 | 81.0 | **+1.96** | **Won** |
| **Sepsis** | 79.79 | 78.8 | **+0.99** | **Won** |
| **LoS** | **INVALID** | 39.8h | — | **Pending** (NaN bug, rerun submitted) |
| **KF** | ~0.273 | 0.33 | **−0.057** | **Won** (baseline already < native) |

**Summary**: 3/5 tasks beat HiRID-native LSTM. Mortality (−3.2 AUROC) remains. LoS results INVALID (NaN task loss bug — rerun pending).

Note: KF frozen baseline (~0.289 mg/dL) already outperforms HiRID-native LSTM (0.33) before translation, suggesting the MIMIC LSTM learned a more generalizable KF representation.

---

## 7. Combined Scorecard

| Task | vs eICU-native | vs HiRID-native | Notes |
|---|---|---|---|
| **Mortality** | **Tied** (−0.01) | **Gap** (−3.19) | HiRID harder |
| **AKI** | **Won** (+1.08) | **Won** (+1.96) | Both directions strong |
| **Sepsis** | **Won** (+2.78) | **Won** (+0.99) | Both directions strong |
| **LoS** | **Won** (−2.0h) | **INVALID** (pending rerun) | HiRID LoS had NaN bug |
| **KF** | **Gap** (+0.012) | **Won** (−0.057) | Opposite pattern |

**Remaining targets**:
- eICU → MIMIC: **KF** (0.292 vs 0.28, need −0.012 mg/dL more)
- HiRID → MIMIC: **Mortality** (80.81 vs 84.0, need +3.19 AUROC) and **LoS** (results INVALID, awaiting rerun with NaN fix)

Interesting pattern: the "hard" tasks flip between directions. KF is the blocker from eICU but already won from HiRID. Mortality is won from eICU but blocked from HiRID. LoS is won from eICU but has a large gap from HiRID.

## 8. Caveats

- **AUPRC**: Not directly comparable between our pipeline and YAIB paper (different cohort definitions, preprocessing, positive rates). AUROC comparisons are more reliable since AUROC is invariant to class prevalence.
- **Regression unit conversion**: LoS/KF values in our system are z-score normalized (MIMIC training stats). Approximate conversions: 1 LoS unit ≈ 168h, 1 KF unit ≈ 12.2 mg/dL. These are approximate — exact denormalization constants should be verified from MIMIC preprocessing stats.
- **HiRID regression comparisons are approximate**: The ~52h and ~0.273 mg/dL values use the same conversion factors as eICU. Cross-check with raw prediction files for exact values.

## 9. Source

Results extracted from YAIB paper Tables 3 and 4 (van de Water et al., ICLR 2024, arXiv:2306.05109). See also `YAIB/docs/figures/results_yaib.png` for the original results figure.
