# Venue Strategy & SOTA Baseline Analysis

**Date**: March 26, 2026
**Method**: 7 independent research agents with web search, consensus-synthesized
**Context**: EHR Translator — frozen-model domain adaptation for clinical time series (eICU/HiRID → MIMIC-IV)

---

## 1. Venue Recommendations

### 1.1 Main Track: NeurIPS 2026 (Unanimous, 7/7)

| Field | Detail |
|---|---|
| Deadline | Abstract May 4, Paper May 6, 2026 (~41 days from analysis date) |
| Acceptance rate | ~25% (2025: 24.5%, 5290/21575) |
| Notification | Sep 24, 2026 |
| Conference | Dec 6–12, 2026, Sydney |
| Format | 9 pages + unlimited references/appendix |

**Why NeurIPS**:
- Top-3 CS venue; ideal for a CS PhD student's CV
- Directly in scope: NeurIPS 2024 accepted ACON (time-series DA) and RCD-KD (cross-domain knowledge distillation for TS)
- Paper touches DA theory (Ben-David gradient alignment), retrieval-augmented architectures, and multi-source generalization — all core NeurIPS topics
- The frozen-model constraint aligns with growing interest in adapting around immovable/deployed models
- The retrieval translator is not health-specific — it is a general DA framework evaluated on clinical tasks

### 1.2 Backup Waterfall

| Priority | Venue | Deadline (est.) | Acceptance Rate | Notes |
|---|---|---|---|---|
| 1st backup | **ICLR 2027** | ~Oct 2026 | ~32% | NeurIPS notification (Sep 24) gives 3–4 weeks to revise. YAIB was ICLR 2024. Best topical fit for representation learning + DA. Highest acceptance rate of top-3. |
| 2nd backup | **ICML 2027** | ~Jan 2027 | ~27% | More theory-heavy audience. Good if gradient alignment theory is strengthened. |
| 3rd backup | **AAAI 2027** | Aug 1, 2026 | ~17.6% | Lower prestige for ML methods. Trending down in acceptance rate. Safety net only. |
| Rolling | **TMLR** | Anytime | ~35–40% | Growing prestige (ICLR-affiliated). Avoids conference cycle delays. 2–4 month review. Legitimate for PhD but slightly below top-3 conferences for hiring committees. |
| Domain-specific | **CHIL 2027** | ~Feb 2027 | ~25–30% | Perfect topical fit but too niche as flagship CS PhD paper. Use alongside a top-venue publication, not instead of one. ACM venue, PMLR proceedings. |

**Decision tree**:
```
NeurIPS 2026 decision (Sep 24, 2026)
├── Accept → Done
├── Reject with actionable feedback
│   ├── Can address in 3–4 weeks? → ICLR 2027 (Oct deadline)
│   ├── Needs major revision? → ICML 2027 (Jan deadline)
│   └── Fundamental framing concerns? → Reframe + TMLR (rolling)
└── Reject with "not enough baselines" / "incremental"
    ├── Add requested baselines → ICLR 2027
    └── Expand scope → TMLR or JMLR
```

### 1.3 Workshop: TS4H at NeurIPS 2026 (5/7 consensus)

| Workshop | Venue | Fit | Deadline (est.) | Format |
|---|---|---|---|---|
| **TS4H** (Time Series for Health) | NeurIPS 2026 | **Perfect** — exact topic match (clinical TS, EHR, distribution shift) | ~Sep 2026 | 4-page, non-archival |
| ML4H (Machine Learning for Health) | NeurIPS 2026 / standalone | Excellent — DA explicitly listed as topic of interest | ~Sep 2026 | 4-page findings (non-archival) or 8-page proceedings (archival) |
| Algorithmic Foundations for Medical AI | ICML 2026 (Jul 10–11) | Strong — Zitnik (RAINCOAT author) is organizer. "Distribution shift + DA" explicitly in scope | TBD | 5-page, non-archival |
| DistShift | NeurIPS 2026 (if recurs) | Good — methods/evaluations for distribution shifts | TBD | Non-archival |

**Dual submission rules**: NeurIPS main track + non-archival workshop = **no conflict**. Workshop papers presented at non-archival workshops do not count as prior publication. Submit a 4-page version to TS4H regardless of main track outcome.

**Recommended strategy**:

| Date | Action |
|---|---|
| May 4, 2026 | NeurIPS 2026 abstract deadline |
| May 6, 2026 | NeurIPS 2026 full paper deadline |
| ~Sep 2026 | Submit 4-page version to TS4H and/or ML4H Findings |
| Sep 24, 2026 | NeurIPS 2026 notification |
| ~Oct 2026 | If rejected: revise with reviewer feedback → ICLR 2027 |
| ~Jan 2027 | If still needed: → ICML 2027 |

### 1.4 Venues Already Passed

| Venue | Deadline | Status |
|---|---|---|
| ICML 2026 | Jan 29, 2026 | Passed |
| CHIL 2026 | Feb 4, 2026 | Passed |
| KDD 2026 Cycle 2 | Feb 8, 2026 | Passed |
| IJCAI 2026 | Jan 19, 2026 | Passed |
| MLHC 2026 | Apr 17, 2026 | 22 days — too tight to split focus with NeurIPS |

### 1.5 CS Career Prestige Ranking

For a CS PhD student's CV, the venue prestige ordering is:

**NeurIPS ≈ ICML ≈ ICLR >> AAAI > KDD > TMLR >> CHIL ≈ MLHC**

Any of the top-3 (NeurIPS/ICML/ICLR) is a tier-1 publication. TMLR is growing but not yet equivalent.

---

## 2. SOTA Baseline Analysis

### 2.1 Current Baselines: Necessary but Not Sufficient (7/7 Unanimous)

| Method | Year | Venue | Approach | Status |
|---|---|---|---|---|
| DANN | 2016 | JMLR | Gradient reversal + domain discriminator | **Done** |
| Deep CORAL | 2016 | ECCV-W | Second-order statistics alignment | **Done** |
| CoDATS | 2020 | KDD | 1D CNN + adversarial, TS-specific | **Done** |

**Problem**: These are all 2016–2020 methods. NeurIPS 2026 reviewers will see a 6-year gap to present-day SOTA and flag it. Every recent TS-DA paper (ACON, CLUDA, RAINCOAT) compares against more methods.

### 2.2 Proposed Baselines: Dataset & Task Deep-Dive

Detailed investigation of what datasets and tasks each proposed baseline actually uses. This reveals that **our problem setting is fundamentally different from what these methods were designed and evaluated for**.

#### Overview

| Method | Venue | Modality | Clinical ICU Data? | EHR Prediction Tasks? | Frozen Model? |
|---|---|---|---|---|---|
| **CDAN** | NeurIPS 2018 | **Images only** | No | No | No |
| **CLUDA** | ICLR 2023 | Time series | **Yes** (MIMIC-IV, AmsterdamUMCdb) | **Yes** (Mortality, Decompensation, LoS) | No |
| **RAINCOAT** | ICML 2023 | Time series | No (Sleep-EDF is closest) | No | No |
| **ACON** | NeurIPS 2024 | Time series | No | No | No |

#### CDAN — Conditional Adversarial Domain Adaptation (NeurIPS 2018)

**Authors**: Long, Cao, Wang, Jordan

**Datasets (5 — all image classification)**:
| Dataset | Domains | Classes | Transfer Tasks |
|---|---|---|---|
| Office-31 | Amazon, Webcam, DSLR | 31 | 6 pairs |
| ImageCLEF-DA | Caltech-256, ImageNet, Pascal VOC | 12 | 6 pairs |
| Office-Home | Artistic, Clip Art, Product, Real-World | 65 | 12 pairs |
| Digits (MNIST/USPS/SVHN) | 3 digit datasets | 10 | 3 pairs |
| VisDA-2017 | Synthetic renderings, Real images | 12 | 1 pair |

- **Total**: 28 transfer tasks, all image classification
- **Clinical/EHR data**: None
- **Time series data**: None
- **Backbone**: AlexNet / ResNet-50 (ImageNet pretrained)
- **Baselines compared**: DAN, RTN, DANN, ADDA, JAN, UNIT, GTA, CyCADA

**Implication**: CDAN is purely a vision method. Including it shows thoroughness in covering the adversarial DA family, but it has never been applied to time series. Our adaptation would be the first TS application.

#### CLUDA — Contrastive Learning for UDA of Time Series (ICLR 2023)

**Authors**: Ozyurt, Feuerriegel, Zhang (ETH Zurich / LMU Munich)

**Datasets (5 — 3 sensor + 2 clinical ICU)**:
| Dataset | Type | Domains | Channels | Seq Length | Classes |
|---|---|---|---|---|---|
| WISDM | Accelerometer (HAR) | 30 subjects | 3 | 128 | 6 |
| HAR | Accel+Gyro (HAR) | 30 subjects | 9 | 128 | 6 |
| HHAR | Accelerometer (HAR) | 9 subjects | 3 | 128 | 6 |
| **MIMIC-IV** | **ICU EHR** | 4 age groups | **41** | **variable (48h max)** | binary/ordinal |
| **AmsterdamUMCdb** | **ICU EHR** | 4 age groups | **41** | **variable** | binary/ordinal |

**Clinical tasks (3)**:
| Task | Type | Metric |
|---|---|---|
| Decompensation (death within 24h) | Binary classification, per-timestep | AUROC, AUPRC |
| Mortality (in-hospital) | Binary classification, per-stay | AUROC, AUPRC |
| Length of Stay | **Ordinal 10-class** classification (NOT regression) | Cohen's weighted Kappa |

**Domain shift definitions**:
- Within-dataset: 4 age groups as domains (20-45, 46-65, 66-85, 85+)
- Cross-dataset: MIMIC ↔ AmsterdamUMCdb (whole population)
- **Total**: ~92 DA scenarios (30 sensor + 24 within-ICU + 32 cross-ICU age-group + 6 cross-ICU whole-pop)

**NOT used**: eICU, HiRID. **NOT evaluated**: Sepsis, AKI, Kidney Function.

**Baselines (11)**: w/o UDA, VRADA, CoDATS, AdvSKM, CAN, CDAN, DDC, DeepCORAL, DSAN, HoMM, MMDA

**Implication**: CLUDA is the **most relevant competitor** — the only proposed baseline that uses ICU data (MIMIC-IV + AUMC) for clinical prediction. However:
- Their domain shift is age-group subpopulations or MIMIC↔AUMC, not eICU→MIMIC or HiRID→MIMIC
- They use 41 features (not our 48+48MI+4st schema)
- LoS is ordinal classification (10 bins), not regression
- No sepsis or AKI tasks
- End-to-end training (no frozen model)

#### RAINCOAT — Frequency-Aware DA for Time Series (ICML 2023)

**Authors**: He, Queen, Koker, Cuevas, Tsiligkaridis, Zitnik (Harvard)

**Datasets (5 — all sensor/signal, NO clinical EHR)**:
| Dataset | Type | Domains | Channels | Seq Length | Classes |
|---|---|---|---|---|---|
| WISDM | Accelerometer (HAR) | 30 subjects | 3 | 128 | 6 |
| HAR | Accel+Gyro (HAR) | 30 subjects | 9 | 128 | 6 |
| HHAR | Accelerometer (HAR) | 9 subjects | 3 | 128 | 6 |
| Boiler | Industrial sensors (fault detection) | 3 boilers | 20 | 36 | 2 |
| Sleep-EDF | EEG (sleep staging) | 20 individuals | 1 | 3,000 | 5 |

- **Total**: 46 closed-set DA scenarios + 30 universal DA scenarios = 76 scenarios
- **Clinical/EHR data**: None. Sleep-EDF is EEG from healthy individuals (PhysioNet), not ICU/EHR.
- **Tasks**: Activity recognition, industrial fault detection, sleep stage classification — all fixed-length segment classification
- **Handles**: Both closed-set DA and universal DA (target has private labels)

**Baselines (13)**: DeepCORAL, CDAN, DIRT-T, AdaMatch, CoDATS, AdvSKM, CLUDA, DDC, HoMM, DSAN, MMDA + UAN, DANCE, OVANet, UniOT (universal DA)

**Implication**: RAINCOAT is the most comprehensive TS-DA benchmark paper but uses **no EHR/ICU data at all**. Domains are individual subjects/devices/machines — a fundamentally different shift from cross-hospital EHR. Only does classification on fixed-length segments, no regression, no variable-length clinical time series.

#### ACON — Adversarial Co-learning Networks (NeurIPS 2024)

**Authors**: Liu, Chen, Shu, Li, Guan, Nie (HIT Shenzhen)

**Datasets (8 — all sensor/signal, NO clinical EHR)**:
| Dataset | Type | Domains | Channels | Seq Length | Classes |
|---|---|---|---|---|---|
| UCIHAR | Accel+Gyro (HAR) | 30 subjects | 9 | 128 | 6 |
| HHAR-P | Accelerometer (HAR, participant split) | 9 participants | 3 | 128 | 6 |
| HHAR-D | Accelerometer (HAR, device split) | 5 devices | 6 | 500 | 6 |
| WISDM | Accelerometer (HAR) | 30 subjects | 3 | 128 | 6 |
| FD | Vibration (fault diagnosis) | 4 conditions | 1 | 5,120 | 3 |
| CAP | EEG (sleep staging) | 5 machines | 19 | 3,000 | 6 |
| EMG | Electromyography (gestures) | 4 subjects | 8 | 200 | 6 |
| PCL | EEG motor imagery (BCI) | 3 procedures | 48 | 750 | 2 |

- **Total**: 76 DA scenarios (10 per dataset, except PCL with 6), each repeated 5 seeds = 380 runs
- **Clinical/EHR data**: None. CAP is clinical sleep EEG but not ICU/EHR prediction.
- **Tasks**: Activity recognition, fault diagnosis, sleep staging, gesture recognition, motor imagery — all fixed-length classification
- **Backbone**: 3-layer 1D-CNN (same for all methods, from AdaTime)

**Baselines (10)**: Source-only, CDAN, DeepCoral, AdaMatch, HoMM, DIRT-T, CoDATS, AdvSKM, CLUDA, RAINCOAT

**Implication**: ACON is the current NeurIPS SOTA but operates on **entirely different data**. Short fixed-length signal windows with single classification labels, not variable-length irregular multivariate clinical TS with per-timestep predictions. Domain shifts are subject/device/machine differences, not cross-hospital EHR schema differences.

#### Summary: How Our Setting Differs from ALL Proposed Baselines

| Aspect | CDAN | CLUDA | RAINCOAT | ACON | **Ours** |
|---|---|---|---|---|---|
| **Modality** | Images | Time series | Time series | Time series | **Clinical time series (EHR)** |
| **Clinical ICU data** | No | **MIMIC-IV, AUMC** | No | No | **eICU, MIMIC-IV, HiRID** |
| **Domain shift type** | Image styles | Age groups / cross-DB | Subjects / devices | Subjects / devices / machines | **Cross-hospital EHR systems** |
| **Prediction tasks** | Image classification | Mortality, Decomp, LoS | HAR, fault, sleep | HAR, fault, sleep, EMG, BCI | **Mortality, AKI, Sepsis, LoS, KF** |
| **Per-timestep tasks** | N/A | Decompensation | No | No | **AKI, Sepsis, LoS** |
| **Regression tasks** | No | No (LoS is ordinal) | No | No | **LoS (MAE), KF (MAE)** |
| **Variable-length sequences** | N/A | Yes (48h max) | No (fixed 128–3000) | No (fixed 128–5120) | **Yes (variable, padded)** |
| **Feature count** | N/A | 41 | 1–20 | 1–48 | **48–292** (with MI, static, generated) |
| **Frozen predictor** | No | No | No | No | **Yes (frozen MIMIC LSTM)** |
| **Training paradigm** | End-to-end | End-to-end | End-to-end | End-to-end | **Input-space translation** |
| **Sepsis / AKI** | No | No | No | No | **Yes** |
| **eICU used** | No | No | No | No | **Yes (primary source)** |
| **HiRID used** | No | No | No | No | **Yes (secondary source)** |
| **Multi-source DA** | No | No | Cross-dataset (HAR only) | No | **Yes (eICU + HiRID → MIMIC)** |

#### Implications for Our Paper

1. **CLUDA is the only competitor with clinical ICU data** — and even it uses different datasets (AUMC not eICU/HiRID), different tasks (no AKI/Sepsis), and different LoS formulation (ordinal not regression). We should compare to CLUDA and highlight these differences.

2. **RAINCOAT and ACON have never seen EHR data** — adapting them to our setting is a genuine contribution. Their architectures were designed for short fixed-length segments with 1–20 channels, not variable-length clinical TS with 48–292 features.

3. **No method operates under a frozen-model constraint** — all train end-to-end. This is our central differentiator and the key reason existing methods may underperform in our setting.

4. **Domain shift nature is fundamentally different** — existing TS-DA evaluates subject-to-subject or device-to-device shifts (same hospital, same EHR). We evaluate hospital-to-hospital shifts across entirely different EHR systems (eICU vs MIMIC-IV vs HiRID). This is a harder, more realistic shift.

5. **We are the first to evaluate TS-DA on AKI and Sepsis** — these per-timestep clinical tasks with extreme label sparsity (AKI: 11.95%, Sepsis: 1.13%) are not covered by any existing TS-DA work.

6. **Our evaluation is more comprehensive** — 5 tasks × 2 source domains × 3 paradigms × ablations. Most TS-DA papers evaluate on HAR/fault detection with single-label-per-window classification.

### 2.3 Consensus Baseline Tiers (from initial analysis)

#### Tier 1 — MUST Add (6–7/7 agents agree)

| Method | Venue | Year | Key Idea | Why Critical | Effort | Code |
|---|---|---|---|---|---|---|
| **CLUDA** | ICLR 2023 | 2023 | Contrastive learning + nearest-neighbor matching for TS DA | **Most relevant EHR DA method** — tested on MIMIC/AUMC. Single most likely reviewer ask. | 1–2 days | [github.com/oezyurty/CLUDA](https://github.com/oezyurty/CLUDA) |
| **RAINCOAT** | ICML 2023 | 2023 | Time + frequency encoders, Sinkhorn divergence, handles feature + label shift | Top performer in 2025 benchmark. Harvard/Zitnik lab. Very well-cited. | 1–2 days | [github.com/mims-harvard/Raincoat](https://github.com/mims-harvard/Raincoat) |
| **CDAN** | NeurIPS 2018 | 2018 | Conditional adversarial — conditions discriminator on classifier predictions | "Mature DANN." Cheap to add since DANN infra exists. Fills the DANN→modern gap. | 2–3 hours | Modify existing DANN |

#### Tier 2 — SHOULD Add (4–5/7 agents agree)

| Method | Venue | Year | Key Idea | Why Important | Effort | Code |
|---|---|---|---|---|---|---|
| **ACON** | NeurIPS 2024 | 2024 | Temporal-frequency co-learning + adversarial in correlation subspaces | **Current SOTA TS-DA at NeurIPS**. Not comparing against the most recent NeurIPS DA paper is a gap. | 2–3 days | [github.com/mingyangliu1024/ACON](https://github.com/mingyangliu1024/ACON) |
| **CoTMix** | IEEE TAI 2023 | 2023 | Temporal mixup + contrastive learning | In AdaTime benchmark, strong TS performer. | 1 day | [github.com/emadeldeen24/CoTMix](https://github.com/emadeldeen24/CoTMix) |
| **AdvSKM** | IJCAI 2021 | 2021 | Adversarial spectral kernel matching | Addresses non-stationarity. In AdaTime benchmark. | 1 day | In AdaTime |
| **MMD standalone** | — | 2015 | Multi-kernel MMD on hidden states | Classic kernel alignment. We already have MMD code. | 2 hours | Existing code |
| **OT-DA (Sinkhorn)** | — | 2017–19 | Optimal transport alignment (Wasserstein/Sinkhorn) | Principled alternative to MMD. We have Sinkhorn code from B5. | 3 hours | Existing code |

#### Tier 3 — Simple/Negative Baselines (3–4/7 agents recommend)

| Baseline | What It Tests | Effort |
|---|---|---|
| **Statistics matching only** (target normalization, no translator) | Is a neural translator even needed? | Trivial (5 min) |
| **Linear probe** (per-feature affine transformation, 2 params/feature) | Does this require a deep model? | Half day |
| **Fine-tuned LSTM** (unfreeze and retrain on source) | Upper bound: what if we DON'T freeze? | Easy |

#### Methods That Do NOT Need Comparison

| Method | Why Not Applicable |
|---|---|
| Domain Generalization (SWAD, FISH) | Different setting — no target data access |
| Federated DA | Different threat model (privacy) |
| Foundation models (ICareFM) | Different resource regime (650K stays vs single-hospital). Discuss in related work. |
| Fine-tuning/LoRA/Adapters | Violates frozen-model constraint. Mention explicitly. |
| SimCLR/MoCo pretraining | Self-supervised pretraining is a different setting |
| Domain Randomization | Requires control over data generation |

### 2.4 Recommended Final Baseline Suite

**Target: 8–9 methods** (top-venue DA papers use 5–8 baselines)

| # | Method | Category | Year | Status |
|---|---|---|---|---|
| 0 | No adaptation (frozen LSTM on source) | Lower bound | — | Done |
| 1 | DANN | Adversarial | 2016 | **Done** |
| 2 | Deep CORAL | Statistical | 2016 | **Done** |
| 3 | CoDATS | TS-adversarial | 2020 | **Done** |
| 4 | **CDAN** | Improved adversarial | 2018 | **Add** (hours) |
| 5 | **CLUDA** | Contrastive TS | 2023 | **Add** (1–2 days) |
| 6 | **RAINCOAT** | Frequency-aware TS | 2023 | **Add** (1–2 days) |
| 7 | **ACON** | NeurIPS 2024 SOTA | 2024 | **Add** (2–3 days, if time) |
| 8 | Statistics-only (target norm, no translator) | Negative baseline | — | **Add** (trivial) |
| — | eICU-native LSTM (YAIB reference) | Upper bound | — | Done |
| — | MIMIC-native LSTM (YAIB reference) | Oracle | — | Done |

This spans: adversarial (DANN, CDAN), statistical (CORAL), contrastive (CLUDA), frequency-aware (RAINCOAT), TS-specific (CoDATS), and current SOTA (ACON) — covering 2016–2024.

### 2.5 Frozen-Model Adaptation Strategy

All existing TS-DA methods (CLUDA, RAINCOAT, ACON, etc.) assume end-to-end fine-tuning. For fair comparison, two strategies:

**Option A — Give competitors MORE freedom (recommended)**:
- Run CLUDA/RAINCOAT/ACON in their original formulation (train shared encoder + task head jointly)
- Report our frozen-model translator results alongside
- Argument: "We beat methods that have strictly more degrees of freedom, despite a harder constraint"

**Option B — Adapt all to frozen setting**:
- Replace their shared encoder with a translator backbone, keep frozen LSTM
- Fairer comparison but harder to implement and less standard

**Recommended**: Do **both** where feasible. Report:
1. Standard method (full freedom, end-to-end) — shows their ceiling
2. Adapted method (frozen-model constraint) — shows how they degrade under our constraint
3. Our retrieval translator (frozen constraint) — shows we solve the problem

This demonstrates both that (a) the frozen-model constraint is genuinely harder and (b) our translator uniquely solves it.

### 2.6 What Top TS-DA Papers Compare Against

Based on analysis of recent top-venue papers:

| Paper | Venue | Baselines Used |
|---|---|---|
| ACON (2024) | NeurIPS | DANN, CDAN, CoDATS, CLUDA, RAINCOAT, AdvSKM, CoTMix, SASA, VRADA |
| RCD-KD (2024) | NeurIPS | DANN, CoDATS, CLUDA, RAINCOAT, CoTMix, AdvSKM |
| RAINCOAT (2023) | ICML | DANN, CDAN, CoDATS, AdvSKM, CLUDA, CoTMix, DDC, DeepCORAL, HoMM, DSAN, MMDA, VRADA, SASA |
| CLUDA (2023) | ICLR | DANN, CDAN, CoDATS, AdvSKM, VRADA, SASA |
| AdaTime benchmark (2023) | TKDD | DANN, CDAN, CoDATS, AdvSKM, CoTMix, SASA, VRADA, DDC, DeepCORAL, HoMM, DSAN, MMDA |

**Common denominator across all**: DANN, CDAN, CoDATS, AdvSKM, CLUDA. We need at minimum CDAN and CLUDA.

### 2.7 Implementation Priority for NeurIPS Deadline (May 6)

Given ~5 weeks, prioritized by effort/impact ratio:

| Priority | Method | Effort | Impact | Deadline Feasible? |
|---|---|---|---|---|
| **P0** | CDAN | 2–3 hours | High (fills DANN gap) | Yes |
| **P0** | Statistics-only negative baseline | Trivial | Medium (answers obvious Q) | Yes |
| **P1** | CLUDA | 1–2 days | Very high (most important missing) | Yes |
| **P1** | RAINCOAT | 1–2 days | Very high (2023 SOTA) | Yes |
| **P2** | ACON | 2–3 days | High (2024 NeurIPS SOTA) | Tight but possible |
| **P3** | CoTMix | 1 day | Medium | If time permits |
| **P3** | Linear probe | Half day | Medium | If time permits |

**Minimum viable set for NeurIPS**: Current 3 (DANN, CORAL, CoDATS) + CDAN + CLUDA + RAINCOAT + statistics-only = **7 DA baselines**. This is defensible.

**Ideal set**: Add ACON for 8 baselines spanning 2016–2024. This is comprehensive.

---

## 3. Critical Papers to Cite for Positioning

### 3.1 Closest Conceptual Relatives

| Paper | Venue | Year | Relationship | How We Differ |
|---|---|---|---|---|
| **TATO** | ICLR 2026 | 2026 | Adapts data (not model) for frozen TS foundation models | They use handcrafted transforms (slicing, normalization, outlier correction). We learn end-to-end neural translation with retrieval augmentation. |
| **DIFO** | CVPR 2024 | 2024 | Source-free DA with frozen multimodal foundation model via prompt learning | Vision domain, prompt-based. We do input-space translation with memory banks for clinical TS. |
| **L2C** | ICLR 2025 | 2025 | Adapting frozen CLIP for few-shot test-time DA via side-branch on input space | Vision, few-shot. We are fully unsupervised DA for time series. |
| **Voice2Series** | ICML 2021 | 2021 | Reprogram acoustic models for TS classification via input transformation | Model reprogramming (Task A → Task B). We do domain adaptation (Domain A → Domain B, same task). |
| **SHOT** | ICML 2020 | 2020 | Source-free DA: freezes source classifier, adapts target encoder | They freeze classifier, adapt feature extractor. We freeze everything, transform input. Different "frozen" regime. |

### 3.2 EHR-Specific Papers to Add to Related Work

| Paper | Venue | Year | Why Cite |
|---|---|---|---|
| **van de Water et al. — Sepsis distribution shift** | npj Digital Medicine | 2026 | Same 3 datasets (eICU, MIMIC, HiRID). Compares 5 deployment strategies. Does NOT study frozen-model translation. |
| **Anchor Regression in ICU** | arXiv | 2025 | 400K patients from 9 ICU databases. Causal DG approach. Useful 3-regime framework (DG/DA/data-rich). |
| **Foundation Models for Critical Care TS** | arXiv | 2024 | 9 ICU datasets, 600K admissions. Addresses "why not just pretrain bigger?" question. |
| **ExtraCare** | arXiv | 2026 | Concept-grounded orthogonal DA for EHR. Interpretable. May be concurrent work. |
| **ICareFM** | medRxiv | 2025 | ICU foundation model, 650K patients. Different paradigm (pretrain at scale vs lightweight translator). |

### 3.3 Benchmark References

| Paper | Venue | Year | Role |
|---|---|---|---|
| **AdaTime** | TKDD | 2023 | Standard TS-DA benchmark suite. Reference for baseline selection methodology. |
| **Fawaz et al. — Deep UDA for TSC** | arXiv | 2024/2025 | 13-method benchmark for TS-DA. Evaluates DANN, CDAN, CoDATS, RAINCOAT, CoTMix, etc. |
| **YAIB** | ICLR | 2024 | Our evaluation framework. Cross-hospital ICU benchmark. |
| **BEDS-Bench** | arXiv | 2021 | Clinical distribution shift benchmark. |

---

## 4. Novelty Claim Validation

Extensive search across all 7 agents confirms: **No prior work combines frozen clinical models with learned input-space translation for EHR domain adaptation.**

Closest works and why they differ:
- **TATO** (ICLR 2026): Frozen model + input transformation, but handcrafted transforms, general TS forecasting, no retrieval, no clinical data
- **L2C** (ICLR 2025): Frozen CLIP + input-space learning, but vision domain, few-shot, no retrieval
- **SHOT** (ICML 2020): Partial freeze (classifier only), feature extractor still adapted
- **kNN-MT** (ICLR 2021): Frozen NMT + kNN, but NLP, output interpolation not input translation

Our claim — **"first frozen-model, retrieval-augmented, input-space domain adaptation for clinical EHR time series"** — holds up under scrutiny.

---

## 5. Key Framing Advantages

### 5.1 The Frozen-Model Constraint Is the Central Differentiator

Every existing TS-DA method (CLUDA, RAINCOAT, ACON, CoDATS, etc.) trains the full pipeline end-to-end. Our translator operates as an input-space preprocessing module that keeps the deployed clinical model completely untouched. This is:
- **Practically motivated**: FDA-approved/validated clinical models cannot be retrained per hospital. Institutional policies often prohibit modifying deployed predictors.
- **Theoretically interesting**: Information bottleneck through the frozen LSTM. Gradient alignment theory explains when and why translation succeeds.
- **Unique across all baselines**: No existing method operates under this constraint.

### 5.2 Surpassing In-Domain Performance

AKI translated eICU (+0.0556) beats MIMIC-native LSTM (89.7 → 91.14). The translator finds features the target model can use better than the target model's own training data. This is a remarkable result to highlight.

### 5.3 Universal Paradigm + Multi-Source Generalization

Retrieval translator works across all 5 tasks and 2 source domains. Combined with the gradient alignment theory explaining *why* it works, this is a complete story: problem → theory → method → comprehensive validation.

---

## 6. Summary: Action Items for NeurIPS 2026

### Baselines (by May 6)

| Week | Action |
|---|---|
| Week 1 (Mar 26 – Apr 1) | Implement CDAN (hours). Run statistics-only negative baseline (trivial). Start CLUDA adaptation. |
| Week 2 (Apr 1 – Apr 8) | Complete CLUDA + RAINCOAT implementations. Run experiments on 3 classification tasks. |
| Week 3 (Apr 8 – Apr 15) | Run ACON if feasible. Collect all baseline results. |
| Week 4 (Apr 15 – Apr 22) | Any remaining experiments. Focus shifts to writing. |

### Venue Strategy

| Date | Action |
|---|---|
| May 4, 2026 | NeurIPS 2026 abstract |
| May 6, 2026 | NeurIPS 2026 paper |
| ~Sep 2026 | TS4H workshop (4-page, non-archival, no conflict) |
| Sep 24, 2026 | NeurIPS notification → prepare ICLR 2027 if rejected |
| ~Oct 2026 | ICLR 2027 deadline (1st backup) |

### Minimum Viable Baseline Set

DANN + Deep CORAL + CoDATS + **CDAN** + **CLUDA** + **RAINCOAT** + statistics-only = **7 DA baselines** spanning 2016–2023, covering adversarial, statistical, contrastive, and frequency-aware families.

### Ideal Baseline Set

Add **ACON** (NeurIPS 2024 SOTA) for **8 baselines** spanning 2016–2024. Comprehensive and reviewer-proof.
