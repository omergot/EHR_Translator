# RESULTS LEDGER — Paper-Scope Curated Index

**SSOT rule**: Every Value cell repeats the headline delta for convenience; authoritative numbers live in the linked section. Fix numbers in the master file, not here.

Master: `docs/comprehensive_results_summary.md` | Bootstrap CIs: `docs/neurips/bootstrap_ci_results.md` | Seed-level ablation pool: `docs/neurips/multiseed_ablation_tables.md`

**Apr 22**: Ledger headline values UNCHANGED by Apr 22 seed sweep (champions are `_nf_C0` / `_hp_K5_lr3e5` / `_C4_no_target_task` / `_C5_no_fidelity` configs, not `_v5_cross3` / `_v5_cross2` which got the new seeds). See `multiseed_ablation_tables.md` § Discrepancies for the 11 new seed cells added.

---

## 1. Primary Results — eICU → MIMIC

### 1a. Classification (AUROC / AUCPR)

| Task | Metric | Value | Authority (file § section) | Seed count | CI [95%] | p | Notes |
|---|---|---|---|---|---|---|---|
| Mortality | AUROC Δ | **+0.0459 ± 0.003 (3 seeds)** | `experiments/results/mort_c2_nf_{C0_control,s42,s7777}_mortality.json` (NEW CHAMPION); old CI: `bootstrap_ci_results.md` § eICU L16 | 3 | bootstrap re-run on 3-seed mean pending | — | **New champion: `mort_c2_nf` (n_cross=2, no-fidelity base).** Per-seed Δ: +0.0496 / +0.0421 / +0.0459; translated AUROC 0.8576 / 0.8656 / 0.8596. Previous: `mortality_retr_v4_mmd_local` +0.0457 single-seed. |
| Mortality | AUCPR Δ | +0.0521 | `bootstrap_ci_results.md` § eICU Classification L17 | 1 | [+0.0352, +0.0692] | <0.001 | Config: `mortality_retr_v4_mmd_local` |
| AKI | AUROC Δ | +0.0576 | `bootstrap_ci_results.md` § eICU Classification L18 | 1 (ablation base) | [+0.0549, +0.0563] | <0.001 | Config: `aki_nf_C0_control` (JSON verified) |
| AKI | AUCPR Δ | +0.1734 | `bootstrap_ci_results.md` § eICU Classification (aki_v5_cross3 row L19); `experiments/results/aki_nf_C0_control_aki.json` | 1 | [+0.1590, +0.1623] | <0.001 | aki_nf_C0 AUCPR=+0.1734 from JSON; CI row is for aki_v5_cross3 |
| Sepsis | AUROC Δ | +0.0633 | `experiments/results/sepsis_C4_no_target_task_sepsis.json` | 1 | — | — | Config: `sepsis_C4_no_target_task`; CI from `adaptive_ccr_sepsis`: [+0.0584,+0.0650] |
| Sepsis | AUCPR Δ | +0.0225 | `da_baselines_results.md` § 1c (Our Best row) | 1 | [+0.0217, +0.0251] | <0.001 | Config: `adaptive_ccr_sepsis` per bootstrap_ci L21 |

### 1b. Regression (MAE / RMSE / R²)

| Task | Metric | Value | Authority (file § section) | Seed count | CI [95%] | p | Notes |
|---|---|---|---|---|---|---|---|
| LoS | MAE Δ | −0.0320 | MEMORY.md "Current Best Results" (los_nm_C5); `bootstrap_ci_results.md` L27 (los_retr_v5_cross3 Δ=−0.0196) | 1 | [−0.0197, −0.0194] | <0.001 | CI is for v5_cross3 control, not nm_C5 best |
| LoS | RMSE Δ | −0.0119 | `bootstrap_ci_results.md` § eICU Regression L28 | 1 | [−0.0121, −0.0117] | <0.001 | Config: `los_retr_v5_cross3` |
| LoS | R² Δ | +0.0664 | `bootstrap_ci_results.md` § eICU Regression L29 | 1 | [+0.0653, +0.0674] | <0.001 | Config: `los_retr_v5_cross3` |
| KF | MAE Δ | −0.0103 | MEMORY.md "Current Best Results" (kf_hp_K5_lr3e5); `bootstrap_ci_results.md` L32 (`kf_C5_no_fidelity` Δ=−0.0074) | 1 | [−0.0079, −0.0068] | <0.001 | Best is kf_hp_K5_lr3e5; CI is for kf_C5_no_fidelity |
| KF | RMSE Δ | −0.0123 | `bootstrap_ci_results.md` § eICU Regression L33 | 1 | [−0.0136, −0.0111] | <0.001 | Config: `kf_C5_no_fidelity` |
| KF | R² Δ | +0.0922 | `bootstrap_ci_results.md` § eICU Regression L34 | 1 | [+0.0850, +0.0995] | <0.001 | Config: `kf_C5_no_fidelity` |

---

## 2. Primary Results — HiRID → MIMIC

### 2a. Classification

| Task | Metric | Value | Authority (file § section) | Seed count | CI [95%] | p | Notes |
|---|---|---|---|---|---|---|---|
| Mortality | AUROC Δ | +0.0488 | `bootstrap_ci_results.md` § HiRID Classification L44 | 1 | [+0.0284, +0.0711] | <0.001 | Config: `mortality_hirid_sr` |
| Mortality | AUCPR Δ | +0.0502 | `bootstrap_ci_results.md` § HiRID Classification L45 | 1 | [+0.0167, +0.0859] | 0.002 | Wide CI (2,560 test stays) |
| AKI | AUROC Δ | +0.0776 | `bootstrap_ci_results.md` § HiRID Classification L46 | 1 | [+0.0745, +0.0807] | <0.001 | Config: `aki_hirid_sr` |
| AKI | AUCPR Δ | +0.1471 | `bootstrap_ci_results.md` § HiRID Classification L47 | 1 | [+0.1414, +0.1531] | <0.001 | |
| Sepsis | AUROC Δ | +0.0777 | `bootstrap_ci_results.md` § HiRID Classification L48 | 1 | [+0.0707, +0.0841] | <0.001 | Config: `sepsis_hirid_sr` |
| Sepsis | AUCPR Δ | +0.0525 | `bootstrap_ci_results.md` § HiRID Classification L49 | 1 | [+0.0471, +0.0584] | <0.001 | |

### 2b. Regression — HiRID LoS MAE Double-Recording Resolution

**RESOLVED**: Two values exist for HiRID LoS MAE delta.
- MEMORY.md: **−0.3050** — normalized MAE delta (YAIB normalizes LoS labels; ~51h in raw hours). Not comparable to code metric.
- `bootstrap_ci_results.md` L55: **−0.0452** [−0.0497, −0.0410], p<0.001 — code metric, absolute units matching all other MAE rows.

**Paper-reportable**: −0.0452 from `bootstrap_ci_results.md` L55. The R² Δ=+1.5594 confirms large lift from a very poor HiRID baseline (R²=−3.89). Unit conversion factor needs final verification. **FLAG: user decision recommended before citing in abstract.**

| Task | Metric | Value | Authority | Seed count | CI [95%] | p | Notes |
|---|---|---|---|---|---|---|---|
| LoS (HiRID) | MAE Δ | −0.0452 | `bootstrap_ci_results.md` § HiRID Regression L55 | 1 | [−0.0497, −0.0410] | <0.001 | Absolute units per code metric |
| LoS (HiRID) | R² Δ | +1.5594 | `bootstrap_ci_results.md` § HiRID Regression L57 | 1 | [+1.3338, +1.8344] | <0.001 | Baseline R²=−3.89 (extrapolation task) |
| KF (HiRID) | MAE Δ | +0.0000 | `bootstrap_ci_results.md` § HiRID Regression L58 | 1 | [−0.0009, +0.0010] | 0.482 (n.s.) | Config: `kf_hirid_sr` |

---

## 3. Native-Baseline Comparisons

Reference: `docs/comprehensive_results_summary.md` § 1.1 and `docs/yaib_reference_baselines.md`

| Task | Source | Claim | Our Best AUROC | Threshold | Exceeds? |
|---|---|---|---|---|---|
| Mortality | eICU | Exceeds eICU-native LSTM | 85.36 | 85.5 | No (−0.14) — close |
| Mortality | eICU | Exceeds MIMIC-native LSTM | 85.36 | 86.7 | No (−1.34) |
| AKI | eICU | Exceeds eICU-native LSTM | 91.34 | 90.2 | **YES (+1.14)** |
| AKI | eICU | Exceeds MIMIC-native LSTM | 91.34 | 89.7 | **YES (+1.64)** |
| AKI | eICU | Exceeds best-eICU (GRU) | 91.34 | 90.9 | **YES (+0.44)** |
| Sepsis | eICU | Exceeds eICU-native LSTM | 77.92 | 74.0 | **YES (+3.92)** |
| Sepsis | eICU | Exceeds MIMIC-native LSTM | 77.92 | 82.0 | No (−4.08) |
| Mortality | HiRID | Exceeds HiRID-native LSTM | 80.81 | 84.0 | No (−3.19) |
| AKI | HiRID | Exceeds HiRID-native LSTM | 82.96 | 81.0 | **YES (+1.96)** |
| Sepsis | HiRID | Exceeds HiRID-native LSTM | 79.79 | 78.8 | **YES (+0.99)** |

Authority: `comprehensive_results_summary.md` § 1.1.

---

## 4. Clinical DA Baselines (eICU → MIMIC, right-padded, complete)

### 4a. Frozen-Backbone DA (primary fairness comparison — same constraint as ours)

Authority: `da_baselines_results.md` § 1b and Appendix B; `baseline_strategy_final.md` § 1b

| Method | Mortality AUROC Δ | AKI AUROC Δ | Sepsis AUROC Δ | Notes |
|---|---|---|---|---|
| Statistics-only (affine renorm) | −0.0100 | −0.0112 | −0.0072 | `da_baselines_results.md` § 3g |
| DANN v1 | +0.0355 | +0.0316 | +0.0164 | `da_baselines_results.md` § 3a |
| Deep CORAL v1 | +0.0374 | +0.0308 | +0.0167 | `da_baselines_results.md` § 3b |
| CoDATS v1 | +0.0352 | +0.0126 | −0.0037 | `da_baselines_results.md` § 3c |
| ACON v2 | +0.0369 | +0.0304 | −0.0051 | `da_baselines_results.md` § 3d |
| T3A (TTA, prototype-cosine, n=3 seeds) | −0.0018 | −0.0340 | +0.0065 | `tta_baselines.md` § 2 (partial multi-seed; aggregator `/tmp/agg_tta.py`) |
| SHOT-canonical (TTA, unfreezes feat., n=4/3/2 Mort/AKI/Sepsis) | −0.0058 | **−0.2055** | −0.0110 | `tta_baselines.md` § 2 (sepsis s7777 in-flight on a6000; partial multi-seed; aggregator `/tmp/agg_tta.py`) |
| **Our best** | **+0.0476** | **+0.0576** | **+0.0633** | `da_baselines_results.md` § 4d + JSON |

Advantage: Mortality +27%, AKI +76%, Sepsis +207% vs best frozen-backbone DA. `da_baselines_results.md` § 5. T3A/SHOT-canonical added as TTA baselines (partial multi-seed; n=3 for T3A, n=2–4 for SHOT-canonical; multi-seed SHOT on a6000 still executing; `tta_baselines.md` §§ 1–2; aggregator `/tmp/agg_tta.py`).

### 4b. E2E DA (full retrain — less fair, informative context)

**Our adaptor wins all 18 comparisons (6 methods × 3 tasks) on absolute AUROC.**
Authority: `da_baselines_results.md` § 9c; `e2e_baselines_audit.md` § 9b-ii (right-padded LSTM); `baseline_strategy_final.md` § 0.

| Context | Claim | Authority |
|---|---|---|
| All 18 E2E comparisons (DANN/CORAL/CoDATS/ACON/CLUDA/RAINCOAT × 3 tasks) | We win all 18 on absolute AUROC | `da_baselines_results.md` § 9c Summary Table |
| AUROC deltas (right-padded, absolute): Mortality +0.048, AKI +0.056, Sepsis +0.051 | Our adaptor outperforms best validated E2E | `da_baselines_results.md` § 9b-ii |
| Left-padding bug (resolved) | Historical LSTM per-timestep inflation; fixed Mar 30. All current numbers use right-padding matching YAIB | `baseline_strategy_final.md` L10-14; `e2e_baselines_audit.md` § 9b-iii |

### 4c. E2E Fine-Tuning Upper Bound

Our adaptor wins all 3 clinical classification tasks against a fully-retrained MIMIC LSTM.

| Task | Our AUROC | Fine-Tuned LSTM AUROC | We Win? |
|---|---|---|---|
| Mortality | 0.856 | 0.846 | **YES** |
| AKI | 0.911 | 0.905 | **YES** |
| Sepsis | 0.767 | 0.746 | **YES** |

Authority: `da_baselines_results.md` § 2 (e2e_fine_tuning_upper_bound rows); plan § "E2E fine-tuning upper bound" confirmed Apr 18.

---

## 5. AdaTime (Non-Clinical, 5 Sensor Datasets)

**Protocol**: v4/v5, 40 total epochs, last-epoch, Adam(β=0.5,0.99), no val split — fully compliant with Ragab 2023 (TKDD). **Seed count**: 5 per dataset; all seeds beat prior best on all 5 datasets. Authority: `adatime_experiments_summary.md` § "Results for Paper" + § "Multi-Seed Results"; `experiments/results/bootstrap_cis/adatime_bootstrap_cis.json`.

| Dataset | Our MF1 (5-seed mean) | Best Published E2E Method | Best Published MF1 | Win Margin | Per-seed: all beat prior best? |
|---|---|---|---|---|---|
| HAR | 94.1±0.0 | DIRT-T | 93.7 | +0.4 | Yes (all 5 identical at 94.1) |
| HHAR | 87.0±0.7 | CoTMix | 84.5 | +2.5 | Yes (worst seed 86.0 > 84.5) |
| WISDM | 70.3±1.5 | CoTMix | 66.3 | +4.0 | Yes (worst seed 68.4 > 66.3) |
| SSC | 66.2±0.2 | MMDA | 63.5 | +2.7 | Yes (worst seed 65.9 > 63.5) |
| MFD | 96.1±0.1 | DIRT-T | 92.8 | +3.3 | Yes (worst seed 96.0 > 92.8) |
| **5-dataset mean** | **82.7** | DIRT-T | 77.3 | **+5.4** | **5/5 datasets, 5/5 seeds** |

**11 published E2E DA methods** (Ragab 2023 Table 4): DANN, Deep CORAL, CDAN, DSAN, MMDA, DIRT-T, AdaBN, BNM, SHOT-IM, SDAT, CoTMix. Authority: `adatime_experiments_summary.md` § "Comparison with AdaTime Published Baselines". All p<0.0001 (bootstrap, 2000 replicates).

### 5a. TTA Baselines on AdaTime (multi-seed, 55 scenarios × 5 datasets × 2–3 seeds)

Authority: `tta_baselines.md` § 3a–3c; `experiments/results/tta/*.json`.
Siblings: `tta_failure_analysis.md` (per-method mechanism),
`tta_meta_review.md` (NeurIPS reviewer verdict).
Aggregators: `/tmp/agg_tta.py` (Δ AUROC), `/tmp/agg_tta_f1.py` (Macro-F1).
**n-counts per cell**: T3A 30–31, SHOT-canonical 30, TENT 30–31, SAR 17–20,
EATA 17–20 (SAR/EATA s7777 still running on local GPU 3). All rows are
**partial multi-seed** — re-aggregate with `/tmp/agg_tta.py` when queue drains.

**AdaTime Δ AUROC (mean over scenarios × seeds):**

| Method | HAR | HHAR | WISDM | SSC | MFD |
|---|---|---|---|---|---|
| T3A | −0.0122 | +0.0093 | −0.0011 | +0.0160 | −0.0570 |
| SHOT-canonical | −0.0943 | −0.0094 | −0.0605 | −0.0092 | −0.0566 |
| TENT | +0.0051 | +0.0084 | +0.0424 | +0.0310 | +0.0202 |
| SAR | +0.0052 | +0.0079 | +0.0429 | +0.0302 | +0.0190 |
| EATA | +0.0053 | +0.0083 | +0.0434 | +0.0310 | +0.0193 |

**AdaTime Macro-F1 × 100 — absolute:**

| Method | HAR | HHAR | WISDM | SSC | MFD | Mean |
|---|---|---|---|---|---|---|
| Source-only | 80.2 | 59.1 | 49.5 | 58.5 | 77.7 | 65.0 |
| T3A | 85.0 | 62.3 | 53.5 | 63.8 | 81.0 | 69.1 |
| SHOT-canonical | 60.6 | 76.8 | 56.2 | 62.4 | 82.7 | 67.7 |
| TENT | 83.9 | 64.4 | 57.5 | 63.7 | 93.4 | 72.6 |
| SAR | 83.6 | 65.7 | 56.7 | 62.2 | 93.1 | 72.2 |
| EATA | 83.4 | 65.4 | 56.7 | 62.2 | 93.2 | 72.2 |
| **Ours (retrieval adaptor)** | **94.1** | **87.0** | **70.3** | **66.2** | **96.1** | **82.7** |

**AdaTime Δ Macro-F1 × 100 (vs source-only):**

| Method | HAR | HHAR | WISDM | SSC | MFD | Mean |
|---|---|---|---|---|---|---|
| T3A | +4.4 | +3.6 | +3.7 | +4.8 | +3.4 | **+4.0** |
| SHOT-canonical | **−19.3** | **+17.2** | +7.0 | +3.4 | +5.0 | +2.7 |
| TENT | +3.4 | +5.7 | +7.7 | +4.7 | +15.8 | **+7.5** |
| SAR | +3.6 | +6.1 | +8.6 | +4.5 | +15.4 | **+7.6** |
| EATA | +3.5 | +5.8 | +8.6 | +4.6 | +15.5 | **+7.5** |
| **Ours** | **+13.9** | **+27.9** | **+20.6** | **+7.2** | **+18.4** | **+17.7** |

Our adaptor wins every AdaTime dataset on both absolute Macro-F1 and Δ
Macro-F1; best TTA (SAR / TENT / EATA) plateaus at +7.5–+7.6 mean Δ MF1
vs our +17.7. SHOT-canonical is bimodal (collapses on HAR, rescues HHAR).

### 5b. A3 Pretrain Ablation — both directions (HAR/SSC/MFD p0→p10 + HHAR/WISDM p10→p0)

Authority: `docs/neurips/adatime_pretrain_ablation.md`. Mechanism: `memory/project_pretrain_asymmetry_analysis.md`. Configs: `experiments/.athena_configs/adatime_{har,ssc,mfd}_*_s*.json`. Claim: **C-64**. MF1 × 100.

| Dataset | Champion (p=0, t=40) | p=10 + t=30 s0 | p=5 + t=35 s0 | p=10 + t=40 s0 | Best recovery (multi-seed) | Δ vs champion |
|---|---|---|---|---|---|---|
| HAR | 94.1 ± 0.0 (5-seed) | 92.05 | 92.69 | 92.22 | **92.95 ± 0.67** (5-seed `lowfid_slowhead`) | **−1.15 (1.7σ)** |
| SSC | 66.2 ± 0.2 (5-seed) | 58.82 | 61.13 | 59.12 | — (gate-skipped, |Δ|≫1) | **−7.38** single-seed |
| MFD | 96.1 ± 0.1 (5-seed) | 83.07 | — | 82.98 | — (gate-skipped, |Δ|≫1) | **−13.0 / −13.1** (t=30 / t=40, matched) |

**Reverse direction (2026-04-21, champion p=10 → p=0 + t=30→40, 5-seed):**

| Dataset | p=10 champion (5-seed) | **p=0 new (5-seed)** | Δ | z-score |
|---|---|---|---|---|
| HHAR | 87.0 ± 0.7 | **88.70 ± 0.93** | +1.70 | +3.27σ |
| WISDM | 70.3 ± 1.5 | **74.13 ± 2.48** | +3.83 | +2.95σ |

**Combined finding: all 5/5 AdaTime datasets prefer pretrain=0** under the Adaptor + AdaTime protocol (HAR/SSC/MFD champions already p=0; HHAR/WISDM now improve when flipped to p=0). Win-margin vs published baselines improves: HHAR vs CoTMix was +2.5, now **+4.2**; WISDM vs CoTMix was +4.0, now **+7.8**.

**Phase-2 multi-seed** executed for HAR (11 pretrain=10 recovery variants s0; best s0 `pr_coldstart` 93.68; 5-seed verification of `lowfid_slowhead` yielded 92.95 ± 0.67 — structurally below champion) and now also HHAR/WISDM p=0 (72963–72972, both positive).

**Interpretation.** Pretrain-ON is universally worse at the AdaTime protocol on the 3 champion-OFF datasets. SSC/MFD damage (Δ=−7 to −13 MF1) is far beyond tuning noise; HAR gap closes to ~1 MF1 (1.7σ) but does not disappear. Together with HHAR (p=10 champion +27.9 MF1 over source-only) and WISDM (p=10 champion +20.6), this gives a regime-split — not a universal pretrain recipe.

**Mechanism** (see `docs/neurips/adatime_pretrain_ablation.md` §4): identity-basin — on 1-channel AdaTime (SSC/MFD) Phase-1 autoencoder reconstruction hits numerical zero by epoch 5–10, welding encoder+decoder to identity; 30 task epochs cannot escape. On EHR classification the basin is unreachable (fid/task gradient ratio 3–10×, cos(task,fid)=−0.21 on sepsis), so Phase 1 is the only viable init (C6 confirms catastrophic pretrain=0: mortality_nf −0.1413, sepsis −0.0763, AKI_nf −0.0828 AUROC).

**Open predictions** (agent-authored, falsifiable): HHAR p=0 predicted to lose 2–5 MF1 (large src→target shift needs pretrain scaffold); WISDM p=0 predicted to **win or tie** champion (N=178, λ_fid=0.01 already minimal). Not yet tested.

### 5c. A3 Output-Mode Strict-Toggle Run (Apr 26, claim-strengthening) — refutes `p > 0 → absolute`

8 strict-toggle cells across HAR/HHAR/WISDM (Athena jobs 74033, 74036–74042, single seed unless noted; HHAR `v4_base` has s1 added → n=2). All cells use AdaTime protocol; `output_mode` is the only toggled axis within each pair. Authority: `docs/neurips/playbook_drafts/adatime_claim_strengthening_run.md` Phase 4–5; `docs/adatime_experiments_summary.md` §"Residual vs Absolute Strict-Toggle Run — Apr 26 update"; `docs/neurips/playbook_drafts/output_mode_multivariable_audit.md` Phase 5; `docs/neurips/adatime_input_adapter_playbook.md` §1 (A3) + §6.

| Cell | RES MF1×100 | ABS MF1×100 | Winner | Margin | Wilcoxon p (n=10 paired) |
|---|---:|---:|---|---:|---|
| HAR `cap_T` p=10 | **92.31** | 67.45 | RES | +24.86 | ≈ 0.009 (Cohen's d ≈ +1.40) |
| WISDM `cap_T` p=0 (vs existing RES baseline ≈ 80.38) | ≈80.38 | 51.61 | RES | ~+28.8 | (existing-baseline reference) |
| HHAR `cap_T` p=0 (vs existing RES baseline ≈ 91.73) | ≈91.73 | 79.16 | RES | ~+12.6 | (existing-baseline reference) |
| WISDM `v4_lr67` p=10, λ_fid=0.5 | **71.63** | 58.22 | RES | +13.41 | ≈ 0.093 (borderline) |
| HHAR `v4_base` p=10, s1 (was s0 ABS-win +3.6) | **89.29** | 88.42 | RES | +0.87 | ≈ 0.88 (no preference; ABS wins 6/10) |

**2-seed mean at the only previous ABS-win cell** (HHAR `v4_base`): RES 87.45 ± 1.85 vs ABS 88.81 ± 0.39 → ABS by +1.36 MF1 *within seed σ ≈ 1.4* — i.e. tie, not sign flip.

**New rule (replaces previously-documented `p > 0 → absolute` direction)**:
- AdaTime: `output_mode = "residual"` universally — wins or ties at every measured `pretrain_epochs × λ_fidelity` cell.
- EHR: `output_mode = "absolute"` universally — frozen LSTM + tabular ICU feature regime (5/5 tasks at n=3, C8 strict toggle).

The cross-benchmark split is keyed on the predictor + feature regime, not on `pretrain_epochs`. λ_fidelity is the *magnitude* axis on AdaTime (Pearson r ≈ −0.93 across `p = 0` datasets), not a direction axis. Honest seed-count flag: 6 of the 7 new strict-toggle pairs are n=1 single-seed; HHAR `v4_base` is now n=2.

Code/landing commits: `347db3c` (rewrite of A3/A6), `e9c5b27` (multi-variable audit Phase 5).

---

## 6. Architecture Agnosticism (MAS — LSTM-Trained Adaptor on Frozen GRU/TCN)

LSTM-trained retrieval adaptor evaluated zero-shot on frozen GRU and TCN (no retraining). Authority: MEMORY.md § "MAS"; `gap_analysis_generality_claim.md` L24. Seed count: 1 (multi-seed not yet run).

| Task | LSTM Δ (original) | GRU Δ | GRU retention % | TCN Δ | TCN retention % |
|---|---|---|---|---|---|
| Sepsis | +0.0512 | +0.0240 | 47% | +0.0442 | 86% |
| AKI | +0.0556 | +0.0311 | 56% | +0.0316 | 57% |
| Mortality | +0.0476 | +0.0404 | 85% | +0.0342 | 72% |

**FLAG**: All MAS results are 1-seed. Multi-seed MAS not yet run. Result JSON files in `experiments/results/mas_*.json` have empty metrics fields — detailed numbers sourced from MEMORY.md.

---

## 7. Component Ablations C0–C9

**Design**: One component removed at a time vs C0 (full model). Apr 16 batch: 156 multi-seed configs complete. Authority: MEMORY.md § "Key ablation findings"; `comprehensive_results_summary.md` (ablation tables); `experiments/results/*_C{0-9}_*.json`.

Key findings only (full tables in `comprehensive_results_summary.md` via update-results skill):

| Component | Key finding | Authority |
|---|---|---|
| C4 no target task | Sepsis best (+0.0633); AKI hurts (−0.034) — sparse labels conflict with target task signal | MEMORY.md § ablation findings |
| C5 no fidelity | Sepsis catastrophic (−0.0665); LoS/KF best — fidelity penalizes cumulative features | MEMORY.md § ablation findings |
| C6 no pretrain | Classification catastrophic (AKI −0.0828, Sepsis −0.0763); regression mild | MEMORY.md § ablation findings |
| C8 residual | Mortality_nf best; AKI near-zero; KF −0.0007 | MEMORY.md § ablation findings |
| C2 no feature gate | Inert without fidelity (5 bases tested); active with fidelity | MEMORY.md § ablation findings |

### 7a. Retrieval-Max ablation (Apr 25) — knob-tuned C0 vs C1 reference

Tests whether retrieval-specific knob tuning (`memory_refresh_epochs 3→1` + per-task context knob) closes the C0-vs-C1 gap that the n=3 ablation (§7) reported as a tie/loss. **NEGATIVE on both probed tasks**.

| Task | Config | n=3 Δ AUROC | C1 reference | Margin |
|---|---|---|---|---|
| AKI | `aki_retrMax_mr1_w25` (mr=1, window=25) | +0.0503 ± 0.0054 | +0.0514 ± 0.0017 (Table C C1) | −0.19σ (tie) |
| Sepsis | `sepsis_retrMax_mr1_k8` (mr=1, k=8) | +0.0430 ± 0.0125 | +0.0488 ± 0.0031 (Table E.1 C1) | −0.45σ (loss) |

Authority: `comprehensive_results_summary.md` §25; `multiseed_ablation_tables.md` Table R; `memory/experiment_history.md` Apr 25. Disconfirms "stale memory bank" hypothesis. No follow-up retrMax queued for Mortality/LoS/KF.

---

## 8. Seed Variance

Authority: MEMORY.md § "Seed Variance Analysis"; `docs/neurips/statistical_completeness.md`

| Task | Config | Seeds | Mean Δ | Spread | Server |
|---|---|---|---|---|---|
| AKI | V5 cross3 | 5 | +0.0547 | 0.0027 | Cross-server |
| Mortality | V4+MMD | 3 | +0.0456 | 0.0029 | 3 servers |
| Sepsis | V4+MMD | 3 | +0.0494 | 0.0203 | High variance (1.1% label density) |

---

## 9. Calibration (ECE / Brier) — All 6 Classification Experiments

Authority: `docs/neurips/calibration_analysis.md` § 2 (eICU) and § 3 (HiRID)

| Source | Task | Orig ECE | Trans ECE | Δ ECE | Orig Brier | Trans Brier | Δ Brier |
|---|---|---|---|---|---|---|---|
| eICU | AKI | 0.1913 | 0.1839 | −0.0074 | 0.1365 | 0.1185 | −0.0180 |
| eICU | Mortality | 0.2402 | 0.1986 | −0.0416 | 0.1451 | 0.1187 | −0.0264 |
| eICU | Sepsis | 0.3217 | 0.2484 | −0.0733 | 0.1837 | 0.1067 | −0.0770 |
| HiRID | AKI | 0.2710 | 0.2669 | −0.0041 | 0.2038 | 0.1723 | −0.0315 |
| HiRID | Mortality | 0.3508 | 0.2657 | −0.0851 | 0.2429 | 0.1739 | −0.0690 |
| HiRID | Sepsis | 0.4577 | 0.2632 | −0.1945 | 0.3092 | 0.1390 | −0.1702 |

All 6 experiments show improved calibration. No explicit calibration loss was used.

---

## 10. Computational Cost and Hardware

Authority: `docs/neurips/computational_cost.md` § 2

| Setting | Hardware | Per-task time | Total EHR campaign | AdaTime (5 seeds) |
|---|---|---|---|---|
| EHR classification tasks | V100S-32GB / A6000 / RTX3090 / Athena L40S | 14–23h per task | ~270–450 GPU-hours | ~125 GPU-hours |
| AdaTime (all 5 datasets × 5 seeds) | V100S-32GB / A6000 / Athena L40S-A100 | ~25-38 min/scenario | — | ~125h |
| Overall campaign (Mar 27–Apr 13) | Athena L40S / RTX6K mix + local | — | **~50–60 GPU-hours** (baseline exps only) | — |

Adaptor 2.6–2.9M vs frozen LSTM 0.17–1.42M (ratio 2–15×). `computational_cost.md` § 1.9.

---

## 10a. Adapter Capacity Sweep (A6) — 2026-04-22

Authority: `docs/neurips/adapter_capacity_sweep.md`. Per-seed raw metrics:
`python scripts/gpu_scheduler.py --status --all | grep -E "_cap_[STXT]+_s"` (EHR);
`experiments/results/adatime_cnn_*_cap_{T,XT}_s{0,1,2}.json` on Athena (AdaTime).

### 10a.1 EHR Goal-B (Tiny, ≤1× predictor) — 3-seed

| Task | Variant | Params | Ratio | Mean Δ ± std | vs champion Δ |
|---|---|---:|---:|---:|---:|
| Mortality | `mort_cap_S` | 415,938 | 0.86× | **+0.0475 ± 0.0013** AUROC | matches 3-seed champion +0.0459 |
| AKI | `aki_cap_T` | 1,425,538 | 1.00× | +0.0409 ± 0.0063 AUROC | 71% of 1-seed champion +0.0576 |
| Sepsis | `sepsis_cap_T` | 160,818 | 0.95× | +0.0194 ± 0.0070 AUROC | 31% of champion +0.0633 (n_cross=3 confound) |
| LoS | `los_cap_T` | 1,419,394 | 1.06× | **−0.0305 ± 0.0003** MAE | 95% of 1-seed champion −0.0320 |
| KF | `kf_cap_T` | 551,826 | 0.98× | **−0.0086 ± 0.0004** MAE | **ties 3-seed champion pool** (−0.0086 ± 0.0014) |

### 10a.2 EHR Goal-A (S/M, ≤2× predictor) — 3-seed

| Task | Variant | Params | Ratio | Mean Δ ± std | Retention |
|---|---|---:|---:|---:|---:|
| Sepsis | `sepsis_cap_S` | 299,842 | 1.77× | +0.0425 ± 0.0045 AUROC | 67% |
| Sepsis | `sepsis_cap_M` | 769,858 | 4.54× | +0.0261 ± 0.0059 AUROC | 41% |
| Mortality | `mort_cap_M` | 973,122 | 2.01× | +0.0425 ± 0.0028 AUROC | 93% |
| KF | `kf_cap_S` | 356,482 | 0.63× | −0.0090 ± 0.0002 MAE | 87% |
| KF | `kf_cap_M` | 984,706 | 1.75× | −0.0094 ± 0.0003 MAE (s42 via eval recovery) | 91% |

### 10a.3 AdaTime Goal-B (Tiny / Extra-Tiny, ≤1×) — 3-seed

| Dataset | Variant | Params | Ratio | Mean MF1 ± std | vs champion |
|---|---|---:|---:|---:|---:|
| HAR | `cap_T_drop10_ff144` (p=0, winner) | ~196K | 0.98× | **0.9438** (bit-identical × 3) | **beats** 0.9407 |
| HAR | `cap_T` (plain, p=0) | 195,514 | 0.97× | 0.9329 (× 3) | below gate (0.9368) |
| HHAR | `cap_T_p0` (p=0, winner) | 195,658 | 0.98× | **0.9093 ± 0.003** | **beats** 0.8704 by +0.039 |
| HHAR | `cap_T` (legacy p=10) | 195,658 | 0.98× | 0.8853 ± 0.016 | beats 0.8704 ± 0.007 by +0.015 |
| WISDM | `cap_T_p0` (p=0, winner) | 195,658 | 0.98× | **0.7595 ± 0.016** | **beats** 0.7027 by +0.057 |
| WISDM | `cap_T` (legacy p=10) | 195,658 | 0.98× | 0.6999 ± 0.003 | tie (champion 0.7027 ± 0.015) |
| SSC | `cap_T` (p=0) | 195,514 | 0.99× | 0.6635 ± 0.001 | matches 0.6624 ± 0.002 |
| SSC | `cap_XT` (p=0) | 68,018 | 0.34× | 0.6556 ± 0.0003 | −0.007 vs champion |
| MFD | `cap_T` (p=0) | 195,514 | 0.99× | 0.9626 ± 0.002 | beats 0.9608 ± 0.001 |
| MFD | `cap_XT` (p=0) | 68,018 | 0.34× | **0.9696 ± 0.002** | **beats** 0.53×-champion 0.9608 |

Source-only MF1 (for retention%): HAR 0.7997 · HHAR 0.5650 · WISDM 0.4996 · SSC 0.5757 · MFD 0.7683.

**Status**: 3-seed coverage complete on all 10 (task × benchmark) Goal-B cells as of 2026-04-23.

---

## 11. Unlogged Results (Queue for `update-results` Skill)

JSON files in `experiments/results/` not yet reflected in `comprehensive_results_summary.md`:
- `aki_nf_C{0-9}_*.json` — AKI no-fidelity ablation series (C0 verified: +0.0576/+0.1734)
- `aki_C{0-9}_*.json`, `sepsis_C4_*.json` (C4 verified: +0.0633/+0.0218), `mortality_retr_v4_mmd_local_*.json` (verified: +0.0456/+0.0521)
- `mortality_C*`, `kf_v6_pretrain_kf.json`, `los_v6_pretrain_los.json`, `los_hirid_sr_los.json`, `kf_hirid_sr_kf.json`
- `mas_*.json` — empty metrics; detailed numbers in MEMORY.md only

Ingest via `update-results` skill.

---

## 12. Open Items

Open items: see `docs/neurips/requirements.md`.

In-flight: full multi-seed ablation verification (156 configs, Apr 16 batch complete); planned 3-seed + bootstrap-CI re-runs on current best configs (`aki_nf_C0`, `sepsis_C4_no_target_task`, `los_nm_C5`, `kf_hp_K5_lr3e5`). TTA baselines (T3A/SHOT-canonical on LSTM; T3A/SHOT-canonical/TENT/SAR/EATA on AdaTime) **partial multi-seed** (2026-04-19, `tta_baselines.md` §6): T3A/TENT/AdaTime-SHOT 3-seed complete; LSTM SHOT n=2–4 (a6000 s2222/s7777 in-flight); SAR/EATA n=2 partial (local GPU 3 s7777 still running). Pattern holds: TTA hurts or plateaus, our adaptor wins on every AdaTime dataset (mean Δ MF1 +17.7 vs best-TTA +7.6).

---

## Verification Checklist

- [x] 4 clinical-baseline categories: frozen-backbone DA, E2E DA (18/18), affine, E2E fine-tuning upper-bound (3/3) — §§ 4a–4c
- [x] AdaTime 11-method × 5-dataset × 5-seed — § 5
- [x] HiRID LoS MAE resolved + flagged — § 2b
- [x] Unlogged results — § 11
- [x] `requirements.md` cross-link — § 12
- [x] BEST_RESULTS.md: all 10 task×source champions with seed-count flags
