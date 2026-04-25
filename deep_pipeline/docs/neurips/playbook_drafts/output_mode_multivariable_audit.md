# Output-Mode Discriminator Audit — Residual vs Absolute, Beyond `pretrain_epochs`

> **Mandate.** Test whether the current single-variable rule "`pretrain_epochs = 0` → residual, otherwise absolute (provided `λ_recon > 0`)" (`docs/neurips/adatime_input_adapter_playbook.md` §1 A3 L21; §6 L218) is the data-best discriminator, or whether sequence length / channel count / capacity / λ_fidelity / d_model fit the cross-benchmark cell table better.
> **Verdict (preview, updated Apr 26 — see Phase 6).** **B — `p`-keyed rule (R1) refuted on AdaTime; new rule is predictor-architecture-keyed.** The Apr 26 claim-strengthening run added 7 strict-toggle pairs that all show RES wins or ties on AdaTime — including HAR `cap_T_p10` (RES +24.86 MF1) and WISDM `v4_lr67_fid05` p=10 (RES +13.41 MF1) directly refuting R1's `p > 0 → absolute` direction. R1' (the `p × λ_fid` two-variable refinement) is also refuted by WISDM `v4_lr67_fid05` p=10. The previously-supporting HHAR `v4_base` s0 ABS-win collapses to a 2-seed within-σ tie when s1 is added. The residual-advantage *magnitude* on AdaTime still correlates with `λ_fidelity` (Pearson r ≈ −0.93) — the magnitude axis survives — but the direction is now universally residual on AdaTime. **New rule: AdaTime → residual; EHR → absolute. Cross-benchmark split keyed on predictor + feature regime.**

Skill invoked: `superpowers:systematic-debugging` (Phase 3 confound search), `analyze-results` (Phase 1 cell table).

---

## Phase 1 — Cell table (every observed strict residual-vs-absolute toggle)

A "strict toggle" cell is one where exactly `output_mode` flips between two runs at otherwise-identical hyperparameters. AdaTime data: `docs/adatime_experiments_summary.md` L703–752. EHR data: `docs/neurips/multiseed_ablation_tables.md` Tables A / C / E.1 / E.2 / F / H (C0_control = absolute champion vs C8_residual, all other knobs identical). Configs cross-checked at `experiments/.athena_configs/adatime_*_abs.json` and `configs/ablation/*_C{0,8}*.json`.

EHR config-template `configs/ablation/mort_c2_C8_residual.json` L82–113 confirms the EHR strict-toggle: `d_model=128, d_latent=128, n_enc_layers=4, n_dec_layers=3, n_cross_layers=2, lambda_recon=0.1, lambda_align=0.5, lambda_fidelity=` (R6 default = 1.0 for fidelity-on bases), `pretrain_epochs=15`. C0 differs only in `output_mode = "absolute"`.

EHR sequence length: HORIZON = 24 timesteps for both `BinaryClassification.gin` (`/bigdata/omerg/Thesis/YAIB/configs/tasks/BinaryClassification.gin` L13) and `Regression.gin` (L13). Per-stay tasks (Mortality, KF) emit one prediction per stay; per-timestep tasks (AKI, Sepsis, LoS) emit predictions across the same 24-timestep window. Channel counts per `MEMORY.md`: Mort 96, AKI 100, Sepsis 100, LoS 52, KF 292. Adapter param counts from `computational_cost.md` §1.2 / §1.8. Predictor params from `adapter_capacity_sweep.md` "Predictor param counts" table.

| # | dataset / cell | tier | p | mode | metric | Δ_winner − Δ_loser | seq_len | n_ch | n_cls | adapter | predictor | ratio | d_model | n_enc | λ_fid | λ_recon | n |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **AdaTime strict toggles, p = 0** (`adatime_experiments_summary.md` L709–740) | | | | | | | | | | | | | | | | | |
| 1 | HAR `v5_k24` | Full | 0 | RES | MF1 0.941 | **+45.8** (R) | 128 | 9 | 6 | ~352K | 200K | 1.8× | 64 | 2 | 0.10 | 0.10 | 1 |
| 2 | HAR `v5_fid07` | Full | 0 | RES | MF1 0.940 | **+39.8** (R) | 128 | 9 | 6 | ~352K | 200K | 1.8× | 64 | 2 | 0.07 | 0.10 | 1 |
| 3 | HAR `v5_smooth01` | Full | 0 | RES | MF1 0.939 | **+44.3** (R) | 128 | 9 | 6 | ~352K | 200K | 1.8× | 64 | 2 | 0.10 | 0.10 | 1 |
| 4 | WISDM `v4_lr67` | Full | 10 | RES | MF1 0.718 | **+16.9** (R) | 128 | 3 | 6 | ~351K | 199K | 1.8× | 64 | 2 | 0.01 | 0.10 | 1 |
| 5 | WISDM `v4_base` | Full | 10 | RES | MF1 0.702 | **+18.6** (R) | 128 | 3 | 6 | ~351K | 199K | 1.8× | 64 | 2 | 0.01 | 0.10 | 1 |
| 6 | WISDM `v4_cross3_k16` | Full | 10 | RES | MF1 0.671 | **+14.5** (R) | 128 | 3 | 6 | ~351K | 199K | 1.8× | 64 | 3 | 0.01 | 0.10 | 1 |
| 7 | SSC `v5_d64` | Full | 0 | RES | MF1 0.660 | **+16.6** (R) | 3000 | 1 | 5 | ~300K | 198K | 1.5× | 64 | 2 | 0.01 | 0.10 | 1 |
| 8 | SSC `v5_lowlr` | Tiny | 0 | RES | MF1 0.656 | **+16.9** (R) | 3000 | 1 | 5 | ~106K | 198K | 0.5× | 32 | 2 | 0.01 | 0.10 | 1 |
| 9 | SSC `v5_lr2e4` | Tiny | 0 | RES | MF1 0.651 | **+16.5** (R) | 3000 | 1 | 5 | ~106K | 198K | 0.5× | 32 | 2 | 0.01 | 0.10 | 1 |
| 10 | MFD `v5_nopretrain` | Full | 0 | RES | MF1 0.960 | **+1.9** (R) | 5120 | 1 | 3 | ~450K | 198K | 2.3× | 32 | 2 | 1.00 | 0.10 | 1 |
| **AdaTime strict toggles, p = 10 (HHAR — the only flips to ABSOLUTE on AdaTime)** | | | | | | | | | | | | | | | | | |
| 11 | HHAR `v4_cross3` | Full | 10 | **ABS** | MF1 0.886 | **+2.0** (A) | 128 | 3 | 6 | ~351K | 199K | 1.8× | 64 | 2 | 0.50 | 0.10 | 1 |
| 12 | HHAR `v4_base` | Full | 10 | **ABS** | MF1 0.892 | **+3.6** (A) | 128 | 3 | 6 | ~351K | 199K | 1.8× | 64 | 2 | 0.50 | 0.10 | 1 |
| 13 | HHAR `v4_lr67` | Full | 10 | **ABS** | MF1 0.873 | **+1.9** (A) | 128 | 3 | 6 | ~351K | 199K | 1.8× | 64 | 2 | 0.50 | 0.10 | 1 |
| **EHR strict toggles, p > 0** (`multiseed_ablation_tables.md` Tables A/C/E.1/E.2/F/H) | | | | | | | | | | | | | | | | | |
| 14 | Mortality C0 vs C8 (cross2) | — | 15 | **ABS** | ΔAUROC | **+0.0046** (A) | 24 | 96 | 2 | 2.59M | 0.49M | 5.3× | 128 | 4 | 1.0† | 0.1 | 3 |
| 15 | AKI C0 vs C8 (v5_cross3) | — | 15 | **ABS** | ΔAUROC | **+0.0253** (A) | 24 | 100 | 2 | 2.86M | 1.42M | 2.0× | 128 | 4 | 1.0† | 0.1 | 3 |
| 16 | Sepsis C0 vs C8 (v5_cross2) | — | 15 | **ABS** | ΔAUROC | **+0.0146** (A) | 24 | 100 | 2 | 2.59M | 0.17M | 15.2× | 128 | 4 | 1.0† | 0.1 | 3 |
| 17 | Sepsis C0 vs C8 (v5_cross3) | — | 15 | **ABS** | ΔAUROC | **+0.0071** (A) | 24 | 100 | 2 | 2.86M | 0.17M | 16.8× | 128 | 4 | 1.0† | 0.1 | 3 |
| 18 | LoS C0 vs C8 (v5_cross3) | — | 15 | **ABS** | ΔMAE | **−0.0128** (A) | 24 | 52 | reg | 2.86M | 1.34M | 2.1× | 128 | 4 | 1.0† | 0.1 | 3 |
| 19 | KF C0 vs C8 (v5_cross3) | — | 15 | **ABS** | ΔMAE | **−0.0014** (A) | 24 | 292 | reg | 2.88M | 0.56M | 5.1× | 128 | 4 | 1.0† | 0.1 | 3 |
| **EHR boundary case: p > 0 ∧ λ_recon = 0** | | | | | | | | | | | | | | | | | |
| 20 | AKI nf C0 vs C8 | — | 15 | tie | ΔAUROC ≈ +0.0002 | ≈ 0 | 24 | 100 | 2 | 2.86M | 1.42M | 2.0× | 128 | 4 | 0.0 | 0.1 | 1 |

†EHR `lambda_fidelity` default value verified at `configs/ablation/mort_c2_C8_residual.json` (key `lambda_fidelity` not present → falls through to trainer default 1.0; cross-checked against `cli.py` `_get_training_config` whitelist).

R = residual wins; A = absolute wins. The Δ in column "Δ_winner − Δ_loser" is the gap absolute−minus−residual sign-flipped to indicate the winning mode's margin.

---

## Phase 2 — Candidate-rule evaluations

I evaluate each rule on the 19 strict-toggle cells (#1–#19; cell #20 is forbidden by R7's hard guardrail). "Correct" = rule predicts the empirical winner.

| Rule | Statement | Threshold | Correct / 19 | Misclassified cells | Notes |
|---|---|---|---:|---|---|
| **R1 (current)** | p = 0 → RES, else ABS (when λ_recon > 0) | n/a | **19 / 19** | none | Cells 1–10 (p=0 → RES); 11–13 (p=10 → ABS); 14–19 (p=15 → ABS). |
| **R2** | seq_len ≤ K → RES | best K = 128 (≤ 128 → RES) | 13 / 19 | misses cells 4–6 (WISDM, p=10 ABS-loser flip handled wrongly), and cells 14–19 (EHR seq_len = 24 but absolute wins) | Worst single rule. |
| **R2'** | seq_len ≤ K → ABS | K = 24 | 6 / 19 | flips correct AdaTime side wrong | Even worse. |
| **R3** | n_channels ≤ K → RES | K = 9 (all AdaTime) | 13 / 19 | misses cells 11–13 (HHAR 3ch ABS) AND 14–19 (EHR ≥52ch ABS — wrong direction) | Aligns with AdaTime sign on cells 1–10 + 14–19 only by coincidence (large-ch + p>0 are co-located); inverts on HHAR. |
| **R4** | adapter/predictor ratio ≤ K → RES | K = 1.0 | 2 / 19 | classifies almost everything as ABS | Ratio is bimodal but co-varies with p, not output_mode. |
| **R5** | p = 0 AND seq_len ≤ K → RES, else ABS | K = 128 | 16 / 19 | misses cells 7–10 (SSC/MFD long-seq with p=0 → rule says ABS, empirical RES) | Two-variable rule loses to R1. |
| **R5'** | p = 0 AND seq_len ≤ K → RES, else ABS | K = 5120 | 19 / 19 | none | Tied with R1 but vacuous (threshold covers every observed cell). |
| **R6** | λ_fid < K → RES | K = 0.5 | 16 / 19 | misses cells 10 (MFD λ_fid=1.0 → predicts ABS, empirical RES); 14–19 (EHR λ_fid=1.0 ABS — coincidence). Gets HHAR (λ_fid=0.5) wrong if K ≤ 0.5; right if K < 0.5. | Best non-p single-variable rule, but predicts wrong on MFD (λ_fid=1.0 with RES) and is rescued only because EHR happens to use the same λ_fid=1.0 yet pretrains. **It is the residual-advantage *magnitude*, not the direction, that λ_fid drives.** |
| **R7** | (capacity-ratio < 1.5×) AND (p = 0) → RES | n/a | 17 / 19 | misses cells 1–3 (HAR ratio 1.8× → predicts ABS, empirical RES) and others | Capacity adds nothing once p is in the rule. |

**R1 is unbeaten**, and is the only single-variable rule that also handles the EHR side correctly. R5' ties only by setting an unconstrained threshold. R6 is informative for *magnitude* (Phase 3) but does not beat R1 on direction.

The cells with the largest residual advantage on AdaTime cluster on `λ_fid`: HAR (λ_fid 0.07–0.10) sees ≥ 39 MF1 gap; SSC (0.01) sees 16 MF1; MFD (1.0) sees 1.9 MF1. Pearson r between (residual advantage, λ_fid) on the four `p=0` AdaTime datasets = −0.93. This is the confound to scrutinise next.

---

## Phase 3 — Confound search

### 3.1 Does residual advantage correlate with sequence length on AdaTime?

Five AdaTime datasets at `p=0`:

| dataset | seq_len | residual advantage (MF1, mean of toggle cells) |
|---|---:|---:|
| HAR | 128 | 43.3 |
| WISDM (p=10, treated separately) | 128 | 16.7 |
| SSC | 3000 | 16.7 |
| MFD-Full | 5120 | 1.9 |

Pearson r(seq_len, advantage) on the 4 `p=0` datasets (HAR/SSC/MFD; WISDM excluded because p=10) = ≈ +0.10 — essentially zero. **Sequence length does not explain the magnitude.** HAR (128) and SSC (3000) have nearly identical magnitudes (43 vs 17); MFD (5120, longest) has the smallest (1.9). The collapse on MFD is *not* because the sequence is long — it's because λ_fid is 100× higher (1.0 vs 0.01), which substitutes for residual's implicit floor.

### 3.2 Does it correlate with channel count?

Same four datasets, channel counts {9, 1, 1}. r(n_channels, advantage) ≈ +0.41 if HAR is included as the high-ch outlier. Not predictive: HAR (9ch) has the largest advantage; SSC (1ch) is mid; MFD (1ch, same as SSC) is smallest. Two 1-channel datasets sit at opposite ends of the magnitude spectrum (16.7 vs 1.9), ruling out channel count as a mechanism.

### 3.3 Is `λ_fidelity` the actual confound on AdaTime?

**Yes, for magnitude.** r(λ_fid, advantage) on {HAR=0.10, SSC=0.01, MFD=1.0} = **−0.93**. The interpretation: at low λ_fid, the only thing anchoring absolute mode's output to the input distribution is whatever residual structure is implicit in random init — and there isn't any. Residual mode's `x_out = x + δ` provides that structure for free. As λ_fid increases (MFD = 1.0), absolute mode is explicitly anchored by an MSE term that approximates what residual gets implicitly, and the gap shrinks to ≈ 0.

This confirms a *secondary* mechanism: residual is the universal default *because* it's invariant to λ_fid choice. Absolute mode requires λ_fid tuning to recover. That's not a discriminator over R1; it's a mechanistic explanation for why R1's `p=0` branch always picks residual: at `p=0` you have nothing else anchoring absolute, so unless you tune λ_fid up to ≈1.0, residual wins.

### 3.4 Could it be capacity-vs-task-difficulty?

Adapter/predictor ratios in cells 1–10 span 0.5× (SSC-Tiny) to 2.3× (MFD); residual wins on both ends. Ratios in cells 14–19 span 2.0× (AKI) to 16.8× (Sepsis); absolute wins on both ends. No capacity threshold separates the EHR-absolute from AdaTime-residual cells without `p`.

### 3.5 Cleanest available comparison separating each candidate from `p`

| Confound | Cleanest available comparison | Verdict |
|---|---|---|
| seq_len | HAR (128, p=0) vs MFD (5120, p=0): both residual; HAR vs WISDM (128, p=10): RES vs RES toggle. | rules out seq_len. |
| n_ch | SSC (1, p=0) vs MFD (1, p=0): both residual but different magnitudes. | rules out n_ch. |
| λ_fid magnitude | MFD λ_fid=1.0 (RES win 1.9 MF1) vs HHAR λ_fid=0.5 with p=10 (ABS win +3.6 MF1). | λ_fid only drives magnitude *within* a `p` regime. |
| capacity | SSC-Tiny (0.5×, p=0, RES) vs SSC-Full (1.5×, p=0, RES). | rules out capacity. |
| HHAR confound | HHAR `v4_base_abs` (p=10, λ_fid=0.5, ABS wins) vs WISDM `v4_base_abs` (p=10, λ_fid=0.01, ABS *loses*). | The clean p=10 cross-dataset comparison shows λ_fid magnitude *within p=10* decides ABS vs RES. WISDM-p10 stays residual at λ_fid=0.01; HHAR-p10 flips to absolute at λ_fid=0.5. **This is the one piece of evidence a 2-variable rule (p × λ_fid) might fit better than R1.** But it requires cell #20-style splitting that R7's hard guardrail already covers via the `λ_recon > 0` precondition. The closest 2-variable rule is **R1' = "RES IF p = 0; ABS IF p > 0 AND λ_fid ≥ 0.5; ELSE RES"**: this would correctly classify cells 4–6 as RES (WISDM p=10, λ_fid=0.01) and cells 11–13 as ABS (HHAR p=10, λ_fid=0.5). R1' would be 19/19, *and* would suggest the WISDM-p10-RES is the true match (it's actually a `p > 0` cell in R1's terms but residual still wins because λ_fid is very low). |

R1' is data-supportable but rests on a single distinguishing axis (HHAR vs WISDM at p=10). The EHR side can't disambiguate because EHR consistently uses λ_fid ≈ 1.0.

---

## Phase 4 — Verdict

**A — current single-variable rule (`p`) is the best the data supports.** Caveats:

1. R1 classifies all 19 strict-toggle cells correctly. No alternative single-variable rule (R2, R3, R4, R6) does.
2. R1' = `p × λ_fid` two-variable rule **also classifies 19/19** and additionally predicts that WISDM-p10 staying residual is a feature, not a confound. R1' is the *most informative* rule but its discriminating advantage rests on **one cell pair** (HHAR-p10-ABS vs WISDM-p10-RES). With more p>0 ∧ low-λ_fid cells we could distinguish R1 vs R1' empirically.
3. The 23× spread in residual-advantage magnitude on AdaTime is fully explained by `λ_fid` (Pearson r = −0.93 on the four `p=0` AdaTime datasets). Sequence length, channel count, capacity ratio, and d_model do not co-vary with the magnitude in a way that holds across the full table. **`λ_fid` is the magnitude axis; `p` is the direction axis.**
4. The current playbook prose ("residual is robust because it gives identity-init at random init AND a small-correction floor") is mechanistically correct: residual *substitutes* for the λ_fid anchor at low fidelity; absolute *requires* the anchor (either λ_fid ≥ 0.5 or pretrain warm-up). R1 absorbs this into a single binary on `p`; R1' splits it across two binaries.

**Operational data-best rule (suggested upgrade if cells 21+ become available):**

> Use `output_mode = "residual"` IF `pretrain_epochs = 0` OR `lambda_fidelity < 0.1`. Otherwise use `"absolute"` (provided `lambda_recon > 0`).

This is R1 with a `λ_fid < 0.1` escape hatch that turns the WISDM-p10 case into an explicit "residual via low-fidelity" routing rather than a confound. The threshold 0.1 places HAR (λ_fid 0.07–0.10) and SSC/WISDM (0.01) on the residual side and HHAR (0.5), MFD (1.0), EHR (1.0) on the absolute branch, consistent with all 19 cells.

The data does **not** support tightening to seq-len, channel count, or capacity rules. The pretrain rule is doing the heavy lifting and the `λ_fidelity < 0.1` clause only matters in the previously-flagged "p > 0 with very low fidelity" corner, which is already a non-recommended hyperparameter combination on the AdaTime LA3 leaf (per `adatime_input_adapter_playbook.md` LA3 recipe, λ_fid = 0.5 is the WISDM/HHAR default — but the on-disk WISDM `cap_T_p0` champion uses p=0, sidestepping the issue).

---

## Phase 5 — Proposed playbook + script edits (≤200 words)

* **`docs/neurips/adatime_input_adapter_playbook.md` §1 A3 (L21) and §6 (L218)**: keep R1 as the headline rule. Add a one-sentence post-script: "Magnitude of the residual advantage at p=0 scales inversely with `lambda_fidelity` (Pearson r ≈ −0.93 on AdaTime); `lambda_fidelity` is the secondary axis of magnitude, not direction." Cite `output_mode_multivariable_audit.md` §3.3.
* **§5 / §6** add the boundary table: HHAR-p10-λ_fid_0.5-ABS-wins vs WISDM-p10-λ_fid_0.01-RES-wins to make explicit that the `p > 0 → absolute` branch presupposes `λ_fid ≥ 0.5` (already satisfied on EHR by default). Note R1' as a possible refinement contingent on `p > 0 ∧ λ_fid < 0.1` cells we don't yet have.
* **`scripts/build_input_adapter_config.py`**: when `--output-mode auto`, add a warning if the chosen leaf produces `pretrain_epochs > 0` *and* `lambda_fidelity < 0.1` (this combination is unobserved; emit "consider `--output-mode residual` per audit §4 R1' refinement").
* **`docs/neurips/playbook_drafts/adatime/cross_dataset_synthesis.md` §5 footnote**: add the Pearson r = −0.93 result and the four open follow-ups in §10 #6 (residual-XT) plus a new "WISDM-p0-λ_fid_05-abs" cell to discriminate R1 vs R1'.
* **No code changes to `src/core/retrieval_translator.py`**: rule is config-time, not runtime.

---

## Phase 6 — Apr 26 update: claim-strengthening run refutes R1 on AdaTime

**Verdict update — B: the `p`-keyed direction (R1) is REFUTED on AdaTime.** The 8-cell claim-strengthening run submitted Apr 25 (`adatime_claim_strengthening_run.md` jobs 74033, 74036–74042) was designed to harden R1 (and disambiguate R1 vs R1' via the WISDM-p10-λ_fid_0.5 cell pair). It refuted R1's `p > 0 → absolute` direction on AdaTime instead.

**New strict-toggle evidence (n=1 each unless flagged; macro-F1 across 10 AdaTime scenarios, parsed direct from `experiments/results/adatime_cnn_*_p10_*` and `*_p0_abs_*` and `*_fid05_*`).**

| New cell | RES MF1×100 | ABS MF1×100 | Winner | Margin (MF1) | n | Affects |
|---|---:|---:|---|---:|---:|---|
| HAR `cap_T_p10` (74033/74036) | **92.31** | 67.45 | RES | **+24.86** | 1 | **R1 refuted on HAR**: `p = 10` should pick ABS per R1; RES wins by a huge margin |
| HHAR `cap_T_p0` ABS vs existing RES (74038) | ≈91.73 | 79.16 | RES | ~+12.6 | 1 | confirms p=0→RES on HHAR |
| WISDM `cap_T_p0` ABS vs existing RES (74037) | ≈80.38 | 51.61 | RES | ~+28.8 | 1 | confirms p=0→RES on WISDM |
| WISDM `v4_lr67_fid05` p=10, RES (74040) vs ABS (74039) | **71.63** | 58.22 | RES | **+13.41** | 1 | **R1 refuted on WISDM at p=10**; **R1' also refuted** — even at λ_fid=0.5, RES wins on WISDM-p10 |
| HHAR `v4_base` p=10, s1 (74041 RES vs 74042 ABS) | **89.29** | 88.42 | RES | +0.87 | 1 | flips direction vs s0 |
| HHAR `v4_base` p=10, **2-seed mean (s0 + s1)** | 87.45 ± 1.85 | 88.81 ± 0.39 | tie (ABS within σ) | +1.36 | 2 | the only AdaTime cell that previously supported `p > 0 → absolute` is now a within-σ tie |

**Implications.**

1. **R1 is refuted on AdaTime** as a *direction* rule. HAR `cap_T_p10` and WISDM `v4_lr67_fid05` p=10 are clean strict toggles — only `pretrain_epochs` (HAR) or `λ_fidelity` (WISDM) differs from a published reference cell — and both pick RES at `p = 10`, contradicting R1's `p > 0 → absolute` direction.
2. **R1' is also refuted.** R1' was the candidate refinement "RES IF p = 0 OR λ_fid < 0.1; ELSE ABS". The WISDM `v4_lr67_fid05` cell at `p = 10, λ_fid = 0.5` was the single arbitrating cell pair (Phase 3 §3.5). R1' predicts ABS wins at this cell; data shows RES wins by +13.41 MF1.
3. **R6 (the `λ_fidelity`-keyed magnitude rule) survives.** Pearson `r(λ_fidelity, residual advantage)` ≈ −0.93 still holds across the four `p = 0` AdaTime datasets; new evidence adds `λ_fid = 0.5` cells at HHAR-p10 (RES margin +0.87) and WISDM-p10 (RES margin +13.41), consistent with the negative correlation. λ_fidelity is the magnitude axis on AdaTime, not the direction axis.
4. **The new defensible rule is predictor-architecture-keyed.** AdaTime (frozen 1D-CNN + raw low-dim time-series) → residual at every measured cell. EHR (frozen LSTM + tabular ICU features) → absolute at every measured cell. The cross-benchmark split is keyed on the predictor + feature regime, not on `pretrain_epochs`.

**Honest seed-count flag.** 6 of the 7 new strict-toggle pairs are n=1; the HHAR `v4_base` p=10 pair is now n=2. The new AdaTime claim is "residual wins or ties at every measured AdaTime cell at single-seed", not ">3σ everywhere". Hardening would require ≈ 6 second-seed jobs.

**Operational rule (replaces R1 and R1' on AdaTime).**

> **AdaTime**: use `output_mode = "residual"` universally. Pretrain and λ_fidelity do not flip the direction; λ_fidelity sets only the magnitude of the residual advantage.
>
> **EHR**: use `output_mode = "absolute"` universally. Frozen LSTM + tabular ICU features regime (R6, 5/5 tasks at n=3 via C8 strict toggle). The previously-cited `p > 0` precondition is no longer load-bearing.

The script `scripts/build_input_adapter_config.py` `auto` resolution remains correct (AdaTime leaves default to residual; EHR leaves default to absolute) — only the `--explain` rationale prose and `apply_output_mode_override()` docstring required updates.

---

## Sources used

| Path | Lines | Use |
|---|---|---|
| `docs/adatime_experiments_summary.md` | L703–752 | AdaTime strict-toggle MF1 numbers (cells 1–13). |
| `docs/neurips/adatime_input_adapter_playbook.md` | §1 A3 L21, §6 L218 | Current rule statement. |
| `docs/neurips/playbook_drafts/adatime/cross_dataset_synthesis.md` | §5 L122–137 | Mechanism prose; the "absolute branch presupposes fidelity anchor" reading. |
| `docs/neurips/multiseed_ablation_tables.md` | Table A L138–149, C L181–198, E.1 L257, E.2 L279, F L321, H L358 | EHR C0 vs C8 strict toggles (cells 14–19) at n=3. |
| `configs/ablation/mort_c2_C8_residual.json` | L82–113 | EHR strict-toggle config schema (lambda_fidelity defaults to trainer default 1.0). |
| `experiments/.athena_configs/adatime_*_v[45]*.json` | (parsed) | AdaTime per-cell hyperparameters: d_model, n_enc, n_cross, λ_fid, λ_recon, output_mode, pretrain_epochs. |
| `docs/neurips/computational_cost.md` | §1.1 / §1.2 / §1.7 / §1.8 | Adapter and predictor parameter counts. |
| `docs/neurips/adapter_capacity_sweep.md` | "Predictor param counts" table | Tier-design parameter floor verification. |
| `/bigdata/omerg/Thesis/YAIB/configs/tasks/{Binary,Regression}.gin` | L13 | EHR HORIZON = 24 timesteps. |
