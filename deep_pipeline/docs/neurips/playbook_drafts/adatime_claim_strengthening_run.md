# AdaTime Claim-Strengthening Run — HAR/HHAR/WISDM Confirmatory Ablations

> **Purpose.** Close the thinly-evidenced gaps in the AdaTime input-adapter playbook by running 8 strict-toggle / multi-seed jobs on the fast 128-timestep datasets (HAR, HHAR, WISDM) on Athena. Submitted 2026-04-25 ~15:24 IDT, jobs 74012–74019.
>
> **Anchor docs.** `docs/neurips/adatime_input_adapter_playbook.md` §1 A1/A3, §5; `docs/neurips/playbook_drafts/output_mode_multivariable_audit.md` (Verdict A: rule survives 19/19 strict toggles, but `p > 0 → absolute` direction on AdaTime rests on HHAR `v4_base_abs` ALONE, single-seed); `docs/neurips/playbook_drafts/adatime/cross_dataset_synthesis.md` §10 open follow-ups.

---

## Phase 1 — Audit summary (testable AdaTime gaps)

Three claims in the playbook are direct AdaTime-side targets that HAR/HHAR/WISDM at 128-timestep can confirm or refute under ~2 GPU-hours per cell:

1. **A1 — `p = 0` universal** (5/5 datasets prefer no Phase-1 pretrain). HAR and HHAR/WISDM 5-seed reverse-direction reversals support p=0 strongly (z = +3.27σ HHAR, +2.95σ WISDM). **Gap:** no strict toggle exists at HAR with `p > 0` to test the `cap_T` champion's p=0 claim against an actual p=10 alternative at the same architecture.
2. **A3 — output-mode regime split (`p = 0 → residual`; `p > 0 → absolute` provided λ_recon > 0).** The `p > 0 → absolute` direction on AdaTime rests on **HHAR `v4_base_abs` alone, single-seed**, +3.6 MF1 over residual. **Gap:** no second-dataset confirmation; no second-seed confirmation; no strict (p = 0, residual) vs (p = 0, absolute) toggle at the WISDM/HHAR cap_T_p0 champions.
3. **R1' candidate refinement** (`p > 0 ∧ λ_fid ≥ 0.5 → absolute`; `p > 0 ∧ λ_fid < 0.1 → residual`). Audit Phase-3 §3.5 isolates R1 vs R1' as distinguishable only via the HHAR-p10-λ_fid_0.5 (ABS wins) vs WISDM-p10-λ_fid_0.01 (RES wins) cell pair. **Gap:** no WISDM (p = 10, λ_fid = 0.5) cell to test whether cranking λ_fid alone — at fixed dataset — flips the empirical winner from residual to absolute. This is the single highest-leverage R1-vs-R1' arbitrator.

Capacity claims (Tiny ≥ Full) and retrieval-cluster claims (HAR/SSC/MFD positive vs HHAR/WISDM negative) are already strong at n=3–5; not retested here.

## Phase 2 — Eight chosen experiments

Each line is one Athena job. "Strict toggle" = exactly one knob differs from a published reference cell. Predicted MF1 ranges are anchored by adjacent published cells in `docs/adatime_experiments_summary.md` and `adatime_pretrain_ablation.md`.

| # | Name | Dataset | p | mode | λ_fid | Reference / toggle pair | Predicted MF1 (under R1) | Falsification criterion |
|---|------|---------|---|------|-------|-------------------------|--------------------------|-------------------------|
| 1 | `adatime_har_cap_T_p10_res_s0` | HAR | **10** | RES | 0.10 | strict toggle of `cap_T_s0` (p=0, RES) ⇒ flips p only | 0.91–0.93 (slight drop from 0.9438 cap_T) | If MF1 ≥ 0.945, A1 (p=0 universal on HAR cap_T) is refuted |
| 2 | `adatime_har_cap_T_p10_abs_s0` | HAR | **10** | ABS | 0.10 | strict toggle of #1 ⇒ flips mode only | per A3 (p>0→ABS): wins over #1 by some margin; expected ≥ #1 | If #1 (RES) wins by > 5 MF1, A3's `p>0→absolute` direction is refuted on HAR |
| 3 | `adatime_wisdm_cap_T_p0_abs_s0` | WISDM | 0 | **ABS** | 0.01 | strict toggle of `cap_T_p0_s0` (RES) ⇒ flips mode only | per A3 (p=0→RES): RES baseline (~0.74 n=5) >> ABS; expect ABS in 0.55–0.70 | If ABS ≥ RES (cap_T_p0_s0 ≈ 0.78–0.81), A3's `p=0→residual` direction is refuted on WISDM |
| 4 | `adatime_hhar_cap_T_p0_abs_s0` | HHAR | 0 | **ABS** | 0.5 | strict toggle of `cap_T_p0_s0` (RES) ⇒ flips mode only | per A3 (p=0→RES): RES baseline (~0.89 n=5) > ABS; expect ABS in 0.78–0.86 | If ABS ≥ RES (cap_T_p0_s0 ≈ 0.92), A3's `p=0→residual` direction is refuted on HHAR |
| 5 | `adatime_wisdm_v4_lr67_fid05_abs_s0` | WISDM | 10 | **ABS** | **0.5** | basis: `wisdm_v4_lr67_abs.json` with λ_fid 0.01→0.5 | per R1' (p>0 ∧ λ_fid≥0.5 → ABS): ABS wins; expect 0.65–0.72, beats #6 | If #6 (RES) wins, R1' is refuted; rule reduces to R1 + λ_fid as magnitude only |
| 6 | `adatime_wisdm_v4_lr67_fid05_res_s0` | WISDM | 10 | **RES** | **0.5** | strict toggle of #5 ⇒ flips mode only | per R1 (p>0→ABS): RES loses; under R1' it loses by > λ_fid=0.01's residual margin (16.9 MF1) | If RES wins by ≥ 5 MF1, R1' is the data-best rule; if ABS wins, both R1 & R1' are confirmed |
| 7 | `adatime_hhar_v4_base_abs_s1` | HHAR | 10 | ABS | 0.5 | basis: `v4_base_abs.json`, seed 42 → seed **1** | per A3 (p>0→ABS): ABS wins by a margin near +3.6 MF1 (HHAR v4_base_abs s42) | If RES (#8) wins on s1, the published HHAR p=10 ABS-wins is single-seed lucky; A3 falsified on HHAR |
| 8 | `adatime_hhar_v4_base_res_s1` | HHAR | 10 | RES | 0.5 | basis: `v4_base.json`, seed 42 → seed **1** | per A3 (p>0→ABS): RES loses to #7 | (See #7) |

**Coverage map.** A1: HAR cap_T (#1 vs #2's residual companion + the existing cap_T_s0). A3 `p=0→RES`: WISDM cap_T_p0 (#3) + HHAR cap_T_p0 (#4). A3 `p>0→ABS` second-seed: HHAR (#7, #8). R1 vs R1': WISDM v4_lr67 at λ_fid=0.5 (#5, #6).

## Phase 3 — Configs

All 8 configs created in `experiments/.athena_configs/`. Each is a clean strict toggle vs an existing reference cell — only the targeted knob(s) differ; all other fields preserved verbatim.

| # | Path | Validation |
|---|------|------------|
| 1 | `experiments/.athena_configs/adatime_har_cap_T_p10_res_s0.json` | OK (toggled `pretrain_epochs` 0→10 vs `adatime_har_cap_T_s0.json`) |
| 2 | `experiments/.athena_configs/adatime_har_cap_T_p10_abs_s0.json` | OK (toggled `pretrain_epochs` 0→10 + `output_mode` RES→ABS vs `adatime_har_cap_T_s0.json`; strict toggle of #1) |
| 3 | `experiments/.athena_configs/adatime_wisdm_cap_T_p0_abs_s0.json` | OK (toggled `output_mode` RES→ABS vs `adatime_wisdm_cap_T_p0_s0.json`) |
| 4 | `experiments/.athena_configs/adatime_hhar_cap_T_p0_abs_s0.json` | OK (toggled `output_mode` RES→ABS vs `adatime_hhar_cap_T_p0_s0.json`) |
| 5 | `experiments/.athena_configs/adatime_wisdm_v4_lr67_fid05_abs_s0.json` | OK (toggled `lambda_fidelity` 0.01→0.5 vs `adatime_wisdm_v4_lr67_abs.json`; seed 42→0) |
| 6 | `experiments/.athena_configs/adatime_wisdm_v4_lr67_fid05_res_s0.json` | OK (strict toggle of #5: `output_mode` ABS→RES) |
| 7 | `experiments/.athena_configs/adatime_hhar_v4_base_abs_s1.json` | OK (toggled `seed` 42→1 vs `adatime_hhar_v4_base_abs.json`) |
| 8 | `experiments/.athena_configs/adatime_hhar_v4_base_res_s1.json` | OK (strict toggle of #7: `output_mode` ABS→RES) |

Validation script confirmed all 11 schema fields per config (output_mode, pretrain_epochs, lambda_fidelity, seed, dataset, d_model, n_enc_layers, n_cross_layers, k_neighbors, lr, epochs).

## Phase 4 — Athena submissions

`athena_submit.py --sync` succeeded 2026-04-25 ~15:23 IDT (rsync done; YAIB synced; gin paths fixed; package reinstall completed). All 8 jobs submitted at `--qos 24h_1g` (auto-selected, partitions `l40s-shared,a100-public`).

| # | Name | Athena Job ID | Submitted (IDT) |
|---|------|---|-----------------|
| 1 | adatime_har_cap_T_p10_res_s0 | **74012** | 15:23:36 |
| 2 | adatime_har_cap_T_p10_abs_s0 | **74013** | 15:23:46 |
| 3 | adatime_wisdm_cap_T_p0_abs_s0 | **74014** | 15:23:51 |
| 4 | adatime_hhar_cap_T_p0_abs_s0 | **74015** | 15:23:59 |
| 5 | adatime_wisdm_v4_lr67_fid05_abs_s0 | **74016** | 15:24:04 |
| 6 | adatime_wisdm_v4_lr67_fid05_res_s0 | **74017** | 15:24:14 |
| 7 | adatime_hhar_v4_base_abs_s1 | **74018** | 15:24:19 |
| 8 | adatime_hhar_v4_base_res_s1 | **74019** | 15:24:28 |

Athena QoS limits 2 concurrent jobs per user; remaining 6 wait in queue. Per-job wall-time estimate: ~1–2 hours each (HAR/HHAR/WISDM at 128 timesteps × 30–40 epochs, comparable to existing `cap_T_s*` jobs). **Expected total completion: ~6–10 hours of pipelined runtime, calendar-time ~12–24 hours given QoS gating.**

## Phase 5 — Decision matrix once results land

Each cell pair settles a specific claim with a specific falsification threshold:

| Claim | Confirms if … | Refutes if … |
|-------|---------------|--------------|
| **A1 (HAR cap_T p=0 wins)** | #1 (p=10 RES) MF1 < `cap_T_s0` (≈ 0.94) by ≥ 1 MF1 | #1 ≥ `cap_T_s0`; A1's HAR evidence weakens further |
| **A3 (HAR p>0→ABS)** | #2 (p=10 ABS) > #1 (p=10 RES) | #1 wins by > 5 MF1 → A3 refuted on HAR; rule must split by dataset |
| **A3 (WISDM p=0→RES)** | #3 (p=0 ABS) ≪ `cap_T_p0_s0` (≈ 0.78–0.81) | #3 ≥ `cap_T_p0_s0` → A3's p=0→RES direction refuted on WISDM |
| **A3 (HHAR p=0→RES)** | #4 (p=0 ABS) ≪ `cap_T_p0_s0` (≈ 0.92) | #4 ≥ `cap_T_p0_s0` → A3's p=0→RES direction refuted on HHAR |
| **R1' (λ_fid threshold)** | #5 (ABS) > #6 (RES) at WISDM-p10-λ_fid=0.5 → R1' confirmed; cranking λ_fid alone flips the winner | #6 (RES) wins → R1' is refuted; λ_fid drives only magnitude (Pearson −0.93), not direction |
| **A3 (HHAR p>0→ABS, n>1)** | #7 (ABS s1) > #8 (RES s1) → 2-seed agreement on the pivotal HHAR cell | #8 (RES) wins on s1 → published HHAR `v4_base_abs` win is seed-lucky; A3's `p>0→ABS` evidence collapses to zero on AdaTime |

The most consequential single result is the pair (#7, #8): if it inverts vs published s42, the entire `p > 0 → absolute` direction on AdaTime loses its sole multi-seed support and the rule must be either restricted to EHR or re-derived. If it confirms (ABS wins on s1 too), A3 is solidified to n=2 on HHAR — adequate for a paper claim with a "single-seed flag" softened.

## Phase 6 — Out of scope (and why)

- **n=3 multi-seed for every cell.** Each strict-toggle pair already provides decisive Bayesian evidence at single-seed when the predicted gap is ≥ 5 MF1 and the observed gap matches sign. n=3 is reserved for the pivotal HHAR `v4_base` cell (jobs 7 & 8 add one seed; existing s42 is the second). Adding n=3 to all 6 pairs would double the job count and offer marginal evidence beyond the strict-toggle direction.
- **SSC/MFD ablations.** User explicitly scoped HAR/HHAR/WISDM ("fast 128-timestep"). SSC at 1×3000 and MFD at 1×5120 take 5–10× longer per epoch and do not address the highest-leverage gap (the HHAR-vs-WISDM-at-p=10 R1-vs-R1' arbitration).
- **Cross-tier (Tiny vs Full) toggles at p=10.** Capacity claim (A2) is already strong at n=3 across 5 datasets; not retested.
- **Retrieval-cluster (C0 vs C1) on these new cells.** Retrieval evidence is already strong at n=3 across 5 datasets; the present audit does not need a retrieval factor.
- **Playbook updates.** Per mandate, no playbook prose is changed at this stage. The playbooks will be updated once results land and the decision matrix above resolves each cell.

---

## Tracking

Status: `python scripts/athena_submit.py --status` (live SLURM queue).
Per-job logs: `experiments/logs/<name>.out` on Athena once started.
Results collection: `python scripts/athena_submit.py --collect <name>` once SLURM reports completion.

When all 8 land, follow up with: (a) update audit Phase-3 with new cells; (b) update playbook §1 A3 caveat language (remove "single-seed" qualifier on HHAR, or escalate if refuted); (c) update `docs/neurips/playbook_drafts/adatime/cross_dataset_synthesis.md` §10 open-followups checkbox.

---

## Phase 4 — Outcomes vs predictions (Apr 26 update; actual Athena jobs 74033, 74036–74042)

The 8 jobs all completed cleanly. Result JSONs pulled to `experiments/results/adatime_cnn_*` and parsed via the standard scenario-mean macro-F1 protocol (each scenario stores source-only F1 twice + translator F1). The outcomes refute the previously-documented `pretrain_epochs → output_mode` rule on AdaTime.

### Verified results

| # | Cell | RES MF1×100 | ABS MF1×100 | Winner | Margin (MF1) | Predicted (R1) | Outcome |
|---|---|---:|---:|---|---:|---|---|
| 1/2 | HAR `cap_T_p10` (74033/74036) | **92.31** | 67.45 | RES | **+24.86** | ABS wins (R1: `p > 0 → ABS`) | **R1 refuted on HAR** — RES wins by a huge margin at `p = 10` |
| 3 | WISDM `cap_T_p0` ABS (74037) vs existing RES | ≈80.38 | 51.61 | RES | ~+28.8 | RES wins (R1: `p = 0 → RES`) | **A3 confirmed on WISDM** at `p = 0` |
| 4 | HHAR `cap_T_p0` ABS (74038) vs existing RES | ≈91.73 | 79.16 | RES | ~+12.6 | RES wins (R1: `p = 0 → RES`) | **A3 confirmed on HHAR** at `p = 0` |
| 5/6 | WISDM `v4_lr67_fid05` p=10 (74039/74040) | **71.63** | 58.22 | RES | **+13.41** | ABS wins (R1' arbitrator: at `p = 10, λ_fid = 0.5`, R1' predicts ABS) | **R1 and R1' both refuted on WISDM** — RES wins decisively at `p = 10` even with `λ_fid` cranked from 0.01 to 0.5 |
| 7/8 | HHAR `v4_base` p=10, s1 (74041/74042) | **89.29** | 88.42 | RES | +0.87 | ABS wins (per published s0 +3.6 MF1) | **Direction flips at s1**; 2-seed mean is ABS +1.36 MF1 within σ ≈ 1.4 — i.e. tie, not sign flip |

### Decision-matrix resolution

| Claim | Status |
|---|---|
| **A1 (HAR cap_T p=0 wins)** | Mostly confirmed: HAR `cap_T_p10` RES drops from cap_T_p0_s0's ≈ 0.94 to 0.9231 at `p = 10`, supporting `p = 0` preference on HAR (modest gap, but consistent direction). The big finding here was the 2-line ABS catastrophe at `p = 10`. |
| **A3 (HAR p>0→ABS)** | **REFUTED.** HAR `cap_T_p10` RES (92.31) beats ABS (67.45) by +24.86 MF1. R1's `p > 0 → ABS` direction does not hold on HAR. |
| **A3 (WISDM p=0→RES)** | Confirmed: WISDM `cap_T_p0` ABS lands at 51.61 vs RES baseline ≈ 80.38 — ABS catastrophe at `p = 0`. |
| **A3 (HHAR p=0→RES)** | Confirmed: HHAR `cap_T_p0` ABS lands at 79.16 vs RES baseline ≈ 91.73 — ABS underperforms at `p = 0`. |
| **R1' (λ_fid threshold arbitrator at WISDM-p10-λ_fid_0.5)** | **REFUTED.** WISDM `v4_lr67_fid05` at `p = 10, λ_fid = 0.5`: RES wins by +13.41 MF1. R1' predicted ABS would win once λ_fid was cranked above the 0.1 threshold. The data shows even λ_fid = 0.5 plus pretrain `p = 10` is insufficient to flip WISDM to absolute. |
| **A3 (HHAR p>0→ABS, n>1)** | **REFUTED at single-seed; tied at 2-seed.** s1 reverses to RES +0.87 MF1; 2-seed mean ABS 88.81 ± 0.39 vs RES 87.45 ± 1.85 → ABS by +1.36 MF1 within seed σ. The single AdaTime cell that previously supported `p > 0 → absolute` is now a within-σ tie. |

### Implication: new sharper rule

**The previously-documented `pretrain_epochs → output_mode` rule on AdaTime is refuted.** New rule:

> **AdaTime**: `output_mode = "residual"` universally — wins or ties at every measured `pretrain_epochs × λ_fidelity` cell.
>
> **EHR**: `output_mode = "absolute"` universally — frozen LSTM + tabular ICU feature regime (5/5 tasks at n=3, C8 strict toggle).
>
> The cross-benchmark split is keyed on the **predictor + feature regime**, not on `pretrain_epochs`. λ_fidelity is the *magnitude* axis on AdaTime (Pearson r ≈ −0.93 across `p = 0` datasets), not a direction axis.

The deprecated `p`-keyed rule is preserved as evidence-history in `output_mode_multivariable_audit.md` Phase 6.

### Honest seed-count flag

6 of the 7 new strict-toggle pairs are n=1 single-seed. The HHAR `v4_base` p=10 pair is now n=2. The claim "residual wins or ties at every measured AdaTime cell" is calibrated at this seed count; the >3σ harder claim would require ≈ 6 additional second-seed jobs.
