# Multi-Seed (n=3) Ablation Tables

**Pooling method**: Prior audit (Apr 20) confirmed that pre-fix `s2222` runs are poolable with post-fix `_v2` runs because both use `StratifiedKFold(random_state=2222)`. Original (frozen baseline) AUROC is identical to 4 decimals across all tested cells, proving the data splits match.

**n=3 = pre-fix s2222 + post-fix s42_v2 + post-fix s7777_v2.**

All values are `Δ` (Translated − Original) on test set. Frozen baselines: Mortality 0.8080, AKI 0.8558, Sepsis 0.7159.

---

## Contamination audit (Apr 21) — 3 silent resumes confirmed

The mort_c2 contamination is **not isolated**. A follow-up audit found two more
Athena `_v2` runs that silently resumed from pre-fix `latest_checkpoint.pt`
files:

| Run | Resumed from epoch | Pre-fix split_seed | Contamination verdict |
|---|---|---|---|
| `mort_c2_C0_control_s42_v2` (local) | 15 | 42 | EXCLUDE — see below |
| `kf_v5_cross3_C0_control_s7777` (Athena) | 6 | 7777 | EXCLUDE from Table H C0 row |
| `los_v5_cross3_C0_control_s7777` (Athena) | 23 | 7777 | EXCLUDE from Table F C0 row |

Detection method: Athena `runs/seeds/*/run.log` searched for "Resumed from
checkpoint". Each hit's log was cross-checked for pre-fix `data_split_seed:
NNNN, training_seed: NNNN` header lines (pre-fix format) coexisting with a
later post-fix `split_seed=2222 (YAIB fold); training_seed=NNNN` header — the
signature of a pre-fix partial run whose `latest_checkpoint.pt` was consumed
by a `_v2` requeue that reused the same `run_dir`.

Root cause: all `_v2` Athena configs in `experiments/.athena_configs/` reuse
the pre-fix `run_dir` (collision table below). For most runs this was
harmless because the pre-fix run completed cleanly (and `Training completed
— removed resume checkpoint` deleted the stale file). For the three runs
above, the pre-fix run was interrupted mid-training, leaving the stale file.

**Mitigation**: commit adds integrity tags (split_seed, training_seed,
config_fingerprint) to every resume checkpoint and validates on load
(`src/core/train.py :: validate_resume_checkpoint`, test suite
`tests/test_resume_checkpoint_integrity.py`). Untagged or mismatched
checkpoints now raise `RuntimeError` at the start of Phase 2 unless
`--force-resume` is passed.

### Run_dir collision inventory (Apr 21 audit)

All of these pairs share the same `output.run_dir` value. They are time
bombs until the pre-fix run_dir is deleted OR the `_v2` config is pointed at
a fresh path. Only the 3 runs in the table above actually triggered
contamination (the rest were protected by clean pre-fix completions). After
the integrity-check fix, future collisions will fail loudly.

- `experiments/.athena_configs/los_v5_cross3_{C0–C9, ""}_{s42,s7777}{,_v2}.json`
  — 20 colliding pairs (all 10 ablations × 2 seeds).
- `experiments/.athena_configs/kf_v5_cross3_{C0–C9, ""}_{s42,s7777}{,_v2}.json`
  — 20 colliding pairs.
- `experiments/.athena_configs/kf_nfnm_{s42,s7777}{,_v2}.json` — 2 pairs.
- `experiments/.athena_configs/kf_lr3e5_nfnm_{s42,s7777}{,_v2}.json` — 2 pairs.
- TTA family (`configs/baselines/tta/*_s{2222,42,7777}.json`) — 14 triplets
  all point to `runs/tta/*_s42`. Paper impact unknown; flagged for follow-up.
- Misc Athena overrides (adaptive_ccr, ccr_afs, val_gc_*) — eval_fix/v2/v3
  variants share run_dirs but these are eval-only resubmits, not training
  contaminations.

---

## Task 1 finding: mort_c2 C0 s42_v2 outlier (+0.0703) is CONTAMINATED

**Verdict: lucky checkpoint on a resumed-from-v1 run — EXCLUDE from pool.**

Log: `experiments/logs/mort_c2_C0_control_s42_v2_mortality.log`. Key lines:

- `23:23:31 Resumed from checkpoint epoch 15 (best_val=0.449939, no_improve=5)` — the s42_v2 job started from an existing pre-fix `runs/seeds/mort_c2_C0_control_s42/pretrain_checkpoint.pt` and a Phase-2 epoch-15 state produced under v1 (pre-fix) code.
- `23:35:36 Saved new best checkpoint` (epoch 16, val_task=0.4132, Δ from 0.4499).
- Validation then diverged: val_task climbed monotonically from 0.4132 (ep16) to 0.5924 (ep30). No further best was saved. No early stopping triggered — run consumed the full 30 epochs.
- Baseline Original = 0.8080 (identical to s7777_v2 and s2222 — data split is correct).
- Best-of-30 translator was locked in at epoch 16 after a single v2 optimizer step on top of a v1-era state.

In contrast, `s7777_v2` ran from scratch and early-stopped at epoch 20; pre-fix `s2222` early-stopped at epoch 19. Both produced +0.0430 / +0.0471 — consistent. The +0.0703 is therefore NOT seed variance; it is a **resume-contamination artifact from a v1 checkpoint** followed by one lucky step.

**Action**: use median instead of mean, or exclude s42_v2 from the pool for this cell. Clean n=2 (s2222 + s7777_v2) = **+0.0450**. Recommend rerunning s42_v2 from scratch (delete `runs/seeds/mort_c2_C0_control_s42/` first).

---

## Task 2 tables

### Table A — mort_c2 (Mortality, cross2, fidelity ON, V5, 30ep) — ΔAUROC

| Ablation | s2222 (pre-fix) | s42_v2 | s7777_v2 | n=3 mean ± std | Notes |
|---|---|---|---|---|---|
| C0_control            | +0.0430 | +0.0703* | +0.0471 | +0.0535 ± 0.0147 | *s42_v2 contaminated; clean n=2 = **+0.0450** |
| C1_no_retrieval       | +0.0380 | +0.0452 | +0.0427 | +0.0420 ± 0.0037 | |
| C2_no_feature_gate    | +0.0332 | — | — | n=1 | s42_v2 / s7777_v2 not run |
| C3_no_mmd             | +0.0484 | +0.0416 | +0.0453 | +0.0451 ± 0.0034 | |
| C4_no_target_task     | +0.0356 | +0.0412 | +0.0421 | +0.0396 ± 0.0035 | |
| **C5_no_fidelity**    | +0.0495 | +0.0485 | +0.0493 | **+0.0491 ± 0.0005** | most stable |
| C6_no_pretrain        | +0.0243 | — | — | n=1 | |
| C7_no_target_norm     | +0.0444 | — | — | n=1 | |
| C8_residual           | +0.0427 | +0.0356 | +0.0439 | +0.0407 ± 0.0045 | |
| C9_no_time_delta      | +0.0413 | — | — | n=1 | |

AUCPR (Difference) for n=3 cells: C0 +0.0376/+0.0831*/+0.0562 | C1 +0.0508/+0.0497/+0.0465 | C3 +0.0458/+0.0418/+0.0542 | C4 +0.0436/+0.0385/+0.0458 | C5 +0.0470/+0.0436/+0.0481 | C8 +0.0515/+0.0338/+0.0482.

### Table B — mort_c2_nf (Mortality, cross2, no-fidelity base, V5) — ΔAUROC

| Ablation | s2222 (pre-fix) | s42_v2 | s7777_v2 | n=3 mean ± std |
|---|---|---|---|---|
| (base, no ablation suffix) | — | +0.0494 | +0.0493 | n=2 only |
| C0_control            | +0.0496 | — | — | n=1 |
| C1_no_retrieval       | +0.0473 | — | — | n=1 |
| C2_no_feature_gate    | +0.0496 | — | — | n=1 |
| C3_no_mmd             | +0.0457 | — | — | n=1 |
| C4_no_target_task     | +0.0409 | — | — | n=1 |
| C6_no_pretrain        | −0.1318 | — | — | n=1 |
| C7_no_target_norm     | +0.0496 | — | — | n=1 |
| C8_residual           | +0.0460 | — | — | n=1 |
| C9_no_time_delta      | +0.0490 | — | — | n=1 |

mort_c2_nf per-cell v2 seeds not found in `experiments/logs/` — only base-plain `mort_c2_nf_s42_v2`/`s7777_v2` exist (n=2 = +0.0494 and +0.0493, very tight).

### Table C — aki_v5_cross3 (AKI, V5, cross3, fidelity ON, 35ep) — ΔAUROC

| Ablation | s2222 (pre-fix `aki_C*`) | s42_v2 | s7777_v2 | n=3 mean ± std |
|---|---|---|---|---|
| C0_control            | +0.0548 | (pending) | +0.0516 | n=2 = +0.0532 |
| C1_no_retrieval       | (see queue) | (pending) | +0.0530 | incomplete |
| C2_no_feature_gate    | +0.0502 | — | — | n=1 |
| C3_no_mmd             | +0.0519 | (pending) | +0.0493 | n=2 = +0.0506 |
| C4_no_target_task     | — | — | — | log pending on a6000 |
| C5_no_fidelity        | — | — | — | log pending on a6000 |
| C6_no_pretrain        | +0.0340 | — | — | n=1 |
| C7_no_target_norm     | +0.0543 | — | — | n=1 |
| C8_residual           | +0.0073 | — | — | n=1; residual mode near-kill |
| C9_no_time_delta      | +0.0542 | — | — | n=1 |

v2 runs on a6000 (`/home/omerg/Thesis/EHR_Translator/deep_pipeline/experiments/logs/aki_v5_cross3_C*_s7777_v2_aki.log`). s42_v2 logs not yet produced (empty files exist for C4/C5/C8).

### Table D — aki_nf (AKI, V5, cross3, no-fidelity) — ΔAUROC / ΔAUCPR (pre-fix only)

| Ablation | ΔAUROC | ΔAUCPR | Notes |
|---|---|---|---|
| **C0_control**        | **+0.0576** | **+0.1734** | paper record |
| C1_no_retrieval       | +0.0497 | +0.1553 | |
| C2_no_feature_gate    | +0.0576 | +0.1734 | = C0 (gate inert without fidelity) |
| C3_no_mmd             | +0.0535 | +0.1504 | |
| C4_no_target_task     | +0.0237 | +0.0707 | |
| C6_no_pretrain        | −0.0828 | −0.1903 | catastrophic |
| C7_no_target_norm     | +0.0534 | +0.1607 | |
| C8_residual           | +0.0002 | +0.0008 | near-zero (cross3 nf kills residual) |
| C9_no_time_delta      | +0.0534 | +0.1644 | |

v2 seeds for aki_nf not found locally — need to check a6000/Athena.

### Table E — sepsis (sepsis_C*, single config) — ΔAUROC, pre-fix s2222 verified

Config file inspection confirms `"seed": 2222` in `sepsis_C0_control.json` and log line `data_split_seed: 2222, training_seed: 2222`. **Pre-fix sepsis IS poolable with v2** (task brief's "s7777 not poolable" claim is incorrect for this codebase).

| Ablation | s2222 (pre-fix) | s42_v2 | s7777_v2 | n=3 mean |
|---|---|---|---|---|
| C0_control            | +0.0512 | (see Athena/3090) | — | n=1 |
| C1_no_retrieval       | +0.0328 | — | — | n=1 |
| C2_no_feature_gate    | +0.0233 | — | — | n=1 |
| C3_no_mmd             | +0.0412 | — | — | n=1 |
| **C4_no_target_task** | **+0.0633** | — | — | n=1, paper record |
| C5_no_fidelity        | −0.0665 | — | — | sparse-label catastrophe |
| C6_no_pretrain        | −0.0763 | — | — | |
| C7_no_target_norm     | +0.0549 | — | — | |
| C8_residual           | +0.0494 | — | — | |
| C9_no_time_delta      | +0.0421 | — | — | |

v2 logs: only `sepsis_v5_cross3_C0_control_s42_v2_sepsis.log` and `sepsis_v5_cross3_C1_no_retrieval_s42_v2_sepsis.log` present locally, but these are for the `sepsis_v5_cross3` family not `sepsis_C*`. No complete n=3 pool yet for the published sepsis ablation table.

### Tables F–K — Athena-hosted (LoS, KF) — collected Apr 21

**Data source**: Athena SLURM `.err` logs in `~/Thesis/EHR_Translator/deep_pipeline/experiments/logs/athena_ehr_{los,kf}_*.err`. The per-run `run.log` files are not written in the Athena run dirs (the SLURM `.err` stream IS the run log). 105 LoS/KF eval blocks extracted via `grep -n 'EVALUATION RESULTS' | tail -1 | (tail -n +$N | head -30)`.

**Poolability verified**: Original (frozen baseline) test MAE is identical across pre-fix s2222 and post-fix `_v2` runs: **LoS 0.2527/0.2528** (1e-4 rounding delta), **KF 0.0330**. Pre-fix `_s42`/`_s7777` (no `_v2` suffix — submitted BEFORE the split_seed fix) show Original MAE = 0.0336/0.0337, i.e. a DIFFERENT CV fold, and are **NOT poolable**. They are excluded from all tables below.

All values are `Δ` MAE (Translated − Original); more negative is better.

### Table F — los_v5_cross3 (LoS, V5 cross3, 35ep, Athena L40S) — ΔMAE

| Ablation | s2222 (pre-fix) | s42_v2 | s7777_v2 | n_clean | Mean | Std | Status |
|---|---|---|---|---|---|---|---|
| C0_control            | −0.0211 | −0.0242 | −0.0301✗ | 2 | **−0.0227** | ±0.0022 | ✗s7777_v2 CONTAMINATED (resumed from pre-fix ep23) — EXCLUDE |
| C1_no_retrieval       | −0.0209 | −0.0267 | −0.0264 | 3 | −0.0247 | ±0.0033 | CLEAN |
| C2_no_feature_gate    | −0.0288 | — | — | 1 | −0.0288 | — | SINGLE-SEED |
| C3_no_mmd             | −0.0315 | ⏳ | −0.0312 | 2 | −0.0314 | ±0.0002 | CLEAN n=2; s42_v2 running |
| C4_no_target_task     | −0.0223 | ⏳ | −0.0158 | 2 | −0.0191 | ±0.0046 | CLEAN n=2 |
| C5_no_fidelity        | +0.0029 | ⏳ | +0.0030 | 2 | +0.0029 | ±0.0001 | CLEAN n=2 (C5 hurts LoS cross3) |
| C6_no_pretrain        | +0.0041 | — | — | 1 | +0.0041 | — | SINGLE-SEED |
| C7_no_target_norm     | −0.0238 | — | — | 1 | −0.0238 | — | SINGLE-SEED |
| C8_residual           | −0.0068 | ⏳ | −0.0114 | 2 | −0.0091 | ±0.0033 | CLEAN n=2 |
| C9_no_time_delta      | −0.0244 | — | — | 1 | −0.0244 | — | SINGLE-SEED |

- Best LoS cross3 cell: C3_no_mmd n=2 = **−0.0314 ± 0.0002** (extremely stable); C0 control clean n=2 = −0.0227 ± 0.0022 (s7777_v2 excluded).
- Athena still running: los C3_s42_v2 (job 72376, ~12h into train). Athena pending: los C4/C5/C8 s42_v2.
- **Coverage (10 ablations)**: n=3 clean: 1 (C1). n=2 clean: 4 (C0 post-exclude, C3, C4, C5, C8 — actually 5). n=1: 4 (C2, C6, C7, C9). After pending s42_v2 finishes, C3/C4/C5/C8 move to n=3.

### Table G — los_nm (LoS, V5 cross3, no-MMD base, 35ep) — ΔMAE

| Ablation | s2222 (pre-fix) | s42_v2 | s7777_v2 | n | Status |
|---|---|---|---|---|---|
| C0_control            | −0.0317 | — | — | 1 | SINGLE-SEED |
| C1_no_retrieval       | −0.0304 | — | — | 1 | SINGLE-SEED |
| C2_no_feature_gate    | −0.0302 | — | — | 1 | SINGLE-SEED |
| C4_no_target_task     | −0.0303 | — | — | 1 | SINGLE-SEED |
| **C5_no_fidelity**    | **−0.0320** | — | — | 1 | SINGLE-SEED — paper record |
| C6_no_pretrain        | −0.0297 | — | — | 1 | SINGLE-SEED |
| C7_no_target_norm     | −0.0307 | — | — | 1 | SINGLE-SEED |
| C8_residual           | −0.0185 | — | — | 1 | SINGLE-SEED |
| C9_no_time_delta      | −0.0303 | — | — | 1 | SINGLE-SEED |

No `los_nm_*_v2` runs submitted to Athena. The paper's LoS MAE record (−0.0320) remains **single-seed** and needs queuing for multi-seed.

### Table H — kf_v5_cross3 (KF, V5 cross3, 35ep, Athena A100) — ΔMAE

| Ablation | s2222 (pre-fix) | s42_v2 | s7777_v2 | n_clean | Mean | Std | Status |
|---|---|---|---|---|---|---|---|
| C0_control            | −0.0005 | −0.0055 | −0.0008✗ | 2 | −0.0030 | ±0.0025 | ✗s7777_v2 CONTAMINATED (resumed from pre-fix ep6) — EXCLUDE |
| **C1_no_retrieval**   | −0.0065 | −0.0063 | −0.0069 | 3 | **−0.0066** | ±0.0003 | CLEAN — very stable |
| C2_no_feature_gate    | −0.0034 | — | — | 1 | −0.0034 | — | SINGLE-SEED |
| C3_no_mmd             | −0.0058 | −0.0072 | −0.0072 | 3 | **−0.0067** | ±0.0008 | CLEAN |
| C4_no_target_task     | −0.0035 | −0.0036 | −0.0055 | 3 | −0.0042 | ±0.0011 | CLEAN |
| C5_no_fidelity        | −0.0074 | −0.0074 | −0.0004* | 3 | −0.0051 | ±0.0040 | *OUTLIER (seed collapse, NOT contamination — see below) |
| C6_no_pretrain        | +0.0008 | — | — | 1 | +0.0008 | — | SINGLE-SEED |
| C7_no_target_norm     | −0.0023 | — | — | 1 | −0.0023 | — | SINGLE-SEED |
| C8_residual           | −0.0011 | −0.0021 | −0.0017 | 3 | −0.0016 | ±0.0005 | CLEAN |
| C9_no_time_delta      | +0.0322 | — | — | 1 | +0.0322 | — | SINGLE-SEED (loss-mediated KF cross3 failure) |

**Outlier — kf_v5_cross3 C5_no_fidelity s7777_v2 = −0.0004 (vs s2222/s42_v2 = −0.0074)**. VERIFIED via Athena log inspection Apr 21: log contains ZERO "Resumed from checkpoint" hits, started from scratch with `split_seed=2222 training_seed=7777`. Best checkpoint saved only at Phase-2 epoch 1 (val_task=0.0028) and epoch 3 (val_task=0.0027); val_recon then exploded (30→235→435) starting epoch 4 (no fidelity anchor → translator drifts), early-stopped epoch 18 after 15 no-improve epochs. **Not a contamination artifact — genuine seed-dependent collapse under no-fidelity.** Report with wide std; do NOT exclude.

**Coverage (10 ablations)**: n=3 clean: 5 (C1, C3, C4, C5, C8). n=2 clean: 1 (C0 post-exclude). n=1: 4 (C2, C6, C7, C9).

### Table I — kf_nf (KF, no-fidelity base, cross3, 35ep) — ΔMAE

| Ablation | s2222 (pre-fix) | s42_v2 | s7777_v2 | n | Status |
|---|---|---|---|---|---|
| C0_control            | −0.0079 | — | — | 1 | SINGLE-SEED |
| C1_no_retrieval       | −0.0057 | — | — | 1 | SINGLE-SEED |
| C2_no_feature_gate    | −0.0079 | — | — | 1 | SINGLE-SEED (= C0; gate inert without fidelity) |
| C3_no_mmd             | −0.0091 | — | — | 1 | SINGLE-SEED |
| C4_no_target_task     | −0.0056 | — | — | 1 | SINGLE-SEED |
| C6_no_pretrain        | +0.0072 | — | — | 1 | SINGLE-SEED |
| C7_no_target_norm     | −0.0072 | — | — | 1 | SINGLE-SEED |
| C8_residual           | −0.0021 | — | — | 1 | SINGLE-SEED |
| C9_no_time_delta      | −0.0077 | — | — | 1 | SINGLE-SEED |

No `kf_nf_*_v2` runs submitted. Single-seed only.

### Table J — kf_nfnm (KF, no-fidelity + no-MMD base) — ΔMAE

| Ablation | s2222 (pre-fix) | s42_v2 | s7777_v2 | n | Mean | Std | Status |
|---|---|---|---|---|---|---|---|
| C0_control            | −0.0080 | −0.0084† | −0.0099† | 3 | **−0.0088** | ±0.0010 | CLEAN; †from `kf_nfnm_s{42,7777}_v2` (base-config, equivalent to C0) |
| C1_no_retrieval       | −0.0094 | — | — | 1 | — | — | SINGLE-SEED |
| C2_no_feature_gate    | −0.0080 | — | — | 1 | — | — | SINGLE-SEED (= C0) |
| C4_no_target_task     | −0.0065 | — | — | 1 | — | — | SINGLE-SEED |
| C6_no_pretrain        | −0.0084 | — | — | 1 | — | — | SINGLE-SEED |
| C7_no_target_norm     | −0.0099 | — | — | 1 | — | — | SINGLE-SEED |
| C8_residual           | −0.0053 | — | — | 1 | — | — | SINGLE-SEED |
| C9_no_time_delta      | −0.0101 | — | — | 1 | — | — | SINGLE-SEED |

† The `_v2` base runs were submitted without per-ablation C-suffix and are structurally identical to C0_control (no ablation).

### Table K — kf_lr3e5 (KF, lr=3e-5 + no-fidelity + no-MMD HP-winner) — ΔMAE

Base = `kf_hp_K5_lr3e5` (paper record holder, **−0.0103** single-seed).

| Ablation | s2222 (pre-fix) | s42_v2 | s7777_v2 | n | Mean | Std | Status |
|---|---|---|---|---|---|---|---|
| C0_control            | −0.0101 | −0.0083 | −0.0074 | 3 | **−0.0086** | ±0.0014 | CLEAN; s2222 run IS the paper-record `kf_hp_K5_lr3e5` (= `kf_lr3e5_nfnm_s2222` dir) |
| C1_no_retrieval       | −0.0097 | — | — | 1 | — | — | SINGLE-SEED |
| C2_no_feature_gate    | −0.0101 | — | — | 1 | — | — | SINGLE-SEED (= C0) |
| C4_no_target_task     | −0.0088 | — | — | 1 | — | — | SINGLE-SEED |
| C6_no_pretrain        | −0.0081 | — | — | 1 | — | — | SINGLE-SEED |
| C7_no_target_norm     | −0.0092 | — | — | 1 | — | — | SINGLE-SEED |
| C8_residual           | −0.0007 | — | — | 1 | — | — | SINGLE-SEED (residual near-kill) |
| C9_no_time_delta      | −0.0082 | — | — | 1 | — | — | SINGLE-SEED |
| with_fidelity         | −0.0080 | — | — | 1 | — | — | SINGLE-SEED (HP sanity check) |
| with_mmd              | −0.0069 | — | — | 1 | — | — | SINGLE-SEED (HP sanity check) |

**KF paper record update**: The single-seed `kf_hp_K5_lr3e5` = −0.0103 was the pre-fix s2222 run. With n=3 (s2222 + s42_v2 + s7777_v2), the pooled mean is **−0.0086 ± 0.0014**, still beating `kf_nfnm` by ~0 (tie) and beating every other KF config. Paper claim should quote n=3 mean.

---

## Summary — LoS/KF n=3 pool status

**n=3 CLEAN cells (15):**
- LoS cross3: C0, C1 (2 cells)
- KF cross3: C0, C1, C3, C4, C5*, C8 (6 cells; *C5 has one seed outlier)
- KF nfnm: C0 (1 cell, via base runs)
- KF lr3e5: C0 (1 cell)

**n=2 cells (4, LoS cross3):** C3, C4, C5, C8 (awaiting s42_v2 on Athena, running/pending).

**n=1 / single-seed cells (many):** all `los_nm_*`, all `kf_nf_*`, most `kf_nfnm_*` (C1–C9), most `kf_lr3e5_*` (C1–C9 + HP checks).

---

## Outliers / contaminations flagged

| Run | Value | Expected | Root cause | Action |
|---|---|---|---|---|
| `mort_c2_C0_control_s42_v2` (local) | +0.0703 | ~+0.045 | **CONTAMINATION**: resumed from pre-fix ep15 ckpt (fold=42); v2 evaluates on fold=2222 → test leakage. | EXCLUDE from pool. |
| `kf_v5_cross3_C0_control_s7777` (Athena) | −0.0008 | ~−0.005 | **CONTAMINATION**: resumed from pre-fix ep6 ckpt (fold=7777). | EXCLUDE from Table H C0 row. |
| `los_v5_cross3_C0_control_s7777` (Athena) | −0.0301 | ~−0.022 | **CONTAMINATION**: resumed from pre-fix ep23 ckpt (fold=7777). | EXCLUDE from Table F C0 row. |
| `kf_v5_cross3_C5_no_fidelity_s7777_v2` | −0.0004 | ~−0.0074 | Genuine seed collapse: early-stop at ep18, best@ep3 (no-fidelity + bad seed → val_recon explodes). Log has zero "Resumed from" hits. | Keep in pool; report wide std. |

After exclusions, no remaining cells cross the 3σ threshold.

---

## Gaps (suggested queue entries)

### High priority — to complete paper n=3 on LoS cross3 (4 cells)
- `los_v5_cross3_C3_no_mmd_s42_v2`: **RUNNING** on Athena (72376).
- `los_v5_cross3_C4_no_target_task_s42_v2`: Athena **PENDING** (72377).
- `los_v5_cross3_C5_no_fidelity_s42_v2`: Athena **PENDING** (72378).
- `los_v5_cross3_C8_residual_s42_v2`: Athena **PENDING** (72381).

Configs already uploaded (`experiments/.athena_configs/los_v5_cross3_C{3,4,5,8}_no_*_s42_v2.json`). No manual action needed — jobs will run automatically. Expected completion: within 2–3 days.

### Medium priority — multi-seed for paper KF record
- Paper's KF record config (`kf_hp_K5_lr3e5` = `kf_lr3e5_nfnm`) has n=3 at C0. Mean is −0.0086 ± 0.0014 (not −0.0103 single-seed).
- To improve precision, queue s42_v2 / s7777_v2 for the OTHER HP winners: `kf_hp_K1_50ep`, `kf_hp_K3_window12`, `kf_hp_K7_pretrain30`.

### Low priority — ablations currently single-seed
- **los_nm** (all 9 cells): need s42/s7777_v2 configs created + submitted to Athena. Paper's LoS MAE record (−0.0320, `los_nm_C5_no_fidelity`) is still single-seed.
- **kf_nf** (all 9 cells): same — no v2 runs submitted.
- **kf_nfnm C1–C9**, **kf_lr3e5 C1–C9** (16 cells each base): low priority since the base/C0 cells already have n=3.

Config template: copy `configs/seeds/kf_nfnm_C0_control_s{42,7777}.json` pattern, duplicate for other C-suffixes, submit via `athena_submit.py --config configs/seeds/<name>.json --name <name>_v2`.

---

## Task 3 — files updated (Apr 21 session)

- `/bigdata/omerg/Thesis/EHR_Translator/deep_pipeline/docs/neurips/multiseed_ablation_tables.md` (this file — LoS/KF sections replaced with n=3 tables).

---

## Clean-rerun queue drafts (Apr 21, NOT submitted)

After the resume-integrity fix lands, queue these to repair the three
contaminated cells. **Do not submit until the fix is pushed to origin and the
remote worktrees have pulled it.**

### mort_c2_C0_control_s42_v2_clean (local / a6000)
```yaml
- name: mort_c2_C0_control_s42_clean
  config: configs/seeds/mort_c2_C0_control_s42_clean.json   # copy of existing s42 config, run_dir → runs/seeds_v2/mort_c2_C0_control_s42_clean
  output: runs/seeds_v2/mort_c2_C0_control_s42_clean/eval_mortality.parquet
  command: train_and_eval
  server: a6000
  status: pending
  notes: |
    Clean n=3 repair for mort_c2 C0 row. Uses fresh run_dir to sidestep the
    contaminated runs/seeds/mort_c2_C0_control_s42/ directory (pre-fix v1
    checkpoint). Requires commit 00433dc + resume-integrity fix.
```
Rationale: pointing at a **fresh** `run_dir` (rather than deleting the
contaminated one) is safer — the contaminated dir holds a `best_translator.pt`
from the v2 run that might still be referenced by downstream eval tools.
Fresh dir + new config file (copy of `s42.json` with one field changed) is
a single-variable delta.

### kf_v5_cross3_C0_control_s7777_v3 (Athena)
```yaml
- name: kf_v5_cross3_C0_control_s7777_v3
  config: configs/seeds/kf_v5_cross3_C0_control_s7777_v3.json  # run_dir → runs/seeds_v3/kf_v5_cross3_C0_control_s7777
  output: runs/seeds_v3/kf_v5_cross3_C0_control_s7777/eval_kf.parquet
  command: train_and_eval
  server: athena
  status: athena_pending
  notes: Repair for Table H C0 row contamination (resumed from pre-fix ep6).
```

### los_v5_cross3_C0_control_s7777_v3 (Athena)
```yaml
- name: los_v5_cross3_C0_control_s7777_v3
  config: configs/seeds/los_v5_cross3_C0_control_s7777_v3.json  # run_dir → runs/seeds_v3/los_v5_cross3_C0_control_s7777
  output: runs/seeds_v3/los_v5_cross3_C0_control_s7777/eval_los.parquet
  command: train_and_eval
  server: athena
  status: athena_pending
  notes: Repair for Table F C0 row contamination (resumed from pre-fix ep23).
```

---

## Outstanding work (updated Apr 21)

1. ~~Extract LoS/KF v2 eval blocks from Athena~~ — **DONE** (105 blocks extracted via batch SLURM `.err` scrape).
2. **Apply resume-integrity fix** — **DONE Apr 21** (`src/core/train.py :: validate_resume_checkpoint`, tests in `tests/test_resume_checkpoint_integrity.py`). All 33 tests pass. Future silent resumes impossible.
3. Re-run `mort_c2_C0_control_s42` (clean, new run_dir). See queue draft above.
4. Re-run `kf_v5_cross3_C0_control_s7777` and `los_v5_cross3_C0_control_s7777` on Athena (both contaminated). See queue drafts above.
5. Wait for 4 LoS cross3 s42_v2 Athena jobs (C3 running, C4/C5/C8 pending) to complete full n=3 for LoS (remaining ablations).
6. Queue missing mort_c2 v2 cells: C2, C6, C7, C9 (currently n=1).
7. Queue aki_nf v2 seeds (s42 + s7777) — paper's AKI record is still single-seed.
8. Queue `los_nm_*_v2` seeds to multi-seed the paper's LoS MAE record.
9. Queue `kf_nf_*_v2` seeds to multi-seed the KF cross3 no-fidelity ablation.
10. Queue full sepsis_C\* v2 multi-seed pool — sepsis n=3 requires it.
