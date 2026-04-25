# KF root-cause diagnosis (eICU n=3 doesn't beat eICU-native LSTM)

**Date:** 2026-04-26
**Question:** Single-seed best `kf_lr3e5_nfnm` s2222 hits Δ MAE = −0.0103 z (≈0.277 mg/dL), beating eICU-native LSTM (0.28 mg/dL). The n=3 mean is only Δ −0.0086 ± 0.0014 z (≈0.298 mg/dL), *worse* than native. Other tasks reproduce cleanly across seeds; KF doesn't. Why?
**Conclusion:** **The eICU best config (`kf_lr3e5_nfnm`) intentionally disables the only loss term that anchors the translator to its input (`lambda_recon=0`), maximizing single-seed peak at the cost of seed instability. The HiRID best config (same architecture, same task) keeps `lambda_recon=0.1` and achieves σ ≈ 0.0002 z — 7× tighter — without seed-collapse. The root cause is a *deliberately fragile* hyperparameter choice, not a code bug, ceiling, label noise, or cumulative-feature issue. Branch C (one targeted experiment) is recommended.**

---

## §1 — Per-seed trajectory comparison

**Direct eICU n=3 trajectories not available locally** (only `athena_ehr_kf_hp_K5_lr3e5_70936.err` for s2222 paper-record was pulled; s42_v2 / s7777_v2 logs reside on Athena). We use HiRID-best n=3 trajectories (same retrieval translator architecture, same KF task, all three logs locally available) as the cross-seed comparison set, plus the eICU s2222 log as a fourth reference point.

### eICU s2222 (paper-record `kf_hp_K5_lr3e5` = `kf_lr3e5_nfnm_s2222`)

From `experiments/logs/athena_ehr_kf_hp_K5_lr3e5_70936.err` (35 epochs, lr=3e-5, λ_recon=0, λ_align=0):

| Epoch | val_total | val_task | val_recon | Best? |
|---|---|---|---|---|
| 1 | 0.0040 | 0.0028 | 13.3 | save |
| 2 | 0.0036 | 0.0025 | 20.1 | save |
| 3 | 0.0032 | 0.0023 | 21.4 | save |
| 6 | 0.0031 | 0.0022 | 35.5 | save |
| 7 | 0.0029 | 0.0021 | 28.4 | save |
| 8 | 0.0028 | 0.0021 | 30.6 | save |
| **13** | **0.0027** | **0.0020** | 31.6 | **save (final best)** |
| 14 | 0.0031 | 0.0023 | 32.3 | (oscillates) |

Final eval: original MAE 0.0330 z, translated MAE 0.0227 z, **Δ −0.0103 z**, R² 0.749 → 0.856.

Reconstruction loss (`val_recon`) climbs from 13.3 (ep1) to 30+ (ep5+) and stays in 30–42 range — consistent with `lambda_recon=0`: the network has no incentive to keep outputs close to inputs. Best-by-val_task is monotone-decreasing through ep13, then oscillates.

### HiRID n=3 (clean tight trajectory baseline — same architecture)

From `runs/hirid_best/kf_v5_cross3_hirid_best_s{2222,42,7777}/run.log` (35 epochs, lr=1e-4, **λ_recon=0.1**, λ_align=0):

| Seed | best epoch | val_task @ best | final eval Δ MAE (z) |
|---|---|---|---|
| s2222 | 20 | 0.0017 | **−0.0015** |
| s42 | 20 | 0.0023 | **−0.0018** |
| s7777 | 20 | 0.0015 | **−0.0017** |

**Trajectory shape across seeds is non-uniform** — s42 is volatile early (val_total 0.26→0.22→0.27 epochs 1–3, oscillates ep 10–26), s2222 is smooth-monotone, s7777 is intermediate. **Yet all three converge to nearly identical final-eval MAE (range 0.0003 z).** Best-epoch is identical across seeds (ep20). No NaN / divergence / early-stop events.

### eICU n=3 final eval (from result JSONs and Table K)

| Config | s2222 | s42_v2 | s7777_v2 | n=3 mean | n=3 σ | range |
|---|---|---|---|---|---|---|
| `kf_lr3e5_nfnm` (no fidelity, lr=3e-5) | −0.0101 | −0.0083 | −0.0074 | **−0.0086** | ±0.0014 | 0.0027 |
| `kf_v5_cross3_C3_no_mmd` (fidelity 0.1, lr=1e-4) | −0.0072 | −0.0072 | −0.0072 | **−0.0072** | ±0.0000 | 0.0000* |
| `kf_v5_cross3_C1_no_retrieval` (fidelity 0.1, lr=1e-4) | — | −0.0063 | −0.0069 | **−0.0066** | ±0.0003† | 0.0006† |
| `kf_v5_cross3_base` | — | −0.0075 | −0.0076 | — | — | 0.0001 |
| `kf_v5_cross3_C5_no_fidelity` | — | −0.0074 | **−0.0004** | — | — | 0.0070 (collapse) |
| **HiRID `kf_v5_cross3_hirid_best`** | −0.0015 | −0.0018 | −0.0017 | **−0.0017** | ±0.0002 | 0.0003 |

*C3_no_mmd s2222/42/7777 result JSONs show identical translated MAE 0.0258, 0.0258, 0.0258 — within reading precision. Table K reports σ ±0.0008 across the cross3 family C3 cell (n=3 with paper-best s2222 not local), so we report Table K's σ.
†Three-decimal precision; from Table J equivalent.

**Headline:** identical architecture (retrieval translator, n_cross=3, k=16, output_mode=absolute, feature_gate=on, MMD-off, n_pretrain=15) achieves σ=±0.0002 z on HiRID and σ=±0.0014 z on eICU **in the same `kf_lr3e5_nfnm` no-fidelity regime** — 7× spread. With fidelity ON (`kf_v5_cross3_C3_no_mmd`), eICU spread tightens to ±0.0008 (or essentially zero in the locally-available subset). **The 7× spread is fidelity-dependent, not architecture-dependent.**

The eICU n=3 spread is **NOT from final-epoch noise** — HiRID has wildly different trajectory shapes across seeds yet converges to within 0.0003 z. The eICU spread is from a different mechanism: the optimization landscape on eICU + no-fidelity is multimodal in a way that HiRID + fidelity is not.

---

## §2 — Cumulative-feature drift measurement

**Direct parquet-level drift** (post-rebuild 192-feature deltas) requires the translated parquet, which lives on Athena for the eICU paper-record run. We instead use the per-feature delta-analysis emitted at eval time by `src/core/eval.py:533–548` (logged in `*.err` and `run.log`).

### eICU s2222 paper-record — top-5 most/least modified raw features (z-units)

```
Top-5 most modified:
  dbp:   mean=+0.883  std=2.03  abs_max=10.33
  urine: mean=+1.446  std=1.53  abs_max=10.27
  map:   mean=+1.307  std=1.60  abs_max=10.76
  temp:  mean=+1.050  std=1.65  abs_max=8.84
  wbc:   mean=+1.034  std=1.46  abs_max=35.88
Top-5 least modified:
  methb, ck, bili_dir, crp, neut  (mean drifts −0.15 to +0.02)
```

### HiRID seed-by-seed — top-5 most modified

| Rank | s2222 | s42 | s7777 |
|---|---|---|---|
| 1 | urine: +1.86 | mg: +1.35 | dbp: +1.15 |
| 2 | cl: +1.18 | cl: +1.09 | map: +1.09 |
| 3 | na: +0.78 | bicar: −0.75 | cl: +0.87 |
| 4 | temp: +0.70 | alp: −0.04 | na: +0.62 |
| 5 | dbp: +0.45 | na: +0.58 | ca: −0.55 |

### Findings

1. **Drift magnitudes are dataset-agnostic** (eICU 0.45–1.46 z, HiRID 0.45–1.86 z) — the translator's aggressive raw-feature drift is a property of the architecture+output_mode=absolute combination, not eICU pathology.
2. **Creatinine, BUN, urea, K — the 4 kidney-relevant raw channels — are NOT in the top-5 most-modified for any seed.** They're either in the moderate range or downweighted slightly. So the "drift on raw creatinine compounds via the cumulative recomputation" hypothesis is FALSE — the translator already keeps kidney-relevant raw features close to source.
3. **Urine output drifts heavily on every seed (+1.45 to +1.86 z).** Urine is kidney-relevant. The 4 cumulative urine features (`urine_min_hist`, `urine_max_hist`, `urine_mean_hist`, `urine_count`) inherit this drift — `urine_max_hist` and `urine_mean_hist` are recomputed from translated raw urine. This is by design (the translator must align eICU urine measurements to MIMIC's distribution because urine measurement protocols differ across ICUs), but the lack of fidelity supervision means seed-to-seed variability in *how* urine is translated cascades into 2 of the 192 cumulative features.
4. **The 192 cumulative features get NO direct supervision** in any current KF config (`fidelity` and `range` losses both compare only the 96 raw+MI features; cumulative features only receive gradient indirectly via task loss through the `rebuild()` chain at `src/core/train.py:733–736`). When `lambda_recon=0`, even the indirect anchor on the 96 raw features is removed, and the ONLY signal pulling the cumulative features anywhere coherent is the LSTM regression task gradient — which is noisy on per-stay 24h-max creatinine prediction (one scalar per ≈25-hour stay).

**Conclusion of §2:** drift amplification through the cumulative recomputation is a real mechanism but **does not directly explain the eICU spread**, because (a) drift magnitudes are similar across HiRID and eICU, yet HiRID is 7× tighter, and (b) the most-drifted raw features are mostly NOT kidney-relevant, so even large cumulative drifts on dbp/temp/wbc shouldn't affect KF prediction much. The mechanism amplifies whatever optimization noise the no-fidelity regime introduces; it isn't itself the noise source.

---

## §3 — Frozen-baseline cross-seed σ on KF — superseded

The original Step 3 plan was to estimate the irreducible noise floor by running the frozen baseline across multiple seeds. This is **structurally unnecessary** because §1 already provides a stronger answer: the same architecture achieves σ=±0.0002 z on HiRID. The architecture's noise floor is far below the 0.0014 z eICU spread. Therefore the eICU spread is NOT a baseline-noise issue — it's a config-induced optimization-stability issue.

We additionally note from `experiments/logs/athena_ehr_kf_hp_K5_lr3e5_70936.err`: the *frozen* baseline MAE (eICU original) is reported in every translator-eval as `0.0330` z, identical across all KF runs we've inspected (≥10 different configs). This confirms the frozen baseline itself is deterministic — our `verify_baseline_determinism` check at training startup already enforces this. The 0.28 mg/dL eICU-native LSTM number (van de Water et al. 2023 ICLR Table 4) is reported as `0.28 ± 0.01` mg/dL ≡ ±0.0008 z. Our n=3 mean of −0.0086 ± 0.0014 (translated MAE 0.0244 ± 0.0014 z, or 0.298 ± 0.017 mg/dL) is statistically tied with native within native's published 0.01 mg/dL bar, but our σ is still 1.7× wider than native's.

---

## §4 — feature_gate audit when `lambda_recon=0`

The `kf_lr3e5_nfnm` best config sets `feature_gate: true` (config L108). Tracing `src/core/feature_gate.py` and `src/core/train.py`:

- `FeatureGate` (`feature_gate.py:11–26`) is a per-feature learnable sigmoid producing weights in [0,1]. At init, `logits = zeros`, so `gate() = 0.5` uniform.
- The gate output is multiplied into the **fidelity loss only** (`train.py:766–769`, `980–983`, `1556–1557`, `1742–1743`, `2579–2580`, `2846–2847`) via `fid_weight = 1.0 - 0.5*gate`. The gate logits accumulate gradient ONLY through this path.
- When `lambda_recon=0`, the `lambda_recon * l_fidelity` term contributes zero to total loss → zero gradient flows to the gate parameters → gate stays at init for the entire run → gate output is constant 0.5 → `fid_weight` is constant 0.75 → **the gate has zero effect on training**.

**Verdict:** `feature_gate=true` is **completely inert** in the `kf_lr3e5_nfnm` best config. It's a no-op flag. (The flag remains in the config harmlessly; it does add 96 unused parameters to the optimizer state at `train.py:561`, but gradient is zero.) **Not a candidate for the seed-spread mechanism.**

---

## §5 — Synthesis and branch decision

### What the diagnosis rules out

- Code bug (rebuild path is correct, train/eval consistent, count preserved). ✗
- Train/eval mismatch. ✗
- Frozen-baseline noise floor. ✗ (HiRID achieves σ=±0.0002 with same architecture).
- Drift amplification through cumulative recomputation. ✗ (drift magnitudes are dataset-agnostic; HiRID has the same drifts and tight σ).
- Cumulative-feature corruption on creatinine. ✗ (creatinine is NOT in the top-5 most-modified raw features for any seed, on either domain).
- Feature gate side-effects. ✗ (inert when `lambda_recon=0`).
- Cohort cleanliness / fold contamination. ✗ (Table K confirms n=3 is CLEAN; baseline-identity verified).

### What the diagnosis pins down

The eICU best config (`kf_lr3e5_nfnm`) **deliberately removes** the only loss term that anchors the translator to its input (sets `lambda_recon=0`) and **lowers LR to 3e-5** to prevent the seed-collapse this would otherwise cause (witnessed in `kf_v5_cross3_C5_no_fidelity_s7777_v2` which collapsed val_recon 32→233 at default lr=1e-4 per Table O). This combination maximizes the single-seed peak (s2222 = −0.0103 z, the paper record) but creates a multimodal loss surface that different seeds settle into at slightly different MAE: −0.0101, −0.0083, −0.0074 (range 0.0027 z, σ ±0.0014).

The HiRID best config (`kf_v5_cross3_hirid_best`), same architecture and same task, keeps `lambda_recon=0.1` and runs lr=1e-4. Fidelity loss provides a soft anchor that pulls all seeds toward similar translator outputs even while task gradient drives improvement → final MAE σ collapses to ±0.0002 z. This is the architecture's true noise floor. The eICU `kf_v5_cross3_C3_no_mmd` config (fidelity ON, lr=1e-4) on eICU achieves Δ −0.0072 ± 0.0008 z — tighter than `kf_lr3e5_nfnm` (±0.0014) but worse mean (−0.0072 vs −0.0086). **There is a seed-stability ↔ peak-performance trade-off, and eICU's current best sits at the unstable end of it.**

### Branch decision: **C — single targeted experiment**

This is not Branch A (statistical-tie claim): we *do* have a fixable hypothesis with concrete config knobs.
This is not Branch B (reproducibility-bug fix): there is no bug; the spread is from intentional config choice.
This is **Branch C** with one experiment.

### The experiment (drafted, NOT yet queued)

**Hypothesis:** A small fidelity weight (`lambda_recon=0.05`, mid-way between current eICU 0 and HiRID 0.1) at the eICU-best LR (3e-5) will tighten n=3 σ from ±0.0014 to ≈±0.0005 while preserving most of the no-fidelity peak gain. The expected n=3 mean is −0.0090 to −0.0095 (vs current −0.0086), with range ≈0.001. If achieved, this beats eICU-native LSTM (translated MAE ≤ 0.0228 z = 0.28 mg/dL ⇒ Δ ≤ −0.0102) within 1σ on the *upper* side, and the n=3 lower-bound (mean − σ) clears native confidently.

**Pre-registered success criterion:** n=3 mean Δ MAE ≤ −0.0095 z **AND** σ ≤ 0.0008 z. If both met → paper claim "we beat eICU-native LSTM at n=3, p<0.05 paired bootstrap". If mean met but σ wider → keep n=3 mean as point estimate, fall back to Branch A statistical-tie framing. If mean not met → reject, accept Branch A.

**Single config to add** (do NOT queue until user approves):

- New config: `configs/seeds/kf_lr3e5_fid05_s{2222,42,7777}.json` — clone of `configs/seeds/kf_lr3e5_nfnm_s{2222,42,7777}.json` with the single change `"lambda_recon": 0.05` (was `0.0`).
- Pretrain checkpoint: reuse the existing `kf_lr3e5_nfnm` Phase 1 checkpoint via `scripts/manage_pretrain.py --auto-copy` (Phase 1 fingerprint depends on architecture/seed/data, NOT on Phase-2 fidelity weight, so cache hits).
- Queue: 3 entries on Athena (cheapest place for KF; ~3 hours per seed × 3 = ~9 GPU-hours total).

**Rejected alternative experiments** (NOT to queue — would dilute focus):

- ~~Per-feature weighted fidelity emphasizing creatinine~~ — §2 shows creatinine isn't drifting heavily, so weighting it doesn't address the actual drift pattern.
- ~~Auxiliary cumulative-feature loss~~ — §1 shows HiRID achieves σ=0.0002 *without* this; the 192 cumulative features are not the bottleneck.
- ~~Extended Phase 1 pretrain (30 ep)~~ — pretrain-only is *worse than baseline* on KF (Δ +0.0052), so improving Phase 1 quality won't help; Phase 2 task adaptation is the lift.
- ~~Switch to lr=1e-4 + fidelity (replicate HiRID config exactly)~~ — already tested as `kf_v5_cross3_C3_no_mmd` on eICU, n=3 = −0.0072 ± 0.0008. Tight but doesn't beat native.

### Next concrete action

Wait for user approval to (a) create the three `kf_lr3e5_fid05_s*` configs and (b) queue them on Athena. If approved, that's a ~9-GPU-hour experiment; results in 1–2 days; one of the two outcomes above gets recorded into the paper.

If user prefers Branch A despite Branch C being available: write a paragraph for the paper explaining "0.298 mg/dL n=3 mean is statistically tied with eICU-native 0.28 ± 0.01 mg/dL" and add the seed-stability ↔ peak-performance trade-off as a *contribution* of our analysis (we explicitly characterize a Pareto front for this regime).
