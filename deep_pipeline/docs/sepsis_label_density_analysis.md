# Sepsis Label Density Analysis & Negative Subsampling Strategy

**Date**: 2026-02-21
**Context**: After AKI experiments confirmed label density as the primary bottleneck for sepsis translation, this document analyzes the sepsis data distribution and proposes negative stay subsampling to match AKI-like label density.

---

## 1. The Problem: Label Sparsity Causes Gradient Failure

### 1.1 Three-Task Comparison

| Metric | Sepsis | AKI | Mortality |
|---|---|---|---|
| Per-timestep positive rate | **1.13%** | **11.95%** | 5.52% |
| Per-stay positive rate | **4.57%** | **37.79%** | 5.52% |
| Total stays (eICU) | 123,412 | 165,285 | 113,390 |
| Positive stays | 5,639 | 62,463 | 6,250 |
| **All-negative stays** | **117,773 (95.4%)** | ~102,822 (62.2%) | ~107,140 (94.5%) |
| Best ΔAUCROC (delta) | **+0.0102** | +0.0242 | +0.0329 |
| Best ΔAUCROC (shared latent) | -0.0172 | +0.0370 | +0.0441 |

### 1.2 Gradient Alignment: The Root Cause

Measured at translator parameters (first 4 batches of epoch 0):

| Metric | Sepsis | Mortality | Interpretation |
|---|---|---|---|
| Task gradient norm | 1.052 | 2.163 | Mortality task gradient 2x stronger |
| Fidelity/Task ratio | 5.73x | 2.81x | Fidelity dominates sepsis 2x more |
| **cos(task, fidelity)** | **-0.21** | **+0.84** | **Destructive vs cooperative** |

In sepsis, within a typical positive stay: ~1-2 timesteps say "increase prediction" while ~35 say "decrease prediction." These contradictory micro-gradients largely cancel. Meanwhile fidelity says "change nothing." The net result: fidelity dominates with tiny, noisy task perturbations.

### 1.3 AKI Proved Label Density is THE Bottleneck

AKI is per-timestep + causal (identical structure to sepsis) but with 11.95% positive rate. Both translators work at full scale:
- Delta: +0.0242 AUCROC, +0.0781 AUCPR
- Shared latent: +0.0370 AUCROC, +0.1021 AUCPR

**Conclusion**: Per-timestep structure and causal attention are NOT the bottleneck. Label density is.

---

## 2. Sepsis Data Statistics

### 2.1 Stay-Level Label Distribution

| Category | Count | Percentage |
|---|---|---|
| Total stays | 123,412 | 100% |
| All-negative stays (zero sepsis) | 117,773 | **95.4%** |
| Positive stays (≥1 positive TS) | 5,639 | 4.6% |

**95.4% of training stays contribute zero task gradient — only fidelity gradient.**

### 2.2 Positive Stay Characteristics

| Metric | Value |
|---|---|
| Mean positive timesteps per stay | 12.88 |
| Median positive timesteps per stay | 13 |
| Range | 7–13 (fixed sepsis definition window) |
| Mean negative timesteps per positive stay | ~26.8 |
| Within-stay positive rate | ~32.5% |

Within positive stays, the per-timestep positive rate is **32.45%** — nearly identical to AKI's 36.5% within-positive-stay rate.

### 2.3 Missingness Patterns

| Metric | Value |
|---|---|
| Mean missingness across all stays | 85.7% |
| Median missingness | 85.3% |
| Minimum missingness (any stay) | **64%** |
| Stays with >80% missingness | 96.7% |
| Stays with >90% missingness | 10.1% |

**Missingness is universal** — every stay has >50% missing features. This is not a useful filtering lever.

### 2.4 Missingness-Label Correlation

| Category | Stays | Positive Rate |
|---|---|---|
| ≤80% missingness | 4,039 | **16.1%** |
| >80% missingness | 119,373 | 4.2% |

Low-missingness stays are enriched for positives (sicker patients get more tests). This is a correlation, not a causal filter.

### 2.5 Source vs Target Comparison

| Metric | eICU (source) | MIMIC (target) |
|---|---|---|
| Total stays | 123,412 | 82,753 |
| Per-stay positive rate | 4.57% | 4.73% |
| Per-TS positive rate | 1.13% | 1.09% |
| Mean missingness | 85.7% | 84.0% |
| Max sequence length | 169 | 169 |

Both domains have matching label distributions — the problem is symmetric.

---

## 3. Why Current Oversampling Doesn't Solve This

With `oversampling_factor=20`, the effective per-stay sampling rate reaches ~49% (positive stays sampled 20x more often). This is **higher than AKI's native 37.8%**.

But it doesn't help because:
1. **Same 5,639 positive stays repeated 20x** → overfitting to those specific feature patterns
2. **117K negative stays still process** → fidelity gradients from negatives drown out task signal
3. **No new positive diversity created** — just more forward passes on the same data
4. **Per-timestep rate within each stay unchanged** — still 1.1% across full batches

---

## 4. Strategy: Negative Stay Subsampling

### 4.1 Rationale

Instead of oversampling positives (which repeats data), **remove most negative stays**. This:
- Eliminates stays that produce zero task gradient (only fidelity noise)
- Matches the conditions where translation is known to work (AKI-like density)
- Makes epochs ~10x faster (13K vs 123K stays)
- Each positive stay seen 1x/epoch (no overfitting from repetition)
- Preserves negative diversity for fidelity learning

### 4.2 Subsampling Targets

| Scenario | Keep Pos | Keep Neg | Total | Per-Stay Pos% | Per-TS Pos% |
|---|---|---|---|---|---|
| **Baseline** | 5,639 | 117,773 | 123,412 | 4.6% | 1.13% |
| Match AKI per-stay (37.8%) | 5,639 | 9,281 | 14,920 | 37.8% | 10.2% |
| **Match AKI per-TS (12.0%)** | **5,639** | **7,300** | **12,939** | **43.6%** | **12.0%** |
| Positive only | 5,639 | 0 | 5,639 | 100% | 32.5% |

### 4.3 Chosen Target: Match AKI Per-TS Rate (~12%)

**Configuration**: Keep all 5,639 positive stays + 7,300 random negative stays = 12,939 total.

**Why this target**:
- Matches the exact conditions where both translators work (+0.037 AUCROC for AKI SL)
- Balanced ratio: 7.3K negatives vs 5.6K positives — fidelity still has diverse negative patterns
- Per-TS rate of 12.0% matches AKI's 11.95% — direct comparability
- No oversampling needed — natural ratio is already balanced
- 10x faster epochs than full data

### 4.4 Validity Assessment

**Why removing negatives is valid for domain adaptation**:
- The translator's job for negative stays is "preserve features" (identity-like) — fidelity loss handles this alone
- 7,300 negative stays still provide diverse feature patterns for fidelity learning
- At test time, the translator evaluates on ALL stays (including negatives) — the test set is unchanged
- The MIMIC target data remains unchanged — alignment quality is preserved

**What could go wrong**:
- If specific negative feature patterns are important for fidelity learning, 7.3K may not be enough diversity
- Random selection of negatives may miss edge cases → could try stratified selection by missingness
- The translator may learn a bias toward positive-patient feature patterns

**Mitigation**: Compare against baseline on full test set (all stays). If negative-patient translation degrades, increase negative count.

---

## 5. Experiment Design

### 5.1 Filtered Cohort Creation

Create filtered eICU sepsis cohort at `/bigdata/omerg/Thesis/cohort_data/sepsis/eicu_filtered_aki_density/`:
1. Identify all positive stays from `outc.parquet` (≥1 positive label)
2. Randomly sample 7,300 negative stays (seed=2222 for reproducibility)
3. Filter `dyn.parquet`, `outc.parquet`, `sta.parquet` to keep only selected stays
4. Copy `preproc/` directory (static recipe unchanged)

### 5.2 Experiments to Run

| # | Method | Config Base | Expected GPU Hours | GPU |
|---|---|---|---|---|
| 1 | Delta-based (d128, causal) | AKI delta full | ~1-2h | 0 |
| 2 | Shared Latent v3 | AKI SL full | ~3-4h | 0 (after #1) |

Both use:
- Same model architecture as AKI configs (proven to work at similar density)
- VLB enabled (per-timestep task, long sequences)
- No oversampling (not needed — ratio already balanced)
- Shuffle enabled
- 30 epochs, patience 7

### 5.3 Success Criteria

| Metric | Current Best | Target (Conservative) | Target (Optimistic) |
|---|---|---|---|
| ΔAUCROC | +0.0025 | +0.010 | +0.025 |
| ΔAUCPR | +0.0008 | +0.005 | +0.020 |

If subsampling matches AKI scaling, we expect 4-10x improvement over current best.

---

## 6. Experimental Results (Feb 21)

### 6.1 Filtered Cohort

Created at `/bigdata/omerg/Thesis/cohort_data/sepsis/eicu_filtered_aki_density/`:
- 12,939 stays total (5,639 positive + 7,300 random negative)
- `dyn.parquet`: 609,881 rows (7.6 MB), `outc.parquet`: 159K rows, `sta.parquet`: 12,939 stays
- 1 null sex value imputed to 'Male' (mode) for YAIB LabelEncoder compatibility
- Seed: 2222

### 6.2 Results: Filtered Test Set

| Method | ΔAUCROC | ΔAUCPR | ΔBrier | ΔECE | Config |
|---|---|---|---|---|---|
| **Shared Latent v3** | **+0.0450** | **+0.0740** | +0.0606 | +0.1259 | d128/lat128, causal, VLB, batch=16, 15+30ep |
| Delta-based | +0.0052 | +0.0005 | +0.0639 | +0.1100 | d128, causal, VLB, batch=64, 30ep |

Baselines (filtered test): Delta AUCROC=0.6764, SL AUCROC=0.6753. Translated: Delta 0.6816, SL 0.7202.

### 6.3 Results: Original Full Test Set — INVALID (Data Leakage)

> **DATA LEAKAGE DISCOVERED (Feb 21)**: YAIB's `StratifiedShuffleSplit` splits each data_dir independently based on its stay_id set. When training on the filtered cohort and evaluating on the original cohort, YAIB generates different train/test splits for each. **893 filtered-train stays appeared in the original test set**, containing **79.1% of all positive timesteps** (11,506 of 14,540). The results below are INVALID.

~~Models trained on filtered cohort, evaluated on the original unfiltered eICU test set:~~

| Method | ~~ΔAUCROC~~ | ~~ΔAUCPR~~ | Status |
|---|---|---|---|
| ~~Shared Latent v3~~ | ~~+0.0805~~ | ~~+0.0238~~ | **INVALID (leakage)** |
| ~~Delta-based~~ | ~~+0.0002~~ | ~~-0.0003~~ | **INVALID (leakage)** |

### 6.4 Leakage-Free Fix: In-Split Negative Subsampling

To fix the leakage, `_apply_negative_subsampling()` was added to `src/cli.py`. Instead of creating a separate filtered cohort, this subsamples negative stays **within YAIB's train split** of the original (unfiltered) eICU data. Config key: `training.negative_subsample_count: 7300`.

After subsampling: 11,247 training stays (3,947 pos + 7,300 neg = 35.1% per-stay positive rate). Test set is unchanged (24,683 stays, 4.57% positive rate).

| Method | Baseline AUCROC | Translated AUCROC | **ΔAUCROC** | **ΔAUCPR** | Status |
|---|---|---|---|---|---|
| Delta-based (subsampled) | 0.7160 | 0.7144 | **-0.0016** | -0.0002 | Complete |
| Shared Latent v3 (subsampled) | 0.7159 | 0.7158 | **-0.0001** | -0.0009 | Complete |
| Previous best (C2, no subsample) | 0.7159 | 0.7184 | +0.0025 | +0.0008 | Confirmed |

**Neither method benefits from negative subsampling.** The +0.0805 was entirely due to leakage. The label density hypothesis — that matching AKI-like training density would unlock SL for sepsis — is **disproven**. The difference between AKI (works with SL) and sepsis (doesn't) is not just about training label density.

Configs: `configs/sepsis_subsample_{delta,sl}_full.json`

### 6.5 Analysis

**Why the leakage inflated results so dramatically:**
The 893 leaked stays contained 79.1% of positive timesteps in the test set. Since the model was trained on these exact stays, it could accurately predict their labels, producing a massive AUCROC inflation. This disproportionately affected the SL method (+0.0805) because SL's latent space memorized the training distribution more effectively.

**Filtered test results (Section 6.2) remain valid:** Both train and eval use the same data_dir, so YAIB generates the same train/test split. No leakage occurs within a single data_dir.

**Comparison to AKI (the inspiration):**

| Task | Training Density | ΔAUCROC (SL, full test) | ΔAUCPR (SL, full test) |
|---|---|---|---|
| AKI | Native 11.95% | +0.0370 | +0.1021 |
| Sepsis (subsampled, clean) | Matched ~12% | **-0.0001** | -0.0009 |
| Sepsis (unfiltered) | 1.13% | -0.0172 | -0.0011 |

**The density hypothesis is disproven.** Matching AKI-like training density does NOT help sepsis. The difference between AKI and sepsis must involve other factors: test-time label density (AKI 11.95% vs sepsis 1.13%), per-stay positive rate (AKI 37.8% vs sepsis 4.6%), or sequence structure differences.

### 6.6 Run Details

| Run | GPU | Duration | Best Epoch | Status |
|---|---|---|---|---|
| Delta filtered (leaked) | 0 | ~1h | — | INVALID |
| SL filtered (leaked) | 0 | ~3h | 30 | INVALID |
| Delta subsampled (clean) | 0 | ~30min | 19 | Complete: -0.0016 |
| SL subsampled (clean) | 0 | ~2.5h | 8 (early stopped) | Complete: -0.0001 |

Logs: `experiments/logs/sepsis_subsample_{delta,sl}_full.log`
Configs: `configs/sepsis_subsample_{delta,sl}_full.json`

---

## 7. Target Task Loss Breakthrough (Feb 23)

### 7.1 Approach: MIMIC Target Task Loss

Instead of modifying training label density, target task loss adds gradient signal from the MIMIC domain. Config: `lambda_target_task: 0.5`.

**For delta-based**: MIMIC data → translator → frozen LSTM → loss against MIMIC labels. The translator receives gradient from both eICU (source task) and MIMIC (target task) simultaneously.

**For shared latent**: Additionally includes `lambda_label_pred: 0.1` — an MLP head on the latent space predicts labels from both domains, bypassing the frozen LSTM entirely.

### 7.2 Results

| Method | ΔAUCROC | ΔAUCPR | ΔBrier | ΔECE | Notes |
|---|---|---|---|---|---|
| Previous best (C2 GradNorm) | +0.0025 | +0.0008 | — | — | Delta-based |
| **Delta + target task (full)** | **+0.0102** | **+0.0056** | **-0.0460** | **-0.0430** | **New best (4x improvement)** |
| SL + MIMIC labels (full) | -0.0071 | -0.0034 | +0.1304 | +0.1002 | SL still hurts |
| Delta + target task (filtered 12%) | -0.0047 | -0.0009 | +0.0545 | +0.0797 | Subsampling + TTL doesn't help |
| SL + MIMIC labels (filtered 12%) | -0.0073 | -0.0020 | +0.1812 | +0.2191 | Worst combination |

### 7.3 Why Target Task Loss Bypasses the Label Density Problem

The gradient bottleneck for sepsis has two components:
1. **Weak eICU task gradient** (1.1% positive rate → sparse, contradictory signal)
2. **Destructive task-fidelity interference** (cos = -0.21)

Target task loss addresses #2 by providing a **second, coherent task gradient** from the MIMIC domain:
- The frozen LSTM was trained on MIMIC — its gradients on MIMIC data are well-calibrated
- The MIMIC task gradient aligns with the translation direction ("make output more MIMIC-like")
- This changes the effective gradient alignment from destructive to cooperative
- The translator learns both "improve eICU predictions" and "preserve MIMIC prediction quality"

Importantly, the improvement (+0.0102) comes from **full data** — the 123K stays provide diverse patterns for both the eICU and MIMIC task losses. This is why the filtered version (-0.0047) fails: reduced diversity undermines the target task signal.

### 7.4 Calibration Improvement

Unlike most experiments, delta + target task loss **improves calibration** (Brier -0.046, ECE -0.043). The dual task loss prevents the translator from distorting the feature space in ways that shift prediction probabilities, acting as implicit calibration regularization.

### 7.5 Subsampling + Target Task Loss: Compound Failure

Combining negative subsampling with target task loss produces worse results than either alone:
- Filtered delta + TTL: -0.0047 (vs full delta + TTL: +0.0102)
- Filtered SL + TTL: -0.0073 (vs full SL + TTL: -0.0071)

This confirms that **data volume matters more than density** for the target task loss mechanism. The diverse negative stays (117K) contribute meaningful fidelity learning that the target task loss builds upon.

---

## 8. Future Directions (Updated Feb 23)

### 8.1 Subsampling Conclusion
Negative subsampling does NOT help sepsis, even when combined with target task loss. The label density hypothesis is disproven.

### 8.2 Current Best Strategy
Delta-based + target task loss on full data: **+0.0102 AUCROC**. This is the confirmed approach for sepsis.

### 8.3 Remaining Approaches for Sepsis
- **Extended training / hyperparameter tuning**: Delta + TTL best at ep20/30, may benefit from more epochs or different lambda_target_task
- **Combine with C3 (cosine fidelity)** or A3 (padding-aware): Compound TTL with proven mortality improvements
- **Per-stay aggregation + TTL**: Aggregate per-timestep predictions to per-stay before loss, combined with target task loss
- **Curriculum training**: Start with high lambda_target_task (learn MIMIC alignment), decay over training to emphasize eICU task
- **Multi-seed validation**: Confirm +0.0102 is stable

### 8.4 Calibration (High Priority for Mortality/AKI)
- Temperature scaling on the full test set to fix Brier/ECE degradation
- Note: sepsis delta + TTL already shows improved calibration (-0.046 Brier)

---

## 9. Related Documents

- **[docs/comprehensive_results_summary.md](comprehensive_results_summary.md)** — Master results including AKI comparison
- **[docs/gradient_bottleneck_analysis.md](gradient_bottleneck_analysis.md)** — Gradient alignment measurements
- **[docs/shared_latent_results.md](shared_latent_results.md)** — Why shared latent fails on sepsis
- **[docs/experiment_results_abc.md](experiment_results_abc.md)** — A/B/C debug results
