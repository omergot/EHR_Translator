# Investigation: Why Mortality Works But Sepsis Doesn't

> **Role**: Controlled experiments systematically ruling out hypotheses (attention mode, capacity, data size). Identifies sequence length + task structure as the root cause. Updated Feb 2026 with gradient alignment analysis and AKI comparison.
> **See also**: [gradient_bottleneck_analysis.md](gradient_bottleneck_analysis.md) (quantified gradient evidence confirming this), [comprehensive_results_summary.md](comprehensive_results_summary.md) (master results), [experiment_results_mmd_mlm.md](experiment_results_mmd_mlm.md) (earlier MMD/MLM results that motivated this investigation)

**Date**: Feb 13-14, 2026 (updated Feb 19)
**Question**: The mortality24 task with bidirectional attention achieves +0.023 AUCROC, while sepsis with causal attention barely moves (+0.002). What causes this gap?

---

## Background

Prior work established that the causal transformer translator on sepsis produces minimal improvement (+0.001 to +0.002 AUCROC) across all loss configurations (baseline, MMD, MLM, MLM+MMD). See [experiment_results_mmd_mlm.md](experiment_results_mmd_mlm.md) for full details.

Meanwhile, a mortality24 run with bidirectional attention had achieved AUCROC 0.8079 -> 0.8309 (+0.0230). This investigation isolates which factors explain the gap.

## Differences Between Mortality and Sepsis Runs

| Factor | Mortality24 | Sepsis (debug) | Sepsis (full) |
|---|---|---|---|
| **Attention mode** | Bidirectional | Causal | Causal |
| **Sequence length** | 25 timesteps | 169 timesteps | 169 timesteps |
| **d_model** | 128 | 64 | 128 |
| **Training data** | Full (79K stays) | Debug 20% (25K stays) | Full (86K stays) |
| **Positive rate (stay-level)** | 5.5% | 4.6% | 4.6% |
| **Task structure** | Per-stay label | Per-timestep label | Per-timestep label |
| **lambda_range** | 0.5 | 0.001 | 0 |
| **best_metric** | val_total | val_task | val_task |
| **AUCROC delta** | **+0.023** | **+0.002** | **+0.001** |

## Control Experiments on Mortality24

To isolate the effect of each factor, we ran controlled experiments on mortality24, changing one variable at a time from the bidirectional d128 baseline.

### Experiment Setup

All experiments use:
- Task: Mortality24, eICU -> MIMIC-IV
- Full data (79K train stays, 23K test stays)
- Baseline: Frozen LSTM (hidden_dim=185, layer_dim=2)
- lr=1e-4, batch_size=64, lambda_fidelity=0.1, lambda_range=0.5
- best_metric=val_total, 20 epochs, no early stopping
- seed=2222, deterministic training

### Results (Full Data, 20 Epochs)

| Experiment | Attention | d_model | AUCROC | delta | AUCPR | delta | val_task (best) |
|---|---|---|---|---|---|---|---|
| Baseline (no translator) | - | - | 0.8079 | - | 0.2965 | - | - |
| Bidirectional d128 | bidir | 128 | 0.8331 | **+0.0251** | 0.3256 | +0.0292 | 0.4872 |
| Causal d128 | causal | 128 | 0.8320 | **+0.0240** | 0.3238 | +0.0273 | 0.4919 |
| Causal d64 | causal | 64 | 0.8149 | **+0.0070** | 0.3096 | +0.0131 | 0.5147 |

### Results (Debug 20%, 20 Epochs, Shuffle Comparison — Feb 20)

| Experiment | Attention | d_model | Shuffle | AUCROC delta | AUCPR delta | Epochs | Best Ep | Early Stop? |
|---|---|---|---|---|---|---|---|---|
| Baseline (debug) | - | - | - | 0.8215 baseline | 0.3204 baseline | - | - | - |
| Bidir d128 | bidir | 128 | False | **+0.0059** | +0.0055 | 20/20 | 20 | No |
| Bidir d128 | bidir | 128 | True | **+0.0064** | +0.0064 | 20/20 | 20 | No |
| Bidir d64 | bidir | 64 | False | **+0.0042** | +0.0063 | 20/20 | 20 | No |
| Bidir d64 | bidir | 64 | True | **+0.0046** | +0.0061 | 20/20 | 18 | No |

**Shuffle effect**: Marginal for delta-based (~+0.0005 AUCROC). All debug runs were still improving at epoch 20.

### Configs
- `configs/mortality24_bidir_repro_config.json` — Bidirectional d128 (full)
- `configs/mortality24_causal_config.json` — Causal d128 (full)
- `configs/mortality24_causal_d64_config.json` — Causal d64 (full)
- `configs/mortality24_delta_d{128,64}_debug_shuf{0,1}.json` — Debug shuffle variants

### Run outputs
- `runs/mortality24_bidir_repro/` — loss curves + checkpoint
- `runs/mortality24_causal/` — loss curves + checkpoint
- `runs/mortality24_causal_d64/` — loss curves + checkpoint
- `runs/mortality24_delta_d{128,64}_debug_shuf{0,1}/` — debug shuffle experiment runs

---

## Existing Full-Data Sepsis Results

From earlier runs in run.log (Feb 8-9, 2026), using `configs/sample_transformer_config.json` (d_model=128, causal, full data, lambda_range=0):

| Experiment | Attention | d_model | Data | AUCROC | delta |
|---|---|---|---|---|---|
| Sepsis full data | causal | 128 | full (86K stays) | 0.7166 | **+0.0006** |
| Sepsis debug | causal | 64 | 20% (25K stays) | 0.7210 | **+0.0017** |

Full-data sepsis with d_model=128 performed **worse** than debug, confirming data size is not the bottleneck.

---

## Conclusions

### Factor 1: Causal vs Bidirectional Attention — NOT the bottleneck

On mortality24 (25 timesteps), causal attention (+0.0240) matches bidirectional (+0.0251). The causal model even trained slightly faster in early epochs. With short sequences, causal attention has sufficient context at every position.

### Factor 2: Model Capacity (d_model) — Significant but not sufficient

d_model=64 reduces mortality improvement from +0.024 to +0.007 (3.4x reduction). This is a real effect: the d64 model's val_task plateaued at ~0.52 from epoch 7, while d128 reached ~0.49.

However, even mortality d64 (+0.007) is 3.5x better than sepsis d64 (+0.002). And sepsis d128 full data (+0.001) is even worse. So capacity alone doesn't explain the gap.

### Factor 3: Data Size — NOT the bottleneck for sepsis, IS the bottleneck for mortality

Full-data sepsis (+0.0006) performed worse than debug sepsis (+0.0017). More data didn't help sepsis. This rules out the 20% subsetting as a limiting factor for sepsis.

However, for mortality, data scaling experiments (Feb 20) show near-linear improvement with data volume:

| Data % | Train Stays | ΔAUCROC | ΔAUCPR |
|---|---|---|---|
| 20% | 15,873 | +0.0064 | +0.0064 |
| 40% | 31,746 | +0.0134 | +0.0120 |
| 60% | 47,619 | +0.0218 | +0.0226 |
| 80% | 63,492 | +0.0282 | +0.0276 |

All d128, causal, shuffle=true, 30 epochs. The 20% result comes from a 20-epoch run. Extended training (40 epochs) at 20% scale caused overfitting (AUCROC dropped from +0.0064 to +0.0048), confirming that the data volume—not epoch count—is the binding constraint for mortality.

### Factor 4: Sequence Length + Task Structure — Primary bottleneck

After ruling out attention mode, capacity, and data size, the remaining differences are:

1. **Sequence length: 25 vs 169 timesteps**
   - With 169 timesteps, even bidirectional attention faces a harder transformation problem — the translator must learn coherent changes across 6.8x more positions
   - The LSTM baseline processes sequences left-to-right; meaningful early-timestep transformations require understanding long-range patterns
   - Computational cost: forward pass is ~10x slower for 169 vs 25 timesteps (0.3s vs 0.03s), limiting the number of gradient steps per epoch

2. **Task structure: per-timestep vs per-stay**
   - Mortality: one binary label per stay. The LSTM makes a single prediction from the final hidden state. The translator just needs the overall trajectory to look right.
   - Sepsis: a label at every timestep. The LSTM's prediction at timestep t depends on the hidden state accumulated from timesteps 0..t. The translator must get EVERY timestep right, not just the overall pattern.
   - This makes the gradient signal much more diffuse for sepsis — each timestep's loss is tiny (1.1% positive rate at timestep level), and the translator must satisfy all of them simultaneously.

### Summary Table

| Factor | Ruled Out? | Evidence |
|---|---|---|
| Causal vs bidirectional | Yes | Causal matches bidir on mortality (+0.024 vs +0.025) |
| Model capacity (d_model) | Partial | Matters (~3x effect) but doesn't explain the full gap |
| Data size | Yes | Full data sepsis (+0.001) worse than debug (+0.002) |
| Sequence length (169 vs 25) | **Primary suspect** | Cannot be tested in isolation (tied to task) |
| Task structure (per-timestep vs per-stay) | **Primary suspect** | Cannot be tested in isolation (tied to task) |

---

## Implications for Next Steps

1. **The causal transformer architecture may be fundamentally limited for per-timestep tasks with long sequences.** The combination of 169 timesteps + per-timestep labels creates a much harder optimization landscape than 25 timesteps + per-stay labels.

2. **Model capacity helps but isn't sufficient.** Even d_model=128 on full data gives +0.001 for sepsis, while d_model=64 gives +0.007 for mortality.

3. **Possible directions:**
   - Try sepsis with truncated sequences (e.g., last 25 timesteps only) to test sequence length hypothesis
   - Try AKI task (per-timestep like sepsis but different cohort/sequence lengths)
   - Investigate whether the translator makes meaningful feature-level changes on sepsis (delta analysis)
   - Consider alternative architectures better suited for long-sequence per-timestep tasks
   - Focus on mortality task where the translator demonstrably works (+0.025 AUCROC)

---

## Update (Feb 19): Gradient Alignment Analysis

After completing all A/B/C experiments, full-data validation, and shared latent experiments (mortality +0.0441, sepsis -0.017 to -0.043), we revisited the original investigation question with new quantitative evidence. The gradient alignment discovery provides the clearest explanation yet.

### The Gradient Alignment Finding

The gradient diagnostic code (implemented in `src/core/train.py`) logs cosine similarity between task and fidelity gradient vectors. This reveals **cooperative vs destructive interference**:

| Metric | Mortality | Sepsis | Interpretation |
|---|---|---|---|
| Task grad norm | 2.163 | 1.052 | Mortality 2x stronger |
| Fidelity grad norm | 6.728 | 6.996 | Similar magnitude |
| Fid/Task ratio | **2.81x** | **5.73x** | Sepsis 2x harder to overcome |
| **cos(task, fidelity)** | **+0.84** | **-0.21** | **Cooperative vs destructive** |

**Key discovery**: In mortality, task and fidelity gradients point in the **same direction** (cos=+0.84). They cooperate — fidelity regularizes magnitude while task guides direction. In sepsis, they point in **opposite directions** (cos=-0.21). They cancel each other — the parameter update is dominated by destructive interference.

This is arguably the single most important finding explaining the performance gap. It means:
- **Mortality**: Fidelity acts as a *helpful regularizer* (same direction, limits step size)
- **Sepsis**: Fidelity acts as an *active adversary* (opposite direction, fights the task signal)

### Why Gradients Align for Mortality but Not Sepsis

**Mortality (per-stay label)**: All 24 timesteps contribute to one prediction. The task gradient says "adjust the entire stay trajectory" — this is structurally similar to "preserve the overall pattern" (what fidelity wants). Both losses push toward coherent, small adjustments.

**Sepsis (per-timestep labels)**: Each timestep has its own label. ~35 negative timesteps say "decrease prediction" while ~1-2 positive timesteps say "increase prediction." The net task gradient is a confused mix of opposing directions that happens to be roughly orthogonal to (or opposing) the fidelity direction of "don't change anything."

### Comprehensive Factor Comparison (Updated)

| Factor | Mortality24 | Sepsis | Impact |
|---|---|---|---|
| **Labels** | 1 per stay | 1 per timestep (1.1% pos) | Gradient coherence vs contradiction |
| **Positive rate** | 5.5% per-stay | 1.1% per-timestep | 5x sparser |
| **Sequence length** | Fixed 24 | Median 37, max 169 | 0% vs 73% padding |
| **Attention** | Bidirectional (full) | Causal window=25 | Rich vs limited context |
| **Task grad norm** | 2.163 | 1.052 | 2x weaker |
| **Fidelity/Task ratio** | 2.81x | 5.73x | 2x harder to overcome |
| **Gradient alignment** | **cos = +0.84** | **cos = -0.21** | **Cooperative vs destructive** |
| **Best ΔAUCROC** | +0.0441 (shared latent) | +0.0025 (C2 GradNorm) | 18x gap |

### Does More Labels = More Signal? No — the Opposite

A natural intuition is that sepsis, with labels at every timestep, should have *more* training signal than mortality's single per-stay label. **This is wrong.** More labels means more *contradictory* gradients:

- Within a single positive sepsis stay: ~1-2 positive timesteps generate "increase sepsis" gradients while ~35 negative timesteps generate "decrease sepsis" gradients. These largely cancel.
- Mortality's single per-stay label creates one coherent signal across all 24 timesteps — no internal contradiction.

Oversampling (f=20) increased per-stay positive frequency to 48.8% but didn't change the per-timestep rate (still 1.1%). Each positive stay still has ~35 negative timesteps per 1-2 positive ones. This is why oversampling helped (+0.006) but didn't solve the problem.

### Shared Latent Results Confirm the Pattern

The shared latent experiments (Feb 18-19) provide the strongest confirmation:

| Approach | Mortality ΔAUCROC | Sepsis ΔAUCROC |
|---|---|---|
| Shared Latent v3 | **+0.0441** | **-0.0325** |
| Shared Latent v1 | +0.0415 | -0.0172 |
| Best delta-based | +0.0285 (A3) | +0.0025 (C2) |

Shared latent works for mortality because MMD alignment provides dense gradient that bypasses the frozen LSTM bottleneck. It fails for sepsis because reconstruction loss (dense, strong) overwhelms the weak task signal, and the model optimizes for reconstruction rather than task performance. The training dynamics show this clearly: sepsis shared latent has best AUCROC at epoch 1 and deteriorates from there.

---

## Update (Feb 19): AKI Hypothesis

AKI (Acute Kidney Injury) was identified as the ideal controlled experiment: per-timestep + causal like sepsis, but with 11.95% positive rate (10.6x higher). If AKI translation succeeds → label density is the bottleneck. If it fails → per-timestep structure itself is the issue.

---

## AKI Experiment Results (Feb 20): Label Density Confirmed as Root Cause

### AKI Debug Results

| Metric | Baseline | Delta-based (d128) | Shared Latent (d128/latent128) |
|---|---|---|---|
| **AUCROC** | 0.8600 | 0.8707 (**+0.0107**) | 0.8760 (**+0.0160**) |
| **AUCPR** | 0.5718 | 0.6231 (**+0.0513**) | 0.6207 (**+0.0489**) |
| Brier | 0.1340 | 0.1240 (-0.0100) | 0.1245 (-0.0095) |

Configs: `configs/aki_delta_debug.json`, `configs/aki_shared_latent_debug.json`. Both causal attention, VLB, no oversampling.

### AKI Full-Data Validation (Feb 20-21)

| Metric | Baseline (full) | Delta Full (d128) | Delta Δ | **SL Full (v3)** | **SL Δ** |
|---|---|---|---|---|---|
| **AUCROC** | 0.8558 | 0.8800 | **+0.0242** | 0.8928 | **+0.0370** |
| **AUCPR** | 0.5678 | 0.6460 | **+0.0781** | 0.6699 | **+0.1021** |
| Brier | 0.1365 | 0.1253 | **-0.0112** | 0.1253 | **-0.0111** |
| ECE | 0.1913 | 0.1880 | **-0.0032** | 0.1925 | +0.0012 |

Configs: `configs/aki_delta_full.json` (d128, causal, VLB, 20 epochs), `configs/aki_shared_latent_full.json` (v3, VLB, shuffle=true, 15 pretrain + 30 joint epochs).

**Delta debug → full**: +0.0107 → **+0.0242** AUCROC (2.3x). **SL debug → full**: +0.0160 → **+0.0370** AUCROC (2.3x). Both methods show identical 2.3x scaling.

The shared latent AUCPR improvement (**+0.1021**) is the **largest across any task or method** in the project. Both translators also improve calibration (Brier -0.011).

### Updated Three-Task Comparison (Full-Data Results)

| Dimension | Sepsis | **AKI** | Mortality24 |
|---|---|---|---|
| **Task structure** | Per-timestep | Per-timestep | Per-stay |
| **Per-timestep pos rate** | **1.13%** | **11.95%** | 5.52% |
| **Per-stay pos rate** | **4.57%** | **37.79%** | 5.52% |
| **Median seq length** | 38 | **28** | 25 |
| **Max seq length** | 169 | 169 | 25 |
| **Padding** | ~73% | ~58% | 0% |
| **Attention mode** | Causal | Causal | Bidirectional |
| **Delta-based ΔAUCROC (full)** | +0.003 | **+0.0242** | **+0.0329** |
| **Shared latent ΔAUCROC (full)** | -0.017 to -0.043 | **+0.0370** | **+0.0441** |
| **Shared latent ΔAUCPR (full)** | -0.001 to -0.009 | **+0.1021** | **+0.0456** |

### What AKI Proved

1. **Per-timestep structure is NOT the bottleneck**: AKI is per-timestep like sepsis, and both translators work.
2. **Causal attention is NOT the bottleneck**: AKI uses causal attention and both translators work.
3. **Label density IS the bottleneck**: AKI's 11.95% per-timestep rate (vs sepsis 1.13%) is the key difference.
4. **Shared latent works on per-timestep tasks with dense labels**: AKI shared latent +0.016, vs sepsis -0.017 to -0.043.
5. **In-stay positive ratio is the mechanism**: AKI has ~10 positive per 25 negative timesteps per stay (coherent signal). Sepsis has ~1-2 positive per 35 negative (contradictory gradients).

---

## Definitive Root Cause Ranking (Updated Feb 20)

| Rank | Factor | Evidence | Confirmed by AKI? |
|---|---|---|---|
| 1 | **Label density / gradient coherence** | AKI (11.95% pos) works, sepsis (1.13%) fails | **Yes — definitive** |
| 2 | **Gradient alignment** (cos +0.84 vs -0.21) | Logged diagnostic data | Likely (dense labels → cooperative gradients) |
| 3 | **Fidelity/task ratio** (5.7x vs 2.8x) | Logged gradient norms | Likely (more task signal → lower ratio) |
| 4 | **Sequence length** | AKI median=28 (works) vs sepsis median=38 | **Contributing, not primary** |
| 5 | **Padding waste** (~73% sepsis) | Bucket batching helps speed, not metrics | Mostly addressed |
| 6 | **Causal attention** | AKI uses causal and works | **Ruled out** |
| 7 | **Per-timestep structure** | AKI is per-timestep and works | **Ruled out** |
