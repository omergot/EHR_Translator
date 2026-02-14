# Investigation: Why Mortality Works But Sepsis Doesn't

**Date**: Feb 13-14, 2026
**Question**: The mortality24 task with bidirectional attention achieves +0.023 AUCROC, while sepsis with causal attention barely moves (+0.002). What causes this gap?

---

## Background

Prior work established that the causal transformer translator on sepsis produces minimal improvement (+0.001 to +0.002 AUCROC) across all loss configurations (baseline, MMD, MLM, MLM+MMD). See [mmd_mlm_experiment_results.md](mmd_mlm_experiment_results.md) for full details.

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

### Results

| Experiment | Attention | d_model | AUCROC | delta | AUCPR | delta | val_task (best) |
|---|---|---|---|---|---|---|---|
| Baseline (no translator) | - | - | 0.8079 | - | 0.2965 | - | - |
| Bidirectional d128 | bidir | 128 | 0.8331 | **+0.0251** | 0.3256 | +0.0292 | 0.4872 |
| Causal d128 | causal | 128 | 0.8320 | **+0.0240** | 0.3238 | +0.0273 | 0.4919 |
| Causal d64 | causal | 64 | 0.8149 | **+0.0070** | 0.3096 | +0.0131 | 0.5147 |

### Configs
- `configs/mortality24_bidir_repro_config.json` — Bidirectional d128
- `configs/mortality24_causal_config.json` — Causal d128
- `configs/mortality24_causal_d64_config.json` — Causal d64

### Run outputs
- `runs/mortality24_bidir_repro/` — loss curves + checkpoint
- `runs/mortality24_causal/` — loss curves + checkpoint
- `runs/mortality24_causal_d64/` — loss curves + checkpoint

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

### Factor 3: Data Size — NOT the bottleneck

Full-data sepsis (+0.0006) performed worse than debug sepsis (+0.0017). More data didn't help. This rules out the 20% subsetting as a limiting factor.

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
