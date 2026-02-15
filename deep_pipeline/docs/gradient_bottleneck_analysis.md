# Gradient Bottleneck Analysis and Next Steps

> **Role**: Main findings document — start here for current project status.
> **See also**: [architecture.md](architecture.md) (model reference), [gradient_flow_mechanics.md](gradient_flow_mechanics.md) (how gradients flow), [investigation_mortality_vs_sepsis.md](investigation_mortality_vs_sepsis.md) (controlled factor isolation experiments)

## Summary of All Experiments

### Results Table (Sepsis, debug=20%, d_model=64 unless noted)

| Experiment | Config | AUCROC Delta | AUCPR Delta | Best val_task | Early Stop |
|---|---|---|---|---|---|
| A: Baseline (30ep) | task+fid+range | +0.0017 | -0.0004 | 0.6764 | ep 17 |
| B: MMD only | lambda_mmd=1.0 | +0.0013 | -0.0002 | ~0.67 | ep 17 |
| C: MMD+Trans | mmd+transition | +0.0013 | -0.0006 | 0.6700 | ep 17 |
| D: MLM only | 10ep pretrain | -0.0005 | -0.0006 | 0.6683 | ep 18 |
| E: MLM+MMD | pretrain+mmd | +0.0016 | -0.0001 | 0.6697 | ep 20 |
| W=25 baseline | window=25 | +0.0022 | - | 0.6732 | - |
| **Oversample f=3** | W=25, f=3 | -0.0001 | -0.0014 | 0.6556 | ep 18 |
| **Oversample f=10** | W=25, f=10 | +0.0047 | -0.0008 | - | ~ep 13 |
| **Oversample f=20** | W=25, f=20 | **+0.0059** | +0.0003 | - | ~ep 8 |
| Oversample f=20, d128 | W=25, f=20 | +0.0019 | -0.0008 | 0.6975 | ep 8 |
| **f=20 grad diag** | W=25, f=20, fid=0.1 | **+0.0059** | +0.0003 | 0.6977 | ep 7 |
| **f=20 no-fidelity** | W=25, f=20, fid=0.0 | **-0.1013** | -0.0126 | 1.1141 | ep 6 |

### Control Experiments (Mortality vs Sepsis)

| Experiment | Task | Attention | d_model | AUCROC Delta | AUCPR Delta |
|---|---|---|---|---|---|
| **Mortality full 30ep** | Mortality24 | bidirectional | 128 | **+0.0264** | **+0.0296** |
| Mortality bidir d128 (prev) | Mortality24 | bidirectional | 128 | +0.0251 | - |
| Mortality causal d128 | Mortality24 | causal | 128 | +0.0240 | - |
| Mortality causal d64 | Mortality24 | causal | 64 | +0.0070 | - |
| Sepsis causal d128 full | Sepsis | causal | 128 | +0.0006 | - |
| Sepsis causal d64 debug | Sepsis | causal | 64 | +0.0017 | - |

Mortality full 30ep: trained 28 epochs (early stopped at patience=5, best at epoch 23), val_task 0.5303→0.4866.

### What We've Ruled Out

| Hypothesis | Evidence Against |
|---|---|
| Causal vs bidirectional attention | Mortality causal (+0.024) matches bidirectional (+0.025) |
| Insufficient model capacity | Sepsis d128 (+0.001) worse than d64 (+0.002); mortality d64 (+0.007) still 3.5x better than sepsis d64 |
| Not enough training data | Full-data sepsis (+0.0006) worse than 20%-debug (+0.0017) |
| Attention span too long | W=25 window (+0.0022) similar to full attention (+0.0017) |
| Wrong loss function | MMD, MLM, fidelity tuning all give +0.001-0.002 |
| Not enough positive exposure | Oversampling f=20 helped (+0.006), confirming gradient frequency matters |

## Gradient Diagnostic Results (Confirmed)

### Experiment: Gradient Magnitude Logging (f=20, lambda_fidelity=0.1)

Measured per-component gradient L2 norms at the translator parameters for the first 4 batches of epoch 1:

| Batch | Task Grad | Fidelity Grad | Range Grad | Fid/Task Ratio |
|---|---|---|---|---|
| 0 | 0.718 | **6.996** | 0.004 | **9.74x** |
| 1 | 0.757 | **4.974** | 0.002 | **6.57x** |
| 2 | 0.995 | **3.371** | 0.000 | **3.39x** |
| 3 | 0.738 | **3.919** | 0.000 | **5.31x** |

**Finding: Fidelity gradient is 3-10x larger than task gradient.** The translator overwhelmingly learns "don't change anything" rather than "improve predictions." Range gradient is negligible (~0.001x task).

### Experiment: Zero Fidelity (f=20, lambda_fidelity=0.0)

Removing fidelity entirely caused the translator to **diverge catastrophically**:

| Epoch | val_task | val_fidelity (unweighted) | val_range |
|---|---|---|---|
| 1 | 1.1141 | **317.6** | 13.99 |
| 2 | 1.2173 | 263.7 | 9.43 |
| 3 | 1.2891 | 171.8 | 5.23 |
| 4 | 1.2466 | 100.5 | 2.22 |
| 5 | 1.3302 | 107.3 | 2.19 |
| 6 | 1.1998 | 74.6 | 1.17 |

**Result: AUCROC -0.1013** (0.7193 → 0.6181). Without fidelity, the translator produces huge deltas (fidelity=317 at epoch 1 vs ~0.86 with fidelity), destroying the input data. The task loss also degrades (1.11+ vs 0.70) — the translated data is so distorted that even the task predictions worsen.

### Interpretation

The gradient diagnostic reveals a **Goldilocks problem**:
- **lambda_fidelity=0.1**: fidelity gradient 6x too strong, suppresses task learning → +0.006 AUCROC
- **lambda_fidelity=0.0**: no regularization, translator diverges → -0.101 AUCROC
- **The optimal lambda_fidelity is somewhere between 0 and 0.1**, but simply tuning it won't solve the fundamental issue: the task gradient is inherently weak and noisy

This confirms that the bottleneck is not "fidelity is too strong" but rather "**the task gradient doesn't carry enough information to guide meaningful translations.**" Even without competing fidelity gradient, the task signal alone sends the translator off a cliff — it changes features aggressively in random directions because the per-timestep classification signal is too sparse to indicate *how* to change them.

## The Gradient Bottleneck: A Precise Explanation

### The Data

Sepsis training set: **86,387 stays** with **4,497,315 total timestep-rows**.

- Average stay length: **52 timesteps** (range varies; padded to 169 in batches)
- Per-stay positive rate: **4.57%** (3,945 positive stays out of 86,387)
- Per-timestep positive rate: **1.13%** (50,840 positive timesteps out of 4,497,315)
- Class weight applied: **44.2x** for positive class

### Where the Gradient Comes From

The translator receives gradient from two competing sources:

**1. Task loss** (weight 1.0) — computed only on `M_label` timesteps (the labeled subset of non-padded timesteps):

```
logits = frozen_LSTM(translated_input)        # (B, T, 2)
prediction = logits[M_label]                  # select labeled timesteps
loss = CrossEntropy(prediction, labels, weight=[0.51, 44.2])
```

The gradient flows backward: `loss → logits → LSTM hidden states → LSTM inputs → translator output → translator parameters`. The frozen LSTM is differentiable (just not updated), so gradients pass through it.

**2. Fidelity loss** (weight 0.1) — computed on all non-padded timesteps:

```
diff = (translator_output - original_input) ** 2   # (B, T, 48)
loss = mean(sum_features(diff), over non-padded timesteps)
```

This gradient is simple: it pushes every translator output toward its corresponding input. It directly opposes the task loss.

### Why the Net Signal is Near-Zero

Consider a typical training batch (batch_size=64):

**Step 1: How many labeled timesteps?**
- 64 stays x ~52 real timesteps = ~3,328 non-padded timesteps per batch
- Not all timesteps have labels (label mask is a subset) — but assume most do for sepsis
- Of those, ~1.1% are positive = ~37 positive timesteps
- With class weight 44.2x, the positive gradient is amplified, but still represents a small fraction of the total gradient mass

**Step 2: What does the task gradient say?**
- At each positive timestep: "change the input so the LSTM predicts higher sepsis probability"
- At each negative timestep (~3,291): "change the input so the LSTM predicts lower sepsis probability"
- The negative gradient dominates by ~90:1 in raw count (partially offset by 44.2x weighting)
- But critically: **the translator doesn't know which direction to change features** — the LSTM is a black box, and the gradient through 161-dimensional hidden states is noisy

**Step 3: What does the fidelity gradient say?**
- At all ~3,328 timesteps, across all 48 features: "don't change anything"
- This is a **dense, clear signal** that opposes the sparse, noisy task signal
- With lambda_fidelity=0.1, fidelity contributes 0.1x the gradient of a same-magnitude task loss

**Step 4: The net update**

The optimizer step combines:
- A noisy, sparse task gradient that says "change *something* at ~37 positive timesteps" (but disagrees with the ~3,291 negative timesteps about *what* to change)
- A clear, dense fidelity gradient that says "change nothing" at all ~3,328 timesteps
- A range gradient that activates only for out-of-bounds values (often near-zero)

**Result: the net parameter update is dominated by fidelity (regularize toward identity) with tiny, noisy perturbations from task loss.** The translator learns to make extremely small deltas that barely move the LSTM's predictions.

### Why Mortality Works and Sepsis Doesn't

Mortality24 has fundamentally different gradient properties:

| Property | Sepsis | Mortality24 |
|---|---|---|
| Label type | Per-timestep | Per-stay (one label) |
| Sequence length | ~52 real (padded to 169) | ~25 timesteps |
| Positive rate | 1.13% per-timestep, 4.57% per-stay | ~12% per-stay |
| Label density | Sparse: most timesteps labeled, but signal diffuse | Dense: one label, entire stay contributes |
| Gradient coherence | Each timestep gets a tiny gradient; they may disagree | All timesteps get gradient from one clear signal |

For mortality, the single per-stay label means the LSTM's final hidden state (which aggregates the full sequence) receives one strong gradient. This propagates backward coherently to all timesteps. The translator gets a consistent signal: "transform the entire stay so the final prediction changes."

For sepsis, each timestep gets its own tiny gradient. Neighboring timesteps may have opposite labels (rare but possible), and the overwhelming majority (98.9%) say "predict negative." The translator receives contradictory micro-signals that largely cancel out.

### Why Oversampling Helped

Oversampling f=20 increased the effective positive rate from 4.57% to 48.8% of stays. This means:
- ~31 positive stays per batch instead of ~3
- More diverse positive examples → less gradient noise
- Task gradient for "improve positive prediction" is 10x denser
- But the per-timestep imbalance within each positive stay is unchanged (still ~1.1% of timesteps are positive)

This explains why f=20 helped (+0.006) but didn't solve the problem — it addressed stay-level frequency but not timestep-level signal diffuseness.

## Verifying the Gradient Bottleneck Hypothesis

All three diagnostics are implemented in `src/core/train.py` `_run_epoch()`. They run automatically on the first 4 batches of epoch 0 and log with prefixes `[grad-diag]` and `[grad-ts]`.

### Diagnostic 1: Gradient Magnitude Logging -- IMPLEMENTED, CONFIRMED

Measures per-component gradient L2 norms at the translator parameters. Logs as `[grad-diag]`.

**Results (f=20, lambda_fidelity=0.1, W=25):**

| Batch | Task Grad | Fidelity Grad | Range Grad | Fid/Task Ratio |
|---|---|---|---|---|
| 0 | 0.718 | **6.996** | 0.004 | **9.74x** |
| 1 | 0.757 | **4.974** | 0.002 | **6.57x** |
| 2 | 0.995 | **3.371** | 0.000 | **3.39x** |
| 3 | 0.738 | **3.919** | 0.000 | **5.31x** |

**Conclusion:** Fidelity gradient overwhelms task by 3-10x. Range gradient negligible.

### Diagnostic 2: Zero-Fidelity Experiment -- IMPLEMENTED, CONFIRMED

Ran f=20 with `lambda_fidelity=0.0`. Translator diverged catastrophically:
- **AUCROC: -0.1013** (0.7193 → 0.6181)
- Fidelity (unweighted) reached 317 at epoch 1 (vs ~0.86 normally)
- Task loss also degraded (1.11 vs 0.70)

**Conclusion:** Task gradient alone is too noisy to guide useful translations. Fidelity is necessary but too strong at 0.1.

### Diagnostic 3: Per-Timestep Gradient Analysis -- IMPLEMENTED

Measures L2 norm of task gradient at the translator output (`x_val_out`) broken down by:
- **Positive labeled timesteps** (y >= 1 & M_label)
- **Negative labeled timesteps** (y < 1 & M_label)
- **Unlabeled non-padded timesteps** (~M_label & ~M_pad)

Also logs counts (n_pos, n_neg, n_unlabeled, n_pad) to reveal batch composition. Logs as `[grad-ts]`.

**Results — Sepsis W=25 f=20:**

| Batch | pos_norm | neg_norm | ratio_pos/neg | n_pos | n_neg | n_unlabeled | n_pad |
|---|---|---|---|---|---|---|---|
| 0 | 0.001402 | 0.000398 | 3.52 | 403 | 2486 | 0 | 7927 |
| 1 | 0.001301 | 0.000341 | 3.81 | 390 | 2519 | 0 | 7907 |
| 2 | 0.001903 | 0.000356 | 5.34 | 317 | 2470 | 0 | 8029 |
| 3 | 0.001474 | 0.000403 | 3.66 | 411 | 3013 | 0 | 7392 |

**Results — Sepsis W=6 f=20:**

| Batch | pos_norm | neg_norm | ratio_pos/neg | n_pos | n_neg | n_unlabeled | n_pad |
|---|---|---|---|---|---|---|---|
| 0 | 0.001244 | 0.000361 | 3.44 | 324 | 2571 | 0 | 7921 |
| 1 | 0.001354 | 0.000351 | 3.86 | 362 | 2595 | 0 | 7859 |
| 2 | 0.001375 | 0.000348 | 3.95 | 423 | 2710 | 0 | 7683 |
| 3 | 0.001460 | 0.000399 | 3.66 | 441 | 2684 | 0 | 7691 |

**Results — Mortality full 30ep (bidirectional, d_model=128):**

| Batch | pos_norm | neg_norm | ratio_pos/neg | n_pos | n_neg | n_unlabeled | n_pad |
|---|---|---|---|---|---|---|---|
| 0 | 0.013095 | 0.001032 | **12.69** | 2 | 62 | 1536 | 0 |
| 1 | 0.039971 | 0.001088 | **36.74** | 2 | 62 | 1536 | 0 |
| 2 | 0.012419 | 0.000943 | **13.17** | 2 | 62 | 1536 | 0 |
| 3 | 0.004092 | 0.001109 | **3.69** | 1 | 63 | 1536 | 0 |

**Mortality Gradient Magnitude (epoch 0):**

| Batch | Task Grad | Fidelity Grad | Range Grad | Fid/Task Ratio |
|---|---|---|---|---|
| 0 | 2.397 | 6.728 | 0.398 | **2.81x** |
| 1 | 2.116 | 6.385 | 0.377 | **3.02x** |
| 2 | 1.314 | 5.468 | 0.549 | **4.16x** |
| 3 | 2.825 | 6.397 | 3.348 | **2.26x** |

### Comparative Analysis: Sepsis vs Mortality Per-Timestep Gradients

| Metric | Sepsis (W=25) | Mortality | Interpretation |
|---|---|---|---|
| Task grad norm | 0.7-1.0 | **1.3-2.8** | Mortality task gradient 2-3x stronger |
| Fid/task ratio | 3.2-9.8x | **2.3-4.2x** | Fidelity dominates less in mortality |
| pos_norm | ~0.0015 | **~0.017** | Positive gradient 11x stronger per-timestep |
| neg_norm | ~0.0004 | **~0.001** | Negative gradient 2.5x stronger |
| pos/neg ratio | 3.5-5.4x | **3.7-36.7x** | Class weight much more effective in mortality |
| n_pos per batch | ~380 | ~2 | Sepsis has more positive timesteps but diffuse |
| n_neg per batch | ~2700 | ~62 | Sepsis overwhelmed by negatives |
| n_unlabeled | **0** | **1536** | Mortality: most timesteps are "context" for the per-stay label |
| n_pad | **~7900** | **0** | Sepsis: 73% padding waste; Mortality: no padding |
| AUCROC delta | +0.0059 | **+0.0264** | 4.5x better performance |

**Key Insights:**

1. **Mortality has NO padding** (fixed 24-timestep sequences) — all computation contributes to learning. Sepsis wastes ~73% of computation on zero-padded timesteps.

2. **Mortality has per-stay labels** — only 64 timesteps per batch (1 per stay) receive labels; the other 1536 are "unlabeled context" that receives gradient only through LSTM temporal propagation. This concentrates the label signal.

3. **The LSTM amplifies positive gradient contrast in mortality** — with only 1-2 positive stays per batch and per-stay labels, the 44x class weight creates 13-37x pos/neg gradient ratio (vs only 3.5-5.4x in sepsis). The LSTM's final hidden state aggregates the full sequence, so the gradient propagates coherently to all timesteps.

4. **Sepsis per-timestep labels diffuse the gradient** — 380+ positive timesteps each get a tiny gradient that the LSTM attenuates as it propagates. The net effect: despite 44x class weight, the pos/neg ratio at the translator output is only 3.5-5.4x. The gradient signal is spread thin.

5. **Fidelity dominance is the limiting factor** — even with oversampling and class weighting, the fidelity gradient (3-10x for sepsis, 2-4x for mortality) remains the dominant signal. Mortality succeeds because its task gradient is inherently 2-3x stronger AND the fidelity ratio is lower.

## Recommended Next Steps (Ranked by Expected Impact)

### Tier 1: Quick Wins (< 1 hour each)

**1a. Zero-fidelity baseline** (config change only)
- Set `lambda_fidelity=0.0` with f=20, W=25
- Tests whether removing the opposing gradient signal helps
- Expected: +0.007-0.010 if gradient bottleneck theory is correct

**1b. Gradient magnitude logging** (diagnostic)
- Add the logging code from Diagnostic 1 above
- Run one epoch with current best config (f=20)
- Provides concrete numbers on the gradient competition

### Tier 2: Medium Effort (half day each)

**2a. Hidden-state MMD**
- Instead of matching raw features (input-space MMD), match the frozen LSTM's hidden states between translated-eICU and real-MIMIC data
- This provides dense gradient at every timestep, in a task-relevant 161-dimensional space
- The LSTM already compresses temporal context, so alignment happens where it matters
- Implementation: extract `h_t` from LSTM after forward pass on both domains, compute MMD on those

```
translated_eICU → frozen_LSTM → h_source  (B, T, 161)
real_MIMIC      → frozen_LSTM → h_target  (B, T, 161)
L_hidden_mmd = MMD(h_source[~pad], h_target[~pad])
```

- This is complementary to oversampling: oversampling gives denser task gradient, hidden-MMD gives dense domain-alignment gradient
- Expected: +0.008-0.015 combined with f=20

**2b. Focal loss for task component**
- Replace standard cross-entropy with focal loss (gamma=2)
- Down-weights easy negatives, concentrates gradient on hard examples near the decision boundary
- These hard examples are exactly where the translator's changes matter most
- Complementary to oversampling (frequency vs quality of gradient)

### Tier 3: Higher Effort (1-2 days)

**3a. Adversarial domain discriminator**
- Train a small discriminator: "is this LSTM hidden state from translated-eICU or real-MIMIC?"
- Translator tries to fool it (gradient reversal layer)
- Unlike fixed-kernel MMD, the discriminator *learns* what features matter to match
- Provides dense, adaptive gradient signal at every timestep
- Classic DANN approach, well-established in domain adaptation literature

**3b. CORAL (Correlation Alignment)**
- Align second-order statistics (covariance) of LSTM hidden states between domains
- Simpler than adversarial, no training instability
- Captures cross-feature correlations that per-feature MMD misses
- `L_coral = ||Cov(h_source) - Cov(h_target)||_F^2 / (4 * d^2)`

### Tier 4: Exploratory

**4a. Task-specific gradient weighting**
- Weight the task loss contribution of each timestep by how "informative" it is
- E.g., higher weight for timesteps near sepsis onset, lower for stable periods
- Could use the baseline model's prediction entropy as a proxy for informativeness

**4b. Progressive training**
- Start with high lambda_fidelity (learn identity), gradually reduce to 0
- This gives the translator a curriculum: first learn to preserve data, then learn to improve it
- Avoids early divergence while eventually removing the opposing gradient

## Why Hidden-State MMD is the Most Promising Next Step

The current input-space MMD (which didn't help) aligns raw features: "make eICU heart rate distribution match MIMIC heart rate distribution." But this is indirect — the frozen LSTM may not care about raw heart rate distributions. It cares about its own internal representations.

Hidden-state MMD aligns what the LSTM actually sees: "make the LSTM's internal processing of translated eICU look like its processing of real MIMIC." This is:

1. **Dense**: gradient at every non-padded timestep (not just labeled ones)
2. **Task-relevant**: the LSTM hidden state is exactly what feeds into the classification head
3. **Complementary to task loss**: task loss says "improve predictions," hidden-MMD says "make representations similar" — these don't conflict like task vs fidelity
4. **No opposing signal**: unlike fidelity loss, hidden-MMD doesn't push toward identity — it pushes toward MIMIC-like representations
5. **Already implemented infrastructure**: the MMD utility and target data loading are already in place; only the extraction point changes (LSTM hidden states instead of raw features)
