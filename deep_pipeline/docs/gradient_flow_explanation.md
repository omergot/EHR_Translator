# Gradient Flow in the Translator Training Loop

This document traces exactly how a weight update occurs for a single training step, using a concrete example of a sepsis patient with a 30-timestep stay (padded to 169).

## Forward Pass

### Step 1: Extract

`schema_resolver.extract()` decomposes the batch `(data, labels, mask)` + static:
- `x_val`: (B, 169, 48) — feature values
- `x_miss`: (B, 169, 48) — missingness indicators
- `m_pad`: (B, 169) — True for t=30..168 (padded), False for t=0..29 (real)
- `M_label`: (B, 169) — the label mask from YAIB (which timesteps have labels)

### Step 2: Translator Forward

The EHRTranslator processes all 169 timesteps:
1. Triplet projection (value + missingness + time-delta), sensor embeddings, temporal encoding
2. AxialBlock attention (variable-wise + temporal, causal with optional sliding window)
3. FiLM conditioning from static features
4. `delta = delta_head(h)` → (B, 169, 48)
5. `x_val_out = x_val + delta`
6. `x_val_out.masked_fill(m_pad[:, :, None], 0.0)` — deltas at t=30..168 forced to zero

The translator produces **30 real deltas** (t=0..29) and 139 zeros.

### Step 3: Rebuild

`x_val_out` is recombined with `x_miss` and `x_static` into the full YAIB tensor (B, 169, 100).

### Step 4: Frozen LSTM Forward

```python
logits = yaib_runtime.forward((x_yaib_translated, labels, label_mask))
```

The LSTM processes all 169 timesteps sequentially (it doesn't know about padding). It produces `logits` of shape (B, 169, 2) — a sepsis prediction at every timestep.

### Step 5: Task Loss

```python
prediction = masked_select(logits, label_mask)  # flatten to labeled timesteps only
target = masked_select(labels, label_mask)       # corresponding labels
loss = CrossEntropyLoss(weight=loss_weights)(prediction, target)
```

The loss is **only computed on timesteps where `label_mask` is True**. For sepsis, `label_mask` is a subset of non-padded timesteps (YAIB determines which hours get labels — not necessarily all 30). The loss is a single scalar — the average cross-entropy across all labeled timesteps in the batch.

### Step 6: Auxiliary Losses

- `l_fidelity`: MSE between `x_val_out` and `x_val`, averaged over **all** non-padded timesteps
- `l_range`: penalty for out-of-bounds features, averaged over all non-padded timesteps
- `l_total = l_task + 0.1 * l_fidelity + 0.001 * l_range`

## Backward Pass

### Step 7: `l_total.backward()`

#### Task loss → LSTM logits

The CE loss gradient ∂L_task/∂logit_t is nonzero only at labeled timesteps. Say timesteps {5, 10, 15, 20, 25, 29} have labels — the gradient is nonzero at those 6 positions and zero everywhere else.

#### LSTM logits → translated input

The LSTM computes `h_t = f(input_t, h_{t-1})`. For a labeled timestep t=25, there are two gradient paths to `input_25`:

**Direct path** (1 LSTM cell):
```
∂L/∂input_25 = (∂L/∂logit_25) × (∂logit_25/∂h_25) × (∂h_25/∂input_25)
```

**Indirect paths** (through future labeled timesteps):
```
∂L/∂input_25 += (∂L/∂logit_29) × (∂logit_29/∂h_29) × (∂h_29/∂h_28) × ... × (∂h_26/∂h_25) × (∂h_25/∂input_25)
```

The direct path is short — one LSTM cell. The indirect path from t=29 is 4 LSTM steps. For t=5 with a label at t=29, the indirect path is 24 steps — which attenuates but isn't catastrophic for an LSTM.

**Key point**: Unlike what one might assume, the gradient for the translator's output at timestep t does NOT need to flow through the entire 169-step LSTM chain. It primarily comes from the loss at t itself (direct, short path), with attenuating contributions from future labeled timesteps.

#### Translated input → deltas

`x_val_out = x_val + delta`, so `∂L/∂delta_t = ∂L/∂x_val_out_t` (identity, gradient passes through unchanged).

#### Deltas → shared transformer parameters

`delta_t = delta_head(h_t)` where `h_t` is the transformer hidden state at timestep t. The gradient for the shared transformer parameters θ is:

```
∂L_task/∂θ = Σ_{t=0..29} (∂L_task/∂delta_t) × (∂delta_t/∂θ)
```

Each of the 30 real timesteps contributes a gradient direction for the **same shared weights**. These get summed. The fidelity loss adds another 30 gradient contributions (one per non-padded timestep).

### Step 8: `optimizer.step()`

AdamW takes the accumulated gradient and updates all translator parameters.

## What This Means

The task loss gradient is nonzero only at labeled timesteps, but must update weights that affect all timesteps. In a typical sepsis batch:

- ~6 out of 30 timesteps have labels
- ~5 of those are negative (label=0) at the ~1.1% positive rate
- The task gradient is dominated by "predict negative everywhere"
- The fidelity loss (0.1 weight) acts on all 30 timesteps saying "don't change anything"
- These compete: the net update for shared transformer weights is the sum of these opposing signals

## Sliding Window Experiment Result

The W=25 sliding window experiment (limiting temporal attention to 25 positions) tested whether the attention span over 169 positions was the bottleneck:

| Experiment | val_task (best) | Test AUCROC delta |
|---|---|---|
| Baseline (full causal) | 0.6801 | +0.0030 |
| W=25 sliding window | 0.6732 | +0.0022 |

The window made the model learn **faster and more aggressively** (val_task dropped further), but test AUCROC didn't improve. This means:

- **Ruled out**: attention over too many positions hurting the transformer's representations
- **Consistent with**: the model can overfit the training distribution but struggles to learn generalizable translations, likely due to the sparse positive labels and the competing loss signals described above
