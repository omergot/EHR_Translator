# Strategy Evaluation: MMD + MLM Approach

> **Role**: Historical — documents why MMD+MLM was chosen over alternatives (forecasting, distillation). Written before experiments.
> **See also**: [implementation_plan_mmd_mlm.md](implementation_plan_mmd_mlm.md) (how it was built), [experiment_results_mmd_mlm.md](experiment_results_mmd_mlm.md) (results: +0.001-0.002), [gradient_bottleneck_analysis.md](gradient_bottleneck_analysis.md) (why it didn't help enough)

## Context

The causal translator for Sepsis/AKI barely learns anything useful, while the bidirectional ("cheaty") translator achieves massive gains:

| Model | Best val_task | Test AUCROC | Test AUCPR | Delta AUCROC |
|-------|-------------|-------------|------------|--------------|
| **Bidirectional (cheaty)** | 0.2684 | **0.9569** | **0.3651** | **+0.2473** |
| Causal (d=64, full data) | 0.6752 | 0.7208 | 0.0304 | +0.0015 |
| Causal (d=64, 20% subset) | 0.6666 | 0.7199 | 0.0303 | +0.0006 |
| Causal (d=128, full data) | 0.6691 | 0.7188 | 0.0298 | -0.0005 |
| Original (no translation) | — | 0.7193 | 0.0309 | baseline |

**Root cause analysis:** The causal model's only training signal is task loss through a frozen LSTM on a 1.1% positive-rate binary classification task. This gradient is indirect (loss -> LSTM -> translator), noisy, and extremely weak. The model has almost nothing actionable to learn from.

**Why bidirectional works so well:** With bidirectional attention, delta_t (translation correction at time t) is computed using future timesteps t+1, ..., T. The LSTM then processes translated data where each input already encodes future knowledge. This is genuine information leakage — the AUCROC of 0.96 is unrealistically high for sepsis prediction.

---

## Idea 1: Pretrain Forecaster + Pseudo-Future Bidirectional

**Concept:** Train a forecaster model F to predict x[t+1:t+w] from x[0:t]. During translator training, generate pseudo-future timesteps, then use bidirectional attention over [real_past + predicted_future].

### Pros
- Conceptually clean approximation of bidirectional context
- Forecaster is a well-defined self-supervised task
- Could be a thesis contribution ("forecast-augmented domain adaptation")
- If forecast captures broad trends (patient deteriorating/improving), even rough future context helps

### Cons
- **EHR data is highly stochastic and sparse** — 48 features, most missing at any timestep. Forecasting accuracy will be poor
- **Domain mismatch** — forecaster predicts SOURCE domain futures, but the translator maps source -> target. Predicted futures are in the wrong domain
- **Error propagation** — autoregressive prediction over w steps compounds errors
- **The cheaty model's power is from *specific* future events** (e.g., "creatinine will spike at t+6"), not average trends. A smoothed forecast doesn't provide this signal
- **Computational cost** — two models, forecaster must run at every training step
- **Training complexity** — forecaster quality directly limits translator quality

### Verdict
Medium potential, high complexity. **Expected recovery: 10-25% of bidirectional gap.** The fundamental limitation is that a forecast is an averaged/smoothed version of the future, losing the specific details that made bidirectional attention so powerful.

---

## Idea 2: Latent Space Forecasting

**Concept:** Same as Idea 1, but predict hidden representations h[t+1] instead of raw x[t+1].

### Pros
- Latent space is smoother and more amenable to prediction
- Avoids predicting noisy raw sensor values
- Integrates naturally (concatenate latent forecasts to real latent states)

### Cons
- **Same fundamental limitations as Idea 1** — error propagation, domain mismatch
- **Chicken-and-egg problem** — need a trained encoder to forecast in its latent space, but the encoder is what we're training
- **Requires multi-stage training** — pretrain encoder, train forecaster, fine-tune translator
- Less interpretable — harder to verify forecast quality

### Verdict
This is a **refinement of Idea 1**, not a standalone alternative. Adds ~5-10% over raw forecasting if pursuing Idea 1. Same fundamental limitations apply.

---

## Idea 3a: Oracle Distillation from Cheaty Model

**Concept:** Use knowledge distillation — train the causal translator to minimize MSE between its output deltas and the cheaty bidirectional translator's output deltas.

```
L = alpha * L_task + beta * MSE(delta_causal, delta_cheaty.detach()) + gamma * L_fidelity
```

### Pros
- Simplest to implement (1 extra MSE term)
- No forecasting needed — sidesteps prediction problem entirely
- Strong supervision signal from a model that achieves AUCROC 0.96
- Can serve as a **diagnostic**: if it generalizes, domain-level knowledge dominates; if not, future leakage dominates

### Cons
- **Epistemically questionable for a thesis** — laundering information leakage. The cheaty model's deltas encode future information; the student learns to approximate these future-aware translations
- **Generalization risk** — the causal model can't produce future-dependent deltas at test time. It may learn noisy compromises that don't generalize
- **Reviewers will ask** "isn't the student model just learning to approximate future-aware translations?"

### What the cheaty model's deltas encode (in decreasing transferability)
- **(a) Domain statistics** (e.g., "shift creatinine by +0.3 for MIMIC") — fully transferable
- **(b) Patient-type-aware translation** (e.g., "this trajectory resembles MIMIC kidney patients") — partially transferable
- **(c) Future-leaked adjustments** (e.g., "encode that sepsis onsets at t+50") — NOT transferable

### Verdict
High potential for a **quick diagnostic experiment**, but hard to defend in a thesis if it works. **Expected recovery: 20-50% of bidirectional gap.**

---

## Idea 3b: Distribution Matching with Real MIMIC Data (CHOSEN APPROACH)

**Concept:** Instead of distilling from the cheaty model, use **real MIMIC data** as the supervision target. The translator's goal is to make eICU look like MIMIC — and we literally have MIMIC data. Add an MMD (Maximum Mean Discrepancy) loss that directly pushes translated eICU distributions toward real MIMIC distributions.

```
L = alpha * L_task + beta * L_MMD(translated_eicu, real_mimic) + gamma * L_fidelity
```

### Pros
- **No information leakage whatsoever** — MIMIC data is the target domain ground truth
- **Directly addresses the weak gradient problem** — MMD gives the translator direct, dense, per-feature feedback without routing through the frozen LSTM
- **Well-grounded in domain adaptation literature** — MMD, DAN, JAN are established methods
- **Thesis-worthy** — "Distribution-matching regularization for causal EHR domain adaptation"
- **Combines cleanly** with MLM pretraining and all other approaches
- **Goes beyond LinearRegressionTranslator** — captures conditional, temporal, and non-linear feature relationships

### Cons
- Requires loading MIMIC data in parallel during training (additional I/O and memory)
- MMD kernel bandwidth selection requires care (solved by multi-kernel MMD)
- Standard MMD treats samples as independent, losing temporal structure (solved by transition MMD)
- Marginal distribution matching alone may not be sufficient — may need multiple levels of MMD

### MMD Levels (progressive complexity)
1. **Per-timestep marginal MMD** — pool features across valid timesteps, compute multi-kernel MMD. Direct, strong signal.
2. **Transition MMD** — match distribution of (x[t+1] - x[t]) between domains. Captures temporal dynamics.
3. **Frozen LSTM hidden state MMD** — match LSTM internal representations. Most principled (matches what the baseline model "sees"), but adds computation.

### Implementation: Multi-Kernel MMD
Use sum of RBF kernels with different bandwidths (median heuristic * [0.1, 0.5, 1, 2, 10]) to capture patterns at multiple scales. Standard practice, robust, no tuning needed.

### Verdict
**Strongest single idea.** Clean, defensible, directly addresses the core problem (weak gradients), and well-supported by domain adaptation literature. **Expected to provide meaningful improvement over the current near-zero gain.**

---

## Idea 4: MLM (Masked Language Modeling) Pretraining

**Concept:** Pretrain the translator backbone by masking random timesteps (~15%) and predicting them using **bidirectional attention** (legitimate since no labels are involved). Then fine-tune with causal attention + task loss + MMD.

### Pros
- **Fully defensible** — bidirectional pretraining with self-supervised objective has no information leakage
- **Thesis-worthy contribution** — "Bidirectional pretraining improves causal domain adaptation for clinical time series"
- Learns temporal dynamics, feature correlations, patient trajectory patterns
- Better weight initialization -> faster convergence, potentially better local optima
- **Complementary** to MMD approach — can combine both
- Can pretrain on **both eICU + MIMIC data** (maximizes data, teaches model about both domains)

### Cons
- Won't fully close the causal/bidirectional gap — the causal constraint is still the bottleneck during fine-tuning
- Additional training phase (though on same data with different objective)
- Pretraining objective (reconstruct features) != downstream objective (domain adaptation)
- Pretrained bidirectional representations may partially degrade when switching to causal fine-tuning

### Key insight
During pretraining, the model learns temporal priors (e.g., "after creatinine rises, BUN typically follows"). These priors persist in the weights even under causal attention, giving the model implicit knowledge about likely futures without explicitly seeing them.

### Data split for pretraining
**No split needed.** MLM is self-supervised (no labels), so all training data can be used for both pretraining and translator training. Better yet: pretrain on **both eICU + MIMIC** data to learn temporal patterns from both domains.

### Verdict
**Most principled approach, excellent complement to MMD.** Expected to improve convergence and representation quality. **Expected recovery: 15-35% of bidirectional gap when combined with MMD.**

---

## Recommendation: Combined Approach (Idea 3b + Idea 4)

### Phase 1: MMD Domain Matching (primary contribution)
Add multi-kernel MMD loss with real MIMIC data. This directly addresses the weak gradient problem that makes the causal model fail.

### Phase 2: MLM Pretraining (complementary contribution)
Pretrain the translator backbone with bidirectional MLM on both eICU + MIMIC. This provides better initialization for the MMD fine-tuning phase.

### Phase 3 (optional): Distillation Diagnostic
Run Idea 3a as a diagnostic experiment to quantify how much of the cheaty model's knowledge is domain-level vs. future-leaking. This informs interpretation of results.

### Expected outcome
The combination should provide substantially better results than either alone. The MMD gives the translator a clear optimization target (match MIMIC distributions), while MLM pretraining gives it better temporal representations to achieve that matching.
