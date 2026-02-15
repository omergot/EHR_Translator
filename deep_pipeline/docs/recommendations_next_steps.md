# Recommendations and Next Steps

> **Role**: Forward-looking recommendations doc. Combines literature review, codebase analysis, and lessons from all experiments to date.
> **See also**: [gradient_bottleneck_analysis.md](gradient_bottleneck_analysis.md) (current status), [architecture.md](architecture.md) (model reference), [investigation_mortality_vs_sepsis.md](investigation_mortality_vs_sepsis.md) (what we've ruled out)

## Context: Where We Are

Current results are far from the goal: mortality shows +0.026 AUCROC and sepsis only +0.006, both insufficient for a meaningful domain adaptation contribution. The root cause is a gradient bottleneck: fidelity gradient is 3-10x stronger than task gradient, and the task signal itself is sparse (1.1% positive rate per-timestep, 73% padding waste). We've ruled out attention mode, model capacity, data size, and attention span. Oversampling (f=20) helped by 3x, confirming gradient density matters. Substantial architectural and training changes are needed to achieve clinically meaningful improvements.

---

## A. Input Shaping and Sequence Processing

The sepsis data has severe structural inefficiency: sequences padded to 169 timesteps with ~73% being padding zeros. This wastes computation and dilutes gradients. Mortality has zero padding (fixed 24-step sequences) and works. Input shaping changes can recover significant signal without changing the model.

### A1. Variable-Length Batching (Eliminate Padding)

**Problem**: Every sepsis batch pads to the longest sequence. With max_length=169 but mean=52, 73% of every batch is zeros that consume memory, attention computation, and produce zero gradients.

**Solution**: Sort stays by length and group into buckets. Each batch pads only to its own maximum, not the global maximum. A batch of 64 stays with lengths 40-55 pads to 55 instead of 169.

**Implementation**: Use a `BucketBatchSampler` that sorts by sequence length and groups similar-length stays:

```python
# In cli.py, replace standard DataLoader batching
from torch.utils.data import Sampler

class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [sorted_indices[i:i+batch_size]
                        for i in range(0, len(sorted_indices), batch_size)]
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        yield from self.batches
```

**Expected impact**: 2-3x more gradient-producing timesteps per batch. Same compute budget, denser signal. Free performance gain.

### A2. Sequence Chunking with Sliding Windows

**Problem**: Even with variable-length batching, long stays (100+ timesteps) produce diffuse gradients because the translator must get every timestep right. Mortality works partly because its sequences are only 25 steps long.

**Solution**: Chunk long sequences into overlapping windows during training. A 100-timestep stay becomes 4 overlapping windows of 30 timesteps each. Each window is an independent training sample with its own labels.

**Key details**:
- Window size W=25-30 (matching mortality's effective length)
- Overlap of 5-10 timesteps for context continuity
- Labels carried per-window (only the non-overlapping portion counts for loss)
- At inference time, run full sequences (no chunking)

**Why this is different from `temporal_attention_window`**: The existing W=25 window limits the *attention span* but still processes the full 169-timestep padded sequence. Chunking actually splits the data into shorter sequences, eliminating padding entirely and reducing the problem to mortality-like scale.

**Expected impact**: Transforms the sepsis problem structurally to resemble mortality (short sequences, no padding, denser labels per window). This directly addresses the primary bottleneck identified in the investigation.

### A3. Padding-Aware Fidelity Loss

**Problem**: Fidelity loss is computed on all non-padded timesteps equally, but the task loss only gets gradient at labeled timesteps. This asymmetry makes fidelity dominate everywhere.

**Solution**: Weight fidelity loss per-timestep based on label proximity. Timesteps near positive labels get lower fidelity weight (allowing the translator more freedom to modify them), while timesteps far from any label get higher fidelity weight (don't change what doesn't matter).

```python
# Compute distance-to-nearest-positive-label for each timestep
# Use as inverse weight for fidelity: allow more change near positive events
pos_mask = (labels >= 1) & M_label
# ... compute temporal distance per timestep to nearest positive ...
fidelity_weight = 1.0 / (1.0 + alpha * proximity_score)
l_fidelity = masked_mean(fidelity_weight * diff.sum(dim=-1), ~M_pad)
```

This lets the translator make larger changes exactly where the task loss provides signal, and enforces identity where there's no task information. It directly reduces the gradient conflict at the timesteps that matter most.

### A4. Truncate-and-Pack Instead of Pad

**Problem**: YAIB's dataloader pads all sequences to max length. For sepsis with max=169 but median=52, this is extremely wasteful.

**Solution**: Set a hard `max_seq_len` (e.g., 72 = median + 1 std). Truncate longer stays from the *beginning* (keep recent history, which is most predictive for sepsis onset). Pack remaining sequences with minimal padding.

**Tradeoff**: Loses early history for long stays, but the frozen LSTM was trained on the same data and most of its signal comes from recent timesteps anyway. The gradient analysis shows that even full-length sequences produce near-zero learning signal -- better to have dense short sequences.

---

## B. Latent Space Alignment and Domain Bridging

The core thesis idea: eICU and MIMIC represent the same clinical reality (same patients, same diseases, same physiology) observed through different measurement systems. In some latent space, they should be identical. The translator's job is to find and exploit this shared latent space.

### B1. Model-Agnostic Hidden Representation Alignment

**Problem**: The gradient bottleneck analysis recommends "hidden-state MMD" on LSTM hidden states. But the baseline model could be GRU, TCN, or Transformer (YAIB supports all four: `LSTMNet`, `GRUNet`, `TemporalConvNet`, `Transformer`). We need a model-agnostic way to extract representations.

**Solution**: Use PyTorch forward hooks to extract the *penultimate layer activations* from any baseline model. The penultimate layer (just before the classification head) is the model's learned representation space regardless of architecture. This is the standard approach in transfer learning and domain adaptation.

```python
class BaselineRepresentationExtractor:
    def __init__(self, model):
        self.model = model
        self.representations = None
        # Register hook on the last layer before classification head
        self._find_and_hook_penultimate(model)

    def _find_and_hook_penultimate(self, model):
        """Walk the model to find the layer feeding into the final Linear."""
        layers = list(model.children())
        # YAIB models: sequential backbone → linear head
        # Hook the backbone's output
        target = layers[-2] if len(layers) > 1 else layers[-1]
        target.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.representations = output
```

For YAIB specifically, all DL models follow the pattern `backbone → classification_head`. The backbone output is `(B, T, hidden_dim)` for all architectures. We hook after the backbone and before the head.

**Why this works for any architecture**:
- LSTM: `h_t` at each timestep = backbone output
- GRU: same as LSTM
- TCN: temporal convolution output at each timestep
- Transformer: encoder output at each timestep

All produce `(B, T, D)` representations that feed into the same classification head. Aligning these representations is equivalent to saying "make the baseline model *see* the translated eICU data the same way it sees MIMIC data."

### B2. Shared Encoder Latent Space (Train a Domain-Invariant Encoder)

**Idea**: Instead of aligning through the frozen baseline's representations, train a separate encoder that maps both domains into a shared latent space. This encoder learns what's common between eICU and MIMIC.

**Architecture**:
```
eICU features  ──→ [Shared Encoder] ──→ z_eICU  ──→ shared latent space
MIMIC features ──→ [Shared Encoder] ──→ z_MIMIC ──→ shared latent space
```

Train the shared encoder with:
1. **Reconstruction loss**: decode z back to features (autoencoder)
2. **Domain confusion loss**: a discriminator cannot tell z_eICU from z_MIMIC
3. **Clinical preservation loss**: task predictions from z match original predictions

The translator then operates as: eICU → shared latent → translate in latent space → decode to MIMIC-like features.

**Key advantage**: The latent space is *designed* to be domain-invariant, not just incidentally so. The encoder learns to strip away measurement-system artifacts while preserving clinical content.

### B3. Nearest-Neighbor Translation in Latent Space

**Core idea** (from the user): Once eICU and MIMIC samples are in a shared latent space, translate each eICU sample based on its *position among MIMIC samples* in that space. Find the nearest MIMIC neighbors and interpolate.

**Detailed approach**:

**Step 1: Build the shared latent space.** Use any method from B1/B2 to embed both domains. The frozen baseline's penultimate representations are the simplest starting point (no additional training needed).

**Step 2: For each eICU sample, find its k nearest MIMIC neighbors in latent space.** This identifies which MIMIC patients are "clinically similar" to this eICU patient.

**Step 3: Compute a translation vector from the neighborhood.** The key insight: the translation should map the eICU sample toward the local MIMIC manifold, not toward the MIMIC centroid. Three options:

**(a) Weighted Average Offset:**
```
For eICU sample x_e with latent embedding z_e:
  Find k nearest MIMIC neighbors: {z_m1, z_m2, ..., z_mk} with features {x_m1, ..., x_mk}
  Weights w_i = softmax(-||z_e - z_mi|| / temperature)
  Translation target = Σ w_i * x_mi
  Delta = translation_target - x_e
```
This is essentially kernel regression in latent space. The translator learns to do this parametrically.

**(b) Barycentric Interpolation:**
```
Express z_e as a convex combination of its k nearest MIMIC neighbors:
  z_e ≈ Σ α_i * z_mi  (solve for α_i via least squares with α_i >= 0, Σα_i = 1)
  Then: translated_x = Σ α_i * x_mi
```
This respects the local geometry of the MIMIC manifold. If the eICU patient sits between two MIMIC patient types, the translation blends their feature profiles.

**(c) Train the Transformer to Learn This Mapping:**
Rather than computing nearest neighbors at inference, use the nearest-neighbor translation as a *training signal*. The transformer learns to approximate the kNN mapping:

```
L_neighbor = MSE(translator(x_eICU), kNN_translation(x_eICU))
```

This combines the interpretability of kNN with the generalization of a neural network. The translator learns the local structure of the cross-domain mapping without needing to do kNN at test time.

**Why this is principled**: It exploits the assumption that the same patient in different hospitals would map to the same latent location. The kNN lookup finds "the same kind of patient in MIMIC" and uses their actual features as the translation target. This provides a dense, per-sample supervision signal that doesn't depend on task labels at all.

**Practical considerations**:
- Precompute MIMIC embeddings and build a FAISS index for fast kNN lookup
- k=5-10 is typical; temperature controls how local vs. global the interpolation is
- This can replace or supplement fidelity loss (it's a smarter version of "don't change too much")

### B4. Contrastive Domain Alignment

**Idea**: Use contrastive learning (InfoNCE) to pull together eICU-MIMIC pairs that represent similar clinical states, and push apart dissimilar pairs.

**Pair construction**: Match eICU and MIMIC patients by clinical similarity (same diagnosis, similar APACHE/SOFA scores, similar demographics). These "pseudo-pairs" define the positive set. This is feasible because both datasets include rich metadata.

```
L_contrastive = -log(exp(sim(z_eICU, z_MIMIC_pos) / τ) /
                     Σ exp(sim(z_eICU, z_MIMIC_neg) / τ))
```

**Advantage over MMD**: MMD aligns entire distributions without regard to which samples should match. Contrastive alignment respects the fine structure: a young trauma patient in eICU should map near young trauma patients in MIMIC, not near elderly heart failure patients.

### B5. Optimal Transport for Sample-Level Alignment

**Idea**: Use optimal transport (OT) to compute the minimum-cost mapping between eICU and MIMIC sample distributions. OT naturally provides per-sample correspondences.

**Implementation** with the POT library (`pip install POT`):

```python
import ot

# Compute cost matrix between eICU and MIMIC embeddings
C = ot.dist(z_eicu, z_mimic)  # (n_eicu, n_mimic)

# Sinkhorn divergence (differentiable, smooth gradients)
loss = ot.sinkhorn2(a, b, C, reg=0.1)
```

**Advantages over MMD**:
- Provides a *transport plan* T[i,j] = how much of eICU sample i should map to MIMIC sample j
- Smoother gradients than MMD (entropic regularization)
- Can handle unbalanced distributions (different class rates between domains) via unbalanced OT
- The transport plan itself is interpretable: it shows which eICU patients correspond to which MIMIC patients

**Connection to B3**: The OT transport plan is a principled version of the nearest-neighbor idea. Instead of hard kNN, OT finds the globally optimal soft assignment between samples.

### B6. Domain-Adversarial Training on Baseline Representations

**Idea**: Classic DANN (Domain-Adversarial Neural Network) applied to the baseline model's representations. A discriminator tries to distinguish translated-eICU from MIMIC; the translator tries to fool it.

**Architecture**:
```
translated_eICU ──→ [Frozen Baseline] ──→ h_source ──→ [Discriminator] ──→ "source"
real_MIMIC      ──→ [Frozen Baseline] ──→ h_target ──→ [Discriminator] ──→ "target"
                                                            ↑
                                              gradient reversal layer
```

**Key difference from MMD**: The discriminator *learns* which aspects of the representation space differ between domains, and focuses the alignment signal there. MMD uses fixed kernels that may not capture the relevant differences.

**Stability considerations**: Use Wasserstein distance with gradient penalty instead of vanilla GAN loss (more stable, no mode collapse). AdaDiag (PMC 2022) validated this exact setup for clinical EHR domain adaptation with lambda=0.2.

---

## C. Training Signal Improvements

### C1. Focal Loss (Replace BCE Task Loss)

**Problem**: Standard cross-entropy treats all samples equally. With 1.1% positive rate, the gradient is dominated by easy negatives that are already correctly classified. The task gradient is not just sparse, it's *low quality*.

**Solution**: Focal loss `FL = -α(1-pt)^γ · log(pt)` down-weights easy examples and focuses gradient on hard examples near the decision boundary.

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal).mean()
```

Start with `gamma=2.0, alpha=0.75`. This is compatible with oversampling: oversampling increases *frequency* of positive examples, focal loss increases *quality* of the gradient per example. They're complementary.

**Expected impact**: Denser effective task gradient. The 3-10x fidelity/task ratio should decrease because the task gradient concentrates on the most informative samples rather than averaging over 99% easy negatives.

### C2. GradNorm Dynamic Loss Weighting

**Problem**: Fixed lambda values (fidelity=0.1, range=1e-3, mmd=1.0) require manual tuning, and the gradient diagnostic shows the balance shifts during training. What's right at epoch 0 isn't right at epoch 20.

**Solution**: Dynamically adjust loss weights each step to equalize gradient magnitudes across components. The existing gradient diagnostic code already computes per-component norms -- make it a training mechanism:

```python
# Each step, compute gradient norms per component
# Adjust weights inversely to normalize:
target_ratio = 1.0  # equal contribution
for component in [fidelity, range, mmd]:
    actual_ratio = grad_norm[component] / grad_norm[task]
    component.weight *= (target_ratio / actual_ratio) ** alpha  # alpha=0.5 for smoothness
```

This eliminates the lambda search entirely and auto-adapts as the translator's output distribution changes during training.

### C3. Cosine Similarity Fidelity

**Problem**: MSE fidelity produces gradients proportional to absolute error. Features with large scales (e.g., heart rate ~80) dominate over features with small scales (e.g., lactate ~1.5), creating an uneven gradient landscape.

**Solution**: Replace MSE with cosine similarity:

```python
l_fidelity = 1.0 - F.cosine_similarity(x_val_out, x_val, dim=-1).mean()
```

Cosine similarity is scale-invariant: it only measures whether the *direction* of the feature vector changed. This produces more balanced gradients across features with different magnitudes. DynaGraph (Nature npj Digital Medicine, 2025) validated this substitution for clinical EHR multi-loss training.

---

## D. Evaluation and Diagnostics

### D1. Calibration Metrics — **Implemented**

> **Status**: Implemented in `src/core/eval.py`. Brier score and ECE are computed in all evaluator paths (`TranslatorEvaluator`, `TransformerTranslatorEvaluator`). Results appear automatically in evaluation logs.

**Problem**: AUROC measures discrimination (ranking) but not calibration (absolute probability accuracy). Domain adaptation can preserve ranking while destroying calibration: the translator might shift all predictions up or down without changing their order.

**Solution**: Add to `TransformerTranslatorEvaluator`:

```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

brier = brier_score_loss(y_true, y_prob)
fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
ece = np.mean(np.abs(fraction_pos - mean_pred))  # Expected Calibration Error
```

This is ~10 lines of code and immediately reveals if translations preserve the baseline's probability estimates. Critical for clinical deployment arguments in a thesis.

### D2. Per-Feature Delta Analysis — **Implemented**

> **Status**: Implemented in `src/core/eval.py` (`TransformerTranslatorEvaluator.translate_and_evaluate`). Running-stats accumulators track per-feature mean, std, abs_max, and fraction near zero. Logs top-5 most/least modified features with `[delta-analysis]` prefix.

**Problem**: We know the translator produces small deltas overall, but not *which* features it changes or whether those changes are clinically sensible.

**Solution**: After each evaluation, log per-feature statistics of the translator's deltas:

```python
# In evaluator, accumulate deltas across test set
deltas = x_val_out - x_val  # (B, T, 48)
per_feature_mean = deltas[~M_pad].mean(dim=0)  # (48,)
per_feature_std = deltas[~M_pad].std(dim=0)    # (48,)
per_feature_max = deltas[~M_pad].abs().max(dim=0).values
# Log feature name → delta stats
```

This tells you: is the translator modifying clinically relevant features (labs, vitals) or just noise features? Are the modifications in a plausible range? Combined with SHAP analysis on the baseline model before/after translation, this reveals whether the translator is doing something clinically meaningful.

### D3. Gradient Dynamics Over Training (Extend Current Diagnostic) — **Implemented**

> **Status**: Implemented in `src/core/train.py`. Gradient diagnostics now run periodically (epoch 0 detailed + batch 0 at `epochs // 4` intervals). Cosine similarity between task/fidelity gradients added (`cos_task_fid`). All `[grad-diag]` and `[grad-ts]` messages include `epoch=`.

**Problem**: Gradient norms are logged only for batches 0-3 of epoch 0. The gradient landscape changes as the translator learns.

**Solution**: Log gradient norms for 2-3 batches every N epochs (e.g., every 5). Additionally:

- **Cosine similarity between task and fidelity gradients**: if negative, they're actively fighting each other. This is a stronger diagnostic than magnitude ratio alone.
- **Gradient variance across batches**: high variance = noisy signal = need larger effective batch size or gradient accumulation.

```python
# Add: gradient direction conflict metric
task_grad_vec = torch.cat([p.grad.flatten() for p in translator.parameters() if p.grad is not None])
fid_grad_vec = ...  # same, after fidelity backward
cos_sim = F.cosine_similarity(task_grad_vec.unsqueeze(0), fid_grad_vec.unsqueeze(0))
logging.info("[grad-diag] task_fid_cosine=%.4f", cos_sim.item())
# Negative = directly opposing; Near 0 = orthogonal; Positive = aligned
```

### D4. Multi-Seed Statistical Significance

> **Deferred**: Multi-seed significance testing is deferred to the final results phase, after the best model/training configuration is identified. Running 5 seeds per experiment during development is too compute-intensive.

**Problem**: Best result is +0.0059 AUCROC. At this scale, we need to prove it isn't noise.

**Solution**: Run each experiment configuration with 5 different seeds. Report mean +/- std. Use DeLong's test for AUROC significance (paired comparison against baseline).

This is essential for thesis credibility. A result of +0.0059 +/- 0.003 (p=0.04) is publishable. A result of +0.0059 +/- 0.008 (p=0.3) is not.

### D5. Oracle Noise Bound

> **Deferred**: Oracle Noise Bound Multi-seed significance testing is deferred to the final results phase, after the best model/training configuration is identified.

**Problem**: No lower bound to contextualize results. Is +0.006 good or bad?

**Solution**: Run a "random translator" that adds Gaussian noise with the same mean/std as the learned translator's deltas. The comparison triangle:

| Translator | AUCROC Delta | Interpretation |
|---|---|---|
| Random noise | (expected negative) | Lower bound: any positive delta is meaningful |
| Causal learned | +0.006 | Our result |
| Bidirectional (cheaty) | +0.247 | Upper bound: ceiling with full information |

This frames the contribution and proves the translator is learning something beyond random perturbation.

---

## E. Experimental Methodology

> **Deferred**: The full ablation matrix and multi-task experiment tracks (E1, E2) are deferred to the final results phase. These require a stable best configuration to ablate against, and significant compute budget. Focus on model/training improvements first.

### E1. Systematic Ablation Matrix

Test key improvements in a factorial design to find interactions:

| Exp | Focal | Oversampling | Hidden-MMD | Dynamic λ | Chunking |
|---|---|---|---|---|---|
| 1 | - | f=20 | - | - | - | (current best baseline)
| 2 | γ=2 | f=20 | - | - | - |
| 3 | - | f=20 | λ=0.2 | - | - |
| 4 | γ=2 | f=20 | λ=0.2 | - | - |
| 5 | γ=2 | f=20 | λ=0.2 | yes | - |
| 6 | γ=2 | f=20 | λ=0.2 | yes | W=30 |

Run each with 3 seeds minimum. This reveals which improvements are additive vs. redundant.

### E2. Task-Specific Experiment Tracks

Don't only optimize for sepsis. Run the top 2-3 configurations on all three tasks:

- **Sepsis**: The hard case (per-timestep, long sequences, sparse labels)
- **Mortality24**: The working case (per-stay, short sequences, higher positive rate)
- **AKI**: Middle ground (per-timestep but different dynamics)

A method that improves all three tasks is a much stronger thesis contribution than one tuned to a single task.

---

## Priority Ranking

| # | Recommendation | Section | Effort | Impact | Notes |
|---|---|---|---|---|---|
| 1 | Focal loss | C1 | Low (15 lines) | High | Directly amplifies sparse task signal |
| 2 | ~~Calibration metrics~~ | D1 | ~~Low (10 lines)~~ | ~~High~~ | **Done** — Brier + ECE in all evaluators |
| 3 | Variable-length batching | A1 | Low (30 lines) | High | Eliminates 73% padding waste |
| 4 | Sequence chunking | A2 | Medium | High | Transforms sepsis to mortality-like structure |
| 5 | Penultimate-layer alignment | B1 | Medium | High | Model-agnostic, dense signal |
| 6 | GradNorm dynamic weighting | C2 | Medium | High | Eliminates lambda tuning |
| 7 | kNN translation in latent space | B3 | Medium | High | Per-sample supervision, label-free |
| 8 | ~~Gradient dynamics over training~~ | D3 | ~~Low (20 lines)~~ | ~~Medium~~ | **Done** — Periodic + cosine similarity |
| 9 | Multi-seed significance | D4 | Low (scripting) | Critical | Required for thesis validity |
| 10 | ~~Per-feature delta analysis~~ | D2 | ~~Low (15 lines)~~ | ~~Medium~~ | **Done** — Top-5 most/least modified features |
| 11 | Padding-aware fidelity | A3 | Low (10 lines) | Medium | Reduces gradient conflict at key timesteps |
| 12 | Optimal transport | B5 | Medium | Medium | Principled alternative to MMD |
| 13 | Cosine fidelity | C3 | Low (3 lines) | Medium | Better cross-feature gradient balance |
| 14 | Contrastive alignment | B4 | High | Medium | Needs pseudo-pair construction |
| 15 | Shared encoder | B2 | High | High | Most ambitious; strong thesis contribution |
| 16 | Ablation matrix | E1 | High (compute) | Critical | Required for thesis completeness |
| 17 | Oracle noise bound | D5 | Low | Medium | Frames contribution |

**Recommended execution order**: D1-D3 diagnostics are done. Start with 1 and 3 (focal loss + variable-length batching), then 4-5 (transform the problem structure), then 6-7 (training signal and latent alignment). Items 9 and 16 should run in parallel with development as final validation.
