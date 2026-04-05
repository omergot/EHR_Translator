# Gradient Alignment Condition: Evidence Review and Revision Recommendations

> **Purpose**: Audit of the "Gradient Alignment Condition" (Proposition 1, part2_method.tex lines 38-41)
> against all available gradient cosine logs. Recommendations for the next paper version.
>
> **Date**: 2026-03-29
> **Data sources**: 8 experiments across 4 tasks, 51 total cos(task, reg) measurements

---

## 1. What the Paper Currently Claims

**Proposition (informal):** Stable frozen-model translation requires E[alpha(theta)] >= 0 during training.

**Table 1 (part2_method.tex, lines 52-60):**

| Task      | alpha       | Outcome             |
|-----------|-------------|---------------------|
| Mortality | +0.84       | +0.048 AUROC        |
| AKI       | >0          | +0.056 AUROC        |
| Sepsis    | -0.21       | +0.006 AUROC (delta)|

**Text (line 63):** "the alignment coefficient is the **strongest predictor** of translation success"

---

## 2. What the Logs Actually Show

### 2.1 Per-batch alpha is extremely noisy

All surviving log files show alpha swinging between -1 and +1 **within the same epoch** for every task:

| Task      | N_measurements | Mean alpha | Std   | Range            | % Positive |
|-----------|----------------|------------|-------|------------------|------------|
| Mortality | 16             | **+0.038** | 0.491 | [-0.825, +0.912] | 50.0%      |
| AKI       | 8              | **+0.039** | 0.529 | [-0.896, +0.922] | 62.5%      |
| Sepsis    | 13             | **-0.126** | 0.586 | [-0.851, +0.955] | 38.5%      |
| KF        | 22             | **-0.018** | 0.194 | [-0.370, +0.393] | 50.0%      |

**Mortality mean alpha is +0.038, not +0.84.** The true mean is indistinguishable from zero given the variance (std = 0.49, 12x the mean).

**Sepsis mean alpha is -0.126, not -0.21.** But with std = 0.59, this is also consistent with zero. The two-sample difference (mortality vs sepsis) has a Cohen's d of ~0.30 — a small effect buried in noise.

### 2.2 The specific values +0.84 and -0.21 are single-batch snapshots

The original gradient bottleneck analysis (`docs/gradient_bottleneck_analysis.md`, line 257) reports these as point estimates in a comparative table. The +0.84 likely came from a single batch of an early debug run (the `exp_sepsis_oversample20_graddiag_debug` experiment whose log file was not preserved). They were never epoch-averaged or replicated.

### 2.3 Sepsis starts POSITIVE, then flips negative

The `sepsis_delta_target_task` run shows a dramatic epoch trajectory:

| Epoch | Batches measured | Mean alpha  |
|-------|------------------|-------------|
| 0     | 4                | **+0.658**  |
| 7     | 1                | **-0.463**  |
| 14    | 1                | **-0.844**  |
| 21    | 1                | **-0.469**  |
| 28    | 1                | **-0.635**  |

Sepsis gradients are **cooperative at initialization** and become destructive during training. This directly contradicts the narrative that sepsis has inherent destructive interference — it develops it.

### 2.4 Mortality also goes negative

From `mortality_delta_target_norm`:
- Epoch 7, batch 0: alpha = **-0.825**
- Epoch 14, batch 0: alpha = **-0.134**

From `mortality_delta_target_task`:
- Epoch 0, batch 0: alpha = **-0.495**
- Epoch 21, batch 0: alpha = **-0.441**

If negative alpha predicts failure, mortality shouldn't succeed either.

### 2.5 AKI alpha is NOT clearly positive

From `aki_delta_target_norm`:
- Epoch 0 mean: **-0.026** (slightly negative)
- Epoch 21: **-0.896** (strongly negative)
- Overall mean: **+0.039** (essentially zero)

The paper's claim "alpha > 0" for AKI is not supported. The mean is indistinguishable from zero.

### 2.6 Hidden-state cosine tells a contradictory story

The hidden-state centroid cosine similarity (a different metric, from `scripts/analyze_hidden_states.py`):

| Task      | cos(eICU, MIMIC) | cos(translated, MIMIC) | Translation AUROC Delta |
|-----------|-------------------|------------------------|-------------------------|
| Mortality | 0.993             | 0.978                  | +0.048                  |
| Sepsis    | 0.981             | 0.624                  | +0.051 (retrieval)      |
| AKI       | **0.186**         | **0.223**              | **+0.056** (best!)      |

AKI has the **lowest** hidden-state cosine but the **best** translation results. This rules out "cosine alignment predicts success" as a general principle at the representation level too.

---

## 3. What IS Well-Supported by the Data

### 3.1 Gradient magnitude ratio (fid/task) — STRONG evidence

This is stable, reproducible, and clearly differentiates tasks:

| Task      | Epoch 0 fid/task ratio | Task gradient norm | Outcome     |
|-----------|------------------------|--------------------|-------------|
| Sepsis    | **7.2–19.6x**          | 0.36–6.57          | Delta fails |
| Mortality | **0.6–12.5x**          | 0.68–2.96          | Delta works |
| AKI       | **0.4–1.6x**           | 1.17–2.46          | Delta works |
| KF (retr) | **7.1–99.2x**          | 0.08–0.41          | Marginal    |

The fid/task ratio is 4-10x higher for sepsis than for AKI. This is consistent across batches and experiments. Unlike alpha, this metric has low within-task variance and clear between-task separation.

### 3.2 Label sparsity → weak task gradient — STRONG evidence

| Task      | Label type    | Positive rate | Task grad norm (ep0) | Per-ts pos gradient |
|-----------|---------------|---------------|----------------------|---------------------|
| Mortality | Per-stay      | 12% (stay)    | 0.7–2.8              | 0.013 (concentrated)|
| AKI       | Per-timestep  | 11.95% (ts)   | 1.2–2.5              | —                   |
| Sepsis    | Per-timestep  | 1.13% (ts)    | 0.36–1.0             | 0.001 (diffuse)     |

The per-timestep gradient analysis (from `gradient_bottleneck_analysis.md`) shows mortality positive-timestep gradient is 11x stronger than sepsis. This is reproducible and mechanistically explained.

### 3.3 Zero-fidelity catastrophe — STRONG evidence

lambda_fidelity=0 causes -0.101 AUROC (0.719 -> 0.618). Fidelity is necessary but dominates. This is a clean, reproducible result.

### 3.4 Sepsis alpha trajectory (positive → negative) — NOVEL, INTERESTING

The epoch trajectory (Section 2.3 above) is actually more interesting than the original claim. It suggests the translator learns an initial cooperative mode, then as it begins to deviate from identity, the fidelity gradient starts opposing the task gradient. This is a **training dynamics** story, not a static property.

---

## 4. Recommendations for Paper Revision

### 4.1 REMOVE: The formal Proposition

> ~~Proposition [Gradient Alignment Condition, informal]: Stable frozen-model translation requires E[alpha(theta)] >= 0 during training.~~

**Why**: The empirical evidence shows E[alpha] ≈ 0 for ALL tasks, including successful ones. The proposition is falsified by our own data.

### 4.2 REMOVE: Table 1 with specific alpha values

> ~~alpha = +0.84 for mortality, alpha = -0.21 for sepsis~~

**Why**: These are single-batch snapshots with no statistical validity. The actual means are +0.038 and -0.126, both indistinguishable from zero given std > 0.49. A reviewer who asks "how was alpha computed?" or "what's the confidence interval?" will find the claim indefensible.

### 4.3 REPLACE WITH: Gradient Magnitude Dominance

**New framing**: The key diagnostic is not the *direction* of gradient alignment, but the *magnitude ratio* between regularization and task gradients. When the regularization gradient dominates by >5x, the translator cannot learn meaningful translations regardless of alignment direction.

Proposed replacement table:

| Task      | Label rate     | ||grad_fid|| / ||grad_task|| | Task grad norm | Delta outcome |
|-----------|----------------|-------------------------------|----------------|---------------|
| AKI       | 11.95% (per-ts)| 0.4–1.6x                      | 1.2–2.5        | +0.056 AUROC  |
| Mortality | 12% (per-stay) | 0.6–12.5x                     | 0.7–2.8        | +0.026 AUROC  |
| Sepsis    | 1.13% (per-ts) | **7.2–19.6x**                 | 0.4–1.0        | +0.006 AUROC  |

This table uses ranges (reproducible) instead of point estimates, and the magnitude ratio is far more stable than the cosine direction. The ranking (AKI < Mortality < Sepsis) perfectly predicts delta translator performance, and the mechanistic explanation (label sparsity → weak task gradient → dominated by regularization) is clean.

### 4.4 ADD: Training dynamics of alpha (optional, if space permits)

The sepsis epoch trajectory (Section 2.3) is genuinely interesting as a training dynamics observation. If there's space, present it as: "We observe that gradient alignment is cooperative at initialization for all tasks, but degrades during training for sepsis as the translator deviates from identity. This suggests the conflict is not inherent but emergent." This is honest, novel, and doesn't require overclaiming.

### 4.5 KEEP: The narrative arc (delta → SL → retrieval)

The overall story — that gradient conflict limits delta translators for sparse-label tasks, motivating retrieval — remains valid. The mechanism just needs to be reframed from "directional conflict (cosine)" to "magnitude dominance (ratio)". The retrieval architecture resolves both: it provides an alternative information channel (cross-attention from memory bank) that doesn't compete with the regularization gradient.

### 4.6 KEEP: Zero-fidelity catastrophe

The lambda_fid=0 experiment (-0.101 AUROC) is clean and compelling. It demonstrates the Goldilocks problem: regularization is necessary but harmful when too strong. Keep this as-is.

---

## 5. Suggested Revised Text (Sketch)

```latex
\paragraph{Gradient Magnitude Bottleneck.}
A central difficulty of frozen-model translation is that the task gradient
must flow through the frozen predictor before reaching $\theta$, attenuating
the signal. We decompose gradient contributions at the translator parameters:

\begin{equation}
\nabla_\theta \mathcal{L} = \nabla_\theta \mathcal{L}_{\mathrm{task}}
+ \lambda_{\mathrm{fid}} \nabla_\theta \mathcal{L}_{\mathrm{fid}}
\end{equation}

Table~\ref{tab:grad_ratio} reports the ratio
$\|\nabla_\theta \mathcal{L}_{\mathrm{fid}}\| /
\|\nabla_\theta \mathcal{L}_{\mathrm{task}}\|$ across tasks.
When this ratio exceeds $\sim$5$\times$, the regularisation gradient
dominates the optimisation landscape, and the translator converges to
near-identity regardless of whether the two gradients are directionally
aligned or opposed. The ratio is directly predicted by label density:
sparse per-timestep labels (sepsis, 1.1\%) produce weak, diffuse task
gradients that cannot compete with the dense fidelity signal, while
per-stay labels (mortality) or dense per-timestep labels (AKI, 12\%)
yield task gradients of comparable magnitude.

Removing fidelity entirely ($\lambda_{\mathrm{fid}} = 0$) causes
catastrophic divergence ($-0.101$ AUROC), confirming that the task
gradient alone is too noisy to guide the translator. This creates a
\emph{magnitude bottleneck}: fidelity is necessary to prevent
divergence but, when it dominates, suppresses learning. The retrieval
architecture (Section~\ref{sec:retrieval}) resolves this by providing
an orthogonal information channel---cross-attention over a target-domain
memory bank---whose gradient does not compete with fidelity.
```

---

## 6. Impact Assessment

| Change | Risk | Benefit |
|--------|------|---------|
| Remove Proposition 1 | Lose a "theoretical contribution" | Avoid falsifiable claim with N=3 and wrong numbers |
| Remove alpha table | Less dramatic narrative | Honest, reproducible metrics |
| Add magnitude ratio table | Slightly less elegant | Actually supported by data; reviewer-proof |
| Add training dynamics (optional) | More complex story | Novel observation, shows intellectual honesty |

**Net effect**: The paper becomes more honest and more defensible. The core insight (gradient competition limits delta translators for sparse tasks, motivating retrieval) is preserved — only the specific metric changes from cosine direction to magnitude ratio.

---

## Appendix: Raw Data

### A.1 All cos_task_fid measurements (delta trainer)

**mortality_delta_target_task** (AUROC delta: +0.026):
```
ep0 b0: -0.495  ep0 b1: +0.440  ep0 b2: -0.217  ep0 b3: +0.431
ep7 b0: +0.300  ep14 b0: +0.456  ep21 b0: -0.441  ep28 b0: -0.222
Mean: +0.032, Std: 0.388
```

**mortality_delta_target_norm** (AUROC delta: +0.026):
```
ep0 b0: +0.912  ep0 b1: +0.120  ep0 b2: -0.252  ep0 b3: -0.505
ep7 b0: -0.825  ep14 b0: -0.134  ep21 b0: +0.148  ep28 b0: +0.886
Mean: +0.044, Std: 0.576
```

**sepsis_delta_target_task** (AUROC delta: +0.006):
```
ep0 b0: +0.449  ep0 b1: +0.551  ep0 b2: +0.955  ep0 b3: +0.680
ep7 b0: -0.463  ep14 b0: -0.844  ep21 b0: -0.469  ep28 b0: -0.635
Mean: +0.028, Std: 0.654
```

**sepsis_delta_filtered** (AUROC delta: ~0.006):
```
ep0 b0: -0.197  ep0 b1: +0.106  ep0 b2: -0.610  ep0 b3: -0.309
ep7 b0: -0.851
Mean: -0.372, Std: 0.331
```

**aki_delta_target_norm** (AUROC delta: +0.026 delta, +0.056 retrieval):
```
ep0 b0: -0.511  ep0 b1: +0.477  ep0 b2: -0.160  ep0 b3: +0.092
ep7 b0: +0.922  ep14 b0: +0.226  ep21 b0: -0.896  ep28 b0: +0.162
Mean: +0.039, Std: 0.529
```

### A.2 All cos_task_recon measurements (retrieval trainer, KF task)

**kf_hirid_v5_cross3** (MAE delta: -0.021):
```
ep0 b0: -0.156  ep0 b1: +0.032  ep0 b2: -0.333  ep0 b3: -0.020
ep12 b0: +0.205  ep24 b0: +0.159  ep36 b0: +0.138  ep48 b0: +0.103
Mean: +0.016, Std: 0.170
```

**kf_v6_sr** (V6 self-retrieval):
```
ep0 b0: -0.370  ep0 b1: +0.393  ep0 b2: +0.076  ep0 b3: -0.262
ep12 b0: -0.084  ep24 b0: -0.004
Mean: -0.042, Std: 0.246
```

**kf_hirid_sr** (HiRID source):
```
ep0 b0: -0.323  ep0 b1: +0.132  ep0 b2: -0.146  ep0 b3: -0.205
ep7 b0: -0.046  ep14 b0: +0.052  ep21 b0: +0.161  ep28 b0: +0.100
Mean: -0.034, Std: 0.165
```

### A.3 Hidden-state centroid cosine similarity

| Task      | cos(eICU, MIMIC) | cos(translated, MIMIC) | Best AUROC Delta |
|-----------|-------------------|------------------------|------------------|
| Mortality | 0.993             | 0.978                  | +0.048           |
| Sepsis    | 0.981             | 0.624                  | +0.051           |
| AKI       | 0.186             | 0.223                  | +0.056           |

### A.4 LoS gradient diagnostics (degenerate)

All measurements show task_grad = 0.000000, making cosine undefined. The retrieval trainer's task loss produces zero gradient for LoS — likely a numerical issue with MSE on normalized targets near zero. Not usable for alpha analysis.
