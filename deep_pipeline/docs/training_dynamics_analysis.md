# Training Dynamics Analysis & Improvement Roadmap

**Date**: 2026-03-19
**Scope**: Convergence patterns, overfitting analysis, loss component decomposition, architecture decision rules, and prioritized improvement recommendations across all tasks and paradigms.
**Purpose**: Reference document for implementing training improvements in future sessions.

---

## 1. Training Dynamics Overview

### 1.1 Convergence Patterns by Task and Paradigm

| Task | Paradigm | Avg Best Epoch | Epoch Budget | Utilization | Pattern |
|---|---|---|---|---|---|
| **AKI** | Retrieval V5 | 22-26 | 30 | 72% | Trains productively, best epochs late |
| **AKI** | SL+FG | 20-25 | 30 | 73% | Steady improvement, late plateau |
| **Sepsis** | Retrieval V4 | 7-11 | 30 | 49% | Early plateau, wastes 50%+ compute |
| **Sepsis** | SL | oscillatory | 30 | N/A | val_task bounces 0.9-1.4, unreliable |
| **Mortality** | Retrieval V4 | 7-12 | 30 | 43% | Quick convergence, early plateau |
| **Mortality** | SL+FG | 12-15 | 30 | 47% | Moderate convergence, less overfitting |
| **Mortality** | Delta | 22-28 | 30 | 83% | Still improving at epoch 30 |

### 1.2 Key Observations

- **AKI is the only task that fully uses its epoch budget.** Dense labels (11.95%) provide enough gradient signal to keep learning through epoch 25+.
- **Sepsis and mortality plateau early.** Sparse labels (1.1% and 5.5% respectively) exhaust useful gradient information by epoch 7-12, after which the model overfits.
- **Delta experiments train the slowest** (still improving at epoch 30 for mortality). The additive delta architecture has fewer parameters and no pretrain phase, so it converges more gradually. May benefit from 50-epoch budgets.
- **SL experiments overfit less than retrieval** due to reconstruction loss acting as implicit regularization. The encoder-decoder must maintain general feature reconstruction quality, which constrains how far the model can deviate.
- **Early signal reliability varies by task** (from convergence analysis, n=110 logs):
  - Sepsis retrieval: val_task at epoch 3 predicts final AUCROC (Spearman rho=-0.73, p<0.05)
  - AKI retrieval: val_task at epoch 5 predicts final AUCROC (rho=-0.62, p<0.05)
  - Mortality: val_task does NOT predict final AUCROC within any paradigm (requires full runs)

---

## 2. The Overfitting Problem

### 2.1 Train-Val Gap Growth

The single biggest training issue: **constant lr=1e-4 causes massive train-val gaps after the best epoch**. The model keeps memorizing training patterns without improving generalization.

| Experiment | Best Epoch | Gap at Best | Gap at End | Growth Factor |
|---|---|---|---|---|
| AKI V5 cross3 | ~26 | 1.4% | 16.1% | 11.5x |
| Sepsis V4 MMD | ~9 | 3.5% | 34.4% | 9.8x |
| Mortality V4 MMD | ~10 | 2.8% | 72.1% | 25.8x |
| AKI SL+FG (mean) | ~22 | 1.8% | 5.4% | 3.0x |
| Sepsis SL (typical) | oscillatory | N/A | N/A | N/A |

### 2.2 Paradigm-Level Differences

- **Retrieval overfits the most**: The memory bank provides strong, specific signals from real MIMIC data. The model can learn to exploit training-set-specific patterns in the memory bank that don't generalize.
- **SL overfits less**: Reconstruction loss (lambda_recon) acts as implicit regularization. The model must maintain general feature reconstruction quality across all 48-292 features, preventing task-loss-driven overfitting. AKI SL+FG gap only reaches 5.4% vs 16.1% for retrieval.
- **Delta overfits least**: The additive delta constraint naturally limits how far outputs can deviate from inputs. But delta also converges slowest and achieves lower peak performance.

### 2.3 The Mortality Paradox

Mortality has the worst end-of-training gap (72.1%) despite having moderate label density (5.5%). Root cause: per-stay labels mean each stay contributes one gradient signal to the entire 24-timestep sequence. The model can memorize the mapping from small input perturbations to specific stay-level outcomes. With only ~6,200 positive stays in training, there are few unique positive patterns to learn.

### 2.4 Why Early Stopping Is Necessary but Insufficient

Current mitigation: `early_stopping_patience` (10 epochs by default). This catches the worst overfitting but:
- Wastes 30-60% of compute on epochs that will never beat the best
- Does not prevent the model from entering the overfitting regime in the first place
- patience=10 is too generous for sepsis/mortality (best epoch is 7-12, so 10 more wasted epochs)
- The underlying learning rate is too high for the later epochs, causing the model to oscillate around the optimum rather than converge toward it

---

## 3. Loss Component Analysis

### 3.1 Loss Decomposition at Best Epoch (Record Experiments)

At the best epoch for each record-holding experiment, the relative contribution of each loss component:

| Loss Component | AKI V5 cross3 | Sepsis V4 MMD | Mortality V4 | Role |
|---|---|---|---|---|
| **task** (eICU) | ~60% | ~58% | ~65% | Primary learning signal |
| **target_task** (MIMIC) | ~26% | ~28% | ~24% | Self-reconstruction on MIMIC |
| **recon** | <1% | ~2% | ~1% | Autoencoder reconstruction (self-corrects) |
| **align/MMD** | ~4% | ~4% | ~3% | Distributional alignment |
| **fidelity** | ~5% | ~4% | ~4% | Input preservation |
| **label_pred** | ~3% | ~2% | ~1% | Auxiliary label prediction |
| **range** | <1% | <1% | <1% | Boundary enforcement |
| **smooth** | 0% | 0% | <1% | Temporal smoothness |
| **imp_reg** | 0% | 0% | 0% | Importance regularization |

### 3.2 Key Insight: target_task Steals Gradient Capacity

The `target_task` loss (MIMIC self-reconstruction through the frozen LSTM) consumes 24-28% of the gradient budget. This loss keeps improving throughout training (the translator gets better at reconstructing MIMIC data) even when `val_task` (eICU translation quality) plateaus.

**This is the core dynamic**: the model improves at MIMIC self-reconstruction but stops improving at cross-domain translation. The target_task gradient continues to update parameters in ways that help MIMIC reconstruction but not eICU-to-MIMIC translation.

### 3.3 Loss Trajectory Patterns

- **recon** starts at ~10% of total loss at epoch 1 and drops to <1% by the best epoch. The autoencoder quickly learns to reconstruct inputs accurately. This is expected behavior and confirms Phase 1 pretrain is working.
- **align/MMD** provides ~3-4% of gradient. Small but crucial -- for sepsis, adding MMD was the key to the V4 record (+0.0512 vs +0.0499 without). MMD provides a distributional alignment signal that complements instance-level k-NN retrieval.
- **range, smooth, imp_reg** are negligible in all record experiments. These auxiliary losses were explored in ablations and found to be neutral or harmful (smoothness hurts sepsis AUCPR, importance reg causes weight collapse).
- **label_pred** provides ~1-5% of signal. Useful auxiliary signal but not dominant. Was bugged in V3 (only active in Phase 1), fixed in V4.

---

## 4. Early Stopping Metric: val_task Is Correct

### 4.1 Metric Comparison

In all experiments tested, **val_task selected an equal or better checkpoint than val_total**:

- `val_task`: validation loss on eICU data through the frozen LSTM (direct measure of translation quality)
- `val_total`: sum of all loss components on validation data

### 4.2 Why val_total Fails

`val_total` continues improving even after `val_task` plateaus because:
1. `val_recon` keeps dropping (autoencoder improves at self-reconstruction)
2. `val_target_task` keeps dropping (model improves at MIMIC-domain prediction)
3. `val_align` keeps dropping (latent distributions align further)

None of these auxiliary improvements translate to better eICU-to-MIMIC translation quality. Selecting the checkpoint by `val_total` would choose a later, more overfit checkpoint.

### 4.3 Recommendation

Continue using `val_task` as `best_metric` in all configs. No change needed.

---

## 5. Why n_cross_layers=3 Helps AKI but Hurts Sepsis

### 5.1 The Label Density Bottleneck

Each CrossAttentionBlock adds ~265K trainable parameters. More parameters require more gradient information to train effectively.

| Task | Positive Rate | Informative Gradients/Sequence | Can Train 3 CrossAttn Layers? |
|---|---|---|---|
| AKI | 11.95% | ~20 per sequence | Yes -- sufficient gradient mass |
| Mortality | 5.52% | ~1 per stay (per-stay label) | Borderline (V4 tied with SL) |
| Sepsis | 1.13% | ~1.9 per sequence | No -- gradient too sparse |

### 5.2 Empirical Evidence

| Experiment | n_cross_layers | AUCROC delta | vs 1-layer |
|---|---|---|---|
| AKI V5 cross3 | 3 | **+0.0556** | **+0.0087** (significant gain) |
| Sepsis V5 cross3 | 3 | +0.0448 | -0.0064 (regression from V4 record) |
| Mortality V4 | 1 | +0.0470 | N/A (no 3-layer comparison) |

### 5.3 Confounding in the Sepsis V5 Cross3 Experiment

**IMPORTANT**: The `sepsis_retr_v5_cross3` experiment was **confounded**. Multiple parameters changed simultaneously from the V4 record config:

| Parameter | V4 Record (sepsis_retr_v4_mmd) | V5 Cross3 (sepsis_retr_v5_cross3) |
|---|---|---|
| n_cross_layers | 2 | **3** (changed) |
| n_dec_layers | 2 | **3** (changed) |
| epochs | 30 | **50** (changed) |
| patience | 10 | **15** (changed) |
| V5 fixes | No | Yes (double-encode fix, overlapping bank) |

Because multiple variables changed, we cannot definitively attribute the sepsis regression to n_cross_layers alone. **A clean ablation is needed** (Section 8.1).

### 5.4 Additional Contributing Factor: Gradient Alignment

Beyond label density, sepsis has destructive gradient alignment:
- Sepsis: cos(task, fidelity) = **-0.21** (gradients fight)
- Mortality: cos(task, fidelity) = **+0.84** (gradients cooperate)

With 3 cross-attention layers, the gradient must flow through more parameters. When the task gradient is already destructively interfering with fidelity, adding more parameters amplifies the noise, making it harder for the model to find a useful direction.

### 5.5 Threshold Estimate

Based on the AKI/Sepsis comparison, the empirical threshold for benefiting from n_cross_layers=3 appears to be approximately **10% label density**. Below this, the task gradient is too sparse to effectively train the additional parameters.

---

## 6. Architecture Decision Rule

### 6.1 Two Templates, One Knob

```
IF label_density >= 10% OR task_type == "regression":
    n_cross_layers = 3
ELSE:
    n_cross_layers = 2
```

Note: Regression tasks (LoS, KF) use MSE loss which provides dense gradient at every labeled timestep, similar to high-density classification tasks.

### 6.2 Shared Parameters Across All Tasks

| Parameter | Value | Notes |
|---|---|---|
| `d_model` | 128 | Encoder/decoder hidden dimension |
| `d_latent` | 128 | Latent space dimension |
| `n_enc_layers` | 4 | Encoder transformer layers |
| `n_dec_layers` | 3 | Decoder transformer layers |
| `k_neighbors` | 16 | k-NN retrieval neighbors |
| `retrieval_window` | 6 | Memory bank window size |
| `window_stride` | 3 | Overlapping memory bank (V5) |
| `lambda_fidelity` | 0.1 | Feature preservation |
| `lambda_range` | 0.1 | Output range enforcement |
| `lambda_align` | 0.5 | MMD distributional alignment |
| `lambda_smooth` | 0.0 | Temporal smoothness (disabled -- hurts sepsis AUCPR) |
| `lambda_importance_reg` | 0.0 | Weight sparsity (disabled -- causes collapse) |
| `lambda_label_pred` | 0.1 | Auxiliary label prediction |
| `lambda_target_task` | 0.5 | MIMIC self-reconstruction |
| `feature_gate` | true | Per-feature learned weighting |
| `output_mode` | "absolute" | Direct output (not additive residual) |
| `use_target_normalization` | true | Affine renorm of source features |

### 6.3 Task-Specific Parameters

| Parameter | AKI | Sepsis | Mortality | LoS | KF |
|---|---|---|---|---|---|
| `n_cross_layers` | **3** | **2** | **2** | **3** | **3** |
| `temporal_attention_mode` | causal | causal | bidirectional | causal | bidirectional |
| `variable_length_batching` | true | true | **false** | true | **false** |
| `batch_size` | 16 | 16 | 16 | 16 | 16 |
| `epochs` | 30 | 30 | 30 | 30 | 30 |
| `early_stopping_patience` | 10 | 10 | 10 | 10 | 10 |

---

## 7. Recommended Improvements (Priority Order)

### 7.1 LR Scheduling (HIGH IMPACT)

**Problem**: Constant lr=1e-4 causes 16-72% train-val gaps after the best epoch. The model overshoots the optimum and oscillates in the overfitting regime.

**Evidence**:
- AKI V5 cross3: gap grows from 1.4% to 16.1% (11.5x)
- Sepsis V4 MMD: gap grows from 3.5% to 34.4% (9.8x)
- Mortality V4: gap grows from 2.8% to 72.1% (25.8x)
- All tasks show continued train loss improvement while val loss stagnates or worsens

**Solution**: Two options, both well-supported by the evidence:

Option A -- `CosineAnnealingLR(T_max=max_epochs, eta_min=1e-6)`:
- Smoothly decays lr from 1e-4 to 1e-6 over the full training run
- Pros: Simple, no hyperparameter tuning, makes later epochs productive (smaller steps near convergence)
- Cons: Fixed schedule, doesn't adapt to actual training dynamics

Option B -- `ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)`:
- Halves lr when val_task stops improving for 5 epochs
- Pros: Adapts to actual convergence, could extend productive training for AKI (where plateau comes late)
- Cons: More hyperparameters, interacts with early stopping (patience overlap)

**Recommendation**: Start with Option A (cosine) for simplicity. If results are good, stick with it. Try Option B only if cosine doesn't help.

**Expected impact**: Later best epochs, smaller train-val gap, potentially better final val_task. The gain should be largest for mortality (72% overfitting, lot of room) and sepsis (34% overfitting, plus early plateau could shift later).

### 7.2 Reduce lambda_target_task for Sparse Tasks (MODERATE IMPACT)

**Problem**: The target_task loss (MIMIC self-reconstruction through frozen LSTM) consumes 24-28% of gradient budget. This signal improves throughout training even when eICU translation quality plateaus. For sparse-label tasks (sepsis, mortality), the target_task gradient may dominate the task gradient and steer parameters toward MIMIC-specific patterns that don't help translation.

**Evidence**:
- target_task loss keeps decreasing monotonically even after val_task plateaus
- Sepsis val_task often oscillates while val_target_task drops steadily
- The gradient from target_task is dense (every MIMIC timestep contributes) while task gradient is sparse (only positive eICU timesteps provide informative signal)

**Solution**: Reduce `lambda_target_task` from 0.5 to 0.2-0.3 for sepsis and mortality.

| Task | Current lambda_target_task | Proposed | Rationale |
|---|---|---|---|
| AKI | 0.5 | **0.5** (keep) | Dense labels can compete with target_task gradient |
| Sepsis | 0.5 | **0.2** | Sparse labels drowned by target_task; reduce its weight |
| Mortality | 0.5 | **0.3** | Moderate density; slight reduction may help |
| LoS | 0.5 | **0.5** (keep) | Dense regression signal per timestep |
| KF | 0.5 | **0.3** | Per-stay label, similar to mortality |

**Implementation**: Config-level change only. No code modifications needed.

### 7.3 Gradient Accumulation for Sepsis (MODERATE IMPACT)

**Problem**: At 1.1% positive rate and batch_size=16, many batches contain zero or very few positive timesteps. This creates noisy, uninformative gradient updates where the model only sees negative examples.

**Quantification**:
- Average sequence length: ~52 timesteps (padded to 169)
- batch_size=16: ~832 real timesteps per batch
- At 1.1% positive rate: ~9 positive timesteps per batch
- With oversampling_factor=1 (no oversampling): ~4.57% positive stays, so ~0.7 positive stays per batch of 16
- Many batches have 0 positive stays, providing pure-negative gradients

**Solution**: Add `accumulate_grad_batches` parameter. Instead of calling optimizer.step() every batch, accumulate gradients over N batches and step once.

| Setting | Effective Batch Size | Positive Timesteps per Update | Memory Impact |
|---|---|---|---|
| accumulate=1 (current) | 16 | ~9 | Baseline |
| accumulate=2 | 32 (effective) | ~18 | None (no extra memory) |
| accumulate=4 | 64 (effective) | ~37 | None |

**Implementation**: Modify `RetrievalTranslatorTrainer._run_epoch()` to accumulate gradients. See Section 10.2.

**Expected impact**: More stable gradient estimates for sepsis. Unlikely to change AKI or mortality significantly (they already have enough positive samples per batch).

### 7.4 Task-Specific Epoch Budgets (COMPUTE SAVINGS)

**Problem**: Current default is 30 epochs with patience=10 for all tasks. This wastes significant compute:
- Sepsis best epoch is 7-11 with patience=10 meaning early stop at epoch 17-21. The remaining 9-13 budgeted epochs are never used, but the data loading and Phase 1 pretrain (the real bottleneck) already happened.
- Mortality is similar: best epoch 7-12, early stop at 17-22.
- AKI fully uses its budget (best epoch 22-26) but could benefit from more epochs.
- Delta experiments are still improving at epoch 30.

**Proposed budget adjustments**:

| Task | Paradigm | Current Budget | Proposed Budget | Proposed Patience | Compute Saved |
|---|---|---|---|---|---|
| AKI | Retrieval | 30, pat=10 | **50, pat=15** | 15 | None (extends) |
| AKI | SL | 30, pat=10 | 30, pat=10 | 10 | None (keep) |
| Sepsis | Retrieval | 30, pat=10 | **20, pat=8** | 8 | ~30-40% Phase 2 |
| Mortality | Retrieval | 30, pat=10 | **25, pat=10** | 10 | ~15% Phase 2 |
| Mortality | Delta | 30, pat=5 | **50, pat=15** | 15 | None (extends) |

**Note**: These should be applied AFTER implementing LR scheduling (Section 7.1). With cosine LR, the later epochs become more productive and the optimal budget may differ.

### 7.5 Clean Sepsis Ablation (EVIDENCE GAP)

**Problem**: The sepsis V5 cross3 experiment was confounded (see Section 5.3). We do not have clean evidence for whether n_cross_layers=3 helps or hurts sepsis specifically.

**Required experiment**: Take the V4 record config (`sepsis_retr_v4_mmd.json`) and change ONLY `n_cross_layers` from 2 to 3. Keep all other parameters identical:
- n_dec_layers=2 (NOT 3)
- epochs=30 (NOT 50)
- patience=10 (NOT 15)
- V4 code (no double-encode fix, no overlapping bank)

This isolates the cross-attention depth effect. If the result is below the V4 record (+0.0512), it confirms n_cross_layers=3 hurts sepsis. If above, it reopens the architecture question.

---

## 8. Experiment Queue for Improvements

### 8.1 Priority 1: Evidence Gaps (run first)

| # | Name | Base Config | Change | Expected Time | Purpose |
|---|---|---|---|---|---|
| 1 | `sepsis_v4_cross3_clean` | `sepsis_retr_v4_mmd.json` | n_cross_layers: 2 -> 3 | ~6h | Clean ablation of cross-attention depth |

### 8.2 Priority 2: LR Scheduling (highest expected impact)

| # | Name | Base Config | Change | Expected Time | Purpose |
|---|---|---|---|---|---|
| 2 | `sepsis_v4_cosine_lr` | `sepsis_retr_v4_mmd.json` | +CosineAnnealingLR | ~6h | Test LR scheduling on worst overfitter |
| 3 | `aki_v5_cosine_lr` | `aki_v5_cross3.json` | +CosineAnnealingLR | ~10h | Can we beat the AKI record? |
| 4 | `mortality_v4_cosine_lr` | `mortality_retr_v4_mmd.json` | +CosineAnnealingLR | ~5h | Test on highest gap (72%) |

### 8.3 Priority 3: Loss Weight Tuning (moderate impact, config-only)

| # | Name | Base Config | Change | Expected Time | Purpose |
|---|---|---|---|---|---|
| 5 | `sepsis_v4_low_target_task` | `sepsis_retr_v4_mmd.json` | lambda_target_task: 0.5 -> 0.2 | ~6h | Reduce MIMIC gradient competition |
| 6 | `mortality_v4_low_target_task` | `mortality_retr_v4_mmd.json` | lambda_target_task: 0.5 -> 0.3 | ~5h | Same rationale for mortality |

### 8.4 Priority 4: Gradient Accumulation (sepsis-specific)

| # | Name | Base Config | Change | Expected Time | Purpose |
|---|---|---|---|---|---|
| 7 | `sepsis_v4_grad_accum4` | `sepsis_retr_v4_mmd.json` | accumulate_grad_batches: 4 | ~6h | Stabilize sparse gradients |

### 8.5 Combination Experiments (after individual ablations)

Once individual effects are measured, combine the best:

| # | Name | Changes | Purpose |
|---|---|---|---|
| 8 | `sepsis_v4_cosine_low_tt` | cosine LR + lambda_target_task=0.2 | Best 2 improvements combined |
| 9 | `sepsis_v4_full_improve` | cosine LR + low target_task + grad_accum | All improvements combined |
| 10 | `aki_v5_cosine_extended` | cosine LR + epochs=50 + patience=15 | Extended training with LR decay |

---

## 9. Current Best Results (Reference)

### 9.1 Classification (AUCROC delta from frozen baseline)

| Task | Frozen Baseline | Best Result | Absolute | Best Config | YAIB eICU-native | Status |
|---|---|---|---|---|---|---|
| Mortality | 0.8079 | +0.0476 | **0.8555** | SL+FG | 0.855 | Matched |
| AKI | 0.8558 | +0.0556 | **0.9114** | Retr V5 cross3 | 0.902 | **Surpassed** |
| Sepsis | 0.7159 | +0.0512 | **0.7671** | Retr V4+MMD | 0.740 | **Surpassed** |

### 9.2 Regression (MAE)

| Task | Frozen Baseline | Translated | Improvement | MIMIC-native | YAIB eICU-native | Status |
|---|---|---|---|---|---|---|
| LoS | 42.5h | **39.2h** | -3.3h | 40.9h | 39.2h | Matched eICU-native, surpassed MIMIC |
| KF | 0.403 mg/dL | **0.382 mg/dL** | -0.021 | 0.346 | 0.28 | Improved, gap remains |

### 9.3 Statistical Significance

All improvements are highly significant (paired cluster bootstrap, p<0.001, 95% CIs exclude zero):

| Task | AUCROC delta | 95% CI |
|---|---|---|
| AKI (V5 cross3) | +0.0556 | [+0.0513, +0.0554] (from SL seeds; V5 CIs pending) |
| Mortality (SL+FG) | +0.0467 | [+0.0374, +0.0558] |
| Sepsis (V4 MMD) | +0.0510 | [+0.0389, +0.0635] |

### 9.4 Cross-Server Reproducibility

| Task | Label Density | Local (V100S) | A6000 | Std | Verdict |
|---|---|---|---|---|---|
| AKI | 11.95% | +0.0529 | +0.0527 | +/-0.0012 | Excellent |
| Mortality | 5.52% | +0.0462 | +0.0425 | +/-0.0025 | Good |
| Sepsis | 1.13% | +0.0443 | +0.0384 | +/-0.0048 | Moderate |

Stability inversely correlated with label sparsity. Multi-seed reporting essential for sepsis.

---

## 10. Implementation Notes

### 10.1 Files to Modify for LR Scheduling

**`src/core/train.py`** -- All 3 trainer classes need scheduler support:

1. In `__init__()` of each trainer: create scheduler after optimizer initialization
   ```python
   # After self.optimizer = ...
   lr_schedule = training_config.get('lr_schedule', None)
   lr_min = training_config.get('lr_min', 1e-6)
   if lr_schedule == 'cosine':
       from torch.optim.lr_scheduler import CosineAnnealingLR
       self.scheduler = CosineAnnealingLR(
           self.optimizer, T_max=self.epochs, eta_min=lr_min
       )
   elif lr_schedule == 'plateau':
       from torch.optim.lr_scheduler import ReduceLROnPlateau
       self.scheduler = ReduceLROnPlateau(
           self.optimizer, mode='min', patience=5,
           factor=0.5, min_lr=lr_min
       )
   else:
       self.scheduler = None
   ```

2. After each epoch (in the main training loop), step the scheduler:
   ```python
   # After validation
   if self.scheduler is not None:
       if isinstance(self.scheduler, ReduceLROnPlateau):
           self.scheduler.step(val_task_loss)
       else:
           self.scheduler.step()
   ```

3. Log the current learning rate:
   ```python
   current_lr = self.optimizer.param_groups[0]['lr']
   logging.info(f"  lr={current_lr:.2e}")
   ```

4. Save/restore scheduler state in checkpoints:
   ```python
   # Save
   checkpoint['scheduler_state'] = self.scheduler.state_dict() if self.scheduler else None
   # Load
   if self.scheduler and 'scheduler_state' in checkpoint and checkpoint['scheduler_state']:
       self.scheduler.load_state_dict(checkpoint['scheduler_state'])
   ```

**`src/cli.py`** -- Add to `_get_training_config()` whitelist:
```python
'lr_schedule',  # 'cosine', 'plateau', or None
'lr_min',       # minimum learning rate (default 1e-6)
```

**Config files** -- Add to training section:
```json
{
    "training": {
        "lr_schedule": "cosine",
        "lr_min": 1e-6
    }
}
```

### 10.2 Files to Modify for Gradient Accumulation

**`src/core/train.py`** -- Modify `_run_epoch()` in `RetrievalTranslatorTrainer`:

```python
# In __init__
self.accumulate_grad_batches = training_config.get('accumulate_grad_batches', 1)

# In _run_epoch, replace the optimizer step block:
loss_total.backward()

if (batch_idx + 1) % self.accumulate_grad_batches == 0:
    self.optimizer.step()
    self.optimizer.zero_grad()

# Don't forget the final partial accumulation at the end of the epoch:
if (batch_idx + 1) % self.accumulate_grad_batches != 0:
    self.optimizer.step()
    self.optimizer.zero_grad()
```

**Note**: Scale the loss by `1/accumulate_grad_batches` before `.backward()` to keep effective learning rate consistent, OR adjust the lr proportionally.

**`src/cli.py`** -- Add to `_get_training_config()` whitelist:
```python
'accumulate_grad_batches',  # int, default 1 (no accumulation)
```

### 10.3 Backward Compatibility Requirements

All new config keys MUST default to disabled behavior:
- `lr_schedule`: default `None` (no scheduling, constant lr as before)
- `lr_min`: default `1e-6` (only used if lr_schedule is set)
- `accumulate_grad_batches`: default `1` (step every batch, identical to current behavior)

**Existing experiments must produce identical results with the new code.** This is non-negotiable -- any change to default behavior would invalidate all prior results and make cross-experiment comparisons meaningless.

### 10.4 The `_get_training_config()` Whitelist (CRITICAL)

The `_get_training_config()` function in `src/cli.py` explicitly lists all config keys that are passed to the trainer. **New training config keys MUST be added here or they are silently dropped.** This is the #1 source of "config change had no effect" bugs in this codebase.

Every time you add a new config key:
1. Add it to the JSON config file
2. Add it to `_get_training_config()` in `src/cli.py`
3. Add it to the trainer's `__init__()` with a backward-compatible default
4. Test that the value actually reaches the trainer (add a `logging.info()` at startup)

---

## 11. Relationship to Prior Analyses

This document synthesizes findings from:

| Document | Key Contribution to This Analysis |
|---|---|
| `docs/gradient_bottleneck_analysis.md` | Gradient magnitude ratios, cos(task,fidelity) alignment, per-timestep gradient norms |
| `docs/convergence_analysis/convergence_report.md` | Early signal reliability (Spearman rho), wall-clock timing, ranking stability |
| `docs/comprehensive_results_summary.md` Sec 20-27 | V3-V5 ablation results, multi-seed variance, bootstrap CIs |
| `docs/sepsis_label_density_analysis.md` | Why sparse labels cause gradient noise, AKI vs sepsis controlled comparison |
| `memory/MEMORY.md` | Current records, task-specific strategy, active state |

---

## 12. Summary of Actionable Items

| Priority | Item | Type | Expected Impact | Effort |
|---|---|---|---|---|
| **P0** | Clean sepsis cross3 ablation | Experiment | Evidence gap closure | Config-only |
| **P1** | CosineAnnealingLR scheduling | Code + config | Reduced overfitting, later best epochs | ~2h coding |
| **P2** | Reduce lambda_target_task (sepsis/mortality) | Config-only | Better gradient allocation | Minutes |
| **P3** | Gradient accumulation (sepsis) | Code + config | Stabilized gradient estimates | ~1h coding |
| **P4** | Task-specific epoch budgets | Config-only | 15-40% compute savings | Minutes |
| **P5** | Extended delta training (50 epochs) | Config-only | Better delta results | Minutes |

The single highest-impact improvement is likely LR scheduling (P1), because it directly addresses the largest observed problem (16-72% overfitting) and is well-understood in the deep learning literature. The implementation is straightforward and backward compatible.

---

## 13. Phase 1 (Autoencoder Pretraining) Investigation

### 13.1 What Phase 1 Does

Phase 1 trains the encoder-decoder on **MIMIC target data only** before Phase 2 joint training. Both `LatentTranslatorTrainer` and `RetrievalTranslatorTrainer` share this pattern.

**Loss**: `loss = l_recon + lambda_label_pred * l_label_pred`
- Reconstruction: MSE between decoded output and MIMIC input, summed across features, averaged across valid timesteps
- Label prediction (optional): BCE for classification / MSE for regression, from a lightweight MLP head on the latent

**What gets trained**: Encoder (triplet_proj, sensor_emb, enc_blocks, to_latent), decoder (dec_blocks, output_head), projections (latent_to_cross, from_latent), label_pred_head, **and cross-attention blocks** — but with degenerate zero context (see 13.5).

**After Phase 1**: Optimizer and GradScaler are **completely reset**. Phase 2 starts with fresh momentum.

### 13.2 Reconstruction Loss Convergence

Phase 1 recon loss follows a power-law decay (~24 × epoch^(-0.85)) and is **NOT converged at 15 epochs**:

| Task | Features | Ep 1 | Ep 5 | Ep 10 | Ep 15 | Ep 10→15 Δ |
|---|---|---|---|---|---|---|
| AKI | 48 | 24.2-25.0 | 11.7-12.7 | 5.7-6.4 | 4.3-4.9 | **-22 to -27%** |
| Sepsis | 48 | 23.8-24.1 | 9.6-10.3 | 5.3-5.9 | 4.3-4.6 | **-19 to -22%** |
| Mortality | 48 | 24.1-25.4 | 12.4-17.8 | 5.7-11.7 | 4.6-8.7 | **-19 to -23%** |
| LoS | 48 | 24.9 | 10.3 | 5.6 | 4.3 | -22% |
| **KF** | **48** | **26.1** | **17.2** | **10.7** | **7.0** | **-35% (still high)** |

All tasks with 48 features converge to recon ~4.3-4.9 except **KF which converges significantly slower** (7.0 at ep 15). KF has different data distribution (shorter stays, different feature fill patterns) making reconstruction harder. KF would benefit most from extended pretraining.

### 13.3 Label Prediction Is Ineffective in Phase 1

Label_pred converges to near-random-guess loss for all tasks:

| Task | Positive Rate | label_pred @ep15 | Random Guess BCE | Margin |
|---|---|---|---|---|
| AKI | 11.95% | 0.462-0.466 | 0.443 | +0.02 (barely above random) |
| Sepsis | 1.13% | 0.109-0.110 | 0.067 | +0.04 |
| Mortality | 5.52% | 0.244-0.254 | 0.200 | +0.05 |
| LoS (MSE) | continuous | 0.061 | — | — |
| KF (MSE) | continuous | 0.003 | — | — |

**Evidence it may hurt**: `aki_v4_no_labelpred` (lambda_label_pred=0 in both phases) achieved **+0.0492 AUCROC**, better than `aki_retr_v4_mmd` (+0.0469) which used lambda_label_pred=0.1. The label_pred head learns a degenerate predict-base-rate solution, providing no useful encoder gradient.

### 13.4 Phase 1 Checkpoint Reuse

Reuse is safe when architecture (d_latent, d_model, n_enc/dec_layers), target data, pretrain_epochs, and seed match. Phase 2 hyperparams (k_neighbors, lambda_*, n_cross_layers, window_stride) do NOT affect Phase 1.

**Verified**: label_pred converges to 0.462-0.466 across ALL AKI retrieval runs regardless of Phase 2 config.

**Time savings**:
| Task | Phase 1 Time (V100) | Phase 2 Time | Phase 1 % of Total |
|---|---|---|---|
| AKI | ~97 min | ~1560 min (30 ep) | **6%** |
| Sepsis | ~112 min | ~225 min (30 ep) | **33%** |
| Mortality | ~57 min | ~105 min (30 ep) | **35%** |

Checkpoint reuse saves meaningful time for sepsis/mortality (33-35%) but is marginal for AKI (6%).

### 13.5 Critical Design Issue: Cross-Attention with Zero Context

The `RetrievalTranslator.decode()` method (used in Phase 1) passes **zero tensors** as context:
```python
zero_context = torch.zeros(B, T, 1, self.d_model, ...)
for block in self.cross_blocks:
    h = block(h, zero_context, m_pad)
```

**Problem**: The cross-attention layers learn a degenerate pass-through mode in Phase 1 (attending to all-zero KV = learned bias). In Phase 2, they must completely relearn to attend to real retrieved MIMIC neighbors. This transition is driven by the task gradient — which is noisy and sparse, especially for sepsis.

**Impact**: This may explain why the cross-attention layers (n_cross_layers) show task-dependent effectiveness:
- AKI (dense labels): enough task gradient to retrain cross-attention from zero-context init
- Sepsis (sparse labels): insufficient gradient to overcome the degenerate Phase 1 init

### 13.6 Potential Phase 1 Improvements

#### P0: Self-Retrieval in Phase 1 (HIGH IMPACT)
Build a MIMIC memory bank during Phase 1. For each batch, query the bank and pass real context to cross-attention instead of zeros. This fixes the biggest design flaw — cross-attention learns meaningful patterns before Phase 2.

**Implementation**: Call `_build_memory_bank()` before Phase 1 loop. In `_pretrain_epoch`, use `forward_with_retrieval()` instead of `encode()+decode()`. Rebuild bank every `memory_refresh_epochs`.

**Expected benefit**: Better cross-attention init → faster Phase 2 convergence → potentially higher final performance, especially for label-sparse tasks.

#### P1: Extended Pretraining (30 epochs) (MODERATE IMPACT)
Loss still drops 19-27% between epoch 10→15. Extrapolating the power-law decay:
- 30 epochs → recon ~3.0-3.5 (vs 4.3-4.9 at ep 15)
- KF specifically needs this (recon=7.0 at ep 15 vs ~4.5 for others)

**Cost**: ~200 min for AKI (vs 97 min), but only 6% of total. For sepsis, Phase 1 would grow from 33% to ~45% of total — still worthwhile if Phase 2 convergence improves.

#### P2: Remove Label Prediction from Phase 1 (LOW IMPACT, EASY)
Near-zero useful signal. The `aki_v4_no_labelpred` experiment without it did +0.0023 better. Currently the same `lambda_label_pred` applies to both phases — could add a separate `pretrain_label_pred` config key, or just set to 0.

#### P3: Feature-Weighted Reconstruction (MODERATE IMPACT)
Phase 1 treats all 48 features equally. Use FeatureGate weights or LSTM gradient magnitudes from a prior experiment to prioritize task-relevant features in the reconstruction loss.

#### P4: Masked Autoencoder Objective (MODERATE IMPACT, MORE WORK)
Mask 30-50% of features per timestep, reconstruct only masked. Forces contextual learning instead of trivial pass-through. Aligns better with the translation task (encoder must infer missing information).

#### P5: Dual-Domain Phase 1 (EXPLORATORY)
Train on both MIMIC + eICU. Gives the shared encoder exposure to both domains. Risk: encoder learns domain shortcuts. Would need alignment loss (MMD) already in Phase 1.

### 13.7 Recommended Phase 1 Experiment Queue

| Priority | Experiment | Change from Baseline | Expected Impact |
|---|---|---|---|
| **P0** | `aki_v5_self_retrieval_pretrain` | Self-retrieval in Phase 1 | Better cross-attn init |
| **P0** | `sepsis_v4_self_retrieval_pretrain` | Same for sepsis | Largest benefit (sparse labels) |
| **P1** | `kf_v5_pretrain30` | pretrain_epochs=30 | Fix KF slow convergence |
| **P2** | `aki_v5_no_pretrain_labelpred` | lambda_label_pred=0 in Phase 1 only | Remove useless signal |
| **P3** | `aki_v5_weighted_recon` | Feature-weighted Phase 1 recon | Focus capacity on task features |

---

## 14. Updated Summary of All Actionable Items

| Priority | Item | Type | Expected Impact | Effort | Section |
|---|---|---|---|---|---|
| **P0** | Self-retrieval in Phase 1 | Code | Fix cross-attn zero-context flaw | ~3h coding | 13.6 |
| **P0** | Clean sepsis cross3 ablation | Config | Evidence gap closure | Minutes | 7 |
| **P1** | CosineAnnealingLR scheduling | Code + config | Reduced overfitting (16-72% gap) | ~2h coding | 7 |
| **P1** | Extended Phase 1 for KF (30 ep) | Config | Fix KF slow convergence | Minutes | 13.6 |
| **P2** | Reduce lambda_target_task (sepsis/mort) | Config | Better gradient allocation | Minutes | 7 |
| **P2** | Remove label_pred from Phase 1 | Config or code | Remove useless gradient | Minutes-1h | 13.6 |
| **P3** | Gradient accumulation (sepsis) | Code + config | Stabilized gradient estimates | ~1h coding | 7 |
| **P3** | Feature-weighted Phase 1 recon | Code + config | Focus recon on task features | ~2h coding | 13.6 |
| **P4** | Task-specific epoch budgets | Config | 15-40% compute savings | Minutes | 7 |
| **P5** | Masked autoencoder Phase 1 | Code | Better representations | ~4h coding | 13.6 |
| **P5** | Dual-domain Phase 1 | Code + design | Domain-invariant encoder | ~4h coding | 13.6 |
