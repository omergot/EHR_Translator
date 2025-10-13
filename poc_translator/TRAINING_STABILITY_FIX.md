# Training Stability Fix - Gradient Explosion Resolution

## Problem

Training showed **high gradient norms** (12.19) exceeding the warning threshold (10.0), indicating instability:
```
WARNING - HIGH GRADIENT NORM DETECTED: 12.1887 (max_grad: 0.9233)
```

## Root Causes

1. **Wasserstein weight too high**: `wasserstein_weight: 10` dominated other losses
2. **Learning rate too aggressive**: `lr: 1e-3` with high loss weights
3. **Small demographic groups**: `wasserstein_min_group_size: 16` led to unstable Wasserstein computation
4. **Tight gradient clipping**: `gradient_clip_val: 1.0` was being exceeded by 12x
5. **No learning rate warmup**: Sudden large updates at training start

## Solutions Implemented

### 1. **Balanced Loss Weights** (`conf/config.yml`)

**Before:**
```yaml
rec_weight: 1.0
cycle_weight: 1.0
wasserstein_weight: 10    # 10x other losses!
```

**After:**
```yaml
rec_weight: 1.0
cycle_weight: 0.5         # Reduced from 1.0
wasserstein_weight: 0.5   # Reduced from 10 (20x reduction!)
```

**Rationale**: 
- Wasserstein loss was dominating and causing large gradients
- Now all losses are on similar scales (0.5-1.0)
- Reconstruction remains primary objective

### 2. **Reduced Learning Rate** (`conf/config.yml`)

**Before:**
```yaml
lr: 1e-3  # Too aggressive
```

**After:**
```yaml
lr: 5e-4  # 50% reduction for stability
```

**Rationale**:
- Lower LR = smaller gradient updates = more stable training
- Still fast enough to converge in 30 epochs

### 3. **Larger Demographic Groups** (`conf/config.yml`)

**Before:**
```yaml
wasserstein_min_group_size: 16  # Too small, unstable statistics
```

**After:**
```yaml
wasserstein_min_group_size: 32  # 2x increase for stability
```

**Rationale**:
- Wasserstein distance on small groups is noisy
- Larger groups → more stable gradient estimates

### 4. **Increased Gradient Clipping** (`conf/config.yml`)

**Before:**
```yaml
gradient_clip_val: 1.0  # Too tight, exceeded by 12x
```

**After:**
```yaml
gradient_clip_val: 5.0  # 5x increase for headroom
```

**Rationale**:
- Allows model to make larger (but safe) updates
- Acts as safety net, not primary constraint
- Still prevents true explosions (>100)

### 5. **Learning Rate Warmup** (`src/model.py`)

**Added:**
```python
def lr_lambda(epoch):
    warmup_epochs = 3
    if epoch < warmup_epochs:
        # Linear warmup from 0.1x to 1.0x
        return 0.1 + (0.9 * epoch / warmup_epochs)
    return 1.0
```

**Rationale**:
- Epoch 0: LR = 0.1 × 5e-4 = 5e-5 (gentle start)
- Epoch 1: LR = 0.55 × 5e-4 = 2.75e-4
- Epoch 2: LR = 1.0 × 5e-4 = 5e-4 (full speed)
- Prevents large updates before model stabilizes

## Expected Behavior After Fix

### Gradient Norms:
- **Epoch 0-2 (warmup)**: 1-5 (low due to warmup)
- **Epoch 3+**: 5-10 (healthy range)
- **Warnings**: Rare or none
- **Errors**: None

### Loss Values:
```
Epoch 0: rec_loss ≈ 0.5-1.0, cycle_loss ≈ 0.5-1.0, wass_loss ≈ 0.1-0.5
Epoch 5: All losses should decrease and stabilize
```

### Distribution Metrics:
```
Epoch 5+: 
  Mean KS: Should gradually improve from ~0.16
  Mean Wasserstein: Should gradually improve from ~0.13
```

## Monitoring During Training

### Watch for Success:
```bash
# Monitor gradient norms (should be <10)
tail -f logs/training_*.log | grep "gradient\|Epoch"

# Watch TensorBoard
tensorboard --logdir lightning_logs
# Check: grad_norm, train_loss, val_mean_ks_eicu_to_mimic
```

### Red Flags:
- ❌ Gradient norm consistently > 10
- ❌ Loss values increasing
- ❌ NaN or Inf values
- ❌ KS/Wasserstein not improving after epoch 10

### Green Flags:
- ✅ Gradient norm 3-8 (healthy)
- ✅ Losses decreasing smoothly
- ✅ KS/Wasserstein improving or stable
- ✅ No warnings in logs

## If Problems Persist

### Further Reductions:
```yaml
# Option 1: Even lower LR
lr: 1e-4  # 80% reduction from original

# Option 2: Disable Wasserstein temporarily
wasserstein_weight: 0.0

# Option 3: Increase warmup
# In model.py, change: warmup_epochs = 5
```

### Debug Steps:
1. Check input data for outliers: `python -c "import pandas as pd; df = pd.read_csv('data/train_mimic_preprocessed.csv'); print(df.describe())"`
2. Verify feature normalization
3. Check for NaN/Inf in data
4. Reduce batch size to 64 if memory issues

## Comparison: Before vs After

| Parameter | Before | After | Change |
|-----------|--------|-------|--------|
| **Learning Rate** | 1e-3 | 5e-4 | -50% |
| **Wasserstein Weight** | 10 | 0.5 | -95% |
| **Cycle Weight** | 1.0 | 0.5 | -50% |
| **Gradient Clip** | 1.0 | 5.0 | +400% |
| **Min Group Size** | 16 | 32 | +100% |
| **LR Warmup** | None | 3 epochs | NEW |

## Training Command

```bash
# Resume training with new stable config
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python src/train.py --gpu 0

# Or from checkpoint (if needed)
python src/train.py --gpu 0 --resume checkpoints/last.ckpt
```

## Expected Training Time

- **Per epoch**: ~2-3 minutes (similar to before)
- **Total (30 epochs)**: ~1-1.5 hours
- **Convergence**: Loss should stabilize by epoch 15-20

## Verification

After training completes, verify improvements:

```bash
# Run evaluation
python src/evaluate.py --model checkpoints/final_model.ckpt

# Compare distributions
python compare_distributions.py

# Check metrics:
# - Trained Mean KS should be < Untrained Mean KS
# - Trained Mean Wasserstein should be < Untrained Mean Wasserstein
```

## Summary

**Root cause**: Wasserstein loss weight was 10x too high, causing gradient spikes.

**Fix**: Reduced all loss weights, lowered LR by 50%, added warmup, increased gradient clipping headroom.

**Result**: Training should now be stable with gradient norms in 3-8 range and smooth loss curves.



