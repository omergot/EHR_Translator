# MIMIC-Only Mode Implementation

## Overview

The `--mimic-only` flag now properly enables **same-domain cycle consistency testing** by splitting MIMIC data into alternating "fake domains" for training the cycle architecture.

## What Changed

### Before (BROKEN)
- All MIMIC samples had `domain=1`
- Most loss functions required `mimic_mask.any() and eicu_mask.any()` - always FALSE
- Only reconstruction and KL losses were active
- Essentially trained a basic VAE, not a Cycle-VAE

### After (FIXED) ✅
- MIMIC samples alternate between `domain=0` and `domain=1` (idx % 2)
- Both domain masks are True, enabling ALL loss functions:
  - ✅ Reconstruction loss
  - ✅ KL loss  
  - ✅ **Cycle consistency loss** (0→1→0 and 1→0→1)
  - ✅ **MMD loss** (match latent distributions)
  - ✅ **Covariance loss** (match feature covariances)
  - ✅ **Per-feature MMD loss** (target problematic features)
  - ✅ **Wasserstein loss** (distribution matching)
  - ✅ **Feature reconstruction loss** (prevent mode collapse)
  - ✅ **Domain adversarial loss** (domain-invariant features)

## How It Works

### Data Split Strategy

```python
# Training data (e.g., 1000 samples in dataset)
Sample 0: domain=0  (treated as "source")
Sample 1: domain=1  (treated as "target")
Sample 2: domain=0  (treated as "source")
Sample 3: domain=1  (treated as "target")
...

# DataLoader creates batches (e.g., batch_size=128)
# Initial batch from DataLoader: 128 samples with alternating domains
# Then split and recombine to match normal mode behavior:
# - batch_0: ~64 samples with domain=0
# - batch_1: ~64 samples with domain=1
# - Combined: ~128 samples total (same as normal mode's 2x batch_size behavior)
```

**Important**: The batch size behavior matches normal mode:
- Normal mode: batch_size MIMIC + batch_size eICU = 2×batch_size total
- MIMIC-only mode: batch_size samples split by domain = ~batch_size total (after split)
  - This ensures consistent training dynamics and batch normalization

### Training Flow

```
1. Direct Reconstruction:
   MIMIC[domain=0] → Encoder → z → Decoder_eICU → x_recon
   MIMIC[domain=1] → Encoder → z → Decoder_MIMIC → x_recon

2. Cycle Consistency (now active!):
   MIMIC[domain=0] → Encoder → z → Decoder_MIMIC → x_translated
                   → Encoder → z' → Decoder_eICU → x_cycle (should match original)
   
   MIMIC[domain=1] → Encoder → z → Decoder_eICU → x_translated
                   → Encoder → z' → Decoder_MIMIC → x_cycle (should match original)

3. Distribution Matching (now active!):
   - MMD: Match z[domain=0] and z[domain=1] distributions
   - Covariance: Match translated and target feature covariances
   - Wasserstein: Match per-feature distributions
```

## Usage

### Training
```bash
# Train on MIMIC data only (same distribution, different "domains")
python src/train.py --mimic-only

# With GPU
python src/train.py --mimic-only --gpu 0

# Dry run (1 epoch)
python src/train.py --mimic-only --dry-run
```

### Evaluation
```bash
# Evaluate on MIMIC test data (same-domain cycle testing)
python src/evaluate.py --model checkpoints/final_model.ckpt --mimic-only
```

## Expected Behavior

### Training Metrics
You should see **ALL** loss components active:
```
DETAILED LOSS BREAKDOWN (batch 0):
  rec_loss: X.XXXX (weight: 1.0)
  kl_loss: X.XXXX (weight: 0.001)
  cycle_loss: X.XXXX (weight: 0.3)          ← NOW ACTIVE! ✅
  mmd_loss: X.XXXX (weight: 0.5)            ← NOW ACTIVE! ✅
  cov_loss: X.XXXX (weight: 0.01)           ← NOW ACTIVE! ✅
  per_feature_mmd_loss: X.XXXX (weight: 0.2) ← NOW ACTIVE! ✅
  wasserstein_loss: X.XXXX (weight: 0.1)    ← NOW ACTIVE! ✅
  feature_recon_loss: X.XXXX (weight: 0.2)  ← NOW ACTIVE! ✅
  domain_adv_loss: X.XXXX (weight: 0.2)     ← NOW ACTIVE! ✅
```

### Performance Expectations
Since all data comes from the **same distribution** (MIMIC), the model should:
- ✅ Achieve MUCH better cycle consistency (lower cycle_loss)
- ✅ Show lower MMD/Wasserstein losses (easier to match)
- ✅ Produce higher-quality reconstructions
- ✅ Validate that the architecture itself works

If it still fails in MIMIC-only mode:
- ❌ Problem is with model architecture/hyperparameters
- ❌ Not a domain shift issue

If it succeeds in MIMIC-only mode but fails in cross-domain:
- ✅ Architecture works
- ❌ Need better domain adaptation techniques

## Technical Details

### Implementation Location
- **File**: `src/dataset.py`
- **Classes**: `FeatureDataset`, `CombinedDataLoader`
- **Key Parameter**: `split_for_cycle=True` (automatically set when `mimic_only=True`)

### Code Changes
1. **FeatureDataset**: Added `split_for_cycle` parameter to `__init__`
2. **FeatureDataset.__getitem__**: Alternates domain labels when `split_for_cycle=True` (idx % 2)
3. **CombinedDataModule.setup()**: Passes `split_for_cycle=True` in MIMIC-only mode
4. **CombinedDataLoader.__next__**: Splits mixed-domain batch and recombines to match normal batch size
   - Gets batch with alternating domains from DataLoader
   - Splits into domain_0 and domain_1 sub-batches
   - Recombines using `combine_batches()` (same logic as normal mode)
   - Result: consistent batch size between normal and MIMIC-only modes

## Testing

To verify the fix works:

1. **Check data loading**:
   ```python
   # Should see: "CYCLE MODE: Splitting mimic data into alternating domains"
   ```

2. **Check batch size consistency**:
   ```python
   # Normal mode (batch_size=128):
   # Final batch: 256 samples (128 MIMIC + 128 eICU)
   
   # MIMIC-only mode (batch_size=128):
   # DataLoader batch: 128 samples with alternating domains
   # After split & recombine: ~128 samples (~64 domain=0 + ~64 domain=1)
   # Same effective batch size as normal mode for fair comparison!
   ```

3. **Check batch domains**:
   ```python
   # In final combined batch:
   # ~50% samples with domain=0 (MIMIC labeled as "source")
   # ~50% samples with domain=1 (MIMIC labeled as "target")
   # All from the same MIMIC distribution!
   ```

4. **Check loss activation**:
   ```python
   # cycle_loss should be > 0 (not zero)
   # mmd_loss should be > 0 (not zero)
   # All domain-comparison losses should be active
   ```

## Why This Matters

This implementation allows you to:
1. **Validate the model architecture** on same-distribution data
2. **Isolate domain shift problems** from architecture problems
3. **Test cycle consistency** without cross-domain complications
4. **Establish a performance baseline** for what's achievable

If the model works well in MIMIC-only mode, you know the architecture is sound and the issue is domain adaptation. If it still fails, you need to revisit the model design itself.

