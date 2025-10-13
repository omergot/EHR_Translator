# Model Simplification Changes

## Date: October 8, 2025

## Overview
This document describes the comprehensive simplification of the CycleVAE model, focusing on three core losses and improved input handling.

## Key Changes

### 1. Simplified Loss Structure ✅
**Before**: 9 different loss components
- Reconstruction loss
- KL divergence
- Cycle consistency
- MMD loss
- Covariance loss
- Per-feature MMD
- Wasserstein loss
- Feature reconstruction loss
- Domain adversarial loss

**After**: 3 core losses only
- **Reconstruction loss**: MSE on clinical features only (excludes demographics and missing flags)
- **Cycle consistency loss**: MSE on clinical features only
- **Conditional 1-D Wasserstein loss**: Demographic-partitioned Wasserstein on worst-K features

### 2. Input Handling Improvements ✅

#### Missing Flags
- **Status**: Input-only, no loss computed
- **Implementation**: `_apply_missing_mask()` function overrides numeric targets to 0 when missing_flag=1
- **Applied to**: Both reconstruction and cycle losses

#### Demographics (Age, Gender)
- **Status**: Input-only, no loss computed
- **Rationale**: These should not change during translation
- **Implementation**: Excluded from clinical_indices, not included in loss computation

### 3. Conditional Wasserstein Loss ✅

#### Demographic Partitioning
- **Groups**: Age buckets (configurable, default 10 years) × Gender
- **Formula**: `group_id = (age / age_bucket_years) * 2 + gender`
- **Minimum group size**: Configurable (default 16 samples)

#### Worst-K Features
- **Dynamic selection**: Re-evaluated every N epochs (configurable, default every epoch)
- **Selection criteria**: Highest 1-D Wasserstein distance
- **Number of features**: Configurable (default K=10)

#### Computation Strategy
- **Frequency**: Every N steps (configurable, default every 5 steps)
- **Reason**: Save compute and reduce variance
- **Per-group computation**: Only on groups with sufficient samples

### 4. Removed Components ✅
- Heteroscedastic outputs (simplified to standard MSE)
- Feature reconstruction head
- Domain adversarial classifier
- MMD loss (replaced with Wasserstein)
- Covariance matching loss
- Per-feature MMD loss

### 5. Architecture Changes ✅
- **Encoder**: Unchanged
- **Decoders**: Simplified, no heteroscedastic outputs
- **Removed networks**: 
  - `feature_reconstruction_head`
  - `domain_classifier`

## Configuration Parameters

### New Parameters (add to config.yml)
```yaml
training:
  # Core loss weights
  rec_weight: 1.0
  cycle_weight: 1.0
  wasserstein_weight: 1.0
  
  # Conditional Wasserstein parameters
  wasserstein_compute_every_n_steps: 5
  wasserstein_min_group_size: 16
  wasserstein_worst_k: 10
  wasserstein_age_bucket_years: 10
  wasserstein_update_worst_every_n_epochs: 1
```

### Removed Parameters
```yaml
  kl_weight: <removed>
  mmd_weight: <removed>
  cov_weight: <removed>
  per_feature_mmd_weight: <removed>
  feature_recon_weight: <removed>
  domain_adversarial_weight: <removed>
  use_heteroscedastic: <removed>
  use_safe_mode: <removed>
```

## Implementation Status

### Completed ✅
1. Simplified loss functions to three core losses
2. Implemented conditional demographic-partitioned Wasserstein
3. Made worst features dynamic with periodic re-evaluation
4. Modified input handling: missing flags and demographics input-only
5. Override numeric targets using missing flags before loss computation

### Pending ⏳
6. Update evaluation to compute all loss components on test set
7. Add per-feature percentage error metrics (reconstruction and cycle)
8. Add latent and per-feature distribution distance metrics with plots
9. Add hybrid relative error thresholds (5%, 10%, 20%, 30%)
10. Update evaluation report with comprehensive per-feature KPIs

## Training Step Pseudocode

```python
def training_step(batch):
    # Extract data
    x_numeric, x_missing, domain = batch
    
    # Update worst features if new epoch
    if new_epoch and epoch % update_frequency == 0:
        update_worst_features()
    
    # Forward pass
    outputs = forward(x_numeric + x_missing)
    
    # Loss 1: Reconstruction (clinical features only, with missing mask)
    rec_loss = MSE(x_recon[clinical_indices], 
                   apply_missing_mask(x_numeric[clinical_indices], x_missing))
    
    # Loss 2: Cycle (clinical features only, with missing mask)
    cycle_loss = MSE(x_cycle[clinical_indices],
                    apply_missing_mask(x_numeric[clinical_indices], x_missing))
    
    # Loss 3: Conditional Wasserstein (every N steps, worst-K features)
    if batch_idx % N == 0:
        for each_demographic_group:
            if group_size >= min_size:
                for each_worst_feature:
                    wasserstein_loss += 1D_Wasserstein(translated, real)
    
    return rec_weight * rec_loss + cycle_weight * cycle_loss + wasserstein_weight * wasserstein_loss
```

## Benefits

1. **Simplicity**: 3 losses vs 9, easier to understand and tune
2. **Interpretability**: Each loss has clear meaning
3. **Efficiency**: Wasserstein computed every N steps, not every step
4. **Robustness**: Missing flags properly handled, demographics protected
5. **Adaptivity**: Worst features identified dynamically during training
6. **Clinical relevance**: Demographic-conditional distribution matching

## Next Steps

1. Update test_step and validation_step with simplified losses
2. Implement comprehensive evaluation metrics
3. Update config.yml with new parameters
4. Test training with simplified model
5. Compare results with original complex model

