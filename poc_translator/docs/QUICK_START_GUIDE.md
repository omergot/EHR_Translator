# Quick Start Guide: Simplified CycleVAE

## What Changed?

Your model is now **dramatically simpler** with only 3 core losses instead of 9 complex ones:

1. **Reconstruction Loss** - How well the model reconstructs clinical features
2. **Cycle Loss** - How well round-trip translation preserves features  
3. **Conditional Wasserstein Loss** - How well translated data matches real data within demographic groups

## How to Run Training

```bash
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator

# Train the model
python src/train.py --config conf/config.yml

# With specific GPU
python src/train.py --config conf/config.yml --gpu 0
```

## Configuration

All settings are in `conf/config.yml`. Key parameters:

```yaml
training:
  rec_weight: 1.0                              # Reconstruction loss weight
  cycle_weight: 1.0                            # Cycle loss weight
  wasserstein_weight: 1.0                      # Wasserstein loss weight
  wasserstein_compute_every_n_steps: 5         # Compute every N steps
  wasserstein_worst_k: 10                      # Target worst-K features
  wasserstein_age_bucket_years: 10             # Age bucket size
  wasserstein_update_worst_every_n_epochs: 1   # Update worst features every N epochs
```

## What to Monitor During Training

Watch these metrics in the logs:

- `train_rec_loss` - Should decrease steadily
- `train_cycle_loss` - Should decrease steadily  
- `train_wasserstein_loss` - Should decrease (computed every N steps)
- `train_loss` - Total loss (sum of the three)

You'll also see:
```
Updated worst-10 features: indices=[3, 7, 12, ...], Wasserstein distances=[0.45, 0.38, ...]
```
This shows which features are hardest to match (updated every epoch).

## Key Differences from Before

### ✅ What Was Removed
- KL divergence loss
- MMD loss  
- Covariance loss
- Per-feature MMD loss
- Feature reconstruction head
- Domain adversarial classifier
- All heteroscedastic outputs

### ✅ What Was Improved
- **Missing flags**: Now input-only (model doesn't try to predict them)
- **Demographics**: Input-only (Age/Gender don't change during translation)
- **Worst features**: Identified dynamically during training
- **Conditional Wasserstein**: Matches distributions within demographic groups

### ✅ What Was Added
- Comprehensive evaluation helper methods
- Per-feature percentage error metrics  
- Hybrid relative error (handles zeros gracefully)
- Latent space distance metrics
- Distribution distance metrics per feature

## Running Evaluation

```bash
# Run comprehensive evaluation
python src/evaluate.py --config conf/config.yml \
    --model output/checkpoints/final_model.ckpt \
    --comprehensive
```

## Interpreting Results

### Loss Values
- **Reconstruction/Cycle Loss**: Lower is better. Target: < 1.0 for normalized features
- **Wasserstein Loss**: Lower is better. Target: < 0.5 per demographic group

### Evaluation Metrics
- **% within 10%**: What percentage of predictions are within 10% of true value
  - Target: > 80% for most clinical features
- **% within 0.5 IQR**: Robust metric for features with zeros
  - Target: > 70% for most features
- **MAE**: Mean absolute error in original units (bpm, mmol/L, etc.)
  - Interpret clinically: Is 5 bpm error acceptable for heart rate?

## Tuning Tips

### If reconstruction is poor:
- Increase `rec_weight` (e.g., to 2.0 or 5.0)
- Check if data preprocessing is correct

### If cycle consistency is poor:
- Increase `cycle_weight` (e.g., to 2.0)
- Model may need more epochs

### If distribution matching is poor:
- Increase `wasserstein_weight` (e.g., to 2.0)
- Decrease `wasserstein_compute_every_n_steps` (compute more often)
- Increase `wasserstein_worst_k` (target more features)

### If training is slow:
- Increase `wasserstein_compute_every_n_steps` (compute less often)
- Decrease `batch_size`

## Common Issues

### "No common demographic groups"
- Your batch is too small or demographics are too imbalanced
- Solution: Increase `batch_size` or decrease `wasserstein_min_group_size`

### "Worst features not updating"
- Normal if features are stable
- Check logs for "Updated worst-K features" message every N epochs

### "Loss is NaN"
- Should be very rare with new simplified model
- If it happens, check data for extreme values

## Next Steps

1. **Train the model** with default settings
2. **Check test losses** - all three should be reasonable
3. **Run comprehensive evaluation** - see detailed per-feature metrics
4. **Tune weights** based on which aspect needs improvement
5. **Generate plots** using evaluation metrics

## Files to Check

- **Logs**: `logs/training_YYYYMMDD_HHMMSS.log` - Full training log
- **Checkpoints**: `checkpoints/final_model.ckpt` - Trained model
- **Config**: `config_used.yml` - Actual config used for training
- **Evaluation**: `evaluation/` - All evaluation outputs

## Questions?

See these detailed docs:
- `SIMPLIFICATION_CHANGES.md` - What changed and why
- `IMPLEMENTATION_COMPLETE.md` - Complete technical details
- `conf/config.yml` - All available parameters

## Model Architecture Summary

```
Input: [clinical_features | Age | Gender | missing_flags]
         ↓
      Encoder → Latent Space (256-dim)
         ↓
   Decoder (domain-specific)
         ↓
Output: [clinical_features | Age | Gender | missing_flags]

Losses:
1. Reconstruction: clinical_features only (with missing mask)
2. Cycle: clinical_features only (with missing mask)
3. Wasserstein: clinical_features, per demographic group, worst-K features
```

That's it! The model is much simpler now while being more effective for your use case.

