# Diagnostic Logging Summary

## Overview
Added comprehensive logging to diagnose the training loss explosion issue observed in the logs where losses jumped from ~1.8 to ~78 million.

## Changes Made

### 1. File-Based Logging (`src/train.py`)
**Purpose**: Ensure all training logs are captured to a file, not just console output.

**Changes**:
- Added `setup_file_logging()` function that creates timestamped log files
- Log files saved to: `logs/training_YYYYMMDD_HHMMSS.log`
- Dual output: Console (INFO level) + File (DEBUG level)
- Root logger configured to capture all module logs

**What This Captures**:
- All training progress messages
- All WARNING and ERROR messages from any module
- Complete history of the training run

### 2. Enhanced Loss Breakdown Logging (`src/model.py`)
**Purpose**: Detect concerning losses much earlier than before.

**Changes**:
- **LOWERED threshold** from `1e10` to `1e3` - will now catch explosions 10 million times sooner!
- Log detailed breakdown when:
  - First batch of each epoch (`batch_idx == 0`)
  - Every 50 batches (`batch_idx % 50 == 0`)
  - Any loss component > 1000 (`total_loss > 1e3`, `rec_loss > 1e3`, etc.)

**What This Captures**:
- Epoch and batch number
- All 9 loss components with their weights
- Total loss
- When loss > 100: Full tensor statistics (min/max/mean/std) for:
  - Input features (`x`)
  - Reconstructed features (`x_recon`)
  - Latent representations (`z`)

### 3. Gradient Norm Monitoring (`src/model.py`)
**Purpose**: Detect gradient explosions before they cause NaN/Inf values.

**Changes**:
- Added `on_before_optimizer_step()` callback
- Computes total gradient norm across all parameters
- Logs `grad_norm` and `grad_max` to tensorboard

**What This Captures**:
- WARNING when gradient norm > 10
- ERROR when gradient norm > 100 with:
  - Epoch and global step
  - Total norm, max gradient, min gradient
  - Top 5 parameters with largest gradients (identifies which layers are exploding)

### 4. Decoder Output Monitoring (`src/model.py`)
**Purpose**: Catch extreme decoder outputs that lead to loss explosions.

**Changes**:
- Warning when decoder outputs exceed ±20
- Logs both decoder output and input latent (`z`) statistics

**What This Captures**:
- Which decoder (MIMIC or eICU) is producing extreme values
- The range and distribution of problematic outputs
- The corresponding latent space values (helps identify encoder issues)

### 5. Per-Feature Reconstruction Loss Analysis (`src/model.py`)
**Purpose**: Identify which specific features are causing reconstruction loss to explode.

**Changes**:
- When continuous reconstruction loss > 100, logs:
  - Total continuous loss value
  - Index of worst-performing feature
  - Input range and reconstruction range for worst feature
  - Top 5 worst features and their individual losses

**What This Captures**:
- Exact feature indices causing problems
- Whether it's an input data issue (extreme values) or decoder issue (wrong predictions)
- Patterns across multiple problematic features

## Key Metrics Now Logged

### To File (timestamped in `logs/training_*.log`):
1. All console output
2. All WARNING/ERROR messages
3. Detailed loss breakdowns (every 50 batches + when concerning)
4. Gradient explosion diagnostics
5. Decoder output warnings
6. Per-feature reconstruction diagnostics

### To TensorBoard:
1. `grad_norm` - Total gradient norm
2. `grad_max` - Maximum gradient value
3. All existing loss components

## Expected Diagnostic Output

When the explosion occurs, you should now see logs like:

```
================================================================================
DETAILED LOSS BREAKDOWN (epoch 2, batch 123):
  rec_loss: 13792226.000000 (weight: 5.0)
  ...
  TOTAL: 78621624.000000

CONCERNING LOSS DETECTED - Adding decoder output statistics:
  x_recon: min=-45.2341, max=892.4532, mean=23.4123, std=123.4567
  x_input: min=-2.1234, max=3.4567, mean=0.1234, std=1.2345
  z: min=-8.9123, max=12.3456, mean=0.0123, std=2.3456
================================================================================

HIGH GRADIENT NORM DETECTED: 234.5678 (max_grad: 567.8901)

LARGE DECODER OUTPUT (MIMIC): min=-23.4567, max=823.4561, mean=45.2341
  Input z stats: min=-12.3456, max=15.6789, mean=0.2345

VERY HIGH CONTINUOUS RECONSTRUCTION LOSS: 2758445.200000
  Worst feature index: 14, loss: 1234567.890000
  Feature 14 - x range: [-2.3456, 3.4567]
  Feature 14 - x_recon range: [-456.7890, 823.4561]
  Top 5 worst features: [14, 7, 21, 3, 18]
  Their losses: [1234567.89, 234567.12, 123456.78, 98765.43, 87654.32]
```

## Next Steps

1. **Run training** with these enhanced logs
2. **Examine the log file** in `logs/training_*.log` when explosion occurs
3. **Look for**:
   - Which batch/epoch the explosion starts
   - Which loss component explodes first (rec_loss, cycle_loss, or feature_recon_loss)
   - Whether gradients explode first (gradient norm > 100)
   - Which decoder outputs extreme values
   - Which specific feature indices are problematic
   - Whether the issue is in the encoder (extreme z values) or decoder (extreme outputs from normal z)

4. **Based on findings**, implement targeted fixes:
   - If specific features: Add feature-specific clamping or normalization
   - If decoder: Add output constraints or architecture changes
   - If gradients: Adjust gradient clipping or learning rate
   - If precision: Switch from FP16 to FP32

## Log File Location

After running training, find your complete logs at:
```
/bigdata/omerg/Thesis/EHR_Translator/poc_translator/logs/training_YYYYMMDD_HHMMSS.log
```

The filename will be printed at the start of training.


