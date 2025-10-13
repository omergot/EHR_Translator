# Validation Split and Decoder Explosion Logging - Implementation Summary

## Date: October 13, 2025

## Overview
This document summarizes the changes made to:
1. Add validation data splits to enable `on_train_epoch_end()` metrics computation
2. Add comprehensive logging to diagnose decoder output explosions

---

## 1. Validation Split Implementation

### Changes Made:

#### A. Configuration (`conf/config.yml`)
- **Changed:** Data splits from train/test only to train/val/test
- **Before:** `train_split: 0.8, test_split: 0.2`
- **After:** `train_split: 0.7, val_split: 0.15, test_split: 0.15`

#### B. Preprocessing (`src/preprocess.py`)

**`split_data()` method (lines 798-834):**
- Now creates **three-way split** (train/val/test) instead of two-way (train/test)
- Uses stratified splitting to ensure proper patient-level separation
- First splits off test set, then splits remaining into train/val

**`save_splits()` method (lines 836-878):**
- Now saves **6 files** instead of 4:
  - `train_mimic_preprocessed.csv`
  - `val_mimic_preprocessed.csv` ✨ NEW
  - `test_mimic_preprocessed.csv`
  - `train_eicu_preprocessed.csv`
  - `val_eicu_preprocessed.csv` ✨ NEW
  - `test_eicu_preprocessed.csv`
- Updated `split_info.json` to include validation statistics

**Main preprocessing pipeline (lines 972-1038):**
- Updated to handle 3-tuple returns from `split_data()`
- Added validation data transformation step
- Added validation data to monotonicity verification
- Updated plotting to include validation data

#### C. DataModule (`src/dataset.py`)

**Added validation dataset attributes (lines 138-139):**
```python
self.val_mimic_dataset = None
self.val_eicu_dataset = None
```

**Updated `setup()` method (lines 152-178):**
- Now loads validation data from `val_mimic_preprocessed.csv` and `val_eicu_preprocessed.csv`
- Creates validation datasets for both MIMIC and eICU
- Supports both standard mode and MIMIC-only mode

**Added `val_dataloader()` method (lines 213-226):**
- Returns a `CombinedDataLoader` with validation data
- Uses `shuffle=False` for deterministic validation
- Uses same balance strategy as training for consistency

### Result:
✅ `on_train_epoch_end()` in `model.py` will now successfully call `val_dataloader()` and compute distribution metrics during training

---

## 2. Decoder Explosion Diagnostic Logging

### Changes Made:

#### A. Enhanced Forward Pass Logging (`src/model.py`, lines 434-494)

**MIMIC Decoder (lines 434-463):**
Added detailed breakdown when decoder output exceeds threshold (20):
- Final output statistics (min, max, mean)
- Decoder output **before skip connection**
- Skip connection contribution (separate from decoder)
- Input x statistics
- Latent z statistics  
- Skip connection parameters (scale and bias)
- Feature indices with extreme values

**eICU Decoder (lines 465-494):**
Same comprehensive logging as MIMIC decoder

**Example Output:**
```
⚠️  LARGE DECODER OUTPUT (MIMIC) - DETAILED BREAKDOWN:
  Final output: min=-24.5122, max=4.6557, mean=0.2571
  Decoder output (before skip): min=-15.2133, max=3.1234, mean=0.1234
  Skip contribution: min=-9.3456, max=1.8765, mean=0.0987
  Input x: min=-5.4321, max=8.7654, mean=0.3456
  Latent z: min=-11.6405, max=8.0213, mean=-0.0690
  Skip params - scale: min=0.8234, max=1.2345, mean=1.0123, std=0.0456
  Skip params - bias: min=-0.5432, max=0.3456, mean=-0.0123, std=0.1234
  Extreme values found at 42 locations
  Feature indices with extreme values (first 10): [3, 7, 12, 15, 18, 21, 24, 27, 30, 33]
```

#### B. Epoch-End Skip Parameter Logging (`src/model.py`, lines 990-1013)

**Added to `on_train_epoch_end()`:**
- Logs skip connection parameters at the **end of each epoch**
- Tracks evolution over training to identify divergence
- Logs to both console (logger.info) and metrics (self.log)

**Metrics logged:**
- `skip_scale_mimic_mean`, `skip_scale_mimic_max`
- `skip_bias_mimic_max_abs`
- `skip_scale_eicu_mean`, `skip_scale_eicu_max`
- `skip_bias_eicu_max_abs`

**Example Output:**
```
================================================================================
EPOCH 5 - Skip Connection Parameters:
  MIMIC Decoder:
    skip_scale: min=0.8234, max=1.2345, mean=1.0123, std=0.0456
    skip_bias:  min=-0.5432, max=0.3456, mean=-0.0123, std=0.1234
  eICU Decoder:
    skip_scale: min=0.7891, max=1.3456, mean=0.9987, std=0.0567
    skip_bias:  min=-0.6789, max=0.4567, mean=-0.0234, std=0.1456
================================================================================
```

#### C. Skip Parameter Gradient Logging (`src/model.py`, lines 1295-1315)

**Added to `on_before_optimizer_step()`:**
- Tracks gradients of skip connection parameters **at every step**
- Logs to metrics for continuous monitoring
- Warns when gradients are high

**Metrics logged:**
- `grad_skip_scale_mimic`
- `grad_skip_bias_mimic`
- `grad_skip_scale_eicu`
- `grad_skip_bias_eicu`

**Enhanced warnings:**
- When gradient norm > 10: Shows skip parameter gradients
- When gradient norm > 100: Full gradient explosion analysis

---

## 3. What to Look For in Next Training Run

### A. Validation Split
Watch for:
- ✅ No warning about `val_dataloader` not implemented
- ✅ Distribution metrics (KS statistics, Wasserstein distances) logged at epoch end
- ✅ Validation dataset sizes in logs

### B. Decoder Explosion Diagnosis

**When you see the warning again, the detailed breakdown will tell you:**

1. **If decoder output is the problem:**
   - Large "Decoder output (before skip)" values
   - → Problem is in the decoder network weights

2. **If skip connection is the problem:**
   - Large "Skip contribution" values
   - Large "Skip params - scale" or "bias" values
   - → Skip connection parameters are diverging

3. **If input is the problem:**
   - Large "Input x" values
   - → Data normalization or preprocessing issue

4. **If latent space is the problem:**
   - Large "Latent z" values
   - → Encoder producing extreme latent codes

5. **Which features are affected:**
   - "Feature indices with extreme values" shows which features
   - Cross-reference with feature_spec.json to identify features

### C. Tracking Over Time

**Epoch-end logs show:**
- If skip_scale is drifting far from 1.0
- If skip_bias is growing large
- Trends that might predict explosions

**Step-level gradient logs show:**
- If skip parameter gradients are consistently large
- If gradients spike before explosions

---

## 4. Next Steps After Training

Once you have the new logs, we can:

1. **Identify root cause** from the detailed breakdown
2. **Apply targeted fixes**:
   - If skip connection: Add regularization or constraints
   - If decoder: Adjust architecture or initialization
   - If latent space: Constrain encoder output
   - If specific features: Investigate those features

3. **Monitor progress** using the epoch-level metrics

---

## Files Modified

1. `/bigdata/omerg/Thesis/EHR_Translator/poc_translator/conf/config.yml`
2. `/bigdata/omerg/Thesis/EHR_Translator/poc_translator/src/preprocess.py`
3. `/bigdata/omerg/Thesis/EHR_Translator/poc_translator/src/dataset.py`
4. `/bigdata/omerg/Thesis/EHR_Translator/poc_translator/src/model.py`

## Action Required Before Training

⚠️ **You must re-run preprocessing** to generate the validation split files:

```bash
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python src/preprocess.py --config conf/config.yml --fit
```

This will create:
- `data/val_mimic_preprocessed.csv`
- `data/val_eicu_preprocessed.csv`

Without these files, training will fail when trying to load validation data.

