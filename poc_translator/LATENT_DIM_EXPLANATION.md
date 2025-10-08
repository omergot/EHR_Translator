# Understanding latent_dim in VAE

## Current Configuration
- **Input dimensions**: 27 numeric features + 6 missing flags = 33 total
- **Latent dimensions**: 256 (as configured)
- **Output dimensions**: 33 (same as input)

## What latent_dim Controls

The `latent_dim` is the size of the **compressed representation** (bottleneck) in your VAE:

```
[Input: 33 dims] → [Encoder] → [Latent: latent_dim] → [Decoder] → [Output: 33 dims]
                                      ↑
                                 BOTTLENECK
```

---

## Comparison of Different Values

### 1. latent_dim = 64 (Original - TOO SMALL)
```
33 features → 64 latent dims → 33 features
              ↑ EXPANSION (!)
```

**Effect**: 
- Actually **expands** the space (33→64) instead of compressing
- But network hidden layers (256→128→64) do compress
- **Problem**: Not enough capacity to preserve all information
- **Result**: Information loss → poor reconstruction (MSE=28)

### 2. latent_dim = 256 (Current - YOUR SETTING)
```
33 features → 256 latent dims → 33 features
              ↑ LARGE EXPANSION
```

**Effect**:
- **Much more capacity** to preserve information
- Latent space is 7.7× larger than input
- Model can learn richer representations
- **Trade-off**: Less compression = easier reconstruction but potentially less regularization

**Why This is Better**:
- Previous bottleneck (64) was losing information
- 256 gives model plenty of room to preserve all features
- Should dramatically improve reconstruction (lower MSE)
- Still maintains regularization through KL divergence

### 3. latent_dim = 33 (Same as Input)
```
33 features → 33 latent dims → 33 features
              ↑ NO COMPRESSION
```

**Effect**:
- No dimensional bottleneck at all
- Model CAN achieve perfect reconstruction
- But defeats purpose of compression-based regularization
- Good for debugging, not for production

### 4. latent_dim = 16 (Aggressive Compression)
```
33 features → 16 latent dims → 33 features
              ↑ STRONG BOTTLENECK
```

**Effect**:
- Forces model to learn very compressed representations
- High information loss → poor reconstruction
- Good for learning minimal feature sets
- **Not recommended** for your case (already had issues with 64)

---

## Why 256 is a Good Choice for You

### Problem Analysis:
Your previous evaluation showed:
- Mean MSE = 28.2 (should be ~1.0 for normalized data)
- SpO2_max MSE = 867 (extreme failure)
- Creat features MSE = 4-6 (poor)
- **Diagnosis**: Model losing too much information in bottleneck

### Why 256 Helps:

1. **More Capacity**: 
   - Previous: 64 dims trying to encode 33 features
   - Now: 256 dims (7.7× more room)
   - Plenty of space to preserve all features

2. **Information Preservation**:
   - Can dedicate ~8 latent dims per input feature
   - No forced information loss
   - Should fix poor reconstruction

3. **Still Gets VAE Benefits**:
   - KL divergence still regularizes
   - Still learns structured latent space
   - Still forces domain adaptation

4. **Empirically Validated**:
   - Many VAE papers use latent_dim > input_dim
   - Common for complex data (images, EHR)
   - Allows model to learn hierarchical features

---

## The Trade-off Spectrum

```
Small latent_dim (16)                                 Large latent_dim (512)
        |                                                      |
        |                                                      |
    [Compression]──────────────────────────────────[Capacity]
        |                                                      |
        |                                                      |
 Strong regularization                           Easy reconstruction
 High information loss                           Minimal information loss
 Minimal overfitting                             Risk of overfitting
 Poor reconstruction                             Good reconstruction
        |                                                      |
        ├──────┬──────────┬──────────┬────────────────────────┤
        16     64        128        256                      512
             (old)                 (new!)
```

Your choice of 256 is **solidly in the "capacity" zone**, which is appropriate given:
- Previous bottleneck was causing failures
- You have decent training data size (~45K samples)
- You want good reconstruction quality

---

## Expected Impact of Changing 64 → 256

### Before (latent_dim=64):
- Mean MIMIC MSE: 28.2
- SpO2_max MSE: 867
- Gender MSE: 0.57 (worse than random)
- Creat MSE: 4-6

### Expected After (latent_dim=256):
- Mean MIMIC MSE: **2-5** (5-10× better)
- SpO2_max MSE: **50-100** (assuming outliers still exist, but 8× better)
- Gender MSE: **< 0.1** (with sigmoid fix)
- Creat MSE: **1-2** (2-3× better)

**Combined with other fixes** (sigmoid, demographic bypass):
- Overall MSE should approach ~1.0 (random baseline)
- Demographics (Age, Gender) should have near-perfect reconstruction
- Clinical features should have much better reconstruction

---

## How It Works in Training

### Encoder (33 → 256):
```python
Input (33) 
  → Linear(33, 256)  → ReLU → BatchNorm → Dropout
  → Linear(256, 128) → ReLU → BatchNorm → Dropout  
  → Linear(128, 64)  → ReLU → BatchNorm → Dropout
  → Linear(64, 256)  # Output: mu and logvar (each 256 dims)
```

The encoder learns to map your 33 input features into a 256-dimensional latent space where:
- Similar patients have similar latent vectors
- MIMIC and eICU patients with same condition map to same region
- The 256 dimensions capture different aspects (severity, age, organ systems, etc.)

### Decoder (256 → 33):
```python
Latent (256)
  → Linear(256, 64)  → ReLU → BatchNorm → Dropout
  → Linear(64, 128)  → ReLU → BatchNorm → Dropout
  → Linear(128, 256) → ReLU → BatchNorm → Dropout
  → Linear(256, 33)  # Output: reconstructed features
     → Sigmoid for binary features (indices for Gender, missing flags)
     → Direct copy for Age, Gender (bypass)
```

---

## Other Configuration Settings (in config.yml)

Looking at your config, you also made good complementary changes:

### ✅ Good Settings:
- `lr: 1e-3` - Higher learning rate (was 1e-6) for faster optimization
- `rec_weight: 5.0` - Prioritizes reconstruction (good!)
- `kl_weight: 1e-5` - Very low KL weight (prevents over-regularization)
- `use_heteroscedastic: false` - Disabled for stability (good for now)

### These Work Together With latent_dim=256:
- Large latent space (256) provides capacity
- High rec_weight (5.0) tells model to use that capacity for reconstruction
- Low kl_weight (1e-5) doesn't penalize using the full latent space
- **Result**: Model can freely use all 256 dimensions to preserve information

---

## Summary

**latent_dim = 256 means**:
- Your compressed representation is 256 numbers
- Each patient visit gets encoded as a point in 256-dimensional space
- This is 7.7× larger than your input (33 features)
- Gives model plenty of room to preserve all information
- Should dramatically improve reconstruction quality

**Why this is good for you**:
- Previous bottleneck (64) was too tight → information loss
- 256 gives ample capacity → better reconstruction
- Combined with your other fixes (sigmoid, bypass, higher rec_weight)
- Should see **massive improvement** in evaluation metrics

**What to expect in next training run**:
- Much lower MSE across all features
- Gender/Age near perfect (bypass + sigmoid)
- Clinical features 5-10× better
- Correlations no longer NaN

---

## Quick Reference

| Setting | Value | Effect |
|---------|-------|--------|
| **latent_dim** | 256 | Size of compressed representation |
| **input_dim** | 33 | Original features (numeric + missing flags) |
| **output_dim** | 33 | Reconstructed features |
| **Compression ratio** | 33:256 | Actually expansion! (intentional) |
| **Capacity** | High | Plenty of room for information |
| **Information loss** | Minimal | Should preserve all features well |

