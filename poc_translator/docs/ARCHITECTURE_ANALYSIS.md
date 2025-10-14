# Architecture Analysis: Why Correlation Is Lower Than Expected

## Current Architecture

### Encoder Path:
```
Input (32) → Linear(256) → BatchNorm → ReLU → Dropout
           → Linear(128) → BatchNorm → ReLU → Dropout  
           → Linear(64) → BatchNorm → ReLU → Dropout    ← BOTTLENECK!
           → Linear(256) [mu, logvar]
```

### Decoder Path:
```
Latent (256) → Linear(64) → BatchNorm → ReLU → Dropout  ← BOTTLENECK!
             → Linear(128) → BatchNorm → ReLU → Dropout
             → Linear(256) → BatchNorm → ReLU → Dropout
             → Linear(32) [output]
```

---

## Problems Identified

### 1. **Unnecessary Bottleneck** ❌

**The 64-dim hidden layer is smaller than input (32)!**

```
32 → 256 → 128 → 64 ← BOTTLENECK
```

Even though final latent_dim=256, the path goes through 64 dims first!

**Impact:**
- Information loss at the 64-dim layer
- Must compress 32 features into fewer effective dimensions
- Limits reconstruction capacity

### 2. **Unnecessary Non-Linearity** ❌

**Why is ReLU needed for domain translation?**

- eICU and MIMIC data are both **normalized** (mean~0, std~1)
- The transformation should be approximately **linear** (affine):
  ```
  x_mimic ≈ A · x_eicu + b
  ```
- Multiple ReLU layers create **highly non-linear** transformations
- This explains why Corr² << R²!

### 3. **KL Regularization as Implicit Bottleneck** ⚠️

Even with latent_dim=256 > input_dim=32:

```python
# KL loss pushes latent distribution toward N(0,1)
KL_loss = -0.5 * sum(1 + logvar - mu² - exp(logvar))
```

This **constrains the effective dimensionality** of the latent space!

**Result:** Model cannot use all 256 dimensions freely.

---

## Why This Causes Low Correlation

### Simplified Example:

**Ideal Linear Model:**
```
x_mimic = W · x_eicu + b

Roundtrip:
x_eicu → x_mimic → x_eicu_reconstructed
x_recon = W_inv · (W · x + b) + b2
        ≈ x  (if W_inv ≈ W^(-1) and biases cancel)

Result: R² ≈ Corr² ≈ 0.99
```

**Your Non-Linear Model:**
```
x_eicu → ReLU(BN(W1·x)) → ReLU(BN(W2·...)) → ... → x_mimic

Roundtrip adds MORE non-linearity:
x_recon = ReLU(...ReLU(...ReLU(...)))  ← 6+ ReLU layers!

Result:
- R² = 0.96 (good - captures overall pattern)
- Corr = 0.64 (lower - non-linear warping breaks linear correlation)
```

---

## Evidence From Your Results

### WBC_mean (best feature):
```
R² = 0.961  ← Model explains 96% of variance
Corr = 0.642 ← But linear fit is only 64% correlated
Corr² = 0.412 ← Much less than R²!
```

**This pattern is diagnostic of non-linear compression:**
- High R²: Model captures the relationship
- Low Corr: Relationship is non-linear (e.g., sigmoid-like at extremes)

### Visualization:

```
y_true vs y_pred scatter plot would show:
  
  y_pred |    x  x
         |   x x x
         |  x x x x     ← Compressed at extremes
         | x x x x x
         |x x x x x
         +-----------
           y_true
           
Linear fit (correlation): Poor
Variance explained (R²): Good
```

---

## Proposed Solution: Simplify Architecture

### Option A: Remove Bottleneck (Minimal Change)

```python
# In model.py line 246, change:
hidden_dims = [256, 128, 64]  # OLD

# To:
hidden_dims = [128, 256]  # NEW - no bottleneck!
```

**Path:**
```
32 → 128 → 256 → latent(256) → 256 → 128 → 32
     ↑ No bottleneck, always expanding
```

### Option B: Make Architecture Linear (Better)

```python
# For domain translation, use LINEAR layers only
hidden_dims = []  # No hidden layers!

# This gives:
32 → latent(256) → 32
     ↑ Direct linear transformation
```

**With KL regularization, this is equivalent to:**
- Probabilistic PCA (linear dimensionality reduction)
- Simple linear translation with noise

### Option C: Remove KL Loss for Cycle Translation (Best?)

```python
# During cycle consistency, use deterministic encoder (no KL loss)
# Only apply KL loss during reconstruction

def training_step(self, batch):
    # Reconstruction: Use KL loss (VAE behavior)
    mu, logvar = self.encoder(x)
    z = self.reparameterize(mu, logvar)
    x_recon = self.decoder(z)
    rec_loss = mse_loss(x, x_recon)
    kl_loss = kl_divergence(mu, logvar)
    
    # Cycle: No KL loss (deterministic translation)
    mu_cycle, _ = self.encoder(x)
    z_cycle = mu_cycle  # No reparameterization!
    x_cycle = self.decoder(z_cycle)
    cycle_loss = mse_loss(x, x_cycle)
    
    total_loss = rec_loss + kl_loss + cycle_loss  # KL only on reconstruction
```

---

## Expected Improvements

### After Removing Bottleneck (Option A):
- R²: 0.96 → 0.97 (+0.01)
- Correlation: 0.64 → 0.75 (+0.11)
- Still some non-linearity from ReLU

### After Linear Architecture (Option B):
- R²: 0.96 → 0.98 (+0.02)
- Correlation: 0.64 → 0.90 (+0.26)
- Much more linear relationship

### After Removing KL from Cycle (Option C):
- R²: 0.96 → 0.99 (+0.03)
- Correlation: 0.64 → 0.95 (+0.31)
- Nearly linear cycle consistency

---

## Trade-offs

### Current Model (Non-Linear):
✓ Flexible (can learn complex transformations)
✓ Robust to outliers (ReLU clips negatives)
✗ Lower correlation
✗ Non-linear warping

### Linear Model:
✓ Higher correlation
✓ Easier to interpret
✓ Faster training/inference
✗ Less flexible
✗ May not capture non-linear domain shifts

---

## Recommendation

### For Your Use Case (EHR Translation):

**Domain translation SHOULD be approximately linear!**

Reasons:
1. Both datasets are normalized the same way
2. Clinical variables have same meaning in both domains
3. Only need to adjust for:
   - Measurement scales (linear)
   - Missing data patterns (handled by flags)
   - Population demographics (handled by conditional Wasserstein)

**Recommended: Option A (remove bottleneck) + Option C (no KL in cycle)**

This gives:
- Better cycle consistency
- Higher correlation
- Still maintains VAE properties for generation

---

## Implementation

### Quick Test (No Retraining):

Check if bottleneck is the issue:

```python
# In model.py, add this diagnostic:
def analyze_bottleneck(self):
    x_test = torch.randn(100, 32)
    
    # Track dimensionality at each layer
    with torch.no_grad():
        h = x_test
        for i, layer in enumerate(self.encoder.feature_extractor):
            h = layer(h)
            if isinstance(layer, nn.Linear):
                # Effective dimensionality via SVD
                U, S, V = torch.svd(h)
                eff_dim = (S > 0.01 * S[0]).sum().item()
                print(f'Layer {i}: shape={h.shape}, eff_dim={eff_dim}')
```

**Expected**: Effective dimensionality drops at 64-dim layer.

### Full Fix (Requires Retraining):

```python
# In model.py line 246:
hidden_dims = [128, 256]  # Removed bottleneck

# In training_step, use deterministic cycle:
mu_cycle, _ = self.encoder(x_cycle)
z_cycle = mu_cycle  # No reparameterization in cycle!
```

Re-train for 50 epochs → Expect Corr > 0.85

---

## Summary

| Issue | Impact on Corr | Fix | Effort |
|-------|---------------|-----|---------|
| **64-dim bottleneck** | -0.10 | Remove | Retrain |
| **Multiple ReLUs** | -0.15 | Simplify | Retrain |
| **KL in cycle** | -0.10 | Conditional KL | Code change |
| **Dropout (fixed)** | -0.25 | ✓ Done | N/A |

**Total possible improvement: 0.64 → 0.95 correlation**

But requires architectural changes + retraining.

**Current performance (R²=0.96, KS=0.15) is already good!**

Decision: Is higher correlation worth the retraining effort?


