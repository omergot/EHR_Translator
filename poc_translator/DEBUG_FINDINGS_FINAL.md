# Debug Findings: Gradient Explosion & Correlation Issues

## Executive Summary

### Issue 1: Gradient Explosion ⚠️ CRITICAL
- **Symptom**: `inf` gradients in `decoder_*.fc_out.bias` at epochs 17, 22, 28, 34, 40, 45
- **Root Cause**: Unbounded decoder output + stochastic noise accumulation
- **Impact**: Training instability, potential model divergence

### Issue 2: Low Correlation Despite High R² 🔴 MAJOR BUG
- **Symptom**: R²=0.96 but Corr=0.39 (expected Corr=0.98 since R²=corr²)
- **Root Cause**: `translate_*` functions use `reparameterize()` which adds Gaussian noise
- **Impact**: Roundtrip evaluation is measuring noisy reconstruction, not deterministic quality

---

## Issue 1: Gradient Explosion Analysis

### Pattern Observations

**Explosion Events:**
| Epoch | Step  | Parameter | x_recon_max |
|-------|-------|-----------|-------------|
| 17    | 6101  | decoder_eicu.fc_out.bias | 4.3984 |
| 22    | 8111  | decoder_mimic.fc_out.bias | 4.1016 |
| 28    | 10174 | decoder_mimic.fc_out.bias | 4.0820 |
| 34    | 12192 | decoder_mimic.fc_out.bias | 4.6914 |
| 40    | 14335 | decoder_mimic.fc_out.bias | 4.3711 |
| 45    | 16385 | decoder_eicu.fc_out.bias | 4.3125 |

**Key Patterns:**
1. ✓ Alternates between `decoder_eicu` and `decoder_mimic`
2. ✓ Always in `fc_out.bias` (never weights!)
3. ✓ `x_recon_max` increases over time (4.1 → 4.7)
4. ✓ Data range is stable: x ∈ [-3.8, 4.2], z ∈ [-15.2, 15.2]

### Root Cause

```python
# In Decoder class (model.py:136)
self.fc_out = nn.Linear(prev_dim, output_dim)  # NO activation function!

# In Decoder.forward (model.py:161)
x_recon = self.fc_out(features)  # Unbounded output!
```

**The Problem:**
1. Decoder output has NO activation function → unbounded
2. When `fc_out.bias` drifts positive → outputs exceed data range
3. `x_recon` > 4.5 (while data max is 4.2) → large reconstruction loss
4. Large loss gradient → explodes bias further → `inf` gradient

**Why it alternates:**
- Training uses both decoders equally
- When one decoder explodes, gradients are clipped/skipped
- The other decoder continues training normally
- Eventually that one also explodes
- Cycle repeats

### Solution: Add Gradient Clipping

```python
# In training config (config.yml)
trainer:
  gradient_clip_val: 1.0  # Clip gradients to max norm of 1.0
  gradient_clip_algorithm: 'norm'
```

Or add output bounds:
```python
# In Decoder.forward (model.py:161)
x_recon = self.fc_out(features)
x_recon = torch.clamp(x_recon, min=-10, max=10)  # Enforce normalized data range
```

---

## Issue 2: Low Correlation - Stochastic Translation Bug

### The Discrepancy

**Observed:**
| Feature | R² | Correlation | Expected Corr (√R²) | Discrepancy |
|---------|-----|-------------|---------------------|-------------|
| WBC_mean | 0.961 | 0.385 | 0.980 | 0.596 |
| Creat_max | 0.958 | 0.327 | 0.979 | 0.652 |
| Na_mean | 0.958 | 0.474 | 0.979 | 0.505 |

**For simple linear regression: R² = corr²**
- Expected: corr ≈ 0.98
- Actual: corr ≈ 0.39
- **Ratio: ~0.4 consistently across all features!**

### Root Cause: Stochastic Reparameterization

**In `translate_eicu_to_mimic` (model.py:1070):**
```python
mu, logvar = self.encoder(x_eicu)
z = self.encoder.reparameterize(mu, logvar)  # ← ADDS NOISE!
x_mimic = self.decoder_mimic(z)
```

**The `reparameterize` function:**
```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std  # ← Gaussian noise!
```

**Impact on roundtrip:**
```
Original:     x_eicu
   ↓ (encode + reparameterize)
Latent:       z = μ₁ + ε₁·exp(0.5·logvar₁)  ← noise #1
   ↓ (decode to MIMIC)
Translated:   x_eicu_to_mimic
   ↓ (encode + reparameterize)
Latent:       z = μ₂ + ε₂·exp(0.5·logvar₂)  ← noise #2
   ↓ (decode back to eICU)
Roundtrip:    x_eicu_roundtrip
```

**Total noise: TWO independent Gaussian samples!**

### Why This Explains the Numbers

**Theoretical analysis:**
If we have:
- `y_pred = y_true + noise`
- `noise ~ N(0, σ²)`

Then:
```
R² = 1 - Var(noise) / Var(y_true)
corr² = Cov(y_true, y_pred)² / (Var(y_true) · Var(y_pred))
```

With noise:
```
Cov(y_true, y_pred) = Cov(y_true, y_true + noise) = Var(y_true)
Var(y_pred) = Var(y_true + noise) = Var(y_true) + σ²

corr² = Var(y_true)² / [Var(y_true) · (Var(y_true) + σ²)]
      = Var(y_true) / (Var(y_true) + σ²)
```

**If σ² = Var(y_true) (noise equals signal):**
- R² ≈ 0.5
- corr² ≈ 0.5
- corr ≈ 0.71

**If σ² = 5·Var(y_true) (noise >> signal):**
- R² can still be high (model fits mean trend)
- corr² ≈ 1/6 ≈ 0.17
- corr ≈ 0.41 ← **This matches our observation!**

### Why R² is High Despite Noise

R² measures:
```
R² = 1 - SS_res / SS_tot
   = 1 - Σ(y_true - y_pred)² / Σ(y_true - ȳ_true)²
```

**Key insight**: R² can be high if:
1. The model correctly predicts the MEAN trend (deterministic component)
2. Even if individual predictions are noisy

**Correlation measures**:
```
corr = Cov(y_true, y_pred) / (σ_true · σ_pred)
```

Correlation is REDUCED by noise in y_pred (increases σ_pred).

### Solution: Use Deterministic Translation for Evaluation

**Change `translate_*` functions to NOT use reparameterization:**

```python
def translate_eicu_to_mimic(self, x_eicu: torch.Tensor) -> torch.Tensor:
    """Translate eICU to MIMIC (DETERMINISTIC for evaluation)"""
    self.eval()
    with torch.no_grad():
        mu, logvar = self.encoder(x_eicu)
        mu = torch.clamp(mu, min=-20, max=20)
        
        # FIX: Use mu directly (deterministic), NOT reparameterize!
        z = mu  # ← NO NOISE!
        
        decoder_output = self.decoder_mimic(z)
        # ... rest unchanged
```

**Alternative: Add separate evaluation functions:**

```python
def translate_eicu_to_mimic_deterministic(self, x_eicu: torch.Tensor) -> torch.Tensor:
    """Deterministic translation for evaluation (no stochasticity)"""
    self.eval()
    with torch.no_grad():
        mu, _ = self.encoder(x_eicu)
        mu = torch.clamp(mu, min=-20, max=20)
        z = mu  # Use mu directly, no sampling
        
        decoder_output = self.decoder_mimic(z)
        if self.use_heteroscedastic:
            x_mimic, _ = decoder_output
        else:
            x_mimic = decoder_output
        return x_mimic

def translate_mimic_to_eicu_deterministic(self, x_mimic: torch.Tensor) -> torch.Tensor:
    """Deterministic translation for evaluation (no stochasticity)"""
    self.eval()
    with torch.no_grad():
        mu, _ = self.encoder(x_mimic)
        mu = torch.clamp(mu, min=-20, max=20)
        z = mu  # Use mu directly, no sampling
        
        decoder_output = self.decoder_eicu(z)
        if self.use_heteroscedastic:
            x_eicu, _ = decoder_output
        else:
            x_eicu = decoder_output
        return x_eicu
```

**Then update comprehensive_evaluator.py:**

```python
# In run_evaluation (comprehensive_evaluator.py:115-126)
# OLD:
x_eicu_to_mimic = self.model.translate_eicu_to_mimic(x_eicu)
x_mimic_to_eicu = self.model.translate_mimic_to_eicu(x_mimic)
x_eicu_roundtrip = self.model.translate_mimic_to_eicu(x_eicu_to_mimic)
x_mimic_roundtrip = self.model.translate_eicu_to_mimic(x_mimic_to_eicu)

# NEW:
x_eicu_to_mimic = self.model.translate_eicu_to_mimic_deterministic(x_eicu)
x_mimic_to_eicu = self.model.translate_mimic_to_eicu_deterministic(x_mimic)
x_eicu_roundtrip = self.model.translate_mimic_to_eicu_deterministic(x_eicu_to_mimic)
x_mimic_roundtrip = self.model.translate_eicu_to_mimic_deterministic(x_mimic_to_eicu)
```

### Expected Impact

After fix:
- **Correlation**: 0.39 → 0.95+ (should match √R²)
- **Roundtrip quality**: Much clearer signal
- **No change to training** (reparameterization still used in training for VAE regularization)

---

## Priority & Implementation Plan

### Priority 1: Fix Correlation Bug (CRITICAL) ⚠️
**Impact**: Current evaluation metrics are meaningless!

**Steps**:
1. Add `translate_*_deterministic` methods to `CycleVAE` (10 min)
2. Update `comprehensive_evaluator.py` to use deterministic translations (5 min)
3. Re-run evaluation (5 min)
4. Verify: corr ≈ √R² (1 min)

**Total time**: ~20 minutes

### Priority 2: Fix Gradient Explosion (HIGH) 🔴
**Impact**: Training instability, potential divergence

**Steps**:
1. Add gradient clipping to config (2 min)
2. Add output clamping to decoder (5 min)
3. Re-train model (varies)

**Total time**: ~10 minutes + retraining

---

## Verification Tests

### Test 1: Deterministic Translation
```python
# Should give IDENTICAL results for same input
x = load_test_data()
out1 = model.translate_eicu_to_mimic_deterministic(x)
out2 = model.translate_eicu_to_mimic_deterministic(x)
assert torch.allclose(out1, out2, atol=1e-6)  # Must be identical!
```

### Test 2: Correlation = √R²
```python
# After fix, correlation should match R²
results = evaluator.run_evaluation(...)
df = results['correlation_metrics']['summary_df']

for _, row in df.iterrows():
    r2 = row['mimic_r2']
    corr = row['mimic_correlation']
    expected_corr = np.sqrt(r2) if r2 >= 0 else 0
    assert abs(corr - expected_corr) < 0.05  # Tolerance for numerical errors
```

### Test 3: No Gradient Explosion
```python
# After fix, no infinite gradients
for epoch in range(50):
    train_epoch()
    assert all(torch.isfinite(p.grad) for p in model.parameters() if p.grad is not None)
```

---

## Conclusion

Both issues stem from implementation choices that seemed reasonable but had unintended consequences:

1. **Gradient Explosion**: Unbounded decoder outputs allowed bias drift → explosion
   - Fix: Gradient clipping + output bounds

2. **Low Correlation**: Stochastic translation added noise → reduced correlation
   - Fix: Use deterministic (mu only) for evaluation

**Neither is a fundamental model problem** - both are fixable with small code changes!


