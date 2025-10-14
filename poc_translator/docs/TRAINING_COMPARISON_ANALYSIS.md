# Training Comparison Analysis: Before vs After Fixes
**Date:** October 13, 2025  
**Comparison:** Training at 12:44 (before fixes) vs 14:03 (after fixes)

---

## Executive Summary

After implementing the 5 critical fixes (KL loss, latent clamping, loss rebalancing, SpO2_max blacklist), the training shows **mixed results**:

### ✅ **Major Wins:**
1. **Latent space properly constrained** (|z| ≤ 5.0 vs previous ~14)
2. **Skip connections much more stable** (std ~0.06 vs previous ~0.19)
3. **KL loss successfully integrated** (values ~4-5)
4. **Feature blacklist working** (SpO2_max excluded from worst-K selection)

### ⚠️ **Concerns:**
1. **Distribution metrics slightly WORSE** (KS +5.2%, Wass +2.1%)
2. **Still 1 gradient explosion** (but with clamped latents - different failure mode)
3. **Training cut shorter** (20 epochs vs 28)
4. **Same problematic features persist** (WBC_std, Creat_std, Na_std)

### 🎯 **Bottom Line:**
The fixes **solved the instability** but **didn't improve distribution matching**. The model is now **stable but still stuck** on hard features.

---

## Detailed Metrics Comparison

### Training Distribution Metrics

|Metric|Before (Epoch 0)|Before (Final Ep 27)|After (Final Ep 19)|Change vs Before|
|------|----------------|--------------------|--------------------|----------------|
|**KS (eICU→MIMIC)**|0.185177|0.167490|**0.176222**|+0.008732 (+5.2%) ⬆️ WORSE|
|**Wass (eICU→MIMIC)**|0.136533|0.127426|**0.130056**|+0.002630 (+2.1%) ⬆️ WORSE|
|**KS (MIMIC→eICU)**|0.204082|0.179968|**0.178641**|-0.001327 (-0.7%) ⬇️ Better|
|**Wass (MIMIC→eICU)**|0.148148|0.130988|**0.132205**|+0.001217 (+0.9%) ⬆️ WORSE|

**Note:** "After" trained for fewer epochs (19 vs 27), so not fully converged yet.

### Overall Improvement from Start

|Metric|Before (Ep 0→27)|After (Ep 0→19)|
|------|----------------|---------------|
|**KS Improvement**|0.0177 (9.5%)|0.009 (4.9%)|
|**Wass Improvement**|0.0091 (6.7%)|0.006 (4.5%)|

**Interpretation:** Less improvement after fixes, **BUT** before-training had 28 epochs vs 19 epochs after. The comparison isn't entirely fair.

---

## Gradient Explosion Analysis

### Before Fixes:
- **2 explosions** at Epochs 19 and 25
- **Affected:** `decoder_mimic.fc_out.bias` and `decoder_eicu.fc_out.bias`
- **Cause:** Extreme latent codes (|z| up to 13.99)
- **Pattern:** Final decoder layer couldn't handle unbounded latent space

### After Fixes:
- **1 explosion** at Epoch 14
- **Affected:** `encoder.fc_mu.weight` ← **DIFFERENT LAYER!**
- **Latent range at explosion:** z ∈ [-5.0000, 5.0000] ← **Clamping worked!**
- **Cause:** **New failure mode** - encoder's mu prediction itself exploded

**Key Insight:** The clamping **prevented decoder explosions** but revealed a **different instability** in the encoder. The KL loss weight (0.0007 at epoch 14) may still be too weak to fully regularize encoder output.

---

## Latent Space Stability

| Metric | Before | After | Assessment |
|--------|--------|-------|------------|
| **Max \|z\| observed** | 13.99 | 5.00 | ✅ **Clamping works perfectly** |
| **Encoder regularization** | None (KL loss missing) | KL loss ~4-5 | ✅ **Added** |
| **KL weight** | N/A | 0.0007-0.0009 (ramping up) | ⚠️ **May be too small** |

**Recommendation:** Consider increasing `kl_weight_target` from 0.001 to 0.005-0.01 for stronger regularization.

---

## Skip Connection Parameter Evolution

### MIMIC Decoder:

| Epoch | Before: skip_scale (mean, std) | After: skip_scale (mean, std) |
|-------|-------------------------------|-------------------------------|
| 0 | (0.976, 0.063) | (0.976, 0.063) |
| ~14 | (0.945, 0.119) | (0.970, 0.068) ✅ **More stable** |
| Final | (0.912, 0.194) | (0.977, 0.059) ✅ **Much more stable** |

### eICU Decoder:

| Epoch | Before: skip_scale (mean, std) | After: skip_scale (mean, std) |
|-------|-------------------------------|-------------------------------|
| 0 | (0.979, 0.061) | (0.974, 0.058) |
| ~14 | (0.965, 0.087) | (0.974, 0.058) ✅ **More stable** |
| Final | (0.945, 0.159) | (0.980, 0.059) ✅ **Much more stable** |

**Assessment:** 
- ✅ Skip parameters are **dramatically more stable** after fixes
- ✅ Mean stays very close to 1.0 (identity-like behavior)
- ✅ Standard deviation **3x lower** (0.06 vs 0.19)
- **Interpretation:** Loss rebalancing (higher rec_weight) reduced pressure on skip connections to compensate for poor reconstruction

---

## Feature Blacklist Effectiveness

### Before Fixes (Worst-5 Selection):
```
Example from Epoch 17:
Updated worst-5 features: indices=[13, 8, 14, 21, 15]
```
(No feature names logged, but SpO2_max always appeared)

### After Fixes (Worst-5 Selection):
```
Epoch 14: indices=[21, 22, 20, 23, 8], names=['Creat_max', 'Creat_mean', 'Creat_min', 'Creat_std', 'SpO2_min']
Epoch 17: indices=[21, 8, 14, 22, 20], names=['Creat_max', 'SpO2_min', 'WBC_mean', 'Creat_mean', 'Creat_min']
Epoch 19: indices=[21, 16, 20, 22, 13], names=['Creat_max', 'Na_min', 'Creat_min', 'Creat_mean', 'WBC_max']
```

**Assessment:**
- ✅ **Blacklist is working** - SpO2_max no longer in worst-5 selection
- ⚠️ **But** SpO2_max still appears in overall "Worst 5 by KS" metrics (KS=0.998)
- ✅ Model now focuses on **learnable features** (Creat group, WBC, Na)
- ⚠️ **New pattern:** Creat features (min/max/mean/std) dominate worst-5 consistently

---

## Evaluation Results (Test Set Performance)

### Round-Trip Consistency (A→B'→A'):

| Metric | Value | Assessment |
|--------|-------|------------|
| eICU MAE | 0.237 (std units) | ✅ Good |
| MIMIC MAE | 0.240 (std units) | ✅ Good |
| eICU % within 0.5 IQR | 99.8% | ✅ Excellent |
| MIMIC % within 0.5 IQR | 99.5% | ✅ Excellent |

### Distribution Matching (Test):

| Metric | eICU→MIMIC | MIMIC→eICU |
|--------|------------|------------|
| **Mean KS** | 0.166 | 0.176 |
| **KS < 0.1 (Excellent)** | 15/24 (62.5%) | 11/24 (45.8%) |
| **KS < 0.2 (Good)** | 19/24 (79.2%) | 20/24 (83.3%) |
| **Mean Wasserstein** | 0.139 | N/A |

### Downstream Task (Mortality Prediction):

| Metric | eICU Raw | eICU Translated | Improvement |
|--------|----------|-----------------|-------------|
| **AUC** | 0.496 | 0.503 | +0.007 |
| **AP** | 0.194 | 0.199 | +0.005 |

**Assessment:**
- ✅ Round-trip consistency is **excellent** (99%+ within IQR)
- ⚠️ Distribution matching is **moderate** (62.5% excellent features)
- ⚠️ Downstream improvement is **marginal** (~0.7% AUC)
- ❌ Downstream performance overall is **poor** (AUC ~0.5 = random)

**Key Finding:** Model preserves structure well (cycle consistency) but **doesn't improve downstream task**. This suggests **feature distributions aren't clinically meaningful** after translation.

---

## Persistent Problematic Features

Features that **consistently appear in worst 5** across both before and after:

1. **SpO2_max** - KS = 0.998 (constant per dataset, unfixable)
2. **WBC_std** - Wass ~0.47 (white blood cell variance)
3. **Creat_std** - Wass ~0.38 (creatinine variance)
4. **Na_std** - Wass ~0.34 (sodium variance)
5. **SpO2_min** - Wass ~0.40

**New Pattern After Fixes:**
- **Creatinine features (min/max/mean/std)** now dominate worst-5 selection
- This suggests Creat has **distributional differences** that are hard to match
- Unlike SpO2_max (which is constant), Creat features have real variation but **different distributions** between datasets

---

## Analysis: Why Distribution Matching Got Slightly Worse

### Hypothesis 1: Loss Rebalancing Effect
**Before:**
```yaml
rec_weight: 0.2
cycle_weight: 0.2
wasserstein_weight: 1.0  # Dominant
```

**After:**
```yaml
rec_weight: 1.0          # 5x increase
cycle_weight: 0.5        # 2.5x increase
wasserstein_weight: 0.5  # 0.5x decrease
```

**Analysis:**
- Model now **prioritizes reconstruction** over distribution matching
- This is **intentional** - we want stable training first
- **Trade-off:** Better stability, slightly worse distributions

### Hypothesis 2: Shorter Training
- Before: 28 epochs
- After: 20 epochs (reduced in config)
- Less time to converge → metrics not fully optimized yet

### Hypothesis 3: KL Loss Trade-off
- KL loss constrains latent space
- Constrained latent → **less flexibility** for distribution matching
- **Trade-off:** More stable, less expressive

### Hypothesis 4: Creatinine Focus
- Blacklist redirected Wasserstein loss from SpO2_max → Creat features
- Creat group appears to have **fundamental distribution differences**
- Model is now **stuck on a different hard problem**

---

## Key Insights

### ✅ What Worked:
1. **Latent clamping** - No more decoder explosions
2. **Loss rebalancing** - Much more stable skip connections
3. **Feature blacklist** - Model focuses on learnable features
4. **KL loss** - Latent space regularized (though may need stronger weight)

### ⚠️ What Didn't Work as Expected:
1. **Distribution matching slightly worse** - Metrics regressed ~2-5%
2. **New explosion mode** - Encoder mu layer exploded (not decoder)
3. **Different stuck features** - Creat group now problematic instead of SpO2

### ❌ Unchanged Problems:
1. **Downstream task still fails** - AUC ~0.5 (random performance)
2. **Standard deviation features still hard** - _std features consistently worst
3. **Training still stagnates** - Just at a different local minimum

---

## Recommended Next Steps

### 🔴 **HIGH PRIORITY - Address New Instability:**

**1. Strengthen KL Regularization**
```yaml
# In config.yml:
kl_weight_target: 0.005  # Increase from 0.001 (5x stronger)
```
**Rationale:** KL weight of 0.0007-0.0009 was too weak to prevent encoder explosion.

**2. Add Encoder Output Clamping (Safety)**
```python
# In model.py, Encoder.forward():
def forward(self, x):
    features = self.feature_extractor(x)
    mu = self.fc_mu(features)
    logvar = self.fc_logvar(features)
    
    # Clamp mu to prevent extreme encoder predictions
    mu = torch.clamp(mu, min=-10, max=10)  # NEW
    
    return mu, logvar
```

**3. Increase Training to Full 30 Epochs**
- Current: 20 epochs (too short for convergence)
- Restore: 30 epochs
- **Rationale:** Need fair comparison with before-training (28 epochs)

### 🟡 **MEDIUM PRIORITY - Improve Distribution Matching:**

**4. Investigate Creatinine Distribution Mismatch**
```python
# Similar to SpO2_max analysis:
print("MIMIC Creat features:", mimic[['Creat_min', 'Creat_max', 'Creat_mean', 'Creat_std']].describe())
print("eICU Creat features:", eicu[['Creat_min', 'Creat_max', 'Creat_mean', 'Creat_std']].describe())
```
**Rationale:** Creat group now dominates worst-5. May need preprocessing fix or blacklist expansion.

**5. Consider Excluding SpO2_max Entirely from Model Input**
- Currently: Blacklisted from worst-K but still processed
- Alternative: Remove from preprocessing entirely
- **Benefit:** Model doesn't waste capacity on unlearnable feature

**6. Add Separate Loss for Variance Matching**
```python
# For _std features specifically:
std_features = ['WBC_std', 'Creat_std', 'Na_std', 'RR_std', 'HR_std', 'SpO2_std', 'Temp_std', 'MAP_std']
std_indices = [i for i, name in enumerate(feature_names) if any(f in name for f in std_features)]

variance_loss = F.mse_loss(
    x_recon[:, std_indices].std(dim=0),
    x[:, std_indices].std(dim=0)
)
total_loss += 0.1 * variance_loss  # Small weight
```

### 🟢 **LOW PRIORITY - Optimization:**

**7. Tune Loss Weight Balance**
Current weights may need fine-tuning:
```yaml
rec_weight: 1.0          # Keep
cycle_weight: 0.3        # Decrease slightly
wasserstein_weight: 0.7  # Increase slightly
kl_weight_target: 0.005  # Increase
```

**8. Increase KL Warmup Period**
```yaml
kl_warmup_epochs: 30  # From 20 - gentler ramp
```

---

## Comparison to Previous Training

### Convergence Comparison:

| Aspect | Before Fixes | After Fixes |
|--------|-------------|-------------|
| **Stagnation starts** | Epoch ~17 | Epoch ~15 (earlier!) |
| **Final KS** | 0.167 | 0.176 (+5.2% worse) |
| **Final Wass** | 0.127 | 0.130 (+2.1% worse) |
| **Gradient explosions** | 2 (decoder bias) | 1 (encoder mu) |
| **Latent range** | [-14, +14] | [-5, +5] ✅ |
| **Skip stability** | std=0.19 | std=0.06 ✅ |

### Training Dynamics:

| Phase | Before | After |
|-------|--------|-------|
| **Epochs 0-9** | Active learning, large changes | Active learning, large changes |
| **Epochs 10-16** | Gradual convergence | **Earlier stagnation** ⚠️ |
| **Epochs 17+** | Stagnation | (Not trained) |

**Key Difference:** After fixes, model **stagnates earlier** (epoch 15 vs 17). This suggests:
- KL loss constraining exploration
- Loss rebalancing prioritizing stability over distribution matching
- **Trade-off accepted** for training stability

---

## Conclusion

### The Good News:
1. ✅ **Training is much more stable** - skip connections don't diverge
2. ✅ **Latent space is properly regularized** - no more |z|>10 explosions
3. ✅ **Model focuses on learnable features** - blacklist works

### The Bad News:
1. ❌ **Distribution matching slightly degraded** - not improved
2. ⚠️ **New instability emerged** - encoder mu explosion (need stronger KL)
3. ⚠️ **Still stuck on hard features** - just different ones (Creat vs SpO2)
4. ❌ **Downstream task still fails** - AUC ~0.5 (random)

### The Verdict:
**The fixes achieved their primary goal (stability) but revealed that the fundamental problem is deeper:**
- Some features have **irreconcilable distributional differences** (SpO2_max, Creat group, _std features)
- Standard deviation features may require **architecture changes** (separate variance modeling)
- The model may be **fundamentally limited** by VAE architecture for this task

### Recommended Action:
1. **Implement high-priority fixes** (stronger KL, encoder clamping, full 30 epochs)
2. **Run another training** to see if convergence improves with more epochs
3. **If still stuck:** Consider architectural changes (separate variance heads, heteroscedastic outputs for _std features)

The fixes were **necessary but not sufficient**. We have a **stable but still limited** model.


