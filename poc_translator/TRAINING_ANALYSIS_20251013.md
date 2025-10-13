# Training Analysis - October 13, 2025 Run

## Executive Summary

The training run from **12:44:37 to 12:59:02** (27 epochs completed) shows **clear signs of training stagnation** despite initial improvement. Two gradient explosions occurred, and the model is stuck on consistently problematic features.

---

## 1. Training Stagnation Analysis

### Overall Performance
- **Starting Point (Epoch 0):**
  - KS Distance (eICU→MIMIC): 0.185177
  - Wasserstein Distance (eICU→MIMIC): 0.136533

- **Ending Point (Epoch 27):**
  - KS Distance (eICU→MIMIC): 0.167490
  - Wasserstein Distance (eICU→MIMIC): 0.127426

- **Overall Improvement:**
  - KS: +0.0177 (9.55% improvement)
  - Wasserstein: +0.0091 (6.67% improvement)

### Stagnation Evidence

#### Last 5 Epochs (23-27) Statistics:
```
Epoch    KS          Wasserstein
23       0.169392    0.128532
24       0.169194    0.127755
25       0.170209    0.129405
26       0.169118    0.127228
27       0.167490    0.127426

Variance:
  KS:          0.000000783  ⚠️ EXTREMELY LOW
  Wasserstein: 0.000000644  ⚠️ EXTREMELY LOW
```

**The model is essentially flat-lined** - metrics are oscillating in a very narrow range with no meaningful improvement since ~epoch 16.

### Training Dynamics Pattern

**Phase 1 (Epochs 0-9): Active Learning**
- Large fluctuations (e.g., Epoch 2 spike: KS jumped from 0.177 → 0.209)
- Net improvement: KS improved by 0.0165
- Model exploring solution space

**Phase 2 (Epochs 10-16): Convergence**
- Gradual improvement with smaller changes
- Model finding local optimum
- KS improved from 0.168 → 0.165

**Phase 3 (Epochs 17-27): Stagnation**
- Tiny oscillations around KS ≈ 0.165-0.170
- **No meaningful progress**
- Model is stuck in local minimum

---

## 2. Problematic Features Analysis

### Consistently Worst Features (Appear in "Worst 5" across ALL 28 epochs):

1. **SpO2_max: KS = 0.997855**
   - ⚠️ **CRITICAL ISSUE** - This value is **constant across all epochs**
   - Model is **completely unable to match** this distribution
   - This is essentially a failed feature

2. **WBC_std: Wasserstein ≈ 0.44-0.47**
   - Appears in worst 5 in **100% of epochs**
   - Minimal improvement over training
   - White blood cell count standard deviation mismatch

3. **Creat_std: Wasserstein ≈ 0.34-0.42**
   - Appears in worst 5 in **100% of epochs**
   - Creatinine standard deviation mismatch
   - Moderate improvement but still problematic

4. **Na_std: Wasserstein ≈ 0.23-0.35**
   - Appears in worst 5 in **100% of epochs**
   - Sodium standard deviation mismatch
   - Some improvement but plateaued

5. **SpO2_min: Wasserstein ≈ 0.39-0.61**
   - Appears in worst 5 in **100% of epochs**
   - Large initial problem (0.61) that improved to ~0.40
   - But still consistently problematic

### Why These Features Are Stuck

**Pattern Recognition:**
- All worst features are either **SpO2** or **standard deviation (_std) features**
- The model is targeting these features (they're in the "worst-5" that Wasserstein loss focuses on)
- **But the loss is not helping** - they remain in the worst 5 epoch after epoch

**Possible Root Causes:**

1. **SpO2_max - Fundamental Data Issue:**
   - KS = 0.997855 (basically 1.0) means distributions are completely non-overlapping
   - This suggests either:
     - Measurement scale difference between databases (e.g., 0-100 vs 0-1)
     - Data quality issue (outliers, censoring)
     - Preprocessing error specific to SpO2_max

2. **Standard Deviation Features (_std) - Statistical Challenge:**
   - Std deviation is a second-order statistic
   - Model reconstructs means well but struggles with variance matching
   - The decoder architecture may not have capacity to independently control mean and variance
   - Skip connections preserve input variance, limiting the model's ability to adjust

---

## 3. Gradient Explosion Analysis

### Explosion Events:
```
Epoch 19, Step 6000:
  - decoder_mimic.fc_out.bias: inf
  - decoder_eicu.fc_out.bias: inf
  - Skip connection grads: ~0.0016 (normal)
  - Last batch: z range [-13.65, 11.10]

Epoch 25, Step 8001:
  - decoder_mimic.fc_out.bias: inf
  - decoder_eicu.fc_out.bias: inf
  - Skip connection grads: ~0.0005 (normal)
  - Last batch: z range [-13.99, 13.42]
```

### Key Observations:

1. **Specific Layer Problem:**
   - ONLY `fc_out.bias` (final decoder layer bias) explodes
   - All other parameters have normal gradients
   - Skip connection parameters have **very small** gradients (0.0005-0.0016)

2. **Skip Connections Are NOT the Problem:**
   - Despite detailed logging added, skip connection grads are tiny
   - Skip parameters are **stable**:
     - MIMIC decoder: scale ≈ 0.91-0.94, bias ≈ -0.003 to 0.005
     - eICU decoder: scale ≈ 0.94-0.95, bias ≈ -0.004 to 0.003
   - Skip params show healthy variance (std ≈ 0.08-0.19) but no divergence

3. **Latent Space Issue:**
   - z values are **very large**: range ~[-14, +13]
   - Expected range for normal latent: ≈ [-3, +3]
   - Encoder is producing extreme latent codes
   - Final bias layer receives these extreme values multiplied by weight matrix

### Why Explosions Happen:

**Sequence of Events:**
1. Encoder produces extreme latent codes (|z| > 10)
2. Decoder layers amplify these through non-linearities
3. Final fc_out layer sees extreme inputs
4. Bias gradient = sum of errors across batch
5. If reconstruction error is large on problematic features (SpO2_max, WBC_std), and inputs are extreme, bias gradient explodes
6. **inf gradients trigger** when: large_error × large_activation = overflow

**Root Cause:**
- **Latent space is not properly regularized**
- KL divergence loss (encoder regularization) is either:
  - Too weak (low weight)
  - Disabled/not working properly
  - Being overcome by reconstruction pressure on hard features

---

## 4. Skip Connection Parameter Evolution

### MIMIC Decoder Skip Parameters:

```
Epoch    skip_scale (min, max, mean)          skip_bias (min, max, mean)
0        (0.766, 1.075, 0.976)                (-0.258, 0.261, -0.004)
5        (0.703, 1.142, 0.973)                (-0.282, 0.276, -0.025)
10       (0.672, 1.109, 0.962)                (-0.265, 0.204, -0.008)
15       (0.549, 1.128, 0.945)                (-0.309, 0.176, -0.011)
20       (0.419, 1.128, 0.933)                (-0.320, 0.178, +0.001)
25       (0.351, 1.179, 0.919)                (-0.325, 0.178, +0.004)
27       (0.333, 1.173, 0.912)                (-0.326, 0.181, +0.003)
```

### eICU Decoder Skip Parameters:

```
Epoch    skip_scale (min, max, mean)          skip_bias (min, max, mean)
0        (0.771, 1.031, 0.979)                (-0.259, 0.263, -0.006)
5        (0.746, 1.054, 0.976)                (-0.274, 0.363, -0.032)
10       (0.691, 1.095, 0.970)                (-0.276, 0.300, -0.016)
15       (0.602, 1.078, 0.965)                (-0.296, 0.259, -0.005)
20       (0.426, 1.140, 0.948)                (-0.308, 0.174, -0.001)
25       (0.375, 1.182, 0.945)                (-0.303, 0.179, +0.003)
27       (0.357, 1.183, 0.945)                (-0.301, 0.166, +0.001)
```

### Analysis:

1. **Range Expansion:**
   - Min values decreasing: 0.77 → 0.33 (MIMIC), 0.77 → 0.36 (eICU)
   - Max values increasing: 1.08 → 1.18 (MIMIC), 1.03 → 1.18 (eICU)
   - **Standard deviation growing:** 0.06 → 0.19 (MIMIC), 0.06 → 0.16 (eICU)
   
2. **Mean Drift:**
   - MIMIC mean: 0.976 → 0.912 (drift down by 6.5%)
   - eICU mean: 0.979 → 0.945 (drift down by 3.5%)
   - Model is learning to **reduce skip connection strength** on average

3. **Interpretation:**
   - Skip connections **moving away from identity** (1.0)
   - Model trying to rely more on decoder network, less on input passthrough
   - This is **expected behavior** for difficult features where input≠output
   - BUT: The expanding range suggests model is **struggling** - some features need strong skip (near 1.0), others need weak (near 0.3)

4. **Not Causing Explosions:**
   - Parameters are stable (no runaway growth)
   - Gradients are tiny (0.0005-0.0016)
   - Changes are gradual and controlled

---

## 5. Root Cause Summary

### Why Training is Stuck:

1. **SpO2_max Distribution Mismatch (KS = 0.998):**
   - This feature is **fundamentally unlearnable** in current form
   - Likely a **data preprocessing issue** not a model issue
   - Model wastes capacity trying to fix this unfixable feature

2. **Standard Deviation Feature Challenge:**
   - WBC_std, Creat_std, Na_std remain in worst 5 across ALL epochs
   - Model architecture (VAE with skip connections) may not have capacity to independently control mean vs variance
   - Current loss function doesn't provide enough signal for variance matching

3. **Latent Space Explosion:**
   - Encoder producing extreme codes (|z| > 10-14)
   - KL divergence not constraining latent space enough
   - This causes gradient instabilities in final decoder layer

4. **Local Minimum:**
   - Model found a solution that's "good enough" for easy features
   - Hard features (SpO2, _std) not improving
   - Wasserstein loss targeting worst features but not helping
   - **The loss weights might need rebalancing**

### Why Gradient Explosions Happen:

**Direct Cause:** Final decoder bias layer (`fc_out.bias`)
- Receives extreme inputs from blown-up latent codes
- Reconstruction error on hard features creates large gradients
- Multiplication of large error × large activation = overflow to inf

**Underlying Cause:** Unconstrained latent space
- Encoder not properly regularized by KL divergence
- Latent codes growing to compensate for difficult reconstructions
- This is a **symptom** of the hard features problem

---

## 6. Recommendations

### Immediate Actions:

1. **Investigate SpO2_max Data:**
   ```python
   # Check raw data distributions
   import pandas as pd
   mimic = pd.read_csv('data/train_mimic_preprocessed.csv')
   eicu = pd.read_csv('data/train_eicu_preprocessed.csv')
   
   print("MIMIC SpO2_max:", mimic['SpO2_max'].describe())
   print("eICU SpO2_max:", eicu['SpO2_max'].describe())
   ```
   - Look for scale mismatch (0-100 vs 0-1)
   - Check for outliers or censoring
   - May need to **exclude this feature** if unfixable

2. **Strengthen Latent Space Regularization:**
   - Current: KL warmup over 20 epochs (but what's the final weight?)
   - Add explicit latent clamping: `z = torch.clamp(z, min=-5, max=5)`
   - Increase KL divergence loss weight

3. **Consider Architecture Changes for Variance:**
   - Current decoder doesn't separately model mean and variance
   - Could add auxiliary loss specifically for matching feature standard deviations
   - Or use heteroscedastic outputs (already have this for some modes?)

### Longer-term Fixes:

4. **Feature Engineering:**
   - Consider removing SpO2_max entirely if it's unlearnable
   - Or transform it differently (log scale, quantile normalization)

5. **Loss Rebalancing:**
   - Current: `rec_weight=0.2, cycle_weight=0.2, wasserstein_weight=1.0`
   - Wasserstein is 5x higher but not helping worst features
   - Consider: Add a "variance matching loss" with moderate weight

6. **Adaptive Worst-K Selection:**
   - Currently targets worst-5 features
   - But if they're consistently the same 5 and not improving, this may be counterproductive
   - Consider: Exclude truly unlearnable features from worst-K selection

---

## 7. Training Metrics Summary

| Metric | Initial | Final | Change | Status |
|--------|---------|-------|--------|--------|
| KS (eICU→MIMIC) | 0.1852 | 0.1675 | -9.5% | ⚠️ Stagnant |
| Wasserstein (eICU→MIMIC) | 0.1365 | 0.1274 | -6.7% | ⚠️ Stagnant |
| Skip scale (MIMIC mean) | 0.976 | 0.912 | -6.5% | ✓ Stable |
| Skip scale (eICU mean) | 0.979 | 0.945 | -3.5% | ✓ Stable |
| Gradient Explosions | - | 2 | - | ⚠️ Concerning |
| SpO2_max KS | 0.998 | 0.998 | 0.0% | ❌ Failed |

**Conclusion:** Model has converged to a local minimum. Further training without intervention will not improve performance on hard features.

