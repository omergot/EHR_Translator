# Model Fixes Summary

## Date: October 5, 2025

## Issues Identified

### 1. Binary Feature Handling ✅ FIXED
**Problem**: Gender and missing flags were being reconstructed without sigmoid activation, leading to:
- Gender MSE = 0.565 (2.3x worse than random!)
- Model outputting ~0.5 constantly (ignoring input)
- Correlation = NaN due to zero variance in predictions

**Solution Implemented**:
- Modified `Decoder` class to apply `torch.sigmoid()` to:
  - Gender (at index `numeric_dim - 1`)
  - All missing flags (indices `numeric_dim` to `numeric_dim + missing_dim`)
- Updated decoder initialization to pass `numeric_dim` and `missing_dim` parameters

**Files Changed**: `src/model.py` (lines 93-176, 250-253)

### 2. Demographic Feature Bypass ✅ FIXED
**Problem**: Age and Gender were being compressed through the latent bottleneck (64 dims), losing information

**Solution Implemented**:
- Added direct bypass in `forward()` method
- Demographics are now copied directly from input to reconstruction
- Skips latent compression for Age and Gender
- Ensures perfect reconstruction of demographic features

**Files Changed**: `src/model.py` (lines 402-408)

**Code Added**:
```python
# IMPROVEMENT: Bypass latent bottleneck for demographics (Age, Gender)
if len(self.demographic_indices) > 0:
    for demo_idx in self.demographic_indices:
        x_recon[:, demo_idx] = x[:, demo_idx]
```

### 3. SpO2 Outlier Issue ⚠️ IDENTIFIED (NOT FIXED YET)
**Problem**: Extreme outliers in SpO2 features after preprocessing

**From Data Verification Report**:
```
SpO2_max: mean=44.717, std=7676.334 (!!!)
SpO2_mean: mean=1.748, std=307.328
```

This is causing:
- SpO2_max MSE = 867 (complete failure in evaluation)
- Distribution mismatch (KS statistic > 0.94)

**Root Cause**: 
RobustScaler (median/IQR) is robust to outliers in the data used to fit the scaler, but if test data has extreme values outside the training range, they get scaled to very large numbers.

**Possible Solutions** (TO BE IMPLEMENTED):
1. **Option A - Clip during preprocessing**:
   ```python
   # After scaling, clip to reasonable range
   scaled_data = np.clip(scaled_data, -10, 10)
   ```

2. **Option B - Better outlier handling in raw data**:
   - Investigate raw SpO2 values before scaling
   - Remove physiologically impossible values (SpO2 > 100 or < 0)
   - May need to check `extract_poc_features.py`

3. **Option C - Feature-specific loss weighting**:
   - Down-weight SpO2 in reconstruction loss
   - Prevent extreme outliers from dominating training

**Recommendation**: Start with Option A (clipping) as quick fix, then investigate Option B for proper solution.

---

## 4. Why `latent_dim=input_dim` (Not Implemented - EXPLANATION ONLY)

**Question**: What would setting `latent_dim = input_dim` do?

**Answer**: This converts the VAE into a **standard autoencoder with NO compression**.

### Current Setup (latent_dim=64):
```
Input (27 features) → Encoder → Latent Space (64 dims) → Decoder → Output (27 features)
                               ↑ BOTTLENECK
```

### With latent_dim=input_dim (27):
```
Input (27 features) → Encoder → Latent Space (27 dims) → Decoder → Output (27 features)
                               ↑ NO BOTTLENECK
```

### Effects:

**Pros**:
- ✓ **Perfect reconstruction possible** - no information loss
- ✓ **Easier to learn** - model has capacity to memorize
- ✓ **Better for debugging** - can isolate if compression is the issue
- ✓ **Good baseline** - compare against compressed versions

**Cons**:
- ✗ **No dimensionality reduction** - defeats purpose of latent representation
- ✗ **Overfitting risk** - model may just memorize training data
- ✗ **No regularization** - KL divergence becomes meaningless
- ✗ **Defeats domain adaptation** - need compression to force domains into shared space

### When to Use:
- **Debugging**: If this STILL fails, problem is NOT the bottleneck (it's data, loss function, or architecture)
- **Baseline**: Compare MSE with compressed versions to quantify information loss
- **Ablation study**: Understand contribution of compression to domain adaptation

### Why NOT for Production:
The entire point of a VAE for domain translation is to:
1. Compress both domains into a shared latent space
2. Force model to learn domain-invariant features
3. Use compression as implicit regularization

With `latent_dim=input_dim`, you lose all these benefits. The model becomes a simple autoencoder that can't do meaningful domain adaptation.

---

## Testing Recommendations

### 1. Quick Validation (5 min)
```bash
# Check if sigmoid is working
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python -c "
import torch
from src.model import CycleVAE
import yaml
with open('conf/config.yml') as f:
    config = yaml.safe_load(f)
# ... load model and check Gender output range ...
"
```

### 2. Full Retraining (30 min)
```bash
cd /bigdata/omerg/Thesis/EHR_Translator/poc_translator
python scripts/train.py --config conf/config.yml --mimic-only
```

**Expected Improvements**:
- Gender MSE: 0.565 → < 0.1
- Age MSE: 0.565 → < 0.1 (perfect reconstruction via bypass)
- Correlation: NaN → > 0.9 (due to variance in predictions)

### 3. Evaluation
```bash
python src/evaluate.py --checkpoint checkpoints/final_model.ckpt
```

**What to Check**:
- Gender R² should be >> 0.9 (was -1.3)
- Age R² should be = 1.0 (bypass ensures perfect reconstruction)
- Correlations should NOT be NaN
- Overall MSE should decrease significantly

---

## Files Modified

1. `/bigdata/omerg/Thesis/EHR_Translator/poc_translator/src/model.py`:
   - `Decoder.__init__()`: Added `numeric_dim`, `missing_dim` parameters
   - `Decoder.forward()`: Added sigmoid activation for binary features
   - `CycleVAE.__init__()`: Pass dimensions to decoders
   - `CycleVAE.forward()`: Added demographic bypass

---

## Next Steps

1. ✅ **DONE**: Fix binary feature handling (sigmoid)
2. ✅ **DONE**: Add demographic bypass
3. ⏳ **PENDING**: Fix SpO2 outliers (choose Option A, B, or C)
4. ⏳ **PENDING**: Retrain model and evaluate
5. ⏳ **PENDING**: If still poor, increase `latent_dim: 64 → 128`
6. ⏳ **PENDING**: If still poor, increase `rec_weight: 1.0 → 10.0`
7. ⏳ **PENDING**: Consider `latent_dim=input_dim` as diagnostic test

---

## Critical Notes

- **Sigmoid fix is crucial**: Binary features MUST be in [0,1] range
- **Bypass is optional but recommended**: Ensures demographics aren't lost
- **SpO2 outliers MUST be addressed**: Currently dominating loss function
- **RobustScaler limitations**: Works great for train data, but test outliers can explode

