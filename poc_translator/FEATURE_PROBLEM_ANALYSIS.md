# Feature Problem Analysis & Proposed Solutions

## 🔍 Investigation Summary

We investigated why certain features perform poorly in cycle reconstruction:
- **Std features**: WBC_std, Na_std, Creat_std (KS: 0.35-0.43, R²: 0.31-0.65)
- **Min features**: RR_min (R²: -0.03), HR_min (R²: 0.28)
- **Mean features**: RR_mean (R²: 0.33), HR_mean (R²: 0.56)

---

## 📊 Key Findings

### 1. **Std Features: Log1p + RobustScaler Works, But Distribution Still Challenging**

**Current preprocessing (CORRECT)**:
```
Raw std → log1p(std) → RobustScaler → Normalized
```

**After preprocessing**:
- WBC_std: 47.8% negative, Median=0, Skew=0.67
- Na_std: 42.1% negative, Median=0, Skew=0.41
- Creat_std: 47.6% negative, Median=0, Skew=1.68

**Why ~50% negative?**
- Log1p(std) transforms positive skew → still right-skewed
- RobustScaler centers at median → negative values are below median
- This is **mathematically correct!**

**Why model struggles**:
1. **Creat_std has heavy right tail** (skew=1.68 AFTER log1p!)
   - Extreme values not handled well by MSE
   - Model compresses tail → systematic bias

2. **Std features have weak signal**
   - Represent within-window variability
   - Lower information content than min/mean/max
   - Harder to preserve through latent bottleneck

3. **Zero-inflation around median**
   - 3-11% exactly zero
   - Dense cluster near median → model learns to predict median
   - Loses tail information

---

### 2. **RR_min & HR_min: Heavy Discretization**

**RR_min** (R²=-0.03, CATASTROPHIC):
- **Only 27 unique values** after preprocessing!
- Top 10 values account for 84.4% of data
- Median spacing: 0.091 (large gaps)
- **Histogram shows spike pattern** - discrete distribution

**Why this happens**:
```
Original: RR measured in breaths/minute (integers: 10, 12, 15, etc.)
→ POC aggregation: min/max/mean/std per window
→ Normalization: Discrete values → discrete normalized values
→ **Still fundamentally discrete!**
```

**HR_min** (R²=0.28, POOR):
- 112 unique values (better than RR_min, but still discrete)
- No single dominant value
- More spread but still quantized

**Why model fails**:
- VAE encoder/decoder are **continuous** (regression)
- Trying to fit discrete distribution with continuous output
- MSE loss: model outputs mean/median → misses discrete structure
- Cycle translation: discretization lost → continuous output → bad reconstruction

---

### 3. **RR_mean & HR_mean: Moderate Discretization**

**RR_mean** (R²=0.33, POOR):
- 5,069 unique values (much better than RR_min)
- But still shows peaked distribution
- Skewness: 0.90 (right-skewed)

**HR_mean** (R²=0.56, ACCEPTABLE):
- 7,818 unique values (nearly continuous)
- Moderate skew: 0.40
- Performs better (R²=0.56) but not great

**Pattern**: Min features are more discrete → worse performance

---

## 🎯 Root Causes Summary

| Feature | Root Cause | Evidence |
|---------|------------|----------|
| **Creat_std** | Heavy right tail (skew=1.68 after log1p) | KS=0.43, systematic bias |
| **WBC_std, Na_std** | Weak signal + zero-inflation | KS=0.35-0.39, median=0 |
| **RR_min** | Severe discretization (27 unique values) | R²=-0.03, spike pattern |
| **HR_min** | Moderate discretization (112 unique) | R²=0.28, quantization |
| **RR_mean** | Mild discretization + right skew | R²=0.33, skew=0.90 |

---

## 💡 Proposed Solutions

### **Solution 1: Improve Std Feature Handling** (Priority: MEDIUM)

**Problem**: Creat_std still skewed (1.68) after log1p, heavy tail

**Option A: Box-Cox or Yeo-Johnson transformation** (RECOMMENDED)
```python
# In apply_feature_specific_transforms:
from scipy.stats import yeojohnson

def apply_feature_specific_transforms(self, data, feature_analysis):
    # For std features with extreme skew
    EXTREME_SKEW_FEATURES = {
        'Creat_std': True,  # Skew > 1.5 after log1p
        'WBC_std': False,   # Skew < 1.0, log1p sufficient
        'Na_std': False
    }
    
    for col in feature_analysis["std_features"]:
        if EXTREME_SKEW_FEATURES.get(col, False):
            # Yeo-Johnson handles negative values (after scaling)
            data[col], fitted_lambda = yeojohnson(data[col])
            logger.info(f"Applied Yeo-Johnson to {col} (lambda={fitted_lambda:.3f})")
        else:
            # Standard log1p for other std features
            data[col] = np.log1p(data[col])
    
    return data
```

**Option B: Quantile transformation** (More aggressive)
```python
from sklearn.preprocessing import QuantileTransformer

# For all std features
qt = QuantileTransformer(output_distribution='normal')
std_cols = [f for f in data.columns if f.endswith('_std')]
data[std_cols] = qt.fit_transform(data[std_cols])
```

**Expected improvement**: Creat_std KS: 0.43 → 0.30

---

### **Solution 2: Handle Discretization in RR_min, HR_min** (Priority: HIGH)

**Problem**: Only 27-112 unique values, model outputs continuous

**Option A: Treat as ordinal/categorical** (RECOMMENDED)
```python
# In model.py, separate discrete features from continuous

class CycleVAE:
    def __init__(self, config, feature_spec):
        # Identify discrete features
        self.discrete_features = ['RR_min', 'HR_min']  # Low unique count
        self.discrete_indices = [self.get_feature_idx(f) 
                                for f in self.discrete_features]
        
        # Create quantization bins (learned from training data)
        self.quantization_bins = {}
        for feat in self.discrete_features:
            # Get unique values from training data
            unique_vals = np.unique(train_data[feat])
            self.quantization_bins[feat] = unique_vals
    
    def decode_with_quantization(self, x_decoded):
        # Post-process discrete features
        for feat_idx, feat_name in zip(self.discrete_indices, 
                                       self.discrete_features):
            # Find nearest discrete value
            bins = self.quantization_bins[feat_name]
            x_decoded[:, feat_idx] = self.quantize_to_nearest(
                x_decoded[:, feat_idx], bins
            )
        return x_decoded
    
    def quantize_to_nearest(self, values, bins):
        # Snap to nearest bin value
        bins_tensor = torch.tensor(bins, device=values.device)
        dists = torch.abs(values.unsqueeze(-1) - bins_tensor)
        nearest_idx = torch.argmin(dists, dim=-1)
        return bins_tensor[nearest_idx]
```

**Option B: Weighted loss for discrete features**
```python
# Increase reconstruction loss weight for discrete features
FEATURE_LOSS_WEIGHTS = {
    'RR_min': 3.0,   # Heavily penalize errors (only 27 values)
    'HR_min': 2.0,   # Moderately penalize
    # Others: 1.0 (default)
}

def weighted_reconstruction_loss(pred, target, weights_dict):
    per_feature_loss = (pred - target) ** 2
    weights = torch.tensor([weights_dict.get(f, 1.0) 
                           for f in feature_names])
    return (per_feature_loss * weights).mean()
```

**Option C: Add discretization-aware loss**
```python
# For discrete features, penalize not just value but "bin membership"
def discrete_aware_loss(pred, target, bins):
    # Continuous loss
    mse = (pred - target) ** 2
    
    # Bin membership loss (are they in same bin?)
    pred_bin = torch.argmin(torch.abs(pred.unsqueeze(-1) - bins), dim=-1)
    target_bin = torch.argmin(torch.abs(target.unsqueeze(-1) - bins), dim=-1)
    bin_mismatch = (pred_bin != target_bin).float()
    
    return mse + 0.5 * bin_mismatch
```

**Expected improvement**: RR_min R²: -0.03 → 0.30, HR_min R²: 0.28 → 0.50

---

### **Solution 3: Exclude SpO2_max** (Priority: CRITICAL)

**Problem**: Degenerate (IQR=0), R²=-2.23

**Implementation**:
```python
# In model.py
class CycleVAE:
    def __init__(self, config, feature_spec):
        # Exclude degenerate features from cycle loss
        self.excluded_from_cycle = ['SpO2_max']
        self.excluded_indices = [self.get_feature_idx(f) 
                                for f in self.excluded_from_cycle]
        
        # Create mask for cycle loss
        self.cycle_loss_mask = torch.ones(self.input_dim)
        self.cycle_loss_mask[self.excluded_indices] = 0.0
    
    def compute_cycle_loss(self, x_orig, x_cycle):
        # Mask out excluded features
        masked_orig = x_orig * self.cycle_loss_mask
        masked_cycle = x_cycle * self.cycle_loss_mask
        
        # Normalize by number of included features
        n_included = self.cycle_loss_mask.sum()
        loss = F.mse_loss(masked_cycle, masked_orig, reduction='sum')
        return loss / n_included
```

**Expected improvement**: SpO2_max won't drag down overall metrics

---

### **Solution 4: Add Min≤Mean≤Max Enforcement** (Priority: MEDIUM)

**Problem**: After cycle, semantic constraints can be violated

**Implementation**:
```python
# In model.py decoder
def decode_with_constraints(self, z, domain):
    x_decoded = self.decoder(z, domain)
    
    # Enforce min ≤ mean ≤ max for each clinical feature
    for feature in ['HR', 'RR', 'SpO2', 'Temp', 'WBC', 'Na', 'Creat']:
        min_idx = self.feature_indices[f'{feature}_min']
        mean_idx = self.feature_indices[f'{feature}_mean']
        max_idx = self.feature_indices[f'{feature}_max']
        
        # Soft enforcement via clamping
        x_decoded[:, min_idx] = torch.minimum(
            x_decoded[:, min_idx], 
            x_decoded[:, mean_idx]
        )
        x_decoded[:, max_idx] = torch.maximum(
            x_decoded[:, max_idx],
            x_decoded[:, mean_idx]
        )
    
    return x_decoded
```

**Expected improvement**: No violations, more semantically valid outputs

---

## 🚀 Implementation Priority & Expected Impact

| Priority | Solution | Effort | Expected Impact |
|----------|----------|--------|-----------------|
| **CRITICAL** | Exclude SpO2_max from cycle loss | 30 min | R²: -2.23 → excluded |
| **HIGH** | Quantize RR_min, HR_min outputs | 2 hours | RR_min R²: -0.03 → 0.30 |
| **MEDIUM** | Better transform for Creat_std | 1 hour | KS: 0.43 → 0.30 |
| **MEDIUM** | Enforce min≤mean≤max constraints | 1 hour | Semantic validity ✓ |
| **LOW** | Quantile transform all std features | 2 hours | Minor KS improvement |

**Total effort**: 4-6 hours for high-impact fixes

**Expected overall improvement**:
- Features with R² > 0.5: **67% → 80%+**
- Features with KS < 0.2: **71% → 85%+**
- No negative R² features
- Semantically valid outputs

---

## 📋 Next Steps

1. **Immediate**: Implement SpO2_max exclusion (30 min)
2. **Phase 1**: Add discretization handling for RR_min, HR_min (2 hours)
3. **Phase 2**: Improve Creat_std transformation (1 hour)
4. **Phase 3**: Add semantic constraints (1 hour)
5. **Validation**: Re-run evaluation, verify improvements

---

## 🔬 Technical Notes

### Why log1p + RobustScaler is correct but insufficient:

The preprocessing **is doing its job** - it's making the data more normal and reducing outliers. The issue is:

1. **Some features are fundamentally hard** (discrete, degenerate)
2. **MSE loss assumes continuous Gaussian** (not true for discrete/skewed)
3. **VAE latent bottleneck loses information** (especially for weak signals)

The solutions target **model-level** improvements, not preprocessing changes.

### Why we see ~50% negative after scaling:

This is **expected and correct**:
- RobustScaler centers at **median** (50th percentile)
- So 50% of values below median → negative
- 50% of values above median → positive
- If you see 42-48% negative, it means slight skew remains (correct!)

