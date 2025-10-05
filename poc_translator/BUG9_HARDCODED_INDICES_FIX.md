# Bug #9: Hardcoded Feature Indices in Model

## Problem
Even after fixing the dataset column ordering (Bug #8), training still failed with CUDA index out of bounds error:
```
../aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [16,0,0], thread: [0,0,0] 
Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
```

Error occurred at `model.py` line 570:
```python
x_recon_cont = x_recon[:, continuous_indices]
```

## Root Cause
In `model.py` line 562-563, the feature indices were **hardcoded**:

```python
continuous_indices = list(range(34))  # Indices 0-33 (continuous features)  
binary_indices = list(range(34, min(42, x.shape[1])))   # Indices 34+ (missing indicators)
```

This assumed:
- 34 continuous features (8 features × 4 stats + 2 demographics = 34)
- 8 missing indicators
- **Total: 42 features**

But after removing Temp and MAP (Bug #4), we now have:
- 26 numeric features (6 features × 4 stats + 2 demographics = 26)
- 6 missing indicators  
- **Total: 32 features**

So `continuous_indices = list(range(34))` tried to access indices 0-33, but we only have 0-31! ❌

## The Fix

Changed `src/model.py` lines 559-563 to calculate indices dynamically:

```python
# CRITICAL FIX: Calculate indices dynamically based on actual dimensions
# numeric_dim includes all continuous features (clinical + demographic)
# missing_dim includes all binary missing indicators
continuous_indices = list(range(self.numeric_dim))  # All numeric features
binary_indices = list(range(self.numeric_dim, self.numeric_dim + self.missing_dim))  # All missing indicators
```

Now:
- `continuous_indices = list(range(26))` → indices 0-25 ✓
- `binary_indices = list(range(26, 32))` → indices 26-31 ✓

## Verification
Ran dry-run training - no more CUDA errors! The model correctly:
- Uses `input_dim=32` (26 numeric + 6 missing)
- Accesses valid indices 0-31
- Training starts successfully

## Files Modified
- `src/model.py` - Lines 559-563 (dynamic index calculation)

## Impact
This was the **9th critical bug** that was preventing training. The bug was introduced when we removed features but forgot to update the hardcoded indices in the model.

## Key Lesson
**Never hardcode array indices or dimensions!** Always calculate them dynamically from the actual data structure, especially when features can be added/removed via configuration.

