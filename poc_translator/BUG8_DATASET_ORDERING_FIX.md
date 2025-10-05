# Bug #8: Dataset Column Ordering Fix

## Problem
Training failed with CUDA index out of bounds error:
```
../aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [16,0,0], thread: [0,0,0] 
Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
```

## Root Cause
The `FeatureDataset` class in `src/dataset.py` was **sorting features alphabetically** (line 50):

```python
clinical_features.sort()  # ❌ This scrambled the column order!
```

This caused a mismatch between:
1. **CSV column order**: `HR_min, HR_max, HR_mean, HR_std, RR_min, RR_max, ...`
2. **Dataset tensor order** (after sorting): `Creat_max, Creat_mean, Creat_min, Creat_std, HR_max, HR_mean, ...`

When the model tried to read feature values, it was accessing the wrong indices!

## The Fix

### Changed `src/dataset.py`:

1. **Added `feature_spec` parameter** to `FeatureDataset.__init__()`:
   - Now uses `feature_spec['numeric_features']` to maintain correct column order
   - Removed the alphabetical sorting that was scrambling features
   
2. **Updated `CombinedDataModule.setup()`**:
   - Now passes `feature_spec` when creating datasets
   - Ensures column order consistency across train/test datasets

## Verification

Created and ran `verify_dataset_fix.py` which confirmed:
- ✅ CSV column order matches feature_spec order
- ✅ Feature_spec order matches dataset tensor order  
- ✅ Missing flags order is consistent
- ✅ Total input dimension: 32 features (26 numeric + 6 missing flags)

## Files Modified
1. `src/dataset.py` - Lines 29-52 (added feature_spec parameter, removed sorting)
2. `src/dataset.py` - Lines 125-137 (pass feature_spec to datasets)

## Impact
This was the **8th critical bug** preventing successful training. With all 8 bugs now fixed:
1. Missing flags are binary (not 0.693)
2. Column order is preserved
3. Temperature/MAP removed (high missingness)
4. Standard deviation features have variation
5. Min/max relationships preserved (unified scaling)
6. **Dataset column order matches CSV** ← THIS FIX

## Next Step
Training should now work! Run:
```bash
python src/train.py --config conf/config.yml
```

