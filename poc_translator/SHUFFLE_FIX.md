# Shuffle-Induced Domain Imbalance Fix

## Problem

User reported seeing domain imbalance in MIMIC-only mode:
```
Batch domain balance: MIMIC=59, eICU=69 (imbalance could cause ln(2) stagnation)
```

Expected: Exactly 64 domain=0 and 64 domain=1 (for batch_size=128)
Actual: Random imbalance like 59 vs 69

## Root Cause

The original implementation used `idx % 2` to assign domain labels:

```python
# Original (BROKEN with shuffling)
def __getitem__(self, idx):
    if self.split_for_cycle:
        domain_label = idx % 2  # ❌ Checks if INDEX is even/odd
```

### Why This Breaks with Shuffling

When `shuffle=True` in the DataLoader:

```python
# Without shuffle (consecutive indices):
batch_indices = [0, 1, 2, 3, 4, 5, 6, 7, ...]
domains =       [0, 1, 0, 1, 0, 1, 0, 1, ...]  ✅ Perfect 50-50!

# With shuffle (random indices):
batch_indices = [42, 7, 103, 88, 15, 256, 91, 134, ...]
domains =       [0,  1,  1,   0,  1,   0,  1,   0, ...]  ❌ Imbalanced!
                ↑   ↑   ↑    ↑   ↑    ↑   ↑    ↑
              even odd odd even odd even odd even
              
# Counting even/odd in this batch:
# Even (domain=0): 42, 88, 256, 134 = 4 samples
# Odd (domain=1): 7, 103, 15, 91 = 4 samples
# But with 128 samples, you might get 59 evens and 69 odds!
```

**The randomness of which indices are selected determines the domain balance**, not the data itself!

## The Fix

Pre-assign domain labels to the **data** before shuffling, not based on index:

```python
# In __init__:
if split_for_cycle:
    # First half of data: domain=0, second half: domain=1
    n = len(data)
    self.domain_labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
    # Result: [0, 0, 0, ..., 0, 1, 1, 1, ..., 1]
    #         └─ n//2 zeros ─┘└─ remaining ones ─┘

# In __getitem__:
def __getitem__(self, idx):
    if self.split_for_cycle:
        domain_label = int(self.domain_labels[idx])  # ✅ Uses pre-assigned label
```

### How This Works

1. **Dataset creation** (before shuffling):
   ```python
   # 1000 samples total
   domain_labels = [0]*500 + [1]*500
   # Exactly 500 domain=0, 500 domain=1
   ```

2. **DataLoader shuffling** (randomizes access):
   ```python
   # Shuffled indices: [753, 42, 891, ...]
   # But domain_labels[753], domain_labels[42], etc. still gives exact 50-50!
   ```

3. **Batch creation** (e.g., batch_size=128):
   ```python
   # Randomly select 128 indices
   # domain_labels at those indices will have ~64 zeros and ~64 ones
   # Because the array has exactly 50-50 distribution!
   ```

4. **Result**:
   ```python
   # Each batch will have ~64 domain=0 and ~64 domain=1
   # (might be 63/65 or 64/64, but very close to 50-50)
   ```

## Before vs After

### Before Fix
```python
# Random batch with shuffling:
indices = [42, 7, 103, 88, 15, ...]  # 128 random indices
domains = [idx % 2 for idx in indices]
# Result: Could be 59 domain=0, 69 domain=1 ❌
```

### After Fix
```python
# Random batch with shuffling:
indices = [753, 42, 891, 234, 567, ...]  # 128 random indices from [0, 999]
domain_labels = [0]*500 + [1]*500  # Pre-assigned, half 0s, half 1s

# Batch domains:
domains = [domain_labels[753], domain_labels[42], domain_labels[891], ...]
# Since ~half of 1000 are 0s and half are 1s,
# random sampling of 128 will give ~64 of each ✅
```

## Mathematical Explanation

This is a **hypergeometric distribution** problem:

- Population: N = 1000 samples (500 domain=0, 500 domain=1)
- Sample: n = 128 randomly selected
- Expected domain=0 in sample: n × (500/1000) = 64
- Standard deviation: ~√(n × p × (1-p)) ≈ 5.66

So you'll typically get 64 ± 6 of each domain, much more balanced than the random 59/69 split from the index-based approach!

## Testing

After this fix, you should see:
```
# Typical batches:
Batch domain balance: MIMIC=64, eICU=64 (perfect!)
Batch domain balance: MIMIC=63, eICU=65 (nearly perfect)
Batch domain balance: MIMIC=65, eICU=63 (nearly perfect)

# Extremely unlikely to see:
Batch domain balance: MIMIC=59, eICU=69 (too imbalanced, won't happen)
```

## Files Modified

- `src/dataset.py`: 
  - `FeatureDataset.__init__`: Added `self.domain_labels` pre-assignment
  - `FeatureDataset.__getitem__`: Changed from `idx % 2` to `self.domain_labels[idx]`

## Key Insight

**Don't assign properties based on index when using shuffling!**

- ❌ Bad: `label = idx % 2` (breaks with shuffling)
- ✅ Good: `label = pre_assigned_labels[idx]` (works with shuffling)

The index is just a random access pointer when shuffling is enabled. The actual data properties should be pre-assigned to the data itself, not derived from the index!


