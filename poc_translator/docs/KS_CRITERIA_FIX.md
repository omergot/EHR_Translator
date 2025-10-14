# KS Test Criteria Fix: Removing P-value Dependency

## Problem

The original evaluation criteria for distribution matching used:
```python
df['eicu_to_mimic_good'] = (df['eicu_to_mimic_ks'] < 0.3) & (df['eicu_to_mimic_pvalue'] > 0.05)
```

This caused **all features to be marked as "bad"** (0/24 good) because:
1. With large sample sizes (N=5731 per domain), **all p-values < 0.05**
2. The p-value threshold (p > 0.05) was impossible to meet with large N
3. P-values test statistical significance, not effect size
4. P-values are **uninformative for large samples** - even tiny KS differences become "significant"

## Solution

Changed to use **KS statistic magnitude only** (effect size), with three quality tiers:

```python
# KS interpretation: <0.1=excellent, <0.2=good, <0.3=acceptable, >0.5=poor
df['eicu_to_mimic_excellent'] = (df['eicu_to_mimic_ks'] < 0.1)
df['eicu_to_mimic_good'] = (df['eicu_to_mimic_ks'] < 0.2)
df['eicu_to_mimic_acceptable'] = (df['eicu_to_mimic_ks'] < 0.3)
```

### KS Statistic Interpretation Guide

| KS Range | Quality | Interpretation |
|----------|---------|----------------|
| < 0.1 | Excellent | Distributions nearly identical |
| 0.1-0.2 | Good | Distributions very similar |
| 0.2-0.3 | Acceptable | Noticeable but acceptable difference |
| 0.3-0.5 | Poor | Significant distribution mismatch |
| > 0.5 | Very Poor | Major distribution mismatch |

### Why P-values Don't Help Here

With large N, the KS test becomes **overly sensitive**:
- Even KS=0.05 (excellent match) → p < 0.001 (significant)
- The p-value tells us "distributions differ" but not **how much**
- We care about **effect size** (KS magnitude), not statistical significance

### Example from Actual Data

**Before fix:**
```
Distribution Matching (KS < 0.3, p > 0.05):
- eICU→MIMIC Translation: 0/24 features (0.0%)  ❌
- MIMIC→eICU Translation: 0/24 features (0.0%)  ❌
```

**After fix (same data, better interpretation):**
```
Distribution Matching (KS statistic - effect size):
eICU→MIMIC Translation:
- Excellent (KS<0.1): 5/24 (20.8%)
- Good (KS<0.2): 17/24 (70.8%)      ✓
- Acceptable (KS<0.3): 20/24 (83.3%) ✓
- Mean KS: 0.183

MIMIC→eICU Translation:
- Excellent (KS<0.1): 11/24 (45.8%)
- Good (KS<0.2): 17/24 (70.8%)      ✓
- Acceptable (KS<0.3): 20/24 (83.3%) ✓
- Mean KS: 0.176
```

Much more informative! The model is actually performing **quite well** on distribution matching.

## Changes Made

### 1. `comprehensive_evaluator.py`
- Lines 434-441: Added three-tier KS-based quality flags (excellent/good/acceptable)
- Lines 1066-1092: Updated console output to show all three tiers + mean KS
- Removed p-value from quality criteria

### 2. `evaluate.py`
- Lines 840-866: Updated executive summary section with three-tier breakdown
- Lines 1077-1103: Updated distribution analysis section
- Lines 1195-1198: Updated recommendations to use "acceptable" threshold
- Added note explaining p-values not used with large N

## Files Modified

- `src/comprehensive_evaluator.py`
- `src/evaluate.py`

## Backward Compatibility

The CSV output (`ks_analysis.csv`) still contains:
- `eicu_to_mimic_pvalue` (for reference)
- `eicu_to_mimic_significant` (for reference)
- **New columns**: `eicu_to_mimic_excellent`, `eicu_to_mimic_good`, `eicu_to_mimic_acceptable`

Old reports will need regeneration to use the new criteria.

## Validation

To verify the fix works, re-run comprehensive evaluation:
```bash
python src/evaluate.py --config conf/config.yml --model output/model.ckpt \
  --output-dir output --comprehensive --mimic-only
```

Expected output should now show meaningful distribution matching percentages (not 0/24).

## References

- Massey, F.J. (1951). "The Kolmogorov-Smirnov Test for Goodness of Fit"
- Common guideline: KS < 0.2 indicates distributions are similar enough for practical purposes
- P-value interpretation changes dramatically with sample size (see: Sullivan & Feinn, 2012, "Using Effect Size")

