"""
Sepsis source (eICU) data analysis for filtering strategies.
Analyzes label distribution, missingness, and filtering scenarios.
"""

import polars as pl
import numpy as np

# ============================================================
# Load data
# ============================================================
EICU_BASE = "/bigdata/omerg/Thesis/cohort_data/sepsis/eicu"
MIIV_BASE = "/bigdata/omerg/Thesis/cohort_data/sepsis/miiv"

print("=" * 80)
print("SEPSIS SOURCE (eICU) DATA ANALYSIS FOR FILTERING STRATEGIES")
print("=" * 80)

outc = pl.read_parquet(f"{EICU_BASE}/outc.parquet")
dyn = pl.read_parquet(f"{EICU_BASE}/dyn.parquet")
sta = pl.read_parquet(f"{EICU_BASE}/sta.parquet")

print(f"\n--- Raw data shapes ---")
print(f"Outcomes (outc): {outc.shape}  columns: {outc.columns}")
print(f"Dynamic  (dyn):  {dyn.shape}  columns (first 10): {dyn.columns[:10]}...")
print(f"Static   (sta):  {sta.shape}  columns: {sta.columns}")

# Identify the stay ID column and label column
print(f"\nOutcome dtypes:\n{outc.dtypes}")
print(f"Outcome head:\n{outc.head(10)}")

stay_col = outc.columns[0]
time_col = outc.columns[1] if len(outc.columns) > 2 else None
label_col = outc.columns[-1]

print(f"\nDetected: stay_col={stay_col}, time_col={time_col}, label_col={label_col}")

# ============================================================
# 1. ALL-NEGATIVE STAYS
# ============================================================
print("\n" + "=" * 80)
print("1. ALL-NEGATIVE STAYS ANALYSIS")
print("=" * 80)

stay_labels = outc.group_by(stay_col).agg([
    pl.col(label_col).sum().alias("n_positive"),
    pl.col(label_col).count().alias("n_timesteps"),
])

total_stays = stay_labels.height
all_negative = stay_labels.filter(pl.col("n_positive") == 0)
n_all_negative = all_negative.height
positive_stays = stay_labels.filter(pl.col("n_positive") > 0)
n_positive_stays = positive_stays.height

total_timesteps = outc.height
total_positive_ts = outc.select(pl.col(label_col).sum()).item()

print(f"Total stays:          {total_stays}")
print(f"All-negative stays:   {n_all_negative} ({100*n_all_negative/total_stays:.1f}%)")
print(f"Positive stays (>=1): {n_positive_stays} ({100*n_positive_stays/total_stays:.1f}%)")
print(f"Total timesteps:      {total_timesteps}")
print(f"Total positive TS:    {total_positive_ts} ({100*total_positive_ts/total_timesteps:.2f}%)")
print(f"Per-stay positive rate: {100*n_positive_stays/total_stays:.2f}%")

# ============================================================
# 2. POSITIVE TIMESTEP DISTRIBUTION (within positive stays)
# ============================================================
print("\n" + "=" * 80)
print("2. POSITIVE TIMESTEP DISTRIBUTION (within positive stays)")
print("=" * 80)

pos_counts = positive_stays.select("n_positive").to_series().to_numpy()
print(f"Positive stays: {len(pos_counts)}")
print(f"  Min positive timesteps:    {np.min(pos_counts)}")
print(f"  Max positive timesteps:    {np.max(pos_counts)}")
print(f"  Mean positive timesteps:   {np.mean(pos_counts):.1f}")
print(f"  Median positive timesteps: {np.median(pos_counts):.1f}")
print(f"  Std:                       {np.std(pos_counts):.1f}")

bins = [1, 2, 3, 5, 10, 20, 50, 100, 200]
print(f"\nHistogram of positive timesteps per stay:")
for i in range(len(bins)):
    if i == 0:
        count = np.sum(pos_counts == bins[i])
        print(f"  Exactly {bins[i]}:    {count} ({100*count/len(pos_counts):.1f}%)")
    else:
        count = np.sum((pos_counts >= bins[i-1]+1) & (pos_counts <= bins[i]))
        if i < len(bins) - 1:
            print(f"  {bins[i-1]+1:3d}-{bins[i]:3d}:       {count} ({100*count/len(pos_counts):.1f}%)")
        else:
            count_last = np.sum(pos_counts >= bins[i-1]+1)
            print(f"  {bins[i-1]+1:3d}+:         {count_last} ({100*count_last/len(pos_counts):.1f}%)")

print(f"\nPercentiles of positive timesteps:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p:2d}: {np.percentile(pos_counts, p):.0f}")

# ============================================================
# 3. MISSING DATA PER STAY
# ============================================================
print("\n" + "=" * 80)
print("3. MISSING DATA PER STAY")
print("=" * 80)

dyn_cols = dyn.columns
print(f"Dynamic columns ({len(dyn_cols)} total):")
print(f"  First 15: {dyn_cols[:15]}")

dyn_stay_col = dyn_cols[0]
dyn_time_col = dyn_cols[1]

feature_cols = [c for c in dyn_cols if c not in [dyn_stay_col, dyn_time_col]]
n_features = len(feature_cols)
print(f"Number of feature columns: {n_features}")

print(f"\nComputing per-stay missingness (this may take a moment)...")

missingness_per_stay = dyn.select([
    pl.col(dyn_stay_col),
    pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int32) for c in feature_cols]).alias("null_count"),
]).group_by(dyn_stay_col).agg([
    pl.col("null_count").sum().alias("total_nulls"),
    pl.col("null_count").count().alias("n_timesteps"),
]).with_columns(
    (pl.col("total_nulls") / (pl.col("n_timesteps") * n_features)).alias("miss_rate")
)

miss_rates = missingness_per_stay.select("miss_rate").to_series().to_numpy()

print(f"\nMissingness rate distribution across {len(miss_rates)} stays:")
print(f"  Mean:   {np.mean(miss_rates):.4f} ({100*np.mean(miss_rates):.1f}%)")
print(f"  Median: {np.median(miss_rates):.4f} ({100*np.median(miss_rates):.1f}%)")
print(f"  Std:    {np.std(miss_rates):.4f}")
print(f"  Min:    {np.min(miss_rates):.4f}")
print(f"  Max:    {np.max(miss_rates):.4f}")

thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
print(f"\nStays above missingness thresholds:")
for t in thresholds:
    n_above = np.sum(miss_rates > t)
    print(f"  >{100*t:.0f}% missing: {n_above:5d} stays ({100*n_above/len(miss_rates):.1f}%)")

print(f"\nPercentiles of missingness rate:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p:2d}: {100*np.percentile(miss_rates, p):.1f}%")

# ============================================================
# 4. CROSS-TABULATION: Missingness vs Positive/Negative
# ============================================================
print("\n" + "=" * 80)
print("4. CROSS-TABULATION: Missingness vs Label Status")
print("=" * 80)

cross = missingness_per_stay.join(stay_labels, on=dyn_stay_col, how="inner", suffix="_label")
cross = cross.with_columns(
    (pl.col("n_positive") > 0).alias("is_positive"),
)

for threshold, label in [(0.80, "HIGH (>80% missing)"), (0.50, "LOW (<50% missing)")]:
    if "HIGH" in label:
        subset = cross.filter(pl.col("miss_rate") > threshold)
    else:
        subset = cross.filter(pl.col("miss_rate") < threshold)
    
    n_sub = subset.height
    n_pos = subset.filter(pl.col("is_positive")).height
    n_neg = n_sub - n_pos
    
    print(f"\n{label}: {n_sub} stays")
    print(f"  Positive stays: {n_pos} ({100*n_pos/max(n_sub,1):.1f}%)")
    print(f"  Negative stays: {n_neg} ({100*n_neg/max(n_sub,1):.1f}%)")

for lo, hi, label in [(0.50, 0.80, "MEDIUM (50-80% missing)"), (0.0, 0.50, "LOW (<50% missing)")]:
    subset = cross.filter((pl.col("miss_rate") >= lo) & (pl.col("miss_rate") <= hi))
    n_sub = subset.height
    n_pos = subset.filter(pl.col("is_positive")).height
    n_neg = n_sub - n_pos
    print(f"\n{label}: {n_sub} stays")
    print(f"  Positive stays: {n_pos} ({100*n_pos/max(n_sub,1):.1f}%)")
    print(f"  Negative stays: {n_neg} ({100*n_neg/max(n_sub,1):.1f}%)")

# ============================================================
# 5. FILTERING SCENARIOS
# ============================================================
print("\n" + "=" * 80)
print("5. FILTERING SCENARIOS")
print("=" * 80)

def compute_scenario_stats(scenario_name, stay_ids_to_keep, outc_df, s_col, l_col, stay_labels_df):
    kept_labels = stay_labels_df.filter(pl.col(s_col).is_in(stay_ids_to_keep))
    kept_outc = outc_df.filter(pl.col(s_col).is_in(stay_ids_to_keep))
    
    n_stays = kept_labels.height
    n_pos_stays = kept_labels.filter(pl.col("n_positive") > 0).height
    n_total_ts = kept_outc.height
    n_pos_ts = kept_outc.select(pl.col(l_col).sum()).item()
    
    per_stay_rate = n_pos_stays / max(n_stays, 1)
    per_ts_rate = n_pos_ts / max(n_total_ts, 1)
    
    print(f"\n  [{scenario_name}]")
    print(f"  Stays remaining:      {n_stays} (removed {total_stays - n_stays})")
    print(f"  Positive stays:       {n_pos_stays} ({100*per_stay_rate:.1f}%)")
    print(f"  Total timesteps:      {n_total_ts}")
    print(f"  Positive timesteps:   {n_pos_ts} ({100*per_ts_rate:.2f}%)")
    print(f"  Per-stay pos rate:    {100*per_stay_rate:.2f}%")
    print(f"  Per-timestep pos rate:{100*per_ts_rate:.2f}%")
    
    return n_stays, n_pos_stays, n_total_ts, n_pos_ts

# Baseline
print(f"\n  [BASELINE - No filtering]")
print(f"  Stays:                {total_stays}")
print(f"  Positive stays:       {n_positive_stays} ({100*n_positive_stays/total_stays:.1f}%)")
print(f"  Total timesteps:      {total_timesteps}")
print(f"  Positive timesteps:   {total_positive_ts} ({100*total_positive_ts/total_timesteps:.2f}%)")

all_stay_ids = stay_labels.select(stay_col).to_series()
positive_stay_ids = positive_stays.select(stay_col).to_series()

# a. Remove all-negative stays
compute_scenario_stats("5a. Keep only positive stays (>=1 pos TS)",
                       positive_stay_ids, outc, stay_col, label_col, stay_labels)

# b. Remove >90% missingness
miss_df = missingness_per_stay.select([dyn_stay_col, "miss_rate"])
keep_90 = miss_df.filter(pl.col("miss_rate") <= 0.90).select(dyn_stay_col).to_series()
compute_scenario_stats("5b. Remove >90% missingness stays",
                       keep_90, outc, stay_col, label_col, stay_labels)

# c. Remove >80% missingness
keep_80 = miss_df.filter(pl.col("miss_rate") <= 0.80).select(dyn_stay_col).to_series()
compute_scenario_stats("5c. Remove >80% missingness stays",
                       keep_80, outc, stay_col, label_col, stay_labels)

# d. Current oversampling_factor=20 effective ratio
print(f"\n  [5d. Oversampling analysis (factor=20)]")
print(f"  Current per-stay positive rate: {100*n_positive_stays/total_stays:.2f}%")
print(f"  With oversampling_factor=20:")
effective_pos = n_positive_stays * 20
effective_total = (total_stays - n_positive_stays) + effective_pos
effective_rate = effective_pos / effective_total
print(f"    Effective positive stays: {effective_pos} (vs {total_stays - n_positive_stays} negative)")
print(f"    Effective per-stay rate: {100*effective_rate:.1f}%")
print(f"    Note: This oversamples at the STAY level in the sampler,")
print(f"    but per-timestep rate within each stay is unchanged.")

# e. Combinations
keep_e1 = set(positive_stay_ids.to_list()) & set(keep_90.to_list())
compute_scenario_stats("5e1. Positive stays + <=90% miss",
                       list(keep_e1), outc, stay_col, label_col, stay_labels)

keep_e2 = set(positive_stay_ids.to_list()) & set(keep_80.to_list())
compute_scenario_stats("5e2. Positive stays + <=80% miss",
                       list(keep_e2), outc, stay_col, label_col, stay_labels)

neg_stay_ids = all_negative.select(stay_col).to_series()
neg_high_miss = set(neg_stay_ids.to_list()) - set(keep_80.to_list())
keep_e3 = set(all_stay_ids.to_list()) - neg_high_miss
compute_scenario_stats("5e3. Remove high-miss negative stays only (keep all pos, remove neg >80%)",
                       list(keep_e3), outc, stay_col, label_col, stay_labels)

# ============================================================
# 6. PER-TIMESTEP POSITIVE RATE WITHIN POSITIVE STAYS
# ============================================================
print("\n" + "=" * 80)
print("6. PER-TIMESTEP POSITIVE RATE WITHIN POSITIVE STAYS")
print("=" * 80)

pos_outc = outc.filter(pl.col(stay_col).is_in(positive_stay_ids))
pos_ts_total = pos_outc.height
pos_ts_positive = pos_outc.select(pl.col(label_col).sum()).item()

print(f"Positive stays only: {n_positive_stays} stays")
print(f"  Total timesteps:    {pos_ts_total}")
print(f"  Positive timesteps: {pos_ts_positive} ({100*pos_ts_positive/pos_ts_total:.2f}%)")
print(f"  vs full dataset:    {100*total_positive_ts/total_timesteps:.2f}%")
print(f"  Improvement factor: {(pos_ts_positive/pos_ts_total) / (total_positive_ts/total_timesteps):.2f}x")

pos_stay_fracs = positive_stays.with_columns(
    (pl.col("n_positive") / pl.col("n_timesteps")).alias("pos_frac")
).select("pos_frac").to_series().to_numpy()

print(f"\nPer-stay positive fraction distribution (within positive stays):")
print(f"  Mean:   {100*np.mean(pos_stay_fracs):.1f}%")
print(f"  Median: {100*np.median(pos_stay_fracs):.1f}%")
print(f"  Min:    {100*np.min(pos_stay_fracs):.1f}%")
print(f"  Max:    {100*np.max(pos_stay_fracs):.1f}%")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}:   {100*np.percentile(pos_stay_fracs, p):.1f}%")

# ============================================================
# 7. MIMIC (TARGET) DATA COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("7. MIMIC-IV (TARGET) DATA COMPARISON")
print("=" * 80)

try:
    mimic_outc = pl.read_parquet(f"{MIIV_BASE}/outc.parquet")
    mimic_dyn = pl.read_parquet(f"{MIIV_BASE}/dyn.parquet")
    
    m_stay_col = mimic_outc.columns[0]
    m_label_col = mimic_outc.columns[-1]
    
    m_stay_labels = mimic_outc.group_by(m_stay_col).agg([
        pl.col(m_label_col).sum().alias("n_positive"),
        pl.col(m_label_col).count().alias("n_timesteps"),
    ])
    
    m_total_stays = m_stay_labels.height
    m_pos_stays = m_stay_labels.filter(pl.col("n_positive") > 0).height
    m_neg_stays = m_total_stays - m_pos_stays
    m_total_ts = mimic_outc.height
    m_pos_ts = mimic_outc.select(pl.col(m_label_col).sum()).item()
    
    print(f"MIMIC-IV sepsis data:")
    print(f"  Total stays:          {m_total_stays}")
    print(f"  Positive stays:       {m_pos_stays} ({100*m_pos_stays/m_total_stays:.1f}%)")
    print(f"  All-negative stays:   {m_neg_stays} ({100*m_neg_stays/m_total_stays:.1f}%)")
    print(f"  Total timesteps:      {m_total_ts}")
    print(f"  Positive timesteps:   {m_pos_ts} ({100*m_pos_ts/m_total_ts:.2f}%)")
    
    # MIMIC missingness
    m_feature_cols = [c for c in mimic_dyn.columns if c not in [mimic_dyn.columns[0], mimic_dyn.columns[1]]]
    m_n_features = len(m_feature_cols)
    
    m_miss_per_stay = mimic_dyn.select([
        pl.col(mimic_dyn.columns[0]),
        pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int32) for c in m_feature_cols]).alias("null_count"),
    ]).group_by(mimic_dyn.columns[0]).agg([
        pl.col("null_count").sum().alias("total_nulls"),
        pl.col("null_count").count().alias("n_timesteps"),
    ]).with_columns(
        (pl.col("total_nulls") / (pl.col("n_timesteps") * m_n_features)).alias("miss_rate")
    )
    
    m_miss_rates = m_miss_per_stay.select("miss_rate").to_series().to_numpy()
    print(f"\n  MIMIC missingness:")
    print(f"  Mean:   {100*np.mean(m_miss_rates):.1f}%")
    print(f"  Median: {100*np.median(m_miss_rates):.1f}%")
    
    m_seq_lens = m_stay_labels.select("n_timesteps").to_series().to_numpy()
    e_seq_lens = stay_labels.select("n_timesteps").to_series().to_numpy()
    
    print(f"\n  Sequence length comparison:")
    print(f"                eICU         MIMIC-IV")
    print(f"  Mean:         {np.mean(e_seq_lens):.1f}        {np.mean(m_seq_lens):.1f}")
    print(f"  Median:       {np.median(e_seq_lens):.0f}           {np.median(m_seq_lens):.0f}")
    print(f"  Max:          {np.max(e_seq_lens)}           {np.max(m_seq_lens)}")
    
    print(f"\n  --- COMPARISON SUMMARY ---")
    print(f"                        eICU (source)    MIMIC-IV (target)")
    print(f"  Total stays:          {total_stays:>10}       {m_total_stays:>10}")
    print(f"  Per-stay pos rate:    {100*n_positive_stays/total_stays:>9.1f}%       {100*m_pos_stays/m_total_stays:>9.1f}%")
    print(f"  Per-TS pos rate:      {100*total_positive_ts/total_timesteps:>9.2f}%       {100*m_pos_ts/m_total_ts:>9.2f}%")
    print(f"  Mean missingness:     {100*np.mean(miss_rates):>9.1f}%       {100*np.mean(m_miss_rates):>9.1f}%")
    
except FileNotFoundError as e:
    print(f"  Could not load MIMIC data: {e}")
except Exception as e:
    import traceback
    print(f"  Error analyzing MIMIC data: {e}")
    traceback.print_exc()

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY: FILTERING SCENARIO COMPARISON TABLE")
print("=" * 80)

print(f"\n{'Scenario':<55} {'Stays':>7} {'Pos%':>7} {'TS':>10} {'TS Pos%':>8}")
print("-" * 90)

scenarios = [
    ("Baseline (no filter)", total_stays, n_positive_stays/total_stays, total_timesteps, total_positive_ts/total_timesteps),
]

for name, keep_ids in [
    ("Keep positive stays only", positive_stay_ids.to_list()),
    ("Remove >90% miss", keep_90.to_list()),
    ("Remove >80% miss", keep_80.to_list()),
    ("Positive + <=90% miss", list(keep_e1)),
    ("Positive + <=80% miss", list(keep_e2)),
    ("Remove neg >80% miss (keep all pos)", list(keep_e3)),
]:
    kl = stay_labels.filter(pl.col(stay_col).is_in(keep_ids))
    ko = outc.filter(pl.col(stay_col).is_in(keep_ids))
    ns = kl.height
    nps = kl.filter(pl.col("n_positive") > 0).height
    nts = ko.height
    npts = ko.select(pl.col(label_col).sum()).item()
    scenarios.append((name, ns, nps/max(ns,1), nts, npts/max(nts,1)))

for name, ns, psr, nts, ptsr in scenarios:
    print(f"{name:<55} {ns:>7} {100*psr:>6.1f}% {nts:>10} {100*ptsr:>7.2f}%")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
