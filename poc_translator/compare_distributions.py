#!/usr/bin/env python3
"""
Compare distribution matching quality between trained and untrained models.
This is the REAL test - does translation actually match the target distribution?

Reconstruction is trivial with skip connections, but translation quality is what matters.
"""

import torch
import numpy as np
import pandas as pd
import json
import yaml
import sys
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, 'src')
from model import CycleVAE

def compute_ks_statistics(source_data, translated_data, target_data, feature_names):
    """
    Compute KS and Wasserstein statistics to measure how well translated data matches target distribution.
    
    Uses KS statistic threshold instead of p-value (more reliable for large n).
    KS < 0.1: distributions match well
    
    Returns:
        DataFrame with KS statistics, Wasserstein distance, and p-values
    """
    KS_THRESHOLD = 0.1  # Threshold for considering distributions as matching
    n_features = source_data.shape[1]
    results = []
    
    for i in range(n_features):
        # KS test: translated vs target (how well does translation match target?)
        ks_stat_translation, p_val_translation = stats.ks_2samp(translated_data[:, i], target_data[:, i])
        
        # KS test: source vs target (baseline - how different are the domains?)
        ks_stat_baseline, p_val_baseline = stats.ks_2samp(source_data[:, i], target_data[:, i])
        
        # Wasserstein distance: translated vs target
        wass_translation = stats.wasserstein_distance(translated_data[:, i], target_data[:, i])
        
        # Wasserstein distance: source vs target (baseline)
        wass_baseline = stats.wasserstein_distance(source_data[:, i], target_data[:, i])
        
        results.append({
            'feature': feature_names[i] if i < len(feature_names) else f'Feature_{i}',
            'ks_translation': ks_stat_translation,
            'p_val_translation': p_val_translation,
            'ks_baseline': ks_stat_baseline,
            'p_val_baseline': p_val_baseline,
            'wass_translation': wass_translation,
            'wass_baseline': wass_baseline,
            'ks_improvement': ks_stat_baseline - ks_stat_translation,  # Positive = better translation
            'wass_improvement': wass_baseline - wass_translation,  # Positive = better translation
            'translation_good': ks_stat_translation < KS_THRESHOLD  # KS threshold (not p-value for large n)
        })
    
    return pd.DataFrame(results)

def compute_distribution_distances(source_data, translated_data, target_data):
    """Compute mean absolute difference in distribution statistics."""
    # Mean differences
    target_means = target_data.mean(axis=0)
    translated_means = translated_data.mean(axis=0)
    source_means = source_data.mean(axis=0)
    
    mean_error_translation = np.abs(translated_means - target_means).mean()
    mean_error_baseline = np.abs(source_means - target_means).mean()
    
    # Std differences
    target_stds = target_data.std(axis=0)
    translated_stds = translated_data.std(axis=0)
    source_stds = source_data.std(axis=0)
    
    std_error_translation = np.abs(translated_stds - target_stds).mean()
    std_error_baseline = np.abs(source_stds - target_stds).mean()
    
    return {
        'mean_error_translation': mean_error_translation,
        'mean_error_baseline': mean_error_baseline,
        'std_error_translation': std_error_translation,
        'std_error_baseline': std_error_baseline
    }

def evaluate_translation_quality(model, x_source, x_target, direction_name):
    """Evaluate how well the model translates from source to target domain."""
    device = next(model.parameters()).device
    x_source_dev = x_source.to(device)
    x_target_dev = x_target.to(device)
    
    # Compute IQR for outlier detection
    numeric_dim = model.numeric_dim
    all_numeric = torch.cat([x_source_dev[:, :numeric_dim], x_target_dev[:, :numeric_dim]], dim=0)
    model.feature_iqr = model.compute_feature_iqr(all_numeric)
    
    model.eval()
    with torch.no_grad():
        if 'eICU→MIMIC' in direction_name:
            x_translated = model.translate_eicu_to_mimic_deterministic(x_source_dev)
        else:  # MIMIC→eICU
            x_translated = model.translate_mimic_to_eicu_deterministic(x_source_dev)
    
    x_source_np = x_source_dev.cpu().numpy()
    x_translated_np = x_translated.cpu().numpy()
    x_target_np = x_target_dev.cpu().numpy()
    
    return x_source_np, x_translated_np, x_target_np

def main():
    print("="*80)
    print("DISTRIBUTION MATCHING COMPARISON: Trained vs Untrained")
    print("="*80)
    print("\nThis compares the REAL test: Does translation match target distribution?")
    print("Using REAL eICU and REAL MIMIC test data with different distributions.")
    print("(Reconstruction is trivial with skip connections, so we focus on translation)\n")
    
    # Load configuration
    with open('feature_spec.json') as f:
        feature_spec = json.load(f)
    with open('config_used.yml') as f:
        config = yaml.safe_load(f)
    
    # Load test data - use REAL eICU and MIMIC data
    print("Loading test data...")
    eicu_df = pd.read_csv('data/test_eicu_preprocessed.csv')
    mimic_df = pd.read_csv('data/test_mimic_preprocessed.csv')
    print(f"  eICU test samples: {len(eicu_df)}")
    print(f"  MIMIC test samples: {len(mimic_df)}")
    
    # Convert to tensors
    numeric_features = feature_spec['numeric_features']
    missing_features = feature_spec['missing_features']
    all_features = numeric_features + missing_features
    
    eicu_numeric = torch.FloatTensor(eicu_df[[f for f in numeric_features if f in eicu_df.columns]].values)
    eicu_missing = torch.FloatTensor(eicu_df[[f for f in missing_features if f in eicu_df.columns]].values)
    mimic_numeric = torch.FloatTensor(mimic_df[[f for f in numeric_features if f in mimic_df.columns]].values)
    mimic_missing = torch.FloatTensor(mimic_df[[f for f in missing_features if f in mimic_df.columns]].values)
    
    x_eicu = torch.cat([eicu_numeric, eicu_missing], dim=1)
    x_mimic = torch.cat([mimic_numeric, mimic_missing], dim=1)
    
    print(f"  MIMIC samples: {len(mimic_df)}")
    print(f"  eICU samples: {len(eicu_df)}")
    
    # =====================================================
    # 1. TRAINED MODEL
    # =====================================================
    print("\n" + "="*80)
    print("1. TRAINED MODEL - Translation Quality")
    print("="*80)
    
    checkpoint = torch.load('checkpoints/final_model.ckpt', map_location='cpu', weights_only=False)
    trained_model = CycleVAE(config, feature_spec)
    trained_model.load_state_dict(checkpoint['state_dict'])
    
    # Evaluate eICU→MIMIC translation
    print("\nEvaluating eICU→MIMIC translation...")
    eicu_src_t, eicu_to_mimic_t, mimic_tgt_t = evaluate_translation_quality(
        trained_model, x_eicu, x_mimic, 'eICU→MIMIC'
    )
    
    # Only evaluate clinical features (exclude demographics and missing flags)
    demographic_features = feature_spec.get('demographic_features', ['Age', 'Gender'])
    clinical_features = [f for f in numeric_features if f not in demographic_features]
    clinical_indices = [i for i, f in enumerate(all_features) if f in clinical_features]
    
    eicu_src_clinical_t = eicu_src_t[:, clinical_indices]
    eicu_to_mimic_clinical_t = eicu_to_mimic_t[:, clinical_indices]
    mimic_tgt_clinical_t = mimic_tgt_t[:, clinical_indices]
    
    ks_trained = compute_ks_statistics(eicu_src_clinical_t, eicu_to_mimic_clinical_t, 
                                       mimic_tgt_clinical_t, clinical_features)
    dist_trained = compute_distribution_distances(eicu_src_clinical_t, eicu_to_mimic_clinical_t,
                                                  mimic_tgt_clinical_t)
    
    print(f"\nKS Statistics (eICU→MIMIC):")
    print(f"  Mean KS (source vs target):      {ks_trained['ks_baseline'].mean():.6f}")
    print(f"  Mean KS (translated vs target):  {ks_trained['ks_translation'].mean():.6f}")
    print(f"  KS Improvement:                   {ks_trained['ks_improvement'].mean():.6f}")
    print(f"  Features matching distribution:   {ks_trained['translation_good'].sum()}/{len(ks_trained)}")
    
    print(f"\nWasserstein Distance (eICU→MIMIC):")
    print(f"  Mean Wass (source vs target):    {ks_trained['wass_baseline'].mean():.6f}")
    print(f"  Mean Wass (translated vs target): {ks_trained['wass_translation'].mean():.6f}")
    print(f"  Wass Improvement:                 {ks_trained['wass_improvement'].mean():.6f}")
    
    print(f"\nDistribution Distance:")
    print(f"  Mean error (source):              {dist_trained['mean_error_baseline']:.6f}")
    print(f"  Mean error (translated):          {dist_trained['mean_error_translation']:.6f}")
    print(f"  Std error (source):               {dist_trained['std_error_baseline']:.6f}")
    print(f"  Std error (translated):           {dist_trained['std_error_translation']:.6f}")
    
    # =====================================================
    # 2. UNTRAINED MODEL
    # =====================================================
    print("\n" + "="*80)
    print("2. UNTRAINED MODEL - Translation Quality")
    print("="*80)
    
    untrained_model = CycleVAE(config, feature_spec)
    # Don't load checkpoint - random initialization
    
    print("\nEvaluating eICU→MIMIC translation...")
    eicu_src_u, eicu_to_mimic_u, mimic_tgt_u = evaluate_translation_quality(
        untrained_model, x_eicu, x_mimic, 'eICU→MIMIC'
    )
    
    eicu_src_clinical_u = eicu_src_u[:, clinical_indices]
    eicu_to_mimic_clinical_u = eicu_to_mimic_u[:, clinical_indices]
    mimic_tgt_clinical_u = mimic_tgt_u[:, clinical_indices]
    
    ks_untrained = compute_ks_statistics(eicu_src_clinical_u, eicu_to_mimic_clinical_u,
                                         mimic_tgt_clinical_u, clinical_features)
    dist_untrained = compute_distribution_distances(eicu_src_clinical_u, eicu_to_mimic_clinical_u,
                                                    mimic_tgt_clinical_u)
    
    print(f"\nKS Statistics (eICU→MIMIC):")
    print(f"  Mean KS (source vs target):      {ks_untrained['ks_baseline'].mean():.6f}")
    print(f"  Mean KS (translated vs target):  {ks_untrained['ks_translation'].mean():.6f}")
    print(f"  KS Improvement:                   {ks_untrained['ks_improvement'].mean():.6f}")
    print(f"  Features matching distribution:   {ks_untrained['translation_good'].sum()}/{len(ks_untrained)}")
    
    print(f"\nWasserstein Distance (eICU→MIMIC):")
    print(f"  Mean Wass (source vs target):    {ks_untrained['wass_baseline'].mean():.6f}")
    print(f"  Mean Wass (translated vs target): {ks_untrained['wass_translation'].mean():.6f}")
    print(f"  Wass Improvement:                 {ks_untrained['wass_improvement'].mean():.6f}")
    
    print(f"\nDistribution Distance:")
    print(f"  Mean error (source):              {dist_untrained['mean_error_baseline']:.6f}")
    print(f"  Mean error (translated):          {dist_untrained['mean_error_translation']:.6f}")
    print(f"  Std error (source):               {dist_untrained['std_error_baseline']:.6f}")
    print(f"  Std error (translated):           {dist_untrained['std_error_translation']:.6f}")
    
    # =====================================================
    # 3. COMPARISON
    # =====================================================
    print("\n" + "="*80)
    print("COMPARISON: Distribution Matching Quality")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Metric': [
            'Mean KS (translated vs target)',
            'Mean KS improvement over source',
            'Mean Wasserstein (translated vs target)',
            'Mean Wasserstein improvement',
            'Features matching target distribution',
            'Mean absolute error (means)',
            'Mean absolute error (stds)'
        ],
        'Trained': [
            f"{ks_trained['ks_translation'].mean():.6f}",
            f"{ks_trained['ks_improvement'].mean():.6f}",
            f"{ks_trained['wass_translation'].mean():.6f}",
            f"{ks_trained['wass_improvement'].mean():.6f}",
            f"{ks_trained['translation_good'].sum()}/{len(ks_trained)}",
            f"{dist_trained['mean_error_translation']:.6f}",
            f"{dist_trained['std_error_translation']:.6f}"
        ],
        'Untrained': [
            f"{ks_untrained['ks_translation'].mean():.6f}",
            f"{ks_untrained['ks_improvement'].mean():.6f}",
            f"{ks_untrained['wass_translation'].mean():.6f}",
            f"{ks_untrained['wass_improvement'].mean():.6f}",
            f"{ks_untrained['translation_good'].sum()}/{len(ks_untrained)}",
            f"{dist_untrained['mean_error_translation']:.6f}",
            f"{dist_untrained['std_error_translation']:.6f}"
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # =====================================================
    # 4. DETAILED ANALYSIS
    # =====================================================
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    ks_improvement = ks_untrained['ks_translation'].mean() - ks_trained['ks_translation'].mean()
    wass_improvement = ks_untrained['wass_translation'].mean() - ks_trained['wass_translation'].mean()
    mean_improvement = dist_untrained['mean_error_translation'] - dist_trained['mean_error_translation']
    
    print(f"\nTrained model improvements over untrained:")
    print(f"  KS distance reduction:        {ks_improvement:.6f} ({ks_improvement/ks_untrained['ks_translation'].mean()*100:.1f}%)")
    print(f"  Wasserstein reduction:        {wass_improvement:.6f} ({wass_improvement/ks_untrained['wass_translation'].mean()*100:.1f}%)")
    print(f"  Mean error reduction:         {mean_improvement:.6f} ({mean_improvement/dist_untrained['mean_error_translation']*100:.1f}%)")
    print(f"  Additional features matching: +{ks_trained['translation_good'].sum() - ks_untrained['translation_good'].sum()}")
    
    if ks_improvement > 0.01:
        print("\n✅ EXCELLENT: Training significantly improved distribution matching!")
        print("   The model learned to translate between domains effectively.")
    elif ks_improvement > 0.001:
        print("\n✅ GOOD: Training improved distribution matching.")
    elif ks_improvement > 0:
        print("\n⚠  MODEST: Training provided some improvement.")
    else:
        print("\n❌ WARNING: Training did not improve distribution matching.")
    
    # Save results
    output_dir = Path('evaluation_comparison')
    output_dir.mkdir(exist_ok=True)
    
    # Merge KS and Wasserstein results
    comparison_df = pd.DataFrame({
        'feature': clinical_features,
        'trained_ks': ks_trained['ks_translation'].values,
        'untrained_ks': ks_untrained['ks_translation'].values,
        'ks_improvement': ks_untrained['ks_translation'].values - ks_trained['ks_translation'].values,
        'trained_wasserstein': ks_trained['wass_translation'].values,
        'untrained_wasserstein': ks_untrained['wass_translation'].values,
        'wass_improvement': ks_untrained['wass_translation'].values - ks_trained['wass_translation'].values,
        'trained_matches_distribution': ks_trained['translation_good'].values,
        'untrained_matches_distribution': ks_untrained['translation_good'].values
    })
    
    comparison_df.to_csv(output_dir / 'distribution_comparison.csv', index=False)
    print(f"\n✓ Detailed results saved to: {output_dir / 'distribution_comparison.csv'}")
    
    # =====================================================
    # 5. DETAILED CSV ANALYSIS
    # =====================================================
    print("\n" + "="*80)
    print("DETAILED PER-FEATURE ANALYSIS")
    print("="*80)
    
    # Sort by KS improvement (best to worst)
    df_sorted_ks = comparison_df.sort_values('ks_improvement', ascending=False)
    df_sorted_wass = comparison_df.sort_values('wass_improvement', ascending=False)
    
    print("\n📊 ALL FEATURES - Sorted by KS Improvement (Best to Worst):")
    print("-" * 80)
    for idx, row in df_sorted_ks.iterrows():
        improvement_pct = (row['ks_improvement'] / row['untrained_ks'] * 100) if row['untrained_ks'] > 0 else 0
        status = "✓" if row['ks_improvement'] > 0 else "✗" if row['ks_improvement'] < 0 else "="
        match_status = "✓" if row['trained_matches_distribution'] else "✗"
        print(f"  {status} {row['feature']:12s}: KS {row['ks_improvement']:+.6f} ({improvement_pct:+.1f}%) [{match_status}]")
        print(f"                  Trained: {row['trained_ks']:.6f} | Untrained: {row['untrained_ks']:.6f}")
    
    print("\n📊 ALL FEATURES - Sorted by Wasserstein Improvement (Best to Worst):")
    print("-" * 80)
    for idx, row in df_sorted_wass.iterrows():
        improvement_pct = (row['wass_improvement'] / row['untrained_wasserstein'] * 100) if row['untrained_wasserstein'] > 0 else 0
        status = "✓" if row['wass_improvement'] > 0 else "✗" if row['wass_improvement'] < 0 else "="
        match_status = "✓" if row['trained_matches_distribution'] else "✗"
        print(f"  {status} {row['feature']:12s}: Wass {row['wass_improvement']:+.6f} ({improvement_pct:+.1f}%) [{match_status}]")
        print(f"                  Trained: {row['trained_wasserstein']:.6f} | Untrained: {row['untrained_wasserstein']:.6f}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    n_features = len(comparison_df)
    ks_improved = (comparison_df['ks_improvement'] > 0).sum()
    ks_regressed = (comparison_df['ks_improvement'] < 0).sum()
    ks_unchanged = (comparison_df['ks_improvement'] == 0).sum()
    
    wass_improved = (comparison_df['wass_improvement'] > 0).sum()
    wass_regressed = (comparison_df['wass_improvement'] < 0).sum()
    wass_unchanged = (comparison_df['wass_improvement'] == 0).sum()
    
    print(f"\nKS Statistic:")
    print(f"  ✓ Improved:   {ks_improved}/{n_features} ({ks_improved/n_features*100:.1f}%)")
    print(f"  ✗ Regressed:  {ks_regressed}/{n_features} ({ks_regressed/n_features*100:.1f}%)")
    print(f"  = Unchanged:  {ks_unchanged}/{n_features} ({ks_unchanged/n_features*100:.1f}%)")
    print(f"  Mean improvement: {comparison_df['ks_improvement'].mean():.6f}")
    print(f"  Median improvement: {comparison_df['ks_improvement'].median():.6f}")
    print(f"  Std improvement: {comparison_df['ks_improvement'].std():.6f}")
    
    print(f"\nWasserstein Distance:")
    print(f"  ✓ Improved:   {wass_improved}/{n_features} ({wass_improved/n_features*100:.1f}%)")
    print(f"  ✗ Regressed:  {wass_regressed}/{n_features} ({wass_regressed/n_features*100:.1f}%)")
    print(f"  = Unchanged:  {wass_unchanged}/{n_features} ({wass_unchanged/n_features*100:.1f}%)")
    print(f"  Mean improvement: {comparison_df['wass_improvement'].mean():.6f}")
    print(f"  Median improvement: {comparison_df['wass_improvement'].median():.6f}")
    print(f"  Std improvement: {comparison_df['wass_improvement'].std():.6f}")
    
    # Correlation analysis
    correlation = comparison_df['ks_improvement'].corr(comparison_df['wass_improvement'])
    print(f"\n📈 Correlation between KS and Wasserstein improvements: {correlation:.4f}")
    
    if correlation > 0.5:
        print("   → Strong agreement: Both metrics improve together")
    elif correlation > 0.2:
        print("   → Moderate agreement: Metrics somewhat aligned")
    elif correlation > -0.2:
        print("   → Weak correlation: Metrics capture different aspects")
    else:
        print("   → Disagreement: Metrics may be conflicting")
    
    # Identify features with disagreement (KS improved but Wass regressed, or vice versa)
    ks_up_wass_down = comparison_df[(comparison_df['ks_improvement'] > 0) & (comparison_df['wass_improvement'] < 0)]
    ks_down_wass_up = comparison_df[(comparison_df['ks_improvement'] < 0) & (comparison_df['wass_improvement'] > 0)]
    
    if len(ks_up_wass_down) > 0:
        print(f"\n🔀 {len(ks_up_wass_down)} features with CONFLICTING improvements (KS↑ but Wass↓):")
        for idx, row in ks_up_wass_down.iterrows():
            print(f"   {row['feature']:12s}: KS {row['ks_improvement']:+.6f}, Wass {row['wass_improvement']:+.6f}")
    
    if len(ks_down_wass_up) > 0:
        print(f"\n🔀 {len(ks_down_wass_up)} features with CONFLICTING improvements (KS↓ but Wass↑):")
        for idx, row in ks_down_wass_up.iterrows():
            print(f"   {row['feature']:12s}: KS {row['ks_improvement']:+.6f}, Wass {row['wass_improvement']:+.6f}")
    
    # Identify features where training made them match distribution
    newly_matching = comparison_df[comparison_df['trained_matches_distribution'] & ~comparison_df['untrained_matches_distribution']]
    lost_matching = comparison_df[~comparison_df['trained_matches_distribution'] & comparison_df['untrained_matches_distribution']]
    
    if len(newly_matching) > 0:
        print(f"\n✨ {len(newly_matching)} features NOW match distribution (after training):")
        for idx, row in newly_matching.iterrows():
            print(f"   {row['feature']:12s}: KS {row['untrained_ks']:.4f} → {row['trained_ks']:.4f}")
    
    if len(lost_matching) > 0:
        print(f"\n💔 {len(lost_matching)} features LOST distribution match (after training):")
        for idx, row in lost_matching.iterrows():
            print(f"   {row['feature']:12s}: KS {row['untrained_ks']:.4f} → {row['trained_ks']:.4f}")
    
    # Overall verdict with more nuance
    print("\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)
    
    overall_ks_improvement = comparison_df['ks_improvement'].mean()
    overall_wass_improvement = comparison_df['wass_improvement'].mean()
    
    print(f"\nOverall Training Impact:")
    print(f"  KS Distance:        {'✓ Improved' if overall_ks_improvement > 0 else '✗ Regressed'} by {abs(overall_ks_improvement):.6f}")
    print(f"  Wasserstein:        {'✓ Improved' if overall_wass_improvement > 0 else '✗ Regressed'} by {abs(overall_wass_improvement):.6f}")
    print(f"  Features improved:  {ks_improved}/{n_features} (KS), {wass_improved}/{n_features} (Wass)")
    
    if overall_ks_improvement > 0.001 and overall_wass_improvement > 0.001:
        print("\n🎉 EXCELLENT: Training improved both KS and Wasserstein metrics!")
        print("   The model learned meaningful translations.")
    elif overall_wass_improvement > 0.001:
        print("\n✅ GOOD: Training improved Wasserstein distance (training objective)!")
        print("   KS may not show improvement, but the model learned distribution matching.")
    elif ks_improved > n_features * 0.6:
        print("\n👍 MODERATE: Training improved majority of features.")
        print("   Some features benefited more than others.")
    else:
        print("\n⚠️  MIXED: Training showed limited overall improvement.")
        print("   Consider: higher Wasserstein weight, more epochs, or architecture changes.")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()


