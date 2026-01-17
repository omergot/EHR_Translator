#!/usr/bin/env python3
"""
Unit Correction Script for eICU Cache Data

This script corrects unit mismatches identified in the distribution analysis:
1. C-Reactive Protein: Convert mg/L to mg/dL (divide by 10)
2. RBC: Skipped (issue is on MIMIC side with mixed units)

Author: Domain Translation Project
Date: December 2025
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def load_data(input_path):
    """Load eICU cache data"""
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    return df


def print_feature_stats(df, feature_name, label=""):
    """Print statistics for a specific feature"""
    feature_data = df[df['feature_name'] == feature_name]
    
    if len(feature_data) == 0:
        print(f"  {label}No data found for '{feature_name}'")
        return
    
    # Convert to numeric for statistics
    numeric_vals = pd.to_numeric(feature_data['feature_value'], errors='coerce').dropna()
    
    print(f"  {label}Feature: {feature_name}")
    print(f"    Total rows: {len(feature_data):,}")
    print(f"    Valid numeric values: {len(numeric_vals):,}")
    print(f"    Unique patients: {feature_data['example_id'].nunique():,}")
    
    if len(numeric_vals) > 0:
        print(f"    Value range: [{numeric_vals.min():.4f}, {numeric_vals.max():.4f}]")
        print(f"    Mean ± SD: {numeric_vals.mean():.4f} ± {numeric_vals.std():.4f}")
        print(f"    Median [IQR]: {numeric_vals.median():.4f} [{numeric_vals.quantile(0.75) - numeric_vals.quantile(0.25):.4f}]")
    
    if 'unit' in feature_data.columns:
        units = feature_data['unit'].value_counts()
        print(f"    Units: {dict(units)}")
        print(f"    Sample values (first 5): {numeric_vals.head().tolist()}")


def correct_crp(df, dry_run=False):
    """
    Correct C-Reactive Protein unit mismatch
    Convert from mg/L to mg/dL by dividing by 10
    
    Note: In eICU cache, the feature is named 'CRP' (not 'C-Reactive Protein')
    """
    feature_name = 'CRP'
    
    print(f"\n{'='*80}")
    print(f"CORRECTING: {feature_name}")
    print(f"{'='*80}")
    print("Conversion: mg/L → mg/dL (divide by 10)")
    
    # Print before statistics
    print("\nBEFORE CORRECTION:")
    print_feature_stats(df, feature_name, "  ")
    
    if not dry_run:
        # Apply correction
        mask = df['feature_name'] == feature_name
        n_affected = mask.sum()
        
        if n_affected > 0:
            # Divide feature_value by 10
            df.loc[mask, 'feature_value'] = pd.to_numeric(df.loc[mask, 'feature_value'], errors='coerce') / 10
            
            # Update unit column
            if 'unit' in df.columns:
                df.loc[mask, 'unit'] = 'mg/dL'
            
            print(f"\n✓ Applied correction to {n_affected:,} rows")
            
            # Print after statistics
            print("\nAFTER CORRECTION:")
            print_feature_stats(df, feature_name, "  ")
        else:
            print(f"\n⚠ No rows found for '{feature_name}'")
    else:
        print("\n[DRY RUN] No changes applied")
    
    return df


def validate_data(df_original, df_corrected):
    """Validate that correction maintained data integrity"""
    print(f"\n{'='*80}")
    print("VALIDATION")
    print(f"{'='*80}")
    
    # Check row count
    print(f"Row count preserved: {len(df_original):,} → {len(df_corrected):,} ", end="")
    if len(df_original) == len(df_corrected):
        print("✓")
    else:
        print("✗ WARNING")
    
    # Check column count
    print(f"Column count preserved: {len(df_original.columns)} → {len(df_corrected.columns)} ", end="")
    if len(df_original.columns) == len(df_corrected.columns):
        print("✓")
    else:
        print("✗ WARNING")
    
    # Check feature names
    orig_features = set(df_original['feature_name'].unique())
    corr_features = set(df_corrected['feature_name'].unique())
    print(f"Feature names preserved: {len(orig_features)} → {len(corr_features)} ", end="")
    if orig_features == corr_features:
        print("✓")
    else:
        print("✗ WARNING")
    
    # Check for NaN introduction
    orig_nan = df_original['feature_value'].isna().sum()
    corr_nan = df_corrected['feature_value'].isna().sum()
    print(f"NaN values: {orig_nan:,} → {corr_nan:,} ", end="")
    if corr_nan <= orig_nan:
        print("✓")
    else:
        print(f"✗ WARNING: {corr_nan - orig_nan:,} new NaN values introduced")


def save_data(df, output_path):
    """Save corrected data"""
    print(f"\n{'='*80}")
    print("SAVING CORRECTED DATA")
    print(f"{'='*80}")
    print(f"Output path: {output_path}")
    
    df.to_csv(output_path, index=False)
    
    # Verify saved file
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"✓ File saved successfully")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Rows: {len(df):,}")


def main():
    parser = argparse.ArgumentParser(
        description='Correct unit mismatches in eICU cache data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard usage
  python correct_eicu_units.py
  
  # Dry run (no changes)
  python correct_eicu_units.py --dry-run
  
  # Custom paths
  python correct_eicu_units.py --input /path/to/input.csv --output /path/to/output.csv
        """
    )
    
    parser.add_argument(
        '--input',
        default='/bigdata/omerg/RatchetEHR/tmp/tmp/cache_data_bsi_test_100',
        help='Input eICU cache file path'
    )
    
    parser.add_argument(
        '--output',
        default='/bigdata/omerg/RatchetEHR/tmp/tmp/cache_data_bsi_test_100_corrected',
        help='Output corrected cache file path'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without applying corrections'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("eICU CACHE UNIT CORRECTION SCRIPT")
    print("="*80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Mode:   {'DRY RUN' if args.dry_run else 'APPLY CORRECTIONS'}")
    print()
    
    # Load data
    df = load_data(args.input)
    df_original = df.copy()
    
    # Apply corrections
    print("\n" + "="*80)
    print("CORRECTIONS TO APPLY")
    print("="*80)
    print("1. C-Reactive Protein: mg/L → mg/dL (÷10)")
    print("2. RBC: SKIPPED (issue is on MIMIC side)")
    
    # C-Reactive Protein correction
    df = correct_crp(df, dry_run=args.dry_run)
    
    if not args.dry_run:
        # Validate
        validate_data(df_original, df)
        
        # Save
        save_data(df, args.output)
        
        print("\n" + "="*80)
        print("✓ CORRECTION COMPLETE")
        print("="*80)
        print(f"Corrected file saved to: {args.output}")
        print("\nNext steps:")
        print("1. Update EICU_DATA_PATH in compare_distributions.py to point to corrected file")
        print("2. Re-run distribution analysis: python3 scripts/compare_distributions.py")
    else:
        print("\n" + "="*80)
        print("DRY RUN COMPLETE - NO CHANGES APPLIED")
        print("="*80)
        print("Remove --dry-run flag to apply corrections")


if __name__ == "__main__":
    main()

