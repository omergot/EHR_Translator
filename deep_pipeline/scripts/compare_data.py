#!/usr/bin/env python3
"""
Script to compare:
1. Two recipe directories (preprocessing recipe pickle files)
2. Two parquet files (dyn.parquet, sta.parquet, or outc.parquet)

Usage:
    # Compare recipes
    python compare_data.py --compare-recipes \
        cohort_data/sepsis/miiv/preproc/s_2222_r_0_f_0_t_None_d_False_recipe \
        cohort_data/sepsis/eicu/preproc/s_2222_r_0_f_0_t_None_d_False_recipe

    # Compare parquet files
    python compare_data.py --compare-parquet \
        cohort_data/sepsis/eicu/dyn.parquet \
        cohort_data/sepsis/miiv/dyn.parquet

    # Compare static files
    python compare_data.py --compare-parquet \
        cohort_data/sepsis/eicu/sta.parquet \
        cohort_data/sepsis/miiv/sta.parquet
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import sys

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available")

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("Warning: polars not available")

if not HAS_PANDAS and not HAS_POLARS:
    raise ImportError("Neither pandas nor polars available. Please install one of them.")


def load_recipe(recipe_path: Path):
    """Load a recipe pickle file."""
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")
    
    with open(recipe_path, "rb") as f:
        recipe = pickle.load(f)
    return recipe


def compare_recipes(recipe1_path: Path, recipe2_path: Path):
    """Compare two recipe pickle files."""
    print("=" * 80)
    print("COMPARING RECIPES")
    print("=" * 80)
    print(f"\nRecipe 1: {recipe1_path}")
    print(f"Recipe 2: {recipe2_path}\n")
    
    # Load recipes
    print("Loading recipes...")
    recipe1 = load_recipe(recipe1_path)
    recipe2 = load_recipe(recipe2_path)
    print("✓ Recipes loaded successfully\n")
    
    # Compare basic attributes
    print("-" * 80)
    print("BASIC ATTRIBUTES")
    print("-" * 80)
    
    attrs1 = set(dir(recipe1))
    attrs2 = set(dir(recipe2))
    
    common_attrs = attrs1 & attrs2
    only_in_1 = attrs1 - attrs2
    only_in_2 = attrs2 - attrs1
    
    print(f"Common attributes: {len(common_attrs)}")
    if only_in_1:
        print(f"Attributes only in recipe 1: {only_in_1}")
    if only_in_2:
        print(f"Attributes only in recipe 2: {only_in_2}")
    
    # Compare steps
    print("\n" + "-" * 80)
    print("RECIPE STEPS")
    print("-" * 80)
    
    steps1 = getattr(recipe1, 'steps', None)
    steps2 = getattr(recipe2, 'steps', None)
    
    if steps1 is not None and steps2 is not None:
        print(f"Recipe 1 has {len(steps1)} steps")
        print(f"Recipe 2 has {len(steps2)} steps")
        
        if len(steps1) != len(steps2):
            print("⚠ WARNING: Different number of steps!")
        else:
            print("✓ Same number of steps")
        
        print("\nStep types:")
        for i, (s1, s2) in enumerate(zip(steps1, steps2)):
            type1 = type(s1).__name__
            type2 = type(s2).__name__
            match = "✓" if type1 == type2 else "✗"
            print(f"  Step {i}: {match} {type1} vs {type2}")
    
    # Compare data if available
    print("\n" + "-" * 80)
    print("RECIPE DATA")
    print("-" * 80)
    
    data1 = getattr(recipe1, 'data', None)
    data2 = getattr(recipe2, 'data', None)
    
    if data1 is not None and data2 is not None:
        print("Both recipes have data attribute")
        
        # Try to compare as DataFrames
        if HAS_PANDAS:
            if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
                print(f"Recipe 1 data shape: {data1.shape}")
                print(f"Recipe 2 data shape: {data2.shape}")
                print(f"Recipe 1 columns: {list(data1.columns)[:10]}... (showing first 10)")
                print(f"Recipe 2 columns: {list(data2.columns)[:10]}... (showing first 10)")
                
                cols1 = set(data1.columns)
                cols2 = set(data2.columns)
                common_cols = cols1 & cols2
                only_in_1_cols = cols1 - cols2
                only_in_2_cols = cols2 - cols1
                
                print(f"\nCommon columns: {len(common_cols)}")
                if only_in_1_cols:
                    print(f"Columns only in recipe 1: {list(only_in_1_cols)[:10]}...")
                if only_in_2_cols:
                    print(f"Columns only in recipe 2: {list(only_in_2_cols)[:10]}...")
        elif HAS_POLARS:
            if isinstance(data1, pl.DataFrame) and isinstance(data2, pl.DataFrame):
                print(f"Recipe 1 data shape: {data1.shape}")
                print(f"Recipe 2 data shape: {data2.shape}")
                print(f"Recipe 1 columns: {list(data1.columns)[:10]}... (showing first 10)")
                print(f"Recipe 2 columns: {list(data2.columns)[:10]}... (showing first 10)")
                
                cols1 = set(data1.columns)
                cols2 = set(data2.columns)
                common_cols = cols1 & cols2
                only_in_1_cols = cols1 - cols2
                only_in_2_cols = cols2 - cols1
                
                print(f"\nCommon columns: {len(common_cols)}")
                if only_in_1_cols:
                    print(f"Columns only in recipe 1: {list(only_in_1_cols)[:10]}...")
                if only_in_2_cols:
                    print(f"Columns only in recipe 2: {list(only_in_2_cols)[:10]}...")
    else:
        if data1 is None:
            print("Recipe 1 does not have data attribute")
        if data2 is None:
            print("Recipe 2 does not have data attribute")
    
    # Compare normalization statistics
    print("\n" + "-" * 80)
    print("NORMALIZATION STATISTICS")
    print("-" * 80)
    
    stats1 = extract_normalization_stats(recipe1)
    stats2 = extract_normalization_stats(recipe2)
    
    if stats1 and stats2:
        print("Both recipes have normalization statistics")
        
        if stats1['columns'] and stats2['columns']:
            cols1 = set(stats1['columns'])
            cols2 = set(stats2['columns'])
            common_cols = cols1 & cols2
            
            print(f"Common normalized columns: {len(common_cols)}")
            
            if common_cols:
                print("\nComparing statistics for common columns (first 5):")
                for col in list(common_cols)[:5]:
                    idx1 = stats1['columns'].index(col)
                    idx2 = stats2['columns'].index(col)
                    
                    mean1 = stats1['mean'][idx1]
                    mean2 = stats2['mean'][idx2]
                    std1 = stats1['std'][idx1]
                    std2 = stats2['std'][idx2]
                    
                    mean_diff = abs(mean1 - mean2)
                    std_diff = abs(std1 - std2)
                    
                    print(f"  {col}:")
                    print(f"    Mean: {mean1:.4f} vs {mean2:.4f} (diff: {mean_diff:.4f})")
                    print(f"    Std:  {std1:.4f} vs {std2:.4f} (diff: {std_diff:.4f})")
    else:
        if not stats1:
            print("Recipe 1 does not have normalization statistics")
        if not stats2:
            print("Recipe 2 does not have normalization statistics")
    
    print("\n" + "=" * 80)


def extract_normalization_stats(recipe) -> Optional[Dict[str, Any]]:
    """Extract normalization statistics from a recipe."""
    if hasattr(recipe, 'steps') and recipe.steps:
        for step in recipe.steps:
            if hasattr(step, '__class__') and 'StepScale' in step.__class__.__name__:
                if hasattr(step, 'sklearn_transformer'):
                    scaler = step.sklearn_transformer
                    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                        columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
                        return {
                            'mean': scaler.mean_,
                            'std': scaler.scale_,
                            'columns': list(columns) if columns is not None else None
                        }
    return None


def load_parquet(file_path: Path):
    """Load a parquet file using pandas or polars."""
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    if HAS_POLARS:
        return pl.read_parquet(file_path)
    elif HAS_PANDAS:
        return pd.read_parquet(file_path)
    else:
        raise ImportError("Neither polars nor pandas available")


def compare_parquet_files(file1_path: Path, file2_path: Path):
    """Compare two parquet files."""
    print("=" * 80)
    print("COMPARING PARQUET FILES")
    print("=" * 80)
    print(f"\nFile 1: {file1_path}")
    print(f"File 2: {file2_path}\n")
    
    # Load files
    print("Loading parquet files...")
    df1 = load_parquet(file1_path)
    df2 = load_parquet(file2_path)
    print("✓ Files loaded successfully\n")
    
    # Basic information
    print("-" * 80)
    print("BASIC INFORMATION")
    print("-" * 80)
    
    if HAS_POLARS:
        print(f"File 1 shape: {df1.shape} (rows: {df1.height}, cols: {df1.width})")
        print(f"File 2 shape: {df2.shape} (rows: {df2.height}, cols: {df2.width})")
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
    else:  # pandas
        print(f"File 1 shape: {df1.shape} (rows: {len(df1)}, cols: {len(df1.columns)})")
        print(f"File 2 shape: {df2.shape} (rows: {len(df2)}, cols: {len(df2.columns)})")
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
    
    # Column comparison
    print("\n" + "-" * 80)
    print("COLUMN COMPARISON")
    print("-" * 80)
    
    common_cols = cols1 & cols2
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    print(f"Common columns: {len(common_cols)}")
    print(f"Columns only in file 1: {len(only_in_1)}")
    print(f"Columns only in file 2: {len(only_in_2)}")
    
    if only_in_1:
        print(f"\nColumns only in file 1: {list(only_in_1)[:20]}...")
    if only_in_2:
        print(f"Columns only in file 2: {list(only_in_2)[:20]}...")
    
    # Data type comparison for common columns
    if common_cols:
        print("\n" + "-" * 80)
        print("DATA TYPE COMPARISON (Common Columns)")
        print("-" * 80)
        
        dtype_diffs = []
        for col in sorted(common_cols)[:20]:  # Show first 20
            if HAS_POLARS:
                dtype1 = str(df1[col].dtype)
                dtype2 = str(df2[col].dtype)
            else:
                dtype1 = str(df1[col].dtype)
                dtype2 = str(df2[col].dtype)
            
            if dtype1 != dtype2:
                dtype_diffs.append((col, dtype1, dtype2))
        
        if dtype_diffs:
            print("⚠ WARNING: Data type differences found:")
            for col, dt1, dt2 in dtype_diffs[:10]:
                print(f"  {col}: {dt1} vs {dt2}")
        else:
            print("✓ All common columns have matching data types")
    
    # Statistical comparison for numeric columns
    if common_cols:
        print("\n" + "-" * 80)
        print("STATISTICAL COMPARISON (Common Numeric Columns)")
        print("-" * 80)
        
        # Get numeric columns
        if HAS_POLARS:
            numeric_cols = [col for col in common_cols 
                          if df1[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]
        else:
            numeric_cols = [col for col in common_cols 
                          if df1[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        if numeric_cols:
            print(f"Found {len(numeric_cols)} numeric columns in common")
            print("\nComparing statistics (first 10 numeric columns):")
            
            for col in numeric_cols[:10]:
                if HAS_POLARS:
                    stats1 = df1[col].describe()
                    stats2 = df2[col].describe()
                    
                    mean1 = stats1.filter(pl.col("statistic") == "mean")["value"][0]
                    mean2 = stats2.filter(pl.col("statistic") == "mean")["value"][0]
                    std1 = stats1.filter(pl.col("statistic") == "std")["value"][0]
                    std2 = stats2.filter(pl.col("statistic") == "std")["value"][0]
                    min1 = stats1.filter(pl.col("statistic") == "min")["value"][0]
                    min2 = stats2.filter(pl.col("statistic") == "min")["value"][0]
                    max1 = stats1.filter(pl.col("statistic") == "max")["value"][0]
                    max2 = stats2.filter(pl.col("statistic") == "max")["value"][0]
                else:
                    mean1 = df1[col].mean()
                    mean2 = df2[col].mean()
                    std1 = df1[col].std()
                    std2 = df2[col].std()
                    min1 = df1[col].min()
                    min2 = df2[col].min()
                    max1 = df1[col].max()
                    max2 = df2[col].max()
                
                print(f"\n  {col}:")
                print(f"    Mean: {mean1:.4f} vs {mean2:.4f} (diff: {abs(mean1 - mean2):.4f})")
                print(f"    Std:  {std1:.4f} vs {std2:.4f} (diff: {abs(std1 - std2):.4f})")
                print(f"    Min:  {min1:.4f} vs {min2:.4f}")
                print(f"    Max:  {max1:.4f} vs {max2:.4f}")
        else:
            print("No numeric columns found in common")
    
    # Check for stay_id if present
    if 'stay_id' in common_cols:
        print("\n" + "-" * 80)
        print("STAY ID COMPARISON")
        print("-" * 80)
        
        if HAS_POLARS:
            stays1 = set(df1['stay_id'].unique().to_list())
            stays2 = set(df2['stay_id'].unique().to_list())
        else:
            stays1 = set(df1['stay_id'].unique())
            stays2 = set(df2['stay_id'].unique())
        
        common_stays = stays1 & stays2
        only_in_1_stays = stays1 - stays2
        only_in_2_stays = stays2 - stays1
        
        print(f"Unique stays in file 1: {len(stays1)}")
        print(f"Unique stays in file 2: {len(stays2)}")
        print(f"Common stays: {len(common_stays)}")
        print(f"Stays only in file 1: {len(only_in_1_stays)}")
        print(f"Stays only in file 2: {len(only_in_2_stays)}")
    
    print("\n" + "=" * 80)


def explain_outc_parquet():
    """Explain what the outc.parquet file is."""
    print("=" * 80)
    print("EXPLANATION: outc.parquet FILE")
    print("=" * 80)
    print("""
The outc.parquet (outcome) file contains the target labels/outcomes for each patient stay.

Structure:
  - stay_id: Unique identifier for each ICU stay
  - label: The target variable (e.g., binary classification label: 0 or 1)
           For mortality prediction: 1 = death, 0 = survival
           For other tasks: may contain different label values

Purpose:
  - Used during training to provide ground truth labels
  - Used during evaluation to compute metrics (accuracy, AUC, etc.)
  - Links patient stays to their outcomes

Relationship to other files:
  - dyn.parquet: Contains time-series features (vitals, lab values) over time
                 Each row corresponds to a time point for a stay_id
  - sta.parquet: Contains static features (age, sex, height, weight)
                 Each row corresponds to a stay_id (one row per stay)
  - outc.parquet: Contains outcome labels
                 Each row corresponds to a stay_id (one row per stay, or multiple rows if multiple outcomes per stay)

Note: The outcome file is typically used to:
  1. Extract unique stay IDs for data splitting
  2. Provide labels for supervised learning
  3. Stratify data splits (e.g., stratified k-fold cross-validation)
    """)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare recipes or parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two recipe files
  python compare_data.py --compare-recipes \\
      cohort_data/sepsis/miiv/preproc/s_2222_r_0_f_0_t_None_d_False_recipe \\
      cohort_data/sepsis/eicu/preproc/s_2222_r_0_f_0_t_None_d_False_recipe

  # Compare two dynamic parquet files
  python compare_data.py --compare-parquet \\
      cohort_data/sepsis/eicu/dyn.parquet \\
      cohort_data/sepsis/miiv/dyn.parquet

  # Compare two static parquet files
  python compare_data.py --compare-parquet \\
      cohort_data/sepsis/eicu/sta.parquet \\
      cohort_data/sepsis/miiv/sta.parquet

  # Explain what outc.parquet is
  python compare_data.py --explain-outc
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--compare-recipes',
        nargs=2,
        metavar=('RECIPE1', 'RECIPE2'),
        help='Compare two recipe pickle files'
    )
    group.add_argument(
        '--compare-parquet',
        nargs=2,
        metavar=('FILE1', 'FILE2'),
        help='Compare two parquet files'
    )
    group.add_argument(
        '--explain-outc',
        action='store_true',
        help='Explain what the outc.parquet file is'
    )
    
    args = parser.parse_args()
    
    if args.explain_outc:
        explain_outc_parquet()
    elif args.compare_recipes:
        recipe1_path = Path(args.compare_recipes[0])
        recipe2_path = Path(args.compare_recipes[1])
        compare_recipes(recipe1_path, recipe2_path)
    elif args.compare_parquet:
        file1_path = Path(args.compare_parquet[0])
        file2_path = Path(args.compare_parquet[1])
        compare_parquet_files(file1_path, file2_path)


if __name__ == "__main__":
    main()






