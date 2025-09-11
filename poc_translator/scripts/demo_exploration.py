#!/usr/bin/env python3
"""
Demo script showing how to use database exploration results
This demonstrates how to load and analyze the JSON output from explore_databases.py
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_latest_exploration_data(analysis_dir):
    """Load the most recent exploration data"""
    json_files = list(Path(analysis_dir).glob("database_structure_data_*.json"))
    
    if not json_files:
        print(f"No exploration data found in {analysis_dir}")
        print("Run explore_databases.py first!")
        return None
    
    # Get the most recent file
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading data from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def analyze_exploration_data(data):
    """Analyze the exploration data and show useful insights"""
    
    print("\n" + "="*80)
    print("DATABASE EXPLORATION ANALYSIS")
    print("="*80)
    
    for db_name in ['eicu', 'mimic']:
        if db_name not in data:
            continue
            
        db_data = data[db_name]
        print(f"\n{db_name.upper()} DATABASE INSIGHTS:")
        print("-" * 40)
        
        # Collect all tables with their feature potential
        all_tables = []
        for schema_name, schema_info in db_data['schemas'].items():
            for table_name, table_info in schema_info['tables'].items():
                high_features = sum(1 for f in table_info['feature_candidates'] if f.get('potential') == 'high')
                medium_features = sum(1 for f in table_info['feature_candidates'] if f.get('potential') == 'medium')
                
                all_tables.append({
                    'full_name': f"{schema_name}.{table_name}",
                    'schema': schema_name,
                    'table': table_name,
                    'rows': table_info['row_count'],
                    'columns': table_info['column_count'],
                    'high_potential': high_features,
                    'medium_potential': medium_features,
                    'total_potential': high_features + medium_features
                })
        
        # Sort by feature potential
        all_tables.sort(key=lambda x: x['total_potential'], reverse=True)
        
        # Show top 10 tables for feature extraction
        print("\nTOP 10 TABLES FOR FEATURE EXTRACTION:")
        print("Rank | Table | Rows | Cols | High | Med | Total Features")
        print("-----|-------|------|------|------|-----|---------------")
        
        for i, table in enumerate(all_tables[:10], 1):
            print(f"{i:4d} | {table['full_name']:<25} | {str(table['rows']):<8} | {table['columns']:4d} | {table['high_potential']:4d} | {table['medium_potential']:3d} | {table['total_potential']:6d}")
        
        # Show feature distribution
        print(f"\nFEATURE DISTRIBUTION:")
        total_high = sum(t['high_potential'] for t in all_tables)
        total_medium = sum(t['medium_potential'] for t in all_tables)
        total_tables = len(all_tables)
        
        print(f"- Total tables analyzed: {total_tables}")
        print(f"- High potential features: {total_high}")
        print(f"- Medium potential features: {total_medium}")
        print(f"- Tables with >5 potential features: {sum(1 for t in all_tables if t['total_potential'] > 5)}")
        
        # Show key schemas
        print(f"\nSCHEMA BREAKDOWN:")
        schema_stats = {}
        for table in all_tables:
            schema = table['schema']
            if schema not in schema_stats:
                schema_stats[schema] = {'tables': 0, 'features': 0}
            schema_stats[schema]['tables'] += 1
            schema_stats[schema]['features'] += table['total_potential']
        
        for schema, stats in sorted(schema_stats.items(), key=lambda x: x[1]['features'], reverse=True):
            print(f"- {schema}: {stats['tables']} tables, {stats['features']} potential features")

def generate_feature_extraction_suggestions(data):
    """Generate specific suggestions for feature extraction"""
    
    print(f"\n{'='*80}")
    print("FEATURE EXTRACTION SUGGESTIONS")
    print("="*80)
    
    suggestions = {
        'eicu': [],
        'mimic': []
    }
    
    for db_name in ['eicu', 'mimic']:
        if db_name not in data:
            continue
            
        db_data = data[db_name]
        
        # Find tables with high-value numerical features
        for schema_name, schema_info in db_data['schemas'].items():
            for table_name, table_info in schema_info['tables'].items():
                
                # Get high potential numerical columns
                numerical_features = [
                    f['column'] for f in table_info['feature_candidates'] 
                    if f['category'] == 'numerical' and f.get('potential') == 'high'
                ]
                
                if len(numerical_features) >= 3:  # At least 3 high-potential numerical features
                    suggestions[db_name].append({
                        'table': f"{schema_name}.{table_name}",
                        'features': numerical_features,
                        'rows': table_info['row_count']
                    })
    
    for db_name, db_suggestions in suggestions.items():
        if not db_suggestions:
            continue
            
        print(f"\n{db_name.upper()} SPECIFIC FEATURE EXTRACTION TARGETS:")
        print("-" * 50)
        
        # Sort by number of features
        db_suggestions.sort(key=lambda x: len(x['features']), reverse=True)
        
        for suggestion in db_suggestions[:5]:  # Top 5
            print(f"\n📊 **{suggestion['table']}** ({suggestion['rows']} rows)")
            print(f"   Key features: {', '.join(suggestion['features'][:8])}")
            if len(suggestion['features']) > 8:
                print(f"   + {len(suggestion['features']) - 8} more features...")

def main():
    """Main demo function"""
    print("Database Exploration Analysis Demo")
    print("This script analyzes the output from explore_databases.py")
    
    # Look for analysis data
    analysis_dir = Path(__file__).parent.parent / "database_analysis"
    
    if not analysis_dir.exists():
        print(f"\nError: Analysis directory not found: {analysis_dir}")
        print("Please run explore_databases.py first to generate the analysis data.")
        return
    
    # Load the latest exploration data
    data = load_latest_exploration_data(analysis_dir)
    
    if not data:
        return
    
    # Analyze the data
    analyze_exploration_data(data)
    
    # Generate suggestions
    generate_feature_extraction_suggestions(data)
    
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review the tables and features identified above")
    print("2. Run extract_all_features.py to get the actual data")
    print("3. Use the table recommendations to focus your feature engineering")
    print("4. Consider the high-potential features for your specific use case")

if __name__ == "__main__":
    main()
