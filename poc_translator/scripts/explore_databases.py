#!/usr/bin/env python3
"""
Database Structure Explorer for eICU and MIMIC-IV
Analyzes database schemas, tables, columns, and data types to provide comprehensive structure summary.
"""

import pandas as pd
import psycopg2
import yaml
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def load_config():
    """Load configuration from config.yml"""
    config_path = Path(__file__).parent.parent / "conf" / "config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_db_connection(conn_string):
    """Get database connection"""
    try:
        conn = psycopg2.connect(conn_string)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def get_database_schemas(conn, db_name):
    """Get all schemas in the database"""
    query = """
    SELECT schema_name 
    FROM information_schema.schemata 
    WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
    ORDER BY schema_name
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        return df['schema_name'].tolist()
    except Exception as e:
        print(f"Error getting schemas for {db_name}: {e}")
        return []

def get_schema_tables(conn, schema_name, db_name):
    """Get all tables in a schema"""
    query = f"""
    SELECT 
        table_name,
        table_type
    FROM information_schema.tables 
    WHERE table_schema = '{schema_name}'
    ORDER BY table_name
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error getting tables for schema {schema_name} in {db_name}: {e}")
        return pd.DataFrame()

def get_table_info(conn, schema_name, table_name, db_name):
    """Get detailed information about a table"""
    
    # Get column information
    column_query = f"""
    SELECT 
        column_name,
        data_type,
        is_nullable,
        column_default,
        character_maximum_length,
        numeric_precision,
        numeric_scale
    FROM information_schema.columns 
    WHERE table_schema = '{schema_name}' 
        AND table_name = '{table_name}'
    ORDER BY ordinal_position
    """
    
    try:
        columns_df = pd.read_sql_query(column_query, conn)
        
        # Get row count
        count_query = f"SELECT COUNT(*) as row_count FROM {schema_name}.{table_name}"
        try:
            count_result = pd.read_sql_query(count_query, conn)
            row_count = count_result['row_count'].iloc[0]
        except:
            row_count = "Access denied or table empty"
        
        # Get sample data (first 3 rows)
        sample_query = f"SELECT * FROM {schema_name}.{table_name} LIMIT 3"
        try:
            sample_df = pd.read_sql_query(sample_query, conn)
        except:
            sample_df = pd.DataFrame()
            
        return {
            'columns': columns_df,
            'row_count': row_count,
            'sample_data': sample_df
        }
        
    except Exception as e:
        print(f"Error getting info for table {schema_name}.{table_name} in {db_name}: {e}")
        return {
            'columns': pd.DataFrame(),
            'row_count': "Error",
            'sample_data': pd.DataFrame()
        }

def analyze_potential_features(table_info, schema_name, table_name):
    """Analyze which columns could be useful features"""
    if table_info['columns'].empty:
        return []
    
    numerical_types = ['integer', 'bigint', 'smallint', 'decimal', 'numeric', 'real', 'double precision', 'money']
    temporal_types = ['timestamp', 'timestamp without time zone', 'timestamp with time zone', 'date', 'time']
    text_types = ['text', 'character varying', 'character', 'varchar', 'char']
    
    feature_candidates = []
    
    for _, col in table_info['columns'].iterrows():
        col_name = col['column_name']
        col_type = col['data_type'].lower()
        
        feature_info = {
            'column': col_name,
            'type': col_type,
            'nullable': col['is_nullable'],
            'category': 'other'
        }
        
        # Categorize potential features
        if any(num_type in col_type for num_type in numerical_types):
            feature_info['category'] = 'numerical'
            
            # Check if it's likely a measurement
            measurement_keywords = ['value', 'result', 'level', 'count', 'rate', 'pressure', 'temp', 'weight', 'height', 'age', 'dose', 'amount']
            if any(keyword in col_name.lower() for keyword in measurement_keywords):
                feature_info['potential'] = 'high'
            else:
                feature_info['potential'] = 'medium'
                
        elif any(temp_type in col_type for temp_type in temporal_types):
            feature_info['category'] = 'temporal'
            feature_info['potential'] = 'medium'
            
        elif any(text_type in col_type for text_type in text_types):
            feature_info['category'] = 'categorical'
            
            # Check if it's likely a useful categorical feature
            if 'name' in col_name.lower() or 'label' in col_name.lower() or 'type' in col_name.lower():
                feature_info['potential'] = 'medium'
            else:
                feature_info['potential'] = 'low'
        
        feature_candidates.append(feature_info)
    
    return feature_candidates

def explore_database(conn, db_name):
    """Comprehensive database exploration"""
    print(f"\n{'='*80}")
    print(f"EXPLORING {db_name.upper()} DATABASE")
    print(f"{'='*80}")
    
    database_summary = {
        'database_name': db_name,
        'exploration_timestamp': datetime.now().isoformat(),
        'schemas': {}
    }
    
    # Get all schemas
    schemas = get_database_schemas(conn, db_name)
    print(f"\nFound {len(schemas)} schemas: {schemas}")
    
    for schema_name in schemas:
        print(f"\n{'-'*60}")
        print(f"SCHEMA: {schema_name}")
        print(f"{'-'*60}")
        
        schema_info = {
            'tables': {},
            'summary': {
                'total_tables': 0,
                'total_columns': 0,
                'high_potential_features': 0,
                'medium_potential_features': 0
            }
        }
        
        # Get tables in schema
        tables_df = get_schema_tables(conn, schema_name, db_name)
        
        if tables_df.empty:
            print(f"No accessible tables found in schema {schema_name}")
            continue
            
        print(f"Found {len(tables_df)} tables in {schema_name}")
        schema_info['summary']['total_tables'] = len(tables_df)
        
        for _, table_row in tables_df.iterrows():
            table_name = table_row['table_name']
            table_type = table_row['table_type']
            
            print(f"\n  Table: {table_name} ({table_type})")
            
            # Get detailed table information
            table_info = get_table_info(conn, schema_name, table_name, db_name)
            
            if not table_info['columns'].empty:
                num_columns = len(table_info['columns'])
                print(f"    Columns: {num_columns}")
                print(f"    Rows: {table_info['row_count']}")
                
                schema_info['summary']['total_columns'] += num_columns
                
                # Analyze potential features
                feature_candidates = analyze_potential_features(table_info, schema_name, table_name)
                
                high_potential = sum(1 for f in feature_candidates if f.get('potential') == 'high')
                medium_potential = sum(1 for f in feature_candidates if f.get('potential') == 'medium')
                
                schema_info['summary']['high_potential_features'] += high_potential
                schema_info['summary']['medium_potential_features'] += medium_potential
                
                print(f"    High potential features: {high_potential}")
                print(f"    Medium potential features: {medium_potential}")
                
                # Store detailed information
                schema_info['tables'][table_name] = {
                    'type': table_type,
                    'row_count': table_info['row_count'],
                    'column_count': num_columns,
                    'columns': table_info['columns'].to_dict('records'),
                    'feature_candidates': feature_candidates,
                    'sample_data_shape': table_info['sample_data'].shape if not table_info['sample_data'].empty else (0, 0)
                }
                
                # Show some interesting columns
                numerical_cols = [f['column'] for f in feature_candidates if f['category'] == 'numerical' and f.get('potential') in ['high', 'medium']]
                if numerical_cols:
                    print(f"    Key numerical columns: {numerical_cols[:5]}" + ('...' if len(numerical_cols) > 5 else ''))
            
            else:
                print(f"    No accessible column information")
                schema_info['tables'][table_name] = {
                    'type': table_type,
                    'row_count': "Access denied",
                    'column_count': 0,
                    'columns': [],
                    'feature_candidates': [],
                    'sample_data_shape': (0, 0)
                }
        
        database_summary['schemas'][schema_name] = schema_info
    
    return database_summary

def generate_summary_report(eicu_summary, mimic_summary, output_dir):
    """Generate comprehensive summary report"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"database_structure_report_{timestamp}.md"
    json_file = output_dir / f"database_structure_data_{timestamp}.json"
    
    # Save raw data as JSON
    combined_data = {
        'eicu': eicu_summary,
        'mimic': mimic_summary
    }
    
    with open(json_file, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)
    
    # Generate markdown report
    with open(report_file, 'w') as f:
        f.write("# Database Structure Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        
        # Overall statistics
        total_schemas = len(eicu_summary['schemas']) + len(mimic_summary['schemas'])
        total_tables = sum(schema['summary']['total_tables'] for schema in eicu_summary['schemas'].values()) + \
                      sum(schema['summary']['total_tables'] for schema in mimic_summary['schemas'].values())
        total_columns = sum(schema['summary']['total_columns'] for schema in eicu_summary['schemas'].values()) + \
                       sum(schema['summary']['total_columns'] for schema in mimic_summary['schemas'].values())
        
        f.write(f"- **Total Schemas Analyzed**: {total_schemas}\n")
        f.write(f"- **Total Tables Found**: {total_tables}\n") 
        f.write(f"- **Total Columns Analyzed**: {total_columns}\n\n")
        
        # Database-specific analysis
        for db_name, summary in [('eICU', eicu_summary), ('MIMIC-IV', mimic_summary)]:
            f.write(f"## {db_name} Database Analysis\n\n")
            
            if not summary['schemas']:
                f.write("No accessible schemas found.\n\n")
                continue
                
            f.write(f"**Schemas**: {list(summary['schemas'].keys())}\n\n")
            
            for schema_name, schema_info in summary['schemas'].items():
                f.write(f"### Schema: `{schema_name}`\n\n")
                f.write(f"- **Tables**: {schema_info['summary']['total_tables']}\n")
                f.write(f"- **Total Columns**: {schema_info['summary']['total_columns']}\n")
                f.write(f"- **High Potential Features**: {schema_info['summary']['high_potential_features']}\n")
                f.write(f"- **Medium Potential Features**: {schema_info['summary']['medium_potential_features']}\n\n")
                
                # Table details
                f.write("#### Tables Overview\n\n")
                f.write("| Table Name | Type | Rows | Columns | High Potential Features | Key Numerical Columns |\n")
                f.write("|------------|------|------|---------|------------------------|------------------------|\n")
                
                for table_name, table_info in schema_info['tables'].items():
                    high_features = sum(1 for f in table_info['feature_candidates'] if f.get('potential') == 'high')
                    numerical_cols = [f['column'] for f in table_info['feature_candidates'] 
                                    if f['category'] == 'numerical' and f.get('potential') in ['high', 'medium']]
                    key_cols = ', '.join(numerical_cols[:3]) + ('...' if len(numerical_cols) > 3 else '')
                    
                    f.write(f"| {table_name} | {table_info['type']} | {table_info['row_count']} | {table_info['column_count']} | {high_features} | {key_cols} |\n")
                
                f.write("\n")
                
                # Feature candidates summary
                f.write("#### Key Feature-Rich Tables\n\n")
                feature_rich_tables = [(name, info) for name, info in schema_info['tables'].items() 
                                     if sum(1 for f in info['feature_candidates'] if f.get('potential') in ['high', 'medium']) > 5]
                
                if feature_rich_tables:
                    for table_name, table_info in feature_rich_tables[:5]:  # Top 5
                        f.write(f"**{table_name}**\n")
                        high_potential = [f['column'] for f in table_info['feature_candidates'] if f.get('potential') == 'high']
                        medium_potential = [f['column'] for f in table_info['feature_candidates'] if f.get('potential') == 'medium']
                        
                        if high_potential:
                            f.write(f"- High potential: {', '.join(high_potential[:10])}\n")
                        if medium_potential:
                            f.write(f"- Medium potential: {', '.join(medium_potential[:10])}\n")
                        f.write("\n")
                else:
                    f.write("No feature-rich tables found in this schema.\n\n")
        
        f.write("## Feature Extraction Recommendations\n\n")
        
        # Generate recommendations based on analysis
        f.write("### eICU Recommended Tables for Feature Extraction:\n")
        eicu_recommendations = []
        for schema_name, schema_info in eicu_summary['schemas'].items():
            for table_name, table_info in schema_info['tables'].items():
                feature_score = sum(1 for f in table_info['feature_candidates'] if f.get('potential') in ['high', 'medium'])
                if feature_score > 3:
                    eicu_recommendations.append((f"{schema_name}.{table_name}", feature_score, table_info['row_count']))
        
        eicu_recommendations.sort(key=lambda x: x[1], reverse=True)
        for table, score, rows in eicu_recommendations[:10]:
            f.write(f"- **{table}** ({score} potential features, {rows} rows)\n")
        
        f.write("\n### MIMIC-IV Recommended Tables for Feature Extraction:\n")
        mimic_recommendations = []
        for schema_name, schema_info in mimic_summary['schemas'].items():
            for table_name, table_info in schema_info['tables'].items():
                feature_score = sum(1 for f in table_info['feature_candidates'] if f.get('potential') in ['high', 'medium'])
                if feature_score > 3:
                    mimic_recommendations.append((f"{schema_name}.{table_name}", feature_score, table_info['row_count']))
        
        mimic_recommendations.sort(key=lambda x: x[1], reverse=True)
        for table, score, rows in mimic_recommendations[:10]:
            f.write(f"- **{table}** ({score} potential features, {rows} rows)\n")
        
        f.write(f"\n---\n*Report generated by explore_databases.py*\n")
        f.write(f"*Raw data available in: {json_file.name}*\n")
    
    return report_file, json_file

def main():
    """Main execution function"""
    print("Starting comprehensive database structure exploration...")
    
    # Load configuration
    config = load_config()
    
    # Create output directory
    output_dir = Path(config['paths']['output_dir']) / "database_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Connect to databases
    print("Connecting to databases...")
    mimic_conn = get_db_connection(config['db']['mimic_conn'])
    eicu_conn = get_db_connection(config['db']['eicu_conn'])
    
    if not mimic_conn:
        print("Failed to connect to MIMIC database")
        return
        
    if not eicu_conn:
        print("Failed to connect to eICU database") 
        return
    
    try:
        # Explore eICU database
        eicu_summary = explore_database(eicu_conn, "eICU")
        
        # Explore MIMIC database  
        mimic_summary = explore_database(mimic_conn, "MIMIC-IV")
        
        # Generate comprehensive report
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'='*80}")
        
        report_file, json_file = generate_summary_report(eicu_summary, mimic_summary, output_dir)
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Summary report: {report_file}")
        print(f"📁 Raw data: {json_file}")
        
    except Exception as e:
        print(f"Error during exploration: {e}")
        raise
    finally:
        if eicu_conn:
            eicu_conn.close()
        if mimic_conn:
            mimic_conn.close()

if __name__ == "__main__":
    main()
