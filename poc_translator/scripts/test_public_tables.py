#!/usr/bin/env python3
"""
Test Script for Public Table Queries

Quick test to verify the updated make_queries.py approach works
"""

import psycopg2
import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent.parent / "conf" / "config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_queries():
    config = load_config()
    conn = psycopg2.connect(config['db']['mimic_conn'])
    
    print("Testing new public table approach...")
    print("="*60)
    
    # Test MIMIC query
    mimic_test = """
    WITH icu_window AS (
      SELECT 
        i.stay_id as icustay_id, 
        i.subject_id as subject_id, 
        i.hadm_id as hadm_id, 
        i.intime,
        i.intime + INTERVAL '24 hours' as intime_plus_24h
      FROM mimiciv_icu.icustays i
      JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
      WHERE i.intime IS NOT NULL
        AND i.hadm_id IS NOT NULL
        AND EXTRACT(epoch FROM (i.outtime - i.intime))/3600 >= 24
        AND p.anchor_age >= 18
    )
    SELECT COUNT(*) as mimic_public_count FROM icu_window;
    """
    
    # Test eICU query  
    eicu_test = """
    WITH stays AS (
      SELECT 
        p.patientunitstayid
      FROM eicu_crd.patient p
      WHERE p.patientunitstayid IS NOT NULL
        AND p.unitdischargeoffset >= 24 * 60
        AND p.age != '' AND p.age != 'Unknown' AND p.age IS NOT NULL
        AND CASE 
            WHEN p.age = '> 89' THEN 90 
            ELSE CAST(p.age AS INTEGER) 
            END >= 18
    )
    SELECT COUNT(*) as eicu_public_count FROM stays;
    """
    
    try:
        with conn.cursor() as cur:
            print("🔍 Testing MIMIC public table query...")
            cur.execute(mimic_test)
            mimic_count = cur.fetchone()[0]
            print(f"   ✅ MIMIC Public Tables: {mimic_count:,} adult ICU stays (≥24h)")
            
            print("\n🔍 Testing eICU public table query...")
            cur.execute(eicu_test)
            eicu_count = cur.fetchone()[0]
            print(f"   ✅ eICU Public Tables:  {eicu_count:,} adult ICU stays (≥24h)")
            
            print(f"\n📈 COMPARISON TO STRICT COHORTS:")
            print(f"   MIMIC: {mimic_count:,} vs 12,139 strict = {mimic_count/12139:.1f}x increase")
            if eicu_count > 0:
                print(f"   eICU:  {eicu_count:,} vs ~3,500 strict = {eicu_count/3500:.1f}x increase (estimated)")
            
            print(f"\n✅ Public table queries working!")
            print(f"✅ Ready to regenerate SQL files with: python sql/make_queries.py")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    conn.close()

if __name__ == "__main__":
    test_queries()
