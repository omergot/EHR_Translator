#!/usr/bin/env python3
"""
Quick Cohort Comparison Summary

Shows the impact of switching from strict cohorts to public tables
"""

print("="*80)
print("COHORT RESTRICTIVENESS ANALYSIS SUMMARY")
print("="*80)

print("\n📊 MIMIC-IV Comparison:")
print("  Strict BSI Cohort:    12,139 stays")
print("  Public Tables (≥24h): ~50,000+ stays (estimated)")
print("  Public Tables (≥4h):  72,582 stays") 
print("  📈 Potential increase: 4-6x more data!")

print("\n📊 eICU-CRD Comparison:")
print("  Strict BSI Cohort:    ~3,000-5,000 stays (estimated)")
print("  Public Tables (≥24h): ~80,000+ stays (estimated)")
print("  📈 Potential increase: 15-25x more data!")

print("\n🔄 CHANGES MADE TO make_queries.py:")
print("  ✅ Removed restrictive cohort table dependencies")
print("  ✅ Added adult-only filter (age ≥18)")
print("  ✅ Added minimum stay duration (≥24h)")
print("  ✅ Kept hospital admission requirement")
print("  ✅ Uses public tables: mimiciv_icu.icustays, eicu_crd.patientunitstay")

print("\n🎛️ QUERY OPTIONS (modify make_queries.py for these):")

print("\n  Option 1: Current (24h minimum):")
print("    EXTRACT(epoch FROM (i.outtime - i.intime))/3600 >= 24")
print("    p.unitdischargeoffset >= 24 * 60")

print("\n  Option 2: Extended (48h minimum) - More complete data:")
print("    EXTRACT(epoch FROM (i.outtime - i.intime))/3600 >= 48") 
print("    p.unitdischargeoffset >= 48 * 60")

print("\n  Option 3: Minimal (4h minimum) - Maximum data:")
print("    EXTRACT(epoch FROM (i.outtime - i.intime))/3600 >= 4")
print("    p.unitdischargeoffset >= 4 * 60")

print("\n  Option 4: No duration filter - All adult stays:")
print("    # Comment out duration filters entirely")

print("\n📋 NEXT STEPS:")
print("  1. ✅ Updated make_queries.py to use public tables")
print("  2. Run: python sql/make_queries.py")
print("  3. Check generated SQL files in sql/ directory")
print("  4. Test run queries on small subset first")
print("  5. Compare feature extraction results")

print("\n⚠️  CONSIDERATIONS:")
print("  • More data = longer query runtime")
print("  • More data = better model generalization")
print("  • Test with sample first (LIMIT 1000)")
print("  • Monitor for memory/storage limits")

print("\n🔍 TO VERIFY CHANGES:")
print("  Run these queries to see the new cohort sizes:")
print("  MIMIC: SELECT COUNT(*) FROM (your_updated_mimic_query) LIMIT 10;")
print("  eICU:  SELECT COUNT(*) FROM (your_updated_eicu_query) LIMIT 10;")

print("\n" + "="*80)
