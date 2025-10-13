#!/usr/bin/env python3
"""
Debug why np.ascontiguousarray causes correlation to be 1.0
"""

import numpy as np

# Create two different arrays
a = np.array([-0.11238825, -0.00893997, -1.0434227, -0.00893997, -0.7330779])
b = np.array([-0.10476812, 0.06233264, -1.0588174, -0.02338839, -0.65009147])

print("Original arrays:")
print(f"a: {a}")
print(f"b: {b}")
print(f"Equal? {np.array_equal(a, b)}")
print()

# Test correlation
corr1 = np.corrcoef(a, b)[0, 1]
print(f"Correlation (direct): {corr1:.10f}")
print()

# Convert to contiguous
a_cont = np.ascontiguousarray(a)
b_cont = np.ascontiguousarray(b)

print("After np.ascontiguousarray:")
print(f"a_cont: {a_cont}")
print(f"b_cont: {b_cont}")
print(f"Equal? {np.array_equal(a_cont, b_cont)}")
print(f"Same object? {a_cont is b_cont}")
print(f"Share memory? {np.shares_memory(a_cont, b_cont)}")
print()

corr2 = np.corrcoef(a_cont, b_cont)[0, 1]
print(f"Correlation (contiguous): {corr2:.10f}")
print()

# Try with actual data structure
print("=" * 60)
print("Testing with 2D array slicing (like our actual case)")
print("=" * 60)

# Create a 2D array similar to our clinical features
data = np.random.randn(100, 5)
data_rt = data + np.random.randn(100, 5) * 0.1  # Add some noise

# Extract column 0 (this creates a view)
col_0 = data[:, 0]
col_0_rt = data_rt[:, 0]

print(f"col_0 is view? {col_0.base is not None}")
print(f"col_0_rt is view? {col_0_rt.base is not None}")
print(f"C_CONTIGUOUS: {col_0.flags['C_CONTIGUOUS']}, {col_0_rt.flags['C_CONTIGUOUS']}")
print()

corr_direct = np.corrcoef(col_0, col_0_rt)[0, 1]
print(f"Correlation (direct views): {corr_direct:.10f}")

col_0_cont = np.ascontiguousarray(col_0)
col_0_rt_cont = np.ascontiguousarray(col_0_rt)

print(f"After ascontiguousarray - Equal? {np.array_equal(col_0_cont, col_0_rt_cont)}")
corr_cont = np.corrcoef(col_0_cont, col_0_rt_cont)[0, 1]
print(f"Correlation (contiguous): {corr_cont:.10f}")

