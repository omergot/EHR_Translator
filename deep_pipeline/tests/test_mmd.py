"""Tests for multi-kernel MMD utility."""

import torch
import pytest

from src.core.mmd import multi_kernel_mmd


def test_same_distribution_mmd_near_zero():
    """MMD between two samples from the same distribution should be ~0."""
    torch.manual_seed(42)
    x = torch.randn(500, 10)
    y = torch.randn(500, 10)
    mmd = multi_kernel_mmd(x, y)
    assert mmd.item() < 0.1, f"MMD between same distributions too large: {mmd.item()}"


def test_different_distribution_mmd_positive():
    """MMD between samples from different distributions should be clearly positive."""
    torch.manual_seed(42)
    x = torch.randn(500, 10)
    y = torch.randn(500, 10) + 3.0  # shifted mean
    mmd = multi_kernel_mmd(x, y)
    assert mmd.item() > 0.5, f"MMD between different distributions too small: {mmd.item()}"


def test_gradient_flows_through_x():
    """Gradient should flow through x (source) but not y (target)."""
    torch.manual_seed(42)
    x = torch.randn(100, 5, requires_grad=True)
    y = torch.randn(100, 5)
    mmd = multi_kernel_mmd(x, y)
    mmd.backward()
    assert x.grad is not None, "No gradient on x"
    assert x.grad.abs().sum() > 0, "Gradient on x is all zeros"


def test_subsampling():
    """Should handle large inputs via subsampling without error."""
    torch.manual_seed(42)
    x = torch.randn(10000, 10)
    y = torch.randn(8000, 10)
    mmd = multi_kernel_mmd(x, y, max_samples=512)
    assert torch.isfinite(mmd), "MMD is not finite after subsampling"


def test_custom_bandwidths():
    """Custom bandwidths should work correctly."""
    torch.manual_seed(42)
    x = torch.randn(200, 5)
    y = torch.randn(200, 5) + 2.0
    mmd = multi_kernel_mmd(x, y, bandwidths=[0.5, 1.0, 2.0])
    assert mmd.item() > 0.0, "MMD with custom bandwidths should be positive for shifted data"


def test_small_sample():
    """Should return 0 gracefully for very small samples."""
    x = torch.randn(1, 5)
    y = torch.randn(1, 5)
    mmd = multi_kernel_mmd(x, y)
    assert mmd.item() == 0.0, "MMD for single-sample inputs should be 0"


def test_dimension_mismatch_raises():
    """Should raise on mismatched feature dimensions."""
    x = torch.randn(10, 5)
    y = torch.randn(10, 3)
    with pytest.raises(ValueError, match="Feature dims must match"):
        multi_kernel_mmd(x, y)


def test_3d_input_raises():
    """Should raise on 3-D input (not pre-flattened)."""
    x = torch.randn(10, 5, 3)
    y = torch.randn(10, 5, 3)
    with pytest.raises(ValueError, match="Expected 2-D tensors"):
        multi_kernel_mmd(x, y)
