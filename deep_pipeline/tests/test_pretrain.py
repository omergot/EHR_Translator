"""Tests for MLM pretraining."""

import torch
import pytest

from src.core.translator import EHRTranslator


def test_set_temporal_mode():
    """Verify set_temporal_mode switches all blocks."""
    translator = EHRTranslator(
        num_features=10, d_model=32, d_time=8, n_layers=2,
        n_heads=4, d_ff=64, static_dim=4,
        temporal_attention_mode="causal",
    )
    assert all(b.use_causal_temporal_attention for b in translator.blocks)

    translator.set_temporal_mode("bidirectional")
    assert all(not b.use_causal_temporal_attention for b in translator.blocks)
    assert translator.temporal_attention_mode == "bidirectional"

    translator.set_temporal_mode("causal")
    assert all(b.use_causal_temporal_attention for b in translator.blocks)
    assert translator.temporal_attention_mode == "causal"


def test_init_and_discard_mlm_head():
    """MLM head init creates params, discard removes them."""
    translator = EHRTranslator(
        num_features=10, d_model=32, d_time=8, n_layers=2,
        n_heads=4, d_ff=64, static_dim=4,
    )
    assert translator.mask_embedding is None
    assert translator.reconstruction_head is None

    translator.init_mlm_head()
    assert translator.mask_embedding is not None
    assert translator.reconstruction_head is not None
    assert translator.mask_embedding.shape == (32,)

    translator.discard_mlm_head()
    assert translator.mask_embedding is None
    assert translator.reconstruction_head is None


def test_forward_mlm_output_shape():
    """forward_mlm produces correct output shape."""
    B, T, F = 4, 20, 10
    translator = EHRTranslator(
        num_features=F, d_model=32, d_time=8, n_layers=2,
        n_heads=4, d_ff=64, static_dim=4,
        temporal_attention_mode="bidirectional",
    )
    translator.init_mlm_head()

    x_val = torch.randn(B, T, F)
    x_miss = torch.zeros(B, T, F)
    t_abs = torch.arange(T).float().unsqueeze(0).expand(B, -1)
    m_pad = torch.zeros(B, T, dtype=torch.bool)
    x_static = torch.randn(B, 4)
    mlm_mask = torch.zeros(B, T, dtype=torch.bool)
    mlm_mask[:, 3] = True
    mlm_mask[:, 7] = True

    out = translator.forward_mlm(x_val, x_miss, t_abs, m_pad, x_static, mlm_mask)
    assert out.shape == (B, T, F)


def test_forward_mlm_gradient_flows():
    """Gradient flows through forward_mlm to backbone parameters."""
    B, T, F = 2, 10, 5
    translator = EHRTranslator(
        num_features=F, d_model=32, d_time=8, n_layers=2,
        n_heads=4, d_ff=64, static_dim=4,
        temporal_attention_mode="bidirectional",
    )
    translator.init_mlm_head()

    x_val = torch.randn(B, T, F)
    x_miss = torch.zeros(B, T, F)
    t_abs = torch.arange(T).float().unsqueeze(0).expand(B, -1)
    m_pad = torch.zeros(B, T, dtype=torch.bool)
    x_static = torch.randn(B, 4)
    mlm_mask = torch.zeros(B, T, dtype=torch.bool)
    mlm_mask[:, 2] = True

    out = translator.forward_mlm(x_val, x_miss, t_abs, m_pad, x_static, mlm_mask)
    loss = out[mlm_mask].pow(2).mean()
    loss.backward()

    # Check gradient on backbone params
    assert translator.triplet_proj.weight.grad is not None
    assert translator.reconstruction_head.weight.grad is not None


def test_weight_transfer_bidirectional_to_causal():
    """Weights are identical after switching mode — only attention mask changes."""
    F = 5
    translator = EHRTranslator(
        num_features=F, d_model=32, d_time=8, n_layers=2,
        n_heads=4, d_ff=64, static_dim=4,
        temporal_attention_mode="bidirectional",
    )
    weights_before = {k: v.clone() for k, v in translator.state_dict().items()}

    translator.set_temporal_mode("causal")
    weights_after = translator.state_dict()

    for key in weights_before:
        assert torch.equal(weights_before[key], weights_after[key]), f"Weight changed after mode switch: {key}"
