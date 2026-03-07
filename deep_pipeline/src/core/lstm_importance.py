"""Extract per-feature importance from a frozen LSTM baseline.

Reads the input-to-hidden weight matrix (weight_ih) from the first LSTM layer,
sums absolute weights per input feature, normalizes to [0, 1], and converts to
logits via inverse sigmoid for initializing FeatureGate.
"""

import logging

import torch
import torch.nn as nn


def extract_lstm_feature_importance(
    model: nn.Module,
    num_dynamic_features: int,
    dynamic_feature_offset: int = 0,
) -> torch.Tensor:
    """Extract per-dynamic-feature importance from LSTM weight_ih.

    Args:
        model: Frozen baseline model containing an LSTM module.
        num_dynamic_features: Number of dynamic features (F) the translator uses.
        dynamic_feature_offset: Starting index of dynamic features in the LSTM
            input vector (accounts for missingness indicators etc. that precede
            the dynamic block in the YAIB input layout).

    Returns:
        (F,) tensor of logits suitable for FeatureGate initialization.
        High logit → high importance → gate opens → less fidelity pressure.
    """
    # Find the first LSTM module
    lstm_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.LSTM):
            lstm_module = module
            logging.info("Found LSTM module '%s' for importance extraction", name)
            break

    if lstm_module is None:
        logging.warning("No LSTM found in baseline model; returning zero logits")
        return torch.zeros(num_dynamic_features)

    # weight_ih_l0: (4*hidden_size, input_size) for the first layer
    w_ih = lstm_module.weight_ih_l0.detach().float()  # (4H, D_in)
    input_size = w_ih.shape[1]

    # Sum absolute weight contributions per input feature
    per_input_importance = w_ih.abs().sum(dim=0)  # (D_in,)

    # Extract the dynamic feature slice
    end_idx = dynamic_feature_offset + num_dynamic_features
    if end_idx > input_size:
        logging.warning(
            "Dynamic feature range [%d:%d] exceeds LSTM input size %d; "
            "clamping and zero-padding",
            dynamic_feature_offset, end_idx, input_size,
        )
        available = per_input_importance[dynamic_feature_offset:input_size]
        dynamic_importance = torch.zeros(num_dynamic_features)
        dynamic_importance[:available.shape[0]] = available
    else:
        dynamic_importance = per_input_importance[dynamic_feature_offset:end_idx]

    # Normalize to [0, 1]
    imp_min = dynamic_importance.min()
    imp_max = dynamic_importance.max()
    if imp_max - imp_min > 1e-8:
        normalized = (dynamic_importance - imp_min) / (imp_max - imp_min)
    else:
        normalized = torch.full_like(dynamic_importance, 0.5)

    # Clamp away from 0/1 for numerical stability of inverse sigmoid
    normalized = normalized.clamp(0.01, 0.99)

    # Inverse sigmoid: logit = log(p / (1-p))
    logits = torch.log(normalized / (1.0 - normalized))

    # Log top features
    top_vals, top_idx = logits.topk(min(10, logits.shape[0]))
    logging.info(
        "[lstm-importance] Top 10 dynamic features by importance: indices=%s logits=%s",
        top_idx.tolist(),
        [f"{v:.3f}" for v in top_vals.tolist()],
    )

    return logits
