"""Per-feature learned importance weights.

Shared module usable across delta, SL, and retrieval translators to weight
per-feature losses (fidelity, reconstruction, distance).
"""

import torch
import torch.nn as nn


class FeatureGate(nn.Module):
    """Learnable sigmoid gate producing per-feature weights in [0, 1]."""

    def __init__(self, num_features: int, init_logits: torch.Tensor | None = None):
        super().__init__()
        if init_logits is not None:
            assert init_logits.shape == (num_features,), (
                f"init_logits shape {init_logits.shape} != ({num_features},)"
            )
            self.logits = nn.Parameter(init_logits.clone())
        else:
            self.logits = nn.Parameter(torch.zeros(num_features))

    def forward(self) -> torch.Tensor:
        """Return (F,) tensor of importance weights in [0, 1]."""
        return torch.sigmoid(self.logits)
