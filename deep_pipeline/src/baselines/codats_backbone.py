"""CoDATS 1D CNN backbone for frozen-LSTM domain adaptation.

Same forward() interface as EHRTranslator so the rest of the pipeline
(SchemaResolver, evaluation, etc.) works unchanged.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoDATS1DCNN(nn.Module):
    """1D CNN feature transformer for CoDATS baseline.

    Transforms source input features (B, T, F) -> (B, T, F) via temporal
    convolutions, matching EHRTranslator's interface for drop-in use.

    Uses residual connection so it starts near identity.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        n_conv_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.2,
        temporal_attention_mode: str = "causal",
    ):
        super().__init__()
        self.num_features = num_features
        self.causal = temporal_attention_mode == "causal"
        self.kernel_size = kernel_size

        # Input projection: features -> d_model channels
        # Input channels = num_features (values) + num_features (missingness) = 2*F
        in_channels = num_features * 2
        layers = []
        ch = in_channels
        for i in range(n_conv_layers):
            out_ch = d_model if i < n_conv_layers - 1 else d_model
            layers.append(nn.Conv1d(ch, out_ch, kernel_size, padding=0))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            ch = out_ch
        self.conv_stack = nn.ModuleList(
            [m for m in layers if isinstance(m, (nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.Dropout))]
        )

        # Output projection: d_model -> num_features (delta)
        self.out_proj = nn.Linear(d_model, num_features)

        # Initialize output near zero for residual identity start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        logging.info(
            "[CoDATS1DCNN] num_features=%d, d_model=%d, n_conv=%d, kernel=%d, causal=%s",
            num_features, d_model, n_conv_layers, kernel_size, self.causal,
        )

    def _causal_pad(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Left-only padding for causal convolution."""
        pad = kernel_size - 1
        return F.pad(x, (pad, 0))

    def _symmetric_pad(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Symmetric padding for bidirectional convolution."""
        pad = (kernel_size - 1) // 2
        pad_right = kernel_size - 1 - pad
        return F.pad(x, (pad, pad_right))

    def forward(
        self,
        x_val: torch.Tensor,
        x_miss: torch.Tensor,
        t_abs: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
        return_forecast: bool = False,
    ) -> torch.Tensor:
        """Same signature as EHRTranslator.forward().

        Args:
            x_val: (B, T, F) dynamic feature values.
            x_miss: (B, T, F) missingness indicators.
            t_abs: (B, T) absolute time indices (unused by CNN).
            m_pad: (B, T) padding mask (True = padded).
            x_static: (B, S) static features (unused by CNN).
            return_forecast: ignored (compatibility only).

        Returns:
            (B, T, F) translated features.
        """
        B, T, F = x_val.shape

        # Concatenate values + missingness as input channels
        x = torch.cat([x_val, x_miss], dim=-1)  # (B, T, 2F)
        x = x.transpose(1, 2)  # (B, 2F, T) for Conv1d

        # Apply conv stack with appropriate padding
        for module in self.conv_stack:
            if isinstance(module, nn.Conv1d):
                if self.causal:
                    x = self._causal_pad(x, module.kernel_size[0])
                else:
                    x = self._symmetric_pad(x, module.kernel_size[0])
                x = module(x)
            else:
                x = module(x)

        x = x.transpose(1, 2)  # (B, T, d_model)

        # Project to feature space (delta)
        delta = self.out_proj(x)  # (B, T, F)

        # Residual: x_val + delta (starts near identity)
        x_out = x_val + delta

        # Enforce padding integrity
        x_out = x_out.masked_fill(m_pad.unsqueeze(-1).bool(), 0.0)

        return x_out
