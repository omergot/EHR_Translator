"""Adapter bridging AdaTime data format to our translator pipeline.

Provides:
  - AdaTimeSchemaResolver: Extracts X_val, X_miss, X_static, t_abs, M_pad
    from AdaTime batch tuples, mimicking SchemaResolver.extract()
  - AdaTimeRuntime: Mimics YAIBRuntime interface for the frozen LSTM baseline
  - Simplified RetrievalTranslator configuration for short fixed-length sequences
"""

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .target_model import LSTMClassifier

logger = logging.getLogger(__name__)


class AdaTimeSchemaResolver:
    """Schema resolver for AdaTime datasets.

    Mimics SchemaResolver.extract() but for AdaTime batch format.
    AdaTime batch: (data, labels, pad_mask, static_features)
    where data is (B, T, C) -- already in timesteps-first format.

    Since AdaTime data has no missing indicators and no separate static features,
    this is much simpler than the YAIB SchemaResolver.
    """

    def __init__(self, num_features: int, static_dim: int = 4):
        self.num_features = num_features
        self.static_dim = static_dim
        # For compatibility with the retrieval translator trainer
        self.dynamic_features = [f"ch_{i}" for i in range(num_features)]

        # Minimal SchemaIndices-like attribute
        class _Indices:
            def __init__(self, dynamic):
                self.dynamic = dynamic
        self.indices = _Indices(list(range(num_features)))

    def extract(self, batch: tuple) -> Dict[str, torch.Tensor]:
        """Extract components from an AdaTime batch.

        Args:
            batch: (data, labels, pad_mask, static) where:
                data: (B, T, C) float tensor
                labels: (B, T) long tensor (per-timestep repeated class label)
                pad_mask: (B, T) bool tensor (True = valid, opposite of EHR convention!)
                static: (B, S) float tensor (zeros for AdaTime)

        Returns:
            dict matching SchemaResolver.extract() output format:
                X_val: (B, T, C) dynamic feature values
                X_miss: (B, T, C) missing indicators (all zeros)
                X_static: (B, S) static features
                t_abs: (B, T) absolute time indices
                M_pad: (B, T) padding mask (True = PADDED, matching EHR convention)
                M_label: (B, T) label mask (True = has label)
                y: (B, T) labels
                X_yaib: (B, T, C) raw data (same as X_val for AdaTime)
        """
        data, labels, pad_mask, static = batch[0], batch[1], batch[2], batch[3]
        B, T, C = data.shape

        # AdaTime pad_mask: True = valid timestep
        # Our pipeline M_pad: True = PADDED (opposite convention!)
        # For AdaTime fixed-length data, all timesteps are valid -> M_pad = False
        m_pad = ~pad_mask.bool()  # Invert: True -> valid becomes False -> not padded

        # No missing indicators in AdaTime
        x_miss = torch.zeros_like(data)

        # Absolute time index: simple 0, 1, 2, ..., T-1
        t_abs = torch.arange(T, device=data.device, dtype=data.dtype).unsqueeze(0).expand(B, -1)

        # Label mask: all True for AdaTime (every sample has a label)
        m_label = torch.ones(B, T, dtype=torch.bool, device=data.device)

        return {
            "X_val": data,           # (B, T, C)
            "X_miss": x_miss,        # (B, T, C) all zeros
            "X_static": static,      # (B, S)
            "t_abs": t_abs,          # (B, T)
            "M_pad": m_pad,          # (B, T) all False for fixed-length
            "M_label": m_label,      # (B, T) all True
            "y": labels,             # (B, T)
            "X_yaib": data,          # (B, T, C)
        }

    def rebuild(
        self,
        x_yaib: torch.Tensor,
        x_val_translated: torch.Tensor,
        x_miss_unchanged: torch.Tensor = None,
        x_static_unchanged: torch.Tensor = None,
        m_pad: torch.Tensor = None,
    ) -> torch.Tensor:
        """Rebuild full data tensor from translated values.

        For AdaTime, this is trivial: the translated values ARE the full data.
        """
        return x_val_translated


class AdaTimeRuntime:
    """Runtime adapter for the frozen target LSTM model.

    Mimics the interface of YAIBRuntime that the translator trainer expects:
      - forward(data_tuple) -> model predictions
      - _model attribute for parameter verification
    """

    def __init__(self, frozen_model: LSTMClassifier, device: str = "cuda"):
        self._model = frozen_model.to(device)
        self.device = device

    def forward(self, batch_tuple: tuple) -> torch.Tensor:
        """Run frozen model on a data batch.

        Args:
            batch_tuple: (translated_data, labels, pad_mask) or (data, labels, pad_mask, static)
                where translated_data is (B, T, C)

        Returns:
            logits: (B, num_classes) classification logits
        """
        data = batch_tuple[0]
        if data.device != torch.device(self.device):
            data = data.to(self.device)
        return self._model(data)

    def forward_features(self, data: torch.Tensor) -> torch.Tensor:
        """Extract features from the frozen model.

        Args:
            data: (B, T, C) input tensor

        Returns:
            features: (B, hidden_dim) feature representation
        """
        return self._model.extract_features(data.to(self.device))


def compute_adatime_renorm_params(
    source_loader,
    target_loader,
    schema_resolver: AdaTimeSchemaResolver,
    device: str,
) -> tuple:
    """Compute cross-domain normalization parameters for AdaTime data.

    Returns (scale, offset) such that: x_renorm = x_source * scale + offset.
    """
    def _compute_stats(loader):
        sum_val = None
        sum_sq = None
        count = 0
        with torch.no_grad():
            for batch in loader:
                batch = tuple(b.to(device) for b in batch)
                parts = schema_resolver.extract(batch)
                x = parts["X_val"]  # (B, T, C)
                B, T, C = x.shape
                if sum_val is None:
                    sum_val = x.sum(dim=(0, 1))
                    sum_sq = (x ** 2).sum(dim=(0, 1))
                else:
                    sum_val += x.sum(dim=(0, 1))
                    sum_sq += (x ** 2).sum(dim=(0, 1))
                count += B * T
        mean = sum_val / count
        std = ((sum_sq / count) - mean ** 2).clamp(min=1e-8).sqrt()
        return mean, std

    src_mean, src_std = _compute_stats(source_loader)
    tgt_mean, tgt_std = _compute_stats(target_loader)
    scale = tgt_std / src_std.clamp(min=1e-8)
    offset = tgt_mean - src_mean * scale
    logger.info(
        "AdaTime renorm: scale range [%.4f, %.4f], offset range [%.4f, %.4f]",
        scale.min().item(), scale.max().item(), offset.min().item(), offset.max().item(),
    )
    return scale.to(device), offset.to(device)
