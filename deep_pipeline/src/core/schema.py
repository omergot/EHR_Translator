from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch


@dataclass(frozen=True)
class SchemaIndices:
    dynamic: List[int]
    missing: List[int]
    static: List[int]


class SchemaResolver:
    def __init__(
        self,
        feature_names: Iterable[str],
        dynamic_features: Iterable[str],
        static_features: Iterable[str],
        allow_missing_static: bool = False,
        missing_prefix: str = "MissingIndicator_",
        group_col: Optional[str] = None,
    ) -> None:
        feature_names = list(feature_names)
        if group_col and group_col in feature_names:
            feature_names = [c for c in feature_names if c != group_col]
        self.feature_names = feature_names
        self.dynamic_features = list(dynamic_features)
        self.static_features = list(static_features)
        self.allow_missing_static = allow_missing_static
        self.missing_prefix = missing_prefix
        self.indices = SchemaIndices(
            dynamic=self._resolve_indices(self.dynamic_features),
            missing=self._resolve_indices(
                [f"{missing_prefix}{name}" for name in self.dynamic_features]
            ),
            static=self._resolve_indices(self.static_features, allow_missing=allow_missing_static),
        )
        missing_dynamic = [name for name, idx in zip(self.dynamic_features, self.indices.dynamic) if idx is None]
        missing_missing = [
            name
            for name, idx in zip(
                [f"{missing_prefix}{name}" for name in self.dynamic_features],
                self.indices.missing,
            )
            if idx is None
        ]
        missing_static = [name for name, idx in zip(self.static_features, self.indices.static) if idx is None]
        if missing_dynamic:
            raise ValueError(f"Missing dynamic features in YAIB batch: {missing_dynamic}")
        if missing_missing:
            raise ValueError(f"Missing missing-indicator features in YAIB batch: {missing_missing}")
        if missing_static and not allow_missing_static:
            raise ValueError(f"Missing static features in YAIB batch: {missing_static}")
        self.indices = SchemaIndices(
            dynamic=[idx for idx in self.indices.dynamic if idx is not None],
            missing=[idx for idx in self.indices.missing if idx is not None],
            static=[idx for idx in self.indices.static if idx is not None],
        )

    def _resolve_indices(self, names: Iterable[str], allow_missing: bool = False) -> List[Optional[int]]:
        indices: List[Optional[int]] = []
        for name in names:
            if name in self.feature_names:
                indices.append(self.feature_names.index(name))
            elif allow_missing:
                indices.append(None)
            else:
                indices.append(None)
        return indices

    def extract(self, batch: tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        data, labels, label_mask = batch[0], batch[1], batch[2]
        static_override = batch[3] if len(batch) > 3 else None
        label_mask = label_mask.bool()
        m_pad = self._infer_pad_mask(data)
        x_val = data[:, :, self.indices.dynamic]
        x_miss = data[:, :, self.indices.missing]
        if static_override is None:
            valid_mask = ~m_pad
            x_static = self._extract_static(data, valid_mask)
        else:
            x_static = static_override
        t_abs = self._build_time_index(data, m_pad)

        return {
            "X_val": x_val,
            "X_miss": x_miss,
            "X_static": x_static,
            "t_abs": t_abs,
            "M_pad": m_pad,
            "M_label": label_mask,
            "y": labels,
            "X_yaib": data,
        }

    def rebuild(
        self,
        x_yaib: torch.Tensor,
        x_val_translated: torch.Tensor,
        x_miss_unchanged: torch.Tensor | None = None,
        x_static_unchanged: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rebuilt = x_yaib.clone()
        rebuilt[:, :, self.indices.dynamic] = x_val_translated
        return rebuilt


    def _infer_pad_mask(self, data: torch.Tensor) -> torch.Tensor:
        return data.abs().sum(dim=-1) == 0

    def _extract_static(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.indices.static:
            return data.new_zeros((data.shape[0], 0))
        batch_size, _, feat_dim = data.shape
        if mask is None:
            row = data[:, 0, :]
        else:
            first_idx = mask.float().argmax(dim=1)
            gather_idx = first_idx.view(batch_size, 1, 1).expand(batch_size, 1, feat_dim)
            row = data.gather(1, gather_idx).squeeze(1)
        return row[:, self.indices.static]

    def _build_time_index(self, data: torch.Tensor, m_pad: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = data.shape
        t = torch.arange(seq_len, device=data.device, dtype=data.dtype).unsqueeze(0)
        t_abs = t.repeat(data.shape[0], 1)
        if m_pad is not None:
            t_abs = t_abs.masked_fill(m_pad, 0.0)
        return t_abs
