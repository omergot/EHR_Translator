from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class SchemaIndices:
    dynamic: List[int]
    missing: List[int]
    static: List[int]
    # generated[i] = (min_idx, max_idx, count_idx, mean_idx) or None per dynamic feature
    generated: List[Optional[Tuple[int, int, int, int]]] = field(default_factory=list)


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

        # Resolve dynamic (required)
        raw_dynamic = self._resolve_indices(self.dynamic_features)
        missing_dynamic = [name for name, idx in zip(self.dynamic_features, raw_dynamic) if idx is None]
        if missing_dynamic:
            raise ValueError(f"Missing dynamic features in YAIB batch: {missing_dynamic}")

        # Resolve missing indicators (optional — LoS LSTM has none)
        mi_names = [f"{missing_prefix}{name}" for name in self.dynamic_features]
        raw_missing = self._resolve_indices(mi_names, allow_missing=True)
        missing_mi = [name for name, idx in zip(mi_names, raw_missing) if idx is None]
        has_mi = len(missing_mi) == 0
        if not has_mi:
            logging.info(
                "[schema] %d/%d MissingIndicator columns not found — MI will be synthesized as zeros",
                len(missing_mi), len(mi_names),
            )

        # Resolve static (optionally missing)
        raw_static = self._resolve_indices(self.static_features, allow_missing=allow_missing_static)
        missing_static = [name for name, idx in zip(self.static_features, raw_static) if idx is None]
        if missing_static and not allow_missing_static:
            raise ValueError(f"Missing static features in YAIB batch: {missing_static}")

        # Detect generated features (cumulative min/max/count/mean per dynamic feature)
        generated = self._detect_generated_features()

        self.indices = SchemaIndices(
            dynamic=[idx for idx in raw_dynamic if idx is not None],
            missing=[idx for idx in raw_missing if idx is not None] if has_mi else [],
            static=[idx for idx in raw_static if idx is not None],
            generated=generated,
        )
        self._has_mi = has_mi
        self._has_generated = any(g is not None for g in generated)
        if self._has_generated:
            n_gen = sum(1 for g in generated if g is not None)
            logging.info("[schema] Detected %d dynamic features with generated (cumulative) columns", n_gen)

    def _detect_generated_features(self) -> List[Optional[Tuple[int, int, int, int]]]:
        """Detect YAIB-generated cumulative stat columns (_min_hist, _max_hist, _count, _mean_hist)."""
        fn_set = set(self.feature_names)
        fn_index = {name: i for i, name in enumerate(self.feature_names)}
        generated: List[Optional[Tuple[int, int, int, int]]] = []
        for name in self.dynamic_features:
            min_col = f"{name}_min_hist"
            max_col = f"{name}_max_hist"
            count_col = f"{name}_count"
            mean_col = f"{name}_mean_hist"
            if min_col in fn_set and max_col in fn_set and count_col in fn_set and mean_col in fn_set:
                generated.append((
                    fn_index[min_col],
                    fn_index[max_col],
                    fn_index[count_col],
                    fn_index[mean_col],
                ))
            else:
                generated.append(None)
        return generated

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
        if self._has_mi:
            x_miss = data[:, :, self.indices.missing]
        else:
            # No MI columns — synthesize zeros (= "not missing")
            x_miss = torch.zeros_like(x_val)
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
        m_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rebuilt = x_yaib.clone()
        rebuilt[:, :, self.indices.dynamic] = x_val_translated

        if self._has_generated and m_pad is not None:
            self._recompute_generated_features(rebuilt, x_val_translated, m_pad)

        return rebuilt

    def _recompute_generated_features(
        self,
        rebuilt: torch.Tensor,
        x_val: torch.Tensor,
        m_pad: torch.Tensor,
    ) -> None:
        """Recompute cumulative min/max/mean from translated values.

        Keeps count_hist unchanged (depends on original missingness, not values).
        """
        valid = ~m_pad  # (B, T)

        for feat_i, gen_tuple in enumerate(self.indices.generated):
            if gen_tuple is None:
                continue
            min_idx, max_idx, count_idx, mean_idx = gen_tuple
            vals = x_val[:, :, feat_i]  # (B, T)

            # Cumulative min: fill padded with +inf, then cummin
            vals_min = vals.clone()
            vals_min[~valid] = float('inf')
            cum_min, _ = torch.cummin(vals_min, dim=1)
            cum_min[~valid] = 0.0
            rebuilt[:, :, min_idx] = cum_min

            # Cumulative max: fill padded with -inf, then cummax
            vals_max = vals.clone()
            vals_max[~valid] = float('-inf')
            cum_max, _ = torch.cummax(vals_max, dim=1)
            cum_max[~valid] = 0.0
            rebuilt[:, :, max_idx] = cum_max

            # count_hist stays from x_yaib.clone() — unchanged

            # Cumulative mean
            vals_mean = vals.clone()
            vals_mean[~valid] = 0.0
            cum_sum = vals_mean.cumsum(dim=1)
            cum_count = valid.float().cumsum(dim=1).clamp(min=1)
            cum_mean = cum_sum / cum_count
            cum_mean[~valid] = 0.0
            rebuilt[:, :, mean_idx] = cum_mean

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
