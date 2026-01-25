import logging
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class Translator(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
    
    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError


class IdentityTranslator(Translator):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self.input_size = input_size
        # Dummy learnable scalar so optimizer isn't empty
        self._dummy = nn.Parameter(torch.zeros(()))
    
    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        data = batch[0]
        return data + (self._dummy * 0.0)


class LinearRegressionTranslator(Translator):
    def __init__(
        self,
        input_size: int,
        source_feature_indices: Iterable[int],
        dynamic_feature_names: Iterable[str],
        output_feature_names: Iterable[str],
        target_feature_indices: Iterable[int] | None = None,
        source_missing_indicator_indices: Iterable[int | None] | None = None,
        target_missing_indicator_indices: Iterable[int | None] | None = None,
        use_missing_indicator_mask: bool = True,
    ):
        super().__init__(input_size=input_size)
        self.input_size = input_size
        self.source_feature_indices = list(source_feature_indices)
        self.dynamic_feature_names = list(dynamic_feature_names)
        self.target_feature_indices = list(target_feature_indices) if target_feature_indices is not None else None
        self.output_feature_names = list(output_feature_names)
        self.source_missing_indicator_indices = (
            list(source_missing_indicator_indices) if source_missing_indicator_indices is not None else []
        )
        self.target_missing_indicator_indices = (
            list(target_missing_indicator_indices) if target_missing_indicator_indices is not None else []
        )
        self.use_missing_indicator_mask = use_missing_indicator_mask
        self.a = None
        self.b = None
        self.tgt_mean = None
        self.tgt_std = None

    def _init_stats(self, num_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts = np.zeros(num_features, dtype=np.int64)
        sums = np.zeros(num_features, dtype=np.float64)
        sumsq = np.zeros(num_features, dtype=np.float64)
        return counts, sums, sumsq

    def _update_stats(
        self,
        data_np: np.ndarray,
        pad_mask: np.ndarray,
        feature_indices: list[int],
        missing_indicator_indices: list[int | None],
        counts: np.ndarray,
        sums: np.ndarray,
        sumsq: np.ndarray,
    ) -> None:
        for idx, feat_idx in enumerate(feature_indices):
            feature_vals = data_np[:, :, feat_idx]
            observed = pad_mask
            indicator_idx = missing_indicator_indices[idx] if idx < len(missing_indicator_indices) else None
            if indicator_idx is not None and self.use_missing_indicator_mask:
                indicator_vals = data_np[:, :, indicator_idx]
                observed = observed & (indicator_vals <= 0.5)
            if not observed.any():
                continue
            vals = feature_vals[observed]
            counts[idx] += vals.size
            sums[idx] += vals.sum()
            sumsq[idx] += (vals ** 2).sum()

    def _compute_mean_std(
        self,
        loader,
        feature_indices: list[int],
        missing_indicator_indices: list[int | None],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts, sums, sumsq = self._init_stats(len(feature_indices))
        for batch in loader:
            data, _, mask = batch
            data_np = data.detach().cpu().numpy()
            pad_mask = mask.detach().cpu().numpy().astype(bool)
            self._update_stats(data_np, pad_mask, feature_indices, missing_indicator_indices, counts, sums, sumsq)
        means = np.zeros_like(sums)
        stds = np.zeros_like(sums)
        valid = counts > 0
        means[valid] = sums[valid] / counts[valid]
        variances = np.zeros_like(sums)
        variances[valid] = (sumsq[valid] / counts[valid]) - (means[valid] ** 2)
        variances = np.maximum(variances, 0.0)
        stds[valid] = np.sqrt(variances[valid])
        return means, stds, counts

    def fit_from_loaders(self, train_loader, target_loader=None) -> "LinearRegressionTranslator":
        if target_loader is None:
            target_loader = train_loader
        if self.target_feature_indices is None:
            self.target_feature_indices = list(self.source_feature_indices)
        if not self.source_missing_indicator_indices:
            self.source_missing_indicator_indices = [None] * len(self.source_feature_indices)
        if not self.target_missing_indicator_indices:
            self.target_missing_indicator_indices = [None] * len(self.target_feature_indices)

        src_mean, src_std, src_count = self._compute_mean_std(
            train_loader, self.source_feature_indices, self.source_missing_indicator_indices
        )
        tgt_mean, tgt_std, tgt_count = self._compute_mean_std(
            target_loader, self.target_feature_indices, self.target_missing_indicator_indices
        )

        a = np.ones_like(src_mean)
        b = np.zeros_like(src_mean)
        skipped = 0
        zero_var = 0
        for i in range(len(src_mean)):
            if src_count[i] == 0 or tgt_count[i] == 0:
                skipped += 1
                continue
            if src_std[i] == 0 or tgt_std[i] == 0:
                zero_var += 1
                a[i] = 0.0
                b[i] = tgt_mean[i]
            else:
                a[i] = tgt_std[i] / src_std[i]
                b[i] = tgt_mean[i] - a[i] * src_mean[i]
        self.a = a
        self.b = b
        self.tgt_mean = tgt_mean
        self.tgt_std = tgt_std
        logging.info(
            "Fitted per-feature mapping for %d features (skipped=%d, zero_var=%d).",
            len(self.output_feature_names),
            skipped,
            zero_var,
        )
        return self

    def plot_feature_maps(
        self,
        source_loader,
        target_loader,
        output_dir: str | Path,
        max_points: int = 50000,
    ) -> None:
        if self.a is None or self.b is None:
            raise ValueError("Mapping not fitted. Call fit_from_loaders() first.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_features = len(self.source_feature_indices)
        collected_x = [[] for _ in range(num_features)]
        collected_y = [[] for _ in range(num_features)]

        for source_batch, target_batch in zip(source_loader, target_loader):
            src_data = source_batch[0].detach().cpu().numpy()
            tgt_data = target_batch[0].detach().cpu().numpy()
            src_mask = source_batch[2].detach().cpu().numpy().astype(bool)
            tgt_mask = target_batch[2].detach().cpu().numpy().astype(bool)
            for idx in range(num_features):
                if len(collected_x[idx]) >= max_points:
                    continue
                src_feat = src_data[:, :, self.source_feature_indices[idx]]
                tgt_feat = tgt_data[:, :, self.target_feature_indices[idx]]
                observed = src_mask & tgt_mask
                src_ind = (
                    self.source_missing_indicator_indices[idx]
                    if idx < len(self.source_missing_indicator_indices)
                    else None
                )
                tgt_ind = (
                    self.target_missing_indicator_indices[idx]
                    if idx < len(self.target_missing_indicator_indices)
                    else None
                )
                if src_ind is not None and self.use_missing_indicator_mask:
                    observed = observed & (src_data[:, :, src_ind] <= 0.5)
                if tgt_ind is not None and self.use_missing_indicator_mask:
                    observed = observed & (tgt_data[:, :, tgt_ind] <= 0.5)
                if not observed.any():
                    continue
                x_vals = src_feat[observed]
                y_vals = tgt_feat[observed]
                remaining = max_points - len(collected_x[idx])
                if x_vals.size > remaining:
                    x_vals = x_vals[:remaining]
                    y_vals = y_vals[:remaining]
                collected_x[idx].extend(x_vals.tolist())
                collected_y[idx].extend(y_vals.tolist())

        for idx, feature_name in enumerate(self.output_feature_names):
            if not collected_x[idx]:
                continue
            x = np.array(collected_x[idx])
            y = np.array(collected_y[idx])
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x, y, s=4, alpha=0.3)
            x_line = np.linspace(np.min(x), np.max(x), 100)
            y_line = self.a[idx] * x_line + self.b[idx]
            ax.plot(x_line, y_line, color="red", linewidth=1)
            ax.set_title(feature_name)
            ax.set_xlabel("eICU")
            ax.set_ylabel("MIMIC-IV")
            fig.tight_layout()
            fig.savefig(output_dir / f"{feature_name}.png", dpi=150)
            plt.close(fig)

    def translate_batch(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        data = batch[0]
        data_np = data.detach().cpu().numpy()
        batch_size, seq_len, feat_dim = data_np.shape
        if feat_dim != self.input_size:
            logging.warning("LinearRegressionTranslator input_size %d != batch features %d; updating.", self.input_size, feat_dim)
            self.input_size = feat_dim
        if self.a is None or self.b is None:
            raise ValueError("LinearRegressionTranslator not fitted. Call fit_from_loaders() first.")
        if self.tgt_mean is None or self.tgt_std is None:
            raise ValueError("LinearRegressionTranslator missing target stats. Refit the model.")
        output_np = data_np.copy()
        pad_mask = batch[2].detach().cpu().numpy().astype(bool)
        for idx, feat_idx in enumerate(self.source_feature_indices):
            observed = pad_mask
            indicator_idx = None
            if idx < len(self.source_missing_indicator_indices):
                indicator_idx = self.source_missing_indicator_indices[idx]
            if indicator_idx is not None and self.use_missing_indicator_mask:
                indicator_vals = data_np[:, :, indicator_idx]
                observed = observed & (indicator_vals <= 0.5)
            if not observed.any():
                continue
            vals = output_np[:, :, feat_idx]
            vals[observed] = self.a[idx] * vals[observed] + self.b[idx]
            # Normalize to target (MIMIC-IV) stats for all non-padding positions
            if self.tgt_std[idx] > 0:
                vals[pad_mask] = (vals[pad_mask] - self.tgt_mean[idx]) / self.tgt_std[idx]
            else:
                vals[pad_mask] = 0.0
            output_np[:, :, feat_idx] = vals
        output = torch.from_numpy(output_np).to(data.device).to(data.dtype)
        return output

    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.translate_batch(batch)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "input_size": self.input_size,
                "source_feature_indices": self.source_feature_indices,
                "dynamic_feature_names": self.dynamic_feature_names,
                "target_feature_indices": self.target_feature_indices,
                "output_feature_names": self.output_feature_names,
                "source_missing_indicator_indices": self.source_missing_indicator_indices,
                "target_missing_indicator_indices": self.target_missing_indicator_indices,
                "use_missing_indicator_mask": self.use_missing_indicator_mask,
                "a": self.a,
                "b": self.b,
                "tgt_mean": self.tgt_mean,
                "tgt_std": self.tgt_std,
            },
            path,
        )

    def load(self, path: str) -> "LinearRegressionTranslator":
        state = joblib.load(path)
        self.input_size = state.get("input_size", self.input_size)
        self.source_feature_indices = list(state["source_feature_indices"])
        self.dynamic_feature_names = list(state.get("dynamic_feature_names") or [])
        self.target_feature_indices = list(state.get("target_feature_indices") or [])
        self.output_feature_names = list(state["output_feature_names"])
        self.source_missing_indicator_indices = list(state.get("source_missing_indicator_indices") or [])
        self.target_missing_indicator_indices = list(state.get("target_missing_indicator_indices") or [])
        self.use_missing_indicator_mask = state.get("use_missing_indicator_mask", self.use_missing_indicator_mask)
        self.a = state.get("a")
        self.b = state.get("b")
        self.tgt_mean = state.get("tgt_mean")
        self.tgt_std = state.get("tgt_std")
        return self
