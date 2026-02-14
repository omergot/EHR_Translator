"""Multi-Kernel Maximum Mean Discrepancy (MK-MMD) loss for domain adaptation."""

import torch


def multi_kernel_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: list[float] | None = None,
    max_samples: int = 4096,
) -> torch.Tensor:
    """Compute unbiased multi-kernel MMD^2 between two sets of samples.

    Args:
        x: Source samples (N, D).
        y: Target samples (M, D).
        bandwidths: RBF kernel bandwidths (sigma values). If None, uses median
            heuristic multiplied by [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0].
        max_samples: Subsample to this many rows per domain to limit memory.

    Returns:
        Scalar MMD^2 tensor with gradients flowing through *x* only
        (y is treated as a fixed target distribution).
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"Expected 2-D tensors, got x={x.dim()}-D, y={y.dim()}-D")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dims must match: x has {x.shape[1]}, y has {y.shape[1]}")

    # Subsample if needed
    if x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_samples]
        x = x[idx]
    if y.shape[0] > max_samples:
        idx = torch.randperm(y.shape[0], device=y.device)[:max_samples]
        y = y[idx]

    # Detach y so gradients only flow through x (source/translated)
    y = y.detach()

    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return x.new_tensor(0.0)

    # Pairwise squared distances
    xx_dist = torch.cdist(x, x, p=2).pow(2)  # (N, N)
    yy_dist = torch.cdist(y, y, p=2).pow(2)  # (M, M)
    xy_dist = torch.cdist(x, y, p=2).pow(2)  # (N, M)

    if bandwidths is None:
        # Median heuristic on a combined subsample
        with torch.no_grad():
            sample_size = min(512, n, m)
            sx = x[:sample_size]
            sy = y[:sample_size]
            combined = torch.cat([sx, sy], dim=0)
            dists = torch.cdist(combined, combined, p=2).pow(2)
            median_dist = dists.median().clamp(min=1e-8)
            multipliers = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
            bandwidths = [median_dist.item() * mult for mult in multipliers]

    # Sum of RBF kernels across all bandwidths
    k_xx = x.new_zeros(n, n)
    k_yy = x.new_zeros(m, m)
    k_xy = x.new_zeros(n, m)

    for bw in bandwidths:
        gamma = 1.0 / (2.0 * max(bw, 1e-8))
        k_xx = k_xx + torch.exp(-gamma * xx_dist)
        k_yy = k_yy + torch.exp(-gamma * yy_dist)
        k_xy = k_xy + torch.exp(-gamma * xy_dist)

    # Unbiased estimator: exclude diagonal for k_xx and k_yy
    diag_xx = torch.diagonal(k_xx)
    diag_yy = torch.diagonal(k_yy)

    sum_xx = (k_xx.sum() - diag_xx.sum()) / (n * (n - 1))
    sum_yy = (k_yy.sum() - diag_yy.sum()) / (m * (m - 1))
    sum_xy = k_xy.mean()

    mmd_sq = sum_xx + sum_yy - 2.0 * sum_xy
    return mmd_sq
