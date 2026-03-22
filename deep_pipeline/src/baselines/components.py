"""Shared components for domain adaptation baselines: GRL, discriminator, CORAL loss."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradients during backprop, scaled by lambda (Ganin et al. 2016)."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """MLP discriminator on LSTM hidden states -> domain logit.

    Input: (N, input_dim) flattened non-padded hidden states.
    Output: (N, 1) logits (no sigmoid).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Deep CORAL loss: ||C_s - C_t||_F^2 / (4 * d^2).

    Args:
        source: (N_s, d) source hidden states (non-padded).
        target: (N_t, d) target hidden states (non-padded).

    Returns:
        Scalar CORAL loss.
    """
    d = source.size(1)
    n_s = source.size(0)
    n_t = target.size(0)

    if n_s < 2 or n_t < 2:
        return source.new_tensor(0.0)

    # Mean-center
    source_mean = source.mean(0, keepdim=True)
    target_mean = target.mean(0, keepdim=True)
    source_c = source - source_mean
    target_c = target - target_mean

    # Covariance: (X^T X) / (n - 1)
    cov_s = (source_c.t() @ source_c) / (n_s - 1)
    cov_t = (target_c.t() @ target_c) / (n_t - 1)

    # Frobenius norm squared / (4 * d^2)
    loss = (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)
    return loss


def grl_lambda_schedule(epoch: int, total_epochs: int) -> float:
    """Progressive GRL lambda: 2/(1+exp(-10*p))-1, p = epoch/total_epochs.

    Ramps from ~0 to ~1 over training (Ganin et al. 2016).
    """
    p = epoch / max(total_epochs, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
