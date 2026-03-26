"""Shared components for DA baselines: GRL, discriminator, CORAL, Sinkhorn, contrastive."""
import logging
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


# ---------------------------------------------------------------------------
# CLUDA components (Ozyurt et al., ICLR 2023)
# ---------------------------------------------------------------------------

class TemporalContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss with k-NN positive pairs (CLUDA).

    For each source hidden state, finds its k nearest neighbours in the
    target hidden-state set (cosine similarity).  Positives = k-NN pairs,
    negatives = all other target timesteps in the mini-batch.

    Also computes a *contextual consistency* term: temporal neighbours in
    source should map to similar regions in target hidden space.
    """

    def __init__(self, temperature: float = 0.07, k_neighbors: int = 5,
                 max_samples: int = 2048):
        super().__init__()
        self.temperature = temperature
        self.k = k_neighbors
        self.max_samples = max_samples

    def forward(
        self,
        h_source: torch.Tensor,
        h_target: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute contrastive + contextual consistency losses.

        Args:
            h_source: (B, T, H) source hidden states.
            h_target: (B, T, H) target hidden states (detached).
            src_mask: (B, T) True for valid (non-padded) source timesteps.
            tgt_mask: (B, T) True for valid (non-padded) target timesteps.

        Returns:
            (contrastive_loss, contextual_loss) both scalars.
        """
        device = h_source.device

        # Flatten to non-padded timesteps
        h_s = h_source[src_mask].float()  # (N_s, H)
        h_t = h_target[tgt_mask].float().detach()  # (N_t, H)

        if h_s.size(0) < 2 or h_t.size(0) < 2:
            zero = h_source.new_tensor(0.0)
            return zero, zero

        # Subsample for memory
        if h_s.size(0) > self.max_samples:
            idx = torch.randperm(h_s.size(0), device=device)[:self.max_samples]
            h_s = h_s[idx]
        if h_t.size(0) > self.max_samples:
            idx = torch.randperm(h_t.size(0), device=device)[:self.max_samples]
            h_t = h_t[idx]

        # L2-normalize
        h_s_norm = F.normalize(h_s, dim=-1)
        h_t_norm = F.normalize(h_t, dim=-1)

        # Cosine similarity: (N_s, N_t)
        sim = h_s_norm @ h_t_norm.t()

        # k-NN: for each source, top-k most similar targets
        k = min(self.k, h_t.size(0))
        topk_sim, topk_idx = sim.topk(k, dim=1)  # (N_s, k)

        # --- InfoNCE contrastive loss ---
        # For each source, use its top-1 NN as positive, rest as negatives
        # logits = sim / temperature, positive = topk_idx[:, 0]
        logits = sim / self.temperature  # (N_s, N_t)
        # Positive: the nearest neighbour for each source
        pos_idx = topk_idx[:, 0]  # (N_s,)
        l_contrastive = F.cross_entropy(logits, pos_idx)

        # --- Contextual consistency loss ---
        # Consecutive source timesteps should have similar NN indices
        # Measure smoothness of the NN assignment across source ordering
        if h_s.size(0) > 1:
            nn_embeds = h_t_norm[pos_idx]  # (N_s, H) - the NN embeddings
            # Cosine similarity between consecutive NN embeddings
            nn_sim = (nn_embeds[:-1] * nn_embeds[1:]).sum(dim=-1)  # (N_s-1,)
            # We want high similarity (smooth mapping), so loss = 1 - mean(sim)
            l_contextual = 1.0 - nn_sim.mean()
        else:
            l_contextual = h_source.new_tensor(0.0)

        return l_contrastive, l_contextual


# ---------------------------------------------------------------------------
# RAINCOAT components (He et al., ICML 2023)
# ---------------------------------------------------------------------------

class SinkhornDivergence(nn.Module):
    """Debiased Sinkhorn divergence (entropic OT) for domain alignment.

    Computes S(x, y) = OT_eps(x, y) - 0.5 * OT_eps(x, x) - 0.5 * OT_eps(y, y)
    using log-domain Sinkhorn iterations for numerical stability.
    """

    def __init__(self, eps: float = 0.1, max_iters: int = 50,
                 max_samples: int = 4096):
        super().__init__()
        self.eps = eps
        self.max_iters = max_iters
        self.max_samples = max_samples

    @staticmethod
    def _cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean cost matrix C_ij = ||x_i - y_j||^2."""
        return torch.cdist(x, y, p=2.0).pow(2)

    def _ot_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute entropic OT cost between x and y via log-domain Sinkhorn."""
        n, m = x.size(0), y.size(0)
        if n == 0 or m == 0:
            return x.new_tensor(0.0)

        C = self._cost_matrix(x, y)  # (n, m)
        # Clamp for numerical stability
        C = C.clamp(max=1e6)

        # Log-domain Sinkhorn
        log_mu = -math.log(n) * torch.ones(n, device=x.device)  # uniform
        log_nu = -math.log(m) * torch.ones(m, device=x.device)  # uniform

        f = torch.zeros(n, device=x.device)
        g = torch.zeros(m, device=x.device)

        for _ in range(self.max_iters):
            f = -self.eps * torch.logsumexp(
                (g.unsqueeze(0) - C) / self.eps + log_nu.unsqueeze(0), dim=1
            )
            g = -self.eps * torch.logsumexp(
                (f.unsqueeze(1) - C) / self.eps + log_mu.unsqueeze(1), dim=0
            )

        # OT cost = <f, mu> + <g, nu>
        return (f * torch.exp(log_mu)).sum() + (g * torch.exp(log_nu)).sum()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute debiased Sinkhorn divergence.

        Args:
            x: (N, D) source points.
            y: (M, D) target points (should be detached).

        Returns:
            Scalar Sinkhorn divergence (non-negative).
        """
        device = x.device

        # Subsample for memory
        if x.size(0) > self.max_samples:
            idx = torch.randperm(x.size(0), device=device)[:self.max_samples]
            x = x[idx]
        if y.size(0) > self.max_samples:
            idx = torch.randperm(y.size(0), device=device)[:self.max_samples]
            y = y[idx]

        if x.size(0) < 2 or y.size(0) < 2:
            return x.new_tensor(0.0)

        ot_xy = self._ot_cost(x, y)
        ot_xx = self._ot_cost(x, x)
        ot_yy = self._ot_cost(y, y)

        return torch.relu(ot_xy - 0.5 * ot_xx - 0.5 * ot_yy)


def frequency_features(
    h: torch.Tensor, m_pad: torch.Tensor,
) -> torch.Tensor:
    """Extract frequency-domain features from hidden state sequences via FFT.

    Args:
        h: (B, T, H) hidden states.
        m_pad: (B, T) padding mask (True = padded).

    Returns:
        h_freq: (N, H) flattened frequency magnitudes (all batch x freq bins).
    """
    # Zero out padded positions before FFT
    h_masked = h.masked_fill(m_pad.unsqueeze(-1), 0.0).float()
    # FFT along temporal axis → complex (B, T//2+1, H)
    h_fft = torch.fft.rfft(h_masked, dim=1)
    h_mag = h_fft.abs()  # magnitude spectrum
    # Flatten: (B * (T//2+1), H)
    return h_mag.reshape(-1, h_mag.size(-1))


# ---------------------------------------------------------------------------
# Stats-only baseline (negative baseline)
# ---------------------------------------------------------------------------

class IdentityDATranslator(nn.Module):
    """Identity translator: returns x_val unchanged.

    Used for the statistics-only baseline where only cross-domain
    normalization is applied (no neural translation).
    """

    def __init__(self):
        super().__init__()
        # Dummy parameter so optimizer doesn't complain
        self._dummy = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x_val: torch.Tensor,
        x_miss: torch.Tensor,
        t_abs: torch.Tensor,
        m_pad: torch.Tensor,
        x_static: torch.Tensor,
        return_forecast: bool = False,
    ) -> torch.Tensor:
        return x_val + (self._dummy * 0.0)
