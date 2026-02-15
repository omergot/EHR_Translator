"""Focal Loss for imbalanced classification (Lin et al., 2017)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        alpha: Balancing factor for the positive class (scalar or per-class tensor).
        weight: Per-class weights (passed to cross_entropy). Overrides alpha if both set.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw logits from the model.
            targets: (N,) integer class labels.
        """
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)

        # Per-sample alpha based on class
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        else:
            alpha_t = 1.0

        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        return (focal_weight * ce).mean()
