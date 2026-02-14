"""MLM (Masked Language Modeling) pretraining for the EHR Translator backbone."""

import logging

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..core.schema import SchemaResolver
from ..core.translator import EHRTranslator


class MLMPretrainer:
    """Pretrain EHRTranslator by masking random timesteps and predicting them.

    Uses bidirectional attention (legitimate since no labels are involved).
    Follows BERT masking strategy: 80% mask token, 10% random, 10% keep.
    """

    def __init__(
        self,
        translator: EHRTranslator,
        schema_resolver: SchemaResolver,
        mask_prob: float = 0.15,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
    ):
        self.translator = translator
        self.schema_resolver = schema_resolver
        self.mask_prob = mask_prob
        self.device = device

        # Initialize MLM-specific components
        self.translator.init_mlm_head()
        self.translator = self.translator.to(device)

        # Optimizer trains all parameters except delta_head (translation-specific)
        mlm_params = [
            p for name, p in self.translator.named_parameters()
            if "delta_head" not in name and "forecast_head" not in name
        ]
        self.optimizer = AdamW(mlm_params, lr=learning_rate, weight_decay=weight_decay)

    def _create_mlm_mask(
        self,
        m_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create MLM masking following BERT strategy.

        Args:
            m_pad: (B, T) padding mask (True = padded).

        Returns:
            mlm_mask: (B, T) bool — True for timesteps selected for MLM.
            replace_mask: (B, T) bool — subset of mlm_mask that should be zeroed (80%).
            random_mask: (B, T) bool — subset of mlm_mask that gets random values (10%).
        """
        valid = ~m_pad
        rand = torch.rand_like(valid.float())
        mlm_mask = valid & (rand < self.mask_prob)

        # BERT strategy: 80% replace, 10% random, 10% keep
        strategy_rand = torch.rand_like(mlm_mask.float())
        replace_mask = mlm_mask & (strategy_rand < 0.8)
        random_mask = mlm_mask & (strategy_rand >= 0.8) & (strategy_rand < 0.9)
        # remaining 10% are kept as-is (still in mlm_mask for loss, but not modified)

        return mlm_mask, replace_mask, random_mask

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one epoch of MLM pretraining.

        Returns:
            Average reconstruction loss over the epoch.
        """
        self.translator.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = tuple(b.to(self.device) for b in batch)
            parts = self.schema_resolver.extract(batch)
            x_val = parts["X_val"]      # (B, T, F)
            x_miss = parts["X_miss"]    # (B, T, F)
            t_abs = parts["t_abs"]      # (B, T)
            m_pad = parts["M_pad"]      # (B, T)
            x_static = parts["X_static"]  # (B, S)

            # Create masks
            mlm_mask, replace_mask, random_mask = self._create_mlm_mask(m_pad)

            # Apply masking to input
            x_masked = x_val.clone()
            # 80%: zero out
            x_masked[replace_mask] = 0.0
            # 10%: random values from another timestep in the batch
            if random_mask.any():
                valid_positions = (~m_pad).nonzero(as_tuple=False)
                if valid_positions.shape[0] > 0:
                    random_idx = valid_positions[
                        torch.randint(valid_positions.shape[0], (random_mask.sum().item(),))
                    ]
                    x_masked[random_mask] = x_val[random_idx[:, 0], random_idx[:, 1]]

            # Forward (bidirectional)
            x_reconstructed = self.translator.forward_mlm(
                x_masked, x_miss, t_abs, m_pad, x_static, mlm_mask
            )

            # Loss: MSE at masked positions only
            if mlm_mask.any():
                loss = F.mse_loss(
                    x_reconstructed[mlm_mask],  # (N_masked, F)
                    x_val[mlm_mask],            # (N_masked, F)
                )
            else:
                loss = x_val.new_tensor(0.0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self, epochs: int, train_loader: DataLoader) -> None:
        """Run MLM pretraining for the specified number of epochs."""
        logging.info("MLM pretraining: %d epochs, mask_prob=%.2f, bidirectional attention", epochs, self.mask_prob)
        for epoch in range(epochs):
            avg_loss = self.train_epoch(train_loader)
            logging.info("MLM pretrain epoch %d/%d - reconstruction_loss=%.4f", epoch + 1, epochs, avg_loss)
        logging.info("MLM pretraining completed.")
