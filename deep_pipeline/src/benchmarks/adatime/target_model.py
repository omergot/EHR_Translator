"""Target model (frozen LSTM/CNN baseline) for AdaTime benchmark.

Trains a classifier on source-domain data and freezes it.
This frozen model is the "baseline" that our translator must improve.

Two architectures:
  - LSTMClassifier: Simple LSTM encoder -> final hidden state -> linear classifier
    Input: (batch, timesteps, channels)
  - AdaTimeCNNClassifier: AdaTime's exact 3-block 1D-CNN backbone + linear head
    Input: (batch, timesteps, channels) -- internally transposes to channels-first
    Output: (batch, num_classes) logits
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    """Simple LSTM classifier for time-series classification.

    Takes (batch, timesteps, features) input, runs LSTM, takes final hidden
    state, projects to num_classes logits.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 6,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, timesteps, channels) input tensor

        Returns:
            logits: (batch, num_classes) classification logits
        """
        # LSTM: output is (batch, T, hidden_dim)
        lstm_out, (h_n, _) = self.lstm(x)

        # Use final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, 2*hidden)
        else:
            h_final = h_n[-1]  # (batch, hidden)

        logits = self.classifier(h_final)  # (batch, num_classes)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the feature representation (pre-classifier hidden state).

        Args:
            x: (batch, timesteps, channels)

        Returns:
            features: (batch, hidden_dim) or (batch, 2*hidden_dim) if bidirectional
        """
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return h_n[-1]


def train_target_model(
    target_train_loader: DataLoader,
    target_val_loader: DataLoader,
    input_channels: int,
    num_classes: int,
    hidden_dim: int = 128,
    num_layers: int = 1,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    save_path: Optional[str] = None,
    patience: int = 10,
) -> LSTMClassifier:
    """Train an LSTM classifier on target domain data.

    Args:
        target_train_loader: Training data DataLoader
        target_val_loader: Validation data DataLoader
        input_channels: Number of input features/channels
        num_classes: Number of output classes
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        save_path: Path to save the best model checkpoint
        patience: Early stopping patience

    Returns:
        Trained (and frozen) LSTMClassifier
    """
    model = LSTMClassifier(
        input_channels=input_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6,
    )

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in target_train_loader:
            x, y_seq, mask, static = batch
            x = x.to(device)
            # Per-sequence label: take first timestep (all same)
            y = y_seq[:, 0].to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.shape[0]
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.shape[0]

        train_acc = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in target_val_loader:
                x, y_seq, mask, static = batch
                x = x.to(device)
                y = y_seq[:, 0].to(device)

                logits = model(x)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += x.shape[0]

        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "[Target LSTM] Epoch %d/%d: train_loss=%.4f, train_acc=%.4f, val_acc=%.4f",
                epoch, epochs, train_loss / max(train_total, 1), train_acc, val_acc,
            )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("[Target LSTM] Early stopping at epoch %d (best val_acc=%.4f)", epoch, best_val_acc)
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("[Target LSTM] Training complete. Best val_acc=%.4f", best_val_acc)

    # Save checkpoint
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_channels": input_channels,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "best_val_acc": best_val_acc,
        }, save_path)
        logger.info("[Target LSTM] Saved to %s", save_path)

    return model


def load_frozen_target_model(
    checkpoint_path: str,
    device: str = "cuda",
) -> LSTMClassifier:
    """Load a frozen target model from checkpoint.

    All parameters are set to requires_grad=False, dropout is disabled.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LSTMClassifier(
        input_channels=checkpoint["input_channels"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Disable dropout
    for module in model.modules():
        if hasattr(module, "dropout") and isinstance(getattr(module, "dropout"), float):
            if module.dropout > 0:
                module.dropout = 0.0

    # Set to train mode (cuDNN RNN backward requires it, same as our EHR pipeline)
    model.train()

    # But force eval on dropout/batchnorm submodules
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.BatchNorm1d)):
            m.eval()

    logger.info(
        "[Target LSTM] Loaded frozen model from %s (val_acc=%.4f)",
        checkpoint_path, checkpoint.get("best_val_acc", -1),
    )
    return model


def freeze_model(model: LSTMClassifier) -> LSTMClassifier:
    """Freeze all parameters in the model (in-place)."""
    for param in model.parameters():
        param.requires_grad = False

    # Disable dropout
    for module in model.modules():
        if hasattr(module, "dropout") and isinstance(getattr(module, "dropout"), float):
            if module.dropout > 0:
                module.dropout = 0.0

    model.train()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.BatchNorm1d)):
            m.eval()

    return model


# ═══════════════════════════════════════════════════════════════════════
#  AdaTime CNN Classifier (replicates AdaTime's exact CNN backbone)
# ═══════════════════════════════════════════════════════════════════════

class AdaTimeCNNClassifier(nn.Module):
    """AdaTime's exact 3-block 1D-CNN backbone + linear classifier head.

    Replicates the CNN class from AdaTime/models/models.py exactly.
    This is the standard backbone used in all AdaTime published baselines.

    Input format: (batch, timesteps, channels)  [our timesteps-first format]
    Internally transposes to (batch, channels, timesteps) for Conv1d.
    Output: (batch, num_classes) logits

    Architecture:
      Block 1: Conv1d(C, mid_ch, kernel_size, stride) + BN + ReLU + MaxPool2 + Dropout
      Block 2: Conv1d(mid_ch, mid_ch*2, 8, 1) + BN + ReLU + MaxPool2
      Block 3: Conv1d(mid_ch*2, final_ch, 8, 1) + BN + ReLU + MaxPool2
      AdaptiveAvgPool1d(features_len)
      Flatten -> Linear(features_len * final_ch, num_classes)
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        mid_channels: int = 64,
        final_out_channels: int = 128,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.5,
        features_len: int = 1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.mid_channels = mid_channels
        self.final_out_channels = final_out_channels
        self.features_len = features_len

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                input_channels, mid_channels,
                kernel_size=kernel_size, stride=stride, bias=False,
                padding=(kernel_size // 2),
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_channels * 2, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(features_len)
        self.classifier = nn.Linear(features_len * final_out_channels, num_classes)

    def _extract_flat_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract flattened features (before classifier).

        Args:
            x: (batch, channels, timesteps) -- channels-first

        Returns:
            flat: (batch, features_len * final_out_channels)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        return x.reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, timesteps, channels) -- our timesteps-first format

        Returns:
            logits: (batch, num_classes)
        """
        # Transpose: (B, T, C) -> (B, C, T) for Conv1d
        x_cf = x.permute(0, 2, 1)
        flat = self._extract_flat_features(x_cf)
        return self.classifier(flat)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract flat feature representation before classifier.

        Args:
            x: (batch, timesteps, channels) -- our timesteps-first format

        Returns:
            features: (batch, features_len * final_out_channels)
        """
        x_cf = x.permute(0, 2, 1)
        return self._extract_flat_features(x_cf)

    def freeze(self):
        """Freeze all parameters, disable dropout, keep BatchNorm in eval mode."""
        for param in self.parameters():
            param.requires_grad = False

        # model.train() is required for cuDNN compatibility even when frozen
        self.train()

        # BatchNorm and Dropout must be in eval() mode to be deterministic
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.Dropout)):
                m.eval()


def train_source_cnn(
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    input_channels: int,
    num_classes: int,
    mid_channels: int = 64,
    final_out_channels: int = 128,
    kernel_size: int = 5,
    stride: int = 1,
    dropout: float = 0.5,
    features_len: int = 1,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    save_path: Optional[str] = None,
    patience: int = 10,
    optimizer_betas: tuple = (0.9, 0.999),
) -> "AdaTimeCNNClassifier":
    """Train AdaTime's CNN on SOURCE domain data, then freeze it.

    Matches AdaTime's NO_ADAPT (source-only) training protocol exactly:
      - Adam optimizer, lr=1e-3, weight_decay=1e-4, betas=(0.5, 0.99)
      - NO learning rate scheduling (AdaTime: "we exclude any LR scheduling")
      - NO early stopping — train for all 40 epochs (AdaTime reports last model)
      - NO validation split — source_train_loader should be the FULL source train set
      - source_val_loader is accepted for logging only (not for model selection)

    This CNN is trained on source data. During translator training, the
    translator transforms TARGET data to look source-like so this frozen
    source CNN can classify it correctly.

    Args:
        source_train_loader: FULL source domain training data (no val split)
        source_val_loader: Source domain val/test data (for logging only, not selection)
        input_channels: Number of input features/channels
        num_classes: Number of output classes
        mid_channels, final_out_channels, kernel_size, stride, dropout, features_len:
            CNN architecture params (mirror AdaTime's CNN class)
        epochs: Number of training epochs (AdaTime uses 40)
        lr: Learning rate (AdaTime uses 1e-3)
        weight_decay: L2 regularization (AdaTime uses 1e-4)
        device: Device to train on
        save_path: Path to save the final model checkpoint
        patience: Unused (kept for backward compatibility — AdaTime has no early stopping)
        optimizer_betas: Adam beta parameters. AdaTime protocol uses (0.5, 0.99).
            Default (0.9, 0.999) for backward compatibility with existing checkpoints.

    Returns:
        Trained (frozen) AdaTimeCNNClassifier
    """
    model = AdaTimeCNNClassifier(
        input_channels=input_channels,
        num_classes=num_classes,
        mid_channels=mid_channels,
        final_out_channels=final_out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dropout=dropout,
        features_len=features_len,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=optimizer_betas,
    )
    logger.info(
        "[Source CNN] Optimizer: Adam(lr=%g, weight_decay=%g, betas=%s), no LR scheduler",
        lr, weight_decay, optimizer_betas,
    )
    # AdaTime protocol: "we exclude any learning rate scheduling schemes"
    # No scheduler — constant LR throughout training.

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in source_train_loader:
            x, y_seq, mask, static = batch
            x = x.to(device)
            # Per-sequence label: take first timestep (all same)
            y = y_seq[:, 0].to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.shape[0]
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.shape[0]

        train_acc = train_correct / max(train_total, 1)
        avg_train_loss = train_loss / max(train_total, 1)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "[Source CNN] Epoch %d/%d: train_loss=%.4f, train_acc=%.4f",
                epoch, epochs, avg_train_loss, train_acc,
            )

    # AdaTime uses the LAST model (not best val) — use last epoch's weights
    logger.info("[Source CNN] Training complete (no early stopping). Using last epoch model.")
    final_train_acc = train_acc

    # Save checkpoint (last epoch model — AdaTime convention)
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_channels": input_channels,
            "num_classes": num_classes,
            "mid_channels": mid_channels,
            "final_out_channels": final_out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "dropout": dropout,
            "features_len": features_len,
            "final_train_acc": final_train_acc,
        }, save_path)
        logger.info("[Source CNN] Saved to %s", save_path)

    # Freeze the model
    model.freeze()
    logger.info("[Source CNN] Model frozen.")

    return model


def load_frozen_source_cnn(
    checkpoint_path: str,
    device: str = "cuda",
) -> "AdaTimeCNNClassifier":
    """Load a frozen source CNN from checkpoint.

    All parameters are set to requires_grad=False.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AdaTimeCNNClassifier(
        input_channels=checkpoint["input_channels"],
        num_classes=checkpoint["num_classes"],
        mid_channels=checkpoint.get("mid_channels", 64),
        final_out_channels=checkpoint.get("final_out_channels", 128),
        kernel_size=checkpoint.get("kernel_size", 5),
        stride=checkpoint.get("stride", 1),
        dropout=checkpoint.get("dropout", 0.5),
        features_len=checkpoint.get("features_len", 1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.freeze()
    logger.info(
        "[Source CNN] Loaded frozen model from %s (final_train_acc=%.4f)",
        checkpoint_path, checkpoint.get("final_train_acc", checkpoint.get("best_val_acc", -1)),
    )
    return model
