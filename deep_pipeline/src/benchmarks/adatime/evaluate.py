"""Evaluation for AdaTime benchmark experiments.

Computes accuracy, F1, and AUROC on target-domain test data using:
  1. Source-only baseline (no adaptation)
  2. Our translator (frozen model + translator)
  3. Target-only upper bound
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adapter import AdaTimeSchemaResolver, AdaTimeRuntime
from .target_model import LSTMClassifier

logger = logging.getLogger(__name__)


def evaluate_accuracy(
    model: LSTMClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate classification accuracy, F1, and AUROC.

    Args:
        model: Frozen LSTM classifier
        data_loader: Test data loader
        device: Device

    Returns:
        dict with acc, f1, auroc, loss
    """
    # NOTE: Keep model in train() mode — cuDNN RNN backward requires it.
    # Since dropout is disabled and batchnorm is frozen for our frozen models,
    # train() and eval() produce identical outputs.
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            x, y_seq, mask, static = batch
            x = x.to(device)
            y = y_seq[:, 0].to(device)  # Per-sequence label

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())
            total_loss += loss.item() * x.shape[0]
            total_samples += x.shape[0]

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    acc = (all_preds == all_labels).float().mean().item()
    avg_loss = total_loss / max(total_samples, 1)

    # F1 score (macro)
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")
        # Multi-class AUROC (one-vs-rest)
        auroc = roc_auc_score(
            all_labels.numpy(), all_probs.numpy(), multi_class="ovr", average="macro",
        )
    except Exception as e:
        logger.warning("Could not compute F1/AUROC: %s", e)
        f1 = 0.0
        auroc = 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "auroc": auroc,
        "loss": avg_loss,
        "n_samples": total_samples,
    }


def evaluate_with_translator(
    frozen_model: LSTMClassifier,
    translator: torch.nn.Module,
    schema_resolver: AdaTimeSchemaResolver,
    data_loader: DataLoader,
    memory_bank=None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate translated source data on the frozen target model.

    The translator transforms source data to look like target data,
    then the frozen target model classifies it.

    Args:
        frozen_model: Frozen target LSTM
        translator: Trained translator network
        schema_resolver: AdaTime schema resolver
        data_loader: Source test data loader
        memory_bank: Pre-built memory bank for retrieval translator
        device: Device

    Returns:
        dict with acc, f1, auroc, loss
    """
    from src.core.retrieval_translator import query_memory_bank

    # NOTE: Keep frozen_model in train() mode (cuDNN RNN backward requirement).
    translator.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(b.to(device) for b in batch)
            parts = schema_resolver.extract(batch)

            x_val = parts["X_val"]
            x_miss = parts["X_miss"]
            t_abs = parts["t_abs"]
            m_pad = parts["M_pad"]
            x_static = parts["X_static"]
            y = parts["y"][:, 0]  # Per-sequence label

            if memory_bank is not None:
                # Retrieval translator: encode -> retrieve -> translate
                latent = translator.encode(x_val, x_miss, t_abs, m_pad, x_static)
                importance_w = translator.get_importance_weights()
                context = query_memory_bank(
                    latent.detach(), m_pad, memory_bank,
                    k_neighbors=16, retrieval_window=6,
                    importance_weights=importance_w,
                )
                x_translated, _ = translator.forward_with_retrieval(
                    x_val, x_miss, t_abs, m_pad, x_static, context, latent=latent,
                )
            else:
                # Non-retrieval translator
                x_translated = translator(x_val, x_miss, t_abs, m_pad, x_static)

            # Run frozen model on translated data
            logits = frozen_model(x_translated)
            loss = F.cross_entropy(logits, y)

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())
            total_loss += loss.item() * x_val.shape[0]
            total_samples += x_val.shape[0]

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    acc = (all_preds == all_labels).float().mean().item()
    avg_loss = total_loss / max(total_samples, 1)

    try:
        from sklearn.metrics import f1_score, roc_auc_score
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")
        auroc = roc_auc_score(
            all_labels.numpy(), all_probs.numpy(), multi_class="ovr", average="macro",
        )
    except Exception as e:
        logger.warning("Could not compute F1/AUROC: %s", e)
        f1 = 0.0
        auroc = 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "auroc": auroc,
        "loss": avg_loss,
        "n_samples": total_samples,
    }


def evaluate_with_chunked_translator(
    frozen_model,
    translator: torch.nn.Module,
    schema_resolver,
    data_loader: DataLoader,
    chunk_bank_latents: torch.Tensor = None,
    chunk_bank_sequences: torch.Tensor = None,
    device: str = "cuda",
    chunk_size: int = 128,
    k_neighbors: int = 8,
    # Legacy parameters (kept for backward compat but not used)
    memory_bank=None,
    retrieval_window: int = 4,
) -> Dict[str, float]:
    """Evaluate full-length sequences translated via chunking through frozen source CNN.

    Each full-length sequence is split into chunk_size chunks. Each chunk is
    independently translated by the retrieval translator using the chunk-level
    latent bank. Translated chunks are concatenated and passed through the frozen
    CNN (AdaptiveAvgPool1d handles any sequence length).

    Args:
        frozen_model: Frozen source CNN classifier
        translator: Trained retrieval translator
        schema_resolver: AdaTime schema resolver
        data_loader: Target test DataLoader (full-length sequences)
        chunk_bank_latents: (N, d_latent) mean-pooled source chunk latents (GPU)
        chunk_bank_sequences: (N, T, d_latent) full source chunk latents (CPU)
        device: Device
        chunk_size: Chunk size (must match training chunk_size)
        k_neighbors: Number of retrieval neighbors

    Returns:
        dict with accuracy, f1, auroc, loss, n_samples
    """
    translator.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(b.to(device) for b in batch)
            parts = schema_resolver.extract(batch)

            x_val = parts["X_val"]        # (B, T, C)
            x_miss = parts["X_miss"]
            t_abs = parts["t_abs"]
            m_pad = parts["M_pad"]
            x_static = parts["X_static"]
            y = parts["y"][:, 0]

            B, T, C = x_val.shape
            n_chunks = T // chunk_size
            if n_chunks == 0:
                # Fallback: pass full sequence directly
                logits = frozen_model(x_val)
            elif chunk_bank_latents is not None:
                # Split into chunks
                x_trimmed = x_val[:, :n_chunks * chunk_size, :]
                x_miss_trimmed = x_miss[:, :n_chunks * chunk_size, :]
                t_abs_trimmed = t_abs[:, :n_chunks * chunk_size]
                m_pad_trimmed = m_pad[:, :n_chunks * chunk_size]

                x_chunks = x_trimmed.reshape(B * n_chunks, chunk_size, C)
                x_miss_c = x_miss_trimmed.reshape(B * n_chunks, chunk_size, C)
                t_abs_c = t_abs_trimmed.reshape(B * n_chunks, chunk_size)
                m_pad_c = m_pad_trimmed.reshape(B * n_chunks, chunk_size)
                x_static_c = x_static.unsqueeze(1).expand(B, n_chunks, -1).reshape(
                    B * n_chunks, x_static.shape[-1]
                )

                # Encode: (B*n_chunks, T, d_latent)
                latent = translator.encode(x_chunks, x_miss_c, t_abs_c, m_pad_c, x_static_c)

                # Query chunk-level bank: mean-pool query, find K nearest
                query_mean = latent.mean(dim=1)  # (B*n_chunks, d_latent)
                B_c = B * n_chunks
                K = k_neighbors
                N = chunk_bank_latents.shape[0]
                d = chunk_bank_latents.shape[-1]

                CHUNK_Q = 512
                topk_indices = torch.zeros(B_c, K, dtype=torch.long, device=device)
                for start in range(0, B_c, CHUNK_Q):
                    end = min(start + CHUNK_Q, B_c)
                    q_chunk = query_mean[start:end]
                    diff = q_chunk.unsqueeze(1) - chunk_bank_latents.unsqueeze(0)
                    dists = diff.pow(2).sum(dim=-1)
                    _, top_idx = dists.topk(K, dim=-1, largest=False)
                    topk_indices[start:end] = top_idx

                # Gather source chunk latent sequences
                bank_seqs = chunk_bank_sequences  # (N, T_bank, d) on CPU
                gathered = bank_seqs[topk_indices.cpu()].to(device)  # (B_c, K, T_bank, d)
                gathered_mean = gathered.mean(dim=2)  # (B_c, K, d)
                context = gathered_mean.unsqueeze(1).expand(B_c, chunk_size, K, d)  # (B_c, T, K, d)

                x_translated_c, _ = translator.forward_with_retrieval(
                    x_chunks, x_miss_c, t_abs_c, m_pad_c, x_static_c, context, latent=latent,
                )

                # Concatenate chunks: (B, n_chunks * chunk_size, C)
                x_translated = x_translated_c.reshape(B, n_chunks * chunk_size, C)
                logits = frozen_model(x_translated)
            else:
                logits = frozen_model(x_val)

            loss = F.cross_entropy(logits, y)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())
            total_loss += loss.item() * y.shape[0]
            total_samples += y.shape[0]

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    acc = (all_preds == all_labels).float().mean().item()
    avg_loss = total_loss / max(total_samples, 1)

    try:
        from sklearn.metrics import f1_score, roc_auc_score
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")
        auroc = roc_auc_score(
            all_labels.numpy(), all_probs.numpy(), multi_class="ovr", average="macro",
        )
    except Exception as e:
        logger.warning("Could not compute F1/AUROC: %s", e)
        f1 = 0.0
        auroc = 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "auroc": auroc,
        "loss": avg_loss,
        "n_samples": total_samples,
    }


def evaluate_source_only(
    frozen_model: LSTMClassifier,
    source_test_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """Lower bound: evaluate frozen target model on raw source data (no adaptation)."""
    logger.info("Evaluating source-only baseline (no adaptation)...")
    return evaluate_accuracy(frozen_model, source_test_loader, device)


def evaluate_target_only(
    target_train_loader: DataLoader,
    target_test_loader: DataLoader,
    input_channels: int,
    num_classes: int,
    device: str = "cuda",
    epochs: int = 50,
) -> Dict[str, float]:
    """Upper bound: train and evaluate on target domain only."""
    from .target_model import train_target_model

    logger.info("Training target-only upper bound model...")
    model = train_target_model(
        target_train_loader, target_test_loader,
        input_channels=input_channels,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
    )
    return evaluate_accuracy(model, target_test_loader, device)


def print_results_table(results: Dict[str, Dict[str, float]], scenario: str):
    """Print a formatted results table for a scenario."""
    logger.info("\n" + "=" * 70)
    logger.info("Results for scenario: %s", scenario)
    logger.info("-" * 70)
    logger.info("%-25s %8s %8s %8s", "Method", "Acc", "F1", "AUROC")
    logger.info("-" * 70)
    for method, metrics in results.items():
        logger.info(
            "%-25s %8.4f %8.4f %8.4f",
            method,
            metrics.get("accuracy", 0),
            metrics.get("f1", 0),
            metrics.get("auroc", 0),
        )
    logger.info("=" * 70)
