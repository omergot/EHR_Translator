#!/usr/bin/env python3
"""Main entry point for AdaTime benchmark experiments.

Usage:
    # Train target model + translator + evaluate for a single scenario
    python scripts/run_adatime.py --dataset HAR --source 2 --target 11 --device cuda:2

    # Run all scenarios for a dataset
    python scripts/run_adatime.py --dataset HAR --all-scenarios --device cuda:2

    # Use AdaTime's CNN backbone (correct comparison against published AdaTime numbers)
    python scripts/run_adatime.py --dataset HAR --all-scenarios --use-cnn --device cuda:2

    # Only train target model
    python scripts/run_adatime.py --dataset HAR --source 2 --target 11 --target-only

    # Only evaluate (load existing checkpoints)
    python scripts/run_adatime.py --dataset HAR --source 2 --target 11 --eval-only

    # Run with config file
    python scripts/run_adatime.py --config configs/benchmarks/adatime_har_2_to_11.json

    # Include DANN baseline for fair comparison
    python scripts/run_adatime.py --dataset HAR --source 2 --target 11 --include-dann
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.adatime.data_loader import create_dataloaders, get_dataset_config, DATASET_CONFIGS
from src.benchmarks.adatime.target_model import (
    train_target_model, load_frozen_target_model, freeze_model,
    train_source_cnn, load_frozen_source_cnn, AdaTimeCNNClassifier,
)
from src.benchmarks.adatime.adapter import AdaTimeSchemaResolver, AdaTimeRuntime
from src.benchmarks.adatime.evaluate import (
    evaluate_accuracy, evaluate_with_translator, evaluate_source_only,
    print_results_table,
)
from src.benchmarks.adatime.trainer import (
    AdaTimeRetrievalTrainer, AdaTimeFrozenDANNTrainer, AdaTimeCNNRetrievalTrainer,
    ChunkedAdaTimeCNNRetrievalTrainer,
)
from src.benchmarks.adatime.evaluate import evaluate_with_chunked_translator
from src.core.retrieval_translator import RetrievalTranslator, build_memory_bank

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None, verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def get_default_config(dataset_name: str, source_id: str, target_id: str, full_length: bool = False) -> dict:
    """Generate default config for a scenario.

    Adapts hyperparameters for long-sequence / low-channel datasets:
      - SSC (1ch, 3000ts): downsample to 256, smaller model, batch=16
      - MFD (1ch, 5120ts): downsample to 256, smaller model, batch=16

    When full_length=True for SSC/MFD:
      - No downsampling (full_length flag set, max_seq_len ignored)
      - chunk_size=128 for translator (same as HAR sequence length)
      - batch_size=8 (each batch expands to 8 * n_chunks in translator)
    """
    ds_config = get_dataset_config(dataset_name)

    # Defaults for short-sequence datasets (HAR, HHAR, WISDM)
    batch_size = 32
    max_seq_len = None
    chunk_size = None  # No chunking for short sequences
    d_latent = 64
    d_model = 64
    d_ff = 256
    n_heads = 4

    # Adjust for long-sequence datasets
    if dataset_name in ("SSC", "MFD"):
        if full_length:
            # Full-length mode: no downsampling, use chunking in trainer
            max_seq_len = None
            chunk_size = 128  # Translate in 128-timestep chunks
            batch_size = 8    # Keeps memory manageable (8 seqs * 23 chunks = 184 chunk-batches)
            d_latent = 32
            d_model = 32
            d_ff = 128
            n_heads = 4
        else:
            max_seq_len = 256  # Downsample from 3000/5120 to 256
            batch_size = 16
            d_latent = 32
            d_model = 32
            d_ff = 128
            n_heads = 4  # d_model=32 with 4 heads -> 8 per head

    return {
        "dataset": dataset_name,
        "source_id": source_id,
        "target_id": target_id,
        "data_path": str(Path(__file__).resolve().parent.parent.parent / "AdaTime" / "data"),
        "device": "cuda:2",
        "seed": 42,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "full_length": full_length,
        "chunk_size": chunk_size,
        "val_fraction": 0.0,  # AdaTime trains on FULL source set (no val split from source)
        "target_model": {
            # LSTM frozen on target domain (original setup)
            "hidden_dim": 128,
            "num_layers": 1,
            "epochs": 50,
            "lr": 1e-3,
            "patience": 10,
        },
        "source_cnn": {
            # CNN frozen on source domain (correct AdaTime comparison setup)
            # Architecture params pulled from dataset config to match AdaTime exactly
            "mid_channels": ds_config.mid_channels,
            "final_out_channels": ds_config.final_out_channels,
            "kernel_size": ds_config.kernel_size,
            "stride": ds_config.stride,
            "dropout": ds_config.dropout,
            "features_len": ds_config.features_len,
            "epochs": 40,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "patience": 10,
        },
        "translator": {
            "type": "retrieval",
            "d_latent": d_latent,
            "d_model": d_model,
            "d_time": 16,
            "n_enc_layers": 2,
            "n_dec_layers": 1,
            "n_cross_layers": 2,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "dropout": 0.1,
            "out_dropout": 0.1,
            "temporal_attention_mode": "bidirectional",
            "output_mode": "residual",
        },
        "training": {
            "epochs": 30,
            "lr": 5e-4,
            "pretrain_epochs": 10,
            "lambda_recon": 0.1,
            "lambda_range": 0.1,
            "lambda_smooth": 0.0,
            "lambda_importance_reg": 0.01,
            "lambda_fidelity": 0.01,
            "k_neighbors": 8,
            "retrieval_window": 4,
            "memory_refresh_epochs": 5,
            "early_stopping_patience": 10,
        },
        "output": {
            "run_dir": f"runs/adatime/{dataset_name}/{source_id}_to_{target_id}",
        },
    }


def run_scenario(
    config: dict,
    include_dann: bool = False,
    eval_only: bool = False,
    target_only: bool = False,
    use_cnn: bool = False,
    full_length: bool = False,
):
    """Run a single source->target scenario.

    Args:
        config: Scenario configuration dict
        include_dann: Whether to include frozen-model DANN baseline
        eval_only: Skip training and load existing checkpoints
        target_only: Only train and evaluate target/source model
        use_cnn: Use AdaTime's 1D-CNN trained on SOURCE domain (correct AdaTime comparison).
                 When True: trains CNN on source, translator maps target→source-like.
                 When False: trains LSTM on target, translator maps source→target-like.
        full_length: Use chunking strategy for long-sequence datasets (SSC, MFD).
                 When True: no downsampling — sequences are kept at full length (3000 for SSC)
                 and split into 128-timestep chunks during translation.
                 The frozen CNN uses AdaptiveAvgPool1d and handles any sequence length.

    Returns:
        dict of results for each method.
    """
    dataset_name = config["dataset"]
    source_id = config["source_id"]
    target_id = config["target_id"]
    data_path = config["data_path"]
    device = config.get("device", "cuda:2")
    seed = config.get("seed", 42)
    batch_size = config.get("batch_size", 32)

    # Determine run directory (full_length → separate _full subdir to avoid checkpoint conflicts)
    if use_cnn:
        base_run_dir = config["output"].get(
            "run_dir",
            f"runs/adatime_cnn/{dataset_name}/{source_id}_to_{target_id}",
        )
        # Always use adatime_cnn subdirectory
        if "adatime_cnn" not in str(base_run_dir):
            base_run_dir = str(base_run_dir).replace("runs/adatime/", "runs/adatime_cnn/")
        run_dir = Path(base_run_dir)
        if full_length and "_full" not in str(run_dir):
            # Use a separate dir for full-length experiments (different checkpoints)
            run_dir = Path(str(run_dir).replace(
                f"runs/adatime_cnn/{dataset_name}/",
                f"runs/adatime_cnn/{dataset_name}_full/",
            ))
        # For variant runs, use a separate translator dir but share source CNN from base dir.
        # The "source_cnn_base_dir" config key allows reusing CNNs across experiments.
        source_cnn_base = config.get("source_cnn_base_dir", None)
    else:
        run_dir = Path(config["output"]["run_dir"])
        source_cnn_base = None

    run_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = f"{dataset_name} {source_id}->{target_id}"
    mode_tag = "[CNN]" if use_cnn else "[LSTM]"
    logger.info("=" * 70)
    logger.info("Running scenario %s: %s", mode_tag, scenario_name)
    logger.info("=" * 70)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ds_config = get_dataset_config(dataset_name)

    # Load data
    max_seq_len = config.get("max_seq_len", None)
    cfg_full_length = config.get("full_length", False) or full_length
    chunk_size = config.get("chunk_size", 128) or 128
    logger.info(
        "Loading data... (max_seq_len=%s, full_length=%s, chunk_size=%s)",
        max_seq_len, cfg_full_length, chunk_size,
    )
    loaders = create_dataloaders(
        data_path=data_path,
        dataset_name=dataset_name,
        source_id=source_id,
        target_id=target_id,
        batch_size=batch_size,
        val_fraction=config.get("val_fraction", 0.0),
        seed=seed,
        max_seq_len=max_seq_len,
        full_length=cfg_full_length,
    )

    results = {}

    if use_cnn:
        # ── CNN mode: train/load frozen CNN on SOURCE domain ──
        source_cnn_ckpt = run_dir / "source_cnn.pt"
        cnn_cfg = config.get("source_cnn", {})

        # For variant runs, fall back to base dir for source CNN
        if not source_cnn_ckpt.exists() and source_cnn_base is not None:
            base_ckpt = Path(source_cnn_base) / "source_cnn.pt"
            if base_ckpt.exists():
                logger.info("Source CNN not in variant dir, using base: %s", base_ckpt)
                source_cnn_ckpt = base_ckpt

        if source_cnn_ckpt.exists() and not target_only:
            logger.info("Loading existing frozen source CNN from %s", source_cnn_ckpt)
            frozen_model = load_frozen_source_cnn(str(source_cnn_ckpt), device=device)
        else:
            # Use AdaTime's per-dataset batch size for CNN training (may differ from
            # translator batch size which is smaller to fit chunked sequences in memory)
            cnn_batch_size = ds_config.batch_size
            if cnn_batch_size != batch_size:
                logger.info(
                    "CNN training uses batch_size=%d (AdaTime default), translator uses %d",
                    cnn_batch_size, batch_size,
                )
                cnn_loaders = create_dataloaders(
                    data_path=data_path,
                    dataset_name=dataset_name,
                    source_id=source_id,
                    target_id=target_id,
                    batch_size=cnn_batch_size,
                    val_fraction=config.get("val_fraction", 0.0),
                    seed=seed,
                    max_seq_len=max_seq_len,
                    full_length=cfg_full_length,
                )
                cnn_source_train = cnn_loaders["source_train"]
                cnn_source_val = cnn_loaders["source_val"]
            else:
                cnn_source_train = loaders["source_train"]
                cnn_source_val = loaders["source_val"]

            # Pass optimizer betas from training config (AdaTime protocol: (0.5, 0.99))
            training_cfg = config.get("training", {})
            cnn_optimizer_betas = tuple(training_cfg.get("optimizer_betas", [0.9, 0.999]))

            logger.info("Training source CNN on source domain (%s)...", source_id)
            frozen_model = train_source_cnn(
                source_train_loader=cnn_source_train,
                source_val_loader=cnn_source_val,
                input_channels=ds_config.input_channels,
                num_classes=ds_config.num_classes,
                mid_channels=cnn_cfg.get("mid_channels", ds_config.mid_channels),
                final_out_channels=cnn_cfg.get("final_out_channels", ds_config.final_out_channels),
                kernel_size=cnn_cfg.get("kernel_size", ds_config.kernel_size),
                stride=cnn_cfg.get("stride", ds_config.stride),
                dropout=cnn_cfg.get("dropout", ds_config.dropout),
                features_len=cnn_cfg.get("features_len", ds_config.features_len),
                epochs=cnn_cfg.get("epochs", 40),
                lr=cnn_cfg.get("lr", 1e-3),
                weight_decay=cnn_cfg.get("weight_decay", 1e-4),
                device=device,
                save_path=str(source_cnn_ckpt),
                patience=cnn_cfg.get("patience", 10),
                optimizer_betas=cnn_optimizer_betas,
            )

        if target_only:
            # Evaluate source CNN directly on target test data (source-only baseline)
            target_metrics = evaluate_accuracy(frozen_model, loaders["target_test"], device)
            logger.info("Source CNN on target test: accuracy=%.4f, f1=%.4f", target_metrics["accuracy"], target_metrics.get("f1", 0))
            results["source_cnn_on_target"] = target_metrics
            return results

        # ── Source-only CNN baseline: evaluate frozen source CNN directly on target data ──
        logger.info("Evaluating source-only CNN baseline (no adaptation)...")
        source_only_metrics = evaluate_accuracy(frozen_model, loaders["target_test"], device)
        results["source_only_cnn"] = source_only_metrics
        logger.info("Source-only CNN accuracy: %.4f, f1: %.4f", source_only_metrics["accuracy"], source_only_metrics.get("f1", 0))

        # Also evaluate on target validation data for consistency
        source_only_val = evaluate_accuracy(frozen_model, loaders["target_val"], device)
        results["source_only_cnn_val"] = source_only_val

        # ── Train CNN retrieval translator (target → source-like) ──
        if not eval_only:
            translator_cfg = config.get("translator", {})
            training_cfg = config.get("training", {})

            schema_resolver = AdaTimeSchemaResolver(
                num_features=ds_config.input_channels,
                static_dim=4,
            )

            translator = RetrievalTranslator(
                num_features=ds_config.input_channels,
                d_latent=translator_cfg.get("d_latent", 64),
                d_model=translator_cfg.get("d_model", 64),
                d_time=translator_cfg.get("d_time", 16),
                n_enc_layers=translator_cfg.get("n_enc_layers", 2),
                n_dec_layers=translator_cfg.get("n_dec_layers", 1),
                n_cross_layers=translator_cfg.get("n_cross_layers", 2),
                n_heads=translator_cfg.get("n_heads", 4),
                d_ff=translator_cfg.get("d_ff", 256),
                dropout=translator_cfg.get("dropout", 0.1),
                out_dropout=translator_cfg.get("out_dropout", 0.1),
                static_dim=4,
                temporal_attention_mode=translator_cfg.get("temporal_attention_mode", "bidirectional"),
                output_mode=translator_cfg.get("output_mode", "residual"),
            )

            if cfg_full_length:
                # ── Full-length mode: ChunkedAdaTimeCNNRetrievalTrainer ──
                logger.info(
                    "Using ChunkedAdaTimeCNNRetrievalTrainer (chunk_size=%d)", chunk_size,
                )
                ctx_aware = config.get("context_aware", False)
                trainer = ChunkedAdaTimeCNNRetrievalTrainer(
                    frozen_model=frozen_model,
                    translator=translator,
                    schema_resolver=schema_resolver,
                    source_train_loader=loaders["source_train"],  # Phase 1 + memory bank on SOURCE
                    num_classes=ds_config.num_classes,
                    chunk_size=chunk_size,
                    learning_rate=training_cfg.get("lr", 5e-4),
                    weight_decay=training_cfg.get("weight_decay", 1e-5),
                    lambda_recon=training_cfg.get("lambda_recon", 0.1),
                    lambda_range=training_cfg.get("lambda_range", 0.1),
                    lambda_smooth=training_cfg.get("lambda_smooth", 0.0),
                    lambda_importance_reg=training_cfg.get("lambda_importance_reg", 0.01),
                    lambda_fidelity=training_cfg.get("lambda_fidelity", 0.01),
                    pretrain_epochs=training_cfg.get("pretrain_epochs", 10),
                    k_neighbors=training_cfg.get("k_neighbors", 8),
                    retrieval_window=training_cfg.get("retrieval_window", 4),
                    memory_refresh_epochs=training_cfg.get("memory_refresh_epochs", 5),
                    early_stopping_patience=training_cfg.get("early_stopping_patience", 10),
                    use_last_epoch=training_cfg.get("use_last_epoch", False),
                    run_dir=str(run_dir / "translator"),
                    device=device,
                    optimizer_type=training_cfg.get("optimizer_type", "adamw"),
                    optimizer_betas=tuple(training_cfg.get("optimizer_betas", [0.9, 0.999])),
                    context_aware=ctx_aware,
                    drop_last_chunk=config.get("drop_last_chunk", False),
                )

                trainer.train(
                    epochs=training_cfg.get("epochs", 30),
                    target_train_loader=loaders["target_train"],
                    target_val_loader=loaders["target_val"],
                )

                # Rebuild chunk-level latent bank for evaluation
                logger.info("Rebuilding chunk latent bank for final evaluation...")
                trainer._build_chunk_latent_bank()

                # Evaluate on target TEST data using chunked evaluation
                translator_metrics = evaluate_with_chunked_translator(
                    frozen_model=frozen_model,
                    translator=translator,
                    chunk_bank_latents=trainer._chunk_bank_latents,
                    chunk_bank_sequences=trainer._chunk_bank_sequences,
                    schema_resolver=schema_resolver,
                    data_loader=loaders["target_test"],
                    device=device,
                    chunk_size=chunk_size,
                    k_neighbors=training_cfg.get("k_neighbors", 8),
                    context_aware=ctx_aware,
                    drop_last_chunk=config.get("drop_last_chunk", False),
                )
                results["translator_cnn_full"] = translator_metrics
                logger.info(
                    "Translator (CNN, full-length) accuracy: %.4f, f1: %.4f, auroc: %.4f",
                    translator_metrics["accuracy"], translator_metrics.get("f1", 0),
                    translator_metrics.get("auroc", 0),
                )
            else:
                # ── Standard downsampled mode: AdaTimeCNNRetrievalTrainer ──
                trainer = AdaTimeCNNRetrievalTrainer(
                    frozen_model=frozen_model,
                    translator=translator,
                    schema_resolver=schema_resolver,
                    source_train_loader=loaders["source_train"],  # Phase 1 + memory bank on SOURCE
                    num_classes=ds_config.num_classes,
                    learning_rate=training_cfg.get("lr", 5e-4),
                    weight_decay=training_cfg.get("weight_decay", 1e-5),
                    lambda_recon=training_cfg.get("lambda_recon", 0.1),
                    lambda_range=training_cfg.get("lambda_range", 0.1),
                    lambda_smooth=training_cfg.get("lambda_smooth", 0.0),
                    lambda_importance_reg=training_cfg.get("lambda_importance_reg", 0.01),
                    lambda_fidelity=training_cfg.get("lambda_fidelity", 0.01),
                    pretrain_epochs=training_cfg.get("pretrain_epochs", 10),
                    k_neighbors=training_cfg.get("k_neighbors", 8),
                    retrieval_window=training_cfg.get("retrieval_window", 4),
                    memory_refresh_epochs=training_cfg.get("memory_refresh_epochs", 5),
                    early_stopping_patience=training_cfg.get("early_stopping_patience", 10),
                    best_metric=training_cfg.get("best_metric", "val_acc"),
                    use_last_epoch=training_cfg.get("use_last_epoch", False),
                    run_dir=str(run_dir / "translator"),
                    pretrain_fallback_dir=str(Path(config["source_cnn_base_dir"]) / "translator") if config.get("source_cnn_base_dir") else None,
                    device=device,
                    optimizer_type=training_cfg.get("optimizer_type", "adamw"),
                    optimizer_betas=tuple(training_cfg.get("optimizer_betas", [0.9, 0.999])),
                )

                # Phase 2: translator trains on TARGET data, validated on TARGET data
                trainer.train(
                    epochs=training_cfg.get("epochs", 30),
                    target_train_loader=loaders["target_train"],
                    target_val_loader=loaders["target_val"],
                )

                # Evaluate translator on target TEST data (through frozen source CNN)
                # Build SOURCE memory bank for evaluation
                memory_bank = build_memory_bank(
                    encoder=translator,
                    target_loader=loaders["source_train"],  # SOURCE data for memory bank
                    schema_resolver=schema_resolver,
                    device=device,
                    window_size=training_cfg.get("retrieval_window", 4),
                )

                translator_metrics = evaluate_with_translator(
                    frozen_model=frozen_model,
                    translator=translator,
                    schema_resolver=schema_resolver,
                    data_loader=loaders["target_test"],  # Evaluate on TARGET test data
                    memory_bank=memory_bank,
                    device=device,
                )
                results["translator_cnn"] = translator_metrics
                logger.info(
                    "Translator (CNN) accuracy: %.4f, f1: %.4f, auroc: %.4f",
                    translator_metrics["accuracy"], translator_metrics.get("f1", 0), translator_metrics.get("auroc", 0),
                )
        else:
            # Load existing CNN translator checkpoint
            ckpt_path = run_dir / "translator" / "best_checkpoint.pt"
            if ckpt_path.exists():
                logger.info("Loading CNN translator from %s", ckpt_path)
                translator_cfg = config.get("translator", {})
                schema_resolver = AdaTimeSchemaResolver(
                    num_features=ds_config.input_channels, static_dim=4,
                )
                translator = RetrievalTranslator(
                    num_features=ds_config.input_channels,
                    d_latent=translator_cfg.get("d_latent", 64),
                    d_model=translator_cfg.get("d_model", 64),
                    d_time=translator_cfg.get("d_time", 16),
                    n_enc_layers=translator_cfg.get("n_enc_layers", 2),
                    n_dec_layers=translator_cfg.get("n_dec_layers", 1),
                    n_cross_layers=translator_cfg.get("n_cross_layers", 2),
                    n_heads=translator_cfg.get("n_heads", 4),
                    d_ff=translator_cfg.get("d_ff", 256),
                    dropout=translator_cfg.get("dropout", 0.1),
                    out_dropout=translator_cfg.get("out_dropout", 0.1),
                    static_dim=4,
                    temporal_attention_mode=translator_cfg.get("temporal_attention_mode", "bidirectional"),
                    output_mode=translator_cfg.get("output_mode", "residual"),
                ).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                translator.load_state_dict(ckpt["translator"])

                training_cfg = config.get("training", {})
                memory_bank = build_memory_bank(
                    encoder=translator,
                    target_loader=loaders["source_train"],  # SOURCE memory bank
                    schema_resolver=schema_resolver,
                    device=device,
                    window_size=training_cfg.get("retrieval_window", 4),
                )
                translator_metrics = evaluate_with_translator(
                    frozen_model=frozen_model,
                    translator=translator,
                    schema_resolver=schema_resolver,
                    data_loader=loaders["target_test"],  # Evaluate on target test data
                    memory_bank=memory_bank,
                    device=device,
                )
                results["translator_cnn"] = translator_metrics
                logger.info("Translator (CNN) accuracy: %.4f", translator_metrics["accuracy"])
            else:
                logger.warning("No CNN translator checkpoint found at %s", ckpt_path)

    else:
        # ── LSTM mode (original setup): train/load frozen LSTM on TARGET domain ──
        target_ckpt = run_dir / "target_model.pt"
        target_cfg = config.get("target_model", {})

        if target_ckpt.exists() and not target_only:
            logger.info("Loading existing frozen target model from %s", target_ckpt)
            frozen_model = load_frozen_target_model(str(target_ckpt), device=device)
        else:
            logger.info("Training target model on target domain...")
            frozen_model = train_target_model(
                target_train_loader=loaders["target_train"],
                target_val_loader=loaders["target_val"],
                input_channels=ds_config.input_channels,
                num_classes=ds_config.num_classes,
                hidden_dim=target_cfg.get("hidden_dim", 128),
                num_layers=target_cfg.get("num_layers", 1),
                epochs=target_cfg.get("epochs", 50),
                lr=target_cfg.get("lr", 1e-3),
                device=device,
                save_path=str(target_ckpt),
                patience=target_cfg.get("patience", 10),
            )
            freeze_model(frozen_model)

        if target_only:
            target_metrics = evaluate_accuracy(frozen_model, loaders["target_test"], device)
            logger.info("Target-only test accuracy: %.4f", target_metrics["accuracy"])
            results["target_only"] = target_metrics
            return results

        # ── Step 2: Evaluate source-only baseline ──
        logger.info("Evaluating source-only baseline (no adaptation)...")
        source_only_metrics = evaluate_accuracy(frozen_model, loaders["source_val"], device)
        results["source_only"] = source_only_metrics
        logger.info("Source-only accuracy: %.4f", source_only_metrics["accuracy"])

        # ── Step 3: Train and evaluate retrieval translator ──
        if not eval_only:
            translator_cfg = config.get("translator", {})
            training_cfg = config.get("training", {})

            schema_resolver = AdaTimeSchemaResolver(
                num_features=ds_config.input_channels,
                static_dim=4,
            )

            translator = RetrievalTranslator(
                num_features=ds_config.input_channels,
                d_latent=translator_cfg.get("d_latent", 64),
                d_model=translator_cfg.get("d_model", 64),
                d_time=translator_cfg.get("d_time", 16),
                n_enc_layers=translator_cfg.get("n_enc_layers", 2),
                n_dec_layers=translator_cfg.get("n_dec_layers", 1),
                n_cross_layers=translator_cfg.get("n_cross_layers", 2),
                n_heads=translator_cfg.get("n_heads", 4),
                d_ff=translator_cfg.get("d_ff", 256),
                dropout=translator_cfg.get("dropout", 0.1),
                out_dropout=translator_cfg.get("out_dropout", 0.1),
                static_dim=4,
                temporal_attention_mode=translator_cfg.get("temporal_attention_mode", "bidirectional"),
                output_mode=translator_cfg.get("output_mode", "residual"),
            )

            trainer = AdaTimeRetrievalTrainer(
                frozen_model=frozen_model,
                translator=translator,
                schema_resolver=schema_resolver,
                target_train_loader=loaders["target_train"],
                num_classes=ds_config.num_classes,
                learning_rate=training_cfg.get("lr", 5e-4),
                weight_decay=training_cfg.get("weight_decay", 1e-5),
                lambda_recon=training_cfg.get("lambda_recon", 0.1),
                lambda_range=training_cfg.get("lambda_range", 0.1),
                lambda_smooth=training_cfg.get("lambda_smooth", 0.0),
                lambda_importance_reg=training_cfg.get("lambda_importance_reg", 0.01),
                lambda_fidelity=training_cfg.get("lambda_fidelity", 0.01),
                pretrain_epochs=training_cfg.get("pretrain_epochs", 10),
                k_neighbors=training_cfg.get("k_neighbors", 8),
                retrieval_window=training_cfg.get("retrieval_window", 4),
                memory_refresh_epochs=training_cfg.get("memory_refresh_epochs", 5),
                early_stopping_patience=training_cfg.get("early_stopping_patience", 10),
                run_dir=str(run_dir / "translator"),
                device=device,
                optimizer_type=training_cfg.get("optimizer_type", "adamw"),
                optimizer_betas=tuple(training_cfg.get("optimizer_betas", [0.9, 0.999])),
            )

            trainer.train(
                epochs=training_cfg.get("epochs", 30),
                train_loader=loaders["source_train"],
                val_loader=loaders["source_val"],
            )

            # Evaluate translator on source test data (through frozen model)
            # Build memory bank for evaluation
            memory_bank = build_memory_bank(
                encoder=translator,
                target_loader=loaders["target_train"],
                schema_resolver=schema_resolver,
                device=device,
                window_size=training_cfg.get("retrieval_window", 4),
            )

            translator_metrics = evaluate_with_translator(
                frozen_model=frozen_model,
                translator=translator,
                schema_resolver=schema_resolver,
                data_loader=loaders["source_val"],
                memory_bank=memory_bank,
                device=device,
            )
            results["translator"] = translator_metrics
            logger.info("Translator accuracy: %.4f", translator_metrics["accuracy"])
        else:
            # Load existing translator checkpoint
            ckpt_path = run_dir / "translator" / "best_checkpoint.pt"
            if ckpt_path.exists():
                logger.info("Loading translator from %s", ckpt_path)
                translator_cfg = config.get("translator", {})
                schema_resolver = AdaTimeSchemaResolver(
                    num_features=ds_config.input_channels, static_dim=4,
                )
                translator = RetrievalTranslator(
                    num_features=ds_config.input_channels,
                    d_latent=translator_cfg.get("d_latent", 64),
                    d_model=translator_cfg.get("d_model", 64),
                    d_time=translator_cfg.get("d_time", 16),
                    n_enc_layers=translator_cfg.get("n_enc_layers", 2),
                    n_dec_layers=translator_cfg.get("n_dec_layers", 1),
                    n_cross_layers=translator_cfg.get("n_cross_layers", 2),
                    n_heads=translator_cfg.get("n_heads", 4),
                    d_ff=translator_cfg.get("d_ff", 256),
                    dropout=translator_cfg.get("dropout", 0.1),
                    out_dropout=translator_cfg.get("out_dropout", 0.1),
                    static_dim=4,
                    temporal_attention_mode=translator_cfg.get("temporal_attention_mode", "bidirectional"),
                    output_mode=translator_cfg.get("output_mode", "residual"),
                ).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                translator.load_state_dict(ckpt["translator"])

                training_cfg = config.get("training", {})
                memory_bank = build_memory_bank(
                    encoder=translator,
                    target_loader=loaders["target_train"],
                    schema_resolver=schema_resolver,
                    device=device,
                    window_size=training_cfg.get("retrieval_window", 4),
                )
                translator_metrics = evaluate_with_translator(
                    frozen_model=frozen_model,
                    translator=translator,
                    schema_resolver=schema_resolver,
                    data_loader=loaders["source_val"],
                    memory_bank=memory_bank,
                    device=device,
                )
                results["translator"] = translator_metrics
                logger.info("Translator accuracy: %.4f", translator_metrics["accuracy"])
            else:
                logger.warning("No translator checkpoint found at %s", ckpt_path)

        # ── Step 4: Optionally run DANN baseline ──
        if include_dann and not eval_only:
            logger.info("Training frozen-model DANN baseline...")
            dann_trainer = AdaTimeFrozenDANNTrainer(
                frozen_model=frozen_model,
                input_channels=ds_config.input_channels,
                num_classes=ds_config.num_classes,
                device=device,
            )
            dann_trainer.train(
                epochs=training_cfg.get("epochs", 30),
                source_train_loader=loaders["source_train"],
                target_train_loader=loaders["target_train"],
                source_val_loader=loaders["source_val"],
            )

            # Evaluate DANN using full metrics
            dann_preds = []
            dann_labels = []
            dann_probs = []
            dann_trainer.adapter.eval()
            with torch.no_grad():
                for batch in loaders["source_val"]:
                    x = batch[0].to(device)
                    y = batch[1][:, 0].to(device)
                    x_adapted = dann_trainer.translate(x)
                    logits = frozen_model(x_adapted)
                    dann_preds.append(logits.argmax(-1).cpu())
                    dann_labels.append(y.cpu())
                    dann_probs.append(torch.nn.functional.softmax(logits, dim=-1).cpu())
            dann_preds = torch.cat(dann_preds)
            dann_labels = torch.cat(dann_labels)
            dann_probs = torch.cat(dann_probs)
            dann_acc = (dann_preds == dann_labels).float().mean().item()
            try:
                from sklearn.metrics import f1_score, roc_auc_score
                dann_f1 = f1_score(dann_labels.numpy(), dann_preds.numpy(), average="macro")
                dann_auroc = roc_auc_score(dann_labels.numpy(), dann_probs.numpy(), multi_class="ovr", average="macro")
            except Exception:
                dann_f1 = 0.0
                dann_auroc = 0.0
            results["dann_frozen"] = {"accuracy": dann_acc, "f1": dann_f1, "auroc": dann_auroc}
            logger.info("DANN (frozen) accuracy: %.4f, f1: %.4f, auroc: %.4f", dann_acc, dann_f1, dann_auroc)

    # Print results table
    print_results_table(results, scenario_name)

    # Save results
    results_path = run_dir / "results.json"
    # Convert to serializable
    serializable = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return results


def run_all_scenarios(
    config_base: dict,
    include_dann: bool = False,
    use_cnn: bool = False,
    full_length: bool = False,
    variant: str = "",
):
    """Run all scenarios for a dataset.

    Args:
        config_base: Base configuration dict
        include_dann: Whether to include frozen-model DANN baseline
        use_cnn: Use AdaTime's 1D-CNN backbone (correct comparison against published numbers)
        full_length: Use chunking strategy for full-length SSC/MFD sequences
        variant: Optional suffix appended to run dir (e.g. "c256", "latent64") to avoid overwriting
    """
    dataset_name = config_base["dataset"]
    ds_config = get_dataset_config(dataset_name)
    if full_length:
        dir_tag = f"{dataset_name}_full{variant}"
    elif variant:
        dir_tag = f"{dataset_name}{variant}"
    else:
        dir_tag = dataset_name
    if full_length:
        base_run_root = f"runs/adatime_cnn/{dir_tag}"
    elif variant:
        base_run_root = f"runs/adatime_cnn/{dir_tag}" if use_cnn else f"runs/adatime/{dir_tag}"
    else:
        base_run_root = "runs/adatime_cnn" if use_cnn else "runs/adatime"

    all_results = {}
    for src_id, trg_id in ds_config.scenarios:
        config = {**config_base}
        config["source_id"] = src_id
        config["target_id"] = trg_id
        if variant and not full_length:
            config["output"] = {
                "run_dir": f"{base_run_root}/{src_id}_to_{trg_id}",
            }
            # Point to original (non-variant) dir for source CNN reuse
            orig_base = "runs/adatime_cnn" if use_cnn else "runs/adatime"
            config["source_cnn_base_dir"] = f"{orig_base}/{dataset_name}/{src_id}_to_{trg_id}"
        else:
            config["output"] = {
                "run_dir": f"runs/adatime_cnn/{dir_tag}/{src_id}_to_{trg_id}",
            }
        scenario_results = run_scenario(
            config, include_dann=include_dann, use_cnn=use_cnn, full_length=full_length,
        )
        all_results[f"{src_id}_to_{trg_id}"] = scenario_results

    # Compute summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: %s (all scenarios, mode=%s)", dataset_name, "CNN" if use_cnn else "LSTM")
    logger.info("=" * 70)

    methods = set()
    for scenario_res in all_results.values():
        methods.update(scenario_res.keys())

    for method in sorted(methods):
        accs = [
            res[method]["accuracy"]
            for res in all_results.values()
            if method in res
        ]
        f1s = [
            res[method]["f1"]
            for res in all_results.values()
            if method in res and "f1" in res[method]
        ]
        if accs:
            logger.info(
                "%-30s: mean_acc=%.4f +/- %.4f, mean_f1=%.4f (n=%d)",
                method, np.mean(accs), np.std(accs), np.mean(f1s) if f1s else 0, len(accs),
            )

    # Save all results
    if full_length:
        output_dir = Path(f"runs/adatime_cnn/{dataset_name}_full")
    else:
        output_dir = Path(f"{base_run_root}/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for scenario, res in all_results.items():
        serializable[scenario] = {
            k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()
        }
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Also save a copy to experiments/results/ for easy access
    exp_results_dir = Path("experiments/results")
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("_full" + (variant if variant else "")) if full_length else (variant if variant else "")
    exp_results_path = exp_results_dir / f"adatime_cnn_{dataset_name.lower()}{suffix}.json"
    with open(exp_results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results also saved to %s", exp_results_path)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="AdaTime benchmark for frozen-model translator")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--dataset", type=str, choices=["HAR", "HHAR", "WISDM", "SSC", "MFD"], default="HAR")
    parser.add_argument("--source", type=str, help="Source domain ID")
    parser.add_argument("--target", type=str, help="Target domain ID")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--data-path", type=str, default=None, help="Override data path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None, help="Translator training epochs (overrides config)")
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None, help="Downsample long sequences to this length")
    parser.add_argument("--all-scenarios", action="store_true", help="Run all scenarios for dataset")
    parser.add_argument("--target-only", action="store_true", help="Only train and eval target model")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate (load checkpoints)")
    parser.add_argument("--include-dann", action="store_true", help="Include DANN baseline")
    parser.add_argument(
        "--use-cnn", action="store_true",
        help=(
            "Use AdaTime's 1D-CNN trained on SOURCE domain (correct AdaTime comparison). "
            "Translator maps TARGET→source-like. Memory bank and Phase 1 use SOURCE data. "
            "Results saved to runs/adatime_cnn/ (separate from LSTM runs in runs/adatime/)."
        ),
    )
    parser.add_argument(
        "--full-length", action="store_true",
        help=(
            "Use full-length sequences for SSC/MFD (no downsampling). "
            "Sequences are chunked into chunk_size=128 windows during translation. "
            "The frozen CNN handles any length via AdaptiveAvgPool1d. "
            "Results saved to runs/adatime_cnn/SSC_full/. "
            "Implies --use-cnn (CNN is required for chunking strategy)."
        ),
    )
    parser.add_argument(
        "--chunk-size", type=int, default=128,
        help="Chunk size for full-length mode (default: 128). Only used with --full-length.",
    )
    parser.add_argument(
        "--variant", type=str, default="",
        help="Suffix appended to run dir (e.g. '_c256', '_latent64') to avoid overwriting other runs.",
    )
    parser.add_argument(
        "--d-latent", type=int, default=None,
        help="Override translator d_latent (and d_model) dimension.",
    )
    parser.add_argument(
        "--lambda-fidelity", type=float, default=None,
        help="Override training lambda_fidelity (default: 0.01).",
    )
    parser.add_argument(
        "--best-metric", type=str, default=None, choices=["val_acc", "val_loss"],
        help=(
            "Metric for early stopping and best model selection. "
            "'val_acc' (default, higher is better) or 'val_loss' (lower is better). "
            "val_loss is recommended for small test sets where accuracy is too coarse."
        ),
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Override early_stopping_patience (default: 10). Use 0 to disable.",
    )
    parser.add_argument(
        "--last-epoch", action="store_true",
        help="AdaTime protocol: use last epoch model instead of best-val checkpoint. "
             "Combined with val_fraction=0.0 this exactly matches AdaTime's evaluation.",
    )
    parser.add_argument(
        "--context-aware", action="store_true",
        help=(
            "Enable context-aware chunking: each chunk's encoder sees the previous "
            "chunk as left context (2*chunk_size input, only current chunk output kept). "
            "Only used with --full-length."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # --full-length implies --use-cnn
    if args.full_length:
        args.use_cnn = True

    # Load or generate config
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        if args.source is None and not args.all_scenarios:
            parser.error("Must specify --source and --target, or --all-scenarios")
        config = get_default_config(
            args.dataset,
            args.source or "2",
            args.target or "11",
            full_length=args.full_length,
        )

    # Override from CLI args
    config["device"] = args.device
    config["seed"] = args.seed
    if args.full_length:
        # Full-length mode: batch_size defaults to 8 (already set in get_default_config)
        # Allow explicit override
        if args.batch_size != 32:  # User explicitly set batch size
            config["batch_size"] = args.batch_size
        config["full_length"] = True
        config["chunk_size"] = args.chunk_size
    else:
        config["batch_size"] = args.batch_size
    if args.data_path:
        config["data_path"] = args.data_path
    if args.max_seq_len is not None:
        config["max_seq_len"] = args.max_seq_len
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.pretrain_epochs is not None:
        config.setdefault("training", {})["pretrain_epochs"] = args.pretrain_epochs
    if args.d_latent is not None:
        config.setdefault("translator", {})["d_latent"] = args.d_latent
        config.setdefault("translator", {})["d_model"] = args.d_latent  # keep d_model == d_latent
    if args.lambda_fidelity is not None:
        config.setdefault("training", {})["lambda_fidelity"] = args.lambda_fidelity
    if args.best_metric is not None:
        config.setdefault("training", {})["best_metric"] = args.best_metric
    if args.patience is not None:
        config.setdefault("training", {})["early_stopping_patience"] = args.patience
    if args.last_epoch:
        config.setdefault("training", {})["use_last_epoch"] = True
    if args.context_aware:
        config["context_aware"] = True

    # Resolve default data path
    if "data_path" not in config or not Path(config["data_path"]).exists():
        # Try common locations
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "AdaTime" / "data",
            Path("/bigdata/omerg/Thesis/AdaTime/data"),
            Path.home() / "AdaTime" / "data",
        ]
        for p in candidates:
            if p.exists():
                config["data_path"] = str(p)
                break

    # Setup logging — use _full suffix for full-length experiments
    if args.full_length:
        log_base = f"runs/adatime_cnn/{config['dataset']}_full"
    else:
        log_base = config.get("output", {}).get("run_dir", f"runs/adatime/{config['dataset']}")
    setup_logging(
        log_file=os.path.join(log_base, "run.log"),
        verbose=args.verbose,
    )

    logger.info("Config: %s", json.dumps(config, indent=2))

    # Run
    if args.all_scenarios:
        run_all_scenarios(
            config,
            include_dann=args.include_dann,
            use_cnn=args.use_cnn,
            full_length=args.full_length,
            variant=args.variant,
        )
    else:
        run_scenario(
            config,
            include_dann=args.include_dann,
            eval_only=args.eval_only,
            target_only=args.target_only,
            use_cnn=args.use_cnn,
            full_length=args.full_length,
        )


if __name__ == "__main__":
    main()
