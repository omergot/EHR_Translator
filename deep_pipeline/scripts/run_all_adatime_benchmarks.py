#!/usr/bin/env python3
"""Run all AdaTime benchmark scenarios for HAR, HHAR, and WISDM.

Runs each scenario with:
  1. Source-only baseline
  2. Retrieval translator
  3. Frozen DANN baseline

Saves comprehensive results to experiments/results/adatime_results.json
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
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
)
from src.benchmarks.adatime.adapter import AdaTimeSchemaResolver, AdaTimeRuntime
from src.benchmarks.adatime.evaluate import (
    evaluate_accuracy, evaluate_with_translator, evaluate_source_only,
    print_results_table,
)
from src.benchmarks.adatime.trainer import AdaTimeRetrievalTrainer, AdaTimeFrozenDANNTrainer
from src.core.retrieval_translator import RetrievalTranslator, build_memory_bank

logger = logging.getLogger(__name__)


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def run_single_scenario(
    dataset_name: str,
    source_id: str,
    target_id: str,
    data_path: str,
    device: str = "cuda:3",
    seed: int = 42,
    batch_size: int = 32,
    translator_epochs: int = 30,
    pretrain_epochs: int = 10,
    target_model_epochs: int = 50,
    max_seq_len: int = None,
    d_model: int = 64,
    d_latent: int = 64,
    d_ff: int = 256,
):
    """Run a single scenario with all methods. Returns dict of results."""
    scenario_name = f"{dataset_name} {source_id}->{target_id}"
    run_dir = Path(f"runs/adatime/{dataset_name}/{source_id}_to_{target_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SCENARIO: %s", scenario_name)
    logger.info("=" * 70)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ds_config = get_dataset_config(dataset_name)

    # Load data
    try:
        loaders = create_dataloaders(
            data_path=data_path,
            dataset_name=dataset_name,
            source_id=source_id,
            target_id=target_id,
            batch_size=batch_size,
            val_fraction=0.0,  # AdaTime trains on FULL source set (no val split from source)
            seed=seed,
            max_seq_len=max_seq_len,
        )
    except FileNotFoundError as e:
        logger.error("Data not found for %s: %s", scenario_name, e)
        return None

    results = {}

    # ── Step 1: Train or load frozen target model ──
    target_ckpt = run_dir / "target_model.pt"
    if target_ckpt.exists():
        logger.info("Loading existing target model from %s", target_ckpt)
        frozen_model = load_frozen_target_model(str(target_ckpt), device=device)
    else:
        logger.info("Training target model on target domain...")
        frozen_model = train_target_model(
            target_train_loader=loaders["target_train"],
            target_val_loader=loaders["target_val"],
            input_channels=ds_config.input_channels,
            num_classes=ds_config.num_classes,
            hidden_dim=128,
            num_layers=1,
            epochs=target_model_epochs,
            lr=1e-3,
            device=device,
            save_path=str(target_ckpt),
            patience=10,
        )
        freeze_model(frozen_model)

    # ── Step 2: Target-only upper bound ──
    target_metrics = evaluate_accuracy(frozen_model, loaders["target_test"], device)
    results["target_only"] = target_metrics
    logger.info("Target-only (upper bound): acc=%.4f, f1=%.4f, auroc=%.4f",
                target_metrics["accuracy"], target_metrics["f1"], target_metrics["auroc"])

    # ── Step 3: Source-only baseline ──
    source_metrics = evaluate_accuracy(frozen_model, loaders["source_val"], device)
    results["source_only"] = source_metrics
    logger.info("Source-only (no adaptation): acc=%.4f, f1=%.4f, auroc=%.4f",
                source_metrics["accuracy"], source_metrics["f1"], source_metrics["auroc"])

    # ── Step 4: Retrieval Translator ──
    try:
        translator_ckpt = run_dir / "translator" / "best_checkpoint.pt"
        schema_resolver = AdaTimeSchemaResolver(
            num_features=ds_config.input_channels, static_dim=4,
        )

        if translator_ckpt.exists():
            logger.info("Loading existing translator from %s", translator_ckpt)
            translator = RetrievalTranslator(
                num_features=ds_config.input_channels,
                d_latent=d_latent, d_model=d_model, d_time=16,
                n_enc_layers=2, n_dec_layers=1, n_cross_layers=2,
                n_heads=4, d_ff=d_ff,
                dropout=0.1, out_dropout=0.1,
                static_dim=4,
                temporal_attention_mode="bidirectional",
                output_mode="residual",
            ).to(device)
            ckpt = torch.load(translator_ckpt, map_location=device, weights_only=False)
            translator.load_state_dict(ckpt["translator"])
        else:
            logger.info("Training retrieval translator...")
            translator = RetrievalTranslator(
                num_features=ds_config.input_channels,
                d_latent=d_latent, d_model=d_model, d_time=16,
                n_enc_layers=2, n_dec_layers=1, n_cross_layers=2,
                n_heads=4, d_ff=d_ff,
                dropout=0.1, out_dropout=0.1,
                static_dim=4,
                temporal_attention_mode="bidirectional",
                output_mode="residual",
            )

            trainer = AdaTimeRetrievalTrainer(
                frozen_model=frozen_model,
                translator=translator,
                schema_resolver=schema_resolver,
                target_train_loader=loaders["target_train"],
                num_classes=ds_config.num_classes,
                learning_rate=5e-4,
                lambda_recon=0.1,
                lambda_range=0.1,
                lambda_smooth=0.0,
                lambda_importance_reg=0.01,
                lambda_fidelity=0.01,
                pretrain_epochs=pretrain_epochs,
                k_neighbors=8,
                retrieval_window=4,
                memory_refresh_epochs=5,
                early_stopping_patience=10,
                run_dir=str(run_dir / "translator"),
                device=device,
            )

            trainer.train(
                epochs=translator_epochs,
                train_loader=loaders["source_train"],
                val_loader=loaders["source_val"],
            )

        # Evaluate translator
        memory_bank = build_memory_bank(
            encoder=translator,
            target_loader=loaders["target_train"],
            schema_resolver=schema_resolver,
            device=device,
            window_size=4,
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
        logger.info("Translator: acc=%.4f, f1=%.4f, auroc=%.4f",
                    translator_metrics["accuracy"], translator_metrics["f1"], translator_metrics["auroc"])
    except Exception as e:
        logger.error("Translator failed for %s: %s", scenario_name, traceback.format_exc())
        results["translator"] = {"accuracy": 0.0, "f1": 0.0, "auroc": 0.0, "error": str(e)}

    # ── Step 5: Frozen DANN baseline ──
    try:
        logger.info("Training frozen DANN baseline...")
        # Reset seed for DANN
        torch.manual_seed(seed)
        np.random.seed(seed)

        dann_trainer = AdaTimeFrozenDANNTrainer(
            frozen_model=frozen_model,
            input_channels=ds_config.input_channels,
            num_classes=ds_config.num_classes,
            device=device,
        )
        dann_trainer.train(
            epochs=translator_epochs,
            source_train_loader=loaders["source_train"],
            target_train_loader=loaders["target_train"],
            source_val_loader=loaders["source_val"],
        )

        # Full DANN evaluation
        dann_preds, dann_labels, dann_probs = [], [], []
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
            dann_auroc = roc_auc_score(
                dann_labels.numpy(), dann_probs.numpy(), multi_class="ovr", average="macro",
            )
        except Exception:
            dann_f1 = 0.0
            dann_auroc = 0.0

        results["dann_frozen"] = {"accuracy": dann_acc, "f1": dann_f1, "auroc": dann_auroc}
        logger.info("DANN (frozen): acc=%.4f, f1=%.4f, auroc=%.4f", dann_acc, dann_f1, dann_auroc)
    except Exception as e:
        logger.error("DANN failed for %s: %s", scenario_name, traceback.format_exc())
        results["dann_frozen"] = {"accuracy": 0.0, "f1": 0.0, "auroc": 0.0, "error": str(e)}

    # Print results table
    print_results_table(results, scenario_name)

    # Save per-scenario results
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()}
    with open(run_dir / "results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all AdaTime benchmarks")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--data-path", default="/bigdata/omerg/Thesis/AdaTime/data")
    parser.add_argument("--datasets", nargs="+", default=["HAR"],
                        choices=["HAR", "HHAR", "WISDM", "SSC", "MFD"])
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Specific scenarios as 'SRC-TGT' pairs (e.g., '2-11 6-23')")
    parser.add_argument("--translator-epochs", type=int, default=30)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Downsample long sequences to this length (default: 256 for SSC/MFD)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    all_results = {}

    for dataset_name in args.datasets:
        ds_config = get_dataset_config(dataset_name)
        all_results[dataset_name] = {}

        # Determine max_seq_len: CLI override > auto-detect for long-sequence datasets
        max_seq_len = args.max_seq_len
        if max_seq_len is None and dataset_name in ("SSC", "MFD"):
            max_seq_len = 256
            logger.info("Auto-setting max_seq_len=%d for %s (original=%d)",
                        max_seq_len, dataset_name, ds_config.sequence_len)

        # Determine model size: smaller for univariate long-sequence datasets
        if dataset_name in ("SSC", "MFD"):
            d_model, d_latent, d_ff = 32, 32, 128
            batch_size = 16
        else:
            d_model, d_latent, d_ff = 64, 64, 256
            batch_size = 32

        if args.scenarios:
            scenarios = [(s.split("-")[0], s.split("-")[1]) for s in args.scenarios]
        else:
            scenarios = ds_config.scenarios

        for src_id, trg_id in scenarios:
            scenario_key = f"{src_id}_to_{trg_id}"
            t0 = time.time()

            try:
                result = run_single_scenario(
                    dataset_name=dataset_name,
                    source_id=src_id,
                    target_id=trg_id,
                    data_path=args.data_path,
                    device=args.device,
                    seed=args.seed,
                    batch_size=batch_size,
                    translator_epochs=args.translator_epochs,
                    pretrain_epochs=args.pretrain_epochs,
                    max_seq_len=max_seq_len,
                    d_model=d_model,
                    d_latent=d_latent,
                    d_ff=d_ff,
                )
                if result is not None:
                    all_results[dataset_name][scenario_key] = result
            except Exception as e:
                logger.error("FATAL ERROR in %s %s->%s: %s", dataset_name, src_id, trg_id, e)
                traceback.print_exc()
                all_results[dataset_name][scenario_key] = {"error": str(e)}

            elapsed = time.time() - t0
            logger.info("Scenario %s %s->%s completed in %.1fs", dataset_name, src_id, trg_id, elapsed)

        # Print dataset summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY: %s", dataset_name)
        logger.info("=" * 80)
        methods = ["source_only", "translator", "dann_frozen"]
        header = f"{'Scenario':<15}" + "".join(f"{'  ' + m + ' acc':>20}" for m in methods)
        logger.info(header)
        logger.info("-" * len(header))

        for scenario_key, result in all_results[dataset_name].items():
            if isinstance(result, dict) and "error" not in result:
                line = f"{scenario_key:<15}"
                for m in methods:
                    if m in result:
                        line += f"{result[m].get('accuracy', 0):>20.4f}"
                    else:
                        line += f"{'N/A':>20}"
                logger.info(line)

        # Per-method averages
        for m in methods:
            accs = [
                result[m]["accuracy"]
                for result in all_results[dataset_name].values()
                if isinstance(result, dict) and m in result and "accuracy" in result[m]
            ]
            if accs:
                logger.info("%-15s mean_acc=%.4f +/- %.4f (n=%d)",
                            m, np.mean(accs), np.std(accs), len(accs))

    # Save comprehensive results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Serialize
    serializable = {}
    for ds, scenarios in all_results.items():
        serializable[ds] = {}
        for scenario, methods in scenarios.items():
            if isinstance(methods, dict):
                serializable[ds][scenario] = {}
                for method, metrics in methods.items():
                    if isinstance(metrics, dict):
                        serializable[ds][scenario][method] = {
                            k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in metrics.items()
                        }
                    else:
                        serializable[ds][scenario][method] = metrics

    with open(output_dir / "adatime_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Comprehensive results saved to %s", output_dir / "adatime_results.json")


if __name__ == "__main__":
    main()
