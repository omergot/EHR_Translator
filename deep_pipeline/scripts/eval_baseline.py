#!/usr/bin/env python
"""Quick script to evaluate a frozen LSTM baseline on a dataset.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_baseline.py --config configs/los_retr_v5_cross3.json
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_baseline.py --config configs/los_retr_v5_cross3.json --data-override /path/to/mimic
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cli import _build_runtime_from_config, _get_training_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config JSON path")
    parser.add_argument("--data-override", default=None, help="Override data_dir (e.g. MIMIC path for sanity check)")
    parser.add_argument("--split", default="test", help="Data split to evaluate")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cached preprocessing (regenerate from scratch)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    training_cfg = _get_training_config(config)
    task_type = training_cfg.get("task_type", "classification")

    data_dir_override = args.data_override
    runtime = _build_runtime_from_config(
        config,
        data_dir_override=data_dir_override,
        batch_size_override=training_cfg.get("batch_size", 16),
        seed_override=training_cfg.get("seed", 2222),
    )
    runtime.load_data(load_cache=not args.no_cache)
    loader = runtime.create_dataloader(args.split, shuffle=False, ram_cache=True)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    runtime.load_baseline_model()
    if hasattr(runtime, "_model") and runtime._model is not None:
        runtime._model = runtime._model.to(device)

    from icu_benchmarks.constants import RunMode

    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = tuple(b.to(device) for b in batch)
            outputs = runtime.forward(batch)
            mask = batch[2].to(outputs.device).bool()
            prediction = torch.masked_select(outputs, mask.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
            target = torch.masked_select(batch[1].to(outputs.device), mask)

            if task_type == "regression":
                raw_pred = prediction[:, 0] if prediction.shape[-1] >= 1 else prediction.squeeze(-1)
                all_preds.append(raw_pred.cpu())
            else:
                if outputs.shape[-1] > 1:
                    proba = torch.softmax(prediction, dim=-1)[:, 1]
                else:
                    proba = torch.sigmoid(prediction).squeeze(-1)
                all_preds.append(proba.cpu())

            all_targets.append(target.cpu())
            total_loss += runtime.compute_loss(outputs, batch).item()
            n_batches += 1

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    data_label = data_dir_override or config["data_dir"]
    logging.info("=" * 60)
    logging.info("BASELINE EVALUATION — %s", Path(data_label).name)
    logging.info("  config: %s", args.config)
    logging.info("  data_dir: %s", data_label)
    logging.info("  split: %s, n_samples: %d, n_batches: %d", args.split, len(preds), n_batches)
    logging.info("  task_type: %s", task_type)

    if task_type == "regression":
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(targets, preds)
        mse = mean_squared_error(targets, preds)
        rmse = mse ** 0.5
        r2 = r2_score(targets, preds)
        logging.info("  MAE: %.4f", mae)
        logging.info("  MSE: %.4f", mse)
        logging.info("  RMSE: %.4f", rmse)
        logging.info("  R2: %.4f", r2)
        logging.info("  loss (MSE): %.4f", total_loss / max(n_batches, 1))
        logging.info("  target stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
                      targets.min(), targets.max(), targets.mean(), targets.std())
        logging.info("  pred stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
                      preds.min(), preds.max(), preds.mean(), preds.std())
    else:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = roc_auc_score(targets, preds)
        auprc = average_precision_score(targets, preds)
        logging.info("  AUCROC: %.4f", auroc)
        logging.info("  AUCPR: %.4f", auprc)
        logging.info("  loss: %.4f", total_loss / max(n_batches, 1))

    logging.info("=" * 60)


if __name__ == "__main__":
    main()
