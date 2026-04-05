#!/usr/bin/env python3
"""Re-run DANN for scenarios missing MF1 scores (HAR/2_to_11, HAR/6_to_23).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/rerun_dann_mf1.py
"""
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.adatime.data_loader import create_dataloaders, get_dataset_config
from src.benchmarks.adatime.target_model import load_frozen_target_model
from src.benchmarks.adatime.trainer import AdaTimeFrozenDANNTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda:0"  # Will be GPU 1 via CUDA_VISIBLE_DEVICES=1
SCENARIOS = [
    ("HAR", 2, 11),
    ("HAR", 6, 23),
]


def run_dann_for_scenario(dataset_name: str, src_id: int, trg_id: int):
    """Train and evaluate DANN for a single scenario, return full metrics."""
    ds_config = get_dataset_config(dataset_name)
    scenario_name = f"{src_id}_to_{trg_id}"
    run_dir = PROJECT_ROOT / "runs" / "adatime" / dataset_name / scenario_name

    logger.info("=== Re-running DANN for %s/%s ===", dataset_name, scenario_name)

    # Load data
    data_path = "/bigdata/omerg/Thesis/AdaTime/data"
    loaders = create_dataloaders(
        data_path=data_path,
        dataset_name=dataset_name,
        source_id=str(src_id),
        target_id=str(trg_id),
        batch_size=64,
        seed=42,
    )

    # Load frozen target model
    target_model_path = run_dir / "target_model.pt"
    frozen_model = load_frozen_target_model(
        str(target_model_path),
        device=DEVICE,
    )

    # Train DANN
    dann_trainer = AdaTimeFrozenDANNTrainer(
        frozen_model=frozen_model,
        input_channels=ds_config.input_channels,
        num_classes=ds_config.num_classes,
        device=DEVICE,
    )
    dann_trainer.train(
        epochs=30,
        source_train_loader=loaders["source_train"],
        target_train_loader=loaders["target_train"],
        source_val_loader=loaders["source_val"],
    )

    # Evaluate with full metrics
    dann_preds = []
    dann_labels = []
    dann_probs = []
    dann_trainer.adapter.eval()
    with torch.no_grad():
        for batch in loaders["source_val"]:
            x = batch[0].to(DEVICE)
            y = batch[1][:, 0].to(DEVICE)
            x_adapted = dann_trainer.translate(x)
            logits = frozen_model(x_adapted)
            dann_preds.append(logits.argmax(-1).cpu())
            dann_labels.append(y.cpu())
            dann_probs.append(F.softmax(logits, dim=-1).cpu())
    dann_preds = torch.cat(dann_preds)
    dann_labels = torch.cat(dann_labels)
    dann_probs = torch.cat(dann_probs)

    dann_acc = (dann_preds == dann_labels).float().mean().item()

    from sklearn.metrics import f1_score, roc_auc_score
    dann_f1 = f1_score(dann_labels.numpy(), dann_preds.numpy(), average="macro")
    try:
        dann_auroc = roc_auc_score(
            dann_labels.numpy(), dann_probs.numpy(),
            multi_class="ovr", average="macro",
        )
    except Exception as e:
        logger.warning("AUROC computation failed: %s", e)
        dann_auroc = 0.0

    logger.info(
        "DANN results: acc=%.4f, mf1=%.4f, auroc=%.4f",
        dann_acc, dann_f1, dann_auroc,
    )

    # Update results.json
    results_path = run_dir / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    results["dann_frozen"] = {
        "accuracy": dann_acc,
        "f1": dann_f1,
        "auroc": dann_auroc,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Updated %s", results_path)

    return {"accuracy": dann_acc, "f1": dann_f1, "auroc": dann_auroc}


if __name__ == "__main__":
    for dataset, src, trg in SCENARIOS:
        result = run_dann_for_scenario(dataset, src, trg)
        print(f"{dataset}/{src}_to_{trg}: MF1={result['f1']:.4f}")
