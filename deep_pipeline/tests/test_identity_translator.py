import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.adapters.yaib import YAIBRuntime
from src.core.eval import TranslatorEvaluator
from src.core.translator import IdentityTranslator


def test_identity_translator_metrics_unchanged():
    """Test that identity translator produces identical metrics to original data."""
    
    config_path = Path(__file__).parent.parent / "configs" / "sample_config.json"
    if not config_path.exists():
        logging.warning(f"Config file not found at {config_path}, skipping test")
        return
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    yaib_runtime = YAIBRuntime(
        data_dir=Path(config["data_dir"]),
        baseline_model_dir=Path(config["baseline_model_dir"]),
        task_config=Path(config["task_config"]),
        model_name=config["model_name"],
        vars=config["vars"],
        file_names=config["file_names"],
        seed=config.get("seed", 42),
        batch_size=config.get("batch_size", 1),
    )
    
    yaib_runtime.load_data()
    yaib_runtime.load_baseline_model()
    
    test_dataset = yaib_runtime.create_dataset('test', ram_cache=True)
    test_loader = yaib_runtime.create_dataloader(test_dataset, shuffle=False)
    
    data_shape = next(iter(test_loader))[0].shape
    input_size = data_shape[-1]
    
    translator = IdentityTranslator(input_size=input_size)
    
    evaluator = TranslatorEvaluator(
        yaib_runtime=yaib_runtime,
        translator=translator,
        device="cpu",
    )
    
    original_metrics = evaluator._evaluate_without_translator(test_loader)
    translated_metrics = evaluator.translate_and_evaluate(test_loader, output_parquet_path=None)
    
    tolerance = 1e-5
    
    for metric in ["AUCROC", "AUCPR", "loss"]:
        diff = abs(translated_metrics[metric] - original_metrics[metric])
        assert diff < tolerance, (
            f"Metric {metric} differs: original={original_metrics[metric]:.6f}, "
            f"translated={translated_metrics[metric]:.6f}, diff={diff:.6f}"
        )
        logging.info(f"✓ {metric}: original={original_metrics[metric]:.6f}, "
                    f"translated={translated_metrics[metric]:.6f}, diff={diff:.6f}")
    
    logging.info("Identity translator test passed: metrics are identical within tolerance")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_identity_translator_metrics_unchanged()





