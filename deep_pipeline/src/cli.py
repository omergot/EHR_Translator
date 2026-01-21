import argparse
import json
import logging
from pathlib import Path

import torch

from .adapters.yaib import YAIBRuntime
from .core.eval import TranslatorEvaluator
from .core.train import TranslatorTrainer
from .core.translator import IdentityTranslator
from icu_benchmarks.constants import RunMode

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def train_translator(args):
    if torch.cuda.is_available() :
        device = [0]  # Use GPU 0 specifically
        logging.info(f"Using GPU 0: {torch.cuda.get_device_name(0)}")

    config = load_config(args.config)
    
    yaib_runtime = YAIBRuntime(
        data_dir=Path(config["data_dir"]),
        baseline_model_dir=Path(config["baseline_model_dir"]),
        task_config=Path(config["task_config"]),
        model_config=Path(config["model_config"]) if config.get("model_config") else None,
        model_name=config["model_name"],
        vars=config["vars"],
        file_names=config["file_names"],
        seed=config.get("seed", 42),
        batch_size=config.get("batch_size", 1),
        percentile_outliers_csv=Path(config["percentile_outliers_csv"])
        if config.get("percentile_outliers_csv")
        else None,
    )
    
    yaib_runtime.load_data()
    
    train_loader = yaib_runtime.create_dataloader('test', shuffle=False)
    val_loader = yaib_runtime.create_dataloader('test', shuffle=False)
    
    data_shape = next(iter(train_loader))[0].shape

    input_size = data_shape[-1]
    translator = IdentityTranslator(input_size=input_size)
    
    trainer = TranslatorTrainer(
        yaib_runtime=yaib_runtime,
        translator=translator,
        learning_rate=config.get("learning_rate", 1e-4),
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    trainer.train(
        epochs=config.get("epochs", 10),
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        patience=config.get("patience", 10),
    )
    
    logging.info("Training completed")


def translate_and_eval(args):
    config = load_config(args.config)
    
    yaib_runtime = YAIBRuntime(
        data_dir=Path(config["data_dir"]),
        baseline_model_dir=Path(config["baseline_model_dir"]),
        task_config=Path(config["task_config"]),
        model_config=Path(config["model_config"]) if config.get("model_config") else None,
        model_name=config["model_name"],
        vars=config["vars"],
        file_names=config["file_names"],
        seed=config.get("seed", 42),
        batch_size=config.get("batch_size", 1),
        percentile_outliers_csv=Path(config["percentile_outliers_csv"])
        if config.get("percentile_outliers_csv")
        else None,
    )
    
    yaib_runtime.load_data()

    test_loader = yaib_runtime.create_dataloader('test', shuffle=False)
    
    data_shape = next(iter(test_loader))[0].shape
    input_size = data_shape[-1]
    
    translator = IdentityTranslator(input_size=input_size)
    
    if args.translator_checkpoint:
        checkpoint = torch.load(args.translator_checkpoint, map_location="cpu")
        translator.load_state_dict(checkpoint["translator_state_dict"])
        logging.info(f"Loaded translator from {args.translator_checkpoint}")
    
    evaluator = TranslatorEvaluator(
        yaib_runtime=yaib_runtime,
        translator=translator,
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    output_path = Path(args.output_parquet)
    results = evaluator.evaluate_original_vs_translated(test_loader, output_path)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print("\nOriginal Test Data:")
    for metric, value in results["original"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTranslated Test Data:")
    for metric, value in results["translated"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nDifference:")
    for metric in results["original"].keys():
        diff = results["translated"][metric] - results["original"][metric]
        print(f"  {metric}: {diff:+.4f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Translator Training Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    train_parser = subparsers.add_parser("train_translator", help="Train translator model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    train_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    eval_parser = subparsers.add_parser("translate_and_eval", help="Translate and evaluate")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    eval_parser.add_argument("--input_test_parquet", type=str, help="Input test parquet (optional, uses config data_dir)")
    eval_parser.add_argument("--output_parquet", type=str, required=True, help="Output path for translated parquet")
    eval_parser.add_argument("--translator_checkpoint", type=str, help="Path to translator checkpoint")
    eval_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command == "train_translator":
        train_translator(args)
    elif args.command == "translate_and_eval":
        translate_and_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()





