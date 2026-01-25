import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from .adapters.yaib import YAIBRuntime
from .core.eval import TranslatorEvaluator
from .core.train import TranslatorTrainer
from .core.translator import IdentityTranslator, LinearRegressionTranslator
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSplit
import pandas as pd

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)

def _build_runtime_from_config(config: dict, data_dir_override: str | None = None) -> YAIBRuntime:
    data_dir = Path(data_dir_override) if data_dir_override else Path(config["data_dir"])
    return YAIBRuntime(
        data_dir=data_dir,
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

def _get_translator_config(config: dict) -> dict:
    return config.get("translator", {})


def _get_translator_type(config: dict) -> str:
    return _get_translator_config(config).get("type", "identity")


def _regression_feature_names(columns, missing_prefix, exclude_features):
    return [
        col
        for col in columns
        if not col.startswith(missing_prefix) and col not in exclude_features
    ]


def _prepare_linear_regression(
    config: dict,
    *,
    split: str,
    shuffle: bool,
) -> tuple[LinearRegressionTranslator, YAIBRuntime, DataLoader, DataLoader | None]:
    translator_cfg = _get_translator_config(config)
    static_features = translator_cfg.get("static_features", ["age", "sex", "height", "weight"])
    exclude_static = translator_cfg.get("exclude_static", False)
    use_missing_indicator_mask = translator_cfg.get("use_missing_indicator_mask", True)
    exclude_features = set(translator_cfg.get("exclude_features", []))
    if exclude_static:
        exclude_features.update(static_features)
    missing_prefix = translator_cfg.get("missing_prefix", "MissingIndicator_")
    batch_size = config.get("batch_size", 64)

    source_runtime = _build_runtime_from_config(config)
    source_runtime.load_data(scaling_override=False)
    source_loader = source_runtime.create_dataloader(split, shuffle=shuffle)
    source_feature_names = source_loader.dataset.get_feature_names()
    group_col = source_runtime.vars.get("GROUP")
    source_feature_names = [col for col in source_feature_names if col != group_col]
    source_dynamic = _regression_feature_names(source_feature_names, missing_prefix, exclude_features)
    source_indicator_indices = [
        source_feature_names.index(f"{missing_prefix}{name}") if f"{missing_prefix}{name}" in source_feature_names else None
        for name in source_dynamic
    ]
    source_input_size = next(iter(source_loader))[0].shape[-1]

    target_data_dir = translator_cfg.get("target_data_dir")
    target_loader = None
    target_indices = None
    output_feature_names = source_dynamic

    if target_data_dir:
        target_runtime = _build_runtime_from_config(config, data_dir_override=target_data_dir)
        target_runtime.load_data(scaling_override=False)
        target_loader = target_runtime.create_dataloader(split, shuffle=shuffle)
        target_feature_names = target_loader.dataset.get_feature_names()
        target_group_col = target_runtime.vars.get("GROUP")
        target_feature_names = [col for col in target_feature_names if col != target_group_col]
        target_dynamic = _regression_feature_names(target_feature_names, missing_prefix, exclude_features)
        output_feature_names = [name for name in source_dynamic if name in target_dynamic]
        if not output_feature_names:
            raise ValueError("No overlapping dynamic features between source and target datasets.")
        source_indices = [source_feature_names.index(name) for name in output_feature_names]
        target_indices = [target_feature_names.index(name) for name in output_feature_names]
        target_indicator_indices = [
            target_feature_names.index(f"{missing_prefix}{name}") if f"{missing_prefix}{name}" in target_feature_names else None
            for name in output_feature_names
        ]
    else:
        source_indices = [source_feature_names.index(name) for name in output_feature_names]
        target_indicator_indices = []

    translator = LinearRegressionTranslator(
        input_size=source_input_size,
        source_feature_indices=source_indices,
        dynamic_feature_names=output_feature_names,
        output_feature_names=output_feature_names,
        target_feature_indices=target_indices,
        source_missing_indicator_indices=[source_indicator_indices[source_dynamic.index(name)] for name in output_feature_names],
        target_missing_indicator_indices=target_indicator_indices,
        use_missing_indicator_mask=use_missing_indicator_mask,
    )
    return translator, source_runtime, source_loader, target_loader


def _linear_regression_metrics(
    translator: LinearRegressionTranslator,
    source_loader: DataLoader,
    target_loader: DataLoader,
) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    source_indices = translator.source_feature_indices
    target_feature_names = target_loader.dataset.get_feature_names()
    # Dataset tensors exclude the GROUP column; align feature names accordingly.
    group_col = getattr(target_loader.dataset, "vars", {}).get("GROUP")
    if group_col in target_feature_names:
        target_feature_names = [col for col in target_feature_names if col != group_col]
    target_indices = [
        target_feature_names.index(name)
        for name in translator.output_feature_names
        if name in target_feature_names
    ]
    if len(target_indices) != len(source_indices):
        logging.warning("Target feature index mismatch; skipping regression metrics.")
        return {}

    y_pred_rows = []
    y_true_rows = []
    for source_batch, target_batch in zip(source_loader, target_loader):
        translated = translator.translate_batch(source_batch).detach().cpu().numpy()
        src_mask = source_batch[2].detach().cpu().numpy().reshape(-1)
        tgt_mask = target_batch[2].detach().cpu().numpy().reshape(-1)
        src_flat = translated.reshape(-1, translated.shape[-1])
        tgt_flat = target_batch[0].detach().cpu().numpy().reshape(-1, target_batch[0].shape[-1])
        if src_mask.any() and tgt_mask.any():
            y_pred_rows.append(src_flat[src_mask][:, source_indices])
            y_true_rows.append(tgt_flat[tgt_mask][:, target_indices])
    if not y_pred_rows or not y_true_rows:
        return {}

    y_pred = np.concatenate(y_pred_rows, axis=0)
    y_true = np.concatenate(y_true_rows, axis=0)
    min_len = min(len(y_pred), len(y_true))
    if min_len == 0:
        return {}
    if len(y_pred) != len(y_true):
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def _save_translation_samples(
    translator: LinearRegressionTranslator,
    loader: DataLoader,
    output_dir: str,
    num_samples: int = 100,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader))
    data, _, mask = batch
    data_np = data.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    translated = translator.translate_batch(batch).detach().cpu().numpy()

    flat_mask = mask_np.reshape(-1)
    flat_before = data_np.reshape(-1, data_np.shape[-1])[flat_mask]
    flat_after = translated.reshape(-1, translated.shape[-1])[flat_mask]
    take = min(num_samples, flat_before.shape[0])
    flat_before = flat_before[:take]
    flat_after = flat_after[:take]

    feature_names = loader.dataset.get_feature_names()
    group_col = getattr(loader.dataset, "vars", {}).get("GROUP")
    if group_col in feature_names:
        feature_names = [col for col in feature_names if col != group_col]

    cols = {}
    for idx, name in enumerate(feature_names):
        cols[f"{name}_before"] = flat_before[:, idx]
        cols[f"{name}_after"] = flat_after[:, idx]
    sample_df = pd.DataFrame(cols)
    sample_df.to_csv(output_path / "translation_samples.csv", index=False)

    if translator.a is not None and translator.b is not None:
        ab_df = pd.DataFrame(
            {
                "feature": translator.output_feature_names,
                "a": translator.a,
                "b": translator.b,
            }
        )
        ab_df.to_csv(output_path / "feature_ab.csv", index=False)


def train_translator(args):
    if torch.cuda.is_available() :
        device = [0]  # Use GPU 0 specifically
        logging.info(f"Using GPU 0: {torch.cuda.get_device_name(0)}")

    config = load_config(args.config)
    translator_type = _get_translator_type(config)

    if translator_type == "linear_regression":
        translator, _, train_loader, target_loader = _prepare_linear_regression(
            config,
            split=DataSplit.train,
            shuffle=False,
        )
        translator.fit_from_loaders(train_loader, target_loader)
        model_path = _get_translator_config(config).get("model_path")
        if model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            translator.save(model_path)
            logging.info("Saved linear regression model to %s", model_path)
        logging.info("Linear regression translator fitted on training set.")
        return
    
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
    
    train_loader = yaib_runtime.create_dataloader('train', shuffle=False)
    val_loader = yaib_runtime.create_dataloader('val', shuffle=False)
    
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
    translator_type = _get_translator_type(config)
    eval_baseline_with_target_norm = config.get("eval_baseline_with_target_normalization", False)

    lr_target_loader = None
    if translator_type == "linear_regression":
        translator, yaib_runtime, test_loader, target_loader = _prepare_linear_regression(
            config,
            split=DataSplit.test,
            shuffle=False,
        )
        lr_target_loader = target_loader
        model_path = _get_translator_config(config).get("model_path")
        if not model_path or not Path(model_path).exists():
            raise ValueError("Linear regression model_path is required and must exist.")
        translator.load(model_path)
        logging.info("Loaded linear regression model from %s", model_path)
        translator_cfg = _get_translator_config(config)
        plot_dir = translator_cfg.get("plot_dir")
        if plot_dir and lr_target_loader is not None:
            max_points = int(translator_cfg.get("plot_max_points", 50000))
            translator.plot_feature_maps(test_loader, lr_target_loader, plot_dir, max_points=max_points)
        sample_dir = translator_cfg.get("sample_dir", plot_dir or "deep_pipeline/data/YAIB/translation_samples")
        if sample_dir:
            _save_translation_samples(
                translator,
                test_loader,
                sample_dir,
                num_samples=int(translator_cfg.get("sample_count", 100)),
            )
    else:
        yaib_runtime = _build_runtime_from_config(config)
        yaib_runtime.load_data()

        test_loader = yaib_runtime.create_dataloader('test', shuffle=False)
        
        data_shape = next(iter(test_loader))[0].shape
        input_size = data_shape[-1]
        translator = IdentityTranslator(input_size=input_size)
    
    if translator_type != "linear_regression" and args.translator_checkpoint:
        checkpoint = torch.load(args.translator_checkpoint, map_location="cpu")
        translator.load_state_dict(checkpoint["translator_state_dict"])
        logging.info(f"Loaded translator from {args.translator_checkpoint}")
    
    evaluator = TranslatorEvaluator(
        yaib_runtime=yaib_runtime,
        translator=translator,
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    output_path = Path(args.output_parquet)
    if translator_type == "linear_regression":
        if eval_baseline_with_target_norm:
            # Apply target normalization only (no translation) on raw inputs.
            original_a = translator.a
            original_b = translator.b
            translator.a = np.ones_like(original_a)
            translator.b = np.zeros_like(original_b)
            original_metrics = evaluator.translate_and_evaluate(test_loader, None)
            translator.a = original_a
            translator.b = original_b
        else:
            # Use eICU-normalized runtime for original metrics.
            norm_runtime = _build_runtime_from_config(config)
            norm_runtime.load_data()
            norm_test_loader = norm_runtime.create_dataloader('test', shuffle=False)
            evaluator.yaib_runtime = norm_runtime
            original_metrics = evaluator._evaluate_without_translator(norm_test_loader)
            evaluator.yaib_runtime = yaib_runtime
        translated_metrics = evaluator.translate_and_evaluate(test_loader, output_path)
        results = {"original": original_metrics, "translated": translated_metrics}
    else:
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

    if translator_type == "linear_regression" and lr_target_loader is not None:
        lr_metrics = _linear_regression_metrics(translator, test_loader, lr_target_loader)
        if lr_metrics:
            print("\nLinear Regression Metrics (translated vs target):")
            for key, value in lr_metrics.items():
                print(f"  {key}: {value:.6f}")


def train_and_eval(args):
    train_translator(args)
    translate_and_eval(args)


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

    train_eval_parser = subparsers.add_parser("train_and_eval", help="Train translator then translate and evaluate")
    train_eval_parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    train_eval_parser.add_argument("--output_parquet", type=str, required=True, help="Output path for translated parquet")
    train_eval_parser.add_argument("--translator_checkpoint", type=str, help="Path to translator checkpoint")
    train_eval_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command == "train_translator":
        train_translator(args)
    elif args.command == "translate_and_eval":
        translate_and_eval(args)
    elif args.command == "train_and_eval":
        train_and_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
