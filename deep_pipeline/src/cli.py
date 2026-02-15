import argparse
import copy
import json
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
import polars as pl

from .adapters.yaib import YAIBRuntime
from .core.eval import TranslatorEvaluator, TransformerTranslatorEvaluator
from .core.train import TranslatorTrainer, TransformerTranslatorTrainer
from .core.translator import IdentityTranslator, LinearRegressionTranslator, EHRTranslator
from .core.schema import SchemaResolver
from .core.static_utils import StaticAugmentedDataset, build_static_matrix_for_dataset, load_static_with_recipe
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSplit
import pandas as pd

def setup_logging(verbose: bool = False, log_file: str | None = "run.log"):
    import os as _os
    # Allow environment variable override for experiment orchestration
    log_file = _os.environ.get("EHR_LOG_FILE", log_file)
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        if log_path.parent != Path("."):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)

def _build_runtime_from_config(
    config: dict,
    data_dir_override: str | None = None,
    batch_size_override: int | None = None,
    seed_override: int | None = None,
) -> YAIBRuntime:
    data_dir = Path(data_dir_override) if data_dir_override else Path(config["data_dir"])
    batch_size = batch_size_override if batch_size_override is not None else config.get("batch_size", 1)
    seed = seed_override if seed_override is not None else config.get("seed", 42)
    vars_cfg = copy.deepcopy(config["vars"])
    file_names_cfg = copy.deepcopy(config["file_names"])
    return YAIBRuntime(
        data_dir=data_dir,
        baseline_model_dir=Path(config["baseline_model_dir"]),
        task_config=Path(config["task_config"]),
        model_config=Path(config["model_config"]) if config.get("model_config") else None,
        model_name=config["model_name"],
        vars=vars_cfg,
        file_names=file_names_cfg,
        seed=seed,
        batch_size=batch_size,
        percentile_outliers_csv=Path(config["percentile_outliers_csv"])
        if config.get("percentile_outliers_csv")
        else None,
    )



def _get_training_config(config: dict) -> dict:
    training = config.get("training", {})
    return {
        "epochs": training.get("epochs", config.get("epochs", 10)),
        "batch_size": training.get("batch_size", config.get("batch_size", 1)),
        "lr": training.get("lr", config.get("learning_rate", 1e-4)),
        "lambda_fidelity": training.get("lambda_fidelity", config.get("lambda_fidelity", 0.01)),
        "lambda_range": training.get("lambda_range", config.get("lambda_range", 1e-3)),
        "lambda_forecast": training.get("lambda_forecast", 0.0),
        "lambda_mmd": training.get("lambda_mmd", 0.0),
        "lambda_mmd_transition": training.get("lambda_mmd_transition", 0.0),
        "early_stopping_patience": training.get("early_stopping_patience", 0),
        "seed": training.get("seed", config.get("seed", 42)),
        "best_metric": training.get("best_metric", config.get("best_metric", "val_total")),
        "oversampling_factor": training.get("oversampling_factor", 0),
        # C1: Focal loss
        "focal_gamma": training.get("focal_gamma", 0),
        "focal_alpha": training.get("focal_alpha", 0.75),
        # C2: GradNorm dynamic weighting
        "use_gradnorm": training.get("use_gradnorm", False),
        "gradnorm_alpha": training.get("gradnorm_alpha", 0.3),
        # C3: Cosine fidelity
        "cosine_fidelity": training.get("cosine_fidelity", False),
        # A1: Variable-length batching
        "variable_length_batching": training.get("variable_length_batching", False),
        # A2: Sequence chunking
        "chunk_size": training.get("chunk_size", 0),
        "chunk_overlap": training.get("chunk_overlap", 5),
        # A3: Padding-aware fidelity
        "padding_aware_fidelity": training.get("padding_aware_fidelity", False),
        "fidelity_proximity_alpha": training.get("fidelity_proximity_alpha", 1.0),
        # A4: Truncate-and-pack
        "max_seq_len": training.get("max_seq_len", 0),
        # B1: Hidden-state MMD
        "lambda_hidden_mmd": training.get("lambda_hidden_mmd", 0.0),
        # B2: Shared encoder
        "lambda_shared_encoder": training.get("lambda_shared_encoder", 0.0),
        # B3: kNN translation
        "lambda_knn": training.get("lambda_knn", 0.0),
        "knn_k": training.get("knn_k", 5),
        "knn_temperature": training.get("knn_temperature", 0.1),
        # B4: Contrastive domain alignment
        "lambda_contrastive": training.get("lambda_contrastive", 0.0),
        "contrastive_temperature": training.get("contrastive_temperature", 0.07),
        # B5: Optimal transport
        "lambda_ot": training.get("lambda_ot", 0.0),
        "ot_reg": training.get("ot_reg", 0.1),
        # B6: Domain-adversarial (DANN)
        "lambda_adversarial": training.get("lambda_adversarial", 0.0),
    }


def _get_output_config(config: dict) -> dict:
    output = config.get("output", {})
    run_dir = output.get("run_dir", config.get("run_dir", config.get("checkpoint_dir", "runs/translator")))
    return {"run_dir": run_dir}




def _get_translator_config(config: dict) -> dict:
    return config.get("translator", {})


def _get_translator_type(config: dict) -> str:
    return _get_translator_config(config).get("type", "identity")


def _get_temporal_attention_mode(translator_cfg: dict) -> str:
    mode = str(translator_cfg.get("temporal_attention_mode", "bidirectional")).strip().lower()
    if mode not in {"bidirectional", "causal"}:
        raise ValueError(
            f"translator.temporal_attention_mode='{mode}' is invalid. "
            "Use 'bidirectional' or 'causal'."
        )
    return mode


def _get_static_recipe(config: dict) -> str:
    paths = config.get("paths", {})
    return paths.get("static_recipe", config.get("static_recipe", ""))


def _augment_loader_with_static(
    loader: DataLoader,
    static_df,
    group_col: str,
    static_features: list[str],
) -> DataLoader:
    static_matrix = build_static_matrix_for_dataset(loader.dataset, static_df, group_col, static_features)
    dataset = StaticAugmentedDataset(loader.dataset, static_matrix)
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        drop_last=loader.drop_last,
        pin_memory=getattr(loader, "pin_memory", False),
        collate_fn=loader.collate_fn,
    )


def _augment_loader_with_zero_static(
    loader: DataLoader,
    static_features: list[str],
) -> DataLoader:
    static_dim = len(static_features)
    static_matrix = torch.zeros((len(loader.dataset), static_dim), dtype=torch.float32)
    dataset = StaticAugmentedDataset(loader.dataset, static_matrix)
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        drop_last=loader.drop_last,
        pin_memory=getattr(loader, "pin_memory", False),
        collate_fn=loader.collate_fn,
    )

def _get_stay_labels(dataset):
    """Return list of per-stay binary labels (1 if any timestep is positive)."""
    labels = []
    for i in range(len(dataset)):
        stay_labels = dataset[i][1]  # (data, labels, mask[, static])
        labels.append(int(stay_labels.max() >= 1))
    return labels


def _apply_oversampling(loader, oversampling_factor):
    """Replace train DataLoader with one using WeightedRandomSampler for positive oversampling."""
    from torch.utils.data import WeightedRandomSampler

    stay_labels = _get_stay_labels(loader.dataset)
    n_pos = sum(stay_labels)
    n_neg = len(stay_labels) - n_pos
    weights = [oversampling_factor if lbl == 1 else 1.0 for lbl in stay_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    logging.info(
        "Oversampling: factor=%.1f, %d pos / %d neg, eff_pos_rate=%.1f%%",
        oversampling_factor, n_pos, n_neg,
        100 * oversampling_factor * n_pos / (oversampling_factor * n_pos + n_neg),
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        drop_last=loader.drop_last,
    )


def _get_bounds_csv(config: dict) -> str:
    paths = config.get("paths", {})
    return paths.get("bounds_csv", config.get("bounds_csv", ""))


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
    debug_mode: bool = False,
    debug_fraction: float = 0.05,
    seed: int = 42,
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
    source_loader = source_runtime.create_dataloader(
        split,
        shuffle=shuffle,
        ram_cache=True,
        subset_fraction=debug_fraction if debug_mode else None,
        subset_seed=seed,
    )
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
        target_loader = target_runtime.create_dataloader(
            split,
            shuffle=shuffle,
            ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
            subset_seed=seed,
        )
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
        device = [0,1,2]  # Use GPU 0 specifically
        logging.info(f"Using GPU 0: {torch.cuda.get_device_name(0)}")

    config = load_config(args.config)
    training_cfg = _get_training_config(config)
    output_cfg = _get_output_config(config)
    debug_mode = config.get("debug", False)
    if debug_mode:
        training_cfg["epochs"] = min(training_cfg["epochs"], 30)
    translator_type = _get_translator_type(config)

    logging.info("=== Training Configuration ===")
    logging.info("  debug: %s", debug_mode)
    logging.info("  translator_type: %s", translator_type)
    for k, v in sorted(training_cfg.items()):
        logging.info("  %s: %s", k, v)
    translator_cfg = _get_translator_config(config)
    for k, v in sorted(translator_cfg.items()):
        logging.info("  translator.%s: %s", k, v)
    logging.info("==============================")

    seed = training_cfg["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if translator_type == "transformer":
        translator_cfg = _get_translator_config(config)
        yaib_runtime = _build_runtime_from_config(
            config,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        yaib_runtime.load_data()
        train_loader = yaib_runtime.create_dataloader(
            'train',
            shuffle=False,
            ram_cache=True,
            subset_fraction=0.2 if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )
        val_loader = yaib_runtime.create_dataloader(
            'val',
            shuffle=False,
            ram_cache=True,
            subset_fraction=0.2 if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )

        feature_names = train_loader.dataset.get_feature_names()
        group_col = yaib_runtime.vars.get("GROUP")
        static_features = config["vars"]["STATIC"]
        static_in_features = all(name in feature_names for name in static_features)
        use_static = config.get("use_static", True)
        if not static_in_features:
            if use_static:
                static_recipe = _get_static_recipe(config)
                if not static_recipe:
                    raise ValueError(
                        "Static features are missing from YAIB inputs; "
                        "provide paths.static_recipe in config for translator conditioning."
                    )
                static_df = load_static_with_recipe(
                    data_dir=Path(config["data_dir"]),
                    file_names=config["file_names"],
                    group_col=group_col,
                    static_features=static_features,
                    recipe_path=Path(static_recipe),
                )
                train_loader = _augment_loader_with_static(train_loader, static_df, group_col, static_features)
                val_loader = _augment_loader_with_static(val_loader, static_df, group_col, static_features)
                logging.info("Using static_recipe for translator conditioning only: %s", static_recipe)
            else:
                train_loader = _augment_loader_with_zero_static(train_loader, static_features)
                val_loader = _augment_loader_with_zero_static(val_loader, static_features)
                logging.info("Static conditioning disabled; using zero static features for translator.")

        oversampling_factor = training_cfg.get("oversampling_factor", 0)
        if oversampling_factor > 0:
            train_loader = _apply_oversampling(train_loader, oversampling_factor)

        schema_resolver = SchemaResolver(
            feature_names=feature_names,
            dynamic_features=config["vars"]["DYNAMIC"],
            static_features=static_features,
            allow_missing_static=not static_in_features,
            missing_prefix=translator_cfg.get("missing_prefix", "MissingIndicator_"),
            group_col=group_col,
        )

        # Load MIMIC target data if target_data_dir is specified (for MMD loss)
        target_train_loader = None
        target_data_dir = config.get("target_data_dir")
        if target_data_dir:
            logging.info("Loading target (MIMIC) data from %s", target_data_dir)
            target_runtime = _build_runtime_from_config(
                config,
                data_dir_override=target_data_dir,
                batch_size_override=training_cfg["batch_size"],
                seed_override=training_cfg["seed"],
            )
            target_runtime.load_data()
            target_train_loader = target_runtime.create_dataloader(
                'train',
                shuffle=True,
                ram_cache=True,
                subset_fraction=0.2 if debug_mode else None,
                subset_seed=training_cfg["seed"],
            )
            # Augment with static features (same handling as eICU)
            target_feature_names = target_train_loader.dataset.get_feature_names()
            target_static_in_features = all(name in target_feature_names for name in static_features)
            if not target_static_in_features:
                if use_static:
                    static_recipe_target = _get_static_recipe(config)
                    if static_recipe_target:
                        target_static_df = load_static_with_recipe(
                            data_dir=Path(target_data_dir),
                            file_names=config["file_names"],
                            group_col=group_col,
                            static_features=static_features,
                            recipe_path=Path(static_recipe_target),
                        )
                        target_train_loader = _augment_loader_with_static(
                            target_train_loader, target_static_df, group_col, static_features
                        )
                    else:
                        target_train_loader = _augment_loader_with_zero_static(target_train_loader, static_features)
                else:
                    target_train_loader = _augment_loader_with_zero_static(target_train_loader, static_features)
            logging.info("Target (MIMIC) train loader: %d batches", len(target_train_loader))

        bounds_csv = _get_bounds_csv(config) or translator_cfg.get("bounds_csv", "")
        if not bounds_csv:
            raise ValueError("bounds_csv must be provided for transformer translator.")

        translator = EHRTranslator(
            num_features=len(schema_resolver.indices.dynamic),
            d_latent=translator_cfg.get("d_latent", 16),
            d_model=translator_cfg.get("d_model", 128),
            d_time=translator_cfg.get("d_time", 16),
            n_layers=translator_cfg.get("n_layers", 4),
            n_heads=translator_cfg.get("n_heads", 8),
            d_ff=translator_cfg.get("d_ff", 512),
            dropout=translator_cfg.get("dropout", 0.2),
            out_dropout=translator_cfg.get("out_dropout", 0.1),
            static_dim=len(static_features),
            temporal_attention_mode=_get_temporal_attention_mode(translator_cfg),
            temporal_attention_window=translator_cfg.get("temporal_attention_window", 0),
        )

        # MLM pretraining phase (optional, controlled by config)
        mlm_pretrain_epochs = translator_cfg.get("mlm_pretrain_epochs", 0)
        if mlm_pretrain_epochs > 0:
            from .core.pretrain import MLMPretrainer

            logging.info("Starting MLM pretraining (%d epochs, bidirectional)", mlm_pretrain_epochs)
            translator.set_temporal_mode("bidirectional")

            pretrainer = MLMPretrainer(
                translator=translator,
                schema_resolver=schema_resolver,
                mask_prob=translator_cfg.get("mlm_mask_prob", 0.15),
                learning_rate=translator_cfg.get("mlm_lr", 1e-4),
                weight_decay=translator_cfg.get("weight_decay", 1e-5),
                device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            )
            pretrainer.train(
                epochs=mlm_pretrain_epochs,
                train_loader=train_loader,
            )

            # Switch back to causal for translator fine-tuning
            original_mode = _get_temporal_attention_mode(translator_cfg)
            translator.set_temporal_mode(original_mode)
            translator.discard_mlm_head()
            translator.delta_head.reset_parameters()
            logging.info("MLM pretraining completed. Switched to %s mode for fine-tuning.", original_mode)

        trainer = TransformerTranslatorTrainer(
            yaib_runtime=yaib_runtime,
            translator=translator,
            schema_resolver=schema_resolver,
            bounds_csv=Path(bounds_csv),
            learning_rate=training_cfg["lr"],
            weight_decay=translator_cfg.get("weight_decay", 1e-5),
            lambda_fidelity=training_cfg["lambda_fidelity"],
            lambda_range=training_cfg["lambda_range"],
            lambda_forecast=training_cfg["lambda_forecast"],
            lambda_mmd=training_cfg["lambda_mmd"],
            lambda_mmd_transition=training_cfg["lambda_mmd_transition"],
            target_train_loader=target_train_loader,
            early_stopping_patience=training_cfg["early_stopping_patience"],
            best_metric=training_cfg["best_metric"],
            run_dir=Path(output_cfg["run_dir"]),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        trainer.train(
            epochs=training_cfg["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
        )
        logging.info("Transformer translator training completed")
        return

    elif translator_type == "linear_regression":
        translator, _, train_loader, target_loader = _prepare_linear_regression(
            config,
            split=DataSplit.train,
            shuffle=False,
            debug_mode=debug_mode,
            seed=training_cfg["seed"],
        )
        translator.fit_from_loaders(train_loader, target_loader)
        model_path = _get_translator_config(config).get("model_path")
        if model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            translator.save(model_path)
            logging.info("Saved linear regression model to %s", model_path)
        logging.info("Linear regression translator fitted on training set.")
        return
    
    yaib_runtime = _build_runtime_from_config(
        config,
        batch_size_override=training_cfg["batch_size"],
        seed_override=training_cfg["seed"],
    )
    
    yaib_runtime.load_data()
    
    train_loader = yaib_runtime.create_dataloader(
        'train',
        shuffle=False,
        ram_cache=True,
        subset_fraction=0.2 if debug_mode else None,
        subset_seed=training_cfg["seed"],
    )
    val_loader = yaib_runtime.create_dataloader(
        'val',
        shuffle=False,
        ram_cache=True,
        subset_fraction=0.2 if debug_mode else None,
        subset_seed=training_cfg["seed"],
    )
    
    data_shape = next(iter(train_loader))[0].shape

    input_size = data_shape[-1]
    translator = IdentityTranslator(input_size=input_size)
    
    trainer = TranslatorTrainer(
        yaib_runtime=yaib_runtime,
        translator=translator,
        learning_rate=training_cfg["lr"],
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    trainer.train(
        epochs=training_cfg["epochs"],
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        patience=config.get("patience", 10),
    )
    
    logging.info("Training completed")


def translate_and_eval(args):
    config = load_config(args.config)
    training_cfg = _get_training_config(config)
    output_cfg = _get_output_config(config)
    translator_type = _get_translator_type(config)
    debug_mode = config.get("debug", False)
    eval_baseline_with_target_norm = config.get("eval_baseline_with_target_normalization", False)

    lr_target_loader = None
    results = None

    if translator_type == "transformer":
        translator_cfg = _get_translator_config(config)
        yaib_runtime = _build_runtime_from_config(
            config,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        yaib_runtime.load_data()
        test_loader = yaib_runtime.create_dataloader(
            'test',
            shuffle=False,
            ram_cache=True,
            subset_fraction=0.2 if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )
        feature_names = test_loader.dataset.get_feature_names()
        group_col = yaib_runtime.vars.get("GROUP")
        static_features = config["vars"]["STATIC"]
        static_in_features = all(name in feature_names for name in static_features)
        use_static = config.get("use_static", True)
        if not static_in_features:
            if use_static:
                static_recipe = _get_static_recipe(config)
                if not static_recipe:
                    raise ValueError(
                        "Static features are missing from YAIB inputs; "
                        "provide paths.static_recipe in config for translator conditioning."
                    )
                static_df = load_static_with_recipe(
                    data_dir=Path(config["data_dir"]),
                    file_names=config["file_names"],
                    group_col=group_col,
                    static_features=static_features,
                    recipe_path=Path(static_recipe),
                )
                for col in ("age", "height", "weight"):
                    if col in static_df.columns:
                        stats = static_df.select(
                            pl.col(col).mean().alias("mean"),
                            pl.col(col).std(ddof=0).alias("std"),
                        ).row(0)
                        logging.info("[static] %s mean=%.6f std=%.6f", col, stats[0], stats[1])
                test_loader = _augment_loader_with_static(test_loader, static_df, group_col, static_features)
                logging.info("Using static_recipe for translator conditioning only: %s", static_recipe)
            else:
                test_loader = _augment_loader_with_zero_static(test_loader, static_features)
                logging.info("Static conditioning disabled; using zero static features for translator.")

        schema_resolver = SchemaResolver(
            feature_names=feature_names,
            dynamic_features=config["vars"]["DYNAMIC"],
            static_features=static_features,
            allow_missing_static=not static_in_features,
            missing_prefix=translator_cfg.get("missing_prefix", "MissingIndicator_"),
            group_col=group_col,
        )

        translator = EHRTranslator(
            num_features=len(schema_resolver.indices.dynamic),
            d_latent=translator_cfg.get("d_latent", 16),
            d_model=translator_cfg.get("d_model", 128),
            d_time=translator_cfg.get("d_time", 16),
            n_layers=translator_cfg.get("n_layers", 4),
            n_heads=translator_cfg.get("n_heads", 8),
            d_ff=translator_cfg.get("d_ff", 512),
            dropout=translator_cfg.get("dropout", 0.2),
            out_dropout=translator_cfg.get("out_dropout", 0.1),
            static_dim=len(static_features),
            temporal_attention_mode=_get_temporal_attention_mode(translator_cfg),
            temporal_attention_window=translator_cfg.get("temporal_attention_window", 0),
        )

        checkpoint_path = args.translator_checkpoint
        if not checkpoint_path:
            checkpoint_path = str(Path(output_cfg["run_dir"]) / "best_translator.pt")
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            translator.load_state_dict(checkpoint["translator_state_dict"], strict=False)
            logging.info("Loaded transformer translator from %s", checkpoint_path)
        else:
            logging.warning("No transformer checkpoint found at %s", checkpoint_path)

        evaluator = TransformerTranslatorEvaluator(
            yaib_runtime=yaib_runtime,
            translator=translator,
            schema_resolver=schema_resolver,
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        output_path = Path(args.output_parquet)
        sample_dir = translator_cfg.get(
            "sample_dir", "/bigdata/omerg/Thesis/EHR_Translator/deep_pipeline/data/YAIB/translation_samples"
        )
        sample_size = int(translator_cfg.get("sample_size", 1000))
        results = evaluator.evaluate_original_vs_translated(
            test_loader,
            output_path,
            sample_output_dir=Path(sample_dir) if sample_dir else None,
            sample_size=sample_size,
            export_full_sequence=getattr(args, "export_full_sequence", True),
        )

    elif translator_type == "linear_regression":
        translator, yaib_runtime, test_loader, target_loader = _prepare_linear_regression(
            config,
            split=DataSplit.test,
            shuffle=False,
            debug_mode=debug_mode,
            seed=training_cfg["seed"],
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

        test_loader = yaib_runtime.create_dataloader(
            'test',
            shuffle=False,
            ram_cache=True,
            subset_fraction=0.2 if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )

        data_shape = next(iter(test_loader))[0].shape
        input_size = data_shape[-1]
        translator = IdentityTranslator(input_size=input_size)

    if translator_type not in {"linear_regression", "transformer"} and args.translator_checkpoint:
        checkpoint = torch.load(args.translator_checkpoint, map_location="cpu")
        translator.load_state_dict(checkpoint["translator_state_dict"], strict=False)
        logging.info(f"Loaded translator from {args.translator_checkpoint}")

    if translator_type != "transformer":
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
                norm_test_loader = norm_runtime.create_dataloader(
                    'test',
                    shuffle=False,
                    ram_cache=True,
                    subset_fraction=0.2 if debug_mode else None,
                    subset_seed=training_cfg["seed"],
                )
                evaluator.yaib_runtime = norm_runtime
                original_metrics = evaluator._evaluate_without_translator(norm_test_loader)
                evaluator.yaib_runtime = yaib_runtime
            translated_metrics = evaluator.translate_and_evaluate(test_loader, output_path)
            results = {"original": original_metrics, "translated": translated_metrics}
        else:
            results = evaluator.evaluate_original_vs_translated(test_loader, output_path)

    if results is None:
        raise RuntimeError("No evaluation results produced.")

    logging.info("=" * 80)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 80)
    logging.info("Original Test Data:")
    for metric, value in results["original"].items():
        logging.info("  %s: %.4f", metric, value)

    logging.info("Translated Test Data:")
    for metric, value in results["translated"].items():
        logging.info("  %s: %.4f", metric, value)

    logging.info("Difference:")
    for metric in results["original"].keys():
        diff = results["translated"][metric] - results["original"][metric]
        logging.info("  %s: %+0.4f", metric, diff)
    logging.info("=" * 80)

    if translator_type == "linear_regression" and lr_target_loader is not None:
        lr_metrics = _linear_regression_metrics(translator, test_loader, lr_target_loader)
        if lr_metrics:
            logging.info("Linear Regression Metrics (translated vs target):")
            for key, value in lr_metrics.items():
                logging.info("  %s: %.6f", key, value)

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
    eval_parser.add_argument("--export_full_sequence", default=True, action=argparse.BooleanOptionalAction, help="Export translated parquet with all non-padded timesteps (not just label mask). Default: true (use --no-export_full_sequence for label-mask only).")
    eval_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    train_eval_parser = subparsers.add_parser("train_and_eval", help="Train translator then translate and evaluate")
    train_eval_parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    train_eval_parser.add_argument("--output_parquet", type=str, required=True, help="Output path for translated parquet")
    train_eval_parser.add_argument("--translator_checkpoint", type=str, help="Path to translator checkpoint")
    train_eval_parser.add_argument("--export_full_sequence", default=True, action=argparse.BooleanOptionalAction, help="Export translated parquet with all non-padded timesteps (not just label mask). Default: true (use --no-export_full_sequence for label-mask only).")
    train_eval_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()

    # Load config early to get log_file setting
    _log_file = "run.log"
    if hasattr(args, "config") and args.config:
        try:
            _cfg = load_config(Path(args.config))
            _log_file = _cfg.get("output", {}).get("log_file", "run.log")
        except Exception:
            pass
    setup_logging(getattr(args, "verbose", False), log_file=_log_file)

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
