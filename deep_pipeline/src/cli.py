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
        # DA Baselines (DANN, CORAL, CoDATS)
        "lambda_coral": training.get("lambda_coral", 0.0),
        "discriminator_hidden_dim": training.get("discriminator_hidden_dim", 256),
        "discriminator_lr": training.get("discriminator_lr", 1e-4),
        "grl_schedule": training.get("grl_schedule", True),
        # Shared latent translator
        "pretrain_epochs": training.get("pretrain_epochs", 10),
        "lambda_align": training.get("lambda_align", 0.5),
        "lambda_recon": training.get("lambda_recon", 0.1),
        # Training data shuffling
        "shuffle": training.get("shuffle", False),
        # Negative subsampling (keep N negative stays, all positive stays)
        "negative_subsample_count": training.get("negative_subsample_count", 0),
        # MIMIC target task loss (for both delta and SL)
        "lambda_target_task": training.get("lambda_target_task", 0.0),
        # Latent label prediction head (SL only)
        "lambda_label_pred": training.get("lambda_label_pred", 0.0),
        # Cross-domain normalization: use MIMIC stats to normalize eICU
        "use_target_normalization": training.get("use_target_normalization", False),
        # Retrieval translator
        "k_neighbors": training.get("k_neighbors", 16),
        "retrieval_window": training.get("retrieval_window", 6),
        "n_cross_layers": training.get("n_cross_layers", 2),
        "output_mode": training.get("output_mode", "residual"),
        "memory_refresh_epochs": training.get("memory_refresh_epochs", 5),
        "window_stride": training.get("window_stride", None),
        "lambda_importance_reg": training.get("lambda_importance_reg", 0.01),
        "lambda_smooth": training.get("lambda_smooth", 0.1),
        # Feature gate (for delta/SL comparison)
        "feature_gate": training.get("feature_gate", False),
        # Importance regularization type (retrieval): "l1" or "entropy"
        "importance_reg_type": training.get("importance_reg_type", "l1"),
        # Cross-domain contrastive alignment (retrieval/SL)
        "lambda_contrastive_align": training.get("lambda_contrastive_align", 0.0),
        # Positive-weighted reconstruction (retrieval/SL)
        "recon_positive_boost": training.get("recon_positive_boost", 0.0),
        # LSTM-informed feature gate initialization
        "lstm_informed_gate": training.get("lstm_informed_gate", False),
        # Separate training seed (weight init, dropout, shuffle) from data split seed
        "training_seed": training.get("training_seed", None),
        # Task type: "classification" (default) or "regression" (LoS, KF)
        "task_type": training.get("task_type", "classification"),
        # V6: LR scheduling
        "lr_scheduler": training.get("lr_scheduler", None),           # "cosine" | "plateau" | null
        "lr_min": training.get("lr_min", 0.0),                        # eta_min for cosine
        "lr_warmup_epochs": training.get("lr_warmup_epochs", 0),      # linear warmup
        # V6: Gradient clipping
        "grad_clip_norm": training.get("grad_clip_norm", 0.0),        # max grad norm (0=disabled)
        # V6: Self-retrieval Phase 1
        "phase1_self_retrieval": training.get("phase1_self_retrieval", False),
        "phase1_memory_refresh_epochs": training.get("phase1_memory_refresh_epochs", None),
        # V6: Gradient accumulation
        "accumulate_grad_batches": training.get("accumulate_grad_batches", 1),
        # V7 (NeurIPS): TCR — target context reconstruction through full retrieval pipeline
        "lambda_tcr": training.get("lambda_tcr", 0.0),
        # V7 (NeurIPS): FJS — initialize FeatureGate from frozen model Jacobian sensitivity
        "jacobian_init": training.get("jacobian_init", False),
        # V7: Adaptive memory bank refresh
        "adaptive_refresh": training.get("adaptive_refresh", False),
        "adaptive_refresh_patience": training.get("adaptive_refresh_patience", 2),
        "adaptive_refresh_metric": training.get("adaptive_refresh_metric", "align"),
        "adaptive_refresh_min_delta": training.get("adaptive_refresh_min_delta", 0.001),
        # DA Baselines V2: CLUDA (ICLR 2023)
        "lambda_cluda_temporal": training.get("lambda_cluda_temporal", 0.0),
        "lambda_cluda_contextual": training.get("lambda_cluda_contextual", 0.0),
        "cluda_temperature": training.get("cluda_temperature", 0.07),
        "cluda_k_neighbors": training.get("cluda_k_neighbors", 5),
        # DA Baselines V2: RAINCOAT (ICML 2023)
        "lambda_raincoat_temporal": training.get("lambda_raincoat_temporal", 0.0),
        "lambda_raincoat_freq": training.get("lambda_raincoat_freq", 0.0),
        "raincoat_sinkhorn_eps": training.get("raincoat_sinkhorn_eps", 0.001),
        "raincoat_sinkhorn_iters": training.get("raincoat_sinkhorn_iters", 1000),
        # DA Baselines V2: ACON (NeurIPS 2024)
        "lambda_acon_temporal": training.get("lambda_acon_temporal", 0.0),
        "lambda_acon_freq": training.get("lambda_acon_freq", 0.0),
        "lambda_acon_cross": training.get("lambda_acon_cross", 0.0),
        "acon_freq_hidden_dim": training.get("acon_freq_hidden_dim", 128),
        # V8: Class-Conditional Retrieval (CCR)
        "ccr_alpha": training.get("ccr_alpha", 0.0),
        # V8: Adaptive Fidelity Scheduling (AFS)
        "fidelity_schedule": training.get("fidelity_schedule", None),
        "fidelity_decay_start_epoch": training.get("fidelity_decay_start_epoch", 5),
        "fidelity_decay_end_epoch": training.get("fidelity_decay_end_epoch", 40),
        "fidelity_min_ratio": training.get("fidelity_min_ratio", 0.1),
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


def _apply_negative_subsampling(loader, n_keep_negative, seed=42):
    """Subsample negative stays from training DataLoader to increase effective label density.

    Keeps ALL positive stays and randomly samples n_keep_negative negative stays.
    Returns a new DataLoader over a Subset of the original dataset.
    """
    from torch.utils.data import Subset

    stay_labels = _get_stay_labels(loader.dataset)
    pos_indices = [i for i, lbl in enumerate(stay_labels) if lbl == 1]
    neg_indices = [i for i, lbl in enumerate(stay_labels) if lbl == 0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    if n_keep_negative >= n_neg:
        logging.info(
            "Negative subsampling: requested %d but only %d negatives exist, keeping all",
            n_keep_negative, n_neg,
        )
        return loader

    rng = np.random.RandomState(seed)
    sampled_neg = rng.choice(neg_indices, size=n_keep_negative, replace=False).tolist()
    keep_indices = sorted(pos_indices + sampled_neg)

    subset_dataset = Subset(loader.dataset, keep_indices)
    new_n_pos = n_pos
    new_n_neg = n_keep_negative
    new_total = new_n_pos + new_n_neg
    logging.info(
        "Negative subsampling: %d pos + %d neg = %d stays (from %d total, removed %d negatives)",
        new_n_pos, new_n_neg, new_total, n_pos + n_neg, n_neg - n_keep_negative,
    )
    logging.info(
        "  Effective per-stay positive rate: %.1f%% (was %.1f%%)",
        100 * new_n_pos / new_total, 100 * n_pos / (n_pos + n_neg),
    )

    return DataLoader(
        subset_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        drop_last=loader.drop_last,
        pin_memory=getattr(loader, "pin_memory", False),
        collate_fn=loader.collate_fn,
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
    debug_fraction = config.get("debug_fraction", 0.2)
    if debug_mode:
        training_cfg["epochs"] = min(training_cfg.get("epochs", 30), 1)
        if training_cfg.get("pretrain_epochs", 0) > 0:
            training_cfg["pretrain_epochs"] = 1
        logging.info("DEBUG MODE: epochs capped to %d, pretrain_epochs capped to %d, data fraction=%.0f%%",
                      training_cfg["epochs"], training_cfg.get("pretrain_epochs", 0), debug_fraction * 100)
    translator_type = _get_translator_type(config)

    logging.info("=== Training Configuration ===")
    logging.info("  debug: %s (fraction: %s)", debug_mode, debug_fraction if debug_mode else "N/A")
    logging.info("  translator_type: %s", translator_type)
    for k, v in sorted(training_cfg.items()):
        logging.info("  %s: %s", k, v)
    translator_cfg = _get_translator_config(config)
    for k, v in sorted(translator_cfg.items()):
        logging.info("  translator.%s: %s", k, v)
    logging.info("==============================")

    # training_seed controls weight init, dropout, shuffle ordering.
    # seed (YAIB split seed) is used separately for _build_runtime_from_config.
    training_seed = training_cfg.get("training_seed") or training_cfg["seed"]
    logging.info("  data_split_seed: %s, training_seed: %s", training_cfg["seed"], training_seed)
    import random
    random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    np.random.seed(training_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if translator_type in ("transformer", "affine"):
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
            subset_fraction=debug_fraction if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )
        val_loader = yaib_runtime.create_dataloader(
            'val',
            shuffle=False,
            ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
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

        task_type = training_cfg.get("task_type", "classification")

        neg_subsample = training_cfg.get("negative_subsample_count", 0)
        if neg_subsample > 0 and task_type != "regression":
            train_loader = _apply_negative_subsampling(
                train_loader, neg_subsample, seed=config.get("seed", 2222)
            )

        oversampling_factor = training_cfg.get("oversampling_factor", 0)
        vlb = training_cfg.get("variable_length_batching", False)
        if not vlb and oversampling_factor > 0 and task_type != "regression":
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
                subset_fraction=debug_fraction if debug_mode else None,
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

        # Variable-length bucket batching (replaces oversampling if both enabled)
        if vlb:
            from .core.bucket_batching import apply_bucket_batching
            logging.info("Applying variable-length bucket batching...")
            train_loader = apply_bucket_batching(
                train_loader,
                batch_size=training_cfg["batch_size"],
                oversampling_factor=oversampling_factor,
                shuffle=True,
                drop_last=True,
            )
            if target_train_loader:
                target_train_loader = apply_bucket_batching(
                    target_train_loader,
                    batch_size=training_cfg["batch_size"],
                    oversampling_factor=0,
                    shuffle=True,
                    drop_last=True,
                )
            val_loader = apply_bucket_batching(
                val_loader,
                batch_size=training_cfg["batch_size"],
                oversampling_factor=0,
                shuffle=False,
                drop_last=False,
            )
        elif oversampling_factor == 0:
            # No VLB and no oversampling — optionally enable shuffling for training
            should_shuffle = training_cfg.get("shuffle", False)
            if should_shuffle:
                train_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    shuffle=True,
                    num_workers=train_loader.num_workers,
                    drop_last=train_loader.drop_last,
                    pin_memory=getattr(train_loader, "pin_memory", False),
                    collate_fn=train_loader.collate_fn,
                )
                logging.info("Training with shuffle=True (config)")
            else:
                logging.info("Training with shuffle=False (default)")

        # Cross-domain normalization
        use_target_norm = training_cfg.get("use_target_normalization", False)
        renorm_params = None
        if use_target_norm and target_train_loader is not None:
            from .core.train import compute_renorm_params
            renorm_params = compute_renorm_params(
                train_loader, target_train_loader, schema_resolver,
                config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            )

        bounds_csv = _get_bounds_csv(config) or translator_cfg.get("bounds_csv", "")
        if not bounds_csv:
            raise ValueError("bounds_csv must be provided for transformer translator.")

        if translator_type == "affine":
            from .core.translator import AffineTranslator
            translator = AffineTranslator(
                num_features=len(schema_resolver.indices.dynamic),
                static_dim=len(static_features),
            )
        else:
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
            training_config=training_cfg,
        )
        if renorm_params is not None:
            trainer.set_renorm_params(*renorm_params)
        trainer.train(
            epochs=training_cfg["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
        )
        logging.info("Transformer translator training completed")
        return

    elif translator_type == "finetune_lstm":
        # -----------------------------------------------------------------
        # Fine-tuned LSTM upper bound: unfreeze MIMIC LSTM, retrain on eICU.
        # Uses IdentityDATranslator (data passes through unchanged) + unfrozen
        # baseline.  This is an UPPER BOUND that violates the frozen constraint.
        # -----------------------------------------------------------------
        from .baselines.components import IdentityDATranslator

        translator_cfg = _get_translator_config(config)
        yaib_runtime = _build_runtime_from_config(
            config,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        yaib_runtime.load_data()
        # Load baseline UNFROZEN — the key difference
        yaib_runtime.load_baseline_model(freeze=False)

        train_loader = yaib_runtime.create_dataloader(
            'train',
            shuffle=False,
            ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )
        val_loader = yaib_runtime.create_dataloader(
            'val',
            shuffle=False,
            ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
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
            else:
                train_loader = _augment_loader_with_zero_static(train_loader, static_features)
                val_loader = _augment_loader_with_zero_static(val_loader, static_features)

        task_type = training_cfg.get("task_type", "classification")

        neg_subsample = training_cfg.get("negative_subsample_count", 0)
        if neg_subsample > 0 and task_type != "regression":
            train_loader = _apply_negative_subsampling(
                train_loader, neg_subsample, seed=config.get("seed", 2222)
            )

        oversampling_factor = training_cfg.get("oversampling_factor", 0)
        vlb = training_cfg.get("variable_length_batching", False)
        if not vlb and oversampling_factor > 0 and task_type != "regression":
            train_loader = _apply_oversampling(train_loader, oversampling_factor)

        schema_resolver = SchemaResolver(
            feature_names=feature_names,
            dynamic_features=config["vars"]["DYNAMIC"],
            static_features=static_features,
            allow_missing_static=not static_in_features,
            missing_prefix=translator_cfg.get("missing_prefix", "MissingIndicator_"),
            group_col=group_col,
        )

        # Load MIMIC target data for cross-domain normalization
        target_train_loader = None
        target_data_dir = config.get("target_data_dir")
        if target_data_dir:
            logging.info("Loading target (MIMIC) data from %s for cross-domain normalization", target_data_dir)
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
                subset_fraction=debug_fraction if debug_mode else None,
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

        # Variable-length bucket batching
        if vlb:
            from .core.bucket_batching import apply_bucket_batching
            logging.info("Applying variable-length bucket batching...")
            train_loader = apply_bucket_batching(
                train_loader,
                batch_size=training_cfg["batch_size"],
                oversampling_factor=oversampling_factor,
                shuffle=True,
                drop_last=True,
            )
            if target_train_loader:
                target_train_loader = apply_bucket_batching(
                    target_train_loader,
                    batch_size=training_cfg["batch_size"],
                    oversampling_factor=0,
                    shuffle=True,
                    drop_last=True,
                )
            val_loader = apply_bucket_batching(
                val_loader,
                batch_size=training_cfg["batch_size"],
                oversampling_factor=0,
                shuffle=False,
                drop_last=False,
            )
        elif oversampling_factor == 0:
            should_shuffle = training_cfg.get("shuffle", False)
            if should_shuffle:
                train_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    shuffle=True,
                    num_workers=train_loader.num_workers,
                    drop_last=train_loader.drop_last,
                    pin_memory=getattr(train_loader, "pin_memory", False),
                    collate_fn=train_loader.collate_fn,
                )

        # Cross-domain normalization
        use_target_norm = training_cfg.get("use_target_normalization", False)
        renorm_params = None
        if use_target_norm and target_train_loader is not None:
            from .core.train import compute_renorm_params
            renorm_params = compute_renorm_params(
                train_loader, target_train_loader, schema_resolver,
                config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            )

        bounds_csv = _get_bounds_csv(config) or translator_cfg.get("bounds_csv", "")
        if not bounds_csv:
            raise ValueError("bounds_csv must be provided for finetune_lstm baseline.")

        # Identity translator: data passes through unchanged
        translator = IdentityDATranslator()

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
            training_config=training_cfg,
            freeze_baseline=False,  # KEY: LSTM is trainable
        )
        if renorm_params is not None:
            trainer.set_renorm_params(*renorm_params)
        trainer.train(
            epochs=training_cfg["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
        )
        logging.info("Fine-tuned LSTM training completed (upper bound baseline)")
        return

    elif translator_type == "shared_latent":
        from .core.latent_translator import SharedLatentTranslator
        from .core.train import LatentTranslatorTrainer

        translator_cfg = _get_translator_config(config)
        yaib_runtime = _build_runtime_from_config(
            config,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        yaib_runtime.load_data()
        train_loader = yaib_runtime.create_dataloader(
            'train', shuffle=False, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )
        val_loader = yaib_runtime.create_dataloader(
            'val', shuffle=False, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
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
                    raise ValueError("Static features missing; provide paths.static_recipe.")
                static_df = load_static_with_recipe(
                    data_dir=Path(config["data_dir"]),
                    file_names=config["file_names"],
                    group_col=group_col,
                    static_features=static_features,
                    recipe_path=Path(static_recipe),
                )
                train_loader = _augment_loader_with_static(train_loader, static_df, group_col, static_features)
                val_loader = _augment_loader_with_static(val_loader, static_df, group_col, static_features)
            else:
                train_loader = _augment_loader_with_zero_static(train_loader, static_features)
                val_loader = _augment_loader_with_zero_static(val_loader, static_features)

        task_type = training_cfg.get("task_type", "classification")

        neg_subsample = training_cfg.get("negative_subsample_count", 0)
        if neg_subsample > 0 and task_type != "regression":
            train_loader = _apply_negative_subsampling(
                train_loader, neg_subsample, seed=config.get("seed", 2222)
            )

        oversampling_factor = training_cfg.get("oversampling_factor", 0)
        vlb = training_cfg.get("variable_length_batching", False)
        if not vlb and oversampling_factor > 0 and task_type != "regression":
            train_loader = _apply_oversampling(train_loader, oversampling_factor)

        schema_resolver = SchemaResolver(
            feature_names=feature_names,
            dynamic_features=config["vars"]["DYNAMIC"],
            static_features=static_features,
            allow_missing_static=not static_in_features,
            missing_prefix=translator_cfg.get("missing_prefix", "MissingIndicator_"),
            group_col=group_col,
        )

        # Target (MIMIC) data is REQUIRED for shared latent
        target_data_dir = config.get("target_data_dir")
        if not target_data_dir:
            raise ValueError("shared_latent translator requires target_data_dir in config")
        logging.info("Loading target (MIMIC) data from %s", target_data_dir)
        target_runtime = _build_runtime_from_config(
            config,
            data_dir_override=target_data_dir,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        target_runtime.load_data()
        target_train_loader = target_runtime.create_dataloader(
            'train', shuffle=True, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )
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

        # Variable-length bucket batching (replaces oversampling if both enabled)
        if vlb:
            from .core.bucket_batching import apply_bucket_batching
            logging.info("Applying variable-length bucket batching...")
            train_loader = apply_bucket_batching(
                train_loader,
                batch_size=training_cfg["batch_size"],
                oversampling_factor=oversampling_factor,
                shuffle=True,
                drop_last=True,
            )
            target_train_loader = apply_bucket_batching(
                target_train_loader,
                batch_size=training_cfg["batch_size"],
                oversampling_factor=0,
                shuffle=True,
                drop_last=True,
            )
            val_loader = apply_bucket_batching(
                val_loader,
                batch_size=training_cfg["batch_size"],
                oversampling_factor=0,
                shuffle=False,
                drop_last=False,
            )
        elif oversampling_factor == 0:
            # No VLB and no oversampling — optionally enable shuffling for training
            should_shuffle = training_cfg.get("shuffle", False)
            if should_shuffle:
                train_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    shuffle=True,
                    num_workers=train_loader.num_workers,
                    drop_last=train_loader.drop_last,
                    pin_memory=getattr(train_loader, "pin_memory", False),
                    collate_fn=train_loader.collate_fn,
                )
                logging.info("Training with shuffle=True (config)")
            else:
                logging.info("Training with shuffle=False (default)")

        # Cross-domain normalization
        use_target_norm = training_cfg.get("use_target_normalization", False)
        renorm_params = None
        if use_target_norm:
            from .core.train import compute_renorm_params
            renorm_params = compute_renorm_params(
                train_loader, target_train_loader, schema_resolver,
                config.get("device", "cuda"),
            )

        bounds_csv = _get_bounds_csv(config) or translator_cfg.get("bounds_csv", "")
        if not bounds_csv:
            raise ValueError("bounds_csv must be provided for shared_latent translator.")

        translator = SharedLatentTranslator(
            num_features=len(schema_resolver.indices.dynamic),
            d_latent=translator_cfg.get("d_latent", 64),
            d_model=translator_cfg.get("d_model", 128),
            d_time=translator_cfg.get("d_time", 16),
            n_enc_layers=translator_cfg.get("n_enc_layers", 3),
            n_dec_layers=translator_cfg.get("n_dec_layers", 2),
            n_heads=translator_cfg.get("n_heads", 8),
            d_ff=translator_cfg.get("d_ff", 512),
            dropout=translator_cfg.get("dropout", 0.2),
            out_dropout=translator_cfg.get("out_dropout", 0.1),
            static_dim=len(static_features),
            temporal_attention_mode=_get_temporal_attention_mode(translator_cfg),
            temporal_attention_window=translator_cfg.get("temporal_attention_window", 0),
        )

        trainer = LatentTranslatorTrainer(
            yaib_runtime=yaib_runtime,
            translator=translator,
            schema_resolver=schema_resolver,
            bounds_csv=Path(bounds_csv),
            target_train_loader=target_train_loader,
            learning_rate=training_cfg["lr"],
            lambda_align=training_cfg.get("lambda_align", 0.5),
            lambda_recon=training_cfg.get("lambda_recon", 0.1),
            lambda_range=training_cfg.get("lambda_range", 0.5),
            pretrain_epochs=training_cfg.get("pretrain_epochs", 10),
            early_stopping_patience=training_cfg["early_stopping_patience"],
            best_metric=training_cfg["best_metric"],
            run_dir=Path(output_cfg["run_dir"]),
            device=config.get("device", "cuda"),
            training_config=training_cfg,
        )
        if renorm_params is not None:
            trainer.set_renorm_params(*renorm_params)
        trainer.train(
            epochs=training_cfg["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
        )
        return

    elif translator_type == "retrieval":
        from .core.retrieval_translator import RetrievalTranslator
        from .core.train import RetrievalTranslatorTrainer

        translator_cfg = _get_translator_config(config)
        yaib_runtime = _build_runtime_from_config(
            config,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        yaib_runtime.load_data()
        train_loader = yaib_runtime.create_dataloader(
            'train', shuffle=False, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )
        val_loader = yaib_runtime.create_dataloader(
            'val', shuffle=False, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
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
                    raise ValueError("Static features missing; provide paths.static_recipe.")
                static_df = load_static_with_recipe(
                    data_dir=Path(config["data_dir"]),
                    file_names=config["file_names"],
                    group_col=group_col,
                    static_features=static_features,
                    recipe_path=Path(static_recipe),
                )
                train_loader = _augment_loader_with_static(train_loader, static_df, group_col, static_features)
                val_loader = _augment_loader_with_static(val_loader, static_df, group_col, static_features)
            else:
                train_loader = _augment_loader_with_zero_static(train_loader, static_features)
                val_loader = _augment_loader_with_zero_static(val_loader, static_features)

        task_type = training_cfg.get("task_type", "classification")

        neg_subsample = training_cfg.get("negative_subsample_count", 0)
        if neg_subsample > 0 and task_type != "regression":
            train_loader = _apply_negative_subsampling(
                train_loader, neg_subsample, seed=config.get("seed", 2222)
            )

        oversampling_factor = training_cfg.get("oversampling_factor", 0)
        vlb = training_cfg.get("variable_length_batching", False)
        if not vlb and oversampling_factor > 0 and task_type != "regression":
            train_loader = _apply_oversampling(train_loader, oversampling_factor)

        schema_resolver = SchemaResolver(
            feature_names=feature_names,
            dynamic_features=config["vars"]["DYNAMIC"],
            static_features=static_features,
            allow_missing_static=not static_in_features,
            missing_prefix=translator_cfg.get("missing_prefix", "MissingIndicator_"),
            group_col=group_col,
        )

        # Target (MIMIC) data is REQUIRED for retrieval
        target_data_dir = config.get("target_data_dir")
        if not target_data_dir:
            raise ValueError("retrieval translator requires target_data_dir in config")
        logging.info("Loading target (MIMIC) data from %s", target_data_dir)
        target_runtime = _build_runtime_from_config(
            config,
            data_dir_override=target_data_dir,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        target_runtime.load_data()
        target_train_loader = target_runtime.create_dataloader(
            'train', shuffle=True, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )
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

        # Variable-length bucket batching
        if vlb:
            from .core.bucket_batching import apply_bucket_batching
            logging.info("Applying variable-length bucket batching...")
            train_loader = apply_bucket_batching(
                train_loader,
                batch_size=training_cfg["batch_size"],
                oversampling_factor=oversampling_factor,
                shuffle=True,
                drop_last=True,
            )
            target_train_loader = apply_bucket_batching(
                target_train_loader,
                batch_size=training_cfg["batch_size"],
                oversampling_factor=0,
                shuffle=True,
                drop_last=True,
            )
            val_loader = apply_bucket_batching(
                val_loader,
                batch_size=training_cfg["batch_size"],
                oversampling_factor=0,
                shuffle=False,
                drop_last=False,
            )
        elif oversampling_factor == 0:
            should_shuffle = training_cfg.get("shuffle", False)
            if should_shuffle:
                train_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    shuffle=True,
                    num_workers=train_loader.num_workers,
                    drop_last=train_loader.drop_last,
                    pin_memory=getattr(train_loader, "pin_memory", False),
                    collate_fn=train_loader.collate_fn,
                )
                logging.info("Training with shuffle=True (config)")
            else:
                logging.info("Training with shuffle=False (default)")

        # Cross-domain normalization
        use_target_norm = training_cfg.get("use_target_normalization", False)
        renorm_params = None
        if use_target_norm:
            from .core.train import compute_renorm_params
            renorm_params = compute_renorm_params(
                train_loader, target_train_loader, schema_resolver,
                config.get("device", "cuda"),
            )

        bounds_csv = _get_bounds_csv(config) or translator_cfg.get("bounds_csv", "")
        if not bounds_csv:
            raise ValueError("bounds_csv must be provided for retrieval translator.")

        translator = RetrievalTranslator(
            num_features=len(schema_resolver.indices.dynamic),
            d_latent=translator_cfg.get("d_latent", 128),
            d_model=translator_cfg.get("d_model", 128),
            d_time=translator_cfg.get("d_time", 16),
            n_enc_layers=translator_cfg.get("n_enc_layers", 4),
            n_dec_layers=translator_cfg.get("n_dec_layers", 2),
            n_cross_layers=training_cfg.get("n_cross_layers", 2),
            n_heads=translator_cfg.get("n_heads", 8),
            d_ff=translator_cfg.get("d_ff", 512),
            dropout=translator_cfg.get("dropout", 0.2),
            out_dropout=translator_cfg.get("out_dropout", 0.1),
            static_dim=len(static_features),
            temporal_attention_mode=_get_temporal_attention_mode(translator_cfg),
            temporal_attention_window=translator_cfg.get("temporal_attention_window", 0),
            output_mode=training_cfg.get("output_mode", "residual"),
        )

        trainer = RetrievalTranslatorTrainer(
            yaib_runtime=yaib_runtime,
            translator=translator,
            schema_resolver=schema_resolver,
            bounds_csv=Path(bounds_csv),
            target_train_loader=target_train_loader,
            learning_rate=training_cfg["lr"],
            lambda_recon=training_cfg.get("lambda_recon", 0.1),
            lambda_range=training_cfg.get("lambda_range", 0.5),
            lambda_smooth=training_cfg.get("lambda_smooth", 0.1),
            lambda_importance_reg=training_cfg.get("lambda_importance_reg", 0.01),
            lambda_align=training_cfg.get("lambda_align", 0.0),
            pretrain_epochs=training_cfg.get("pretrain_epochs", 10),
            k_neighbors=training_cfg.get("k_neighbors", 16),
            retrieval_window=training_cfg.get("retrieval_window", 6),
            memory_refresh_epochs=training_cfg.get("memory_refresh_epochs", 5),
            early_stopping_patience=training_cfg["early_stopping_patience"],
            best_metric=training_cfg["best_metric"],
            run_dir=Path(output_cfg["run_dir"]),
            device=config.get("device", "cuda"),
            training_config=training_cfg,
        )
        if renorm_params is not None:
            trainer.set_renorm_params(*renorm_params)
        trainer.train(
            epochs=training_cfg["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
        )
        return

    elif translator_type == "linear_regression":
        translator, _, train_loader, target_loader = _prepare_linear_regression(
            config,
            split=DataSplit.train,
            shuffle=False,
            debug_mode=debug_mode,
            debug_fraction=debug_fraction,
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

    elif translator_type in ("dann", "coral", "codats", "cluda", "raincoat", "acon", "stats_only"):
        from .baselines.trainer import DABaselineTrainer

        translator_cfg = _get_translator_config(config)
        yaib_runtime = _build_runtime_from_config(
            config,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        yaib_runtime.load_data()
        train_loader = yaib_runtime.create_dataloader(
            'train', shuffle=False, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )
        val_loader = yaib_runtime.create_dataloader(
            'val', shuffle=False, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
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
                    raise ValueError("Static features missing; provide paths.static_recipe.")
                static_df = load_static_with_recipe(
                    data_dir=Path(config["data_dir"]),
                    file_names=config["file_names"],
                    group_col=group_col,
                    static_features=static_features,
                    recipe_path=Path(static_recipe),
                )
                train_loader = _augment_loader_with_static(train_loader, static_df, group_col, static_features)
                val_loader = _augment_loader_with_static(val_loader, static_df, group_col, static_features)
            else:
                train_loader = _augment_loader_with_zero_static(train_loader, static_features)
                val_loader = _augment_loader_with_zero_static(val_loader, static_features)

        task_type = training_cfg.get("task_type", "classification")
        neg_subsample = training_cfg.get("negative_subsample_count", 0)
        if neg_subsample > 0 and task_type != "regression":
            train_loader = _apply_negative_subsampling(
                train_loader, neg_subsample, seed=config.get("seed", 2222)
            )

        oversampling_factor = training_cfg.get("oversampling_factor", 0)
        vlb = training_cfg.get("variable_length_batching", False)
        if not vlb and oversampling_factor > 0 and task_type != "regression":
            train_loader = _apply_oversampling(train_loader, oversampling_factor)

        schema_resolver = SchemaResolver(
            feature_names=feature_names,
            dynamic_features=config["vars"]["DYNAMIC"],
            static_features=static_features,
            allow_missing_static=not static_in_features,
            missing_prefix=translator_cfg.get("missing_prefix", "MissingIndicator_"),
            group_col=group_col,
        )

        # Target (MIMIC) data is REQUIRED for DA baselines
        target_data_dir = config.get("target_data_dir")
        if not target_data_dir:
            raise ValueError(f"{translator_type} baseline requires target_data_dir in config")
        logging.info("Loading target (MIMIC) data from %s", target_data_dir)
        target_runtime = _build_runtime_from_config(
            config,
            data_dir_override=target_data_dir,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        target_runtime.load_data()
        target_train_loader = target_runtime.create_dataloader(
            'train', shuffle=True, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )
        # Augment target with static features
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

        # Variable-length bucket batching
        if vlb:
            from .core.bucket_batching import apply_bucket_batching
            train_loader = apply_bucket_batching(
                train_loader, batch_size=training_cfg["batch_size"],
                oversampling_factor=oversampling_factor, shuffle=True, drop_last=True,
            )
            target_train_loader = apply_bucket_batching(
                target_train_loader, batch_size=training_cfg["batch_size"],
                oversampling_factor=0, shuffle=True, drop_last=True,
            )
            val_loader = apply_bucket_batching(
                val_loader, batch_size=training_cfg["batch_size"],
                oversampling_factor=0, shuffle=False, drop_last=False,
            )
        elif oversampling_factor == 0:
            should_shuffle = training_cfg.get("shuffle", False)
            if should_shuffle:
                train_loader = DataLoader(
                    train_loader.dataset, batch_size=train_loader.batch_size,
                    shuffle=True, num_workers=train_loader.num_workers,
                    drop_last=train_loader.drop_last,
                    pin_memory=getattr(train_loader, "pin_memory", False),
                    collate_fn=train_loader.collate_fn,
                )

        # Cross-domain normalization
        use_target_norm = training_cfg.get("use_target_normalization", False)
        renorm_params = None
        if use_target_norm and target_train_loader is not None:
            from .core.train import compute_renorm_params
            renorm_params = compute_renorm_params(
                train_loader, target_train_loader, schema_resolver,
                config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            )

        bounds_csv = _get_bounds_csv(config) or translator_cfg.get("bounds_csv", "")
        if not bounds_csv:
            raise ValueError("bounds_csv must be provided for DA baselines.")

        # Build translator backbone
        if translator_type == "codats":
            from .baselines.codats_backbone import CoDATS1DCNN
            translator = CoDATS1DCNN(
                num_features=len(schema_resolver.indices.dynamic),
                d_model=translator_cfg.get("d_model", 128),
                n_conv_layers=translator_cfg.get("n_conv_layers", 3),
                kernel_size=translator_cfg.get("kernel_size", 5),
                dropout=translator_cfg.get("dropout", 0.2),
                temporal_attention_mode=_get_temporal_attention_mode(translator_cfg),
            )
        elif translator_type == "stats_only":
            from .baselines.components import IdentityDATranslator
            translator = IdentityDATranslator()
            # Stats-only: save renorm params + identity translator and return
            run_dir = Path(output_cfg["run_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "epoch": 0,
                "translator_state_dict": translator.state_dict(),
                "val_metrics": {},
                "train_metrics": {},
                "renorm_scale": renorm_params[0] if renorm_params else None,
                "renorm_offset": renorm_params[1] if renorm_params else None,
                "da_method": "stats_only",
            }
            torch.save(checkpoint, run_dir / "best_translator.pt")
            torch.save(checkpoint, run_dir / "latest_checkpoint.pt")
            logging.info("[stats_only] Saved renorm-only checkpoint to %s (no training)", run_dir)
            return
        else:
            # DANN, CORAL, CLUDA, RAINCOAT, ACON all use EHRTranslator backbone
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

        trainer = DABaselineTrainer(
            yaib_runtime=yaib_runtime,
            translator=translator,
            schema_resolver=schema_resolver,
            bounds_csv=Path(bounds_csv),
            da_method=translator_type,
            target_train_loader=target_train_loader,
            learning_rate=training_cfg["lr"],
            lambda_fidelity=training_cfg["lambda_fidelity"],
            lambda_range=training_cfg["lambda_range"],
            lambda_adversarial=training_cfg["lambda_adversarial"],
            lambda_coral=training_cfg.get("lambda_coral", 0.0),
            discriminator_hidden_dim=training_cfg.get("discriminator_hidden_dim", 256),
            discriminator_lr=training_cfg.get("discriminator_lr", 1e-4),
            grl_schedule=training_cfg.get("grl_schedule", True),
            early_stopping_patience=training_cfg["early_stopping_patience"],
            best_metric=training_cfg["best_metric"],
            run_dir=Path(output_cfg["run_dir"]),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            training_config=training_cfg,
        )
        if renorm_params is not None:
            trainer.set_renorm_params(*renorm_params)
        trainer.train(
            epochs=training_cfg["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
        )
        logging.info("%s baseline training completed", translator_type.upper())
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
        subset_fraction=debug_fraction if debug_mode else None,
        subset_seed=training_cfg["seed"],
    )
    val_loader = yaib_runtime.create_dataloader(
        'val',
        shuffle=False,
        ram_cache=True,
        subset_fraction=debug_fraction if debug_mode else None,
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
    debug_fraction = config.get("debug_fraction", 0.2)
    eval_baseline_with_target_norm = config.get("eval_baseline_with_target_normalization", False)

    lr_target_loader = None
    results = None

    if translator_type in ("transformer", "affine"):
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
            subset_fraction=debug_fraction if debug_mode else None,
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

        if translator_type == "affine":
            from .core.translator import AffineTranslator
            translator = AffineTranslator(
                num_features=len(schema_resolver.indices.dynamic),
                static_dim=len(static_features),
            )
        else:
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
        renorm_scale = None
        renorm_offset = None
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            translator.load_state_dict(checkpoint["translator_state_dict"], strict=False)
            renorm_scale = checkpoint.get("renorm_scale")
            renorm_offset = checkpoint.get("renorm_offset")
            logging.info("Loaded transformer translator from %s", checkpoint_path)
            if renorm_scale is not None:
                logging.info("Cross-domain renormalization params loaded from checkpoint")
        else:
            logging.warning("No transformer checkpoint found at %s", checkpoint_path)

        evaluator = TransformerTranslatorEvaluator(
            yaib_runtime=yaib_runtime,
            translator=translator,
            schema_resolver=schema_resolver,
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            renorm_scale=renorm_scale,
            renorm_offset=renorm_offset,
            task_type=training_cfg.get("task_type", "classification"),
        )
        output_path = Path(args.output_parquet)
        sample_dir = translator_cfg.get(
            "sample_dir", str(Path(__file__).resolve().parent.parent / "data" / "YAIB" / "translation_samples")
        )
        sample_size = int(translator_cfg.get("sample_size", 1000))
        results = evaluator.evaluate_original_vs_translated(
            test_loader,
            output_path,
            sample_output_dir=Path(sample_dir) if sample_dir else None,
            sample_size=sample_size,
            export_full_sequence=getattr(args, "export_full_sequence", True),
        )

    elif translator_type == "shared_latent":
        from .core.latent_translator import SharedLatentTranslator

        translator_cfg = _get_translator_config(config)
        yaib_runtime = _build_runtime_from_config(
            config,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        yaib_runtime.load_data()
        test_loader = yaib_runtime.create_dataloader(
            'test', shuffle=False, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
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
                if static_recipe:
                    static_df = load_static_with_recipe(
                        data_dir=Path(config["data_dir"]),
                        file_names=config["file_names"],
                        group_col=group_col,
                        static_features=static_features,
                        recipe_path=Path(static_recipe),
                    )
                    test_loader = _augment_loader_with_static(test_loader, static_df, group_col, static_features)
                else:
                    test_loader = _augment_loader_with_zero_static(test_loader, static_features)
            else:
                test_loader = _augment_loader_with_zero_static(test_loader, static_features)

        schema_resolver = SchemaResolver(
            feature_names=feature_names,
            dynamic_features=config["vars"]["DYNAMIC"],
            static_features=static_features,
            allow_missing_static=not static_in_features,
            missing_prefix=translator_cfg.get("missing_prefix", "MissingIndicator_"),
            group_col=group_col,
        )

        translator = SharedLatentTranslator(
            num_features=len(schema_resolver.indices.dynamic),
            d_latent=translator_cfg.get("d_latent", 64),
            d_model=translator_cfg.get("d_model", 128),
            d_time=translator_cfg.get("d_time", 16),
            n_enc_layers=translator_cfg.get("n_enc_layers", 3),
            n_dec_layers=translator_cfg.get("n_dec_layers", 2),
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
        renorm_scale = None
        renorm_offset = None
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            translator.load_state_dict(checkpoint["translator_state_dict"], strict=False)
            renorm_scale = checkpoint.get("renorm_scale")
            renorm_offset = checkpoint.get("renorm_offset")
            logging.info("Loaded shared_latent translator from %s", checkpoint_path)
            if renorm_scale is not None:
                logging.info("Cross-domain renormalization params loaded from checkpoint")
        else:
            logging.warning("No shared_latent checkpoint found at %s", checkpoint_path)

        evaluator = TransformerTranslatorEvaluator(
            yaib_runtime=yaib_runtime,
            translator=translator,
            schema_resolver=schema_resolver,
            device=config.get("device", "cuda"),
            renorm_scale=renorm_scale,
            renorm_offset=renorm_offset,
            task_type=training_cfg.get("task_type", "classification"),
        )
        output_path = Path(args.output_parquet)
        results = evaluator.evaluate_original_vs_translated(
            test_loader, output_path,
            export_full_sequence=getattr(args, "export_full_sequence", True),
        )

    elif translator_type == "retrieval":
        from .core.retrieval_translator import RetrievalTranslator

        translator_cfg = _get_translator_config(config)
        yaib_runtime = _build_runtime_from_config(
            config,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        yaib_runtime.load_data()
        test_loader = yaib_runtime.create_dataloader(
            'test', shuffle=False, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
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
                if static_recipe:
                    static_df = load_static_with_recipe(
                        data_dir=Path(config["data_dir"]),
                        file_names=config["file_names"],
                        group_col=group_col,
                        static_features=static_features,
                        recipe_path=Path(static_recipe),
                    )
                    test_loader = _augment_loader_with_static(test_loader, static_df, group_col, static_features)
                else:
                    test_loader = _augment_loader_with_zero_static(test_loader, static_features)
            else:
                test_loader = _augment_loader_with_zero_static(test_loader, static_features)

        schema_resolver = SchemaResolver(
            feature_names=feature_names,
            dynamic_features=config["vars"]["DYNAMIC"],
            static_features=static_features,
            allow_missing_static=not static_in_features,
            missing_prefix=translator_cfg.get("missing_prefix", "MissingIndicator_"),
            group_col=group_col,
        )

        translator = RetrievalTranslator(
            num_features=len(schema_resolver.indices.dynamic),
            d_latent=translator_cfg.get("d_latent", 128),
            d_model=translator_cfg.get("d_model", 128),
            d_time=translator_cfg.get("d_time", 16),
            n_enc_layers=translator_cfg.get("n_enc_layers", 4),
            n_dec_layers=translator_cfg.get("n_dec_layers", 2),
            n_cross_layers=training_cfg.get("n_cross_layers", 2),
            n_heads=translator_cfg.get("n_heads", 8),
            d_ff=translator_cfg.get("d_ff", 512),
            dropout=translator_cfg.get("dropout", 0.2),
            out_dropout=translator_cfg.get("out_dropout", 0.1),
            static_dim=len(static_features),
            temporal_attention_mode=_get_temporal_attention_mode(translator_cfg),
            temporal_attention_window=translator_cfg.get("temporal_attention_window", 0),
            output_mode=training_cfg.get("output_mode", "residual"),
        )

        checkpoint_path = args.translator_checkpoint
        if not checkpoint_path:
            checkpoint_path = str(Path(output_cfg["run_dir"]) / "best_translator.pt")
        renorm_scale = None
        renorm_offset = None
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            translator.load_state_dict(checkpoint["translator_state_dict"], strict=False)
            renorm_scale = checkpoint.get("renorm_scale")
            renorm_offset = checkpoint.get("renorm_offset")
            logging.info("Loaded retrieval translator from %s", checkpoint_path)
            if renorm_scale is not None:
                logging.info("Cross-domain renormalization params loaded from checkpoint")
        else:
            logging.warning("No retrieval checkpoint found at %s", checkpoint_path)

        # Build memory bank from MIMIC target data for eval-time retrieval
        from .core.retrieval_translator import build_memory_bank
        from .core.eval import RetrievalTranslatorWrapper

        device = config.get("device", "cuda")
        translator.to(device)
        translator.eval()

        target_runtime = _build_runtime_from_config(
            config,
            data_dir_override=config.get("target_data_dir"),
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        target_runtime.load_data()
        target_train_loader = target_runtime.create_dataloader(
            "train", shuffle=False, ram_cache=True,
        )
        # Augment target loader with static features (same as training)
        target_feature_names = target_train_loader.dataset.get_feature_names()
        target_static_in_features = all(name in target_feature_names for name in static_features)
        if not target_static_in_features:
            if use_static:
                static_recipe = _get_static_recipe(config)
                if static_recipe:
                    target_static_df = load_static_with_recipe(
                        data_dir=Path(config["target_data_dir"]),
                        file_names=config["file_names"],
                        group_col=group_col,
                        static_features=static_features,
                        recipe_path=Path(static_recipe),
                    )
                    target_train_loader = _augment_loader_with_static(
                        target_train_loader, target_static_df, group_col, static_features
                    )
                else:
                    target_train_loader = _augment_loader_with_zero_static(target_train_loader, static_features)
            else:
                target_train_loader = _augment_loader_with_zero_static(target_train_loader, static_features)

        logging.info("Building memory bank for eval-time retrieval...")
        memory_bank = build_memory_bank(
            encoder=translator,
            target_loader=target_train_loader,
            schema_resolver=schema_resolver,
            device=device,
            window_size=training_cfg.get("retrieval_window", 6),
            window_stride=training_cfg.get("window_stride", None),
        )

        # Wrap translator with memory bank for full retrieval at eval
        wrapped_translator = RetrievalTranslatorWrapper(
            translator=translator,
            memory_bank=memory_bank,
            k_neighbors=training_cfg.get("k_neighbors", 16),
            retrieval_window=training_cfg.get("retrieval_window", 6),
        )

        evaluator = TransformerTranslatorEvaluator(
            yaib_runtime=yaib_runtime,
            translator=wrapped_translator,
            schema_resolver=schema_resolver,
            device=device,
            renorm_scale=renorm_scale,
            renorm_offset=renorm_offset,
            task_type=training_cfg.get("task_type", "classification"),
        )
        output_path = Path(args.output_parquet)
        results = evaluator.evaluate_original_vs_translated(
            test_loader, output_path,
            export_full_sequence=getattr(args, "export_full_sequence", True),
        )

    elif translator_type == "finetune_lstm":
        # -----------------------------------------------------------------
        # Fine-tuned LSTM upper bound: load fine-tuned LSTM weights from
        # checkpoint, use identity translator, evaluate as normal.
        # -----------------------------------------------------------------
        from .baselines.components import IdentityDATranslator

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
            subset_fraction=debug_fraction if debug_mode else None,
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
                test_loader = _augment_loader_with_static(test_loader, static_df, group_col, static_features)
            else:
                test_loader = _augment_loader_with_zero_static(test_loader, static_features)

        schema_resolver = SchemaResolver(
            feature_names=feature_names,
            dynamic_features=config["vars"]["DYNAMIC"],
            static_features=static_features,
            allow_missing_static=not static_in_features,
            missing_prefix=translator_cfg.get("missing_prefix", "MissingIndicator_"),
            group_col=group_col,
        )

        translator = IdentityDATranslator()

        checkpoint_path = args.translator_checkpoint
        if not checkpoint_path:
            checkpoint_path = str(Path(output_cfg["run_dir"]) / "best_translator.pt")

        # Step 1: Evaluate with ORIGINAL frozen LSTM (the baseline we compare against)
        yaib_runtime.load_baseline_model(freeze=True)
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        yaib_runtime._model = yaib_runtime._model.to(device)
        orig_evaluator = TransformerTranslatorEvaluator(
            yaib_runtime=yaib_runtime,
            translator=translator,
            schema_resolver=schema_resolver,
            device=device,
            renorm_scale=None,
            renorm_offset=None,
            task_type=training_cfg.get("task_type", "classification"),
        )
        logging.info("Evaluating with ORIGINAL frozen LSTM (baseline)...")
        original_metrics, orig_probs, orig_targets = orig_evaluator.evaluate_original(test_loader)

        # Step 2: Load fine-tuned LSTM weights and evaluate
        renorm_scale = None
        renorm_offset = None
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            translator.load_state_dict(checkpoint["translator_state_dict"], strict=False)
            renorm_scale = checkpoint.get("renorm_scale")
            renorm_offset = checkpoint.get("renorm_offset")
            model_state = checkpoint.get("model_state_dict")
            if model_state is not None:
                yaib_runtime._model.load_state_dict(model_state)
                logging.info("Loaded fine-tuned LSTM weights from checkpoint")
            else:
                logging.warning("No model_state_dict in checkpoint — using original frozen LSTM")
            logging.info("Loaded finetune_lstm translator from %s", checkpoint_path)
        else:
            logging.warning("No finetune_lstm checkpoint found at %s", checkpoint_path)

        ft_evaluator = TransformerTranslatorEvaluator(
            yaib_runtime=yaib_runtime,
            translator=translator,
            schema_resolver=schema_resolver,
            device=device,
            renorm_scale=renorm_scale,
            renorm_offset=renorm_offset,
            task_type=training_cfg.get("task_type", "classification"),
        )
        logging.info("Evaluating with FINE-TUNED LSTM...")
        output_path = Path(args.output_parquet)
        translated_metrics, trans_probs, trans_targets = ft_evaluator.translate_and_evaluate(
            test_loader, output_path,
        )

        if output_path:
            if trans_probs is not None:
                from .core.eval import _save_predictions
                _save_predictions(trans_probs, trans_targets, output_path)
            if orig_probs is not None:
                from .core.eval import _save_predictions
                _save_predictions(orig_probs, orig_targets, output_path, suffix=".original")

        results = {
            "original": original_metrics,
            "translated": translated_metrics,
        }

    elif translator_type in ("dann", "coral", "codats", "cluda", "raincoat", "acon", "stats_only"):
        translator_cfg = _get_translator_config(config)
        yaib_runtime = _build_runtime_from_config(
            config,
            batch_size_override=training_cfg["batch_size"],
            seed_override=training_cfg["seed"],
        )
        yaib_runtime.load_data()
        test_loader = yaib_runtime.create_dataloader(
            'test', shuffle=False, ram_cache=True,
            subset_fraction=debug_fraction if debug_mode else None,
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
                    raise ValueError("Static features missing; provide paths.static_recipe.")
                static_df = load_static_with_recipe(
                    data_dir=Path(config["data_dir"]),
                    file_names=config["file_names"],
                    group_col=group_col,
                    static_features=static_features,
                    recipe_path=Path(static_recipe),
                )
                test_loader = _augment_loader_with_static(test_loader, static_df, group_col, static_features)
            else:
                test_loader = _augment_loader_with_zero_static(test_loader, static_features)

        schema_resolver = SchemaResolver(
            feature_names=feature_names,
            dynamic_features=config["vars"]["DYNAMIC"],
            static_features=static_features,
            allow_missing_static=not static_in_features,
            missing_prefix=translator_cfg.get("missing_prefix", "MissingIndicator_"),
            group_col=group_col,
        )

        # Build translator matching what was used during training
        if translator_type == "codats":
            from .baselines.codats_backbone import CoDATS1DCNN
            translator = CoDATS1DCNN(
                num_features=len(schema_resolver.indices.dynamic),
                d_model=translator_cfg.get("d_model", 128),
                n_conv_layers=translator_cfg.get("n_conv_layers", 3),
                kernel_size=translator_cfg.get("kernel_size", 5),
                dropout=translator_cfg.get("dropout", 0.2),
                temporal_attention_mode=_get_temporal_attention_mode(translator_cfg),
            )
        elif translator_type == "stats_only":
            from .baselines.components import IdentityDATranslator
            translator = IdentityDATranslator()
        else:
            # DANN, CORAL, CLUDA, RAINCOAT, ACON all use EHRTranslator
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
        renorm_scale = None
        renorm_offset = None
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            translator.load_state_dict(checkpoint["translator_state_dict"], strict=False)
            renorm_scale = checkpoint.get("renorm_scale")
            renorm_offset = checkpoint.get("renorm_offset")
            logging.info("Loaded %s translator from %s", translator_type, checkpoint_path)
        else:
            logging.warning("No %s checkpoint found at %s", translator_type, checkpoint_path)

        evaluator = TransformerTranslatorEvaluator(
            yaib_runtime=yaib_runtime,
            translator=translator,
            schema_resolver=schema_resolver,
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            renorm_scale=renorm_scale,
            renorm_offset=renorm_offset,
            task_type=training_cfg.get("task_type", "classification"),
        )
        output_path = Path(args.output_parquet)
        results = evaluator.evaluate_original_vs_translated(
            test_loader, output_path,
            export_full_sequence=getattr(args, "export_full_sequence", True),
        )

    elif translator_type == "linear_regression":
        translator, yaib_runtime, test_loader, target_loader = _prepare_linear_regression(
            config,
            split=DataSplit.test,
            shuffle=False,
            debug_mode=debug_mode,
            debug_fraction=debug_fraction,
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
            subset_fraction=debug_fraction if debug_mode else None,
            subset_seed=training_cfg["seed"],
        )

        data_shape = next(iter(test_loader))[0].shape
        input_size = data_shape[-1]
        translator = IdentityTranslator(input_size=input_size)

    if translator_type not in {"linear_regression", "transformer", "affine", "shared_latent", "retrieval", "dann", "coral", "codats", "cluda", "raincoat", "acon", "stats_only", "finetune_lstm"} and args.translator_checkpoint:
        checkpoint = torch.load(args.translator_checkpoint, map_location="cpu")
        translator.load_state_dict(checkpoint["translator_state_dict"], strict=False)
        logging.info(f"Loaded translator from {args.translator_checkpoint}")

    if translator_type not in ("transformer", "affine", "shared_latent", "retrieval", "dann", "coral", "codats", "cluda", "raincoat", "acon", "stats_only", "finetune_lstm"):
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
                original_metrics, _, _ = evaluator.translate_and_evaluate(test_loader, None)
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
                    subset_fraction=debug_fraction if debug_mode else None,
                    subset_seed=training_cfg["seed"],
                )
                evaluator.yaib_runtime = norm_runtime
                original_metrics, _, _ = evaluator._evaluate_without_translator(norm_test_loader)
                evaluator.yaib_runtime = yaib_runtime
            translated_metrics, _, _ = evaluator.translate_and_evaluate(test_loader, output_path)
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


def run_e2e_baseline(args):
    """Train and evaluate an end-to-end DA baseline (CLUDA, RAINCOAT, ACON)."""
    config = load_config(Path(args.config))
    config["_config_path"] = str(args.config)  # for label_mode inference in adapter
    device = config.get("device", "cuda")
    training = config.get("training", {})
    method = config.get("translator", {}).get("type", "e2e_cluda")

    logging.info("=" * 60)
    logging.info("End-to-end DA baseline: %s", method)
    logging.info("=" * 60)

    # --- Step 1: Set up E2E data adapter FIRST (needed for fair baseline) ---
    # NOTE: After the source/target swap in the adapter, source=MIMIC, target=eICU.
    # We train on MIMIC labels (source), align with eICU (target), eval on eICU test.
    from .baselines.end_to_end.data_adapter import YAIBToE2EAdapter

    adapter = YAIBToE2EAdapter(config)
    source_train, source_val, source_test, target_train, target_val, target_test = adapter.get_loaders()

    # Auto-detect number of input channels from data
    num_channels = adapter.num_channels
    training["num_input_channels"] = num_channels
    logging.info("Detected %d input channels, seq_len=%d", num_channels, training.get("seq_len", 48))

    # --- Step 1b: Compute no-adaptation baseline on the TARGET (eICU) test data ---
    # CRITICAL: Both "Original" and "Translated" must be evaluated on identical data
    # for a fair comparison. We run the frozen MIMIC LSTM on windowed eICU test data.
    logging.info("Computing no-adaptation baseline (frozen MIMIC LSTM on E2E windowed eICU test data)...")
    baseline_runtime = _build_runtime_from_config(
        config, batch_size_override=training.get("batch_size", 64),
    )
    baseline_runtime.load_data()
    baseline_runtime.load_baseline_model()
    baseline_runtime._model = baseline_runtime._model.to(device)
    for param in baseline_runtime._model.parameters():
        param.requires_grad = False

    # Evaluate frozen LSTM on the E2E windowed eICU test data (target_test)
    baseline_runtime._model.eval()
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in target_test:
            x, labels, static, vmask = batch
            # x is (B, C, L) in E2E format — transpose to (B, L, C) for LSTM
            x_lstm = x.transpose(1, 2).to(device)  # (B, L, C)
            labels = labels.to(device)
            vmask = vmask.to(device)

            # Run frozen LSTM
            logits = baseline_runtime._model(x_lstm)  # (B, L, num_classes) or (B, num_classes)
            if logits.dim() == 3:
                probs = torch.softmax(logits, dim=-1)[:, :, 1]  # (B, L)
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]  # (B,)

            if labels.dim() == 1:
                # Per-stay: probs is (B, L) but we need one prob per stay.
                # Use the last valid timestep's prediction (matches YAIB eval).
                valid = labels >= 0
                if valid.sum() > 0 and probs.dim() == 2:
                    # Find the last valid timestep index for each sample
                    # vmask is (B, L): True for non-padded timesteps
                    last_valid_idx = vmask.long().cumsum(dim=1).argmax(dim=1)  # (B,)
                    per_stay_probs = probs[torch.arange(probs.size(0), device=probs.device), last_valid_idx]
                    all_probs.append(per_stay_probs[valid].cpu().numpy())
                    all_labels.append(labels[valid].cpu().numpy())
                elif valid.sum() > 0:
                    all_probs.append(probs[valid].cpu().numpy())
                    all_labels.append(labels[valid].cpu().numpy())
            else:
                # Per-timestep
                valid = vmask & (labels >= 0)
                if valid.sum() > 0:
                    all_probs.append(probs[valid].cpu().numpy())
                    all_labels.append(labels[valid].cpu().numpy())

    all_probs_np = np.concatenate(all_probs)
    all_labels_np = np.concatenate(all_labels)
    try:
        orig_auroc = roc_auc_score(all_labels_np, all_probs_np)
    except ValueError:
        orig_auroc = 0.5
    try:
        orig_aucpr = average_precision_score(all_labels_np, all_probs_np)
    except ValueError:
        orig_aucpr = 0.0
    original_metrics = {"AUCROC": orig_auroc, "AUCPR": orig_aucpr}
    logging.info("No-adaptation baseline (frozen MIMIC LSTM on eICU test): AUROC=%.4f AUCPR=%.4f",
                 orig_auroc, orig_aucpr)

    # --- Step 2: Create model + trainer ---
    # Source=MIMIC (labeled), target=eICU (alignment + eval). target_val for early stopping.
    use_source_val_es = training.get("use_source_val_es", False)
    val_loader_for_es = None if use_source_val_es else target_val
    if use_source_val_es:
        logging.info("[E2E] Using SOURCE (MIMIC) validation for early stopping (use_source_val_es=true)")
    if method == "e2e_cluda":
        from .baselines.end_to_end.cluda_model import CLUDAModel, CLUDATrainer
        model = CLUDAModel(config)
        trainer = CLUDATrainer(model, source_train, target_train, source_val, config, device,
                               target_val_loader=val_loader_for_es)
    elif method == "e2e_raincoat":
        from .baselines.end_to_end.raincoat_model import RAINCOATModel, RAINCOATTrainer
        model = RAINCOATModel(config)
        trainer = RAINCOATTrainer(model, source_train, target_train, source_val, config, device,
                                  target_val_loader=val_loader_for_es)
    elif method == "e2e_acon":
        from .baselines.end_to_end.acon_model import ACONModel, ACONTrainer
        model = ACONModel(config)
        trainer = ACONTrainer(model, source_train, target_train, source_val, config, device,
                              target_val_loader=val_loader_for_es)
    elif method == "e2e_dann":
        from .baselines.end_to_end.dann_e2e_model import DANNModel, DANNTrainer
        model = DANNModel(config)
        trainer = DANNTrainer(model, source_train, target_train, source_val, config, device,
                              target_val_loader=val_loader_for_es)
    elif method == "e2e_coral":
        from .baselines.end_to_end.coral_e2e_model import CORALModel, CORALTrainer
        model = CORALModel(config)
        trainer = CORALTrainer(model, source_train, target_train, source_val, config, device,
                               target_val_loader=val_loader_for_es)
    elif method == "e2e_codats":
        from .baselines.end_to_end.codats_e2e_model import CoDATSModel, CoDATSTrainer
        model = CoDATSModel(config)
        trainer = CoDATSTrainer(model, source_train, target_train, source_val, config, device,
                                target_val_loader=val_loader_for_es)
    elif method == "e2e_cdan":
        from .baselines.end_to_end.cdan_model import CDANModel, CDANTrainer
        model = CDANModel(config)
        trainer = CDANTrainer(model, source_train, target_train, source_val, config, device,
                              target_val_loader=val_loader_for_es)
    else:
        raise ValueError(f"Unknown E2E method: {method}. Use e2e_cluda, e2e_raincoat, e2e_acon, e2e_dann, e2e_coral, e2e_codats, or e2e_cdan.")

    # --- Step 4: Train ---
    trainer.train()

    # --- Step 5: Evaluate E2E model on TARGET (eICU) test set ---
    e2e_results = trainer.evaluate(target_test)
    logging.info("E2E method results (on eICU test): %s", e2e_results)

    # --- Step 6: Report results in the standard format ---
    # The format must match what collect_result.py / gpu_scheduler parse:
    #   EVALUATION RESULTS
    #   Original Test Data: AUCROC, AUCPR, ...
    #   Translated Test Data: AUCROC, AUCPR, ...
    #   Difference: AUCROC, AUCPR, ...
    translated_metrics = {
        "AUCROC": e2e_results["auroc"],
        "AUCPR": e2e_results["aucpr"],
        "loss": e2e_results["loss"],
    }

    logging.info("=" * 80)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 80)
    logging.info("Original Test Data:")
    for metric, value in original_metrics.items():
        logging.info("  %s: %.4f", metric, value)
    logging.info("Translated Test Data:")
    for metric, value in translated_metrics.items():
        logging.info("  %s: %.4f", metric, value)
    logging.info("Difference:")
    for metric in original_metrics.keys():
        if metric in translated_metrics:
            diff = translated_metrics[metric] - original_metrics[metric]
            logging.info("  %s: %+0.4f", metric, diff)
    logging.info("=" * 80)


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

    e2e_parser = subparsers.add_parser("run_e2e_baseline", help="Train and evaluate end-to-end DA baseline (CLUDA, RAINCOAT, ACON)")
    e2e_parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    e2e_parser.add_argument("--output_parquet", type=str, required=True, help="Output path for results parquet")
    e2e_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

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
    elif args.command == "run_e2e_baseline":
        run_e2e_baseline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
