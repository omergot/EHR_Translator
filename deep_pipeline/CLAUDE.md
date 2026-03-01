# CLAUDE.md

You are the best data scientist and deep learning engineer in the world who is eager to succeed in our mission to find a translator that improves the metrics.

## Project Overview

**EHR Translator Deep Pipeline** — domain adaptation for EHR time-series (eICU → MIMIC-IV). Train a `Translator` to transform source-domain data so a **strictly frozen** target-domain LSTM baseline performs well. Three paradigms: delta-based (`EHRTranslator`), shared latent (`SharedLatentTranslator`), and retrieval-guided (`RetrievalTranslator`). Three tasks: Mortality24 (per-stay), AKI (per-timestep), Sepsis (per-timestep).

## Commands

```bash
pip install -e .

# Delta-based translator (EHRTranslator)
python run.py train_translator --config configs/<task>_transformer_config.json
python run.py translate_and_eval --config configs/<task>_transformer_config.json --output_parquet out.parquet
python run.py train_and_eval --config configs/<task>_transformer_config.json --output_parquet out.parquet

# Shared latent translator (SharedLatentTranslator) — same CLI, different config
python run.py train_translator --config experiments/configs/sl_v3_mortality.json
python run.py translate_and_eval --config experiments/configs/sl_v3_mortality.json --output_parquet out.parquet

# Retrieval translator (RetrievalTranslator) — requires target_data_dir in config
python run.py train_translator --config configs/sepsis_retrieval_full.json
python run.py translate_and_eval --config configs/sepsis_retrieval_full.json --output_parquet out.parquet

# Tests
pytest tests/
```

## Architecture (Key Points)

- **Entry point**: `run.py` → `src/cli.py` (main orchestrator). Subcommands: `train_translator`, `translate_and_eval`, `train_and_eval`.
- **Data flow**: JSON Config → `YAIBRuntime` (baseline + data) → Translator → frozen LSTM → loss → checkpoint best → eval → parquet.
- **Delta-based** (`src/core/translator.py`): Outputs deltas added to input (starts near identity). `set_temporal_mode()` flips causal/bidirectional.
- **Shared latent** (`src/core/latent_translator.py`): Encoder→latent z→Decoder. Outputs absolute values, not deltas.
- **Retrieval-guided** (`src/core/retrieval_translator.py`): Shared encoder → `MemoryBank` (pre-encoded MIMIC windows) → k-NN per timestep → `CrossAttentionBlock` → Decoder. Instance-level matching, naturally causal.
- **Feature gate** (`src/core/feature_gate.py`): Learnable per-feature sigmoid weights for loss weighting. Shared module usable across all translator types.
- **Delta trainer** (`TransformerTranslatorTrainer` in `src/core/train.py`): task loss + fidelity loss + range loss.
- **SL trainer** (`LatentTranslatorTrainer` at end of `src/core/train.py`): Phase 1 = autoencoder pretrain on MIMIC. Phase 2 = task + MMD alignment + reconstruction + range.
- **Retrieval trainer** (`RetrievalTranslatorTrainer` in `src/core/train.py`): Phase 1 = autoencoder pretrain. Phase 2 = task + fidelity + range + smoothness + importance reg, with memory bank rebuilt every `memory_refresh_epochs`.
- **Task strategy**: SL + target norm for mortality/AKI. Delta + target task + target norm for sepsis. Retrieval translator under active development (sepsis-focused).
- **Cross-domain normalization** (`use_target_normalization`): Affine renorm of source features to target stats. Params saved in checkpoint.

## Safety & Validation (CRITICAL)

These rules prevent catastrophic failures. Violating any one can silently ruin results.

- **Frozen Baseline**: `requires_grad=False` on all baseline params. `verify_baseline_determinism` check at startup.
- **Baseline in train() mode**: Must use `model.train()` not `model.eval()` — cuDNN RNN backward requires it.
- **Padding Integrity**: Translator output `masked_fill` ensures padded timesteps remain exactly 0.0.
- **Time-Travel Rules**: Sepsis/AKI → `temporal_attention_mode="causal"`. Mortality → `"bidirectional"`.
- **lambda_fidelity > 0**: Setting `lambda_fidelity=0.0` causes catastrophic divergence (AUCROC -0.101). Never disable.
- **VLB incompatible with mortality**: `variable_length_batching=true` silently truncates per-stay sequences to length 1. Use `false` for mortality.
- **SL OOM**: SharedLatentTranslator uses ~2.5x more GPU memory. Use `batch_size=16` (not 32) on V100-32GB.
- **Retrieval OOM**: Memory bank (`window_latents`) is GPU-resident. Use `batch_size=16`. Control rebuild cost with `memory_refresh_epochs`.
- **Retrieval detach rule**: Always `src_latent.detach()` before querying memory bank — prevents backprop through k-NN/bank.
- **YAIB leakage rule**: NEVER use different `data_dir` for train and eval. Subsampling must happen within the YAIB split (`_apply_negative_subsampling()`), not via separate cohorts.
- **AMP dtype**: Always `.float()` hidden states before passing to discriminators, loss functions, or MLPs (float16→float32 mismatch).

## Coding Standards

- Logging: `logging.info()` — never `print` in core modules.
- Tensors: Always handle device placement explicitly (`.to(device)`).
- Config backward compat: All new config keys must default to disabled (0/None/False).
- Config files: JSON in `configs/` (base) and `experiments/configs/` (experiments).
- **`_get_training_config()` whitelist** (CRITICAL): This function in `cli.py` explicitly lists all config keys. New training config keys MUST be added here or they are **silently dropped**. This is the #1 source of "config change had no effect" bugs.

## Config Structure

JSON configs with two main sections:
- `"translator"`: `type` ("transformer"|"shared_latent"|"retrieval"), `d_model`, `d_latent`, `n_layers`, `n_enc_layers`, `n_dec_layers`, `n_cross_layers`, `output_mode`, etc.
- `"training"`: `epochs`, `lr`, `batch_size`, `lambda_fidelity`, `lambda_range`, `oversampling_factor`, `variable_length_batching`, `pretrain_epochs`, `lambda_align`, `lambda_recon`, `lambda_target_task`, `lambda_label_pred`, `negative_subsample_count`, `shuffle`, `use_target_normalization`, `early_stopping_patience`, `best_metric`, `k_neighbors`, `retrieval_window`, `n_cross_layers`, `output_mode`, `memory_refresh_epochs`, `lambda_importance_reg`, `lambda_smooth`, `feature_gate`.

## Experiment Queue System

All experiments are managed through `experiments/queue.yaml`. This is the single source of truth.

### Rules for Claude Sessions
- **NEVER launch experiments directly** with `python run.py`. Always add to the queue.
- To add an experiment: append an entry to `experiments/queue.yaml` under the pending section with `status: pending`.
- To prioritize: move the entry higher in the list (scheduler runs top-to-bottom).
- To run the queue: `python scripts/gpu_scheduler.py` (usually already running in a tmux/screen session).
- To check status: `python scripts/gpu_scheduler.py --status`
- Config files must exist before adding to queue. Create the config JSON first, then add the queue entry.

### GPU Rules
- Daytime (09:00-21:00): max 2 GPUs. Prefer GPUs 0, 1. Avoid GPU 3.
- Nighttime (21:00-09:00): max 3 GPUs. Can use GPU 2. GPU 3 only as last resort.
- These rules are enforced by the scheduler automatically.
- If launching a one-off manual experiment (debugging), use GPU 3 to avoid conflicts.

### Queue Entry Format
Each experiment needs: `name` (unique ID), `config` (path to JSON config), `output` (parquet output path), `status: pending`, and optionally `notes`.
