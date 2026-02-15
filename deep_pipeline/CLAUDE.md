# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EHR Translator Deep Pipeline** is a domain adaptation system for electronic health record (EHR) time-series data (e.g., eICU → MIMIC-IV). 

**Core Goal:** Train a `Translator` model to transform source-domain data so that a **strictly frozen** target-domain baseline model (from the YAIB framework) performs well on it. The system must preserve clinical task performance (Sepsis/AKI/Mortality) while maintaining data plausibility.

## Commands

```bash
# Install (editable)
pip install -e .

# Run pipeline (all three subcommands)
python run.py train_translator --config configs/<task>_transformer_config.json
python run.py translate_and_eval --config configs/<task>_transformer_config.json --output_parquet out.parquet
python run.py train_and_eval --config configs/<task>_transformer_config.json --output_parquet out.parquet

# Tests
pytest tests/
pytest tests/test_identity_translator.py
```

## Architecture

### Entry Point and CLI (`run.py` → `src/cli.py`)

`src/cli.py` is the main orchestrator. It parses a JSON config, builds the runtime, creates the translator, trains, and evaluates. Three subcommands: `train_translator`, `translate_and_eval`, `train_and_eval`.

### Core Data Flow

```
JSON Config → YAIBRuntime (loads baseline model + data) → DataLoaders
  → Translator (transform source features) → Baseline model (compute task loss)
  → Multi-component loss backprop → Checkpoint best → Evaluate → Export parquet
```

### Translator Models (`src/core/translator.py`)

Three translator types, selected via `config["translator"]["type"]`:

- **`IdentityTranslator`**: Pass-through baseline (no transformation)
- **`LinearRegressionTranslator`**: Per-feature affine mapping fitted from data statistics
- **`EHRTranslator`** (transformer): The main model. Uses triplet projection (value + missingness + time-delta), per-feature sensor embeddings, sinusoidal temporal encoding, stacked `AxialBlock` layers (variable-wise + temporal attention), and FiLM modulation from static features. Outputs deltas added to input.

### Training (`src/core/train.py`)

- **`TranslatorTrainer`**: Basic trainer for identity/linear translators
- **`TransformerTranslatorTrainer`**: Advanced trainer with three loss components:
  - **Task loss**: Classification loss from baseline model on translated data
  - **Fidelity loss**: MSE between input and output (preserve data integrity)
  - **Range loss**: Penalty for values outside feature bounds from a CSV
- Supports mixed-precision (AMP), gradient accumulation, and debug logging

### Evaluation (`src/core/eval.py`)

- `TranslatorEvaluator` / `TransformerTranslatorEvaluator`: Compute AUROC, AUCPR, loss on translated test data. Export results to parquet.

### Safety & Validation (CRITICAL)
- Frozen Baseline: The baseline model parameters are set to requires_grad=False. A specific verify_baseline_determinism check runs at startup to ensure no internal noise (Dropout/BatchNorm) affects the training signal.

- Padding Integrity: The translator output is explicitly masked (masked_fill) to ensure padded time steps remain exactly 0.0.

- Time-Travel Rules: - Sepsis/AKI: Must use temporal_attention_mode="causal" to prevent looking ahead.

- Mortality: Can use "bidirectional".

### Schema Resolution (`src/core/schema.py`)

`SchemaResolver` manages feature indices across the YAIB batch format. `extract()` deconstructs batches into (x_val, x_miss, x_static, t_abs, m_pad); `rebuild()` reconstructs them after translation.

### YAIB Adapter (`src/adapters/yaib.py`)

`YAIBRuntime` wraps the external YAIB (icu-benchmarks) framework: handles gin config, data preprocessing, baseline model loading, dataloader creation, and forward/loss computation. This isolates all YAIB complexity behind a clean interface.

### Static Feature Handling (`src/core/static_utils.py`)

`StaticAugmentedDataset` wraps a dataset to inject static features as a 4th batch element. Uses `recipys` for preprocessing recipes.

## Config Structure

JSON configs in `configs/` define everything for a run: data paths, baseline model paths, YAIB gin configs, feature lists (DYNAMIC/STATIC), translator type and hyperparameters, training params (lr, epochs, batch_size), output paths, seed, and device. Task-specific configs exist for Sepsis, AKI, and Mortality.

## Key Dependencies

- **PyTorch**: Model definitions, training loops
- **icu-benchmarks (YAIB)**: Baseline models, data preprocessing, gin configs
- **recipys**: Static feature preprocessing recipes
- **polars/pandas**: DataFrame operations for parquet I/O
- **gin-config**: YAIB configuration management
- **scikit-learn**: Evaluation metrics (AUROC, AUCPR)

## Utility Scripts (`scripts/`)

Standalone tools: `generate_static_recipe.py` (build preprocessing recipes), `filter_cohort_by_stay_ids.py` (subset cohorts), `compute_feature_correlation.py` / `compute_feature_ab.py` (feature analysis), `compare_data.py` (dataset comparison), `inspect_linear_regression_pkl.py` (inspect saved linear models).


## Documentation (`docs/`)

### Current State & Key Findings (start here)
- **[docs/gradient_bottleneck_analysis.md](docs/gradient_bottleneck_analysis.md)** — **Main findings document.** Consolidated experiment results table, confirmed gradient bottleneck (fidelity 3-10x task), per-timestep gradient analysis comparing sepsis vs mortality, and ranked next-step recommendations. Start here for current project status.

### Architecture
- **[docs/architecture.md](docs/architecture.md)** — EHRTranslator architecture: triplet embedding, AxialBlock backbone, delta/forecast/reconstruction heads, FiLM static conditioning, temporal attention modes, MLM pretraining breakdown.
- **[docs/gradient_flow_mechanics.md](docs/gradient_flow_mechanics.md)** — Step-by-step trace of forward/backward pass through translator→LSTM→loss, explaining how gradients reach the translator and why the task signal is weak.

### Experiment History (chronological)
- **[docs/strategy_evaluation_mmd_mlm.md](docs/strategy_evaluation_mmd_mlm.md)** — Evaluation of 4 competing approaches (forecasting, distillation, MMD, MLM). Chose MMD+MLM. Explains reasoning.
- **[docs/implementation_plan_mmd_mlm.md](docs/implementation_plan_mmd_mlm.md)** — Implementation plan for MMD domain matching + MLM pretraining (stages 0-5).
- **[docs/experiment_results_mmd_mlm.md](docs/experiment_results_mmd_mlm.md)** — Results from MMD+MLM experiments A-E (Sepsis, debug, 10 and 30 epochs). All variants near +0.001-0.002 AUCROC.
- **[docs/investigation_mortality_vs_sepsis.md](docs/investigation_mortality_vs_sepsis.md)** — Controlled experiments isolating why mortality (+0.025) works but sepsis (+0.002) doesn't. Rules out attention mode, capacity, data size. Identifies sequence length + task structure as root cause.

### Best Results Summary
| Task | Best AUCROC Delta | Config |
|---|---|---|
| Mortality24 | **+0.0264** | Bidirectional, d128, full data, 30ep |
| Sepsis | **+0.0059** | Causal, d64, debug, f=20, W=25 |

## Coding Standards
- Logging: Use logging.info() (never print in core modules).
- Tensors: Always handle device placement explicitly (.to(device)).
- Config: Managed via JSON files in deep_pipeline/configs/