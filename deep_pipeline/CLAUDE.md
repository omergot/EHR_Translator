# CLAUDE.md

You are the best data scientist and deep learning engineer in the world who is eager to succeed in our mission to find a translator that improves the metrics.

## Project Overview

**EHR Translator Deep Pipeline** is a domain adaptation system for electronic health record (EHR) time-series data (eICU → MIMIC-IV).

**Core Goal:** Train a `Translator` model to transform source-domain (eICU) data so that a **strictly frozen** target-domain (MIMIC-IV) LSTM baseline model performs well on it. Two clinical tasks: Mortality24 (per-stay, bidirectional) and Sepsis (per-timestep, causal).

### Current Best Results

| Task | Best AUCROC Δ | Best AUCPR Δ | Method | Baseline AUCROC |
|---|---|---|---|---|
| **Mortality24** | **+0.0441** | **+0.0546** | SL v3 / SL+MIMIC labels | 0.8079 |
| **AKI** | **+0.0370** | **+0.1021** | Shared Latent v3 | 0.8558 |
| **Sepsis** | **+0.0102** | **+0.0056** | Delta + target task loss | 0.7159 |

**Critical finding**: Task structure determines which approach works. Mortality and AKI benefit from shared latent space translation; sepsis benefits from delta-based + MIMIC target task loss (+0.0102 AUCROC, 4x previous best). See "Task-Specific Strategy" below.

## Commands

```bash
# Install (editable)
pip install -e .

# Delta-based translator (EHRTranslator)
python run.py train_translator --config configs/<task>_transformer_config.json
python run.py translate_and_eval --config configs/<task>_transformer_config.json --output_parquet out.parquet
python run.py train_and_eval --config configs/<task>_transformer_config.json --output_parquet out.parquet

# Shared latent translator (SharedLatentTranslator) — same CLI, different config
python run.py train_translator --config experiments/configs/sl_v3_mortality.json
python run.py translate_and_eval --config experiments/configs/sl_v3_mortality.json --output_parquet out.parquet

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

### Translator Models

**Two translation paradigms** — selected via `config["translator"]["type"]`:

#### 1. Delta-Based: `EHRTranslator` (`src/core/translator.py`)

Outputs deltas added to input features (starts near identity). Best for sepsis.

- Triplet projection (value + missingness + time-delta) → per-feature sensor embeddings
- Sinusoidal temporal encoding → stacked `AxialBlock` layers (variable-wise + temporal attention)
- FiLM modulation from static features → delta output head
- `set_temporal_mode()` flips between causal/bidirectional attention

#### 2. Shared Latent Space: `SharedLatentTranslator` (`src/core/latent_translator.py`)

Maps both domains into a shared latent space, decodes to target-like features. Best for mortality.

```
Source eICU features → [Shared Encoder] → Latent z (B,T,d_latent) → [Decoder] → Translated features
Target MIMIC features → [Shared Encoder] → Latent z → [Decoder] → Reconstructed features
```

- Encoder: triplet proj → AxialBlocks → mean pool over features → MLP to latent
- Decoder: MLP from latent → broadcast + learned feature embeddings → AxialBlocks → output
- Outputs absolute values (not deltas) — decoder learns full target distribution
- Reuses `AxialBlock` from translator.py with FiLM static conditioning

#### Other translators
- **`IdentityTranslator`**: Pass-through baseline (no transformation)
- **`LinearRegressionTranslator`**: Per-feature affine mapping fitted from data statistics

### Training (`src/core/train.py`)

**`TransformerTranslatorTrainer`** — for delta-based EHRTranslator:
- **Task loss**: Classification loss from frozen LSTM on translated data
- **Fidelity loss**: MSE between input and output (preserve data integrity). Essential — without it, training diverges catastrophically.
- **Range loss**: Penalty for out-of-bounds feature values
- Supports mixed-precision (AMP), gradient accumulation, debug logging

**`LatentTranslatorTrainer`** — for SharedLatentTranslator:
- **Phase 1 (Pretraining)**: Autoencoder reconstruction on MIMIC target data only
- **Phase 2 (Joint training)**: Task loss + MMD alignment in latent space + reconstruction loss + range loss
- Config keys: `pretrain_epochs`, `lambda_align`, `lambda_recon`, `lambda_range`
- Saves pretrain checkpoints for OOM recovery between phases

### Evaluation (`src/core/eval.py`)

`TranslatorEvaluator` / `TransformerTranslatorEvaluator`: Compute AUROC, AUCPR, Brier, ECE on translated test data. Export results to parquet.

### Key Supporting Modules

| Module | Purpose |
|---|---|
| `src/core/schema.py` | `SchemaResolver` — extract/rebuild YAIB batch format (x_val, x_miss, x_static, t_abs, m_pad) |
| `src/core/bucket_batching.py` | Variable-length batching — groups sequences by length, truncates padding per batch. 3x speedup for long sequences. Config: `training.variable_length_batching: true` |
| `src/core/mmd.py` | Multi-kernel MMD with median heuristic, unbiased estimator |
| `src/core/pretrain.py` | MLM pretrainer with BERT-style masking (80/10/10) |
| `src/core/hidden_extractor.py` | LSTM hidden state extraction via forward hooks |
| `src/core/focal_loss.py` | Focal loss for hard-example mining |
| `src/core/static_utils.py` | `StaticAugmentedDataset` — injects static features as 4th batch element |
| `src/adapters/yaib.py` | `YAIBRuntime` — wraps YAIB framework (gin config, data, baseline model, dataloaders) |

### Safety & Validation (CRITICAL)

- **Frozen Baseline**: Baseline model parameters `requires_grad=False`. `verify_baseline_determinism` check at startup.
- **Padding Integrity**: Translator output `masked_fill` ensures padded timesteps remain exactly 0.0.
- **Time-Travel Rules**: Sepsis/AKI must use `temporal_attention_mode="causal"`. Mortality can use `"bidirectional"`.
- **Baseline model mode**: Must be in `train()` mode (not eval) for cuDNN RNN backward compatibility.

## Task-Specific Strategy (Key Insight)

Mortality and AKI are solved with shared latent translation. Sepsis saw a major breakthrough with delta + MIMIC target task loss (+0.0102 AUCROC, 4x previous best).

| Factor | Mortality24 | AKI | Sepsis |
|---|---|---|---|
| Labels | Per-stay (5.5% pos) | Per-timestep (11.95% pos) | Per-timestep (1.1% pos) |
| Attention | Bidirectional | Causal | Causal |
| Training data | Full (113K stays) | Full (165K stays) | Full (123K stays) |
| Best approach | SL v3 (+0.044) | SL v3 (+0.037) | Delta + target task (+0.010) |
| Delta-based | +0.033 | +0.024 | +0.010 (target task loss) |

**Why shared latent works for mortality and AKI:**
1. MMD alignment in latent space provides direct, dense gradient — no backward through frozen LSTM needed
2. Reconstruction loss stabilizes latent space through decoder path
3. Task loss only needs small adjustments on well-structured latent space

**Why target task loss works for sepsis:**
1. MIMIC target task loss provides direct task-relevant gradient from MIMIC domain labels
2. This bypasses the gradient bottleneck (fidelity 5-10x task gradient) by adding a separate task signal
3. The MIMIC labels (1.1% pos rate) are comparable to eICU, providing consistent task signal
4. Also improves calibration significantly (Brier -0.046, ECE -0.043)

**What didn't work for sepsis:**
1. Shared latent: SL always hurts (-0.007 to -0.043). Per-timestep causal structure incompatible with SL.
2. Negative subsampling: 6 filtered experiments all negative (SL: -0.0001 to -0.0073, delta: -0.0016 to -0.0047)
3. Cross-task transfer: AKI translators don't generalize to sepsis on full data
4. Per-stay MIL aggregation improves stay-level metrics but hurts per-timestep AUCROC

## Gradient Bottleneck (Root Cause of Difficulty)

The fundamental challenge: **fidelity gradient is 3-10x larger than task gradient**. The task signal must flow backward through the frozen LSTM, producing weak, noisy gradients that the fidelity loss dominates.

- Sepsis: 73% padding, 1.1% positive rate, per-timestep labels → extremely diffuse gradient
- Mortality: 0% padding, per-stay labels → concentrated gradient signal
- `lambda_fidelity=0.0` causes catastrophic divergence (AUCROC -0.101)
- Full analysis in `docs/gradient_bottleneck_analysis.md`

## Config Structure

JSON configs in `configs/` (base) and `experiments/configs/` (experiments). Key sections:

```json
{
  "translator": {"type": "transformer|shared_latent", "d_model": 128, "d_latent": 64, ...},
  "training": {
    "epochs": 30, "lr": 1e-4, "batch_size": 64,
    "lambda_fidelity": 1.0, "lambda_range": 0.5,
    "oversampling_factor": 20,
    "variable_length_batching": true,
    "pretrain_epochs": 10, "lambda_align": 0.5, "lambda_recon": 0.1
  }
}
```

## Experiment Infrastructure

| Component | Purpose |
|---|---|
| `experiments/configs/` | Per-experiment JSON configs (debug + full variants) |
| `experiments/collect_result.py` | Parses training logs → JSON results |
| `scripts/aggregate_results.py` | Aggregates results → markdown table |
| `scripts/run_full_parallel.sh` | Runs experiments via git worktrees (one per GPU) |
| `experiments/.state`, `.state_full` | Tracks completed experiment runs |

Experiment branches follow pattern `exp/<experiment_name>` (e.g., `exp/a3_padding_aware`, `exp/shared_latent`).

## Documentation (`docs/`)

### Start Here
- **[docs/comprehensive_results_summary.md](docs/comprehensive_results_summary.md)** — **Master results document.** All experiments, rankings, full-data validation, shared latent results, conclusions, and recommendations. Start here for project status.
- **[docs/shared_latent_results.md](docs/shared_latent_results.md)** — Shared latent space results: mortality (+0.044), sepsis (negative), bucket batching, detailed analysis.

### Architecture & Analysis
- **[docs/architecture.md](docs/architecture.md)** — EHRTranslator architecture details.
- **[docs/gradient_flow_mechanics.md](docs/gradient_flow_mechanics.md)** — Forward/backward pass trace explaining weak task signal.
- **[docs/gradient_bottleneck_analysis.md](docs/gradient_bottleneck_analysis.md)** — Gradient bottleneck analysis with per-timestep data.
- **[docs/optimization_recommendations.md](docs/optimization_recommendations.md)** — Padding/memory optimization strategies.

### Experiment History
- **[docs/recommendations_next_steps.md](docs/recommendations_next_steps.md)** — Pre-A/B/C recommendations (input shaping, latent alignment, training signal, evaluation).
- **[docs/experiment_results_abc.md](docs/experiment_results_abc.md)** — A/B/C series raw results (13 experiments × 2 tasks).
- **[docs/investigation_mortality_vs_sepsis.md](docs/investigation_mortality_vs_sepsis.md)** — Controlled mortality vs sepsis investigation.
- **[docs/experiment_results_mmd_mlm.md](docs/experiment_results_mmd_mlm.md)** — Early MMD+MLM experiments (minimal gains).
- **[docs/shared_latent_plan.md](docs/shared_latent_plan.md)** — Shared latent architecture design and implementation plan.

## Key Dependencies

- **PyTorch**: Model definitions, training loops
- **icu-benchmarks (YAIB)**: Baseline models, data preprocessing, gin configs
- **recipys**: Static feature preprocessing recipes
- **polars/pandas**: DataFrame operations for parquet I/O
- **gin-config**: YAIB configuration management
- **scikit-learn**: Evaluation metrics (AUROC, AUCPR)

## Coding Standards

- Logging: Use `logging.info()` (never `print` in core modules).
- Tensors: Always handle device placement explicitly (`.to(device)`).
- AMP: Cast hidden states to `.float()` before passing to discriminators, loss functions, or MLPs to avoid float16/float32 dtype mismatches.
- Config: All new config keys must default to disabled (0/None/False) for backward compatibility.
- Config files: JSON in `configs/` (base) and `experiments/configs/` (experiments).
