# EHR Translator

Domain adaptation for EHR time-series data. Trains a **Translator** network to transform source-domain electronic health records (eICU) so that a **strictly frozen** target-domain LSTM baseline (trained on MIMIC-IV) performs well on the source data -- closing the distribution gap without retraining the predictor.

## Key Results

Five clinical prediction tasks (3 classification, 2 regression). Deltas are improvements over the frozen eICU-applied baseline.

### Classification

| Task | AUCROC Baseline | AUCROC Translated | AUCROC Delta | AUCPR Delta | eICU-native LSTM (AUCROC / AUCPR) |
|---|---|---|---|---|---|
| **Mortality** (per-stay) | 0.8079 | **0.8555** | **+0.0476** | **+0.0546** | 0.855 / 0.357 |
| **AKI** (per-timestep) | 0.8558 | **0.9114** | **+0.0556** | **+0.1608** | 0.902 / 0.699 |
| **Sepsis** (per-timestep) | 0.7159 | **0.7671** | **+0.0512** | **+0.0225** | 0.740 / 0.040 |

### Regression

| Task | Baseline MAE | Translated MAE | Delta | eICU-native LSTM |
|---|---|---|---|---|
| **Length of Stay** (per-timestep) | 42.5h | **39.2h** | **-3.3h** | 39.2h |
| **Kidney Function** (per-stay) | 0.403 mg/dL | **0.382 mg/dL** | **-0.021** | 0.28 mg/dL |

All 3 classification tasks surpass the eICU-native LSTM AUCROC reference (YAIB, van de Water et al., ICLR 2024). AKI surpasses both the eICU-native LSTM in AUCROC and AUCPR. LoS matches the eICU-native reference.

## Project Structure

```
EHR_Translator/
  deep_pipeline/      # Main codebase (translator training + evaluation)
  poc_translator/     # Early proof-of-concept experiments
  optional_changes/   # Exploratory modifications
  Utils/              # Shared utilities
  runs/               # Experiment outputs
```

All active development is in `deep_pipeline/`. See [`deep_pipeline/docs/`](deep_pipeline/docs/) for detailed documentation.

## Quick Start

```bash
cd deep_pipeline
pip install -e .

# Train a retrieval translator (recommended architecture)
python run.py train_translator --config configs/<task_config>.json

# Translate source data and evaluate against frozen baseline
python run.py translate_and_eval --config configs/<task_config>.json --output_parquet results.parquet

# Train + evaluate in one step
python run.py train_and_eval --config configs/<task_config>.json --output_parquet results.parquet
```

## Translator Architectures

Three paradigms, all operating upstream of the frozen LSTM baseline:

| Paradigm | Key Idea | Best For |
|---|---|---|
| **Delta** (`EHRTranslator`) | Additive residuals on input features | Sepsis (with feature gate) |
| **Shared Latent** (`SharedLatentTranslator`) | Encoder-decoder through shared latent space | Mortality (with feature gate) |
| **Retrieval** (`RetrievalTranslator`) | k-NN lookup against encoded MIMIC windows + cross-attention | Universal (best or tied on all 5 tasks) |

The **Retrieval Translator V5** (with `n_cross_layers=3`) is the recommended universal architecture.

## Documentation

- [Architecture overview](deep_pipeline/docs/architecture.md)
- [Retrieval translator design](deep_pipeline/docs/retrieval_translator_architecture.md)
- [Comprehensive results](deep_pipeline/docs/comprehensive_results_summary.md)
- [YAIB reference baselines](deep_pipeline/docs/yaib_reference_baselines.md)
