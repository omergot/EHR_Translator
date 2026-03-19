# EHR Translator

Domain adaptation for EHR time-series data. Trains a **Translator** network to transform source-domain electronic health records (eICU) so that a **strictly frozen** target-domain LSTM baseline (trained on MIMIC-IV) performs well on the source data -- closing the distribution gap without retraining the predictor.

## Key Results

Five clinical prediction tasks (3 classification, 2 regression). Deltas are improvements over the frozen eICU-applied baseline.

| Task | Metric | Frozen Baseline | Translated | Delta | eICU-native LSTM |
|---|---|---|---|---|---|
| **Mortality** (per-stay) | AUCROC | 0.8079 | **0.8555** | **+0.0476** | 0.855 |
| **AKI** (per-timestep) | AUCROC | 0.8558 | **0.9114** | **+0.0556** | 0.902 |
| **Sepsis** (per-timestep) | AUCROC | 0.7159 | **0.7671** | **+0.0512** | 0.740 |
| **Length of Stay** (per-timestep) | MAE (hours) | 42.5 | **39.2** | **-3.3h** | 39.2 |
| **Kidney Function** (per-stay) | MAE (mg/dL) | 0.403 | **0.382** | **-0.021** | 0.28 |

All 3 classification tasks surpass the eICU-native LSTM reference (YAIB, van de Water et al., ICLR 2024). AKI also surpasses the MIMIC-native LSTM. LoS matches the eICU-native reference.

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
