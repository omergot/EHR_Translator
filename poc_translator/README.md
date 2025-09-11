# Cycle-VAE Domain Translator

A PyTorch Lightning implementation of a Cycle-VAE for translating between MIMIC and eICU clinical datasets. This project enables domain adaptation between different ICU databases by learning a shared latent representation and domain-specific decoders.

## Overview

This repository implements a complete pipeline for:
- **Feature extraction** from MIMIC-IV and eICU databases using aligned feature lists
- **Data preprocessing** with domain-specific scaling and train/val/test splitting
- **Cycle-VAE training** with shared latent space and domain-specific decoders
- **Comprehensive evaluation** including round-trip reconstruction, distributional tests, and downstream task performance

## Architecture

The Cycle-VAE consists of:
- **Shared Encoder**: Maps both MIMIC and eICU data to a common latent space
- **Domain-Specific Decoders**: Separate decoders for MIMIC and eICU reconstruction
- **Cycle Consistency**: Ensures round-trip translation preserves original data
- **MMD Alignment**: Aligns latent distributions between domains

## Repository Structure

```
poc_translator/
├── conf/
│   └── config.yml              # Configuration file
├── sql/
│   └── make_queries.py         # SQL query generator
├── data/
│   └── raw_extractors.py       # Data extraction from databases
├── notebooks/
│   └── quick_viz.ipynb         # Quick visualization notebook
├── src/
│   ├── preprocess.py           # Data preprocessing
│   ├── dataset.py              # PyTorch datasets and dataloaders
│   ├── model.py                # Cycle-VAE model implementation
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── utils.py                # Utility functions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `conf/config.yml` with your database connections:

```yaml
db:
  mimic_conn: "postgresql://postgres:postgres@localhost:5432/mimic"
  eicu_conn: "postgresql://postgres:postgres@localhost:5432/mimic"
  omop_schema: "omop"
  cohort_tables:
    mimic: "mimiciv_bsi_100_2h_test.__mimiciv_bsi_100_2h_cohort"
    eicu: "eicu_bsi_100_2h_test.__eicu_bsi_100_2h_cohort"

paths:
  aligned_eicu_csv: "/path/to/eicu_features_aligned.csv"
  aligned_mimic_csv: "/path/to/mimic_features_aligned.csv"
  output_dir: "/path/to/output"
```

### 3. Database Setup

Ensure you have access to:
- MIMIC-IV database (PostgreSQL)
- eICU database (PostgreSQL)

## Usage

### Step 1: Test Configuration

```bash
python test_config.py
```

This tests database connections and OMOP schema access.

### Step 2: Generate SQL Queries

```bash
python sql/make_queries.py
```

This generates SQL queries for feature extraction using the OMOP schema. Review and update `sql/placeholders.json` with correct concept_ids/columns for your schema.

### Step 2: Extract Raw Data

```bash
# Extract from databases
python data/raw_extractors.py

# Or create sample data for testing
python data/raw_extractors.py --sample
```

### Step 3: Preprocess Data

```bash
python src/preprocess.py --fit
```

This creates:
- Domain-specific scalers
- Train/val/test splits
- Feature specification

### Step 4: Train Model

```bash
# Full training
python src/train.py --config conf/config.yml

# Dry run (1 epoch)
python src/train.py --config conf/config.yml --dry-run
```

### Step 5: Evaluate Model

```bash
python src/evaluate.py --model checkpoints/best.ckpt
```

## Configuration

### Training Parameters

```yaml
training:
  epochs: 120
  batch_size: 128
  lr: 1e-3
  latent_dim: 64
  kl_weight: 1e-3
  cycle_weight: 1.0
  rec_weight: 1.0
  mmd_weight: 0.1
  kl_warmup_epochs: 20
  early_stop_patience: 15
```

### Model Architecture

- **Encoder**: MLP with layers [256, 128, 64]
- **Decoders**: MLP with layers [64, 128, 256]
- **Latent Dimension**: 64
- **Activation**: ReLU with BatchNorm and Dropout

## Evaluation Metrics

The evaluation pipeline provides:

### 1. Round-trip Reconstruction
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Per-feature reconstruction quality

### 2. Distributional Tests
- Kolmogorov-Smirnov (KS) test per feature
- Maximum Mean Discrepancy (MMD) for overall distribution
- Feature-wise significance testing

### 3. Downstream Evaluation
- XGBoost classifier trained on MIMIC
- Performance on:
  - MIMIC test (baseline)
  - eICU test (domain drop)
  - Translated eICU test (improvement)

### 4. Visualizations
- Feature distribution comparisons
- UMAP embeddings
- Round-trip error analysis

## Output Files

After training and evaluation, you'll find:

```
output_dir/
├── data/                      # Preprocessed data splits
├── scalers/                   # Domain-specific scalers
├── checkpoints/              # Model checkpoints
├── logs/                     # TensorBoard logs
├── evaluation/               # Evaluation results
│   ├── round_trip_metrics.csv
│   ├── ks_by_feature.csv
│   ├── mmd_results.json
│   ├── downstream_results.json
│   └── *.png                 # Visualization plots
├── feature_spec.json         # Feature specification
└── config_used.yml          # Configuration used for training
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Wandb (Optional)
Enable in config:
```yaml
logging:
  use_wandb: true
  project: "poc_translator"
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify database credentials in `config.yml`
   - Ensure PostgreSQL is running
   - Check network connectivity

2. **Memory Issues**
   - Reduce batch size in config
   - Use smaller subset of data for testing

3. **CUDA Out of Memory**
   - Reduce batch size
   - Use CPU training: set `accelerator: 'cpu'` in trainer

4. **Missing Dependencies**
   - Install requirements: `pip install -r requirements.txt`
   - For GPU support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Testing

Run individual component tests:

```bash
# Test dataset functionality
python src/dataset.py

# Test model functionality
python src/model.py

# Test utility functions
python src/utils.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cycle_vae_translator,
  title={Cycle-VAE Domain Translator for Clinical Data},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/poc_translator}
}
```

## Acknowledgments

- MIMIC-IV and eICU databases
- PyTorch Lightning team
- The open-source community

## Support

For questions and issues:
1. Check the troubleshooting section
2. Review the code documentation
3. Open an issue on GitHub
