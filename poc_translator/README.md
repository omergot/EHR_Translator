# Cycle Encoder-Decoder for EHR Domain Translation

A PyTorch Lightning implementation of a Cycle Encoder-Decoder for translating between MIMIC-IV and eICU clinical datasets. This project enables domain adaptation between different ICU databases by learning a shared latent representation and domain-specific decoders.

**Author**: Omer Gotfrid (omer.gotfrid@campus.technion.ac.il)

## Overview

This repository implements a complete pipeline for:
- **Feature extraction** from MIMIC-IV and eICU databases using aligned POC (Point-of-Care) features
- **Data preprocessing** with domain-specific robust scaling and train/val/test splitting
- **Cycle Encoder-Decoder training** with shared latent space and domain-specific decoders
- **Comprehensive evaluation** including round-trip reconstruction, distributional tests, and latent space analysis

## Architecture

The Cycle Encoder-Decoder consists of:
- **Shared Encoder**: Maps both MIMIC and eICU data to a common latent space (deterministic encoding using mu)
- **Domain-Specific Decoders**: Separate decoders for MIMIC and eICU reconstruction
- **Three-Loss Training**:
  - **Reconstruction Loss**: Per-feature MSE on clinical features (masked by missing indicators)
  - **Cycle Consistency Loss**: Round-trip translation error on clinical features
  - **Conditional Wasserstein Loss**: 1-D Wasserstein distance on worst-K features, conditioned by age/gender demographics
- **Dynamic Architecture**: Automatically selects latent dimension and hidden layers based on input size
  - Small (< 100 features): latent_dim=16, hidden=[128, 64]
  - Medium (≥ 100 features): latent_dim=32, hidden=[512, 256, 128]

**Note**: While the model includes VAE components (mu, logvar, reparameterization), no KL divergence loss is applied during training. The model functions as a deterministic encoder-decoder for translation tasks.

## Repository Structure

```
poc_translator/
├── conf/
│   └── config.yml              # Configuration file
├── sql/
│   └── make_queries.py         # SQL query generator (legacy)
├── scripts/
│   ├── extract_poc_features.py # Feature extraction from databases
│   ├── compare_distributions.py # Distribution matching analysis
│   ├── compare_trained_vs_untrained.py # Training impact analysis
│   └── README.md               # Scripts documentation
├── data/
│   ├── raw_extractors.py       # Legacy data extraction (not used)
│   ├── *_preprocessed.csv      # Preprocessed train/val/test splits
│   └── split_info.json         # Data split metadata
├── src/
│   ├── preprocess.py           # Data preprocessing with robust scaling
│   ├── dataset.py              # PyTorch datasets and dataloaders
│   ├── model.py                # Cycle Encoder-Decoder implementation
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── comprehensive_evaluator.py # Patient-level evaluation
│   └── utils.py                # Utility functions
├── docs/                       # Development documentation and debug notes
├── evaluation/                 # Evaluation results and plots
├── scalers/                    # Domain-specific robust scalers
├── checkpoints/               # Model checkpoints
├── logs/                      # TensorBoard logs
├── requirements.txt           # Python dependencies
└── README.md                  # This file
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

Edit `conf/config.yml` with your database connections and paths:

```yaml
db:
  mimic_conn: "postgresql://postgres:postgres@localhost:5432/mimic"
  eicu_conn: "postgresql://postgres:postgres@localhost:5432/mimic"

paths:
  eicu_poc_csv: "/path/to/eicu_poc_features.csv"
  mimic_poc_csv: "/path/to/mimic_poc_features.csv"
  output_dir: "/path/to/output"
```

### 3. Database Setup

Ensure you have access to:
- MIMIC-IV database (PostgreSQL)
- eICU database (PostgreSQL)

## Usage

### Step 1: Extract POC Features

```bash
python scripts/extract_poc_features.py
```

This extracts aligned point-of-care features from both databases, including:
- Clinical features: HR, RR, SpO2, Temp, MAP, WBC, Na, Creat
- Demographic features: Age, Gender
- Missing indicators for each clinical feature

Output: `eicu_poc_features.csv` and `mimic_poc_features.csv`

### Step 2: Preprocess Data

```bash
# Standard preprocessing with robust scaling
python src/preprocess.py --config conf/config.yml --fit

# Optional: Run preprocessing audit
python src/preprocess.py --config conf/config.yml --audit

# Optional: Plot feature distributions
python src/preprocess.py --config conf/config.yml --plot-distributions
```

**Preprocessing flags:**
- `--fit`: Fit scalers and create train/val/test splits
- `--audit`: Analyze preprocessing quality and identify problematic features
- `--plot-distributions`: Generate distribution plots by dataset, gender, and age group

This creates:
- Domain-specific RobustScalers (saved in `scalers/`)
- Train/val/test splits (70%/15%/15%)
- Feature specification (`feature_spec.json`)
- Preprocessed CSV files in `data/`

### Step 3: Train Model

```bash
# Full training
python src/train.py --config conf/config.yml

# Dry run (1 epoch only)
python src/train.py --config conf/config.yml --dry-run

# Specify GPU device
python src/train.py --config conf/config.yml --gpu 0

# Data balancing strategy
python src/train.py --config conf/config.yml --balance oversample_minority

# MIMIC-only mode (same-domain validation)
python src/train.py --config conf/config.yml --mimic-only
```

**Training flags:**
- `--extract`: Force data extraction
- `--preprocess`: Force data preprocessing
- `--dry-run`: Run training for 1 epoch only
- `--gpu N`: Specify GPU device (0-3)
- `--balance`: Data balancing strategy (`oversample_minority`, `undersample_majority`, `max`)
- `--audit`: Run preprocessing audit
- `--mimic-only`: Train and test only on MIMIC data

### Step 4: Evaluate Model

```bash
# Standard evaluation
python src/evaluate.py --config conf/config.yml --model checkpoints/best.ckpt

# Comprehensive patient-level evaluation
python src/evaluate.py --config conf/config.yml --model checkpoints/best.ckpt --comprehensive

# MIMIC-only evaluation
python src/evaluate.py --config conf/config.yml --model checkpoints/best.ckpt --mimic-only

# Custom output directory
python src/evaluate.py --config conf/config.yml --model checkpoints/best.ckpt --output-dir custom_eval/
```

**Evaluation flags:**
- `--model`: Path to trained model checkpoint (required)
- `--comprehensive`: Run comprehensive patient-level evaluation
- `--mimic-only`: Evaluate only on MIMIC data
- `--output-dir`: Custom output directory for results

### Step 5: Analyze Distribution Matching (Optional)

```bash
# Compare trained vs untrained model
python scripts/compare_distributions.py

# Analyze training impact on distribution matching
python scripts/compare_trained_vs_untrained.py
```

Output: `dist.txt` with detailed distribution matching statistics

## Configuration

### Training Parameters

```yaml
training:
  epochs: 30                    # Training epochs
  batch_size: 128               # Batch size
  lr: 1e-3                      # Learning rate
  
  # Architecture (auto-selected based on input size)
  latent_dim_auto: true         # Auto-select latent_dim
  latent_dim: 256               # Used if latent_dim_auto: false
  use_residual_blocks: false    # Enable residual connections
  dropout_rate: 0.1             # Dropout rate
  
  # Loss weights (three-loss system)
  rec_weight: 0.2               # Reconstruction loss weight
  cycle_weight: 0.2             # Cycle consistency loss weight
  wasserstein_weight: 1.0       # Conditional Wasserstein loss weight
  
  # Conditional Wasserstein parameters
  wasserstein_compute_every_n_steps: 1     # Compute frequency
  wasserstein_min_group_size: 32           # Min samples per demographic group
  wasserstein_worst_k: 5                   # Number of worst features to target
  wasserstein_age_bucket_years: 10         # Age bucket size
  wasserstein_update_worst_every_n_epochs: 1  # Update frequency
  
  # Optimization
  gradient_clip_val: 10.0       # Gradient clipping threshold
  early_stop_patience: 15       # Early stopping patience
  weight_decay: 1e-4            # L2 regularization
```

### GPU Configuration

```yaml
gpu:
  device: 0                     # GPU device ID (0-3) or 'auto'
  precision: "16-mixed"         # Mixed precision training
  num_workers: 8                # DataLoader workers
```

### Preprocessing Configuration

```yaml
preprocessing:
  max_missing_pct: 0.5          # Remove features with >50% missing values
```

## Evaluation Metrics

The evaluation pipeline provides:

### 1. Round-trip Reconstruction
- **Per-feature MSE/MAE**: Reconstruction error for each clinical feature
- **Cycle MSE/MAE**: Round-trip translation error (eICU→MIMIC→eICU and MIMIC→eICU→MIMIC)
- **Relative errors**: Percentage-based error metrics

### 2. Distributional Tests
- **Kolmogorov-Smirnov (KS) test**: Per-feature distribution similarity
  - KS statistic < 0.1: distributions match well
  - Includes baseline (source vs target) and translation (translated vs target)
- **Wasserstein distance**: Per-feature distributional distance
- **Maximum Mean Discrepancy (MMD)**: Overall latent space distribution alignment

### 3. Latent Space Analysis
- **Latent distance metrics**: MSE and cosine similarity in latent space
- **Domain separation**: Analysis of latent space overlap between domains

### 4. Visualizations
- **Feature distribution comparisons**: Histograms of translated vs target distributions
- **Round-trip error plots**: Per-feature error analysis
- **UMAP embeddings**: Latent space visualization (if UMAP available)

**Note**: Downstream task evaluation (XGBoost classifier) is implemented but not actively used in current workflow.

## Output Files

After training and evaluation, you'll find:

```
output_dir/
├── data/                      # Preprocessed data splits
│   ├── train_mimic_preprocessed.csv
│   ├── train_eicu_preprocessed.csv
│   ├── val_mimic_preprocessed.csv
│   ├── val_eicu_preprocessed.csv
│   ├── test_mimic_preprocessed.csv
│   ├── test_eicu_preprocessed.csv
│   └── split_info.json
├── scalers/                   # Domain-specific scalers
│   ├── mimic_robust_scaler.pkl
│   └── eicu_robust_scaler.pkl
├── checkpoints/              # Model checkpoints
│   ├── best.ckpt
│   └── last.ckpt
├── logs/                     # TensorBoard logs
├── evaluation/               # Evaluation results
│   ├── round_trip_metrics.csv
│   ├── ks_by_feature.csv
│   ├── mmd_results.json
│   ├── evaluation_summary.json
│   ├── feature_distributions_*.png
│   └── round_trip_errors.png
├── docs/                     # Development documentation
├── feature_spec.json         # Feature specification
├── config_used.yml          # Configuration used for training
├── dist.txt                  # Distribution analysis output
└── preprocessing_audit_results.json  # Preprocessing audit
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/
```

View metrics:
- `train_loss`: Total training loss
- `train_rec_loss`: Reconstruction loss
- `train_cycle_loss`: Cycle consistency loss
- `train_wasserstein_loss`: Conditional Wasserstein loss
- `val_loss`: Validation loss

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
   - Reduce `num_workers` in GPU config
   - Use smaller dataset for testing

3. **CUDA Out of Memory**
   - Reduce batch size
   - Use CPU training: set `device: 'cpu'` in GPU config
   - Disable mixed precision: set `precision: "32"` in GPU config

4. **Missing Dependencies**
   - Install requirements: `pip install -r requirements.txt`
   - For GPU support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

5. **Preprocessing Issues**
   - Run audit to identify problematic features: `python src/preprocess.py --audit`
   - Check feature distributions: `python src/preprocess.py --plot-distributions`
   - Review `preprocessing_audit_results.json`

## Testing

Test individual components:

```bash
# Test configuration
python test_config.py

# Test complete pipeline
python test_pipeline.py

# Test dataset functionality
python src/dataset.py

# Test model forward pass
python src/model.py
```

## Development Documentation

See the `docs/` folder for detailed development notes, debugging logs, and implementation details from the project development process.

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
@misc{cycle_encoder_decoder_ehr,
  title={Cycle Encoder-Decoder for EHR Domain Translation},
  author={Omer Gotfrid},
  year={2024},
  email={omer.gotfrid@campus.technion.ac.il},
  institution={Technion - Israel Institute of Technology}
}
```

## Acknowledgments

- MIMIC-IV and eICU databases from PhysioNet
- PyTorch Lightning team
- The open-source community

## Contact

For questions and issues:
- Email: omer.gotfrid@campus.technion.ac.il
- Review the troubleshooting section
- Check the `docs/` folder for development notes
