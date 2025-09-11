#!/usr/bin/env python3
"""
Training Script for Cycle-VAE
Main entry point for training the domain translation model.
"""

import argparse
import yaml
import sys
import logging
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import wandb

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.model import CycleVAE
from src.dataset import CombinedDataModule
from src.preprocess import Preprocessor
from data.raw_extractors import create_sample_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config: dict):
    """Setup logging and wandb if enabled"""
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['project'],
            config=config
        )
        logger.info("Wandb logging enabled")

def create_callbacks(config: dict, output_dir: Path):
    """Create training callbacks"""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="cycle-vae-{epoch:02d}-{train_loss:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # SIMPLIFIED: Early stopping on training loss (no validation)
    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        mode="min",
        patience=config['training']['early_stop_patience'],
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    return callbacks

def create_logger(config: dict, output_dir: Path):
    """Create TensorBoard logger"""
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="cycle_vae",
        version=None
    )
    return logger

def check_data_availability(config: dict) -> bool:
    """Check if required data files exist"""
    output_dir = Path(config['paths']['output_dir'])
    
    required_files = [
        output_dir / "data" / "train_mimic_preprocessed.csv",
        output_dir / "data" / "train_eicu_preprocessed.csv",
        output_dir / "data" / "val_mimic_preprocessed.csv",
        output_dir / "data" / "val_eicu_preprocessed.csv",
        output_dir / "data" / "test_mimic_preprocessed.csv",
        output_dir / "data" / "test_eicu_preprocessed.csv",
        output_dir / "feature_spec.json"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        logger.warning(f"Missing required files: {missing_files}")
        return False
    
    logger.info("All required data files found")
    return True

def prepare_data(config: dict, force_preprocess: bool = False):
    """Prepare data for training"""
    output_dir = Path(config['paths']['output_dir'])
    
    # Check if POC features data exists
    poc_files = [
        Path(config['paths']['mimic_poc_csv']),
        Path(config['paths']['eicu_poc_csv'])
    ]
    
    if not all(f.exists() for f in poc_files):
        logger.error("POC features CSV files not found! Please run extract_poc_features.py first.")
        logger.error(f"Missing files: {[str(f) for f in poc_files if not f.exists()]}")
        raise FileNotFoundError("POC features CSV files are required for preprocessing")
    
    # Check if preprocessed data exists
    if not check_data_availability(config) or force_preprocess:
        logger.info("Preprocessing POC features data...")
        preprocessor = Preprocessor(config)
        feature_spec = preprocessor.preprocess()
        logger.info("Data preprocessing completed")
    else:
        logger.info("Using existing preprocessed data")

def load_feature_spec(config: dict) -> dict:
    """Load feature specification"""
    spec_path = Path(config['paths']['output_dir']) / "feature_spec.json"
    import json
    with open(spec_path, 'r') as f:
        feature_spec = json.load(f)
    return feature_spec

def train_model(config: dict, feature_spec: dict, dry_run: bool = False):
    """Train the Cycle-VAE model"""
    logger.info("Starting model training...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['data']['random_seed'])
    pl.seed_everything(config['data']['random_seed'])
    
    # Create output directory
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(config)
    
    # Create data module with FIXED balancing
    balance_strategy = config.get('balance_strategy', 'oversample_minority')
    data_module = CombinedDataModule(config, feature_spec, balance_strategy=balance_strategy)
    
    # Create model
    model = CycleVAE(config, feature_spec)
    
    # Create callbacks
    callbacks = create_callbacks(config, output_dir)
    
    # Create logger
    tb_logger = create_logger(config, output_dir)
    
    # GPU configuration
    use_gpu = torch.cuda.is_available()
    gpu_config = config.get('gpu', {})
    
    if use_gpu:
        # Get GPU device
        gpu_device = config.get('gpu_device', gpu_config.get('device', 0))  # Command line override or config
        if gpu_device == 'auto':
            devices = 'auto'
        else:
            devices = [int(gpu_device)] if isinstance(gpu_device, (int, str)) else gpu_device
        
        # Get precision
        precision = gpu_config.get('precision', '16-mixed')
        accelerator = 'gpu'
        
        logger.info(f"Using GPU - Device: {devices}, Precision: {precision}")
    else:
        devices = 'auto'
        precision = '32'
        accelerator = 'auto'
        logger.info("CUDA not available - using CPU")
    
    # Create trainer
    trainer_kwargs = {
        'max_epochs': 1 if dry_run else config['training']['epochs'],
        'callbacks': callbacks,
        'logger': tb_logger,
        'accelerator': accelerator,
        'devices': devices,
        'precision': precision,
        'gradient_clip_val': 1.0,
        'log_every_n_steps': config['logging']['log_every_n_steps'],
        'enable_progress_bar': True,
        'enable_model_summary': True,
    }
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Test model
    logger.info("Running final evaluation...")
    trainer.test(model, data_module)
    
    # Save final model
    final_model_path = output_dir / "checkpoints" / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save config used for training
    config_used_path = output_dir / "config_used.yml"
    with open(config_used_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Training config saved to {config_used_path}")
    
    return model, trainer

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Cycle-VAE model')
    parser.add_argument('--config', type=str, default='conf/config.yml', 
                       help='Path to configuration file')
    parser.add_argument('--extract', action='store_true', 
                       help='Force data extraction')
    parser.add_argument('--preprocess', action='store_true', 
                       help='Force data preprocessing')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run training for 1 epoch only')
    parser.add_argument('--gpu', type=int, default=None, 
                       help='GPU device to use')
    parser.add_argument('--balance', type=str, default='oversample_minority',
                       choices=['oversample_minority', 'undersample_majority', 'max'],
                       help='Data balancing strategy')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    config = load_config(config_path)
    
    # Override GPU setting if specified
    if args.gpu is not None:
        config['gpu_device'] = args.gpu
    
    # Override balance strategy if specified
    config['balance_strategy'] = args.balance
    
    logger.info(f"Using configuration: {config_path}")
    logger.info(f"Output directory: {config['paths']['output_dir']}")
    
    try:
        # Prepare data
        prepare_data(config, force_preprocess=args.preprocess)
        
        # Load feature specification
        feature_spec = load_feature_spec(config)
        total_features = len(feature_spec['all_features'])
        logger.info(f"Loaded feature specification with {total_features} features ({feature_spec['n_clinical_features']} clinical + {feature_spec['n_demographic_features']} demographic)")
        
        # Train model
        model, trainer = train_model(config, feature_spec, dry_run=args.dry_run)
        
        logger.info("Training completed successfully!")
        
        if args.dry_run:
            logger.info("Dry run completed - model trained for 1 epoch")
        else:
            logger.info("Full training completed")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
