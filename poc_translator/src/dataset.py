#!/usr/bin/env python3
"""
Dataset and DataModule for Cycle-VAE Training
Handles data loading and batching for MIMIC and eICU data.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import yaml
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureDataset(Dataset):
    """PyTorch Dataset for feature data"""
    
    def __init__(self, data: pd.DataFrame, domain: str = 'mimic', feature_spec: Optional[Dict] = None, 
                 split_for_cycle: bool = False):
        """
        Initialize dataset
        
        Args:
            data: DataFrame with preprocessed features
            domain: 'mimic' or 'eicu'
            feature_spec: Feature specification to maintain correct column order
            split_for_cycle: If True, split samples into alternating domains for cycle training
        """
        self.data = data
        self.domain = domain
        self.split_for_cycle = split_for_cycle
        
        # CRITICAL FIX: Use feature_spec to maintain correct column order from preprocessing!
        # DON'T sort features - this will scramble the order from the CSV!
        if feature_spec is not None:
            # Use the feature_spec order which matches the CSV column order
            self.numeric_features = [col for col in feature_spec['numeric_features'] if col in data.columns]
            self.missing_features = [col for col in feature_spec['missing_features'] if col in data.columns]
        else:
            # Fallback: Preserve CSV column order (DO NOT SORT!)
            self.numeric_features = [col for col in data.columns 
                                    if ('_mean' in col or '_min' in col or '_max' in col or '_std' in col) 
                                    or col in ['Age', 'Gender']]
            self.missing_features = [col for col in data.columns if '_missing' in col]
        
        # Convert to tensors (outliers already clipped in preprocessing)
        self.numeric_tensor = torch.FloatTensor(data[self.numeric_features].values)
        self.missing_tensor = torch.FloatTensor(data[self.missing_features].values)
        
        # Store metadata
        self.metadata = {
            'icu_stay_id': data['icu_stay_id'].values,
            'patient_id': data.get('patient_id', np.zeros(len(data))).values,
        }
        
        # Pre-assign domain labels for cycle mode to ensure exact 50-50 split even with shuffling
        if split_for_cycle:
            # Create domain labels: half 0s, half 1s
            n = len(data)
            self.domain_labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
            
            # CRITICAL: Shuffle the domain labels so 0s and 1s are randomly distributed
            # This ensures that when DataLoader samples any batch, it gets ~50-50 split
            np.random.seed(42)  # Fixed seed for reproducibility
            np.random.shuffle(self.domain_labels)
            
            logger.info(f"CYCLE MODE: Pre-assigned and shuffled domains - {(self.domain_labels == 0).sum()} as domain=0, {(self.domain_labels == 1).sum()} as domain=1")
        else:
            self.domain_labels = None
        
        logger.info(f"Created {domain} dataset with {len(data)} samples")
        logger.info(f"Numeric features: {len(self.numeric_features)}")
        logger.info(f"Missing features: {len(self.missing_features)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        # Determine domain label
        if self.split_for_cycle:
            # MIMIC-only cycle mode: Use pre-assigned domain labels
            # This ensures exact 50-50 split even with DataLoader shuffling
            domain_label = int(self.domain_labels[idx])
        else:
            # Normal mode: Use actual domain
            domain_label = 0 if self.domain == 'eicu' else 1  # 0 for eICU, 1 for MIMIC
        
        return {
            'numeric': self.numeric_tensor[idx],
            'missing': self.missing_tensor[idx],
            'domain': domain_label,
            'icu_stay_id': self.metadata['icu_stay_id'][idx],
            'patient_id': self.metadata['patient_id'][idx],
        }

class CombinedDataModule(pl.LightningDataModule):
    """FIXED: PyTorch Lightning DataModule for combined MIMIC and eICU data"""
    
    def __init__(self, config: Dict, feature_spec: Dict, balance_strategy: str = 'oversample_minority'):
        """
        Initialize DataModule with FIXED balancing
        
        Args:
            config: Configuration dictionary
            feature_spec: Feature specification dictionary
            balance_strategy: How to handle data imbalance
        """
        super().__init__()
        self.config = config
        self.feature_spec = feature_spec
        self.balance_strategy = balance_strategy
        self.output_dir = Path(config['paths']['output_dir'])
        self.mimic_only = config.get('mimic_only', False)
        
        if self.mimic_only:
            logger.info("MIMIC-ONLY MODE: Will use only MIMIC data for training and testing")
        else:
            logger.info(f"Using balance strategy: {balance_strategy}")
        
        # Data paths
        self.data_dir = self.output_dir / "data"
        
        # SIMPLIFIED: Initialize datasets (train/test only)
        self.train_mimic_dataset = None
        self.train_eicu_dataset = None
        self.test_mimic_dataset = None
        self.test_eicu_dataset = None
        
        # Batch size
        self.batch_size = config['training']['batch_size']
        
        logger.info(f"Initialized DataModule with batch size {self.batch_size}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing"""
        logger.info(f"Setting up datasets for stage: {stage}")
        
        if stage == 'fit' or stage is None:
            # Load training data
            train_mimic_data = pd.read_csv(self.data_dir / "train_mimic_preprocessed.csv")
            
            if self.mimic_only:
                # MIMIC-only mode: Split MIMIC data into alternating domains for cycle testing
                self.train_mimic_dataset = FeatureDataset(
                    train_mimic_data, 'mimic', self.feature_spec, split_for_cycle=True
                )
                self.train_eicu_dataset = None
                logger.info(f"Training dataset (MIMIC-only with cycle split) - MIMIC: {len(self.train_mimic_dataset)}")
            else:
                # Standard mode: Load both MIMIC and eICU
                self.train_mimic_dataset = FeatureDataset(train_mimic_data, 'mimic', self.feature_spec)
                train_eicu_data = pd.read_csv(self.data_dir / "train_eicu_preprocessed.csv")
                self.train_eicu_dataset = FeatureDataset(train_eicu_data, 'eicu', self.feature_spec)
                logger.info(f"Training datasets - MIMIC: {len(self.train_mimic_dataset)}, eICU: {len(self.train_eicu_dataset)}")
            
        if stage == 'test' or stage is None:
            # Load test data
            test_mimic_data = pd.read_csv(self.data_dir / "test_mimic_preprocessed.csv")
            
            if self.mimic_only:
                # MIMIC-only mode: Split MIMIC data into alternating domains for cycle testing
                self.test_mimic_dataset = FeatureDataset(
                    test_mimic_data, 'mimic', self.feature_spec, split_for_cycle=True
                )
                self.test_eicu_dataset = None
                logger.info(f"Test dataset (MIMIC-only with cycle split) - MIMIC: {len(self.test_mimic_dataset)}")
            else:
                # Standard mode: Load both MIMIC and eICU
                self.test_mimic_dataset = FeatureDataset(test_mimic_data, 'mimic', self.feature_spec)
                test_eicu_data = pd.read_csv(self.data_dir / "test_eicu_preprocessed.csv")
                self.test_eicu_dataset = FeatureDataset(test_eicu_data, 'eicu', self.feature_spec)
                logger.info(f"Test datasets - MIMIC: {len(self.test_mimic_dataset)}, eICU: {len(self.test_eicu_dataset)}")
    
    def train_dataloader(self):
        """Create training dataloader with FIXED balancing"""
        gpu_config = self.config.get('gpu', {})
        num_workers = gpu_config.get('num_workers', 4)
        
        return CombinedDataLoader(
            mimic_dataset=self.train_mimic_dataset,
            eicu_dataset=self.train_eicu_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            balance_strategy=self.balance_strategy,  # FIXED: Pass balance strategy
            mimic_only=self.mimic_only  # Pass MIMIC-only flag
        )
    
    # REMOVED: No validation dataloader needed
    
    def test_dataloader(self):
        """Create test dataloader"""
        gpu_config = self.config.get('gpu', {})
        num_workers = gpu_config.get('num_workers', 4)
        
        return CombinedDataLoader(
            mimic_dataset=self.test_mimic_dataset,
            eicu_dataset=self.test_eicu_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            mimic_only=self.mimic_only  # Pass MIMIC-only flag
        )
    
    def get_feature_dimensions(self):
        """Get feature dimensions for model initialization"""
        if self.train_mimic_dataset is None:
            raise ValueError("Datasets not initialized. Call setup() first.")
        
        numeric_dim = self.train_mimic_dataset.numeric_tensor.shape[1]
        missing_dim = self.train_mimic_dataset.missing_tensor.shape[1]
        
        return {
            'numeric_dim': numeric_dim,
            'missing_dim': missing_dim,
            'total_dim': numeric_dim + missing_dim
        }

class CombinedDataLoader:
    """Custom DataLoader that alternates between MIMIC and eICU batches"""
    
    def __init__(self, mimic_dataset: FeatureDataset, eicu_dataset: Optional[FeatureDataset], 
                 batch_size: int, shuffle: bool = True, num_workers: int = 4, 
                 balance_strategy: str = 'oversample_minority', mimic_only: bool = False):
        """
        Initialize combined dataloader with FIXED balancing
        
        Args:
            mimic_dataset: MIMIC dataset
            eicu_dataset: eICU dataset (can be None in MIMIC-only mode)
            batch_size: Batch size for each domain
            shuffle: Whether to shuffle data
            num_workers: Number of workers for data loading
            balance_strategy: How to handle imbalance ('oversample_minority', 'undersample_majority', 'max')
            mimic_only: If True, only use MIMIC data
        """
        self.mimic_dataset = mimic_dataset
        self.eicu_dataset = eicu_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.balance_strategy = balance_strategy
        self.mimic_only = mimic_only
        
        if mimic_only:
            logger.info("MIMIC-only DataLoader: Using only MIMIC data")
        else:
            # FIXED: Calculate imbalance and log it
            mimic_size = len(mimic_dataset)
            eicu_size = len(eicu_dataset) if eicu_dataset else 0
            imbalance_ratio = mimic_size / eicu_size if eicu_size > 0 else float('inf')
            logger.info(f"Data imbalance ratio: {imbalance_ratio:.2f}x (MIMIC/eICU)")
            logger.info(f"Using balance strategy: {balance_strategy}")
        
        # Create individual dataloaders
        self.mimic_loader = DataLoader(
            mimic_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            drop_last=True
        )
        
        if not mimic_only and eicu_dataset is not None:
            self.eicu_loader = DataLoader(
                eicu_dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=num_workers,
                drop_last=True
            )
        else:
            self.eicu_loader = None
        
        # Create iterators
        self.mimic_iter = None
        self.eicu_iter = None
        self.reset_iterators()
        
        logger.info(f"Created CombinedDataLoader with batch size {batch_size}")
    
    def reset_iterators(self):
        """Reset dataloader iterators"""
        self.mimic_iter = iter(self.mimic_loader)
        if self.eicu_loader is not None:
            self.eicu_iter = iter(self.eicu_loader)
        else:
            self.eicu_iter = None
    
    def __iter__(self):
        """Create iterator for combined batches"""
        self.reset_iterators()
        return self
    
    def __next__(self):
        """Get next batch with FIXED balancing logic"""
        try:
            # Get MIMIC batch
            try:
                mimic_batch = next(self.mimic_iter)
            except StopIteration:
                # Reset iterators for next epoch
                self.reset_iterators()
                raise StopIteration
            
            # MIMIC-only mode: split batch by alternating domains to match normal batch size
            if self.mimic_only or self.eicu_iter is None:
                # In MIMIC-only mode with split_for_cycle=True, the batch already has
                # alternating domains (0, 1, 0, 1, ...). Split them to match the
                # behavior of normal mode where we have separate MIMIC and eICU batches.
                # This keeps batch size consistent: normal mode returns 2*batch_size,
                # so MIMIC-only mode should too.
                domain = mimic_batch['domain']
                domain_0_mask = (domain == 0)
                domain_1_mask = (domain == 1)
                
                # Create two "fake" batches from the single MIMIC batch
                batch_0 = {
                    'numeric': mimic_batch['numeric'][domain_0_mask],
                    'missing': mimic_batch['missing'][domain_0_mask],
                    'domain': mimic_batch['domain'][domain_0_mask],
                    'icu_stay_id': mimic_batch['icu_stay_id'][domain_0_mask.cpu().numpy()],
                    'patient_id': mimic_batch['patient_id'][domain_0_mask.cpu().numpy()],
                }
                
                batch_1 = {
                    'numeric': mimic_batch['numeric'][domain_1_mask],
                    'missing': mimic_batch['missing'][domain_1_mask],
                    'domain': mimic_batch['domain'][domain_1_mask],
                    'icu_stay_id': mimic_batch['icu_stay_id'][domain_1_mask.cpu().numpy()],
                    'patient_id': mimic_batch['patient_id'][domain_1_mask.cpu().numpy()],
                }
                
                # Combine them just like normal mode (domain_0 + domain_1)
                combined_batch = self.combine_batches(batch_0, batch_1)
                return combined_batch
            
            # Standard mode: combine MIMIC and eICU batches
            # FIXED: Handle imbalanced data by restarting exhausted iterators
            try:
                eicu_batch = next(self.eicu_iter)
            except StopIteration:
                # Restart eICU iterator if exhausted (oversampling)
                self.eicu_iter = iter(self.eicu_loader)
                eicu_batch = next(self.eicu_iter)
            
            # Combine batches
            combined_batch = self.combine_batches(mimic_batch, eicu_batch)
            
            return combined_batch
            
        except StopIteration:
            # Reset iterators for next epoch
            self.reset_iterators()
            raise StopIteration
    
    def combine_batches(self, mimic_batch: Dict, eicu_batch: Dict) -> Dict:
        """Combine MIMIC and eICU batches"""
        combined = {}
        
        # Combine numeric features
        combined['numeric'] = torch.cat([mimic_batch['numeric'], eicu_batch['numeric']], dim=0)
        
        # Combine missing features
        combined['missing'] = torch.cat([mimic_batch['missing'], eicu_batch['missing']], dim=0)
        
        # Combine domain labels
        combined['domain'] = torch.cat([mimic_batch['domain'], eicu_batch['domain']], dim=0)
        
        # Combine metadata
        combined['icu_stay_id'] = np.concatenate([mimic_batch['icu_stay_id'], eicu_batch['icu_stay_id']])
        combined['patient_id'] = np.concatenate([mimic_batch['patient_id'], eicu_batch['patient_id']])
        
        return combined
    
    def __len__(self):
        """Return number of batches per epoch - FIXED to use max instead of min"""
        if self.mimic_only or self.eicu_loader is None:
            return len(self.mimic_loader)
        # CRITICAL FIX: Use max instead of min to get proper training
        return max(len(self.mimic_loader), len(self.eicu_loader))

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    logger.info("Creating sample dataset for testing...")
    
    # Create sample data
    n_samples = 100
    n_features = 40
    
    # Sample MIMIC data
    mimic_data = {
        'icustay_id': range(1, n_samples + 1),
        'subject_id': np.random.randint(1, 1000, n_samples),
        'hadm_id': np.random.randint(1, 500, n_samples)
    }
    
    # Add feature columns
    for i in range(n_features):
        mimic_data[f'feature_{i}_mean'] = np.random.normal(0, 1, n_samples)
        mimic_data[f'feature_{i}_min'] = np.random.normal(-1, 0.5, n_samples)
        mimic_data[f'feature_{i}_max'] = np.random.normal(1, 0.5, n_samples)
        mimic_data[f'feature_{i}_last'] = np.random.normal(0, 1, n_samples)
        mimic_data[f'feature_{i}_missing'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    mimic_df = pd.DataFrame(mimic_data)
    
    # Sample eICU data
    eicu_data = {
        'icustay_id': range(1, n_samples + 1)
    }
    
    # Add feature columns
    for i in range(n_features):
        eicu_data[f'feature_{i}_mean'] = np.random.normal(0, 1, n_samples)
        eicu_data[f'feature_{i}_min'] = np.random.normal(-1, 0.5, n_samples)
        eicu_data[f'feature_{i}_max'] = np.random.normal(1, 0.5, n_samples)
        eicu_data[f'feature_{i}_last'] = np.random.normal(0, 1, n_samples)
        eicu_data[f'feature_{i}_missing'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    eicu_df = pd.DataFrame(eicu_data)
    
    return mimic_df, eicu_df

def test_dataset():
    """Test dataset functionality"""
    logger.info("Testing dataset functionality...")
    
    # Create sample data
    mimic_df, eicu_df = create_sample_dataset()
    
    # Create datasets
    mimic_dataset = FeatureDataset(mimic_df, 'mimic')
    eicu_dataset = FeatureDataset(eicu_df, 'eicu')
    
    # Test single sample
    mimic_sample = mimic_dataset[0]
    eicu_sample = eicu_dataset[0]
    
    logger.info(f"MIMIC sample keys: {mimic_sample.keys()}")
    logger.info(f"MIMIC numeric shape: {mimic_sample['numeric'].shape}")
    logger.info(f"MIMIC missing shape: {mimic_sample['missing'].shape}")
    logger.info(f"MIMIC domain: {mimic_sample['domain']}")
    
    logger.info(f"eICU sample keys: {eicu_sample.keys()}")
    logger.info(f"eICU numeric shape: {eicu_sample['numeric'].shape}")
    logger.info(f"eICU missing shape: {eicu_sample['missing'].shape}")
    logger.info(f"eICU domain: {eicu_sample['domain']}")
    
    # Test dataloader
    dataloader = CombinedDataLoader(
        mimic_dataset=mimic_dataset,
        eicu_dataset=eicu_dataset,
        batch_size=16,
        shuffle=True
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    
    logger.info(f"Combined batch keys: {batch.keys()}")
    logger.info(f"Combined numeric shape: {batch['numeric'].shape}")
    logger.info(f"Combined missing shape: {batch['missing'].shape}")
    logger.info(f"Combined domain shape: {batch['domain'].shape}")
    
    logger.info("Dataset test completed successfully!")

if __name__ == "__main__":
    test_dataset()
