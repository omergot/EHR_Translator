#!/usr/bin/env python3
"""
Cycle-VAE Model for Domain Translation
PyTorch Lightning implementation with shared latent space and domain-specific decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.distributions import Normal
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    """Shared encoder for both domains"""
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        """
        Initialize encoder
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        logger.info(f"Created encoder: {input_dim} -> {hidden_dims} -> {latent_dim}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        features = self.feature_extractor(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z

class Decoder(nn.Module):
    """Domain-specific decoder"""
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        """
        Initialize decoder
        
        Args:
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
        """
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_generator = nn.Sequential(*layers)
        
        # Output layer
        self.fc_out = nn.Linear(prev_dim, output_dim)
        
        logger.info(f"Created decoder: {latent_dim} -> {hidden_dims} -> {output_dim}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            x_recon: Reconstructed features
        """
        features = self.feature_generator(z)
        x_recon = self.fc_out(features)
        
        return x_recon

class CycleVAE(pl.LightningModule):
    """Cycle-VAE model for domain translation"""
    
    def __init__(self, config: Dict, feature_spec: Dict):
        """
        Initialize Cycle-VAE
        
        Args:
            config: Configuration dictionary
            feature_spec: Feature specification dictionary
        """
        super().__init__()
        self.config = config
        self.feature_spec = feature_spec
        
        # Training parameters (ensure they are proper numeric types)
        self.lr = float(config['training']['lr'])
        self.latent_dim = int(config['training']['latent_dim'])
        self.kl_weight = float(config['training']['kl_weight'])
        self.cycle_weight = float(config['training']['cycle_weight'])
        self.rec_weight = float(config['training']['rec_weight'])
        self.mmd_weight = float(config['training']['mmd_weight'])
        self.kl_warmup_epochs = int(config['training']['kl_warmup_epochs'])
        self.weight_decay = float(config['training']['weight_decay'])
        
        # Feature dimensions
        self.numeric_dim = len(feature_spec['numeric_features'])
        self.missing_dim = len(feature_spec['missing_features'])
        self.input_dim = self.numeric_dim + self.missing_dim
        
        # Architecture parameters
        hidden_dims = [256, 128, 64]  # Can be made configurable
        
        # Initialize networks
        self.encoder = Encoder(self.input_dim, hidden_dims, self.latent_dim)
        self.decoder_mimic = Decoder(self.latent_dim, hidden_dims[::-1], self.input_dim)
        self.decoder_eicu = Decoder(self.latent_dim, hidden_dims[::-1], self.input_dim)
        
        # Prior distribution
        self.prior = Normal(0, 1)
        
        logger.info(f"Initialized Cycle-VAE with input_dim={self.input_dim}, latent_dim={self.latent_dim}")
    
    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> Dict:
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, input_dim]
            domain: Domain labels [batch_size] (0=eICU, 1=MIMIC)
            
        Returns:
            Dictionary with outputs
        """
        # Encode
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode based on domain
        mimic_mask = (domain == 1)
        eicu_mask = (domain == 0)
        
        x_recon = torch.zeros_like(x)
        
        if mimic_mask.any():
            x_recon_mimic = self.decoder_mimic(z[mimic_mask])
            x_recon[mimic_mask] = x_recon_mimic.to(x_recon.dtype)
        
        if eicu_mask.any():
            x_recon_eicu = self.decoder_eicu(z[eicu_mask])
            x_recon[eicu_mask] = x_recon_eicu.to(x_recon.dtype)
        
        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'x_recon': x_recon
        }
    
    def cycle_forward(self, x: torch.Tensor, source_domain: int, target_domain: int) -> Dict:
        """
        Cycle forward pass for domain translation
        
        Args:
            x: Input features
            source_domain: Source domain (0=eICU, 1=MIMIC)
            target_domain: Target domain (0=eICU, 1=MIMIC)
            
        Returns:
            Dictionary with cycle outputs
        """
        # Encode
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode to target domain
        if target_domain == 1:  # MIMIC
            x_translated = self.decoder_mimic(z)
        else:  # eICU
            x_translated = self.decoder_eicu(z)
        
        # Encode translated data
        mu_cycle, logvar_cycle = self.encoder(x_translated)
        z_cycle = self.encoder.reparameterize(mu_cycle, logvar_cycle)
        
        # Decode back to source domain
        if source_domain == 1:  # MIMIC
            x_cycle = self.decoder_mimic(z_cycle)
        else:  # eICU
            x_cycle = self.decoder_eicu(z_cycle)
        
        return {
            'x_translated': x_translated,
            'x_cycle': x_cycle,
            'z': z,
            'z_cycle': z_cycle
        }
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss"""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss
    
    def compute_reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss (MSE)"""
        return F.mse_loss(x_recon, x, reduction='sum')
    
    def compute_cycle_loss(self, x: torch.Tensor, x_cycle: torch.Tensor) -> torch.Tensor:
        """Compute cycle consistency loss"""
        return F.mse_loss(x_cycle, x, reduction='sum')
    
    def compute_mmd_loss(self, z_mimic: torch.Tensor, z_eicu: torch.Tensor) -> torch.Tensor:
        """FIXED: Compute Maximum Mean Discrepancy loss with numerical stability"""
        if z_mimic.size(0) == 0 or z_eicu.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # FIXED: Use stable sigma computation
        if not hasattr(self, 'mmd_sigma') or self.mmd_sigma is None:
            # Initialize sigma based on data statistics (once)
            with torch.no_grad():
                # Use median of pairwise distances but with stability
                sample_size = min(100, z_mimic.size(0), z_eicu.size(0))
                distances = torch.cdist(z_mimic[:sample_size], z_eicu[:sample_size])
                self.mmd_sigma = torch.median(distances).item()
                
                # Ensure sigma is reasonable
                if self.mmd_sigma < 1e-6:
                    self.mmd_sigma = 1.0
                    
                logger.info(f"MMD sigma initialized to: {self.mmd_sigma:.6f}")
        
        # MMD computation with fixed sigma
        mmd = self._rbf_mmd(z_mimic, z_eicu, self.mmd_sigma)
        
        # FIXED: Clamp to prevent explosion
        mmd = torch.clamp(mmd, min=0, max=10.0)
        
        return mmd
    
    def _rbf_mmd(self, x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """FIXED: Compute RBF MMD with memory optimization and numerical stability"""
        # Sample if tensors are too large (memory optimization)
        max_samples = 500
        if x.size(0) > max_samples:
            idx_x = torch.randperm(x.size(0), device=x.device)[:max_samples]
            x_sample = x[idx_x]
        else:
            x_sample = x
            
        if y.size(0) > max_samples:
            idx_y = torch.randperm(y.size(0), device=y.device)[:max_samples]
            y_sample = y[idx_y]
        else:
            y_sample = y
        
        # Compute kernel matrices with numerical stability
        gamma = 1.0 / (2 * sigma**2)
        
        xx_dist = torch.cdist(x_sample, x_sample)
        yy_dist = torch.cdist(y_sample, y_sample)
        xy_dist = torch.cdist(x_sample, y_sample)
        
        # RBF kernels
        K_xx = torch.exp(-gamma * xx_dist)
        K_yy = torch.exp(-gamma * yy_dist)
        K_xy = torch.exp(-gamma * xy_dist)
        
        # Remove diagonal for unbiased estimation
        m, n = K_xx.size(0), K_yy.size(0)
        if m > 1:
            K_xx = K_xx - torch.diag(torch.diag(K_xx))
            xx_term = K_xx.sum() / (m * (m - 1))
        else:
            xx_term = K_xx.mean()
            
        if n > 1:
            K_yy = K_yy - torch.diag(torch.diag(K_yy))
            yy_term = K_yy.sum() / (n * (n - 1))
        else:
            yy_term = K_yy.mean()
        
        xy_term = K_xy.mean()
        
        mmd = xx_term + yy_term - 2 * xy_term
        return mmd
    
    def get_kl_weight(self) -> float:
        """Get KL weight with warmup"""
        if self.current_epoch < self.kl_warmup_epochs:
            return self.kl_weight * (self.current_epoch + 1) / self.kl_warmup_epochs
        return self.kl_weight
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step"""
        # Extract data
        x_numeric = batch['numeric']
        x_missing = batch['missing']
        domain = batch['domain']
        
        # Combine features
        x = torch.cat([x_numeric, x_missing], dim=1)
        
        # Separate by domain
        mimic_mask = (domain == 1)
        eicu_mask = (domain == 0)
        
        # Forward pass
        outputs = self.forward(x, domain)
        z, mu, logvar, x_recon = outputs['z'], outputs['mu'], outputs['logvar'], outputs['x_recon']
        
        # Compute losses
        rec_loss = self.compute_reconstruction_loss(x, x_recon)
        kl_loss = self.compute_kl_loss(mu, logvar)
        
        # Cycle consistency loss
        cycle_loss = torch.tensor(0.0, device=self.device)
        if mimic_mask.any() and eicu_mask.any():
            # eICU -> MIMIC -> eICU
            if eicu_mask.any():
                x_eicu = x[eicu_mask]
                cycle_out_eicu = self.cycle_forward(x_eicu, 0, 1)
                cycle_loss += self.compute_cycle_loss(x_eicu, cycle_out_eicu['x_cycle'])
            
            # MIMIC -> eICU -> MIMIC
            if mimic_mask.any():
                x_mimic = x[mimic_mask]
                cycle_out_mimic = self.cycle_forward(x_mimic, 1, 0)
                cycle_loss += self.compute_cycle_loss(x_mimic, cycle_out_mimic['x_cycle'])
        
        # MMD loss
        mmd_loss = torch.tensor(0.0, device=self.device)
        if mimic_mask.any() and eicu_mask.any():
            z_mimic = z[mimic_mask]
            z_eicu = z[eicu_mask]
            mmd_loss = self.compute_mmd_loss(z_mimic, z_eicu)
        
        # Total loss
        kl_weight = self.get_kl_weight()
        total_loss = (
            self.rec_weight * rec_loss +
            kl_weight * kl_loss +
            self.cycle_weight * cycle_loss +
            self.mmd_weight * mmd_loss
        )
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_rec_loss', rec_loss)
        self.log('train_kl_loss', kl_loss)
        self.log('train_cycle_loss', cycle_loss)
        self.log('train_mmd_loss', mmd_loss)
        self.log('kl_weight', kl_weight)
        
        return total_loss
    
    # REMOVED: No validation step needed for train/test only pipeline
    def _validation_step_removed(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step"""
        # Extract data
        x_numeric = batch['numeric']
        x_missing = batch['missing']
        domain = batch['domain']
        
        # Combine features
        x = torch.cat([x_numeric, x_missing], dim=1)
        
        # Forward pass
        outputs = self.forward(x, domain)
        z, mu, logvar, x_recon = outputs['z'], outputs['mu'], outputs['logvar'], outputs['x_recon']
        
        # Compute losses
        rec_loss = self.compute_reconstruction_loss(x, x_recon)
        kl_loss = self.compute_kl_loss(mu, logvar)
        
        # Cycle consistency loss
        cycle_loss = torch.tensor(0.0, device=self.device)
        mimic_mask = (domain == 1)
        eicu_mask = (domain == 0)
        
        if mimic_mask.any() and eicu_mask.any():
            if eicu_mask.any():
                x_eicu = x[eicu_mask]
                cycle_out_eicu = self.cycle_forward(x_eicu, 0, 1)
                cycle_loss += self.compute_cycle_loss(x_eicu, cycle_out_eicu['x_cycle'])
            
            if mimic_mask.any():
                x_mimic = x[mimic_mask]
                cycle_out_mimic = self.cycle_forward(x_mimic, 1, 0)
                cycle_loss += self.compute_cycle_loss(x_mimic, cycle_out_mimic['x_cycle'])
        
        # MMD loss
        mmd_loss = torch.tensor(0.0, device=self.device)
        if mimic_mask.any() and eicu_mask.any():
            z_mimic = z[mimic_mask]
            z_eicu = z[eicu_mask]
            mmd_loss = self.compute_mmd_loss(z_mimic, z_eicu)
        
        # Total loss
        kl_weight = self.get_kl_weight()
        total_loss = (
            self.rec_weight * rec_loss +
            kl_weight * kl_loss +
            self.cycle_weight * cycle_loss +
            self.mmd_weight * mmd_loss
        )
        
        # Logging
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_rec_loss', rec_loss)
        self.log('val_kl_loss', kl_loss)
        self.log('val_cycle_loss', cycle_loss)
        self.log('val_mmd_loss', mmd_loss)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        x = torch.cat([batch['numeric'], batch['missing']], dim=1)
        domain = batch['domain']
        
        # Forward pass
        outputs = self.forward(x, domain)
        
        # Compute losses
        rec_loss = F.mse_loss(outputs['x_recon'], x)
        kl_loss = self.kl_divergence(outputs['mu'], outputs['logvar'])
        cycle_loss = self.cycle_consistency_loss(outputs, x, domain)
        mmd_loss = self.mmd_loss(outputs['z'], domain)
        
        # KL annealing weight
        kl_weight = min(1.0, self.current_epoch / self.kl_warmup_epochs) if self.kl_warmup_epochs > 0 else 1.0
        
        # Total loss
        total_loss = (
            self.rec_weight * rec_loss +
            kl_weight * kl_loss +
            self.cycle_weight * cycle_loss +
            self.mmd_weight * mmd_loss
        )
        
        # Logging
        self.log('test_loss', total_loss, prog_bar=True)
        self.log('test_rec_loss', rec_loss)
        self.log('test_kl_loss', kl_loss)
        self.log('test_cycle_loss', cycle_loss)
        self.log('test_mmd_loss', mmd_loss)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss'
            }
        }
    
    def translate_eicu_to_mimic(self, x_eicu: torch.Tensor) -> torch.Tensor:
        """Translate eICU data to MIMIC format"""
        self.eval()
        with torch.no_grad():
            # Encode
            mu, logvar = self.encoder(x_eicu)
            z = self.encoder.reparameterize(mu, logvar)
            
            # Decode to MIMIC
            x_mimic = self.decoder_mimic(z)
            
        return x_mimic
    
    def translate_mimic_to_eicu(self, x_mimic: torch.Tensor) -> torch.Tensor:
        """Translate MIMIC data to eICU format"""
        self.eval()
        with torch.no_grad():
            # Encode
            mu, logvar = self.encoder(x_mimic)
            z = self.encoder.reparameterize(mu, logvar)
            
            # Decode to eICU
            x_eicu = self.decoder_eicu(z)
            
        return x_eicu

def test_model():
    """Test model functionality"""
    logger.info("Testing Cycle-VAE model...")
    
    # Create sample config and feature spec
    config = {
        'training': {
            'lr': 1e-3,
            'latent_dim': 64,
            'kl_weight': 1e-3,
            'cycle_weight': 1.0,
            'rec_weight': 1.0,
            'mmd_weight': 0.1,
            'kl_warmup_epochs': 20,
            'weight_decay': 1e-5
        }
    }
    
    feature_spec = {
        'numeric_features': [f'feature_{i}_mean' for i in range(40)] + 
                          [f'feature_{i}_min' for i in range(40)] +
                          [f'feature_{i}_max' for i in range(40)] +
                          [f'feature_{i}_last' for i in range(40)],
        'missing_features': [f'feature_{i}_missing' for i in range(40)]
    }
    
    # Create model
    model = CycleVAE(config, feature_spec)
    
    # Create sample batch
    batch_size = 32
    x_numeric = torch.randn(batch_size, 160)  # 40 features * 4 (mean, min, max, last)
    x_missing = torch.randint(0, 2, (batch_size, 40)).float()  # Binary missing flags
    domain = torch.randint(0, 2, (batch_size,))  # Random domain labels
    
    batch = {
        'numeric': x_numeric,
        'missing': x_missing,
        'domain': domain
    }
    
    # Test forward pass
    outputs = model.forward(torch.cat([x_numeric, x_missing], dim=1), domain)
    
    logger.info(f"Model outputs: {outputs.keys()}")
    logger.info(f"Latent shape: {outputs['z'].shape}")
    logger.info(f"Reconstruction shape: {outputs['x_recon'].shape}")
    
    # Test training step
    loss = model.training_step(batch, 0)
    logger.info(f"Training loss: {loss.item()}")
    
    # Test translation
    x_eicu = torch.cat([x_numeric[:8], x_missing[:8]], dim=1)
    x_mimic_translated = model.translate_eicu_to_mimic(x_eicu)
    logger.info(f"Translation shape: {x_mimic_translated.shape}")
    
    logger.info("Model test completed successfully!")

if __name__ == "__main__":
    test_model()
