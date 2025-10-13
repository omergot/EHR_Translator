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
from typing import Dict, Tuple, Optional, List
import logging
import json
from scipy import stats
from sklearn.metrics import mean_squared_error
from pathlib import Path

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    """Shared encoder for both domains"""
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, 
                 use_residual_blocks: bool = False, dropout_rate: float = 0.1):
        """
        Initialize encoder
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
            use_residual_blocks: Use residual blocks (for medium architecture)
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.use_residual_blocks = use_residual_blocks
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        logger.info(f"Created encoder: {input_dim} -> {hidden_dims} -> {latent_dim}, "
                   f"residual_blocks={use_residual_blocks}")
    
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
        Reparameterization trick with numerical stability
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        # CRITICAL FIX: Clamp logvar to prevent explosion in exp()
        logvar_clamped = torch.clamp(logvar, min=-5, max=3)  # exp(3) = 20, exp(-5) = 0.007
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z

class Decoder(nn.Module):
    """IMPROVED: Domain-specific decoder with proper binary feature handling and skip connections"""
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, use_heteroscedastic: bool = True, 
                 numeric_dim: int = None, missing_dim: int = None, use_residual_blocks: bool = False, 
                 dropout_rate: float = 0.1):
        """
        Initialize decoder with optional heteroscedastic outputs and proper binary handling
        
        Args:
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
            use_heteroscedastic: If True, predict both mean and log-variance per feature
            numeric_dim: Number of numeric features (for identifying binary features)
            missing_dim: Number of missing indicator features (binary)
            use_residual_blocks: Use residual blocks (for medium architecture)
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.use_heteroscedastic = use_heteroscedastic
        self.numeric_dim = numeric_dim if numeric_dim is not None else output_dim
        self.missing_dim = missing_dim if missing_dim is not None else 0
        self.use_residual_blocks = use_residual_blocks
        
        # Learnable affine skip connection parameters
        self.skip_scale = nn.Parameter(torch.ones(output_dim))   # a: start near identity
        self.skip_bias = nn.Parameter(torch.zeros(output_dim))   # b: no shift initially
        
        layers = []
        prev_dim = latent_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_generator = nn.Sequential(*layers)
        
        if use_heteroscedastic:
            # Separate heads for mean and log-variance
            self.fc_mu = nn.Linear(prev_dim, output_dim)      # Mean prediction
            self.fc_logvar = nn.Linear(prev_dim, output_dim)  # Log-variance prediction
            logger.info(f"Created heteroscedastic decoder: {latent_dim} -> {hidden_dims} -> {output_dim} (mu + logvar), "
                       f"residual_blocks={use_residual_blocks}")
        else:
            # Standard single output
            self.fc_out = nn.Linear(prev_dim, output_dim)
            logger.info(f"Created standard decoder: {latent_dim} -> {hidden_dims} -> {output_dim}, numeric={numeric_dim}, binary={missing_dim}, "
                       f"residual_blocks={use_residual_blocks}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor or tuple:
        """
        Forward pass with proper sigmoid activation for binary features
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            If heteroscedastic: (mu, logvar) tuple
            If standard: x_recon tensor with sigmoid applied to binary features
        """
        features = self.feature_generator(z)
        
        if self.use_heteroscedastic:
            mu = self.fc_mu(features)
            logvar = self.fc_logvar(features)
            
            # FIXED: Tighter clamp for numerical stability (matches NLL function)
            logvar = torch.clamp(logvar, min=-5, max=3)  # Safer range: exp(5) ≈ 148 max
            
            return mu, logvar
        else:
            x_recon = self.fc_out(features)
            
            # Apply sigmoid to binary features (missing indicators + Gender is at numeric_dim-1)
            if self.missing_dim > 0:
                # Missing indicators are at the end: [numeric_features | missing_indicators]
                binary_start = self.numeric_dim
                binary_end = self.numeric_dim + self.missing_dim
                
                # Also apply sigmoid to Gender (assumed to be at numeric_dim - 1)
                # Gender sigmoid
                x_recon[:, self.numeric_dim - 1] = torch.sigmoid(x_recon[:, self.numeric_dim - 1])
                
                # Missing flags sigmoid
                x_recon[:, binary_start:binary_end] = torch.sigmoid(x_recon[:, binary_start:binary_end])
            
            return x_recon

class CycleVAE(pl.LightningModule):
    """SIMPLIFIED: Cycle-VAE model with only essential losses: reconstruction, cycle, and conditional 1-D Wasserstein"""
    
    def __init__(self, config: Dict, feature_spec: Dict):
        """
        Initialize Cycle-VAE with simplified loss structure
        
        Args:
            config: Configuration dictionary
            feature_spec: Feature specification dictionary
        """
        super().__init__()
        self.config = config
        self.feature_spec = feature_spec
        
        # Feature dimensions (CRITICAL: needed before buffer registration)
        self.numeric_dim = len(feature_spec['numeric_features'])
        self.missing_dim = len(feature_spec['missing_features'])
        self.input_dim = self.numeric_dim + self.missing_dim
        
        # Identify clinical vs demographic features
        self.demographic_features = feature_spec.get('demographic_features', ['Age', 'Gender'])
        self.clinical_features = feature_spec.get('clinical_features', [])
        
        # Get indices for different feature types
        self.demographic_indices = []
        self.missing_flag_indices = list(range(self.numeric_dim, self.numeric_dim + self.missing_dim))
        
        # Find demographic feature indices in numeric features
        for i, feature in enumerate(feature_spec['numeric_features']):
            if any(demo in feature for demo in self.demographic_features):
                self.demographic_indices.append(i)
        
        # Clinical feature indices (everything except demographics and missing flags)
        self.clinical_indices = [i for i in range(self.numeric_dim) if i not in self.demographic_indices]
        
        logger.info(f"Feature indices - Clinical: {len(self.clinical_indices)}, Demographics: {len(self.demographic_indices)}, Missing flags: {len(self.missing_flag_indices)}")
        
        # Removed model-side per-dataset normalization – rely on preprocessing only
        self.normalization_initialized = False
        
        # SIMPLIFIED: Only three training parameters (ensure they are proper numeric types)
        self.lr = float(config['training']['lr'])
        
        # Dynamic architecture based on input size
        latent_dim_auto = config['training'].get('latent_dim_auto', True)
        use_residual_blocks = config['training'].get('use_residual_blocks', False)
        dropout_rate = float(config['training'].get('dropout_rate', 0.1))
        
        if latent_dim_auto:
            if self.input_dim < 100:
                # Small architecture for <100 features
                self.latent_dim = 16
                hidden_dims = [128, 64]
                logger.info(f"Auto-selected SMALL architecture: input_dim={self.input_dim}, latent_dim={self.latent_dim}, hidden_dims={hidden_dims}")
            else:
                # Medium architecture for >=100 features
                self.latent_dim = 32
                hidden_dims = [512, 256, 128]
                logger.info(f"Auto-selected MEDIUM architecture: input_dim={self.input_dim}, latent_dim={self.latent_dim}, hidden_dims={hidden_dims}")
        else:
            # Use config-specified latent_dim and determine hidden_dims based on input size
            self.latent_dim = int(config['training']['latent_dim'])
            if self.input_dim < 100:
                hidden_dims = [128, 64]
            else:
                hidden_dims = [512, 256, 128]
            logger.info(f"Using config-specified latent_dim={self.latent_dim} with hidden_dims={hidden_dims}")
        
        self.rec_weight = float(config['training'].get('rec_weight', 1.0))
        self.cycle_weight = float(config['training'].get('cycle_weight', 1.0))
        self.wasserstein_weight = float(config['training'].get('wasserstein_weight', 1.0))
        
        # Conditional Wasserstein parameters
        self.wasserstein_compute_every_n_steps = int(config['training'].get('wasserstein_compute_every_n_steps', 5))
        self.wasserstein_min_group_size = int(config['training'].get('wasserstein_min_group_size', 16))
        self.wasserstein_worst_k = int(config['training'].get('wasserstein_worst_k', 10))
        self.wasserstein_age_bucket_years = int(config['training'].get('wasserstein_age_bucket_years', 10))
        self.wasserstein_update_worst_every_n_epochs = int(config['training'].get('wasserstein_update_worst_every_n_epochs', 1))
        
        # Track worst-performing features dynamically
        self.worst_feature_indices = []
        self.last_worst_update_epoch = -1
        
        # Other training parameters
        self.kl_warmup_epochs = int(config['training'].get('kl_warmup_epochs', 20))
        self.weight_decay = float(config['training']['weight_decay'])
        self.gradient_clip_val = config['training'].get('gradient_clip_val', 1.0)
        
        # For IQR-based relative error in evaluation
        self.feature_iqr = None  # Will be computed from training data
        
        # Architecture parameters
        self.use_heteroscedastic = False  # SIMPLIFIED: No heteroscedastic outputs
        
        # Initialize networks with dynamic architecture
        self.encoder = Encoder(self.input_dim, hidden_dims, self.latent_dim, 
                              use_residual_blocks=use_residual_blocks, dropout_rate=dropout_rate)
        self.decoder_mimic = Decoder(self.latent_dim, hidden_dims[::-1], self.input_dim, self.use_heteroscedastic,
                                      numeric_dim=self.numeric_dim, missing_dim=self.missing_dim,
                                      use_residual_blocks=use_residual_blocks, dropout_rate=dropout_rate)
        self.decoder_eicu = Decoder(self.latent_dim, hidden_dims[::-1], self.input_dim, self.use_heteroscedastic,
                                     numeric_dim=self.numeric_dim, missing_dim=self.missing_dim,
                                     use_residual_blocks=use_residual_blocks, dropout_rate=dropout_rate)
        
        # Prior distribution
        self.prior = Normal(0, 1)
        
        # CRITICAL FIX: Initialize weights conservatively to prevent initial instability
        self._init_weights()
        
        # Zero-initialize final decoder layers so residual starts at 0
        self._zero_init_decoder_final_layers()
        
        logger.info(f"Initialized Cycle-VAE with input_dim={self.input_dim}, latent_dim={self.latent_dim}")
    
    def _init_weights(self):
        """Initialize network weights conservatively for numerical stability"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Conservative initialization for all networks
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def _zero_init_decoder_final_layers(self):
        """
        Zero-initialize final decoder layers so residual starts at 0.
        This makes the initial model behavior: output = skip_scale * input + skip_bias
        Training gradually learns the residual correction.
        """
        if self.use_heteroscedastic:
            # Zero-init mu and logvar heads for heteroscedastic outputs
            self.decoder_mimic.fc_mu.weight.data.zero_()
            self.decoder_mimic.fc_mu.bias.data.zero_()
            self.decoder_mimic.fc_logvar.weight.data.zero_()
            self.decoder_mimic.fc_logvar.bias.data.zero_()
            
            self.decoder_eicu.fc_mu.weight.data.zero_()
            self.decoder_eicu.fc_mu.bias.data.zero_()
            self.decoder_eicu.fc_logvar.weight.data.zero_()
            self.decoder_eicu.fc_logvar.bias.data.zero_()
            
            logger.info("Zero-initialized final decoder layers (heteroscedastic mode: fc_mu and fc_logvar)")
        else:
            # Zero-init standard output
            self.decoder_mimic.fc_out.weight.data.zero_()
            self.decoder_mimic.fc_out.bias.data.zero_()
            self.decoder_eicu.fc_out.weight.data.zero_()
            self.decoder_eicu.fc_out.bias.data.zero_()
            
            logger.info("Zero-initialized final decoder layers (standard mode: fc_out)")
     
    def initialize_normalization(self, x_mimic: torch.Tensor, x_eicu: torch.Tensor):
        """No-op: model-side normalization removed; kept for backward compatibility."""
        return
    
    def normalize_features(self, x: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """No-op: rely on preprocessing normalization only."""
        return x
    
    def denormalize_features(self, x_norm: torch.Tensor, target_domain: int) -> torch.Tensor:
        """IMPROVEMENT 1: Denormalize features back to original scale"""
        if not self.normalization_initialized:
            return x_norm
        
        if target_domain == 1:  # MIMIC
            return x_norm * self.mimic_feature_std + self.mimic_feature_mean
        else:  # eICU
            return x_norm * self.eicu_feature_std + self.eicu_feature_mean

    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> Dict:
        """
        IMPROVED: Forward pass supporting heteroscedastic outputs
        
        Args:
            x: Input features [batch_size, input_dim]
            domain: Domain labels [batch_size] (0=eICU, 1=MIMIC)
            
        Returns:
            Dictionary with outputs
        """
        # Encode
        mu, logvar = self.encoder(x)
        # MAJOR FIX: Relax latent constraints to preserve domain differences for classifier
        mu = torch.clamp(mu, min=-15, max=15)  # WIDER: Allow domain-specific latent patterns
        logvar = torch.clamp(logvar, min=-5, max=3)  # Keep logvar range tight for stability
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            mu = torch.zeros_like(mu)
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            logvar = torch.full_like(logvar, -2.0)
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode based on domain
        mimic_mask = (domain == 1)
        eicu_mask = (domain == 0)
        
        if self.use_heteroscedastic:
            # Handle heteroscedastic outputs
            x_recon_mu = torch.zeros_like(x)
            x_recon_logvar = torch.zeros_like(x)
            
            if mimic_mask.any():
                mu_mimic, logvar_mimic = self.decoder_mimic(z[mimic_mask])
                # CRITICAL FIX: Tighter constraints for decoder outputs
                mu_mimic = torch.clamp(mu_mimic, min=-10, max=10)  # TIGHTENED: Conservative range for normalized features
                logvar_mimic = torch.clamp(logvar_mimic, min=-5, max=3)  # Keep logvar range tight for stability
                # Check for NaN/Inf and replace with safe values
                if torch.isnan(mu_mimic).any() or torch.isinf(mu_mimic).any():
                    mu_mimic = torch.zeros_like(mu_mimic)
                if torch.isnan(logvar_mimic).any() or torch.isinf(logvar_mimic).any():
                    logvar_mimic = torch.full_like(logvar_mimic, -2.0)  # Safe logvar value
                x_recon_mu[mimic_mask] = mu_mimic.to(x_recon_mu.dtype)
                x_recon_logvar[mimic_mask] = logvar_mimic.to(x_recon_logvar.dtype)
            
            if eicu_mask.any():
                mu_eicu, logvar_eicu = self.decoder_eicu(z[eicu_mask])
                # CRITICAL FIX: Tighter constraints for decoder outputs
                mu_eicu = torch.clamp(mu_eicu, min=-10, max=10)  # TIGHTENED: Conservative range for normalized features
                logvar_eicu = torch.clamp(logvar_eicu, min=-5, max=3)  # Keep logvar range tight for stability
                # Check for NaN/Inf and replace with safe values
                if torch.isnan(mu_eicu).any() or torch.isinf(mu_eicu).any():
                    mu_eicu = torch.zeros_like(mu_eicu)
                if torch.isnan(logvar_eicu).any() or torch.isinf(logvar_eicu).any():
                    logvar_eicu = torch.full_like(logvar_eicu, -2.0)  # Safe logvar value
                x_recon_mu[eicu_mask] = mu_eicu.to(x_recon_mu.dtype) 
                x_recon_logvar[eicu_mask] = logvar_eicu.to(x_recon_logvar.dtype)
            
            return {
                'z': z,
                'mu': mu,
                'logvar': logvar,
                'x_recon': (x_recon_mu, x_recon_logvar)  # Tuple for heteroscedastic
            }
        else:
            # Standard reconstruction with skip connections
            x_recon = torch.zeros_like(x)
            
            if mimic_mask.any():
                # Get decoder output
                decoder_output_mimic = self.decoder_mimic(z[mimic_mask])
                
                # Apply skip connection: output = decoder(z) + (skip_scale * input + skip_bias)
                x_input_mimic = x[mimic_mask]
                skip_contribution = self.decoder_mimic.skip_scale * x_input_mimic + self.decoder_mimic.skip_bias
                x_recon_mimic = decoder_output_mimic + skip_contribution
                
                # DIAGNOSTIC: Check for extreme decoder outputs with detailed breakdown
                if x_recon_mimic.abs().max() > 20:
                    logger.warning(f"⚠️  LARGE DECODER OUTPUT (MIMIC) - DETAILED BREAKDOWN:")
                    logger.warning(f"  Final output: min={x_recon_mimic.min().item():.4f}, max={x_recon_mimic.max().item():.4f}, mean={x_recon_mimic.mean().item():.4f}")
                    logger.warning(f"  Decoder output (before skip): min={decoder_output_mimic.min().item():.4f}, max={decoder_output_mimic.max().item():.4f}, mean={decoder_output_mimic.mean().item():.4f}")
                    logger.warning(f"  Skip contribution: min={skip_contribution.min().item():.4f}, max={skip_contribution.max().item():.4f}, mean={skip_contribution.mean().item():.4f}")
                    logger.warning(f"  Input x: min={x_input_mimic.min().item():.4f}, max={x_input_mimic.max().item():.4f}, mean={x_input_mimic.mean().item():.4f}")
                    logger.warning(f"  Latent z: min={z[mimic_mask].min().item():.4f}, max={z[mimic_mask].max().item():.4f}, mean={z[mimic_mask].mean().item():.4f}")
                    logger.warning(f"  Skip params - scale: min={self.decoder_mimic.skip_scale.min().item():.4f}, max={self.decoder_mimic.skip_scale.max().item():.4f}, mean={self.decoder_mimic.skip_scale.mean().item():.4f}")
                    logger.warning(f"  Skip params - bias: min={self.decoder_mimic.skip_bias.min().item():.4f}, max={self.decoder_mimic.skip_bias.max().item():.4f}, mean={self.decoder_mimic.skip_bias.mean().item():.4f}")
                    
                    # Find which features have extreme values
                    extreme_mask = x_recon_mimic.abs() > 20
                    if extreme_mask.any():
                        extreme_indices = torch.where(extreme_mask)
                        logger.warning(f"  Extreme values found at {extreme_mask.sum().item()} locations")
                        # Log first few extreme feature indices
                        unique_features = torch.unique(extreme_indices[1][:10])
                        logger.warning(f"  Feature indices with extreme values (first 10): {unique_features.tolist()}")
                
                x_recon[mimic_mask] = x_recon_mimic.to(x_recon.dtype)
            
            if eicu_mask.any():
                # Get decoder output
                decoder_output_eicu = self.decoder_eicu(z[eicu_mask])
                
                # Apply skip connection: output = decoder(z) + (skip_scale * input + skip_bias)
                x_input_eicu = x[eicu_mask]
                skip_contribution = self.decoder_eicu.skip_scale * x_input_eicu + self.decoder_eicu.skip_bias
                x_recon_eicu = decoder_output_eicu + skip_contribution
                
                # DIAGNOSTIC: Check for extreme decoder outputs with detailed breakdown
                if x_recon_eicu.abs().max() > 20:
                    logger.warning(f"⚠️  LARGE DECODER OUTPUT (eICU) - DETAILED BREAKDOWN:")
                    logger.warning(f"  Final output: min={x_recon_eicu.min().item():.4f}, max={x_recon_eicu.max().item():.4f}, mean={x_recon_eicu.mean().item():.4f}")
                    logger.warning(f"  Decoder output (before skip): min={decoder_output_eicu.min().item():.4f}, max={decoder_output_eicu.max().item():.4f}, mean={decoder_output_eicu.mean().item():.4f}")
                    logger.warning(f"  Skip contribution: min={skip_contribution.min().item():.4f}, max={skip_contribution.max().item():.4f}, mean={skip_contribution.mean().item():.4f}")
                    logger.warning(f"  Input x: min={x_input_eicu.min().item():.4f}, max={x_input_eicu.max().item():.4f}, mean={x_input_eicu.mean().item():.4f}")
                    logger.warning(f"  Latent z: min={z[eicu_mask].min().item():.4f}, max={z[eicu_mask].max().item():.4f}, mean={z[eicu_mask].mean().item():.4f}")
                    logger.warning(f"  Skip params - scale: min={self.decoder_eicu.skip_scale.min().item():.4f}, max={self.decoder_eicu.skip_scale.max().item():.4f}, mean={self.decoder_eicu.skip_scale.mean().item():.4f}")
                    logger.warning(f"  Skip params - bias: min={self.decoder_eicu.skip_bias.min().item():.4f}, max={self.decoder_eicu.skip_bias.max().item():.4f}, mean={self.decoder_eicu.skip_bias.mean().item():.4f}")
                    
                    # Find which features have extreme values
                    extreme_mask = x_recon_eicu.abs() > 20
                    if extreme_mask.any():
                        extreme_indices = torch.where(extreme_mask)
                        logger.warning(f"  Extreme values found at {extreme_mask.sum().item()} locations")
                        # Log first few extreme feature indices
                        unique_features = torch.unique(extreme_indices[1][:10])
                        logger.warning(f"  Feature indices with extreme values (first 10): {unique_features.tolist()}")
                
                x_recon[eicu_mask] = x_recon_eicu.to(x_recon.dtype)
            
            # KEEP: Bypass latent bottleneck for demographics (Age, Gender)
            # Copy Age and Gender directly from input to reconstruction
            # This is applied AFTER skip connection to ensure demographics are always preserved
            if len(self.demographic_indices) > 0:
                for demo_idx in self.demographic_indices:
                    x_recon[:, demo_idx] = x[:, demo_idx]
                logger.debug(f"Applied demographic bypass for indices: {self.demographic_indices}")
            
            return {
                'z': z,
                'mu': mu,
                'logvar': logvar,
                'x_recon': x_recon  # Tensor for standard
            }
    
    def cycle_forward(self, x: torch.Tensor, source_domain: int, target_domain: int) -> Dict:
        """
        IMPROVED: Cycle forward pass with skip connections
        
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
        
        # Decode to target domain with skip connection
        if target_domain == 1:  # MIMIC
            decoder_output = self.decoder_mimic(z)
            target_decoder = self.decoder_mimic
        else:  # eICU
            decoder_output = self.decoder_eicu(z)
            target_decoder = self.decoder_eicu
        
        # Extract mean and apply skip connection for first translation
        if self.use_heteroscedastic:
            x_translated_mu, x_translated_logvar = decoder_output
            # Apply skip connection: output = decoder(z) + (a * input + b)
            x_translated = x_translated_mu + (target_decoder.skip_scale * x + target_decoder.skip_bias)
            # CRITICAL FIX: Tight constraints for NORMALIZED features in cycle
            x_translated = torch.clamp(x_translated, min=-10, max=10)  # NORMALIZED range!
            if torch.isnan(x_translated).any() or torch.isinf(x_translated).any():
                x_translated = torch.zeros_like(x_translated)
        else:
            # Apply skip connection: output = decoder(z) + (a * input + b)
            x_translated = decoder_output + (target_decoder.skip_scale * x + target_decoder.skip_bias)
            # Safety check for non-heteroscedastic too - NORMALIZED features!
            x_translated = torch.clamp(x_translated, min=-10, max=10)  # NORMALIZED range!
            if torch.isnan(x_translated).any() or torch.isinf(x_translated).any():
                x_translated = torch.zeros_like(x_translated)
        
        # Encode translated data  
        mu_cycle, logvar_cycle = self.encoder(x_translated)
        z_cycle = self.encoder.reparameterize(mu_cycle, logvar_cycle)
        
        # Decode back to source domain with skip connection
        if source_domain == 1:  # MIMIC
            cycle_decoder_output = self.decoder_mimic(z_cycle)
            source_decoder = self.decoder_mimic
        else:  # eICU
            cycle_decoder_output = self.decoder_eicu(z_cycle)
            source_decoder = self.decoder_eicu
        
        # Extract cycle reconstruction and apply skip connection
        if self.use_heteroscedastic:
            x_cycle_mu, x_cycle_logvar = cycle_decoder_output
            # Apply skip connection: output = decoder(z) + (a * input + b)
            # Here input is x_translated (what we're encoding for the cycle back)
            x_cycle = x_cycle_mu + (source_decoder.skip_scale * x_translated + source_decoder.skip_bias)
            # CRITICAL FIX: Tight constraints for NORMALIZED features in cycle  
            x_cycle = torch.clamp(x_cycle, min=-10, max=10)  # NORMALIZED range!  
            if torch.isnan(x_cycle).any() or torch.isinf(x_cycle).any():
                x_cycle = torch.zeros_like(x_cycle)
        else:
            # Apply skip connection: output = decoder(z) + (a * input + b)
            x_cycle = cycle_decoder_output + (source_decoder.skip_scale * x_translated + source_decoder.skip_bias)
            # Safety check for non-heteroscedastic too  
            x_cycle = torch.clamp(x_cycle, min=-10, max=10)  # NORMALIZED range!
            if torch.isnan(x_cycle).any() or torch.isinf(x_cycle).any():
                x_cycle = torch.zeros_like(x_cycle)
        
        return {
            'x_translated': x_translated,
            'x_cycle': x_cycle,
            'z': z,
            'z_cycle': z_cycle
        }
    
    def _apply_missing_mask(self, x_target: torch.Tensor, x_missing: torch.Tensor) -> torch.Tensor:
        """
        Override numeric targets using missing flags
        When missing_flag[i] == 1, set the corresponding numeric features (min, max, mean, std) to 0
        
        Args:
            x_target: Target numeric features [batch_size, numeric_dim]
            x_missing: Missing flags [batch_size, missing_dim]
            
        Returns:
            Masked target with 0s where data is missing
        """
        x_target_masked = x_target.clone()
        
        # Assuming each clinical feature has 4 values (min, max, mean, std) and 1 missing flag
        # E.g., for 6 clinical features: indices 0-3 (HR), 4-7 (RR), 8-11 (SpO2), etc.
        n_clinical_features = self.missing_dim
        features_per_clinical = 4  # min, max, mean, std
        
        for i in range(n_clinical_features):
            # Get missing flag for this clinical feature
            missing_flag = x_missing[:, i]  # [batch_size]
            
            # Zero out all 4 statistics when missing
            start_idx = i * features_per_clinical
            end_idx = start_idx + features_per_clinical
            
            # Use broadcasting to set values to 0 where missing_flag == 1
            if end_idx <= x_target.shape[1]:  # Safety check
                x_target_masked[:, start_idx:end_idx] = x_target_masked[:, start_idx:end_idx] * (1 - missing_flag.unsqueeze(1))
        
        return x_target_masked
    
    def compute_reconstruction_loss(self, x_numeric: torch.Tensor, x_missing: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        SIMPLIFIED: Compute reconstruction loss only on clinical features (not demographics or missing flags)
        Apply missing mask to override targets to 0 when features are missing
        
        Args:
            x_numeric: Original numeric features [batch_size, numeric_dim]
            x_missing: Missing flags [batch_size, missing_dim]
            x_recon: Reconstructed features [batch_size, input_dim] (numeric + missing)
            
        Returns:
            Reconstruction loss (MSE on clinical features only)
        """
        # Extract reconstructed numeric features (ignore reconstructed missing flags)
        x_recon_numeric = x_recon[:, :self.numeric_dim]
        
        # Apply missing mask to target
        x_numeric_masked = self._apply_missing_mask(x_numeric, x_missing)
        
        # Compute loss only on clinical features (exclude demographics)
        if len(self.clinical_indices) > 0:
            x_clinical = x_numeric_masked[:, self.clinical_indices]
            x_recon_clinical = x_recon_numeric[:, self.clinical_indices]
            loss = F.mse_loss(x_recon_clinical, x_clinical, reduction='mean')
        else:
            loss = torch.tensor(0.0, device=x_numeric.device)
        
        return loss
    
    def compute_cycle_loss(self, x_numeric: torch.Tensor, x_missing: torch.Tensor, x_cycle: torch.Tensor) -> torch.Tensor:
        """
        SIMPLIFIED: Compute cycle consistency loss only on clinical features
        Apply missing mask to override targets to 0 when features are missing
        
        Args:
            x_numeric: Original numeric features [batch_size, numeric_dim]
            x_missing: Missing flags [batch_size, missing_dim]
            x_cycle: Cycled features [batch_size, input_dim] (numeric + missing)
            
        Returns:
            Cycle consistency loss (MSE on clinical features only)
        """
        # Extract cycled numeric features (ignore cycled missing flags)
        x_cycle_numeric = x_cycle[:, :self.numeric_dim]
        
        # Apply missing mask to target
        x_numeric_masked = self._apply_missing_mask(x_numeric, x_missing)
        
        # Compute loss only on clinical features (exclude demographics)
        if len(self.clinical_indices) > 0:
            x_clinical = x_numeric_masked[:, self.clinical_indices]
            x_cycle_clinical = x_cycle_numeric[:, self.clinical_indices]
            loss = F.mse_loss(x_cycle_clinical, x_clinical, reduction='mean')
        else:
            loss = torch.tensor(0.0, device=x_numeric.device)
        
        return loss
    
    def _get_demographic_groups(self, age: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        """
        Partition samples into coarse demographic groups based on age buckets and gender
        
        Args:
            age: Age values [batch_size]
            gender: Gender values [batch_size] (0 or 1)
            
        Returns:
            Group IDs [batch_size] - unique ID for each demographic group
        """
        # Create age buckets (e.g., 10-year intervals)
        age_buckets = (age / self.wasserstein_age_bucket_years).long()
        
        # Combine age bucket and gender into a single group ID
        # group_id = age_bucket * 2 + gender
        group_ids = age_buckets * 2 + gender.long()
        
        return group_ids
    
    def _update_worst_features(self, x_translated: torch.Tensor, x_target: torch.Tensor):
        """
        Update worst-performing features based on current 1-D Wasserstein distances
        This is called periodically (every N epochs) to adapt to training progress
        
        Args:
            x_translated: Translated features
            x_target: Target features
        """
        # Only compute clinical features (exclude demographics)
        if len(self.clinical_indices) == 0:
            return
        
        x_trans_clinical = x_translated[:, self.clinical_indices]
        x_target_clinical = x_target[:, self.clinical_indices]
        
        # Compute 1-D Wasserstein for each clinical feature
        feature_wasserstein = []
        for i in range(x_trans_clinical.shape[1]):
            x_trans_i = x_trans_clinical[:, i]
            x_target_i = x_target_clinical[:, i]
            
            # Sort both distributions
            x_trans_sorted, _ = torch.sort(x_trans_i)
            x_target_sorted, _ = torch.sort(x_target_i)
            
            # Handle different sample sizes
            if len(x_trans_sorted) != len(x_target_sorted):
                min_len = min(len(x_trans_sorted), len(x_target_sorted))
                if len(x_trans_sorted) > min_len:
                    indices = torch.linspace(0, len(x_trans_sorted) - 1, min_len, dtype=torch.long, device=x_translated.device)
                    x_trans_sorted = x_trans_sorted[indices]
                else:
                    indices = torch.linspace(0, len(x_target_sorted) - 1, min_len, dtype=torch.long, device=x_translated.device)
                    x_target_sorted = x_target_sorted[indices]
            
            # 1-D Wasserstein distance
            wasserstein_i = torch.mean(torch.abs(x_trans_sorted - x_target_sorted))
            feature_wasserstein.append(wasserstein_i.item())
        
        # Get worst-K features
        feature_wasserstein_tensor = torch.tensor(feature_wasserstein, device=x_translated.device)
        worst_k = min(self.wasserstein_worst_k, len(feature_wasserstein))
        worst_indices = torch.topk(feature_wasserstein_tensor, k=worst_k).indices
        
        # Convert to global indices (clinical_indices)
        self.worst_feature_indices = [self.clinical_indices[i] for i in worst_indices.cpu().numpy()]
        
        logger.info(f"Updated worst-{worst_k} features: indices={self.worst_feature_indices}, "
                   f"Wasserstein distances={[feature_wasserstein[i] for i in worst_indices.cpu().numpy()]}")
    
    def compute_conditional_wasserstein_loss(self, x_translated: torch.Tensor, x_target: torch.Tensor,
                                            age_translated: torch.Tensor, gender_translated: torch.Tensor,
                                            age_target: torch.Tensor, gender_target: torch.Tensor,
                                            batch_idx: int) -> torch.Tensor:
        """
        Compute conditional 1-D Wasserstein loss partitioned by demographics (age buckets × gender)
        Only computed every N steps and only on worst-K features
        
        Args:
            x_translated: Translated features [batch_size, numeric_dim]
            x_target: Target features [batch_size, numeric_dim]
            age_translated: Age values for translated samples [batch_size]
            gender_translated: Gender values for translated samples [batch_size]
            age_target: Age values for target samples [batch_size]
            gender_target: Gender values for target samples [batch_size]
            batch_idx: Current batch index (to determine if we should compute this step)
            
        Returns:
            Conditional Wasserstein loss
        """
        # Only compute every N steps
        if batch_idx % self.wasserstein_compute_every_n_steps != 0:
            return torch.tensor(0.0, device=x_translated.device)
        
        # Check if we have worst features identified
        if len(self.worst_feature_indices) == 0:
            # First time - use all clinical features
            self.worst_feature_indices = self.clinical_indices[:self.wasserstein_worst_k]
        
        # Get demographic groups
        groups_translated = self._get_demographic_groups(age_translated, gender_translated)
        groups_target = self._get_demographic_groups(age_target, gender_target)
        
        # Find common groups with sufficient samples
        unique_groups_translated = torch.unique(groups_translated)
        unique_groups_target = torch.unique(groups_target)
        common_groups = set(unique_groups_translated.cpu().numpy()) & set(unique_groups_target.cpu().numpy())
        
        if len(common_groups) == 0:
            return torch.tensor(0.0, device=x_translated.device)
        
        total_loss = 0.0
        n_groups_processed = 0
        
        # For each common group
        for group_id in common_groups:
            # Get samples in this group
            mask_translated = (groups_translated == group_id)
            mask_target = (groups_target == group_id)
            
            n_trans = mask_translated.sum().item()
            n_target = mask_target.sum().item()
            
            # Skip groups with insufficient samples
            if n_trans < self.wasserstein_min_group_size or n_target < self.wasserstein_min_group_size:
                continue
            
            # Extract group samples
            x_trans_group = x_translated[mask_translated]
            x_target_group = x_target[mask_target]
            
            # Compute 1-D Wasserstein for worst-K features
            group_loss = 0.0
            for feat_idx in self.worst_feature_indices:
                x_trans_feat = x_trans_group[:, feat_idx]
                x_target_feat = x_target_group[:, feat_idx]
                
                # Sort both distributions
                x_trans_sorted, _ = torch.sort(x_trans_feat)
                x_target_sorted, _ = torch.sort(x_target_feat)
                
                # Handle different sample sizes
                if len(x_trans_sorted) != len(x_target_sorted):
                    min_len = min(len(x_trans_sorted), len(x_target_sorted))
                    if len(x_trans_sorted) > min_len:
                        indices = torch.linspace(0, len(x_trans_sorted) - 1, min_len, dtype=torch.long, device=x_translated.device)
                        x_trans_sorted = x_trans_sorted[indices]
                    else:
                        indices = torch.linspace(0, len(x_target_sorted) - 1, min_len, dtype=torch.long, device=x_translated.device)
                        x_target_sorted = x_target_sorted[indices]
                
                # 1-D Wasserstein distance
                wasserstein_feat = torch.mean(torch.abs(x_trans_sorted - x_target_sorted))
                group_loss += wasserstein_feat
            
            # Average over features
            total_loss += group_loss / len(self.worst_feature_indices)
            n_groups_processed += 1
        
        # Average over groups
        if n_groups_processed > 0:
            return total_loss / n_groups_processed
        else:
            return torch.tensor(0.0, device=x_translated.device)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """SIMPLIFIED: Training step with only three losses: reconstruction, cycle, and conditional Wasserstein"""
        # Extract data
        x_numeric = batch['numeric']  # [batch_size, numeric_dim]
        x_missing = batch['missing']  # [batch_size, missing_dim]
        domain = batch['domain']  # [batch_size]
        
        # Combine features for model input
        x = torch.cat([x_numeric, x_missing], dim=1)
        
        # Separate by domain
        mimic_mask = (domain == 1)
        eicu_mask = (domain == 0)
        
        # Update worst features dynamically if it's time
        if (self.current_epoch != self.last_worst_update_epoch and 
            self.current_epoch % self.wasserstein_update_worst_every_n_epochs == 0 and
            mimic_mask.any() and eicu_mask.any()):
            with torch.no_grad():
                # Get translations for worst feature analysis
                x_eicu = x[eicu_mask]
                mu_eicu, _ = self.encoder(x_eicu)
                x_eicu_to_mimic = self.decoder_mimic(mu_eicu)
                x_mimic_real = x[mimic_mask]
                
                # Update worst features based on current Wasserstein distances
                self._update_worst_features(
                    x_eicu_to_mimic[:, :self.numeric_dim], 
                    x_mimic_real[:, :self.numeric_dim]
                )
                self.last_worst_update_epoch = self.current_epoch
        
        # Forward pass
        outputs = self.forward(x, domain)
        z, mu, logvar, x_recon = outputs['z'], outputs['mu'], outputs['logvar'], outputs['x_recon']
        
        # === LOSS 1: Reconstruction Loss ===
        # Only on clinical features, with missing mask applied
        rec_loss = self.compute_reconstruction_loss(x_numeric, x_missing, x_recon)
        
        # === LOSS 2: Cycle Consistency Loss ===
        cycle_loss = torch.tensor(0.0, device=self.device)
        if mimic_mask.any() and eicu_mask.any():
            # eICU -> MIMIC -> eICU
            if eicu_mask.any():
                x_eicu = x[eicu_mask]
                x_eicu_numeric = x_numeric[eicu_mask]
                x_eicu_missing = x_missing[eicu_mask]
                cycle_out_eicu = self.cycle_forward(x_eicu, 0, 1)
                cycle_loss += self.compute_cycle_loss(x_eicu_numeric, x_eicu_missing, cycle_out_eicu['x_cycle'])
            
            # MIMIC -> eICU -> MIMIC
            if mimic_mask.any():
                x_mimic = x[mimic_mask]
                x_mimic_numeric = x_numeric[mimic_mask]
                x_mimic_missing = x_missing[mimic_mask]
                cycle_out_mimic = self.cycle_forward(x_mimic, 1, 0)
                cycle_loss += self.compute_cycle_loss(x_mimic_numeric, x_mimic_missing, cycle_out_mimic['x_cycle'])
        
        # === LOSS 3: Conditional Wasserstein Loss ===
        wasserstein_loss = torch.tensor(0.0, device=self.device)
        if mimic_mask.any() and eicu_mask.any() and self.wasserstein_weight > 0:
            # Get demographic feature indices (Age and Gender)
            age_idx = self.demographic_indices[0] if len(self.demographic_indices) > 0 else self.numeric_dim - 2
            gender_idx = self.demographic_indices[1] if len(self.demographic_indices) > 1 else self.numeric_dim - 1
            
            # Translate eICU -> MIMIC
            x_eicu = x[eicu_mask]
            mu_eicu, _ = self.encoder(x_eicu)
            x_eicu_to_mimic = self.decoder_mimic(mu_eicu)
            x_eicu_to_mimic_numeric = x_eicu_to_mimic[:, :self.numeric_dim]
            
            # Translate MIMIC -> eICU
            x_mimic = x[mimic_mask]
            mu_mimic, _ = self.encoder(x_mimic)
            x_mimic_to_eicu = self.decoder_eicu(mu_mimic)
            x_mimic_to_eicu_numeric = x_mimic_to_eicu[:, :self.numeric_dim]
            
            # Real samples
            x_mimic_real_numeric = x_numeric[mimic_mask]
            x_eicu_real_numeric = x_numeric[eicu_mask]
            
            # Compute conditional Wasserstein: eICU->MIMIC vs real MIMIC
            wasserstein_loss += self.compute_conditional_wasserstein_loss(
                x_eicu_to_mimic_numeric, x_mimic_real_numeric,
                x_eicu_to_mimic_numeric[:, age_idx], x_eicu_to_mimic_numeric[:, gender_idx],
                x_mimic_real_numeric[:, age_idx], x_mimic_real_numeric[:, gender_idx],
                batch_idx
            )
            
            # Compute conditional Wasserstein: MIMIC->eICU vs real eICU
            wasserstein_loss += self.compute_conditional_wasserstein_loss(
                x_mimic_to_eicu_numeric, x_eicu_real_numeric,
                x_mimic_to_eicu_numeric[:, age_idx], x_mimic_to_eicu_numeric[:, gender_idx],
                x_eicu_real_numeric[:, age_idx], x_eicu_real_numeric[:, gender_idx],
                batch_idx
            )
        
        # === Total Loss ===
        total_loss = (
            self.rec_weight * rec_loss +
            self.cycle_weight * cycle_loss +
            self.wasserstein_weight * wasserstein_loss
        )
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_rec_loss', rec_loss)
        self.log('train_cycle_loss', cycle_loss)
        self.log('train_wasserstein_loss', wasserstein_loss)
        
        # Detailed logging every N batches
        if batch_idx % 50 == 0:
            logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                       f"total={total_loss.item():.4f}, rec={rec_loss.item():.4f}, "
                       f"cycle={cycle_loss.item():.4f}, wasserstein={wasserstein_loss.item():.4f}")
        
        # Safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"Loss explosion at batch {batch_idx}!")
            logger.error(f"  rec_loss: {rec_loss.item():.6f}, cycle_loss: {cycle_loss.item():.6f}, "
                        f"wasserstein_loss: {wasserstein_loss.item():.6f}")
            # Return a safe fallback
            return F.mse_loss(x_recon[:, :self.numeric_dim], x_numeric, reduction='mean')

        # Capture last batch stats for diagnostics
        try:
            xr = x_recon[:, :self.numeric_dim]
            self._last_batch_stats = {
                'x_min': float(x_numeric.min().item()),
                'x_max': float(x_numeric.max().item()),
                'z_min': float(z.min().item()),
                'z_max': float(z.max().item()),
                'xr_min': float(xr.min().item()),
                'xr_max': float(xr.max().item()),
                'rec_loss': float(rec_loss.item()),
                'cycle_loss': float(cycle_loss.item() if torch.isfinite(cycle_loss) else 0.0),
                'wass_loss': float(wasserstein_loss.item() if torch.isfinite(wasserstein_loss) else 0.0),
            }
        except Exception:
            pass
        
        return total_loss

    def on_train_epoch_end(self):
        """
        Compute per-feature KS and Wasserstein statistics at the end of each training epoch.
        This helps track distribution matching progress during training.
        Also log skip connection parameters to monitor for explosion.
        """
        # Log skip connection parameters to monitor for divergence
        logger.info("=" * 80)
        logger.info(f"EPOCH {self.current_epoch} - Skip Connection Parameters:")
        logger.info(f"  MIMIC Decoder:")
        logger.info(f"    skip_scale: min={self.decoder_mimic.skip_scale.min().item():.4f}, max={self.decoder_mimic.skip_scale.max().item():.4f}, mean={self.decoder_mimic.skip_scale.mean().item():.4f}, std={self.decoder_mimic.skip_scale.std().item():.4f}")
        logger.info(f"    skip_bias:  min={self.decoder_mimic.skip_bias.min().item():.4f}, max={self.decoder_mimic.skip_bias.max().item():.4f}, mean={self.decoder_mimic.skip_bias.mean().item():.4f}, std={self.decoder_mimic.skip_bias.std().item():.4f}")
        logger.info(f"  eICU Decoder:")
        logger.info(f"    skip_scale: min={self.decoder_eicu.skip_scale.min().item():.4f}, max={self.decoder_eicu.skip_scale.max().item():.4f}, mean={self.decoder_eicu.skip_scale.mean().item():.4f}, std={self.decoder_eicu.skip_scale.std().item():.4f}")
        logger.info(f"    skip_bias:  min={self.decoder_eicu.skip_bias.min().item():.4f}, max={self.decoder_eicu.skip_bias.max().item():.4f}, mean={self.decoder_eicu.skip_bias.mean().item():.4f}, std={self.decoder_eicu.skip_bias.std().item():.4f}")
        logger.info("=" * 80)
        
        # Log to tensorboard/wandb if available
        self.log('skip_scale_mimic_mean', self.decoder_mimic.skip_scale.mean(), on_epoch=True, prog_bar=False)
        self.log('skip_scale_mimic_max', self.decoder_mimic.skip_scale.max(), on_epoch=True, prog_bar=False)
        self.log('skip_bias_mimic_max_abs', self.decoder_mimic.skip_bias.abs().max(), on_epoch=True, prog_bar=False)
        self.log('skip_scale_eicu_mean', self.decoder_eicu.skip_scale.mean(), on_epoch=True, prog_bar=False)
        self.log('skip_scale_eicu_max', self.decoder_eicu.skip_scale.max(), on_epoch=True, prog_bar=False)
        self.log('skip_bias_eicu_max_abs', self.decoder_eicu.skip_bias.abs().max(), on_epoch=True, prog_bar=False)
        
        # Skip if not yet initialized or no datamodule
        if not hasattr(self, 'trainer') or self.trainer is None:
            return
        if not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
            return
        
        try:
            # Get validation dataloaders
            val_dataloaders = self.trainer.datamodule.val_dataloader()
            if val_dataloaders is None:
                return
            
            # Handle both single dataloader and list of dataloaders
            if not isinstance(val_dataloaders, list):
                val_dataloaders = [val_dataloaders]
            
            # Collect data from validation set
            x_eicu_list, x_mimic_list = [], []
            x_eicu_to_mimic_list, x_mimic_to_eicu_list = [], []
            
            self.eval()
            with torch.no_grad():
                for val_loader in val_dataloaders:
                    for batch in val_loader:
                        x_numeric = batch['numeric'].to(self.device)
                        x_missing = batch['missing'].to(self.device)
                        domain = batch['domain'].to(self.device)
                        
                        x = torch.cat([x_numeric, x_missing], dim=1)
                        
                        # Separate by domain
                        mimic_mask = (domain == 1)
                        eicu_mask = (domain == 0)
                        
                        if eicu_mask.any():
                            x_eicu = x[eicu_mask]
                            x_eicu_list.append(x_eicu.cpu())
                            # Translate eICU -> MIMIC
                            x_eicu_to_mimic = self.translate_eicu_to_mimic_deterministic(x_eicu)
                            x_eicu_to_mimic_list.append(x_eicu_to_mimic.cpu())
                        
                        if mimic_mask.any():
                            x_mimic = x[mimic_mask]
                            x_mimic_list.append(x_mimic.cpu())
                            # Translate MIMIC -> eICU
                            x_mimic_to_eicu = self.translate_mimic_to_eicu_deterministic(x_mimic)
                            x_mimic_to_eicu_list.append(x_mimic_to_eicu.cpu())
                        
                        # Limit samples to avoid memory issues
                        if len(x_eicu_list) > 50 or len(x_mimic_list) > 50:
                            break
                    if len(x_eicu_list) > 50 or len(x_mimic_list) > 50:
                        break
            
            self.train()
            
            # Check if we have data
            if not x_eicu_list or not x_mimic_list:
                return
            
            # Concatenate data
            x_eicu = torch.cat(x_eicu_list, dim=0).numpy()
            x_mimic = torch.cat(x_mimic_list, dim=0).numpy()
            x_eicu_to_mimic = torch.cat(x_eicu_to_mimic_list, dim=0).numpy()
            x_mimic_to_eicu = torch.cat(x_mimic_to_eicu_list, dim=0).numpy()
            
            # Get clinical feature indices (exclude demographics)
            demographic_features = ['Age', 'Gender']
            all_features = self.feature_spec.get('numeric_features', []) + self.feature_spec.get('missing_features', [])
            clinical_features = [f for f in self.feature_spec.get('numeric_features', []) if f not in demographic_features]
            clinical_indices = [i for i, f in enumerate(all_features) if f in clinical_features]
            
            # Extract clinical features only
            x_eicu_clinical = x_eicu[:, clinical_indices]
            x_mimic_clinical = x_mimic[:, clinical_indices]
            x_eicu_to_mimic_clinical = x_eicu_to_mimic[:, clinical_indices]
            x_mimic_to_eicu_clinical = x_mimic_to_eicu[:, clinical_indices]
            
            # Compute per-feature metrics
            ks_eicu_to_mimic_list = []
            ks_mimic_to_eicu_list = []
            wass_eicu_to_mimic_list = []
            wass_mimic_to_eicu_list = []
            
            for i, feat_name in enumerate(clinical_features):
                # KS statistics
                ks_e2m, _ = stats.ks_2samp(x_eicu_to_mimic_clinical[:, i], x_mimic_clinical[:, i])
                ks_m2e, _ = stats.ks_2samp(x_mimic_to_eicu_clinical[:, i], x_eicu_clinical[:, i])
                ks_eicu_to_mimic_list.append(ks_e2m)
                ks_mimic_to_eicu_list.append(ks_m2e)
                
                # Wasserstein distance
                wass_e2m = stats.wasserstein_distance(x_eicu_to_mimic_clinical[:, i], x_mimic_clinical[:, i])
                wass_m2e = stats.wasserstein_distance(x_mimic_to_eicu_clinical[:, i], x_eicu_clinical[:, i])
                wass_eicu_to_mimic_list.append(wass_e2m)
                wass_mimic_to_eicu_list.append(wass_m2e)
            
            # Log average metrics
            mean_ks_e2m = np.mean(ks_eicu_to_mimic_list)
            mean_ks_m2e = np.mean(ks_mimic_to_eicu_list)
            mean_wass_e2m = np.mean(wass_eicu_to_mimic_list)
            mean_wass_m2e = np.mean(wass_mimic_to_eicu_list)
            
            self.log('val_mean_ks_eicu_to_mimic', mean_ks_e2m, prog_bar=True)
            self.log('val_mean_ks_mimic_to_eicu', mean_ks_m2e)
            self.log('val_mean_wass_eicu_to_mimic', mean_wass_e2m, prog_bar=True)
            self.log('val_mean_wass_mimic_to_eicu', mean_wass_m2e)
            
            # Log detailed info
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {self.current_epoch} - Distribution Matching Metrics:")
            logger.info(f"  Mean KS (eICU→MIMIC): {mean_ks_e2m:.6f}")
            logger.info(f"  Mean KS (MIMIC→eICU): {mean_ks_m2e:.6f}")
            logger.info(f"  Mean Wasserstein (eICU→MIMIC): {mean_wass_e2m:.6f}")
            logger.info(f"  Mean Wasserstein (MIMIC→eICU): {mean_wass_m2e:.6f}")
            
            # Log worst 5 features for each metric
            ks_e2m_worst_idx = np.argsort(ks_eicu_to_mimic_list)[-5:][::-1]
            wass_e2m_worst_idx = np.argsort(wass_eicu_to_mimic_list)[-5:][::-1]
            
            logger.info(f"\n  Worst 5 features by KS (eICU→MIMIC):")
            for idx in ks_e2m_worst_idx:
                logger.info(f"    {clinical_features[idx]:12s}: {ks_eicu_to_mimic_list[idx]:.6f}")
            
            logger.info(f"\n  Worst 5 features by Wasserstein (eICU→MIMIC):")
            for idx in wass_e2m_worst_idx:
                logger.info(f"    {clinical_features[idx]:12s}: {wass_eicu_to_mimic_list[idx]:.6f}")
            logger.info(f"{'='*80}\n")
            
        except Exception as e:
            logger.warning(f"Failed to compute epoch-end metrics: {e}")
            # Don't fail training if metrics computation fails
            pass

    def test_step(self, batch, batch_idx):
        """SIMPLIFIED: Test step with all three losses computed"""
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
        
        # === LOSS 1: Reconstruction Loss ===
        rec_loss = self.compute_reconstruction_loss(x_numeric, x_missing, x_recon)
        
        # === LOSS 2: Cycle Consistency Loss ===
        cycle_loss = torch.tensor(0.0, device=self.device)
        if mimic_mask.any() and eicu_mask.any():
            # eICU -> MIMIC -> eICU
            if eicu_mask.any():
                x_eicu = x[eicu_mask]
                x_eicu_numeric = x_numeric[eicu_mask]
                x_eicu_missing = x_missing[eicu_mask]
                cycle_out_eicu = self.cycle_forward(x_eicu, 0, 1)
                cycle_loss += self.compute_cycle_loss(x_eicu_numeric, x_eicu_missing, cycle_out_eicu['x_cycle'])
            
            # MIMIC -> eICU -> MIMIC
            if mimic_mask.any():
                x_mimic = x[mimic_mask]
                x_mimic_numeric = x_numeric[mimic_mask]
                x_mimic_missing = x_missing[mimic_mask]
                cycle_out_mimic = self.cycle_forward(x_mimic, 1, 0)
                cycle_loss += self.compute_cycle_loss(x_mimic_numeric, x_mimic_missing, cycle_out_mimic['x_cycle'])
        
        # === LOSS 3: Conditional Wasserstein Loss ===
        # Note: For test, we compute it every batch (not every N steps like training)
        wasserstein_loss = torch.tensor(0.0, device=self.device)
        if mimic_mask.any() and eicu_mask.any() and self.wasserstein_weight > 0:
            # Get demographic feature indices
            age_idx = self.demographic_indices[0] if len(self.demographic_indices) > 0 else self.numeric_dim - 2
            gender_idx = self.demographic_indices[1] if len(self.demographic_indices) > 1 else self.numeric_dim - 1
            
            # Translate eICU -> MIMIC
            x_eicu = x[eicu_mask]
            mu_eicu, _ = self.encoder(x_eicu)
            x_eicu_to_mimic = self.decoder_mimic(mu_eicu)
            x_eicu_to_mimic_numeric = x_eicu_to_mimic[:, :self.numeric_dim]
            
            # Translate MIMIC -> eICU
            x_mimic = x[mimic_mask]
            mu_mimic, _ = self.encoder(x_mimic)
            x_mimic_to_eicu = self.decoder_eicu(mu_mimic)
            x_mimic_to_eicu_numeric = x_mimic_to_eicu[:, :self.numeric_dim]
            
            # Real samples
            x_mimic_real_numeric = x_numeric[mimic_mask]
            x_eicu_real_numeric = x_numeric[eicu_mask]
            
            # Compute conditional Wasserstein (force computation by passing batch_idx=0)
            wasserstein_loss += self.compute_conditional_wasserstein_loss(
                x_eicu_to_mimic_numeric, x_mimic_real_numeric,
                x_eicu_to_mimic_numeric[:, age_idx], x_eicu_to_mimic_numeric[:, gender_idx],
                x_mimic_real_numeric[:, age_idx], x_mimic_real_numeric[:, gender_idx],
                0  # Force computation
            )
            
            wasserstein_loss += self.compute_conditional_wasserstein_loss(
                x_mimic_to_eicu_numeric, x_eicu_real_numeric,
                x_mimic_to_eicu_numeric[:, age_idx], x_mimic_to_eicu_numeric[:, gender_idx],
                x_eicu_real_numeric[:, age_idx], x_eicu_real_numeric[:, gender_idx],
                0  # Force computation
            )
        
        # Total loss
        total_loss = (
            self.rec_weight * rec_loss +
            self.cycle_weight * cycle_loss +
            self.wasserstein_weight * wasserstein_loss
        )
        
        # Safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"Invalid total_loss in test_step batch {batch_idx}")
            total_loss = rec_loss if not torch.isnan(rec_loss) else torch.tensor(0.0, device=self.device)
        
        # Logging with on_epoch=True to aggregate properly
        self.log('test_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_rec_loss', rec_loss, on_step=False, on_epoch=True)
        self.log('test_cycle_loss', cycle_loss, on_step=False, on_epoch=True)
        self.log('test_wasserstein_loss', wasserstein_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer with gradient clipping"""
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
        
        # CRITICAL FIX: Add gradient clipping for numerical stability
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            },
            # Enable gradient clipping
            "gradient_clip_val": self.gradient_clip_val
        }
    
    def on_before_optimizer_step(self, optimizer):
        """DIAGNOSTIC: Log gradient norms to detect explosions, especially in skip connection parameters"""
        # Compute total gradient norm
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
                min_grad = min(min_grad, p.grad.data.abs().min().item())
        
        total_norm = total_norm ** 0.5
        
        # Log gradient statistics
        self.log('grad_norm', total_norm, prog_bar=False)
        self.log('grad_max', max_grad, prog_bar=False)
        
        # Log skip connection parameter gradients (always, to track evolution)
        if self.decoder_mimic.skip_scale.grad is not None:
            skip_scale_grad_mimic = self.decoder_mimic.skip_scale.grad.abs().max().item()
            skip_bias_grad_mimic = self.decoder_mimic.skip_bias.grad.abs().max().item()
            self.log('grad_skip_scale_mimic', skip_scale_grad_mimic, prog_bar=False)
            self.log('grad_skip_bias_mimic', skip_bias_grad_mimic, prog_bar=False)
        
        if self.decoder_eicu.skip_scale.grad is not None:
            skip_scale_grad_eicu = self.decoder_eicu.skip_scale.grad.abs().max().item()
            skip_bias_grad_eicu = self.decoder_eicu.skip_bias.grad.abs().max().item()
            self.log('grad_skip_scale_eicu', skip_scale_grad_eicu, prog_bar=False)
            self.log('grad_skip_bias_eicu', skip_bias_grad_eicu, prog_bar=False)
        
        # Warning for concerning gradient norms
        if total_norm > 10.0:
            logger.warning(f"HIGH GRADIENT NORM DETECTED: {total_norm:.4f} (max_grad: {max_grad:.4f})")
            # Log skip connection grads when gradient norm is high
            if self.decoder_mimic.skip_scale.grad is not None:
                logger.warning(f"  MIMIC skip_scale grad: {skip_scale_grad_mimic:.4f}, skip_bias grad: {skip_bias_grad_mimic:.4f}")
            if self.decoder_eicu.skip_scale.grad is not None:
                logger.warning(f"  eICU skip_scale grad: {skip_scale_grad_eicu:.4f}, skip_bias grad: {skip_bias_grad_eicu:.4f}")
        
        if total_norm > 100.0:
            logger.error(f"GRADIENT EXPLOSION! Norm: {total_norm:.4f}, Max: {max_grad:.4f}, Min: {min_grad:.8f}")
            logger.error(f"  Epoch: {self.current_epoch}, Global step: {self.global_step}")
            
            # Log which parameters have the largest gradients
            large_grad_params = []
            for name, p in self.named_parameters():
                if p.grad is not None:
                    param_max = p.grad.data.abs().max().item()
                    if param_max > 10.0:
                        large_grad_params.append((name, param_max))
            
            if large_grad_params:
                logger.error(f"  Parameters with large gradients:")
                for name, grad_val in sorted(large_grad_params, key=lambda x: x[1], reverse=True)[:5]:
                    logger.error(f"    {name}: {grad_val:.4f}")

        # Diagnostics: if non-finite gradients, print offending parameters and LR
        if not torch.isfinite(torch.tensor(total_norm)) or not torch.isfinite(torch.tensor(max_grad)):
            try:
                # Log LR from the first param group
                for i, pg in enumerate(optimizer.param_groups):
                    logger.error(f"  LR (param_group {i}): {pg.get('lr', 'unknown')}")
                    break
            except Exception:
                pass
            # Log non-finite parameter gradients
            for name, p in self.named_parameters():
                if p.grad is not None:
                    g = p.grad.data
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        logger.error(f"  Non-finite grad in: {name} | max={g.abs().max().item() if torch.isfinite(g.abs().max()) else 'inf'} shape={tuple(g.shape)}")
            # Log last batch stats if available
            if hasattr(self, '_last_batch_stats'):
                s = self._last_batch_stats
                logger.error(f"  Last batch stats: x[min,max]=({s.get('x_min'):.4f},{s.get('x_max'):.4f}), z[min,max]=({s.get('z_min'):.4f},{s.get('z_max'):.4f}), x_recon[min,max]=({s.get('xr_min'):.4f},{s.get('xr_max'):.4f}), rec={s.get('rec_loss'):.6f}, cycle={s.get('cycle_loss'):.6f}, wass={s.get('wass_loss'):.6f}")
    
    def translate_eicu_to_mimic(self, x_eicu: torch.Tensor) -> torch.Tensor:
        """IMPROVED: Translate eICU data to MIMIC format with skip connections"""
        self.eval()
        with torch.no_grad():
            # Encode
            mu, logvar = self.encoder(x_eicu)
            # CRITICAL FIX: More intelligent safety checks for encoder outputs in translation
            mu = torch.clamp(mu, min=-20, max=20)  # Wider range for latent space
            logvar = torch.clamp(logvar, min=-5, max=3)  # Keep logvar range tight for stability
            if torch.isnan(mu).any() or torch.isinf(mu).any():
                mu = torch.zeros_like(mu)
            if torch.isnan(logvar).any() or torch.isinf(logvar).any():
                logvar = torch.full_like(logvar, -2.0)
            z = self.encoder.reparameterize(mu, logvar)
            
            # Decode to MIMIC
            decoder_output = self.decoder_mimic(z)
            
            if self.use_heteroscedastic:
                x_mimic_mu, logvar_out = decoder_output
                # Apply skip connection: output = decoder(z) + (a * input + b)
                x_mimic = x_mimic_mu + (self.decoder_mimic.skip_scale * x_eicu + self.decoder_mimic.skip_bias)
                # CRITICAL FIX: More intelligent safety checks - don't destroy Age feature
                # Only clamp extreme outliers, preserve normal data ranges
                x_mimic = torch.clamp(x_mimic, min=-50, max=100)  # Allow Age 0-100, other features -50 to 100
                if torch.isnan(x_mimic).any() or torch.isinf(x_mimic).any():
                    x_mimic = torch.zeros_like(x_mimic)
            else:
                # Apply skip connection: output = decoder(z) + (a * input + b)
                x_mimic = decoder_output + (self.decoder_mimic.skip_scale * x_eicu + self.decoder_mimic.skip_bias)
                # Safety check for non-heteroscedastic too
                x_mimic = torch.clamp(x_mimic, min=-50, max=100)  # Same wider range
                if torch.isnan(x_mimic).any() or torch.isinf(x_mimic).any():
                    x_mimic = torch.zeros_like(x_mimic)
            
        return x_mimic
    
    def translate_mimic_to_eicu(self, x_mimic: torch.Tensor) -> torch.Tensor:
        """IMPROVED: Translate MIMIC data to eICU format with skip connections"""
        self.eval()
        with torch.no_grad():
            # Encode
            mu, logvar = self.encoder(x_mimic)
            # CRITICAL FIX: More intelligent safety checks for encoder outputs in translation
            mu = torch.clamp(mu, min=-20, max=20)  # Wider range for latent space
            logvar = torch.clamp(logvar, min=-5, max=3)  # Keep logvar range tight for stability
            if torch.isnan(mu).any() or torch.isinf(mu).any():
                mu = torch.zeros_like(mu)
            if torch.isnan(logvar).any() or torch.isinf(logvar).any():
                logvar = torch.full_like(logvar, -2.0)
            z = self.encoder.reparameterize(mu, logvar)
            
            # Decode to eICU
            decoder_output = self.decoder_eicu(z)
            
            if self.use_heteroscedastic:
                x_eicu_mu, logvar_out = decoder_output
                # Apply skip connection: output = decoder(z) + (a * input + b)
                x_eicu = x_eicu_mu + (self.decoder_eicu.skip_scale * x_mimic + self.decoder_eicu.skip_bias)
                # CRITICAL FIX: More intelligent safety checks - don't destroy Age feature
                x_eicu = torch.clamp(x_eicu, min=-50, max=100)  # Allow Age 0-100, other features -50 to 100
                if torch.isnan(x_eicu).any() or torch.isinf(x_eicu).any():
                    x_eicu = torch.zeros_like(x_eicu)
            else:
                # Apply skip connection: output = decoder(z) + (a * input + b)
                x_eicu = decoder_output + (self.decoder_eicu.skip_scale * x_mimic + self.decoder_eicu.skip_bias)
                # Safety check for non-heteroscedastic too
                x_eicu = torch.clamp(x_eicu, min=-50, max=100)  # Same wider range
                if torch.isnan(x_eicu).any() or torch.isinf(x_eicu).any():
                    x_eicu = torch.zeros_like(x_eicu)
            
        return x_eicu
    
    def translate_eicu_to_mimic_deterministic(self, x_eicu: torch.Tensor) -> torch.Tensor:
        """
        DETERMINISTIC translation for evaluation (no stochasticity).
        Uses encoder mean (mu) directly without reparameterization sampling.
        This gives reproducible, noise-free translations for evaluation metrics.
        
        Args:
            x_eicu: eICU features [batch_size, input_dim]
            
        Returns:
            x_mimic: Translated MIMIC features [batch_size, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # Encode - get mu only, ignore logvar
            mu, _ = self.encoder(x_eicu)
            
            # Safety checks for encoder output
            mu = torch.clamp(mu, min=-20, max=20)
            if torch.isnan(mu).any() or torch.isinf(mu).any():
                mu = torch.zeros_like(mu)
            
            # Use mu directly (deterministic, no sampling)
            z = mu
            
            # Decode to MIMIC
            decoder_output = self.decoder_mimic(z)
            
            if self.use_heteroscedastic:
                x_mimic_mu, _ = decoder_output  # Ignore logvar output
                # Apply skip connection: output = decoder(z) + (a * input + b)
                x_mimic = x_mimic_mu + (self.decoder_mimic.skip_scale * x_eicu + self.decoder_mimic.skip_bias)
                x_mimic = torch.clamp(x_mimic, min=-50, max=100)
                if torch.isnan(x_mimic).any() or torch.isinf(x_mimic).any():
                    x_mimic = torch.zeros_like(x_mimic)
            else:
                # Apply skip connection: output = decoder(z) + (a * input + b)
                x_mimic = decoder_output + (self.decoder_mimic.skip_scale * x_eicu + self.decoder_mimic.skip_bias)
                x_mimic = torch.clamp(x_mimic, min=-50, max=100)
                if torch.isnan(x_mimic).any() or torch.isinf(x_mimic).any():
                    x_mimic = torch.zeros_like(x_mimic)
            
        return x_mimic
    
    def translate_mimic_to_eicu_deterministic(self, x_mimic: torch.Tensor) -> torch.Tensor:
        """
        DETERMINISTIC translation for evaluation (no stochasticity).
        Uses encoder mean (mu) directly without reparameterization sampling.
        This gives reproducible, noise-free translations for evaluation metrics.
        
        Args:
            x_mimic: MIMIC features [batch_size, input_dim]
            
        Returns:
            x_eicu: Translated eICU features [batch_size, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # Encode - get mu only, ignore logvar
            mu, _ = self.encoder(x_mimic)
            
            # Safety checks for encoder output
            mu = torch.clamp(mu, min=-20, max=20)
            if torch.isnan(mu).any() or torch.isinf(mu).any():
                mu = torch.zeros_like(mu)
            
            # Use mu directly (deterministic, no sampling)
            z = mu
            
            # Decode to eICU
            decoder_output = self.decoder_eicu(z)
            
            if self.use_heteroscedastic:
                x_eicu_mu, _ = decoder_output  # Ignore logvar output
                # Apply skip connection: output = decoder(z) + (a * input + b)
                x_eicu = x_eicu_mu + (self.decoder_eicu.skip_scale * x_mimic + self.decoder_eicu.skip_bias)
                x_eicu = torch.clamp(x_eicu, min=-50, max=100)
                if torch.isnan(x_eicu).any() or torch.isinf(x_eicu).any():
                    x_eicu = torch.zeros_like(x_eicu)
            else:
                # Apply skip connection: output = decoder(z) + (a * input + b)
                x_eicu = decoder_output + (self.decoder_eicu.skip_scale * x_mimic + self.decoder_eicu.skip_bias)
                x_eicu = torch.clamp(x_eicu, min=-50, max=100)
                if torch.isnan(x_eicu).any() or torch.isinf(x_eicu).any():
                    x_eicu = torch.zeros_like(x_eicu)
            
        return x_eicu
    
    def run_comprehensive_evaluation(self, data_module, output_dir: str):
        """
        IMPROVEMENT 3: Comprehensive evaluation integrated into model
        
        Args:
            data_module: Data module with test data
            output_dir: Output directory for reports
        """
        try:
            logger.info("Running integrated comprehensive evaluation...")
            
            # Get test data
            data_module.setup('test')
            test_loader = data_module.test_dataloader()
            
            # Collect test data
            mimic_data = []
            eicu_data = []
            
            for batch in test_loader:
                x_numeric = batch['numeric']
                x_missing = batch['missing']
                domain = batch['domain']
                
                x = torch.cat([x_numeric, x_missing], dim=1)
                
                mimic_mask = (domain == 1)
                eicu_mask = (domain == 0)
                
                if mimic_mask.any():
                    mimic_data.append(x[mimic_mask])
                if eicu_mask.any():
                    eicu_data.append(x[eicu_mask])
            
            if not mimic_data or not eicu_data:
                logger.warning("Insufficient test data for comprehensive evaluation")
                return
            
            # Combine data
            x_mimic_test = torch.cat(mimic_data, dim=0)
            x_eicu_test = torch.cat(eicu_data, dim=0)
            
            # Limit to reasonable size for evaluation
            max_samples = 1000
            if x_mimic_test.size(0) > max_samples:
                x_mimic_test = x_mimic_test[:max_samples]
            if x_eicu_test.size(0) > max_samples:
                x_eicu_test = x_eicu_test[:max_samples]
            
            logger.info(f"Evaluating with {x_mimic_test.size(0)} MIMIC and {x_eicu_test.size(0)} eICU samples")
            
            # Run evaluation
            evaluation_results = self._evaluate_translation_quality(x_mimic_test, x_eicu_test)
            
            # Log results
            self._log_evaluation_results(evaluation_results)
            
            # Save detailed report
            report_path = Path(output_dir) / "comprehensive_evaluation_report.json"
            with open(report_path, 'w') as f:
                # Convert tensors to lists for JSON serialization
                serializable_results = self._convert_tensors_for_json(evaluation_results)
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Comprehensive evaluation saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            logger.warning("Continuing without comprehensive evaluation")
    
    def _evaluate_translation_quality(self, x_mimic: torch.Tensor, x_eicu: torch.Tensor) -> Dict:
        """IMPROVEMENT 3: Comprehensive translation quality evaluation"""
        self.eval()
        with torch.no_grad():
            # Move data to device
            x_mimic = x_mimic.to(self.device)
            x_eicu = x_eicu.to(self.device)
            
            # Translation: MIMIC -> eICU
            x_mimic_to_eicu = self.translate_mimic_to_eicu(x_mimic)
            
            # Translation: eICU -> MIMIC  
            x_eicu_to_mimic = self.translate_eicu_to_mimic(x_eicu)
            
            # Cycle consistency: MIMIC -> eICU -> MIMIC
            x_mimic_cycle = self.translate_eicu_to_mimic(x_mimic_to_eicu)
            
            # Cycle consistency: eICU -> MIMIC -> eICU
            x_eicu_cycle = self.translate_mimic_to_eicu(x_eicu_to_mimic)
            
            # Compute metrics
            evaluation_results = {
                'mimic_to_eicu_translation': {
                    'distributional_metrics': self._compute_distributional_metrics(x_eicu, x_mimic_to_eicu),
                    'ks_statistics': self._compute_ks_statistics(x_eicu, x_mimic_to_eicu)
                },
                'eicu_to_mimic_translation': {
                    'distributional_metrics': self._compute_distributional_metrics(x_mimic, x_eicu_to_mimic),
                    'ks_statistics': self._compute_ks_statistics(x_mimic, x_eicu_to_mimic)
                },
                'cycle_consistency': {
                    'mimic_cycle_rmse': self._compute_per_feature_rmse(x_mimic, x_mimic_cycle),
                    'eicu_cycle_rmse': self._compute_per_feature_rmse(x_eicu, x_eicu_cycle),
                    'mimic_cycle_corr': self._compute_per_feature_correlation(x_mimic, x_mimic_cycle),
                    'eicu_cycle_corr': self._compute_per_feature_correlation(x_eicu, x_eicu_cycle)
                }
            }
            
            # Summary statistics
            mimic_cycle_rmse = evaluation_results['cycle_consistency']['mimic_cycle_rmse']
            eicu_cycle_rmse = evaluation_results['cycle_consistency']['eicu_cycle_rmse']
            
            evaluation_results['summary'] = {
                'mean_cycle_rmse': (torch.mean(mimic_cycle_rmse) + torch.mean(eicu_cycle_rmse)) / 2,
                'worst_cycle_rmse': torch.max(torch.maximum(mimic_cycle_rmse, eicu_cycle_rmse)),
                'best_cycle_rmse': torch.min(torch.minimum(mimic_cycle_rmse, eicu_cycle_rmse)),
                'mimic_translation_ks_mean': evaluation_results['mimic_to_eicu_translation']['ks_statistics']['mean_ks_statistic'],
                'eicu_translation_ks_mean': evaluation_results['eicu_to_mimic_translation']['ks_statistics']['mean_ks_statistic'],
                'significant_mimic_features': evaluation_results['mimic_to_eicu_translation']['ks_statistics']['significant_features'],
                'significant_eicu_features': evaluation_results['eicu_to_mimic_translation']['ks_statistics']['significant_features']
            }
            
            return evaluation_results
    
    def _compute_per_feature_rmse(self, x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """Compute per-feature RMSE"""
        x_true_np = x_true.detach().cpu().numpy()
        x_pred_np = x_pred.detach().cpu().numpy()
        
        n_features = x_true.shape[1]
        rmse_per_feature = np.zeros(n_features)
        
        for i in range(n_features):
            rmse_per_feature[i] = np.sqrt(mean_squared_error(x_true_np[:, i], x_pred_np[:, i]))
        
        return torch.from_numpy(rmse_per_feature)
    
    def _compute_per_feature_correlation(self, x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """Compute per-feature Pearson correlation"""
        x_true_np = x_true.detach().cpu().numpy()
        x_pred_np = x_pred.detach().cpu().numpy()
        
        n_features = x_true.shape[1]
        correlations = np.zeros(n_features)
        
        for i in range(n_features):
            corr, _ = stats.pearsonr(x_true_np[:, i], x_pred_np[:, i])
            correlations[i] = corr if not np.isnan(corr) else 0.0
        
        return torch.from_numpy(correlations)
    
    def _compute_ks_statistics(self, x_source: torch.Tensor, x_translated: torch.Tensor) -> Dict:
        """Compute Kolmogorov-Smirnov statistics per feature"""
        x_source_np = x_source.detach().cpu().numpy()
        x_translated_np = x_translated.detach().cpu().numpy()
        
        n_features = x_source.shape[1]
        ks_stats = np.zeros(n_features)
        p_values = np.zeros(n_features)
        
        for i in range(n_features):
            ks_stat, p_val = stats.ks_2samp(x_source_np[:, i], x_translated_np[:, i])
            ks_stats[i] = ks_stat
            p_values[i] = p_val
        
        return {
            'ks_statistics': torch.from_numpy(ks_stats),
            'p_values': torch.from_numpy(p_values),
            'significant_features': torch.sum(torch.from_numpy(p_values) < 0.05).item(),
            'mean_ks_statistic': np.mean(ks_stats)
        }
    
    def _compute_distributional_metrics(self, x_source: torch.Tensor, x_translated: torch.Tensor) -> Dict:
        """Comprehensive distributional comparison metrics"""
        x_source_np = x_source.detach().cpu().numpy()
        x_translated_np = x_translated.detach().cpu().numpy()
        
        n_features = x_source.shape[1]
        
        # Mean and std differences
        mean_diff = np.abs(np.mean(x_source_np, axis=0) - np.mean(x_translated_np, axis=0))
        std_diff = np.abs(np.std(x_source_np, axis=0) - np.std(x_translated_np, axis=0))
        
        # Quantile differences (25th, 50th, 75th percentiles)
        q25_diff = np.abs(np.percentile(x_source_np, 25, axis=0) - np.percentile(x_translated_np, 25, axis=0))
        q50_diff = np.abs(np.percentile(x_source_np, 50, axis=0) - np.percentile(x_translated_np, 50, axis=0))
        q75_diff = np.abs(np.percentile(x_source_np, 75, axis=0) - np.percentile(x_translated_np, 75, axis=0))
        
        # Wasserstein (Earth Mover's) distance per feature
        wasserstein_distances = np.zeros(n_features)
        for i in range(n_features):
            # Sort and compute 1-Wasserstein distance
            source_sorted = np.sort(x_source_np[:, i])
            translated_sorted = np.sort(x_translated_np[:, i])
            
            # Handle different sample sizes by resampling to common size
            min_len = min(len(source_sorted), len(translated_sorted))
            if len(source_sorted) != len(translated_sorted):
                if len(source_sorted) > min_len:
                    indices = np.linspace(0, len(source_sorted) - 1, min_len, dtype=int)
                    source_sorted = source_sorted[indices]
                else:
                    indices = np.linspace(0, len(translated_sorted) - 1, min_len, dtype=int)
                    translated_sorted = translated_sorted[indices]
            
            wasserstein_distances[i] = np.mean(np.abs(source_sorted - translated_sorted))
        
        return {
            'mean_absolute_difference': torch.from_numpy(mean_diff),
            'std_absolute_difference': torch.from_numpy(std_diff),
            'q25_difference': torch.from_numpy(q25_diff),
            'q50_difference': torch.from_numpy(q50_diff),
            'q75_difference': torch.from_numpy(q75_diff),
            'wasserstein_distances': torch.from_numpy(wasserstein_distances),
            'summary': {
                'mean_mean_diff': np.mean(mean_diff),
                'mean_std_diff': np.mean(std_diff),
                'mean_wasserstein': np.mean(wasserstein_distances)
            }
        }
    
    def _log_evaluation_results(self, evaluation_results: Dict):
        """Log detailed evaluation results"""
        logger.info("=== COMPREHENSIVE TRANSLATION EVALUATION ===")
        
        summary = evaluation_results['summary']
        logger.info(f"Mean cycle consistency RMSE: {summary['mean_cycle_rmse']:.4f}")
        logger.info(f"Worst feature cycle RMSE: {summary['worst_cycle_rmse']:.4f}")
        logger.info(f"Best feature cycle RMSE: {summary['best_cycle_rmse']:.4f}")
        
        logger.info(f"MIMIC->eICU translation KS mean: {summary['mimic_translation_ks_mean']:.4f}")
        logger.info(f"eICU->MIMIC translation KS mean: {summary['eicu_translation_ks_mean']:.4f}")
        logger.info(f"Significant features MIMIC->eICU: {summary['significant_mimic_features']}")
        logger.info(f"Significant features eICU->MIMIC: {summary['significant_eicu_features']}")
        
        # Detailed feature analysis
        mimic_cycle_rmse = evaluation_results['cycle_consistency']['mimic_cycle_rmse']
        eicu_cycle_rmse = evaluation_results['cycle_consistency']['eicu_cycle_rmse']
        
        # Find worst performing features
        worst_feature_indices = torch.topk(mimic_cycle_rmse + eicu_cycle_rmse, k=10).indices
        best_feature_indices = torch.topk(mimic_cycle_rmse + eicu_cycle_rmse, k=10, largest=False).indices
        
        logger.info(f"Top 10 worst features by cycle RMSE:")
        for i, idx in enumerate(worst_feature_indices):
            mimic_rmse = mimic_cycle_rmse[idx].item()
            eicu_rmse = eicu_cycle_rmse[idx].item()
            logger.info(f"  {i+1}. Feature {idx}: MIMIC={mimic_rmse:.4f}, eICU={eicu_rmse:.4f}")
        
        logger.info(f"Top 10 best features by cycle RMSE:")
        for i, idx in enumerate(best_feature_indices):
            mimic_rmse = mimic_cycle_rmse[idx].item()
            eicu_rmse = eicu_cycle_rmse[idx].item()
            logger.info(f"  {i+1}. Feature {idx}: MIMIC={mimic_rmse:.4f}, eICU={eicu_rmse:.4f}")
        
        # Check if known worst features are still problematic
        known_worst_features = [32]  # Feature 32 from the analysis
        logger.info(f"Analysis of known problematic features:")
        for idx in known_worst_features:
            if idx < len(mimic_cycle_rmse):
                mimic_rmse = mimic_cycle_rmse[idx].item()
                eicu_rmse = eicu_cycle_rmse[idx].item()
                logger.info(f"  Feature {idx}: MIMIC={mimic_rmse:.4f}, eICU={eicu_rmse:.4f}")
    
    def _convert_tensors_for_json(self, obj):
        """Convert tensors to lists for JSON serialization"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_tensors_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_for_json(item) for item in obj]
        else:
            return obj

    def compute_feature_iqr(self, x_train: torch.Tensor) -> torch.Tensor:
        """
        Compute IQR for each feature from training data
        Used for IQR-normalized relative error in evaluation
        
        Args:
            x_train: Training data [n_samples, n_features]
            
        Returns:
            IQR per feature [n_features]
        """
        with torch.no_grad():
            q75 = torch.quantile(x_train, 0.75, dim=0)
            q25 = torch.quantile(x_train, 0.25, dim=0)
            iqr = q75 - q25
            # Avoid division by zero: use small epsilon where IQR is tiny
            iqr = torch.where(iqr < 1e-6, torch.ones_like(iqr), iqr)
            return iqr
    
    def compute_per_feature_percentage_errors(self, x_true: torch.Tensor, x_pred: torch.Tensor,
                                             x_missing: torch.Tensor = None,
                                             mode: str = 'reconstruction') -> Dict:
        """
        Compute per-feature percentage errors with hybrid relative error approach
        
        Args:
            x_true: True values [batch_size, numeric_dim]
            x_pred: Predicted values [batch_size, numeric_dim] (from reconstruction or cycle)
            x_missing: Missing flags [batch_size, missing_dim] (optional)
            mode: 'reconstruction' or 'cycle' for logging
            
        Returns:
            Dictionary with percentage error metrics
        """
        with torch.no_grad():
            # Apply missing mask if provided
            if x_missing is not None:
                x_true = self._apply_missing_mask(x_true, x_missing)
            
            # Only compute on clinical features (exclude demographics)
            if len(self.clinical_indices) == 0:
                return {}
            
            x_true_clinical = x_true[:, self.clinical_indices]
            x_pred_clinical = x_pred[:, self.clinical_indices]
            
            # Absolute error
            abs_error = torch.abs(x_pred_clinical - x_true_clinical)
            
            # Method 1: Relative to true value (for non-zero values)
            # Use safe denominator to avoid division by zero
            min_abs_thresh = 0.01 * torch.abs(x_true_clinical).median(dim=0)[0]  # 1% of median
            min_abs_thresh = torch.clamp(min_abs_thresh, min=1e-6)
            
            # Relative error = abs(pred - true) / max(|true|, min_abs_thresh)
            safe_denominator = torch.maximum(torch.abs(x_true_clinical), min_abs_thresh)
            rel_error_to_true = abs_error / safe_denominator
            
            # Method 2: IQR-normalized relative error (for all values)
            if self.feature_iqr is None:
                # Use IQR from current batch as fallback
                iqr = self.compute_feature_iqr(x_true_clinical)
                iqr_clinical = iqr
            else:
                iqr_clinical = self.feature_iqr[self.clinical_indices]

            # Robustness for ceiling-effect features: if >95% mass at boundary, enforce IQR floor
            # Detect boundary-heavy features on current batch
            batch_min = torch.min(x_true_clinical, dim=0)[0]
            batch_max = torch.max(x_true_clinical, dim=0)[0]
            at_min = (x_true_clinical == batch_min.unsqueeze(0)).float().mean(dim=0)
            at_max = (x_true_clinical == batch_max.unsqueeze(0)).float().mean(dim=0)
            boundary_frac = torch.maximum(at_min, at_max)
            boundary_threshold = float(self.config.get('evaluation', {}).get('boundary_mass_threshold', 0.95))
            boundary_heavy_mask = boundary_frac >= boundary_threshold

            # Enforce a minimum IQR epsilon for boundary-heavy features
            iqr_floor = float(self.config.get('evaluation', {}).get('iqr_min_epsilon_ceiling', 0.05))
            effective_iqr = iqr_clinical.clone()
            if boundary_heavy_mask.any():
                floor_tensor = torch.full_like(effective_iqr, iqr_floor)
                effective_iqr[boundary_heavy_mask] = torch.maximum(effective_iqr[boundary_heavy_mask], floor_tensor[boundary_heavy_mask])

            rel_error_iqr = abs_error / effective_iqr.unsqueeze(0)
            
            # Compute percentage within thresholds
            thresholds = [0.05, 0.10, 0.20, 0.30]
            pct_within_thresholds = {}
            
            for thresh in thresholds:
                # Count samples within threshold
                within_thresh = (rel_error_to_true < thresh).float()
                pct_within = within_thresh.mean(dim=0) * 100  # Percentage per feature
                pct_within_thresholds[f'within_{int(thresh*100)}pct'] = pct_within
            
            # IQR-based thresholds (0.1, 0.5, 1.0 IQR)
            iqr_thresholds = [0.1, 0.5, 1.0]
            pct_within_iqr = {}
            
            for thresh in iqr_thresholds:
                within_thresh = (rel_error_iqr < thresh).float()
                pct_within = within_thresh.mean(dim=0) * 100
                pct_within_iqr[f'within_{thresh}_iqr'] = pct_within

            # Absolute tolerance thresholds (normalized units) for robustness on ceiling features
            abs_thresholds = self.config.get('evaluation', {}).get('absolute_tolerance_thresholds', [0.02, 0.05, 0.10])
            pct_within_abs = {}
            for thresh in abs_thresholds:
                thresh_val = float(thresh)
                within_abs = (abs_error < thresh_val).float()
                pct_within_abs[f'within_{thresh_val}_abs'] = within_abs.mean(dim=0) * 100
            
            # Compute statistics
            mae = abs_error.mean(dim=0)  # Per feature
            median_abs_error = abs_error.median(dim=0)[0]
            
            # Percentiles of error
            percentile_75 = torch.quantile(abs_error, 0.75, dim=0)
            percentile_90 = torch.quantile(abs_error, 0.90, dim=0)
            
            results = {
                'mae': mae,  # [n_clinical_features]
                'median_abs_error': median_abs_error,
                'percentile_75_error': percentile_75,
                'percentile_90_error': percentile_90,
                'rel_error_to_true': rel_error_to_true,  # [batch_size, n_clinical_features]
                'rel_error_iqr': rel_error_iqr,
                'pct_within_thresholds': pct_within_thresholds,  # Dict with per-feature percentages
                'pct_within_iqr': pct_within_iqr,
                'pct_within_abs': pct_within_abs,
                'boundary_heavy_mask': boundary_heavy_mask,
                'effective_iqr': effective_iqr,
                'mode': mode
            }
            
            return results
    
    def compute_latent_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> Dict:
        """
        Compute distance metrics between latent representations
        
        Args:
            z1: First set of latent vectors [n_samples1, latent_dim]
            z2: Second set of latent vectors [n_samples2, latent_dim]
            
        Returns:
            Dictionary with distance metrics
        """
        with torch.no_grad():
            # Mean and std of latent representations
            z1_mean = z1.mean(dim=0)
            z2_mean = z2.mean(dim=0)
            z1_std = z1.std(dim=0)
            z2_std = z2.std(dim=0)
            
            # Euclidean distance between means
            mean_distance = torch.norm(z1_mean - z2_mean, p=2).item()
            
            # Cosine similarity between means
            cosine_sim = F.cosine_similarity(z1_mean.unsqueeze(0), z2_mean.unsqueeze(0)).item()
            
            # KL divergence (assuming Gaussian distributions)
            # KL(p||q) = log(σ2/σ1) + (σ1^2 + (μ1-μ2)^2) / (2σ2^2) - 1/2
            kl_div = torch.log(z2_std / (z1_std + 1e-8)) + \
                     (z1_std**2 + (z1_mean - z2_mean)**2) / (2 * z2_std**2 + 1e-8) - 0.5
            kl_div = kl_div.sum().item()
            
            results = {
                'mean_euclidean_distance': mean_distance,
                'cosine_similarity': cosine_sim,
                'kl_divergence': kl_div,
                'z1_mean_norm': torch.norm(z1_mean, p=2).item(),
                'z2_mean_norm': torch.norm(z2_mean, p=2).item()
            }
            
            return results
    
    def compute_per_feature_distribution_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict:
        """
        Compute distribution distance for each feature
        
        Args:
            x1: First set of features [n_samples1, n_features]
            x2: Second set of features [n_samples2, n_features]
            
        Returns:
            Dictionary with per-feature distance metrics
        """
        with torch.no_grad():
            n_features = x1.shape[1]
            
            # Per-feature Wasserstein distance (1-D)
            wasserstein_distances = []
            ks_statistics = []
            mean_differences = []
            std_differences = []
            
            for i in range(n_features):
                # Wasserstein-1 distance
                x1_feat_sorted, _ = torch.sort(x1[:, i])
                x2_feat_sorted, _ = torch.sort(x2[:, i])
                
                # Handle different sample sizes
                if len(x1_feat_sorted) != len(x2_feat_sorted):
                    min_len = min(len(x1_feat_sorted), len(x2_feat_sorted))
                    if len(x1_feat_sorted) > min_len:
                        indices = torch.linspace(0, len(x1_feat_sorted) - 1, min_len, dtype=torch.long, device=x1.device)
                        x1_feat_sorted = x1_feat_sorted[indices]
                    else:
                        indices = torch.linspace(0, len(x2_feat_sorted) - 1, min_len, dtype=torch.long, device=x2.device)
                        x2_feat_sorted = x2_feat_sorted[indices]
                
                wasserstein_dist = torch.mean(torch.abs(x1_feat_sorted - x2_feat_sorted)).item()
                wasserstein_distances.append(wasserstein_dist)
                
                # KS statistic (approximation using sorted samples)
                # True KS would use scipy, but we can approximate
                ks_stat = torch.max(torch.abs(x1_feat_sorted - x2_feat_sorted)).item()
                ks_statistics.append(ks_stat)
                
                # Mean and std differences
                mean_diff = torch.abs(x1[:, i].mean() - x2[:, i].mean()).item()
                std_diff = torch.abs(x1[:, i].std() - x2[:, i].std()).item()
                mean_differences.append(mean_diff)
                std_differences.append(std_diff)
            
            results = {
                'wasserstein_distances': torch.tensor(wasserstein_distances),  # [n_features]
                'ks_statistics': torch.tensor(ks_statistics),
                'mean_differences': torch.tensor(mean_differences),
                'std_differences': torch.tensor(std_differences)
            }
            
            return results

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
