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
    """IMPROVED: Domain-specific decoder with heteroscedastic output"""
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, use_heteroscedastic: bool = True):
        """
        Initialize decoder with optional heteroscedastic outputs
        
        Args:
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
            use_heteroscedastic: If True, predict both mean and log-variance per feature
        """
        super().__init__()
        self.use_heteroscedastic = use_heteroscedastic
        
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
        
        if use_heteroscedastic:
            # Separate heads for mean and log-variance
            self.fc_mu = nn.Linear(prev_dim, output_dim)      # Mean prediction
            self.fc_logvar = nn.Linear(prev_dim, output_dim)  # Log-variance prediction
            logger.info(f"Created heteroscedastic decoder: {latent_dim} -> {hidden_dims} -> {output_dim} (mu + logvar)")
        else:
            # Standard single output
            self.fc_out = nn.Linear(prev_dim, output_dim)
            logger.info(f"Created standard decoder: {latent_dim} -> {hidden_dims} -> {output_dim}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor or tuple:
        """
        Forward pass
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            If heteroscedastic: (mu, logvar) tuple
            If standard: x_recon tensor
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
        self.cov_weight = float(config['training'].get('cov_weight', 0.1))  # New covariance loss weight
        self.per_feature_mmd_weight = float(config['training'].get('per_feature_mmd_weight', 0.05))
        self.wasserstein_weight = float(config['training'].get('wasserstein_weight', 0.0))  # Per-feature Wasserstein weight
        self.kl_warmup_epochs = int(config['training']['kl_warmup_epochs'])
        self.weight_decay = float(config['training']['weight_decay'])
        
        # Track worst-performing features for targeted MMD
        self.worst_features = config['training'].get('worst_features', [
            'Temp_mean', 'Temp_min', 'Temp_max', 'Temp_std',
            'HR_std', 'SpO2_std', 'MAP_std', 'Creat_mean'
        ])
        
        # Feature dimensions (MOVED UP: needed for demographic indices calculation)
        self.numeric_dim = len(feature_spec['numeric_features'])
        self.missing_dim = len(feature_spec['missing_features'])
        self.input_dim = self.numeric_dim + self.missing_dim
        
        # FIXED: Identify demographic features that need special handling (different scales)
        self.demographic_indices = []
        if hasattr(feature_spec, 'numeric_features'):
            for i, feature in enumerate(feature_spec['numeric_features']):
                if 'Age' in feature or 'Gender' in feature:
                    self.demographic_indices.append(i)
        
        # Fallback if feature_spec doesn't have the expected structure
        if len(self.demographic_indices) == 0:
            # Assume last 2 features are Age, Gender based on typical structure
            self.demographic_indices = [self.input_dim - self.missing_dim - 2, self.input_dim - self.missing_dim - 1]
        
        logger.info(f"Demographic feature indices: {self.demographic_indices}")
        
        # Safety flag to disable heteroscedastic for problematic features
        self.use_safe_mode = config['training'].get('use_safe_mode', False)
        
        # Gradient clipping for numerical stability
        self.gradient_clip_val = config['training'].get('gradient_clip_val', 1.0)
        
        # Covariance loss is now numerically stable - no need for disable mechanism
        
        # Architecture parameters
        hidden_dims = [256, 128, 64]  # Can be made configurable
        self.use_heteroscedastic = config['training'].get('use_heteroscedastic', True)
        
        # Initialize networks
        self.encoder = Encoder(self.input_dim, hidden_dims, self.latent_dim)
        self.decoder_mimic = Decoder(self.latent_dim, hidden_dims[::-1], self.input_dim, self.use_heteroscedastic)
        self.decoder_eicu = Decoder(self.latent_dim, hidden_dims[::-1], self.input_dim, self.use_heteroscedastic)
        
        # Prior distribution
        self.prior = Normal(0, 1)
        
        logger.info(f"Initialized Cycle-VAE with input_dim={self.input_dim}, latent_dim={self.latent_dim}")
    
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
                x_recon_mu[mimic_mask] = mu_mimic.to(x_recon_mu.dtype)
                x_recon_logvar[mimic_mask] = logvar_mimic.to(x_recon_logvar.dtype)
            
            if eicu_mask.any():
                mu_eicu, logvar_eicu = self.decoder_eicu(z[eicu_mask])
                x_recon_mu[eicu_mask] = mu_eicu.to(x_recon_mu.dtype) 
                x_recon_logvar[eicu_mask] = logvar_eicu.to(x_recon_logvar.dtype)
            
            return {
                'z': z,
                'mu': mu,
                'logvar': logvar,
                'x_recon': (x_recon_mu, x_recon_logvar)  # Tuple for heteroscedastic
            }
        else:
            # Standard reconstruction
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
                'x_recon': x_recon  # Tensor for standard
            }
    
    def cycle_forward(self, x: torch.Tensor, source_domain: int, target_domain: int) -> Dict:
        """
        IMPROVED: Cycle forward pass supporting heteroscedastic outputs
        
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
            decoder_output = self.decoder_mimic(z)
        else:  # eICU
            decoder_output = self.decoder_eicu(z)
        
        # Extract mean for next encoding step
        if self.use_heteroscedastic:
            x_translated_mu, x_translated_logvar = decoder_output
            x_translated = x_translated_mu  # Use mean for next step
        else:
            x_translated = decoder_output
        
        # Encode translated data  
        mu_cycle, logvar_cycle = self.encoder(x_translated)
        z_cycle = self.encoder.reparameterize(mu_cycle, logvar_cycle)
        
        # Decode back to source domain
        if source_domain == 1:  # MIMIC
            cycle_decoder_output = self.decoder_mimic(z_cycle)
        else:  # eICU
            cycle_decoder_output = self.decoder_eicu(z_cycle)
        
        # Extract cycle reconstruction
        if self.use_heteroscedastic:
            x_cycle_mu, x_cycle_logvar = cycle_decoder_output
            x_cycle = x_cycle_mu  # Use mean for cycle consistency
        else:
            x_cycle = cycle_decoder_output
        
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
    
    def compute_heteroscedastic_nll(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Numerically stable heteroscedastic negative log-likelihood loss
        
        Args:
            x: True features [batch_size, n_features]
            mu: Predicted mean [batch_size, n_features]  
            logvar: Predicted log-variance [batch_size, n_features]
            
        Returns:
            NLL loss
        """
        # CRITICAL FIX: Tighter clamping for numerical stability
        logvar_clamped = torch.clamp(logvar, min=-5, max=3)  # Much safer range: exp(-(-5)) = exp(5) ≈ 148
        
        # Check for any invalid values
        if torch.isnan(logvar_clamped).any() or torch.isinf(logvar_clamped).any():
            logger.warning("NaN or Inf detected in logvar - using fallback MSE")
            return F.mse_loss(mu, x, reduction='mean')
        
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            logger.warning("NaN or Inf detected in mu - using fallback MSE")
            return F.mse_loss(torch.zeros_like(x), x, reduction='mean')
        
        # Compute NLL with numerical stability
        inv_var = torch.exp(-logvar_clamped)  # Max exp(5) ≈ 148, manageable
        mse_term = (x - mu) ** 2 * inv_var
        
        # Check for explosion in MSE term
        if torch.isnan(mse_term).any() or torch.isinf(mse_term).any():
            logger.warning("MSE term explosion - using fallback MSE")
            return F.mse_loss(mu, x, reduction='mean')
        
        nll = 0.5 * (logvar_clamped + mse_term)
        
        # Stronger regularization to prevent variance collapse/explosion
        logvar_penalty = 0.1 * torch.mean(logvar_clamped ** 2)
        
        total_nll = torch.mean(nll) + logvar_penalty
        
        # Final safety check
        if torch.isnan(total_nll) or torch.isinf(total_nll):
            logger.error("Total NLL explosion detected! Using MSE fallback")
            return F.mse_loss(mu, x, reduction='mean')
        
        return total_nll
    
    def compute_reconstruction_loss(self, x: torch.Tensor, decoder_output) -> torch.Tensor:
        """FIXED: Safe reconstruction loss with demographic feature handling"""
        if self.use_heteroscedastic and not self.use_safe_mode:
            # decoder_output is (mu, logvar) tuple
            mu, logvar = decoder_output
            return self.compute_safe_heteroscedastic_nll(x, mu, logvar)
        else:
            # decoder_output is x_recon tensor  
            x_recon = decoder_output
            if not hasattr(self, 'feature_std') or self.feature_std is None:
                # Initialize per-feature standard deviations (computed once from training data)
                with torch.no_grad():
                    self.feature_std = torch.std(x, dim=0) + 1e-8  # Add epsilon for numerical stability
                    logger.info(f"Initialized feature std: min={self.feature_std.min():.6f}, max={self.feature_std.max():.6f}")
            
            # Per-feature standardized MSE
            normalized_diff = (x - x_recon) / self.feature_std
            per_feature_mse = torch.mean(normalized_diff**2, dim=0)  # [n_features]
            
            # OPTIMIZED: Log worst performing features less frequently for speed
            if self.training and torch.rand(1).item() < 0.005:  # 0.5% of batches (half as often)
                worst_features = torch.topk(per_feature_mse, k=min(5, len(per_feature_mse))).indices
                logger.info(f"Worst reconstruction features: {worst_features.tolist()}, MSE: {per_feature_mse[worst_features].tolist()}")
            
            return per_feature_mse.sum()  # Sum across features
    
    def compute_safe_heteroscedastic_nll(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """FIXED: Safe heteroscedastic NLL with demographic feature protection"""
        # Handle demographic features with standard MSE (they have very different scales)
        if len(self.demographic_indices) > 0:
            clinical_indices = [i for i in range(x.shape[1]) if i not in self.demographic_indices]
            
            total_loss = 0.0
            
            # Clinical features: use heteroscedastic NLL
            if len(clinical_indices) > 0:
                clinical_indices = torch.tensor(clinical_indices, device=x.device)
                x_clinical = x[:, clinical_indices]
                mu_clinical = mu[:, clinical_indices]
                logvar_clinical = logvar[:, clinical_indices]
                
                clinical_loss = self.compute_heteroscedastic_nll(x_clinical, mu_clinical, logvar_clinical)
                total_loss += clinical_loss * (len(clinical_indices) / x.shape[1])  # Weight by proportion
            
            # Demographic features: use standard MSE (safer)
            demographic_indices = torch.tensor(self.demographic_indices, device=x.device)
            x_demo = x[:, demographic_indices]
            mu_demo = mu[:, demographic_indices]
            
            demo_loss = F.mse_loss(mu_demo, x_demo, reduction='mean')
            total_loss += demo_loss * (len(self.demographic_indices) / x.shape[1])  # Weight by proportion
            
            return total_loss
        else:
            # No demographic features identified, use full heteroscedastic
            return self.compute_heteroscedastic_nll(x, mu, logvar)
    
    def compute_cycle_loss(self, x: torch.Tensor, x_cycle: torch.Tensor) -> torch.Tensor:
        """IMPROVED: Per-feature standardized cycle consistency loss (works with heteroscedastic)"""
        # Note: x_cycle is already the mean (mu) from cycle_forward when using heteroscedastic
        if not self.use_heteroscedastic:
            # Standard case - use feature standardization
            if not hasattr(self, 'feature_std') or self.feature_std is None:
                # Initialize if not already done
                with torch.no_grad():
                    self.feature_std = torch.std(x, dim=0) + 1e-8
            
            # Per-feature standardized cycle loss
            normalized_diff = (x - x_cycle) / self.feature_std
            per_feature_cycle_mse = torch.mean(normalized_diff**2, dim=0)
            
            return per_feature_cycle_mse.sum()
        else:
            # Heteroscedastic case - use MSE directly since we're comparing against original data
            return F.mse_loss(x_cycle, x, reduction='mean') * x.shape[1]  # Scale by n_features for consistency
    
    def compute_covariance_loss(self, x_translated: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """FIXED: Numerically stable covariance matching loss"""
        def compute_stable_cov_matrix(x):
            # x: (N, F) - center the features
            if x.shape[0] < 2:
                return torch.eye(x.shape[1], device=x.device, dtype=x.dtype) * 1e-6
            
            x_centered = x - x.mean(dim=0, keepdim=True)
            
            # Add numerical stability
            if torch.isnan(x_centered).any() or torch.isinf(x_centered).any():
                return torch.eye(x.shape[1], device=x.device, dtype=x.dtype) * 1e-6
            
            # Use more stable covariance computation
            n_samples = x.shape[0]
            cov = (x_centered.t() @ x_centered) / max(n_samples - 1, 1)
            
            # Add regularization to prevent singular matrices
            cov = cov + torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * 1e-6
            
            # Clamp to prevent extreme values
            cov = torch.clamp(cov, min=-1e6, max=1e6)
            
            return cov
        
        if x_translated.shape[0] < 2 or x_target.shape[0] < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Ensure tensors are finite
        if torch.isnan(x_translated).any() or torch.isinf(x_translated).any():
            return torch.tensor(0.0, device=self.device)
        if torch.isnan(x_target).any() or torch.isinf(x_target).any():
            return torch.tensor(0.0, device=self.device)
        
        cov_translated = compute_stable_cov_matrix(x_translated)
        cov_target = compute_stable_cov_matrix(x_target)
        
        # Frobenius norm of covariance difference with clipping
        cov_diff = cov_translated - cov_target
        
        if torch.isnan(cov_diff).any() or torch.isinf(cov_diff).any():
            return torch.tensor(0.0, device=self.device)
        
        cov_loss = torch.norm(cov_diff, p='fro')
        
        # Final safety clamp
        cov_loss = torch.clamp(cov_loss, min=0, max=1e3)
        
        if torch.isnan(cov_loss) or torch.isinf(cov_loss):
            return torch.tensor(0.0, device=self.device)
        
        return cov_loss
    
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
    
    def compute_wasserstein_loss(self, x_translated: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Compute per-feature 1-Wasserstein distance (Earth Mover's Distance) - stronger than MMD"""
        if x_translated.shape[0] == 0 or x_target.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_wasserstein = 0.0
        n_features = x_translated.shape[1]
        
        for i in range(n_features):
            # Extract individual feature
            x_trans_i = x_translated[:, i]
            x_target_i = x_target[:, i]
            
            # Sort both distributions - Wasserstein distance is the L1 distance between CDFs
            x_trans_sorted, _ = torch.sort(x_trans_i)
            x_target_sorted, _ = torch.sort(x_target_i)
            
            # Handle different sample sizes by interpolation/resampling
            if len(x_trans_sorted) != len(x_target_sorted):
                min_len = min(len(x_trans_sorted), len(x_target_sorted))
                if len(x_trans_sorted) > min_len:
                    indices = torch.linspace(0, len(x_trans_sorted) - 1, min_len, dtype=torch.long, device=self.device)
                    x_trans_sorted = x_trans_sorted[indices]
                else:
                    indices = torch.linspace(0, len(x_target_sorted) - 1, min_len, dtype=torch.long, device=self.device)
                    x_target_sorted = x_target_sorted[indices]
            
            # 1-Wasserstein distance is L1 norm of sorted samples
            wasserstein_i = torch.mean(torch.abs(x_trans_sorted - x_target_sorted))
            total_wasserstein += wasserstein_i
        
        return total_wasserstein / n_features  # Average across features
    
    def compute_per_feature_mmd(self, x_mimic: torch.Tensor, x_eicu: torch.Tensor, 
                               feature_indices: list = None) -> torch.Tensor:
        """
        Compute MMD loss for specific features to target problematic ones
        
        Args:
            x_mimic: MIMIC data [batch_size, n_features]
            x_eicu: eICU data [batch_size, n_features] 
            feature_indices: List of feature indices to compute MMD for
            
        Returns:
            Summed MMD loss for specified features
        """
        if x_mimic.size(0) == 0 or x_eicu.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        
        if feature_indices is None:
            # Default to all features
            feature_indices = list(range(x_mimic.shape[1]))
        
        # Limit to available features
        max_features = min(x_mimic.shape[1], x_eicu.shape[1])
        feature_indices = [idx for idx in feature_indices if idx < max_features]
        
        if len(feature_indices) == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_mmd = 0.0
        sigma = 1.0  # Fixed sigma for per-feature MMD
        
        for feat_idx in feature_indices:
            # Extract single feature [batch_size, 1]
            mimic_feat = x_mimic[:, feat_idx:feat_idx+1]
            eicu_feat = x_eicu[:, feat_idx:feat_idx+1]
            
            # Compute single-feature MMD
            feat_mmd = self._rbf_mmd_single_feature(mimic_feat, eicu_feat, sigma)
            total_mmd += feat_mmd
            
        return total_mmd / len(feature_indices)  # Average over features
    
    def _rbf_mmd_single_feature(self, x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Compute RBF MMD for a single feature
        
        Args:
            x, y: Single feature tensors [batch_size, 1]
            sigma: RBF kernel bandwidth
            
        Returns:
            MMD value for the feature
        """
        # Sample if too large (memory optimization)
        max_samples = 200
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
        
        # Compute pairwise distances 
        gamma = 1.0 / (2 * sigma**2)
        
        xx_dist = torch.cdist(x_sample, x_sample, p=2)  
        yy_dist = torch.cdist(y_sample, y_sample, p=2)
        xy_dist = torch.cdist(x_sample, y_sample, p=2)
        
        # RBF kernels
        K_xx = torch.exp(-gamma * xx_dist**2)
        K_yy = torch.exp(-gamma * yy_dist**2)
        K_xy = torch.exp(-gamma * xy_dist**2)
        
        # MMD computation
        m, n = K_xx.size(0), K_yy.size(0)
        if m > 1:
            K_xx_diag_removed = K_xx - torch.diag(torch.diag(K_xx))
            xx_term = K_xx_diag_removed.sum() / (m * (m - 1))
        else:
            xx_term = K_xx.mean()
            
        if n > 1:
            K_yy_diag_removed = K_yy - torch.diag(torch.diag(K_yy))
            yy_term = K_yy_diag_removed.sum() / (n * (n - 1))
        else:
            yy_term = K_yy.mean()
        
        xy_term = K_xy.mean()
        
        mmd = xx_term + yy_term - 2 * xy_term
        return torch.clamp(mmd, min=0, max=5.0)  # Prevent explosion
    
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
        
        # IMPROVED: Covariance loss (now numerically stable)
        cov_loss = torch.tensor(0.0, device=self.device)
        if mimic_mask.any() and eicu_mask.any():
            # Translate eICU to MIMIC style and compare covariance with real MIMIC
            x_eicu = x[eicu_mask]
            z_eicu_only = z[eicu_mask]
            decoder_output_mimic = self.decoder_mimic(z_eicu_only)
            x_mimic_real = x[mimic_mask]
            
            # Extract mean if using heteroscedastic
            if self.use_heteroscedastic:
                x_eicu_to_mimic, _ = decoder_output_mimic  # Use mean, ignore logvar
            else:
                x_eicu_to_mimic = decoder_output_mimic
            
            cov_loss += self.compute_covariance_loss(x_eicu_to_mimic, x_mimic_real)
            
            # Translate MIMIC to eICU style and compare covariance with real eICU
            x_mimic = x[mimic_mask]
            z_mimic_only = z[mimic_mask]
            decoder_output_eicu = self.decoder_eicu(z_mimic_only)
            
            # Extract mean if using heteroscedastic  
            if self.use_heteroscedastic:
                x_mimic_to_eicu, _ = decoder_output_eicu  # Use mean, ignore logvar
            else:
                x_mimic_to_eicu = decoder_output_eicu
            
            cov_loss += self.compute_covariance_loss(x_mimic_to_eicu, x_eicu)
            
            # Covariance loss is now numerically stable, but keep one safety check
            if torch.isnan(cov_loss) or torch.isinf(cov_loss):
                logger.warning(f"Covariance instability detected at batch {batch_idx} - using zero")
                cov_loss = torch.tensor(0.0, device=self.device)
            
            # IMPROVED: Per-feature MMD loss for worst-performing features
            per_feature_mmd_loss = torch.tensor(0.0, device=self.device)
            if self.per_feature_mmd_weight > 0:
                # Target specific problematic feature indices (Temperature, HR_std, etc.)
                worst_feature_indices = list(range(min(20, x.shape[1])))  # First 20 features as proxy
                
                # Compare translated eICU→MIMIC with real MIMIC
                per_feature_mmd_loss += self.compute_per_feature_mmd(
                    x_eicu_to_mimic, x_mimic_real, worst_feature_indices)
                
                # Compare translated MIMIC→eICU with real eICU 
                per_feature_mmd_loss += self.compute_per_feature_mmd(
                    x_mimic_to_eicu, x_eicu, worst_feature_indices)
        else:
            per_feature_mmd_loss = torch.tensor(0.0, device=self.device)
        
        # NEW: Wasserstein loss for stronger distributional alignment
        wasserstein_loss = torch.tensor(0.0, device=self.device)
        if mimic_mask.any() and eicu_mask.any() and self.wasserstein_weight > 0:
            # Only compute if we have translations (reuse from above)
            if hasattr(self, '_cached_translations'):
                x_eicu_to_mimic, x_mimic_to_eicu, x_mimic_real, x_eicu_real = self._cached_translations
                wasserstein_loss += self.compute_wasserstein_loss(x_eicu_to_mimic, x_mimic_real)
                wasserstein_loss += self.compute_wasserstein_loss(x_mimic_to_eicu, x_eicu_real)
            else:
                # Recompute translations for Wasserstein (fallback)
                x_eicu = x[eicu_mask]
                x_mimic = x[mimic_mask]
                z_eicu_only = z[eicu_mask]
                z_mimic_only = z[mimic_mask]
                
                decoder_output_mimic = self.decoder_mimic(z_eicu_only)
                x_eicu_to_mimic = decoder_output_mimic[0] if self.use_heteroscedastic else decoder_output_mimic
                
                decoder_output_eicu = self.decoder_eicu(z_mimic_only)
                x_mimic_to_eicu = decoder_output_eicu[0] if self.use_heteroscedastic else decoder_output_eicu
                
                wasserstein_loss += self.compute_wasserstein_loss(x_eicu_to_mimic, x_mimic)
                wasserstein_loss += self.compute_wasserstein_loss(x_mimic_to_eicu, x_eicu)
        
        # Total loss with all components INCLUDING WASSERSTEIN
        kl_weight = self.get_kl_weight()
        total_loss = (
            self.rec_weight * rec_loss +
            kl_weight * kl_loss +
            self.cycle_weight * cycle_loss +
            self.mmd_weight * mmd_loss +
            self.cov_weight * cov_loss +
            self.per_feature_mmd_weight * per_feature_mmd_loss +
            self.wasserstein_weight * wasserstein_loss
        )
        
        # Comprehensive logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_rec_loss', rec_loss)
        self.log('train_kl_loss', kl_loss)
        self.log('train_cycle_loss', cycle_loss)
        self.log('train_mmd_loss', mmd_loss)
        self.log('train_cov_loss', cov_loss)
        self.log('train_per_feature_mmd_loss', per_feature_mmd_loss)
        self.log('train_wasserstein_loss', wasserstein_loss)
        self.log('kl_weight', kl_weight)
        
        # OPTIMIZED: Less frequent detailed monitoring for faster training (every 100 batches)
        if batch_idx % 100 == 0 and mimic_mask.any() and eicu_mask.any():
            self._log_detailed_feature_metrics(x, outputs, domain, mimic_mask, eicu_mask)
        
        # CRITICAL: Final safety check to prevent training crashes
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"Training explosion detected at batch {batch_idx}! Loss components:")
            logger.error(f"  rec_loss: {rec_loss.item():.6f}")
            logger.error(f"  kl_loss: {kl_loss.item():.6f}")
            logger.error(f"  cycle_loss: {cycle_loss.item():.6f}")
            logger.error(f"  mmd_loss: {mmd_loss.item():.6f}")
            logger.error(f"  cov_loss: {cov_loss.item():.6f}")
            logger.error(f"  per_feature_mmd_loss: {per_feature_mmd_loss.item():.6f}")
            
            # Return a safe fallback loss to prevent training crash
            fallback_loss = F.mse_loss(outputs['x_recon'], x, reduction='mean') if not self.use_heteroscedastic else F.mse_loss(outputs['x_recon'][0], x, reduction='mean')
            logger.error(f"Using fallback MSE loss: {fallback_loss.item():.6f}")
            return fallback_loss
        
        return total_loss
    
    def _log_detailed_feature_metrics(self, x: torch.Tensor, outputs: Dict, domain: torch.Tensor,
                                    mimic_mask: torch.Tensor, eicu_mask: torch.Tensor):
        """
        IMPROVED: Log detailed per-feature metrics for monitoring
        """
        with torch.no_grad():
            if self.use_heteroscedastic:
                x_recon_mu, x_recon_logvar = outputs['x_recon']
                x_recon = x_recon_mu  # Use mean for metrics
                
                # Log uncertainty statistics
                mean_uncertainty = torch.mean(torch.exp(0.5 * x_recon_logvar))
                self.log('mean_predicted_uncertainty', mean_uncertainty, prog_bar=False)
            else:
                x_recon = outputs['x_recon']
            
            # Per-feature reconstruction error
            per_feature_mse = torch.mean((x - x_recon) ** 2, dim=0)  # [n_features]
            
            # Log statistics of per-feature performance
            self.log('worst_feature_mse', torch.max(per_feature_mse), prog_bar=False)
            self.log('best_feature_mse', torch.min(per_feature_mse), prog_bar=False) 
            self.log('median_feature_mse', torch.median(per_feature_mse), prog_bar=False)
            self.log('feature_mse_std', torch.std(per_feature_mse), prog_bar=False)
            
            # Domain-specific metrics
            if mimic_mask.any():
                mimic_mse = torch.mean((x[mimic_mask] - x_recon[mimic_mask]) ** 2)
                self.log('mimic_domain_mse', mimic_mse, prog_bar=False)
            
            if eicu_mask.any():
                eicu_mse = torch.mean((x[eicu_mask] - x_recon[eicu_mask]) ** 2)
                self.log('eicu_domain_mse', eicu_mse, prog_bar=False)
            
            # Log top 5 worst and best performing feature indices (for debugging)
            if per_feature_mse.numel() >= 5:
                worst_indices = torch.topk(per_feature_mse, k=5).indices.cpu().numpy()
                best_indices = torch.topk(per_feature_mse, k=5, largest=False).indices.cpu().numpy()
                
                # OPTIMIZED: Log even less frequently to speed up training
                if torch.rand(1).item() < 0.05:  # 5% chance (half as often)
                    logger.info(f"Worst features (indices): {worst_indices.tolist()}, MSE: {per_feature_mse[worst_indices].cpu().numpy()}")
                    logger.info(f"Best features (indices): {best_indices.tolist()}, MSE: {per_feature_mse[best_indices].cpu().numpy()}")
    
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
        """IMPROVED: Translate eICU data to MIMIC format (supports heteroscedastic)"""
        self.eval()
        with torch.no_grad():
            # Encode
            mu, logvar = self.encoder(x_eicu)
            z = self.encoder.reparameterize(mu, logvar)
            
            # Decode to MIMIC
            decoder_output = self.decoder_mimic(z)
            
            if self.use_heteroscedastic:
                x_mimic, _ = decoder_output  # Return mean, ignore variance
            else:
                x_mimic = decoder_output
            
        return x_mimic
    
    def translate_mimic_to_eicu(self, x_mimic: torch.Tensor) -> torch.Tensor:
        """IMPROVED: Translate MIMIC data to eICU format (supports heteroscedastic)"""
        self.eval()
        with torch.no_grad():
            # Encode
            mu, logvar = self.encoder(x_mimic)
            z = self.encoder.reparameterize(mu, logvar)
            
            # Decode to eICU
            decoder_output = self.decoder_eicu(z)
            
            if self.use_heteroscedastic:
                x_eicu, _ = decoder_output  # Return mean, ignore variance
            else:
                x_eicu = decoder_output
            
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
