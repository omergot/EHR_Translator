#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Cycle-VAE
Implements patient-level and feature-level evaluation metrics to assess translation quality.
"""

# CRITICAL FIX: Force single-threaded BLAS to avoid non-deterministic correlation computations
# Multi-threaded BLAS causes race conditions in np.corrcoef, leading to different results on each call
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for Cycle-VAE translation quality.
    
    Evaluates:
    1. Per-feature correlation and R² metrics
    2. KS p-values and significance testing
    3. Missingness-aware stratified evaluation
    4. Demographic group evaluation
    5. Visual distribution comparisons
    6. Paired scatter plots
    7. Per-feature summary statistics
    8. Example patient row comparisons
    """
    
    def __init__(self, model, feature_spec: Dict, output_dir: str):
        """
        Initialize comprehensive evaluator.
        
        Args:
            model: Trained Cycle-VAE model
            feature_spec: Feature specification dictionary
            output_dir: Output directory for results
        """
        self.model = model
        self.feature_spec = feature_spec
        self.output_dir = Path(output_dir)
        self.eval_dir = self.output_dir / "comprehensive_evaluation"
        self.eval_dir.mkdir(exist_ok=True)
        
        # Feature names for better interpretability
        self.numeric_features = feature_spec.get('numeric_features', [])
        self.missing_features = feature_spec.get('missing_features', [])
        self.all_features = self.numeric_features + self.missing_features
        
        # UPDATED: Identify clinical-only features (exclude demographics and missing flags)
        self.demographic_features = feature_spec.get('demographic_features', ['Age', 'Gender'])
        self.clinical_only_features = [
            f for f in self.numeric_features 
            if f not in self.demographic_features
        ]
        
        # Get indices for filtering
        self.clinical_indices = [i for i, f in enumerate(self.all_features) 
                                if f in self.clinical_only_features]
        self.demographic_indices = [i for i, f in enumerate(self.all_features) 
                                   if f in self.demographic_features]
        self.missing_indices = [i for i, f in enumerate(self.all_features) 
                               if f in self.missing_features]
        
        # Create subdirectories
        (self.eval_dir / "plots").mkdir(exist_ok=True)
        (self.eval_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"Comprehensive evaluator initialized. Output directory: {self.eval_dir}")
        logger.info(f"Clinical features: {len(self.clinical_only_features)}, Demographics: {len(self.demographic_features)}, Missing flags: {len(self.missing_features)}")
    
    def evaluate_translation_quality(self, x_eicu: torch.Tensor, x_mimic: torch.Tensor) -> Dict:
        """
        UPDATED: Run comprehensive evaluation with simplified model metrics.
        
        Args:
            x_eicu: eICU test data [n_samples, n_features] (numeric + missing)
            x_mimic: MIMIC test data [n_samples, n_features] (numeric + missing)
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Starting comprehensive translation quality evaluation (SIMPLIFIED MODEL)...")
        logger.info(f"Input data shapes: eICU {x_eicu.shape}, MIMIC {x_mimic.shape}")
        
        # Move data to model device
        device = next(self.model.parameters()).device
        logger.info(f"Moving data to device: {device}")
        x_eicu = x_eicu.to(device)
        x_mimic = x_mimic.to(device)
        
        # Split numeric and missing flags
        numeric_dim = self.model.numeric_dim
        x_eicu_numeric = x_eicu[:, :numeric_dim]
        x_eicu_missing = x_eicu[:, numeric_dim:]
        x_mimic_numeric = x_mimic[:, :numeric_dim]
        x_mimic_missing = x_mimic[:, numeric_dim:]
        
        # CRITICAL: Ensure model is in eval mode (disables dropout/batchnorm)
        self.model.eval()
        logger.info("Model set to eval mode (dropout/batchnorm disabled)")
        
        # Compute feature IQR from data (for robust error metrics)
        logger.info("Computing feature IQR for robust error metrics...")
        all_numeric = torch.cat([x_eicu_numeric, x_mimic_numeric], dim=0)
        self.model.feature_iqr = self.model.compute_feature_iqr(all_numeric)
        
        # Perform translations (DETERMINISTIC for evaluation)
        logger.info("Performing eICU to MIMIC translation (deterministic)...")
        x_eicu_to_mimic = self.model.translate_eicu_to_mimic_deterministic(x_eicu)
        logger.info("Performing MIMIC to eICU translation (deterministic)...")
        x_mimic_to_eicu = self.model.translate_mimic_to_eicu_deterministic(x_mimic)
        
        # Split translated outputs
        x_eicu_to_mimic_numeric = x_eicu_to_mimic[:, :numeric_dim]
        x_mimic_to_eicu_numeric = x_mimic_to_eicu[:, :numeric_dim]
        
        # Round-trip translations (DETERMINISTIC for evaluation)
        logger.info("Performing round-trip translations (deterministic)...")
        x_eicu_roundtrip = self.model.translate_mimic_to_eicu_deterministic(x_eicu_to_mimic)
        x_mimic_roundtrip = self.model.translate_eicu_to_mimic_deterministic(x_mimic_to_eicu)
        
        # Split roundtrip outputs
        x_eicu_roundtrip_numeric = x_eicu_roundtrip[:, :numeric_dim]
        x_mimic_roundtrip_numeric = x_mimic_roundtrip[:, :numeric_dim]
        
        logger.info("Translations completed successfully")
        
        # Run all evaluations
        results = {}
        
        # NEW: Per-feature percentage error metrics (reconstruction)
        logger.info("Computing per-feature percentage errors for reconstruction...")
        try:
            # eICU reconstruction (direct A->A')
            with torch.no_grad():
                outputs_eicu = self.model.forward(x_eicu, torch.zeros(x_eicu.shape[0], dtype=torch.long, device=device))
                x_eicu_recon_numeric = outputs_eicu['x_recon'][:, :numeric_dim]
            
            results['eicu_reconstruction_errors'] = self.model.compute_per_feature_percentage_errors(
                x_eicu_numeric, x_eicu_recon_numeric, x_eicu_missing, mode='reconstruction'
            )
            
            # MIMIC reconstruction
            with torch.no_grad():
                outputs_mimic = self.model.forward(x_mimic, torch.ones(x_mimic.shape[0], dtype=torch.long, device=device))
                x_mimic_recon_numeric = outputs_mimic['x_recon'][:, :numeric_dim]
            
            results['mimic_reconstruction_errors'] = self.model.compute_per_feature_percentage_errors(
                x_mimic_numeric, x_mimic_recon_numeric, x_mimic_missing, mode='reconstruction'
            )
            logger.info("✓ Reconstruction percentage errors completed")
        except Exception as e:
            logger.error(f"✗ Reconstruction percentage errors failed: {e}")
            results['eicu_reconstruction_errors'] = None
            results['mimic_reconstruction_errors'] = None
        
        # NEW: Per-feature percentage error metrics (cycle)
        logger.info("Computing per-feature percentage errors for cycle...")
        try:
            results['eicu_cycle_errors'] = self.model.compute_per_feature_percentage_errors(
                x_eicu_numeric, x_eicu_roundtrip_numeric, x_eicu_missing, mode='cycle'
            )
            results['mimic_cycle_errors'] = self.model.compute_per_feature_percentage_errors(
                x_mimic_numeric, x_mimic_roundtrip_numeric, x_mimic_missing, mode='cycle'
            )
            logger.info("✓ Cycle percentage errors completed")
        except Exception as e:
            logger.error(f"✗ Cycle percentage errors failed: {e}")
            results['eicu_cycle_errors'] = None
            results['mimic_cycle_errors'] = None
        
        # NEW: Latent space distance metrics
        logger.info("Computing latent space distance metrics...")
        try:
            with torch.no_grad():
                z_eicu = self.model.encoder(x_eicu)[0]  # Get mu
                z_mimic = self.model.encoder(x_mimic)[0]
                z_eicu_to_mimic = self.model.encoder(x_eicu_to_mimic)[0]
            
            results['latent_distance_eicu_vs_mimic'] = self.model.compute_latent_distance(z_eicu, z_mimic)
            results['latent_distance_translated_vs_real'] = self.model.compute_latent_distance(
                z_eicu_to_mimic, z_mimic
            )
            logger.info("✓ Latent distance metrics completed")
        except Exception as e:
            logger.error(f"✗ Latent distance metrics failed: {e}")
            results['latent_distance_eicu_vs_mimic'] = None
            results['latent_distance_translated_vs_real'] = None
        
        # NEW: Per-feature distribution distance
        logger.info("Computing per-feature distribution distance...")
        try:
            results['distribution_distance_eicu_to_mimic'] = self.model.compute_per_feature_distribution_distance(
                x_eicu_to_mimic_numeric, x_mimic_numeric
            )
            results['distribution_distance_mimic_to_eicu'] = self.model.compute_per_feature_distribution_distance(
                x_mimic_to_eicu_numeric, x_eicu_numeric
            )
            logger.info("✓ Distribution distance metrics completed")
        except Exception as e:
            logger.error(f"✗ Distribution distance metrics failed: {e}")
            results['distribution_distance_eicu_to_mimic'] = None
            results['distribution_distance_mimic_to_eicu'] = None
        
        # Convert to numpy for legacy analysis
        x_eicu_np = x_eicu.detach().cpu().numpy()
        x_mimic_np = x_mimic.detach().cpu().numpy()
        x_eicu_to_mimic_np = x_eicu_to_mimic.detach().cpu().numpy()
        x_mimic_to_eicu_np = x_mimic_to_eicu.detach().cpu().numpy()
        x_eicu_roundtrip_np = x_eicu_roundtrip.detach().cpu().numpy()
        x_mimic_roundtrip_np = x_mimic_roundtrip.detach().cpu().numpy()
        
        # 1. Per-feature correlation and R² metrics
        logger.info("Computing per-feature correlation and R² metrics...")
        try:
            results['correlation_metrics'] = self._compute_correlation_metrics(
                x_eicu_np, x_eicu_roundtrip_np, x_mimic_np, x_mimic_roundtrip_np
            )
            logger.info("✓ Correlation metrics completed")
        except Exception as e:
            logger.error(f"✗ Correlation metrics failed: {e}")
            results['correlation_metrics'] = None
        
        # 2. KS p-values and significance testing
        logger.info("Computing KS p-values and significance testing...")
        try:
            results['ks_analysis'] = self._compute_ks_analysis(
                x_eicu_np, x_eicu_to_mimic_np, x_mimic_np, x_mimic_to_eicu_np
            )
            logger.info("✓ KS analysis completed")
        except Exception as e:
            logger.error(f"✗ KS analysis failed: {e}")
            results['ks_analysis'] = None
        
        # 3. Missingness-aware stratified evaluation
        logger.info("Computing missingness-aware metrics...")
        try:
            results['missingness_analysis'] = self._compute_missingness_analysis(
                x_eicu_np, x_eicu_to_mimic_np, x_mimic_np, x_mimic_to_eicu_np
            )
            logger.info("✓ Missingness analysis completed")
        except Exception as e:
            logger.error(f"✗ Missingness analysis failed: {e}")
            results['missingness_analysis'] = None
        
        # 4. Demographic group evaluation
        logger.info("Computing demographic group evaluation...")
        try:
            results['demographic_analysis'] = self._compute_demographic_analysis(
                x_eicu_np, x_eicu_to_mimic_np, x_mimic_np, x_mimic_to_eicu_np
            )
            logger.info("✓ Demographic analysis completed")
        except Exception as e:
            logger.error(f"✗ Demographic analysis failed: {e}")
            results['demographic_analysis'] = None
        
        # 5. Per-feature summary statistics
        logger.info("Computing per-feature summary statistics...")
        try:
            results['summary_statistics'] = self._compute_summary_statistics(
                x_eicu_np, x_eicu_to_mimic_np, x_mimic_np, x_mimic_to_eicu_np
            )
            logger.info("✓ Summary statistics completed")
        except Exception as e:
            logger.error(f"✗ Summary statistics failed: {e}")
            results['summary_statistics'] = None
        
        # 6. Create visualizations
        logger.info("Creating visualizations...")
        try:
            self._create_visualizations(
                x_eicu_np, x_eicu_to_mimic_np, x_mimic_np, x_mimic_to_eicu_np,
                x_eicu_roundtrip_np, x_mimic_roundtrip_np
            )
            logger.info("✓ Visualizations completed")
        except Exception as e:
            logger.error(f"✗ Visualizations failed: {e}")
        
        # 7. Example patient row comparisons
        logger.info("Creating example patient comparisons...")
        try:
            results['example_patients'] = self._create_example_comparisons(
                x_eicu_np, x_eicu_to_mimic_np, x_mimic_np, x_mimic_to_eicu_np
            )
            logger.info("✓ Example patients completed")
        except Exception as e:
            logger.error(f"✗ Example patients failed: {e}")
            results['example_patients'] = None
        
        # Save results
        logger.info("Saving results...")
        try:
            self._save_results(results)
            logger.info("✓ Results saved successfully")
        except Exception as e:
            logger.error(f"✗ Saving results failed: {e}")
        
        # Print summary
        logger.info("Printing evaluation summary...")
        try:
            self._print_evaluation_summary(results)
            logger.info("✓ Evaluation summary completed")
        except Exception as e:
            logger.error(f"✗ Evaluation summary failed: {e}")
        
        return results
    
    def _compute_correlation_metrics(self, x_eicu: np.ndarray, x_eicu_roundtrip: np.ndarray,
                                   x_mimic: np.ndarray, x_mimic_roundtrip: np.ndarray) -> Dict:
        """
        Compute per-feature correlation and R² metrics (CLINICAL FEATURES ONLY).
        Excludes demographics and missing flags as they are input-only.
        """
        # Only compute on clinical features
        n_clinical = len(self.clinical_indices)
        
        # Extract clinical features only
        x_eicu_clinical = x_eicu[:, self.clinical_indices]
        x_eicu_roundtrip_clinical = x_eicu_roundtrip[:, self.clinical_indices]
        x_mimic_clinical = x_mimic[:, self.clinical_indices]
        x_mimic_roundtrip_clinical = x_mimic_roundtrip[:, self.clinical_indices]
        
        metrics = {
            'eicu_roundtrip': {
                'r2_scores': np.zeros(n_clinical),
                'correlations': np.zeros(n_clinical),
                'mse': np.zeros(n_clinical),
                'mae': np.zeros(n_clinical)
            },
            'mimic_roundtrip': {
                'r2_scores': np.zeros(n_clinical),
                'correlations': np.zeros(n_clinical),
                'mse': np.zeros(n_clinical),
                'mae': np.zeros(n_clinical)
            }
        }
        
        for i in range(n_clinical):
            # eICU round-trip metrics
            if np.std(x_eicu_clinical[:, i]) > 1e-8:  # Avoid division by zero
                metrics['eicu_roundtrip']['r2_scores'][i] = r2_score(x_eicu_clinical[:, i], x_eicu_roundtrip_clinical[:, i])
                metrics['eicu_roundtrip']['correlations'][i] = np.corrcoef(x_eicu_clinical[:, i], x_eicu_roundtrip_clinical[:, i])[0, 1]
            metrics['eicu_roundtrip']['mse'][i] = mean_squared_error(x_eicu_clinical[:, i], x_eicu_roundtrip_clinical[:, i])
            metrics['eicu_roundtrip']['mae'][i] = mean_absolute_error(x_eicu_clinical[:, i], x_eicu_roundtrip_clinical[:, i])
            
            # MIMIC round-trip metrics
            if np.std(x_mimic_clinical[:, i]) > 1e-8:
                r2_val = r2_score(x_mimic_clinical[:, i], x_mimic_roundtrip_clinical[:, i])
                corr_matrix = np.corrcoef(x_mimic_clinical[:, i], x_mimic_roundtrip_clinical[:, i])
                corr_val = corr_matrix[0, 1]
                
                metrics['mimic_roundtrip']['r2_scores'][i] = r2_val
                metrics['mimic_roundtrip']['correlations'][i] = corr_val
                
            metrics['mimic_roundtrip']['mse'][i] = mean_squared_error(x_mimic_clinical[:, i], x_mimic_roundtrip_clinical[:, i])
            metrics['mimic_roundtrip']['mae'][i] = mean_absolute_error(x_mimic_clinical[:, i], x_mimic_roundtrip_clinical[:, i])
        
        # Create summary DataFrame (clinical features only)
        df = pd.DataFrame({
            'feature_name': self.clinical_only_features,
            'eicu_r2': metrics['eicu_roundtrip']['r2_scores'],
            'eicu_correlation': metrics['eicu_roundtrip']['correlations'],
            'eicu_mse': metrics['eicu_roundtrip']['mse'],
            'eicu_mae': metrics['eicu_roundtrip']['mae'],
            'mimic_r2': metrics['mimic_roundtrip']['r2_scores'],
            'mimic_correlation': metrics['mimic_roundtrip']['correlations'],
            'mimic_mse': metrics['mimic_roundtrip']['mse'],
            'mimic_mae': metrics['mimic_roundtrip']['mae']
        })
        
        # Add quality flags
        df['eicu_good_quality'] = (df['eicu_r2'] > 0.5) & (df['eicu_correlation'] > 0.7)
        df['mimic_good_quality'] = (df['mimic_r2'] > 0.5) & (df['mimic_correlation'] > 0.7)
        
        metrics['summary_df'] = df
        
        logger.info(f"Correlation metrics computed on {n_clinical} clinical features (excluded {len(self.demographic_features)} demographics and {len(self.missing_features)} missing flags)")
        
        return metrics
    
    def _compute_ks_analysis(self, x_eicu: np.ndarray, x_eicu_to_mimic: np.ndarray,
                           x_mimic: np.ndarray, x_mimic_to_eicu: np.ndarray) -> Dict:
        """
        UPDATED: Compute KS and Wasserstein distance (CLINICAL FEATURES ONLY).
        Excludes demographics and missing flags as they are input-only.
        """
        # Only compute on clinical features
        n_clinical = len(self.clinical_indices)
        
        # Extract clinical features only
        x_eicu_clinical = x_eicu[:, self.clinical_indices]
        x_eicu_to_mimic_clinical = x_eicu_to_mimic[:, self.clinical_indices]
        x_mimic_clinical = x_mimic[:, self.clinical_indices]
        x_mimic_to_eicu_clinical = x_mimic_to_eicu[:, self.clinical_indices]
        
        ks_results = {
            'eicu_to_mimic': {
                'ks_stats': np.zeros(n_clinical),
                'p_values': np.zeros(n_clinical),
                'significant': np.zeros(n_clinical, dtype=bool),
                'wasserstein': np.zeros(n_clinical)
            },
            'mimic_to_eicu': {
                'ks_stats': np.zeros(n_clinical),
                'p_values': np.zeros(n_clinical),
                'significant': np.zeros(n_clinical, dtype=bool),
                'wasserstein': np.zeros(n_clinical)
            }
        }
        
        for i in range(n_clinical):
            # eICU to MIMIC translation
            ks_stat, p_val = stats.ks_2samp(x_mimic_clinical[:, i], x_eicu_to_mimic_clinical[:, i])
            wass_dist = stats.wasserstein_distance(x_mimic_clinical[:, i], x_eicu_to_mimic_clinical[:, i])
            ks_results['eicu_to_mimic']['ks_stats'][i] = ks_stat
            ks_results['eicu_to_mimic']['p_values'][i] = p_val
            ks_results['eicu_to_mimic']['significant'][i] = p_val < 0.05
            ks_results['eicu_to_mimic']['wasserstein'][i] = wass_dist
            
            # MIMIC to eICU translation
            ks_stat, p_val = stats.ks_2samp(x_eicu_clinical[:, i], x_mimic_to_eicu_clinical[:, i])
            wass_dist = stats.wasserstein_distance(x_eicu_clinical[:, i], x_mimic_to_eicu_clinical[:, i])
            ks_results['mimic_to_eicu']['ks_stats'][i] = ks_stat
            ks_results['mimic_to_eicu']['p_values'][i] = p_val
            ks_results['mimic_to_eicu']['significant'][i] = p_val < 0.05
            ks_results['mimic_to_eicu']['wasserstein'][i] = wass_dist
        
        # Create summary DataFrame (clinical features only)
        df = pd.DataFrame({
            'feature_name': self.clinical_only_features,
            'eicu_to_mimic_ks': ks_results['eicu_to_mimic']['ks_stats'],
            'eicu_to_mimic_pvalue': ks_results['eicu_to_mimic']['p_values'],
            'eicu_to_mimic_significant': ks_results['eicu_to_mimic']['significant'],
            'eicu_to_mimic_wasserstein': ks_results['eicu_to_mimic']['wasserstein'],
            'mimic_to_eicu_ks': ks_results['mimic_to_eicu']['ks_stats'],
            'mimic_to_eicu_pvalue': ks_results['mimic_to_eicu']['p_values'],
            'mimic_to_eicu_significant': ks_results['mimic_to_eicu']['significant'],
            'mimic_to_eicu_wasserstein': ks_results['mimic_to_eicu']['wasserstein']
        })
        
        # Add quality flags based on KS statistic only (p-value uninformative with large N)
        # KS interpretation: <0.1=excellent, <0.2=good, <0.3=acceptable, >0.5=poor
        df['eicu_to_mimic_excellent'] = (df['eicu_to_mimic_ks'] < 0.1)
        df['eicu_to_mimic_good'] = (df['eicu_to_mimic_ks'] < 0.2)
        df['eicu_to_mimic_acceptable'] = (df['eicu_to_mimic_ks'] < 0.3)
        df['mimic_to_eicu_excellent'] = (df['mimic_to_eicu_ks'] < 0.1)
        df['mimic_to_eicu_good'] = (df['mimic_to_eicu_ks'] < 0.2)
        df['mimic_to_eicu_acceptable'] = (df['mimic_to_eicu_ks'] < 0.3)
        
        ks_results['summary_df'] = df
        
        logger.info(f"KS analysis computed on {n_clinical} clinical features (excluded {len(self.demographic_features)} demographics and {len(self.missing_features)} missing flags)")
        
        return ks_results
    
    def _compute_missingness_analysis(self, x_eicu: np.ndarray, x_eicu_to_mimic: np.ndarray,
                                    x_mimic: np.ndarray, x_mimic_to_eicu: np.ndarray) -> Dict:
        """Compute missingness-aware stratified evaluation."""
        # Assume missing features are at the end of the feature vector
        missing_start_idx = len(self.numeric_features)
        
        # Compute feature counts (number of non-missing features per patient)
        eicu_feature_counts = np.sum(x_eicu[:, :missing_start_idx] != 0, axis=1)
        mimic_feature_counts = np.sum(x_mimic[:, :missing_start_idx] != 0, axis=1)
        
        # Define feature count buckets
        eicu_buckets = pd.cut(eicu_feature_counts, bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        mimic_buckets = pd.cut(mimic_feature_counts, bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        results = {
            'eicu_buckets': eicu_buckets,
            'mimic_buckets': mimic_buckets,
            'bucket_analysis': {}
        }
        
        # Analyze performance by bucket
        for bucket in ['very_low', 'low', 'medium', 'high', 'very_high']:
            eicu_mask = eicu_buckets == bucket
            mimic_mask = mimic_buckets == bucket
            
            if np.sum(eicu_mask) > 10 and np.sum(mimic_mask) > 10:  # Minimum sample size
                # Compute MSE for each bucket
                eicu_mse = mean_squared_error(x_eicu[eicu_mask], x_eicu_to_mimic[eicu_mask])
                mimic_mse = mean_squared_error(x_mimic[mimic_mask], x_mimic_to_eicu[mimic_mask])
                
                results['bucket_analysis'][bucket] = {
                    'eicu_samples': np.sum(eicu_mask),
                    'mimic_samples': np.sum(mimic_mask),
                    'eicu_mse': eicu_mse,
                    'mimic_mse': mimic_mse,
                    'eicu_feature_count_mean': np.mean(eicu_feature_counts[eicu_mask]),
                    'mimic_feature_count_mean': np.mean(mimic_feature_counts[mimic_mask])
                }
        
        return results
    
    def _compute_demographic_analysis(self, x_eicu: np.ndarray, x_eicu_to_mimic: np.ndarray,
                                    x_mimic: np.ndarray, x_mimic_to_eicu: np.ndarray) -> Dict:
        """Compute demographic group evaluation."""
        # Find demographic features (Age, Gender)
        age_idx = None
        gender_idx = None
        
        for i, feature in enumerate(self.all_features):
            if 'Age' in feature:
                age_idx = i
            elif 'Gender' in feature:
                gender_idx = i
        
        results = {}
        
        if age_idx is not None:
            # Age-based analysis
            age_values = x_eicu[:, age_idx]
            age_buckets = pd.cut(age_values, bins=5, labels=['young', 'adult', 'middle', 'senior', 'elderly'])
            
            results['age_analysis'] = {}
            for bucket in ['young', 'adult', 'middle', 'senior', 'elderly']:
                mask = age_buckets == bucket
                if np.sum(mask) > 10:
                    mse = mean_squared_error(x_eicu[mask], x_eicu_to_mimic[mask])
                    results['age_analysis'][bucket] = {
                        'samples': np.sum(mask),
                        'mse': mse,
                        'age_mean': np.mean(age_values[mask])
                    }
        
        if gender_idx is not None:
            # Gender-based analysis
            gender_values = x_eicu[:, gender_idx]
            results['gender_analysis'] = {}
            
            for gender in [0, 1]:  # Assuming binary gender encoding
                mask = gender_values == gender
                if np.sum(mask) > 10:
                    mse = mean_squared_error(x_eicu[mask], x_eicu_to_mimic[mask])
                    results['gender_analysis'][f'gender_{gender}'] = {
                        'samples': np.sum(mask),
                        'mse': mse
                    }
        
        return results
    
    def _compute_summary_statistics(self, x_eicu: np.ndarray, x_eicu_to_mimic: np.ndarray,
                                  x_mimic: np.ndarray, x_mimic_to_eicu: np.ndarray) -> Dict:
        """Compute per-feature summary statistics."""
        n_features = x_eicu.shape[1]
        
        stats_data = []
        for i in range(n_features):
            feature_name = self.all_features[i]
            
            # eICU original
            eicu_orig = x_eicu[:, i]
            # eICU translated to MIMIC
            eicu_trans = x_eicu_to_mimic[:, i]
            # MIMIC original
            mimic_orig = x_mimic[:, i]
            # MIMIC translated to eICU
            mimic_trans = x_mimic_to_eicu[:, i]
            
            stats_data.append({
                'feature_name': feature_name,
                'eicu_orig_mean': np.mean(eicu_orig),
                'eicu_orig_std': np.std(eicu_orig),
                'eicu_orig_median': np.median(eicu_orig),
                'eicu_orig_iqr': np.percentile(eicu_orig, 75) - np.percentile(eicu_orig, 25),
                'eicu_trans_mean': np.mean(eicu_trans),
                'eicu_trans_std': np.std(eicu_trans),
                'eicu_trans_median': np.median(eicu_trans),
                'eicu_trans_iqr': np.percentile(eicu_trans, 75) - np.percentile(eicu_trans, 25),
                'mimic_orig_mean': np.mean(mimic_orig),
                'mimic_orig_std': np.std(mimic_orig),
                'mimic_orig_median': np.median(mimic_orig),
                'mimic_orig_iqr': np.percentile(mimic_orig, 75) - np.percentile(mimic_orig, 25),
                'mimic_trans_mean': np.mean(mimic_trans),
                'mimic_trans_std': np.std(mimic_trans),
                'mimic_trans_median': np.median(mimic_trans),
                'mimic_trans_iqr': np.percentile(mimic_trans, 75) - np.percentile(mimic_trans, 25)
            })
        
        return pd.DataFrame(stats_data)
    
    def _create_visualizations(self, x_eicu: np.ndarray, x_eicu_to_mimic: np.ndarray,
                             x_mimic: np.ndarray, x_mimic_to_eicu: np.ndarray,
                             x_eicu_roundtrip: np.ndarray, x_mimic_roundtrip: np.ndarray):
        """Create comprehensive visualizations."""
        logger.info("Starting visualization creation...")
        
        # 1. Distribution comparisons for key features
        logger.info("Creating distribution comparison plots...")
        try:
            self._plot_distribution_comparisons(x_eicu, x_eicu_to_mimic, x_mimic, x_mimic_to_eicu)
            logger.info("✓ Distribution comparison plots completed")
        except Exception as e:
            logger.error(f"✗ Distribution comparison plots failed: {e}")
        
        # 2. Scatter plots for round-trip consistency
        logger.info("Creating round-trip scatter plots...")
        try:
            self._plot_roundtrip_scatters(x_eicu, x_eicu_roundtrip, x_mimic, x_mimic_roundtrip)
            logger.info("✓ Round-trip scatter plots completed")
        except Exception as e:
            logger.error(f"✗ Round-trip scatter plots failed: {e}")
        
        # 3. Correlation heatmaps
        logger.info("Creating correlation heatmaps...")
        try:
            self._plot_correlation_heatmaps(x_eicu, x_eicu_to_mimic, x_mimic, x_mimic_to_eicu)
            logger.info("✓ Correlation heatmaps completed")
        except Exception as e:
            logger.error(f"✗ Correlation heatmaps failed: {e}")
        
        # 4. Feature quality summary
        logger.info("Creating feature quality summary...")
        try:
            self._plot_feature_quality_summary()
            logger.info("✓ Feature quality summary completed")
        except Exception as e:
            logger.error(f"✗ Feature quality summary failed: {e}")
        
        logger.info("All visualizations completed")
    
    def _plot_distribution_comparisons(self, x_eicu: np.ndarray, x_eicu_to_mimic: np.ndarray,
                                     x_mimic: np.ndarray, x_mimic_to_eicu: np.ndarray):
        """Plot distribution comparisons for key features."""
        # Select key features to visualize
        key_features = ['SpO2_max', 'RR_min', 'Na_std', 'HR_min', 'RR_mean', 'SpO2_mean', 'WBC_std']
        key_indices = []
        key_names = []
        
        for feature in key_features:
            for i, name in enumerate(self.all_features):
                if feature in name:
                    key_indices.append(i)
                    key_names.append(name)
                    break
        
        if not key_indices:
            logger.warning("No key features found for distribution plots")
            return
        
        n_features = len(key_indices)
        fig, axes = plt.subplots(n_features, 2, figsize=(15, 4 * n_features))
        if n_features == 1:
            axes = axes.reshape(1, -1)
        
        for i, (idx, name) in enumerate(zip(key_indices, key_names)):
            # eICU to MIMIC translation
            axes[i, 0].hist(x_mimic[:, idx], bins=50, alpha=0.7, label='True MIMIC', density=True)
            axes[i, 0].hist(x_eicu_to_mimic[:, idx], bins=50, alpha=0.7, label='eICU→MIMIC', density=True)
            axes[i, 0].set_title(f'{name}: eICU→MIMIC Translation')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # MIMIC to eICU translation
            axes[i, 1].hist(x_eicu[:, idx], bins=50, alpha=0.7, label='True eICU', density=True)
            axes[i, 1].hist(x_mimic_to_eicu[:, idx], bins=50, alpha=0.7, label='MIMIC→eICU', density=True)
            axes[i, 1].set_title(f'{name}: MIMIC→eICU Translation')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / "plots" / "distribution_comparisons.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roundtrip_scatters(self, x_eicu: np.ndarray, x_eicu_roundtrip: np.ndarray,
                               x_mimic: np.ndarray, x_mimic_roundtrip: np.ndarray):
        """Plot scatter plots for round-trip consistency."""
        # Select key features
        key_features = ['HR_mean', 'Temp_mean', 'Na_mean', 'Creat_mean']
        key_indices = []
        key_names = []
        
        for feature in key_features:
            for i, name in enumerate(self.all_features):
                if feature in name:
                    key_indices.append(i)
                    key_names.append(name)
                    break
        
        if not key_indices:
            return
        
        n_features = len(key_indices)
        fig, axes = plt.subplots(2, n_features, figsize=(4 * n_features, 8))
        if n_features == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (idx, name) in enumerate(zip(key_indices, key_names)):
            # eICU round-trip
            axes[0, i].scatter(x_eicu[:, idx], x_eicu_roundtrip[:, idx], alpha=0.5, s=1)
            axes[0, i].plot([x_eicu[:, idx].min(), x_eicu[:, idx].max()], 
                           [x_eicu[:, idx].min(), x_eicu[:, idx].max()], 'r--', alpha=0.8)
            axes[0, i].set_xlabel('Original eICU')
            axes[0, i].set_ylabel('Round-trip eICU')
            axes[0, i].set_title(f'{name}: eICU Round-trip')
            axes[0, i].grid(True, alpha=0.3)
            
            # MIMIC round-trip
            axes[1, i].scatter(x_mimic[:, idx], x_mimic_roundtrip[:, idx], alpha=0.5, s=1)
            axes[1, i].plot([x_mimic[:, idx].min(), x_mimic[:, idx].max()], 
                           [x_mimic[:, idx].min(), x_mimic[:, idx].max()], 'r--', alpha=0.8)
            axes[1, i].set_xlabel('Original MIMIC')
            axes[1, i].set_ylabel('Round-trip MIMIC')
            axes[1, i].set_title(f'{name}: MIMIC Round-trip')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / "plots" / "roundtrip_scatters.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmaps(self, x_eicu: np.ndarray, x_eicu_to_mimic: np.ndarray,
                                 x_mimic: np.ndarray, x_mimic_to_eicu: np.ndarray):
        """Plot correlation heatmaps."""
        # Compute correlation matrices
        corr_eicu = np.corrcoef(x_eicu.T)
        corr_mimic = np.corrcoef(x_mimic.T)
        corr_eicu_trans = np.corrcoef(x_eicu_to_mimic.T)
        corr_mimic_trans = np.corrcoef(x_mimic_to_eicu.T)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # eICU original
        sns.heatmap(corr_eicu, ax=axes[0, 0], cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        axes[0, 0].set_title('eICU Original Correlations')
        
        # eICU translated
        sns.heatmap(corr_eicu_trans, ax=axes[0, 1], cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        axes[0, 1].set_title('eICU→MIMIC Translated Correlations')
        
        # MIMIC original
        sns.heatmap(corr_mimic, ax=axes[1, 0], cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        axes[1, 0].set_title('MIMIC Original Correlations')
        
        # MIMIC translated
        sns.heatmap(corr_mimic_trans, ax=axes[1, 1], cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        axes[1, 1].set_title('MIMIC→eICU Translated Correlations')
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / "plots" / "correlation_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_quality_summary(self):
        """Plot feature quality summary."""
        # This will be called after all metrics are computed
        pass
    
    def _create_example_comparisons(self, x_eicu: np.ndarray, x_eicu_to_mimic: np.ndarray,
                                  x_mimic: np.ndarray, x_mimic_to_eicu: np.ndarray) -> Dict:
        """Create example patient row comparisons."""
        # Select a few random patients
        n_examples = 3
        eicu_indices = np.random.choice(len(x_eicu), n_examples, replace=False)
        mimic_indices = np.random.choice(len(x_mimic), n_examples, replace=False)
        
        examples = {
            'eicu_patients': [],
            'mimic_patients': []
        }
        
        for i, (eicu_idx, mimic_idx) in enumerate(zip(eicu_indices, mimic_indices)):
            # eICU example
            eicu_example = {
                'patient_id': f'eicu_{eicu_idx}',
                'original': x_eicu[eicu_idx].tolist(),
                'translated': x_eicu_to_mimic[eicu_idx].tolist(),
                'feature_names': self.all_features
            }
            examples['eicu_patients'].append(eicu_example)
            
            # MIMIC example
            mimic_example = {
                'patient_id': f'mimic_{mimic_idx}',
                'original': x_mimic[mimic_idx].tolist(),
                'translated': x_mimic_to_eicu[mimic_idx].tolist(),
                'feature_names': self.all_features
            }
            examples['mimic_patients'].append(mimic_example)
        
        return examples
    
    def _save_results(self, results: Dict):
        """Save all results to files."""
        # Save correlation metrics
        if 'correlation_metrics' in results:
            results['correlation_metrics']['summary_df'].to_csv(
                self.eval_dir / "data" / "correlation_metrics.csv", index=False
            )
        
        # Save KS analysis
        if 'ks_analysis' in results:
            results['ks_analysis']['summary_df'].to_csv(
                self.eval_dir / "data" / "ks_analysis.csv", index=False
            )
        
        # Save summary statistics
        if 'summary_statistics' in results:
            results['summary_statistics'].to_csv(
                self.eval_dir / "data" / "summary_statistics.csv", index=False
            )
        
        # Save example patients
        if 'example_patients' in results:
            with open(self.eval_dir / "data" / "example_patients.json", 'w') as f:
                json.dump(results['example_patients'], f, indent=2)
        
        # Save all results with validation
        try:
            # Test JSON serialization functionality first
            logger.info("Running JSON serialization test...")
            self._test_json_serialization()
            
            # Convert numpy arrays and pandas DataFrames to JSON serializable format
            serializable_results = self._make_json_serializable(results)
            
            # Validate JSON serialization before writing to file
            logger.info("Validating JSON serialization...")
            json_str = json.dumps(serializable_results, indent=2)
            logger.info("JSON validation successful")
            
            # Write to file
            with open(self.eval_dir / "comprehensive_results.json", 'w') as f:
                f.write(json_str)
            logger.info("Comprehensive results saved successfully")
        except TypeError as e:
            logger.error(f"JSON serialization failed - unsupported data type: {e}")
            # Find the problematic object
            self._debug_json_serialization_error(results)
            raise
        except Exception as e:
            logger.error(f"Failed to save comprehensive results: {e}")
            # Save a backup with timestamp
            backup_path = self.eval_dir / f"comprehensive_results_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            logger.info(f"Attempting to save backup to {backup_path}")
            raise
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays, pandas DataFrames, torch tensors, and all numpy/pandas types to JSON serializable format."""
        import torch
        
        # Handle None explicitly
        if obj is None:
            return None
            
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # Handle pandas types
        elif isinstance(obj, pd.DataFrame):
            # Convert DataFrame to dictionary of lists for JSON serialization
            return obj.to_dict('list')
        elif isinstance(obj, pd.Series):
            # Convert Series to list
            return obj.tolist()
        elif isinstance(obj, pd.Categorical):
            # Convert Categorical to list
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            # Handle pandas timestamp objects
            return str(obj)
            
        # Handle torch tensors
        elif isinstance(obj, torch.Tensor):
            # Convert torch tensor to numpy then to list
            return obj.detach().cpu().numpy().tolist()
            
        # Handle collection types recursively
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [self._make_json_serializable(item) for item in obj]
            
        # Handle ALL numpy scalar types
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj.item())
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj.item())
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj.item())
        elif isinstance(obj, np.complex_):
            return str(obj)  # JSON doesn't support complex numbers
        elif isinstance(obj, np.generic):
            # Catch-all for any remaining numpy types
            try:
                return obj.item()
            except (ValueError, AttributeError):
                return str(obj)
                
        # Handle Python native types that might cause issues
        elif isinstance(obj, (np.str_, bytes)):
            return str(obj)
            
        # Fallback for objects with item() method
        elif hasattr(obj, 'item') and callable(obj.item):
            try:
                return obj.item()
            except (ValueError, TypeError, AttributeError):
                return str(obj)
                
        # Handle standard JSON-serializable types
        elif isinstance(obj, (str, int, float, bool)):
            return obj
            
        # Final fallback
        else:
            # Try to convert to string if all else fails
            try:
                return str(obj)
            except:
                logger.warning(f"Unable to serialize object of type {type(obj)}: {obj}")
                return f"<Unserializable: {type(obj).__name__}>"
    
    def _test_json_serialization(self):
        """Test JSON serialization with problematic data types found in the logs."""
        import json
        
        # Test problematic types from the error logs
        test_data = {
            'numpy_array': np.array([1, 2, 3]),
            'numpy_int64': np.int64(42),
            'numpy_float32': np.float32(3.14),
            'pandas_categorical': pd.Categorical(['A', 'B', 'C']),
            'pandas_series': pd.Series([1, 2, 3]),
            'nested_dict': {
                'nested_array': np.array([4, 5, 6]),
                'nested_scalar': np.float64(2.71)
            },
            'list_with_numpy': [np.int32(1), np.float16(2.5), np.bool_(True)]
        }
        
        try:
            serializable = self._make_json_serializable(test_data)
            json_str = json.dumps(serializable, indent=2)
            logger.info("✅ JSON serialization test passed")
            return True
        except Exception as e:
            logger.error(f"❌ JSON serialization test failed: {e}")
            return False
    
    def _debug_json_serialization_error(self, obj, path="root"):
        """Debug JSON serialization by finding problematic objects."""
        try:
            json.dumps(obj)
            return  # This object is fine
        except TypeError as e:
            logger.error(f"JSON error at {path}: {e}")
            logger.error(f"Object type: {type(obj)}")
            if hasattr(obj, '__dict__'):
                logger.error(f"Object dict: {obj.__dict__}")
            
            # Recursively check dict values
            if isinstance(obj, dict):
                for key, value in obj.items():
                    try:
                        json.dumps(value)
                    except TypeError:
                        logger.error(f"Problematic key: {path}.{key}, type: {type(value)}")
                        if hasattr(value, '__dict__'):
                            logger.error(f"Value dict: {value.__dict__}")
                        # Recurse deeper
                        self._debug_json_serialization_error(value, f"{path}.{key}")
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    try:
                        json.dumps(item)
                    except TypeError:
                        logger.error(f"Problematic index: {path}[{i}], type: {type(item)}")
                        self._debug_json_serialization_error(item, f"{path}[{i}]")
    
    def _print_evaluation_summary(self, results: Dict):
        """UPDATED: Print evaluation summary with simplified model metrics."""
        logger.info("=" * 80)
        logger.info("=== COMPREHENSIVE EVALUATION SUMMARY (SIMPLIFIED MODEL) ===")
        logger.info("=" * 80)
        
        # NEW: Per-feature percentage errors - Reconstruction
        if 'eicu_reconstruction_errors' in results and results['eicu_reconstruction_errors'] is not None:
            logger.info("\n📊 RECONSTRUCTION QUALITY (A→A')")
            logger.info("-" * 80)
            logger.info("  NOTE: Data is normalized - MAE in standard deviation units, use IQR metrics")
            
            eicu_err = results['eicu_reconstruction_errors']
            mimic_err = results['mimic_reconstruction_errors']
            
            # Average across clinical features
            eicu_mae_avg = eicu_err['mae'].mean().item()
            mimic_mae_avg = mimic_err['mae'].mean().item()
            
            logger.info(f"  eICU Reconstruction:")
            logger.info(f"    - MAE: {eicu_mae_avg:.4f} (std dev units)")
            logger.info(f"    - Median Error: {eicu_err['median_abs_error'].mean().item():.4f}")
            logger.info(f"    - % within 0.5 IQR: {eicu_err['pct_within_iqr']['within_0.5_iqr'].mean().item():.1f}%")
            logger.info(f"    - % within 1.0 IQR: {eicu_err['pct_within_iqr']['within_1.0_iqr'].mean().item():.1f}%")
            
            logger.info(f"  MIMIC Reconstruction:")
            logger.info(f"    - MAE: {mimic_mae_avg:.4f} (std dev units)")
            logger.info(f"    - Median Error: {mimic_err['median_abs_error'].mean().item():.4f}")
            logger.info(f"    - % within 0.5 IQR: {mimic_err['pct_within_iqr']['within_0.5_iqr'].mean().item():.1f}%")
            logger.info(f"    - % within 1.0 IQR: {mimic_err['pct_within_iqr']['within_1.0_iqr'].mean().item():.1f}%")
        
        # NEW: Per-feature percentage errors - Cycle
        if 'eicu_cycle_errors' in results and results['eicu_cycle_errors'] is not None:
            logger.info("\n🔄 CYCLE CONSISTENCY (A→B'→A')")
            logger.info("-" * 80)
            logger.info("  NOTE: Data is normalized - use IQR metrics for meaningful percentages")
            
            eicu_cyc = results['eicu_cycle_errors']
            mimic_cyc = results['mimic_cycle_errors']
            
            logger.info(f"  eICU Cycle:")
            logger.info(f"    - MAE: {eicu_cyc['mae'].mean().item():.4f} (std dev units)")
            logger.info(f"    - % within 0.5 IQR: {eicu_cyc['pct_within_iqr']['within_0.5_iqr'].mean().item():.1f}%")
            logger.info(f"    - % within 1.0 IQR: {eicu_cyc['pct_within_iqr']['within_1.0_iqr'].mean().item():.1f}%")
            
            logger.info(f"  MIMIC Cycle:")
            logger.info(f"    - MAE: {mimic_cyc['mae'].mean().item():.4f} (std dev units)")
            logger.info(f"    - % within 0.5 IQR: {mimic_cyc['pct_within_iqr']['within_0.5_iqr'].mean().item():.1f}%")
            logger.info(f"    - % within 1.0 IQR: {mimic_cyc['pct_within_iqr']['within_1.0_iqr'].mean().item():.1f}%")
        
        # NEW: Latent space distance
        if 'latent_distance_eicu_vs_mimic' in results and results['latent_distance_eicu_vs_mimic'] is not None:
            logger.info("\n🧠 LATENT SPACE ANALYSIS")
            logger.info("-" * 80)
            
            latent_orig = results['latent_distance_eicu_vs_mimic']
            latent_trans = results['latent_distance_translated_vs_real']
            
            logger.info(f"  Original (eICU vs MIMIC):")
            logger.info(f"    - Euclidean Distance: {latent_orig['mean_euclidean_distance']:.4f}")
            logger.info(f"    - Cosine Similarity: {latent_orig['cosine_similarity']:.4f}")
            logger.info(f"    - KL Divergence: {latent_orig['kl_divergence']:.4f}")
            
            logger.info(f"  After Translation (eICU→MIMIC vs real MIMIC):")
            logger.info(f"    - Euclidean Distance: {latent_trans['mean_euclidean_distance']:.4f}")
            logger.info(f"    - Cosine Similarity: {latent_trans['cosine_similarity']:.4f}")
            logger.info(f"    - KL Divergence: {latent_trans['kl_divergence']:.4f}")
        
        # NEW: Per-feature distribution distance
        if 'distribution_distance_eicu_to_mimic' in results and results['distribution_distance_eicu_to_mimic'] is not None:
            logger.info("\n📈 DISTRIBUTION MATCHING")
            logger.info("-" * 80)
            
            dist_e2m = results['distribution_distance_eicu_to_mimic']
            
            logger.info(f"  eICU→MIMIC Translation:")
            logger.info(f"    - Mean Wasserstein Distance: {dist_e2m['wasserstein_distances'].mean().item():.4f}")
            logger.info(f"    - Mean KS Statistic: {dist_e2m['ks_statistics'].mean().item():.4f}")
            logger.info(f"    - Mean Diff in Means: {dist_e2m['mean_differences'].mean().item():.4f}")
        
        # Legacy metrics
        if 'correlation_metrics' in results and results['correlation_metrics'] is not None:
            logger.info("\n📊 LEGACY: Feature Quality (R² > 0.5 & correlation > 0.7)")
            logger.info("-" * 80)
            df = results['correlation_metrics']['summary_df']
            good_eicu = df['eicu_good_quality'].sum()
            good_mimic = df['mimic_good_quality'].sum()
            total_features = len(df)
            
            logger.info(f"  eICU round-trip: {good_eicu}/{total_features} ({good_eicu/total_features*100:.1f}%)")
            logger.info(f"  MIMIC round-trip: {good_mimic}/{total_features} ({good_mimic/total_features*100:.1f}%)")
        
        if 'ks_analysis' in results and results['ks_analysis'] is not None:
            logger.info("\n📊 Distribution Matching (KS statistic thresholds)")
            logger.info("-" * 80)
            df = results['ks_analysis']['summary_df']
            total_features = len(df)
            
            # eICU→MIMIC
            excellent_e2m = df['eicu_to_mimic_excellent'].sum()
            good_e2m = df['eicu_to_mimic_good'].sum()
            acceptable_e2m = df['eicu_to_mimic_acceptable'].sum()
            
            logger.info(f"  eICU→MIMIC:")
            logger.info(f"    - Excellent (KS<0.1): {excellent_e2m}/{total_features} ({excellent_e2m/total_features*100:.1f}%)")
            logger.info(f"    - Good (KS<0.2): {good_e2m}/{total_features} ({good_e2m/total_features*100:.1f}%)")
            logger.info(f"    - Acceptable (KS<0.3): {acceptable_e2m}/{total_features} ({acceptable_e2m/total_features*100:.1f}%)")
            logger.info(f"    - Mean KS: {df['eicu_to_mimic_ks'].mean():.3f}")
            
            # MIMIC→eICU
            excellent_m2e = df['mimic_to_eicu_excellent'].sum()
            good_m2e = df['mimic_to_eicu_good'].sum()
            acceptable_m2e = df['mimic_to_eicu_acceptable'].sum()
            
            logger.info(f"  MIMIC→eICU:")
            logger.info(f"    - Excellent (KS<0.1): {excellent_m2e}/{total_features} ({excellent_m2e/total_features*100:.1f}%)")
            logger.info(f"    - Good (KS<0.2): {good_m2e}/{total_features} ({good_m2e/total_features*100:.1f}%)")
            logger.info(f"    - Acceptable (KS<0.3): {acceptable_m2e}/{total_features} ({acceptable_m2e/total_features*100:.1f}%)")
            logger.info(f"    - Mean KS: {df['mimic_to_eicu_ks'].mean():.3f}")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Results saved to: {self.eval_dir}")
        logger.info("=" * 80)
