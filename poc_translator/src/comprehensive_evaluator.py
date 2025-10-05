#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Cycle-VAE
Implements patient-level and feature-level evaluation metrics to assess translation quality.
"""

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
        
        # Create subdirectories
        (self.eval_dir / "plots").mkdir(exist_ok=True)
        (self.eval_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"Comprehensive evaluator initialized. Output directory: {self.eval_dir}")
    
    def evaluate_translation_quality(self, x_eicu: torch.Tensor, x_mimic: torch.Tensor) -> Dict:
        """
        Run comprehensive evaluation of translation quality.
        
        Args:
            x_eicu: eICU test data [n_samples, n_features]
            x_mimic: MIMIC test data [n_samples, n_features]
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Starting comprehensive translation quality evaluation...")
        logger.info(f"Input data shapes: eICU {x_eicu.shape}, MIMIC {x_mimic.shape}")
        
        # Move data to model device
        device = next(self.model.parameters()).device
        logger.info(f"Moving data to device: {device}")
        x_eicu = x_eicu.to(device)
        x_mimic = x_mimic.to(device)
        
        # Perform translations
        logger.info("Performing eICU to MIMIC translation...")
        x_eicu_to_mimic = self.model.translate_eicu_to_mimic(x_eicu)
        logger.info("Performing MIMIC to eICU translation...")
        x_mimic_to_eicu = self.model.translate_mimic_to_eicu(x_mimic)
        
        # Round-trip translations
        logger.info("Performing round-trip translations...")
        x_eicu_roundtrip = self.model.translate_mimic_to_eicu(x_eicu_to_mimic)
        x_mimic_roundtrip = self.model.translate_eicu_to_mimic(x_mimic_to_eicu)
        logger.info("Translations completed successfully")
        
        # Convert to numpy for analysis
        x_eicu_np = x_eicu.detach().cpu().numpy()
        x_mimic_np = x_mimic.detach().cpu().numpy()
        x_eicu_to_mimic_np = x_eicu_to_mimic.detach().cpu().numpy()
        x_mimic_to_eicu_np = x_mimic_to_eicu.detach().cpu().numpy()
        x_eicu_roundtrip_np = x_eicu_roundtrip.detach().cpu().numpy()
        x_mimic_roundtrip_np = x_mimic_roundtrip.detach().cpu().numpy()
        
        # Run all evaluations
        results = {}
        
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
        """Compute per-feature correlation and R² metrics."""
        n_features = x_eicu.shape[1]
        metrics = {
            'eicu_roundtrip': {
                'r2_scores': np.zeros(n_features),
                'correlations': np.zeros(n_features),
                'mse': np.zeros(n_features),
                'mae': np.zeros(n_features)
            },
            'mimic_roundtrip': {
                'r2_scores': np.zeros(n_features),
                'correlations': np.zeros(n_features),
                'mse': np.zeros(n_features),
                'mae': np.zeros(n_features)
            }
        }
        
        for i in range(n_features):
            # eICU round-trip metrics
            if np.std(x_eicu[:, i]) > 1e-8:  # Avoid division by zero
                metrics['eicu_roundtrip']['r2_scores'][i] = r2_score(x_eicu[:, i], x_eicu_roundtrip[:, i])
                metrics['eicu_roundtrip']['correlations'][i] = np.corrcoef(x_eicu[:, i], x_eicu_roundtrip[:, i])[0, 1]
            metrics['eicu_roundtrip']['mse'][i] = mean_squared_error(x_eicu[:, i], x_eicu_roundtrip[:, i])
            metrics['eicu_roundtrip']['mae'][i] = mean_absolute_error(x_eicu[:, i], x_eicu_roundtrip[:, i])
            
            # MIMIC round-trip metrics
            if np.std(x_mimic[:, i]) > 1e-8:
                metrics['mimic_roundtrip']['r2_scores'][i] = r2_score(x_mimic[:, i], x_mimic_roundtrip[:, i])
                metrics['mimic_roundtrip']['correlations'][i] = np.corrcoef(x_mimic[:, i], x_mimic_roundtrip[:, i])[0, 1]
            metrics['mimic_roundtrip']['mse'][i] = mean_squared_error(x_mimic[:, i], x_mimic_roundtrip[:, i])
            metrics['mimic_roundtrip']['mae'][i] = mean_absolute_error(x_mimic[:, i], x_mimic_roundtrip[:, i])
        
        # Create summary DataFrame
        df = pd.DataFrame({
            'feature_name': self.all_features,
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
        
        return metrics
    
    def _compute_ks_analysis(self, x_eicu: np.ndarray, x_eicu_to_mimic: np.ndarray,
                           x_mimic: np.ndarray, x_mimic_to_eicu: np.ndarray) -> Dict:
        """Compute KS p-values and significance testing."""
        n_features = x_eicu.shape[1]
        
        ks_results = {
            'eicu_to_mimic': {
                'ks_stats': np.zeros(n_features),
                'p_values': np.zeros(n_features),
                'significant': np.zeros(n_features, dtype=bool)
            },
            'mimic_to_eicu': {
                'ks_stats': np.zeros(n_features),
                'p_values': np.zeros(n_features),
                'significant': np.zeros(n_features, dtype=bool)
            }
        }
        
        for i in range(n_features):
            # eICU to MIMIC translation
            ks_stat, p_val = stats.ks_2samp(x_mimic[:, i], x_eicu_to_mimic[:, i])
            ks_results['eicu_to_mimic']['ks_stats'][i] = ks_stat
            ks_results['eicu_to_mimic']['p_values'][i] = p_val
            ks_results['eicu_to_mimic']['significant'][i] = p_val < 0.05
            
            # MIMIC to eICU translation
            ks_stat, p_val = stats.ks_2samp(x_eicu[:, i], x_mimic_to_eicu[:, i])
            ks_results['mimic_to_eicu']['ks_stats'][i] = ks_stat
            ks_results['mimic_to_eicu']['p_values'][i] = p_val
            ks_results['mimic_to_eicu']['significant'][i] = p_val < 0.05
        
        # Create summary DataFrame
        df = pd.DataFrame({
            'feature_name': self.all_features,
            'eicu_to_mimic_ks': ks_results['eicu_to_mimic']['ks_stats'],
            'eicu_to_mimic_pvalue': ks_results['eicu_to_mimic']['p_values'],
            'eicu_to_mimic_significant': ks_results['eicu_to_mimic']['significant'],
            'mimic_to_eicu_ks': ks_results['mimic_to_eicu']['ks_stats'],
            'mimic_to_eicu_pvalue': ks_results['mimic_to_eicu']['p_values'],
            'mimic_to_eicu_significant': ks_results['mimic_to_eicu']['significant']
        })
        
        # Add quality flags
        df['eicu_to_mimic_good'] = (df['eicu_to_mimic_ks'] < 0.3) & (df['eicu_to_mimic_pvalue'] > 0.05)
        df['mimic_to_eicu_good'] = (df['mimic_to_eicu_ks'] < 0.3) & (df['mimic_to_eicu_pvalue'] > 0.05)
        
        ks_results['summary_df'] = df
        
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
        key_features = ['HR_mean', 'Temp_mean', 'Na_mean', 'Creat_mean', 'Age']
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
        """Print evaluation summary."""
        logger.info("=== COMPREHENSIVE EVALUATION SUMMARY ===")
        
        # Correlation metrics summary
        if 'correlation_metrics' in results:
            df = results['correlation_metrics']['summary_df']
            good_eicu = df['eicu_good_quality'].sum()
            good_mimic = df['mimic_good_quality'].sum()
            total_features = len(df)
            
            logger.info(f"Feature Quality (R² > 0.5 & correlation > 0.7):")
            logger.info(f"  eICU round-trip: {good_eicu}/{total_features} ({good_eicu/total_features*100:.1f}%)")
            logger.info(f"  MIMIC round-trip: {good_mimic}/{total_features} ({good_mimic/total_features*100:.1f}%)")
        
        # KS analysis summary
        if 'ks_analysis' in results:
            df = results['ks_analysis']['summary_df']
            good_eicu = df['eicu_to_mimic_good'].sum()
            good_mimic = df['mimic_to_eicu_good'].sum()
            
            logger.info(f"Distribution Matching (KS < 0.3 & p > 0.05):")
            logger.info(f"  eICU→MIMIC: {good_eicu}/{total_features} ({good_eicu/total_features*100:.1f}%)")
            logger.info(f"  MIMIC→eICU: {good_mimic}/{total_features} ({good_mimic/total_features*100:.1f}%)")
        
        # Missingness analysis summary
        if 'missingness_analysis' in results:
            logger.info("Missingness Analysis:")
            for bucket, data in results['missingness_analysis']['bucket_analysis'].items():
                logger.info(f"  {bucket}: eICU MSE={data['eicu_mse']:.4f}, MIMIC MSE={data['mimic_mse']:.4f}")
        
        logger.info(f"Results saved to: {self.eval_dir}")
