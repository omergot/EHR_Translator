#!/usr/bin/env python3
"""
Evaluation Script for Cycle-VAE (SIMPLIFIED MODEL)
Performs comprehensive evaluation including:
- Per-feature percentage errors (reconstruction & cycle)
- Latent space distance metrics
- Per-feature distribution distance (Wasserstein, KS)
- Hybrid relative error thresholds (5%, 10%, 20%, 30%)
- Round-trip reconstruction quality
- Distributional tests
- Downstream evaluation

UPDATED: Compatible with simplified 3-loss model (reconstruction, cycle, conditional Wasserstein)
"""

import argparse
import yaml
import sys
import logging
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split
try:
    from umap import UMAP
except ImportError:
    try:
        import umap.umap_ as umap
        UMAP = umap.UMAP
    except ImportError:
        # Fallback - skip UMAP visualization if not available
        UMAP = None
import pickle
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.model import CycleVAE
from src.dataset import CombinedDataModule
from src.utils import mmd_rbf, ks_test_featurewise
from src.comprehensive_evaluator import ComprehensiveEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluator class for comprehensive model evaluation"""
    
    def __init__(self, config: dict, model_path: str):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model checkpoint
        """
        self.config = config
        self.output_dir = Path(config['paths']['output_dir'])
        self.eval_dir = self.output_dir / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Load data
        self.data_module = self.load_data()
        
        # Load feature specification
        self.feature_spec = self.load_feature_spec()
        
        logger.info("Evaluator initialized successfully")
    
    def _get_feature_columns(self, df: pd.DataFrame) -> tuple:
        """Return numeric and missing columns in feature_spec order, filtered by availability."""
        numeric_cols = [c for c in self.feature_spec.get('numeric_features', []) if c in df.columns]
        missing_cols = [c for c in self.feature_spec.get('missing_features', []) if c in df.columns]
        return numeric_cols, missing_cols

    def _split_mimic_for_cycle(self, df: pd.DataFrame) -> tuple:
        """Split a single MIMIC dataframe into two pseudo-domains like FeatureDataset(split_for_cycle).

        Uses a fixed seed and 50/50 split (domain 0 and domain 1) with shuffled labels.
        Returns (domain0_df, domain1_df) with reset indices.
        """
        n = len(df)
        labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
        rng = np.random.RandomState(42)
        rng.shuffle(labels)
        dom0_df = df[labels == 0].reset_index(drop=True)
        dom1_df = df[labels == 1].reset_index(drop=True)
        return dom0_df, dom1_df

    def load_model(self, model_path: str) -> CycleVAE:
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        
        # Load feature spec first
        feature_spec = self.load_feature_spec()
        
        # Create model instance
        model = CycleVAE(self.config, feature_spec)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        
        model.eval()
        logger.info("Model loaded successfully")
        
        return model
    
    def load_data(self) -> CombinedDataModule:
        """Load data module"""
        feature_spec = self.load_feature_spec()
        data_module = CombinedDataModule(self.config, feature_spec)
        data_module.setup('test')
        return data_module
    
    def load_feature_spec(self) -> dict:
        """Load feature specification"""
        spec_path = self.output_dir / "feature_spec.json"
        with open(spec_path, 'r') as f:
            feature_spec = json.load(f)
        return feature_spec
    
    def round_trip_evaluation(self):
        """Evaluate round-trip reconstruction quality"""
        logger.info("Performing round-trip evaluation...")
        
        # Get test data
        test_mimic_data = pd.read_csv(self.output_dir / "data" / "test_mimic_preprocessed.csv")
        if self.config.get('mimic_only', False):
            logger.info("MIMIC-ONLY MODE: Splitting MIMIC test data into two pseudo-domains for round-trip evaluation")
            eicu_df, mimic_df = self._split_mimic_for_cycle(test_mimic_data)
            logger.info(f"Split sizes - pseudo-eICU: {len(eicu_df)}, pseudo-MIMIC: {len(mimic_df)}")
            test_eicu_data = eicu_df
            test_mimic_data = mimic_df
        else:
            test_eicu_data = pd.read_csv(self.output_dir / "data" / "test_eicu_preprocessed.csv")
        
        # Convert to tensors (POC features format)
        mimic_numeric_cols, mimic_missing_cols = self._get_feature_columns(test_mimic_data)
        
        mimic_numeric = torch.FloatTensor(test_mimic_data[mimic_numeric_cols].values)
        mimic_missing = torch.FloatTensor(test_mimic_data[mimic_missing_cols].values)
        mimic_x = torch.cat([mimic_numeric, mimic_missing], dim=1)
        
        eicu_numeric_cols, eicu_missing_cols = self._get_feature_columns(test_eicu_data)
        
        eicu_numeric = torch.FloatTensor(test_eicu_data[eicu_numeric_cols].values)
        eicu_missing = torch.FloatTensor(test_eicu_data[eicu_missing_cols].values)
        eicu_x = torch.cat([eicu_numeric, eicu_missing], dim=1)
        
        # Perform round-trip translation
        with torch.no_grad():
            # eICU -> MIMIC -> eICU (DETERMINISTIC for evaluation)
            eicu_to_mimic = self.model.translate_eicu_to_mimic_deterministic(eicu_x)
            eicu_round_trip = self.model.translate_mimic_to_eicu_deterministic(eicu_to_mimic)
            
            # MIMIC -> eICU -> MIMIC (DETERMINISTIC for evaluation)
            mimic_to_eicu = self.model.translate_mimic_to_eicu_deterministic(mimic_x)
            mimic_round_trip = self.model.translate_eicu_to_mimic_deterministic(mimic_to_eicu)
        
        # Compute metrics
        eicu_mse = torch.mean((eicu_x - eicu_round_trip) ** 2, dim=0)
        eicu_mae = torch.mean(torch.abs(eicu_x - eicu_round_trip), dim=0)
        
        mimic_mse = torch.mean((mimic_x - mimic_round_trip) ** 2, dim=0)
        mimic_mae = torch.mean(torch.abs(mimic_x - mimic_round_trip), dim=0)
        
        # Create results DataFrame (updated for POC features)
        results = []
        col_idx = 0
        
        # Iterate through clinical features
        for feature in self.feature_spec['clinical_features']:
            # IMPORTANT: Order must match the CSV column order (min, max, mean, std)
            for suffix in ['_min', '_max', '_mean', '_std']:
                col_name = f"{feature}{suffix}"
                if col_name in mimic_numeric_cols:
                    results.append({
                        'feature_name': feature,
                        'feature_type': suffix.replace('_', ''),
                        'eicu_mse': eicu_mse[col_idx].item(),
                        'eicu_mae': eicu_mae[col_idx].item(),
                        'mimic_mse': mimic_mse[col_idx].item(),
                        'mimic_mae': mimic_mae[col_idx].item()
                    })
                    col_idx += 1
        
        # Add demographic features
        for feature in self.feature_spec['demographic_features']:
            if feature in mimic_numeric_cols:
                results.append({
                    'feature_name': feature,
                    'feature_type': 'demographic',
                    'eicu_mse': eicu_mse[col_idx].item(),
                    'eicu_mae': eicu_mae[col_idx].item(),
                    'mimic_mse': mimic_mse[col_idx].item(),
                    'mimic_mae': mimic_mae[col_idx].item()
                })
                col_idx += 1
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = self.eval_dir / "round_trip_metrics.csv"
        results_df.to_csv(results_path, index=False)
        
        # Summary statistics
        summary = {
            'eicu_mean_mse': eicu_mse.mean().item(),
            'eicu_mean_mae': eicu_mae.mean().item(),
            'mimic_mean_mse': mimic_mse.mean().item(),
            'mimic_mean_mae': mimic_mae.mean().item(),
            'overall_mean_mse': (eicu_mse.mean() + mimic_mse.mean()).item() / 2,
            'overall_mean_mae': (eicu_mae.mean() + mimic_mae.mean()).item() / 2
        }
        
        logger.info(f"Round-trip evaluation completed. Overall MSE: {summary['overall_mean_mse']:.4f}")
        
        return results_df, summary
    
    def distributional_evaluation(self):
        """Evaluate distributional similarity between translated and target data"""
        logger.info("Performing distributional evaluation...")
        
        # Get test data
        test_mimic_data = pd.read_csv(self.output_dir / "data" / "test_mimic_preprocessed.csv")
        if self.config.get('mimic_only', False):
            logger.info("MIMIC-ONLY MODE: Splitting MIMIC test data into two pseudo-domains for distributional evaluation")
            eicu_df, mimic_df = self._split_mimic_for_cycle(test_mimic_data)
            logger.info(f"Split sizes - pseudo-eICU: {len(eicu_df)}, pseudo-MIMIC: {len(mimic_df)}")
            test_eicu_data = eicu_df
            test_mimic_data = mimic_df
        else:
            test_eicu_data = pd.read_csv(self.output_dir / "data" / "test_eicu_preprocessed.csv")
        
        # Convert to tensors (POC features format)
        mimic_numeric_cols, mimic_missing_cols = self._get_feature_columns(test_mimic_data)
        
        mimic_numeric = torch.FloatTensor(test_mimic_data[mimic_numeric_cols].values)
        mimic_missing = torch.FloatTensor(test_mimic_data[mimic_missing_cols].values)
        mimic_x = torch.cat([mimic_numeric, mimic_missing], dim=1)
        
        eicu_numeric_cols, eicu_missing_cols = self._get_feature_columns(test_eicu_data)
        
        eicu_numeric = torch.FloatTensor(test_eicu_data[eicu_numeric_cols].values)
        eicu_missing = torch.FloatTensor(test_eicu_data[eicu_missing_cols].values)
        eicu_x = torch.cat([eicu_numeric, eicu_missing], dim=1)
        
        # Perform translation (DETERMINISTIC for evaluation)
        with torch.no_grad():
            eicu_to_mimic = self.model.translate_eicu_to_mimic_deterministic(eicu_x)
            mimic_to_eicu = self.model.translate_mimic_to_eicu_deterministic(mimic_x)
        
        # KS test for each feature (updated for POC features)
        ks_results = []
        col_idx = 0
        
        # Test clinical features
        for feature in self.feature_spec['clinical_features']:
            # IMPORTANT: Order must match the CSV column order (min, max, mean, std)
            for suffix in ['_min', '_max', '_mean', '_std']:
                col_name = f"{feature}{suffix}"
                if col_name in mimic_numeric_cols:
                    # KS test: translated eICU vs real MIMIC
                    ks_stat, p_value = stats.ks_2samp(
                        eicu_to_mimic[:, col_idx].numpy(),
                        mimic_x[:, col_idx].numpy()
                    )
                    
                    ks_results.append({
                        'feature_name': feature,
                        'feature_type': suffix.replace('_', ''),
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
                    col_idx += 1
        
        # Test demographic features
        for feature in self.feature_spec['demographic_features']:
            if feature in mimic_numeric_cols:
                # KS test: translated eICU vs real MIMIC
                ks_stat, p_value = stats.ks_2samp(
                    eicu_to_mimic[:, col_idx].numpy(),
                    mimic_x[:, col_idx].numpy()
                )
                
                ks_results.append({
                    'feature_name': feature,
                    'feature_type': 'demographic',
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
                col_idx += 1
        
        ks_df = pd.DataFrame(ks_results)
        
        # MMD test for overall distribution
        mmd_eicu_to_mimic = mmd_rbf(eicu_to_mimic.numpy(), mimic_x.numpy())
        mmd_mimic_to_eicu = mmd_rbf(mimic_to_eicu.numpy(), eicu_x.numpy())
        
        # Save results
        ks_path = self.eval_dir / "ks_by_feature.csv"
        ks_df.to_csv(ks_path, index=False)
        
        mmd_results = {
            'mmd_eicu_to_mimic': mmd_eicu_to_mimic,
            'mmd_mimic_to_eicu': mmd_mimic_to_eicu,
            'mean_ks_statistic': ks_df['ks_statistic'].mean(),
            'significant_features_pct': (ks_df['significant'].sum() / len(ks_df)) * 100
        }
        
        mmd_path = self.eval_dir / "mmd_results.json"
        with open(mmd_path, 'w') as f:
            json.dump(mmd_results, f, indent=2)
        
        logger.info(f"Distributional evaluation completed. MMD eICU->MIMIC: {mmd_eicu_to_mimic:.4f}")
        
        return ks_df, mmd_results
    
    def downstream_evaluation(self):
        """Evaluate downstream task performance"""
        logger.info("Performing downstream evaluation...")
        
        # Load data
        train_mimic_data = pd.read_csv(self.output_dir / "data" / "train_mimic_preprocessed.csv")
        test_mimic_data = pd.read_csv(self.output_dir / "data" / "test_mimic_preprocessed.csv")
        if self.config.get('mimic_only', False):
            logger.info("MIMIC-ONLY MODE: Splitting MIMIC test data into two pseudo-domains for downstream evaluation")
            eicu_df, mimic_df = self._split_mimic_for_cycle(test_mimic_data)
            logger.info(f"Split sizes - pseudo-eICU: {len(eicu_df)}, pseudo-MIMIC: {len(mimic_df)}")
            test_eicu_data = eicu_df
            test_mimic_data = mimic_df
        else:
            test_eicu_data = pd.read_csv(self.output_dir / "data" / "test_eicu_preprocessed.csv")
        
        # Create synthetic target variable (in-hospital mortality)
        # In real scenario, this would come from actual outcome data
        np.random.seed(42)
        train_mimic_data['mortality'] = np.random.choice([0, 1], size=len(train_mimic_data), p=[0.8, 0.2])
        test_mimic_data['mortality'] = np.random.choice([0, 1], size=len(test_mimic_data), p=[0.8, 0.2])
        test_eicu_data['mortality'] = np.random.choice([0, 1], size=len(test_eicu_data), p=[0.8, 0.2])
        
        # Prepare features (POC features format)
        # Use feature_spec order for features
        feature_cols = [c for c in self.feature_spec.get('numeric_features', []) if c in train_mimic_data.columns]
        
        # Train XGBoost on MIMIC training data
        xgb_model = xgb.XGBClassifier(**self.config['downstream']['xgboost_params'], random_state=42)
        xgb_model.fit(train_mimic_data[feature_cols], train_mimic_data['mortality'])
        
        # Evaluate on MIMIC test data (baseline)
        mimic_pred_proba = xgb_model.predict_proba(test_mimic_data[feature_cols])[:, 1]
        mimic_auc = roc_auc_score(test_mimic_data['mortality'], mimic_pred_proba)
        mimic_ap = average_precision_score(test_mimic_data['mortality'], mimic_pred_proba)
        mimic_brier = brier_score_loss(test_mimic_data['mortality'], mimic_pred_proba)
        
        # Evaluate on eICU test data (domain drop)
        eicu_pred_proba = xgb_model.predict_proba(test_eicu_data[feature_cols])[:, 1]
        eicu_auc = roc_auc_score(test_eicu_data['mortality'], eicu_pred_proba)
        eicu_ap = average_precision_score(test_eicu_data['mortality'], eicu_pred_proba)
        eicu_brier = brier_score_loss(test_eicu_data['mortality'], eicu_pred_proba)
        
        # Translate eICU data and evaluate
        eicu_numeric_cols, eicu_missing_cols = self._get_feature_columns(test_eicu_data)
        
        eicu_numeric = torch.FloatTensor(test_eicu_data[eicu_numeric_cols].values)
        eicu_missing = torch.FloatTensor(test_eicu_data[eicu_missing_cols].values)
        eicu_x = torch.cat([eicu_numeric, eicu_missing], dim=1)
        
        with torch.no_grad():
            eicu_translated = self.model.translate_eicu_to_mimic_deterministic(eicu_x)
        
        # Convert back to DataFrame format
        all_feature_cols = feature_cols + [col for col in train_mimic_data.columns if '_missing' in col]
        translated_df = pd.DataFrame(eicu_translated.numpy(), columns=all_feature_cols)
        
        # Use only the numeric features for downstream prediction
        translated_df_features = translated_df[feature_cols]
        
        # Evaluate on translated eICU data
        translated_pred_proba = xgb_model.predict_proba(translated_df_features)[:, 1]
        translated_auc = roc_auc_score(test_eicu_data['mortality'], translated_pred_proba)
        translated_ap = average_precision_score(test_eicu_data['mortality'], translated_pred_proba)
        translated_brier = brier_score_loss(test_eicu_data['mortality'], translated_pred_proba)
        
        # Compile results
        results = {
            'mimic_test': {
                'auc': mimic_auc,
                'average_precision': mimic_ap,
                'brier_score': mimic_brier
            },
            'eicu_test_raw': {
                'auc': eicu_auc,
                'average_precision': eicu_ap,
                'brier_score': eicu_brier
            },
            'eicu_test_translated': {
                'auc': translated_auc,
                'average_precision': translated_ap,
                'brier_score': translated_brier
            },
            'improvement': {
                'auc_improvement': translated_auc - eicu_auc,
                'ap_improvement': translated_ap - eicu_ap,
                'brier_improvement': eicu_brier - translated_brier
            }
        }
        
        # Save results
        results_path = self.eval_dir / "downstream_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save XGBoost model
        model_path = self.eval_dir / "downstream_xgboost.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        
        logger.info(f"Downstream evaluation completed. AUC improvement: {results['improvement']['auc_improvement']:.4f}")
        
        return results
    
    def create_visualizations(self):
        """Create visualization plots"""
        logger.info("Creating visualizations...")
        
        # Get test data
        test_mimic_data = pd.read_csv(self.output_dir / "data" / "test_mimic_preprocessed.csv")
        if self.config.get('mimic_only', False):
            logger.info("MIMIC-ONLY MODE: Splitting MIMIC test data into two pseudo-domains for visualizations")
            eicu_df, mimic_df = self._split_mimic_for_cycle(test_mimic_data)
            logger.info(f"Split sizes - pseudo-eICU: {len(eicu_df)}, pseudo-MIMIC: {len(mimic_df)}")
            test_eicu_data = eicu_df
            test_mimic_data = mimic_df
        else:
            test_eicu_data = pd.read_csv(self.output_dir / "data" / "test_eicu_preprocessed.csv")
        
        # Convert to tensors (POC features format)
        mimic_numeric_cols, mimic_missing_cols = self._get_feature_columns(test_mimic_data)
        
        mimic_numeric = torch.FloatTensor(test_mimic_data[mimic_numeric_cols].values)
        mimic_missing = torch.FloatTensor(test_mimic_data[mimic_missing_cols].values)
        mimic_x = torch.cat([mimic_numeric, mimic_missing], dim=1)
        
        eicu_numeric_cols, eicu_missing_cols = self._get_feature_columns(test_eicu_data)
        
        eicu_numeric = torch.FloatTensor(test_eicu_data[eicu_numeric_cols].values)
        eicu_missing = torch.FloatTensor(test_eicu_data[eicu_missing_cols].values)
        eicu_x = torch.cat([eicu_numeric, eicu_missing], dim=1)
        
        # Perform translation (DETERMINISTIC for evaluation)
        with torch.no_grad():
            eicu_to_mimic = self.model.translate_eicu_to_mimic_deterministic(eicu_x)
            mimic_to_eicu = self.model.translate_mimic_to_eicu_deterministic(mimic_x)
        
        # 1. Feature distribution comparison
        self.plot_feature_distributions(mimic_x, eicu_to_mimic, "eICU_to_MIMIC")
        
        # 2. UMAP visualization
        self.plot_umap_embeddings(mimic_x, eicu_x, eicu_to_mimic, mimic_to_eicu)
        
        # 3. Round-trip error analysis
        self.plot_round_trip_errors(mimic_x, eicu_x)
        
        logger.info("Visualizations created successfully")
    
    def plot_feature_distributions(self, original, translated, title_suffix):
        """Plot feature distribution comparisons"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Feature Distribution Comparison - {title_suffix}')
        
        # Select a few representative features (within bounds of our 42 total features)
        max_features = original.shape[1]  # 42 features total
        feature_indices = [0, min(10, max_features-1), min(20, max_features-1), min(30, max_features-1)]
        
        for i, idx in enumerate(feature_indices):
            ax = axes[i // 2, i % 2]
            
            ax.hist(original[:, idx].numpy(), alpha=0.7, label='Original', bins=30)
            ax.hist(translated[:, idx].numpy(), alpha=0.7, label='Translated', bins=30)
            ax.set_title(f'Feature {idx}')
            ax.legend()
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / f"feature_distributions_{title_suffix.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_umap_embeddings(self, mimic_x, eicu_x, eicu_to_mimic, mimic_to_eicu):
        """Plot UMAP embeddings"""
        if UMAP is None:
            logger.warning("UMAP not available, skipping UMAP visualization")
            return
            
        # Combine all data
        all_data = torch.cat([mimic_x, eicu_x, eicu_to_mimic, mimic_to_eicu], dim=0)
        
        # Create labels
        labels = ['MIMIC'] * len(mimic_x) + ['eICU'] * len(eicu_x) + ['eICU->MIMIC'] * len(eicu_to_mimic) + ['MIMIC->eICU'] * len(mimic_to_eicu)
        
        # Fit UMAP
        reducer = UMAP(random_state=42)
        embeddings = reducer.fit_transform(all_data.numpy())
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        colors = {'MIMIC': 'blue', 'eICU': 'red', 'eICU->MIMIC': 'green', 'MIMIC->eICU': 'orange'}
        
        for label in set(labels):
            mask = [l == label for l in labels]
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                       c=colors[label], label=label, alpha=0.6, s=20)
        
        plt.title('UMAP Embeddings of Original and Translated Data')
        plt.legend()
        plt.savefig(self.eval_dir / "umap_embeddings.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_round_trip_errors(self, mimic_x, eicu_x):
        """Plot round-trip reconstruction errors"""
        with torch.no_grad():
            # eICU round-trip (DETERMINISTIC for evaluation)
            eicu_to_mimic = self.model.translate_eicu_to_mimic_deterministic(eicu_x)
            eicu_round_trip = self.model.translate_mimic_to_eicu_deterministic(eicu_to_mimic)
            eicu_errors = torch.mean((eicu_x - eicu_round_trip) ** 2, dim=0)
            
            # MIMIC round-trip (DETERMINISTIC for evaluation)
            mimic_to_eicu = self.model.translate_mimic_to_eicu_deterministic(mimic_x)
            mimic_round_trip = self.model.translate_eicu_to_mimic_deterministic(mimic_to_eicu)
            mimic_errors = torch.mean((mimic_x - mimic_round_trip) ** 2, dim=0)
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(eicu_errors.numpy(), bins=50, alpha=0.7, label='eICU Round-trip Errors')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Frequency')
        plt.title('eICU Round-trip Reconstruction Errors')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(mimic_errors.numpy(), bins=50, alpha=0.7, label='MIMIC Round-trip Errors')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Frequency')
        plt.title('MIMIC Round-trip Reconstruction Errors')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / "round_trip_errors.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("Starting full evaluation pipeline...")
        
        # Run all evaluations
        round_trip_results, round_trip_summary = self.round_trip_evaluation()
        ks_results, mmd_results = self.distributional_evaluation()
        downstream_results = self.downstream_evaluation()
        
        # Create visualizations
        self.create_visualizations()
        
        # Create summary report
        summary = {
            'round_trip': round_trip_summary,
            'distributional': mmd_results,
            'downstream': downstream_results,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        summary_path = self.eval_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("=== EVALUATION SUMMARY ===")
        logger.info(f"Round-trip MSE: {summary['round_trip']['overall_mean_mse']:.4f}")
        logger.info(f"MMD eICU->MIMIC: {summary['distributional']['mmd_eicu_to_mimic']:.4f}")
        logger.info(f"Downstream AUC improvement: {summary['downstream']['improvement']['auc_improvement']:.4f}")
        
        logger.info("Full evaluation completed successfully!")
        
        return summary
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive patient-level and feature-level evaluation using actual test set"""
        logger.info("Starting comprehensive evaluation...")
        
        # FIXED: Load test data directly from CSV files (same as standard evaluation)
        logger.info("Loading actual test data from CSV files...")
        test_mimic_data = pd.read_csv(self.output_dir / "data" / "test_mimic_preprocessed.csv")
        if self.config.get('mimic_only', False):
            logger.info("MIMIC-ONLY MODE: Splitting MIMIC test data into two pseudo-domains for comprehensive evaluation")
            eicu_df, mimic_df = self._split_mimic_for_cycle(test_mimic_data)
            logger.info(f"Split sizes - pseudo-eICU: {len(eicu_df)}, pseudo-MIMIC: {len(mimic_df)}")
            test_eicu_data = eicu_df
            test_mimic_data = mimic_df
        else:
            test_eicu_data = pd.read_csv(self.output_dir / "data" / "test_eicu_preprocessed.csv")
        
        logger.info(f"Loaded test data - MIMIC: {len(test_mimic_data)} samples, eICU: {len(test_eicu_data)} samples")
        
        # Convert to tensors (feature_spec-ordered columns)
        mimic_numeric_cols, mimic_missing_cols = self._get_feature_columns(test_mimic_data)
        mimic_numeric = torch.FloatTensor(test_mimic_data[mimic_numeric_cols].values)
        mimic_missing = torch.FloatTensor(test_mimic_data[mimic_missing_cols].values)
        x_mimic_test = torch.cat([mimic_numeric, mimic_missing], dim=1)
        
        eicu_numeric_cols, eicu_missing_cols = self._get_feature_columns(test_eicu_data)
        eicu_numeric = torch.FloatTensor(test_eicu_data[eicu_numeric_cols].values)
        eicu_missing = torch.FloatTensor(test_eicu_data[eicu_missing_cols].values)
        x_eicu_test = torch.cat([eicu_numeric, eicu_missing], dim=1)
        
        logger.info(f"Converted to tensors - MIMIC: {x_mimic_test.size(0)} samples, eICU: {x_eicu_test.size(0)} samples")
        logger.info(f"Running comprehensive evaluation with {x_mimic_test.size(0)} MIMIC and {x_eicu_test.size(0)} eICU samples")
        
        # Create comprehensive evaluator
        logger.info("Creating comprehensive evaluator...")
        comprehensive_evaluator = ComprehensiveEvaluator(
            self.model, self.feature_spec, str(self.output_dir)
        )
        
        # Run comprehensive evaluation with error handling
        try:
            logger.info("Starting comprehensive evaluation...")
            comprehensive_results = comprehensive_evaluator.evaluate_translation_quality(
                x_eicu_test, x_mimic_test
            )
            logger.info("Comprehensive evaluation completed successfully!")
            return comprehensive_results
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            logger.error("This might be due to memory issues or model problems")
            logger.info("Continuing with standard evaluation only...")
            return None
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report from results."""
        logger.info("Generating comprehensive evaluation report...")
        
        # Check if comprehensive evaluation was run
        comprehensive_dir = self.output_dir / "comprehensive_evaluation"
        if not comprehensive_dir.exists():
            logger.warning("No comprehensive evaluation results found. Run with --comprehensive flag first.")
            return None
        
        # Load results
        results = {}
        
        # Load comprehensive results
        comprehensive_path = comprehensive_dir / "comprehensive_results.json"
        if comprehensive_path.exists():
            try:
                with open(comprehensive_path, 'r') as f:
                    results['comprehensive'] = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted comprehensive results JSON file: {e}")
                logger.info("Deleting corrupted file - it will be regenerated on next evaluation")
                comprehensive_path.unlink()  # Delete corrupted file
                results['comprehensive'] = None
        
        # Load CSV files
        csv_files = [
            'correlation_metrics.csv',
            'ks_analysis.csv',
            'summary_statistics.csv'
        ]
        
        for csv_file in csv_files:
            csv_path = comprehensive_dir / "data" / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                results[csv_file.replace('.csv', '')] = df
        
        # Generate report sections
        report_sections = []
        
        # 1. Executive Summary
        report_sections.append(self._generate_executive_summary(results))
        
        # 2. Feature Quality Analysis
        report_sections.append(self._generate_feature_quality_analysis(results))
        
        # 3. Per-Feature IQR Analysis
        report_sections.append(self._generate_per_feature_iqr_analysis(results))
        
        # 4. Distribution Analysis
        report_sections.append(self._generate_distribution_analysis(results))
        
        # 5. Missingness Analysis
        report_sections.append(self._generate_missingness_analysis(results))
        
        # 6. Demographic Analysis
        report_sections.append(self._generate_demographic_analysis(results))
        
        # 7. Recommendations
        report_sections.append(self._generate_recommendations(results))
        
        # Combine sections
        full_report = "\n\n".join(report_sections)
        
        # Save report
        report_path = self.eval_dir / "comprehensive_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(full_report)
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        
        return str(report_path)
    
    def _generate_executive_summary(self, results: dict) -> str:
        """UPDATED: Generate executive summary with simplified model metrics."""
        summary = "# Comprehensive Evaluation Report (Simplified Model)\n\n"
        summary += "*Generated for simplified CycleVAE with 3 losses: reconstruction, cycle, conditional Wasserstein*\n\n"
        summary += "*Note: Missing flags, Age, and Gender are excluded from evaluation (input-only)*\n\n"
        summary += "## Executive Summary\n\n"
        
        # NEW: Per-feature percentage error metrics
        comprehensive = results.get('comprehensive', {})
        if comprehensive and 'eicu_reconstruction_errors' in comprehensive:
            summary += "### 📊 Reconstruction Quality (A→A')\n\n"
            summary += "*Note: Data is normalized - MAE in standard deviation units, use IQR metrics*\n\n"
            
            eicu_err = comprehensive['eicu_reconstruction_errors']
            mimic_err = comprehensive['mimic_reconstruction_errors']
            
            if eicu_err and 'pct_within_iqr' in eicu_err:
                summary += "**eICU Reconstruction:**\n"
                summary += f"- MAE: {np.mean(eicu_err.get('mae', [])):.4f} (std dev units)\n"
                summary += f"- % within 0.5 IQR: {np.mean(eicu_err['pct_within_iqr'].get('within_0.5_iqr', [])):.1f}%\n"
                summary += f"- % within 1.0 IQR: {np.mean(eicu_err['pct_within_iqr'].get('within_1.0_iqr', [])):.1f}%\n\n"
            
            if mimic_err and 'pct_within_iqr' in mimic_err:
                summary += "**MIMIC Reconstruction:**\n"
                summary += f"- MAE: {np.mean(mimic_err.get('mae', [])):.4f} (std dev units)\n"
                summary += f"- % within 0.5 IQR: {np.mean(mimic_err['pct_within_iqr'].get('within_0.5_iqr', [])):.1f}%\n"
                summary += f"- % within 1.0 IQR: {np.mean(mimic_err['pct_within_iqr'].get('within_1.0_iqr', [])):.1f}%\n\n"
        
        if comprehensive and 'eicu_cycle_errors' in comprehensive:
            summary += "### 🔄 Cycle Consistency (A→B'→A')\n\n"
            summary += "*Note: Data is normalized - use IQR metrics for meaningful percentages*\n\n"
            
            eicu_cyc = comprehensive['eicu_cycle_errors']
            mimic_cyc = comprehensive['mimic_cycle_errors']
            
            if eicu_cyc and 'pct_within_iqr' in eicu_cyc:
                summary += "**eICU Cycle:**\n"
                summary += f"- MAE: {np.mean(eicu_cyc.get('mae', [])):.4f} (std dev units)\n"
                summary += f"- % within 0.5 IQR: {np.mean(eicu_cyc['pct_within_iqr'].get('within_0.5_iqr', [])):.1f}%\n"
                summary += f"- % within 1.0 IQR: {np.mean(eicu_cyc['pct_within_iqr'].get('within_1.0_iqr', [])):.1f}%\n\n"
            
            if mimic_cyc and 'pct_within_iqr' in mimic_cyc:
                summary += "**MIMIC Cycle:**\n"
                summary += f"- MAE: {np.mean(mimic_cyc.get('mae', [])):.4f} (std dev units)\n"
                summary += f"- % within 0.5 IQR: {np.mean(mimic_cyc['pct_within_iqr'].get('within_0.5_iqr', [])):.1f}%\n"
                summary += f"- % within 1.0 IQR: {np.mean(mimic_cyc['pct_within_iqr'].get('within_1.0_iqr', [])):.1f}%\n\n"
        
        # NEW: Latent space distance
        if comprehensive and 'latent_distance_eicu_vs_mimic' in comprehensive:
            summary += "### 🧠 Latent Space Analysis\n\n"
            
            latent_orig = comprehensive['latent_distance_eicu_vs_mimic']
            latent_trans = comprehensive.get('latent_distance_translated_vs_real', {})
            
            if latent_orig:
                summary += "**Original Domains (eICU vs MIMIC):**\n"
                summary += f"- Euclidean Distance: {latent_orig.get('mean_euclidean_distance', 0):.4f}\n"
                summary += f"- Cosine Similarity: {latent_orig.get('cosine_similarity', 0):.4f}\n"
                summary += f"- KL Divergence: {latent_orig.get('kl_divergence', 0):.4f}\n\n"
            
            if latent_trans:
                summary += "**After Translation (eICU→MIMIC vs real MIMIC):**\n"
                summary += f"- Euclidean Distance: {latent_trans.get('mean_euclidean_distance', 0):.4f}\n"
                summary += f"- Cosine Similarity: {latent_trans.get('cosine_similarity', 0):.4f}\n"
                summary += f"- KL Divergence: {latent_trans.get('kl_divergence', 0):.4f}\n\n"
        
        # NEW: Distribution distance
        if comprehensive and 'distribution_distance_eicu_to_mimic' in comprehensive:
            summary += "### 📈 Distribution Matching\n\n"
            
            dist_e2m = comprehensive['distribution_distance_eicu_to_mimic']
            
            if dist_e2m and 'wasserstein_distances' in dist_e2m:
                wass_dists = dist_e2m['wasserstein_distances']
                ks_stats = dist_e2m.get('ks_statistics', [])
                
                summary += f"- Mean Wasserstein Distance: {np.mean(wass_dists):.4f}\n"
                summary += f"- Mean KS Statistic: {np.mean(ks_stats):.4f}\n\n"
        
        summary += "---\n\n"
        summary += "### Legacy Metrics (Clinical Features Only)\n\n"
        summary += "*Note: R² and Correlation below are computed on **roundtrip/cycle** data (A→B'→A'), not on direct reconstruction*\n\n"
        
        if 'correlation_metrics' in results:
            df = results['correlation_metrics']
            
            # Separate metrics
            r2_good_eicu = (df['eicu_r2'] > 0.5).sum()
            r2_good_mimic = (df['mimic_r2'] > 0.5).sum()
            corr_good_eicu = (df['eicu_correlation'] > 0.7).sum()
            corr_good_mimic = (df['mimic_correlation'] > 0.7).sum()
            both_good_eicu = df['eicu_good_quality'].sum()
            both_good_mimic = df['mimic_good_quality'].sum()
            total_features = len(df)
            
            mean_r2_eicu = df['eicu_r2'].mean()
            mean_r2_mimic = df['mimic_r2'].mean()
            mean_corr_eicu = df['eicu_correlation'].mean()
            mean_corr_mimic = df['mimic_correlation'].mean()
            
            summary += f"**Translation Quality Metrics:**\n"
            summary += f"- **Clinical Features Evaluated**: {total_features}\n\n"
            summary += f"**R² (Variance Explained) - Target: > 0.5:**\n"
            summary += f"- **eICU**: {r2_good_eicu}/{total_features} features ({r2_good_eicu/total_features*100:.1f}%), Mean R²: {mean_r2_eicu:.3f}\n"
            summary += f"- **MIMIC**: {r2_good_mimic}/{total_features} features ({r2_good_mimic/total_features*100:.1f}%), Mean R²: {mean_r2_mimic:.3f}\n\n"
            summary += f"**Correlation (Linear Relationship) - Target: > 0.7:**\n"
            summary += f"- **eICU**: {corr_good_eicu}/{total_features} features ({corr_good_eicu/total_features*100:.1f}%), Mean corr: {mean_corr_eicu:.3f}\n"
            summary += f"- **MIMIC**: {corr_good_mimic}/{total_features} features ({corr_good_mimic/total_features*100:.1f}%), Mean corr: {mean_corr_mimic:.3f}\n\n"
            summary += f"**Combined (R² > 0.5 AND correlation > 0.7):**\n"
            summary += f"- **eICU**: {both_good_eicu}/{total_features} features ({both_good_eicu/total_features*100:.1f}%)\n"
            summary += f"- **MIMIC**: {both_good_mimic}/{total_features} features ({both_good_mimic/total_features*100:.1f}%)\n\n"
        
        if 'ks_analysis' in results:
            df = results['ks_analysis']
            total_features = len(df)
            
            # Use KS-based thresholds only (p-values uninformative with large N)
            excellent_e2m = df['eicu_to_mimic_excellent'].sum()
            good_e2m = df['eicu_to_mimic_good'].sum()
            acceptable_e2m = df['eicu_to_mimic_acceptable'].sum()
            excellent_m2e = df['mimic_to_eicu_excellent'].sum()
            good_m2e = df['mimic_to_eicu_good'].sum()
            acceptable_m2e = df['mimic_to_eicu_acceptable'].sum()
            
            mean_ks_e2m = df['eicu_to_mimic_ks'].mean()
            mean_ks_m2e = df['mimic_to_eicu_ks'].mean()
            
            summary += f"**Distribution Matching (KS statistic - effect size):**\n"
            summary += f"*Note: p-values not used (uninformative with large N={len(df)})*\n\n"
            summary += f"**eICU→MIMIC Translation:**\n"
            summary += f"- Excellent (KS<0.1): {excellent_e2m}/{total_features} ({excellent_e2m/total_features*100:.1f}%)\n"
            summary += f"- Good (KS<0.2): {good_e2m}/{total_features} ({good_e2m/total_features*100:.1f}%)\n"
            summary += f"- Acceptable (KS<0.3): {acceptable_e2m}/{total_features} ({acceptable_e2m/total_features*100:.1f}%)\n"
            summary += f"- Mean KS: {mean_ks_e2m:.3f}\n\n"
            summary += f"**MIMIC→eICU Translation:**\n"
            summary += f"- Excellent (KS<0.1): {excellent_m2e}/{total_features} ({excellent_m2e/total_features*100:.1f}%)\n"
            summary += f"- Good (KS<0.2): {good_m2e}/{total_features} ({good_m2e/total_features*100:.1f}%)\n"
            summary += f"- Acceptable (KS<0.3): {acceptable_m2e}/{total_features} ({acceptable_m2e/total_features*100:.1f}%)\n"
            summary += f"- Mean KS: {mean_ks_m2e:.3f}\n\n"
        
        # Overall assessment
        if 'correlation_metrics' in results and 'ks_analysis' in results:
            # Use "good" threshold (KS<0.2) for overall quality
            overall_quality = (good_e2m + good_m2e) / (2 * total_features)
            if overall_quality > 0.8:
                assessment = "**EXCELLENT** - Model shows strong translation quality"
            elif overall_quality > 0.6:
                assessment = "**GOOD** - Model shows acceptable translation quality"
            elif overall_quality > 0.4:
                assessment = "**FAIR** - Model shows moderate translation quality with room for improvement"
            else:
                assessment = "**POOR** - Model shows weak translation quality requiring significant improvements"
            
            summary += f"### Overall Assessment\n\n{assessment}\n\n"
        
        return summary
    
    def _generate_per_feature_iqr_analysis(self, results: dict) -> str:
        """Generate per-feature IQR analysis section."""
        section = "## Per-Feature IQR Analysis\n\n"
        section += "*Detailed breakdown of IQR-normalized errors for each clinical feature*\n\n"
        
        comprehensive = results.get('comprehensive', {})
        
        # Get feature names from comprehensive evaluator
        feature_names = None
        if 'correlation_metrics' in results:
            feature_names = results['correlation_metrics']['feature_name'].tolist()
        
        # Reconstruction per-feature
        if comprehensive and 'eicu_reconstruction_errors' in comprehensive:
            eicu_err = comprehensive['eicu_reconstruction_errors']
            mimic_err = comprehensive['mimic_reconstruction_errors']
            
            if eicu_err and 'pct_within_iqr' in eicu_err and feature_names:
                section += "### Reconstruction (A→A')\n\n"
                
                # Create per-feature table
                section += "| Feature | eICU % in 0.5 IQR | eICU % in 1.0 IQR | MIMIC % in 0.5 IQR | MIMIC % in 1.0 IQR | eICU % < 0.05 abs | MIMIC % < 0.05 abs |\n"
                section += "|---------|-------------------|-------------------|--------------------|--------------------|-----------------|------------------|\n"
                
                for i, feature in enumerate(feature_names):
                    if i < len(eicu_err['pct_within_iqr']['within_0.5_iqr']):
                        eicu_05 = eicu_err['pct_within_iqr']['within_0.5_iqr'][i]
                        eicu_10 = eicu_err['pct_within_iqr']['within_1.0_iqr'][i]
                        mimic_05 = mimic_err['pct_within_iqr']['within_0.5_iqr'][i]
                        mimic_10 = mimic_err['pct_within_iqr']['within_1.0_iqr'][i]
                        # Absolute tolerance display (use 0.05 if present)
                        eicu_abs = None
                        mimic_abs = None
                        if 'pct_within_abs' in eicu_err and 'within_0.05_abs' in eicu_err['pct_within_abs']:
                            eicu_abs = eicu_err['pct_within_abs']['within_0.05_abs'][i]
                        if 'pct_within_abs' in mimic_err and 'within_0.05_abs' in mimic_err['pct_within_abs']:
                            mimic_abs = mimic_err['pct_within_abs']['within_0.05_abs'][i]
                        eicu_abs_str = f"{eicu_abs:.1f}%" if eicu_abs is not None else "-"
                        mimic_abs_str = f"{mimic_abs:.1f}%" if mimic_abs is not None else "-"
                        
                        section += f"| {feature} | {eicu_05:.1f}% | {eicu_10:.1f}% | {mimic_05:.1f}% | {mimic_10:.1f}% | {eicu_abs_str} | {mimic_abs_str} |\n"
                
                section += "\n"
                
                # Identify best and worst performers
                eicu_05_arr = eicu_err['pct_within_iqr']['within_0.5_iqr']
                mimic_05_arr = mimic_err['pct_within_iqr']['within_0.5_iqr']
                
                # Best performers (highest % within 0.5 IQR)
                eicu_sorted = sorted(enumerate(eicu_05_arr), key=lambda x: x[1], reverse=True)
                mimic_sorted = sorted(enumerate(mimic_05_arr), key=lambda x: x[1], reverse=True)
                
                section += "**Best Performing Features (Reconstruction, % within 0.5 IQR):**\n\n"
                section += "eICU:\n"
                for idx, pct in eicu_sorted[:5]:
                    if idx < len(feature_names):
                        section += f"- {feature_names[idx]}: {pct:.1f}%\n"
                
                section += "\nMIMIC:\n"
                for idx, pct in mimic_sorted[:5]:
                    if idx < len(feature_names):
                        section += f"- {feature_names[idx]}: {pct:.1f}%\n"
                
                section += "\n**Worst Performing Features (Reconstruction, % within 0.5 IQR):**\n\n"
                section += "eICU:\n"
                for idx, pct in list(reversed(eicu_sorted))[:5]:
                    if idx < len(feature_names):
                        section += f"- {feature_names[idx]}: {pct:.1f}%\n"
                
                section += "\nMIMIC:\n"
                for idx, pct in list(reversed(mimic_sorted))[:5]:
                    if idx < len(feature_names):
                        section += f"- {feature_names[idx]}: {pct:.1f}%\n"
                
                section += "\n"
        
        # Cycle per-feature
        if comprehensive and 'eicu_cycle_errors' in comprehensive:
            eicu_cyc = comprehensive['eicu_cycle_errors']
            mimic_cyc = comprehensive['mimic_cycle_errors']
            
            if eicu_cyc and 'pct_within_iqr' in eicu_cyc and feature_names:
                section += "### Cycle Consistency (A→B'→A')\n\n"
                
                # Create per-feature table
                section += "| Feature | eICU % in 0.5 IQR | eICU % in 1.0 IQR | MIMIC % in 0.5 IQR | MIMIC % in 1.0 IQR | eICU % < 0.05 abs | MIMIC % < 0.05 abs |\n"
                section += "|---------|-------------------|-------------------|--------------------|--------------------|-----------------|------------------|\n"
                
                for i, feature in enumerate(feature_names):
                    if i < len(eicu_cyc['pct_within_iqr']['within_0.5_iqr']):
                        eicu_05 = eicu_cyc['pct_within_iqr']['within_0.5_iqr'][i]
                        eicu_10 = eicu_cyc['pct_within_iqr']['within_1.0_iqr'][i]
                        mimic_05 = mimic_cyc['pct_within_iqr']['within_0.5_iqr'][i]
                        mimic_10 = mimic_cyc['pct_within_iqr']['within_1.0_iqr'][i]
                        # Absolute tolerance display (use 0.05 if present)
                        eicu_abs = None
                        mimic_abs = None
                        if 'pct_within_abs' in eicu_cyc and 'within_0.05_abs' in eicu_cyc['pct_within_abs']:
                            eicu_abs = eicu_cyc['pct_within_abs']['within_0.05_abs'][i]
                        if 'pct_within_abs' in mimic_cyc and 'within_0.05_abs' in mimic_cyc['pct_within_abs']:
                            mimic_abs = mimic_cyc['pct_within_abs']['within_0.05_abs'][i]
                        eicu_abs_str = f"{eicu_abs:.1f}%" if eicu_abs is not None else "-"
                        mimic_abs_str = f"{mimic_abs:.1f}%" if mimic_abs is not None else "-"
                        
                        section += f"| {feature} | {eicu_05:.1f}% | {eicu_10:.1f}% | {mimic_05:.1f}% | {mimic_10:.1f}% | {eicu_abs_str} | {mimic_abs_str} |\n"
                
                section += "\n"
                
                # Identify best and worst performers for cycle
                eicu_05_arr = eicu_cyc['pct_within_iqr']['within_0.5_iqr']
                mimic_05_arr = mimic_cyc['pct_within_iqr']['within_0.5_iqr']
                
                eicu_sorted = sorted(enumerate(eicu_05_arr), key=lambda x: x[1], reverse=True)
                mimic_sorted = sorted(enumerate(mimic_05_arr), key=lambda x: x[1], reverse=True)
                
                section += "**Best Performing Features (Cycle, % within 0.5 IQR):**\n\n"
                section += "eICU:\n"
                for idx, pct in eicu_sorted[:5]:
                    if idx < len(feature_names):
                        section += f"- {feature_names[idx]}: {pct:.1f}%\n"
                
                section += "\nMIMIC:\n"
                for idx, pct in mimic_sorted[:5]:
                    if idx < len(feature_names):
                        section += f"- {feature_names[idx]}: {pct:.1f}%\n"
                
                section += "\n**Worst Performing Features (Cycle, % within 0.5 IQR):**\n\n"
                section += "eICU:\n"
                for idx, pct in list(reversed(eicu_sorted))[:5]:
                    if idx < len(feature_names):
                        section += f"- {feature_names[idx]}: {pct:.1f}%\n"
                
                section += "\nMIMIC:\n"
                for idx, pct in list(reversed(mimic_sorted))[:5]:
                    if idx < len(feature_names):
                        section += f"- {feature_names[idx]}: {pct:.1f}%\n"
                
                section += "\n"
        
        return section
    
    def _generate_feature_quality_analysis(self, results: dict) -> str:
        """Generate feature quality analysis."""
        if 'correlation_metrics' not in results:
            return "## Feature Quality Analysis\n\n*No correlation metrics available.*\n"
        
        df = results['correlation_metrics']
        
        analysis = "## Feature Quality Analysis\n\n"
        
        # Best performing features
        best_eicu = df.nlargest(5, 'eicu_r2')[['feature_name', 'eicu_r2', 'eicu_correlation']]
        best_mimic = df.nlargest(5, 'mimic_r2')[['feature_name', 'mimic_r2', 'mimic_correlation']]
        
        analysis += "### Best Performing Features (eICU Round-trip)\n\n"
        analysis += "| Feature | R² | Correlation |\n"
        analysis += "|---------|----|-------------|\n"
        for _, row in best_eicu.iterrows():
            analysis += f"| {row['feature_name']} | {row['eicu_r2']:.3f} | {row['eicu_correlation']:.3f} |\n"
        
        analysis += "\n### Best Performing Features (MIMIC Round-trip)\n\n"
        analysis += "| Feature | R² | Correlation |\n"
        analysis += "|---------|----|-------------|\n"
        for _, row in best_mimic.iterrows():
            analysis += f"| {row['feature_name']} | {row['mimic_r2']:.3f} | {row['mimic_correlation']:.3f} |\n"
        
        # Worst performing features
        worst_eicu = df.nsmallest(5, 'eicu_r2')[['feature_name', 'eicu_r2', 'eicu_correlation']]
        worst_mimic = df.nsmallest(5, 'mimic_r2')[['feature_name', 'mimic_r2', 'mimic_correlation']]
        
        analysis += "\n### Worst Performing Features (eICU Round-trip)\n\n"
        analysis += "| Feature | R² | Correlation |\n"
        analysis += "|---------|----|-------------|\n"
        for _, row in worst_eicu.iterrows():
            analysis += f"| {row['feature_name']} | {row['eicu_r2']:.3f} | {row['eicu_correlation']:.3f} |\n"
        
        analysis += "\n### Worst Performing Features (MIMIC Round-trip)\n\n"
        analysis += "| Feature | R² | Correlation |\n"
        analysis += "|---------|----|-------------|\n"
        for _, row in worst_mimic.iterrows():
            analysis += f"| {row['feature_name']} | {row['mimic_r2']:.3f} | {row['mimic_correlation']:.3f} |\n"
        
        return analysis
    
    def _generate_distribution_analysis(self, results: dict) -> str:
        """Generate distribution analysis."""
        if 'ks_analysis' not in results:
            return "## Distribution Analysis\n\n*No KS analysis available.*\n"
        
        df = results['ks_analysis']
        
        analysis = "## Distribution Analysis\n\n"
        analysis += "*KS statistic thresholds: <0.1=excellent, <0.2=good, <0.3=acceptable*\n\n"
        
        # Features with good distribution matching (KS<0.2)
        good_eicu = df[df['eicu_to_mimic_good']]['feature_name'].tolist()
        good_mimic = df[df['mimic_to_eicu_good']]['feature_name'].tolist()
        excellent_eicu = df[df['eicu_to_mimic_excellent']]['feature_name'].tolist()
        excellent_mimic = df[df['mimic_to_eicu_excellent']]['feature_name'].tolist()
        
        analysis += f"### Features with Good Distribution Matching (KS < 0.2)\n\n"
        analysis += f"- **eICU→MIMIC**: {len(good_eicu)} features\n"
        if good_eicu:
            analysis += f"  - {', '.join(good_eicu[:10])}"
            if len(good_eicu) > 10:
                analysis += f" (and {len(good_eicu) - 10} more)"
            analysis += "\n"
        
        analysis += f"- **MIMIC→eICU**: {len(good_mimic)} features\n"
        if good_mimic:
            analysis += f"  - {', '.join(good_mimic[:10])}"
            if len(good_mimic) > 10:
                analysis += f" (and {len(good_mimic) - 10} more)"
            analysis += "\n"
        
        # Features with poor distribution matching (KS>=0.3)
        poor_eicu = df[~df['eicu_to_mimic_acceptable']]['feature_name'].tolist()
        poor_mimic = df[~df['mimic_to_eicu_acceptable']]['feature_name'].tolist()
        
        analysis += f"\n### Features with Poor Distribution Matching (KS ≥ 0.3)\n\n"
        analysis += f"- **eICU→MIMIC**: {len(poor_eicu)} features\n"
        if poor_eicu:
            analysis += f"  - {', '.join(poor_eicu[:10])}"
            if len(poor_eicu) > 10:
                analysis += f" (and {len(poor_eicu) - 10} more)"
            analysis += "\n"
        
        analysis += f"- **MIMIC→eICU**: {len(poor_mimic)} features\n"
        if poor_mimic:
            analysis += f"  - {', '.join(poor_mimic[:10])}"
            if len(poor_mimic) > 10:
                analysis += f" (and {len(poor_mimic) - 10} more)"
            analysis += "\n"
        
        return analysis
    
    def _generate_missingness_analysis(self, results: dict) -> str:
        """Generate missingness analysis."""
        if not results or 'comprehensive' not in results or not results['comprehensive'] or 'missingness_analysis' not in results['comprehensive']:
            return "## Missingness Analysis\n\n*No missingness analysis available.*\n"
        
        missingness = results['comprehensive']['missingness_analysis']
        
        analysis = "## Missingness Analysis\n\n"
        analysis += "### Performance by Feature Count Buckets\n\n"
        analysis += "| Bucket | eICU Samples | MIMIC Samples | eICU MSE | MIMIC MSE |\n"
        analysis += "|--------|--------------|---------------|----------|----------|\n"
        
        for bucket, data in missingness.get('bucket_analysis', {}).items():
            analysis += f"| {bucket} | {data['eicu_samples']} | {data['mimic_samples']} | {data['eicu_mse']:.4f} | {data['mimic_mse']:.4f} |\n"
        
        return analysis
    
    def _generate_demographic_analysis(self, results: dict) -> str:
        """Generate demographic analysis."""
        if not results or 'comprehensive' not in results or not results['comprehensive'] or 'demographic_analysis' not in results['comprehensive']:
            return "## Demographic Analysis\n\n*No demographic analysis available.*\n"
        
        demo = results['comprehensive']['demographic_analysis']
        
        analysis = "## Demographic Analysis\n\n"
        
        if 'age_analysis' in demo:
            analysis += "### Age-based Performance\n\n"
            analysis += "| Age Group | Samples | MSE | Mean Age |\n"
            analysis += "|-----------|---------|-----|----------|\n"
            
            for group, data in demo['age_analysis'].items():
                analysis += f"| {group} | {data['samples']} | {data['mse']:.4f} | {data['age_mean']:.1f} |\n"
        
        if 'gender_analysis' in demo:
            analysis += "\n### Gender-based Performance\n\n"
            analysis += "| Gender | Samples | MSE |\n"
            analysis += "|--------|---------|-----|\n"
            
            for gender, data in demo['gender_analysis'].items():
                analysis += f"| {gender} | {data['samples']} | {data['mse']:.4f} |\n"
        
        return analysis
    
    def _generate_recommendations(self, results: dict) -> str:
        """Generate recommendations based on results."""
        recommendations = "## Recommendations\n\n"
        
        if 'correlation_metrics' in results:
            df = results['correlation_metrics']
            poor_eicu = df[~df['eicu_good_quality']]['feature_name'].tolist()
            poor_mimic = df[~df['mimic_good_quality']]['feature_name'].tolist()
            
            if poor_eicu or poor_mimic:
                recommendations += "### Feature-specific Improvements\n\n"
                recommendations += "The following features show poor round-trip consistency and may need special attention:\n\n"
                
                if poor_eicu:
                    recommendations += f"- **eICU round-trip issues**: {', '.join(poor_eicu[:5])}"
                    if len(poor_eicu) > 5:
                        recommendations += f" (and {len(poor_eicu) - 5} more)"
                    recommendations += "\n"
                
                if poor_mimic:
                    recommendations += f"- **MIMIC round-trip issues**: {', '.join(poor_mimic[:5])}"
                    if len(poor_mimic) > 5:
                        recommendations += f" (and {len(poor_mimic) - 5} more)"
                    recommendations += "\n"
                
                recommendations += "\n**Suggested actions**:\n"
                recommendations += "- Review feature preprocessing and normalization\n"
                recommendations += "- Consider feature-specific loss weighting\n"
                recommendations += "- Investigate domain-specific feature distributions\n\n"
        
        if 'ks_analysis' in results:
            df = results['ks_analysis']
            # Poor = not acceptable (KS >= 0.3) in either direction
            poor_dist = df[~df['eicu_to_mimic_acceptable'] & ~df['mimic_to_eicu_acceptable']]['feature_name'].tolist()
            
            if poor_dist:
                recommendations += "### Distribution Matching Improvements\n\n"
                recommendations += f"Features with poor distribution matching: {', '.join(poor_dist[:5])}"
                if len(poor_dist) > 5:
                    recommendations += f" (and {len(poor_dist) - 5} more)"
                recommendations += "\n\n"
                recommendations += "**Suggested actions**:\n"
                recommendations += "- Increase MMD loss weight for problematic features\n"
                recommendations += "- Consider per-feature MMD loss\n"
                recommendations += "- Review feature scaling and normalization\n\n"
        
        recommendations += "### General Recommendations\n\n"
        recommendations += "1. **Monitor training stability**: Ensure loss components are balanced\n"
        recommendations += "2. **Validate on held-out data**: Test translation quality on unseen patients\n"
        recommendations += "3. **Consider ensemble methods**: Combine multiple models for better robustness\n"
        recommendations += "4. **Domain adaptation**: Fine-tune on target domain data if available\n"
        recommendations += "5. **Clinical validation**: Validate translations with domain experts\n"
        
        return recommendations

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Cycle-VAE model')
    parser.add_argument('--config', type=str, default='conf/config.yml', 
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for evaluation results')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive patient-level evaluation')
    parser.add_argument('--mimic-only', action='store_true',
                       help='Evaluate only on MIMIC data (for same-domain validation)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(config_path)
    
    # Override output directory if specified
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    
    # Set MIMIC-only mode if specified
    config['mimic_only'] = args.mimic_only
    
    if args.mimic_only:
        logger.info("MIMIC-ONLY MODE: Evaluation will use only MIMIC data")
    
    # Create evaluator
    evaluator = Evaluator(config, args.model)
    
    # Run standard evaluation
    summary = evaluator.run_full_evaluation()
    
    # Run comprehensive evaluation if requested
    if args.comprehensive:
        comprehensive_results = evaluator.run_comprehensive_evaluation()
        if comprehensive_results:
            logger.info("Comprehensive evaluation completed successfully!")
            # Automatically generate report
            report_path = evaluator.generate_comprehensive_report()
            if report_path:
                logger.info(f"Comprehensive report generated: {report_path}")
        else:
            logger.warning("Comprehensive evaluation failed or was skipped")
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
