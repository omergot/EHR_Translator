#!/usr/bin/env python3
"""
Evaluation Script for Cycle-VAE
Performs comprehensive evaluation including round-trip reconstruction, 
distributional tests, and downstream evaluation.
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
        test_eicu_data = pd.read_csv(self.output_dir / "data" / "test_eicu_preprocessed.csv")
        
        # Convert to tensors (POC features format)
        mimic_numeric_cols = [col for col in test_mimic_data.columns 
                             if ('_mean' in col or '_min' in col or '_max' in col or '_std' in col) 
                             or col in ['Age', 'Gender']]
        mimic_missing_cols = [col for col in test_mimic_data.columns if '_missing' in col]
        
        mimic_numeric = torch.FloatTensor(test_mimic_data[mimic_numeric_cols].values)
        mimic_missing = torch.FloatTensor(test_mimic_data[mimic_missing_cols].values)
        mimic_x = torch.cat([mimic_numeric, mimic_missing], dim=1)
        
        eicu_numeric_cols = [col for col in test_eicu_data.columns 
                            if ('_mean' in col or '_min' in col or '_max' in col or '_std' in col) 
                            or col in ['Age', 'Gender']]
        eicu_missing_cols = [col for col in test_eicu_data.columns if '_missing' in col]
        
        eicu_numeric = torch.FloatTensor(test_eicu_data[eicu_numeric_cols].values)
        eicu_missing = torch.FloatTensor(test_eicu_data[eicu_missing_cols].values)
        eicu_x = torch.cat([eicu_numeric, eicu_missing], dim=1)
        
        # Perform round-trip translation
        with torch.no_grad():
            # eICU -> MIMIC -> eICU
            eicu_to_mimic = self.model.translate_eicu_to_mimic(eicu_x)
            eicu_round_trip = self.model.translate_mimic_to_eicu(eicu_to_mimic)
            
            # MIMIC -> eICU -> MIMIC
            mimic_to_eicu = self.model.translate_mimic_to_eicu(mimic_x)
            mimic_round_trip = self.model.translate_eicu_to_mimic(mimic_to_eicu)
        
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
            for suffix in ['_mean', '_min', '_max', '_std']:
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
        test_eicu_data = pd.read_csv(self.output_dir / "data" / "test_eicu_preprocessed.csv")
        
        # Convert to tensors (POC features format)
        mimic_numeric_cols = [col for col in test_mimic_data.columns 
                             if ('_mean' in col or '_min' in col or '_max' in col or '_std' in col) 
                             or col in ['Age', 'Gender']]
        mimic_missing_cols = [col for col in test_mimic_data.columns if '_missing' in col]
        
        mimic_numeric = torch.FloatTensor(test_mimic_data[mimic_numeric_cols].values)
        mimic_missing = torch.FloatTensor(test_mimic_data[mimic_missing_cols].values)
        mimic_x = torch.cat([mimic_numeric, mimic_missing], dim=1)
        
        eicu_numeric_cols = [col for col in test_eicu_data.columns 
                            if ('_mean' in col or '_min' in col or '_max' in col or '_std' in col) 
                            or col in ['Age', 'Gender']]
        eicu_missing_cols = [col for col in test_eicu_data.columns if '_missing' in col]
        
        eicu_numeric = torch.FloatTensor(test_eicu_data[eicu_numeric_cols].values)
        eicu_missing = torch.FloatTensor(test_eicu_data[eicu_missing_cols].values)
        eicu_x = torch.cat([eicu_numeric, eicu_missing], dim=1)
        
        # Perform translation
        with torch.no_grad():
            eicu_to_mimic = self.model.translate_eicu_to_mimic(eicu_x)
            mimic_to_eicu = self.model.translate_mimic_to_eicu(mimic_x)
        
        # KS test for each feature (updated for POC features)
        ks_results = []
        col_idx = 0
        
        # Test clinical features
        for feature in self.feature_spec['clinical_features']:
            for suffix in ['_mean', '_min', '_max', '_std']:
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
        test_eicu_data = pd.read_csv(self.output_dir / "data" / "test_eicu_preprocessed.csv")
        
        # Create synthetic target variable (in-hospital mortality)
        # In real scenario, this would come from actual outcome data
        np.random.seed(42)
        train_mimic_data['mortality'] = np.random.choice([0, 1], size=len(train_mimic_data), p=[0.8, 0.2])
        test_mimic_data['mortality'] = np.random.choice([0, 1], size=len(test_mimic_data), p=[0.8, 0.2])
        test_eicu_data['mortality'] = np.random.choice([0, 1], size=len(test_eicu_data), p=[0.8, 0.2])
        
        # Prepare features (POC features format)
        feature_cols = [col for col in train_mimic_data.columns 
                       if ('_mean' in col or '_min' in col or '_max' in col or '_std' in col) 
                       or col in ['Age', 'Gender']]
        
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
        eicu_numeric_cols = [col for col in test_eicu_data.columns 
                            if ('_mean' in col or '_min' in col or '_max' in col or '_std' in col) 
                            or col in ['Age', 'Gender']]
        eicu_missing_cols = [col for col in test_eicu_data.columns if '_missing' in col]
        
        eicu_numeric = torch.FloatTensor(test_eicu_data[eicu_numeric_cols].values)
        eicu_missing = torch.FloatTensor(test_eicu_data[eicu_missing_cols].values)
        eicu_x = torch.cat([eicu_numeric, eicu_missing], dim=1)
        
        with torch.no_grad():
            eicu_translated = self.model.translate_eicu_to_mimic(eicu_x)
        
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
        test_eicu_data = pd.read_csv(self.output_dir / "data" / "test_eicu_preprocessed.csv")
        
        # Convert to tensors (POC features format)
        mimic_numeric_cols = [col for col in test_mimic_data.columns 
                             if ('_mean' in col or '_min' in col or '_max' in col or '_std' in col) 
                             or col in ['Age', 'Gender']]
        mimic_missing_cols = [col for col in test_mimic_data.columns if '_missing' in col]
        
        mimic_numeric = torch.FloatTensor(test_mimic_data[mimic_numeric_cols].values)
        mimic_missing = torch.FloatTensor(test_mimic_data[mimic_missing_cols].values)
        mimic_x = torch.cat([mimic_numeric, mimic_missing], dim=1)
        
        eicu_numeric_cols = [col for col in test_eicu_data.columns 
                            if ('_mean' in col or '_min' in col or '_max' in col or '_std' in col) 
                            or col in ['Age', 'Gender']]
        eicu_missing_cols = [col for col in test_eicu_data.columns if '_missing' in col]
        
        eicu_numeric = torch.FloatTensor(test_eicu_data[eicu_numeric_cols].values)
        eicu_missing = torch.FloatTensor(test_eicu_data[eicu_missing_cols].values)
        eicu_x = torch.cat([eicu_numeric, eicu_missing], dim=1)
        
        # Perform translation
        with torch.no_grad():
            eicu_to_mimic = self.model.translate_eicu_to_mimic(eicu_x)
            mimic_to_eicu = self.model.translate_mimic_to_eicu(mimic_x)
        
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
            # eICU round-trip
            eicu_to_mimic = self.model.translate_eicu_to_mimic(eicu_x)
            eicu_round_trip = self.model.translate_mimic_to_eicu(eicu_to_mimic)
            eicu_errors = torch.mean((eicu_x - eicu_round_trip) ** 2, dim=0)
            
            # MIMIC round-trip
            mimic_to_eicu = self.model.translate_mimic_to_eicu(mimic_x)
            mimic_round_trip = self.model.translate_eicu_to_mimic(mimic_to_eicu)
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
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(config_path)
    
    # Override output directory if specified
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    
    # Create evaluator
    evaluator = Evaluator(config, args.model)
    
    # Run evaluation
    summary = evaluator.run_full_evaluation()
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
