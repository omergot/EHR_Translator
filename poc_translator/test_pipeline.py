#!/usr/bin/env python3
"""
Test Pipeline Script
Tests the entire Cycle-VAE pipeline end-to-end with sample data.
"""

import sys
import logging
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the entire pipeline"""
    logger.info("Starting pipeline test...")
    
    try:
        # 1. Test configuration loading
        logger.info("1. Testing configuration...")
        config_path = Path(__file__).parent / "conf" / "config.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✓ Configuration loaded successfully")
        
        # 2. Test SQL query generation
        logger.info("2. Testing SQL query generation...")
        from sql.make_queries import main as generate_queries
        generate_queries()
        logger.info("✓ SQL queries generated successfully")
        
        # 3. Test sample data creation
        logger.info("3. Testing sample data creation...")
        from data.raw_extractors import create_sample_data
        mimic_data, eicu_data = create_sample_data()
        logger.info(f"✓ Sample data created - MIMIC: {len(mimic_data)}, eICU: {len(eicu_data)}")
        
        # 4. Test preprocessing
        logger.info("4. Testing preprocessing...")
        from src.preprocess import Preprocessor
        preprocessor = Preprocessor(config)
        feature_spec = preprocessor.preprocess()
        logger.info(f"✓ Preprocessing completed - {feature_spec['n_features']} features")
        
        # 5. Test dataset creation
        logger.info("5. Testing dataset creation...")
        from src.dataset import CombinedDataModule
        data_module = CombinedDataModule(config, feature_spec)
        data_module.setup('fit')
        logger.info("✓ Dataset created successfully")
        
        # 6. Test model creation
        logger.info("6. Testing model creation...")
        from src.model import CycleVAE
        model = CycleVAE(config, feature_spec)
        logger.info(f"✓ Model created - Input dim: {model.input_dim}, Latent dim: {model.latent_dim}")
        
        # 7. Test utility functions
        logger.info("7. Testing utility functions...")
        from src.utils import mmd_rbf, ks_test_featurewise
        import numpy as np
        
        # Create sample data for testing
        X = np.random.normal(0, 1, (100, 10))
        Y = np.random.normal(0.5, 1, (100, 10))
        
        mmd_val = mmd_rbf(X, Y)
        ks_stats, p_values = ks_test_featurewise(X, Y)
        
        logger.info(f"✓ Utility functions tested - MMD: {mmd_val:.4f}, KS mean: {ks_stats.mean():.4f}")
        
        # 8. Test training (dry run)
        logger.info("8. Testing training (dry run)...")
        from src.train import train_model
        model, trainer = train_model(config, feature_spec, dry_run=True)
        logger.info("✓ Training dry run completed successfully")
        
        logger.info("🎉 All pipeline tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("=== CYCLE-VAE PIPELINE TEST ===")
    
    success = test_pipeline()
    
    if success:
        logger.info("✅ Pipeline test completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Update database connections in conf/config.yml")
        logger.info("2. Run: python sql/make_queries.py")
        logger.info("3. Run: python data/raw_extractors.py")
        logger.info("4. Run: python src/preprocess.py --fit")
        logger.info("5. Run: python src/train.py --config conf/config.yml")
        logger.info("6. Run: python src/evaluate.py --model checkpoints/best.ckpt")
    else:
        logger.error("❌ Pipeline test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
