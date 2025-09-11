# GPU Configuration Guide for POC Translator

Your system has **4 Tesla V100 GPUs** with 32GB memory each - excellent for deep learning training!

## 🖥️ Available GPUs
```
GPU 0: Tesla V100S-PCIE-32GB (~29GB available)
GPU 1: Tesla V100-PCIE-32GB  (~29GB available) 
GPU 2: Tesla V100-PCIE-32GB  (~29GB available)
GPU 3: Tesla V100S-PCIE-32GB (~29GB available)
```

## ⚙️ Configuration Options

### 1. Using Config File (Recommended)
The `conf/config.yml` now includes GPU settings:

```yaml
# GPU Configuration
gpu:
  # GPU device to use (0, 1, 2, 3 for your Tesla V100s, or 'auto' for automatic selection)
  device: 0
  # Enable mixed precision training for faster training and lower memory usage
  precision: "16-mixed"
  # Enable optimized attention for V100 (if using transformer-based components)
  enable_flash_attention: false
  # Number of workers for data loading (increase for faster data loading)
  num_workers: 4

training:
  batch_size: 256  # Optimized for V100 memory capacity
```

### 2. Command Line Override
You can override the GPU device from command line:

```bash
# Use specific GPU
python src/train.py --config conf/config.yml --gpu 0

# Use different GPU
python src/train.py --config conf/config.yml --gpu 1

# Auto-select GPU
python src/train.py --config conf/config.yml --gpu auto
```

## 🚀 Training Commands

### Basic Training
```bash
cd /bigdata/omerg/Thesis/poc_translator
python src/train.py --config conf/config.yml
```

### Training with Specific GPU
```bash
# Use GPU 0 (default)
python src/train.py --config conf/config.yml --gpu 0

# Use GPU 1 (if GPU 0 is busy)
python src/train.py --config conf/config.yml --gpu 1
```

### Dry Run Test
```bash
# Quick test with 1 epoch
python src/train.py --config conf/config.yml --dry-run
```

### Full Training with Preprocessing
```bash
# Force preprocessing + training
python src/train.py --config conf/config.yml --preprocess
```

## 💾 Memory Optimization

### Batch Size Recommendations
- **Conservative**: `batch_size: 128` (safe, uses ~8-12GB GPU memory)
- **Optimized**: `batch_size: 256` (current setting, uses ~15-20GB GPU memory)  
- **Aggressive**: `batch_size: 512` (uses ~25-30GB GPU memory, may OOM)

### If You Get Out of Memory (OOM) Errors
1. Reduce batch size in `conf/config.yml`:
   ```yaml
   training:
     batch_size: 128  # Reduce from 256
   ```

2. Or use a different GPU:
   ```bash
   python src/train.py --config conf/config.yml --gpu 1
   ```

## 🔍 Monitoring GPU Usage

### Check GPU Status
```bash
nvidia-smi
```

### Monitor in Real-time
```bash
watch -n 1 nvidia-smi
```

### Check PyTorch GPU Detection
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

## ⚡ Performance Tips

### 1. Mixed Precision Training
- **Enabled by default** with `precision: "16-mixed"`
- Reduces memory usage by ~50%
- Increases training speed by ~1.5-2x on V100s

### 2. Data Loading Optimization
- `num_workers: 4` for parallel data loading
- `pin_memory: true` for faster GPU transfers
- `persistent_workers: true` for reduced overhead

### 3. Multiple GPU Usage (Future)
If you want to use multiple GPUs later, you can modify the config:
```yaml
gpu:
  device: [0, 1]  # Use GPUs 0 and 1
```

## 🧪 Testing GPU Setup

Test if everything works:
```bash
cd /bigdata/omerg/Thesis/poc_translator
python -c "
import torch
import yaml
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ GPU count:', torch.cuda.device_count())
print('✅ Current GPU:', torch.cuda.current_device() if torch.cuda.is_available() else 'None')

# Test config loading
with open('conf/config.yml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ GPU config:', config.get('gpu', 'Not found'))
print('✅ Batch size:', config['training']['batch_size'])
"
```

## 🏃 Quick Start

1. **Verify GPU setup**:
   ```bash
   nvidia-smi
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

2. **Run preprocessing** (if not done):
   ```bash
   python src/preprocess.py --config conf/config.yml --fit
   ```

3. **Start training**:
   ```bash
   python src/train.py --config conf/config.yml
   ```

The training will automatically:
- ✅ Detect and use GPU 0
- ✅ Enable mixed precision training
- ✅ Use optimized batch size (256)
- ✅ Use 4 worker processes for data loading
- ✅ Log GPU usage information

## ⚠️ Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'pytorch_lightning'"
```bash
pip install pytorch-lightning
```

### Problem: "CUDA out of memory"
1. Reduce batch size in config.yml
2. Use a different GPU: `--gpu 1`
3. Close other GPU processes

### Problem: "GPU not detected"
1. Check: `nvidia-smi`
2. Check: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with CUDA support if needed

Happy training! 🎯
