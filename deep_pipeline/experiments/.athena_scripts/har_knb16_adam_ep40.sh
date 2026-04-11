#!/bin/bash
#SBATCH --job-name=har_knb16_adam
#SBATCH --account=aran_prj
#SBATCH --partition=l40s-shared,a100-public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --qos=12h_4g
#SBATCH --output=/home/omer.gotfrid/Thesis/EHR_Translator/deep_pipeline/experiments/logs/har_knb16_adam_%j.out
#SBATCH --error=/home/omer.gotfrid/Thesis/EHR_Translator/deep_pipeline/experiments/logs/har_knb16_adam_%j.err

source ~/miniforge3/etc/profile.d/conda.sh

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if echo "$GPU_NAME" | grep -qi "Blackwell\|RTX PRO 6000\|H200\|H100"; then
    CONDA_ENV="yaib-cu128"
else
    CONDA_ENV="yaib"
fi
conda activate "$CONDA_ENV"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd ~/Thesis/EHR_Translator/deep_pipeline
mkdir -p experiments/logs

echo "=== HAR knb16_adam Start ==="
echo "Node: $(hostname)"
echo "GPU: $GPU_NAME"
echo "Conda env: $CONDA_ENV"
echo "Date: $(date)"
echo "========================================"

python scripts/run_adatime.py \
  --dataset HAR \
  --all-scenarios \
  --use-cnn \
  --last-epoch \
  --epochs 40 \
  --patience 0 \
  --config /home/omer.gotfrid/Thesis/EHR_Translator/deep_pipeline/experiments/.athena_configs/har_knb16_adam_ep40_athena.json \
  --data-path /home/omer.gotfrid/Thesis/AdaTime/data \
  --device cuda:0 \
  --variant _adatime_knb16_adam

echo "=== HAR knb16_adam Complete ==="
echo "Date: $(date)"
