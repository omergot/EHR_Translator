#!/bin/bash
# Run filtered sepsis experiments: delta then shared latent
# GPU 0, sequential (delta ~1-2h, SL ~3-4h with pretrain)
set -e

PROJ_DIR="/bigdata/omerg/Thesis/EHR_Translator/deep_pipeline"
cd "$PROJ_DIR"

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Sepsis Filtered Experiments (AKI density)"
echo "GPU: 0"
echo "Started: $(date)"
echo "=========================================="

# 1. Delta-based
echo ""
echo "[1/2] Delta-based on filtered sepsis cohort..."
echo "Config: configs/sepsis_filtered_delta_full.json"
echo "Start: $(date)"
python run.py train_and_eval \
    --config configs/sepsis_filtered_delta_full.json \
    --output_parquet runs/sepsis_filtered_delta_full/results.parquet \
    2>&1 | tee experiments/logs/sepsis_filtered_delta_full.log
echo "Done: $(date)"

# 2. Shared Latent
echo ""
echo "[2/2] Shared Latent v3 on filtered sepsis cohort..."
echo "Config: configs/sepsis_filtered_sl_full.json"
echo "Start: $(date)"
python run.py train_and_eval \
    --config configs/sepsis_filtered_sl_full.json \
    --output_parquet runs/sepsis_filtered_sl_full/results.parquet \
    2>&1 | tee experiments/logs/sepsis_filtered_sl_full.log
echo "Done: $(date)"

echo ""
echo "=========================================="
echo "All sepsis filtered experiments complete!"
echo "Finished: $(date)"
echo "=========================================="
