#!/bin/bash
# Run AKI-trained translator evaluations on the AKI-sepsis intersection dataset.
#
# This script:
# 1. Temporarily adds `complete_train = True` to the sepsis gin config
#    (so YAIB puts all data into the test split for maximum coverage)
# 2. Runs translate_and_eval for delta and shared latent translators
# 3. Reverts the gin config change
#
# Usage: bash scripts/run_aki_sepsis_intersection_eval.sh [GPU_ID]

set -euo pipefail

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

GIN_FILE="/bigdata/omerg/Thesis/pretrained_models/mimic/Sepsis/LSTM/repetition_0/fold_0/train_config.gin"
GIN_LINE="execute_repeated_cv.complete_train = True"

# Add complete_train to gin config
echo "" >> "$GIN_FILE"
echo "# TEMPORARY: cross-task intersection eval" >> "$GIN_FILE"
echo "$GIN_LINE" >> "$GIN_FILE"
echo "[INFO] Added complete_train=True to $GIN_FILE"

# Cleanup function to revert gin change
cleanup() {
    # Remove the last 3 lines (blank line + comment + setting)
    head -n -3 "$GIN_FILE" > "${GIN_FILE}.tmp" && mv "${GIN_FILE}.tmp" "$GIN_FILE"
    echo "[INFO] Reverted complete_train change from $GIN_FILE"
}
trap cleanup EXIT

echo "=== Evaluating AKI delta translator on sepsis intersection ==="
python run.py translate_and_eval \
    --config configs/aki_delta_eval_on_sepsis_intersection.json \
    --translator_checkpoint runs/aki_delta_full/best_translator.pt \
    --output_parquet runs/aki_delta_eval_on_sepsis_intersection/results.parquet

echo "=== Evaluating AKI shared latent translator on sepsis intersection ==="
python run.py translate_and_eval \
    --config configs/aki_sl_eval_on_sepsis_intersection.json \
    --translator_checkpoint runs/aki_shared_latent_full/best_translator.pt \
    --output_parquet runs/aki_sl_eval_on_sepsis_intersection/results.parquet

echo "=== Done ==="
