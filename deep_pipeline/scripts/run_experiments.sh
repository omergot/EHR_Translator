#!/bin/bash
# Run all A/B/C recommendation experiments sequentially.
# Supports resume via state file tracking.
# Usage: bash scripts/run_experiments.sh [GPU_ID]
set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_FILE="$REPO_DIR/experiments/.state"
GPU="${1:-0}"

mkdir -p "$REPO_DIR/experiments/logs" "$REPO_DIR/experiments/results"
touch "$STATE_FILE"

EXPERIMENTS=(c1_focal c3_cosine_fid a3_padding_fid a1_var_batching a4_truncate
             c2_gradnorm a2_chunking b1_hidden_mmd b3_knn b5_ot b6_dann b4_contrastive b2_shared_enc)
TASKS=(sepsis mortality)

for exp in "${EXPERIMENTS[@]}"; do
  for task in "${TASKS[@]}"; do
    key="${exp}_${task}"
    grep -q "^${key}$" "$STATE_FILE" 2>/dev/null && echo "Skip $key (done)" && continue

    echo ""
    echo "========================================"
    echo "  Running: $key on GPU $GPU"
    echo "  $(date)"
    echo "========================================"

    cd "$REPO_DIR"
    branch="exp/$exp"

    # Check if branch exists
    if ! git rev-parse --verify "$branch" >/dev/null 2>&1; then
      echo "ERROR: Branch $branch does not exist. Skipping $key."
      continue
    fi

    git checkout "$branch" 2>/dev/null

    config="experiments/configs/${exp}_${task}_debug.json"
    if [ ! -f "$config" ]; then
      echo "ERROR: Config $config not found. Skipping $key."
      git checkout master 2>/dev/null
      continue
    fi

    # Run experiment
    if CUDA_VISIBLE_DEVICES="$GPU" python run.py train_and_eval \
        --config "$config" \
        --output_parquet "experiments/results/${exp}_${task}.parquet" 2>&1; then
      echo "$key completed successfully"
    else
      echo "ERROR: $key failed (exit code $?)"
    fi

    # Collect results (parse log)
    python scripts/collect_result.py "$exp" "$task" 2>/dev/null || true

    echo "$key" >> "$STATE_FILE"
    git checkout deep_pipeline 2>/dev/null
  done
done

echo ""
echo "========================================"
echo "  All experiments done. Aggregating..."
echo "========================================"

cd "$REPO_DIR"
git checkout master 2>/dev/null
python scripts/aggregate_results.py
