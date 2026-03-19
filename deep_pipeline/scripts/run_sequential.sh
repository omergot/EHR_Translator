#!/bin/bash
# Run all A/B/C experiments sequentially on GPU 0.
# Uses git checkout to switch branches between experiments.
set -o pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
STATE_FILE="$REPO_DIR/experiments/.state"
mkdir -p "$REPO_DIR/experiments/logs" "$REPO_DIR/experiments/results"
touch "$STATE_FILE"

GPU=0

declare -a JOBS=(
  "c1_focal:sepsis"
  "c1_focal:mortality"
  "c3_cosine_fid:sepsis"
  "c3_cosine_fid:mortality"
  "a3_padding_fid:sepsis"
  "a3_padding_fid:mortality"
  "a1_var_batching:sepsis"
  "a1_var_batching:mortality"
  "a4_truncate:sepsis"
  "a4_truncate:mortality"
  "c2_gradnorm:sepsis"
  "c2_gradnorm:mortality"
  "a2_chunking:sepsis"
  "a2_chunking:mortality"
  "b1_hidden_mmd:sepsis"
  "b1_hidden_mmd:mortality"
  "b3_knn:sepsis"
  "b3_knn:mortality"
  "b5_ot:sepsis"
  "b5_ot:mortality"
  "b6_dann:sepsis"
  "b6_dann:mortality"
  "b4_contrastive:sepsis"
  "b4_contrastive:mortality"
  "b2_shared_enc:sepsis"
  "b2_shared_enc:mortality"
)

total=${#JOBS[@]}
done_count=0

for job in "${JOBS[@]}"; do
  IFS=':' read -r exp task <<< "$job"
  key="${exp}_${task}"

  # Skip if done
  if grep -q "^${key}$" "$STATE_FILE" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] [skip] $key already done"
    done_count=$((done_count + 1))
    continue
  fi

  done_count=$((done_count + 1))
  branch="exp/$exp"
  config="experiments/configs/${exp}_${task}_debug.json"
  log="experiments/logs/${exp}_${task}.log"
  parquet="experiments/results/${exp}_${task}.parquet"

  echo ""
  echo "========================================"
  echo "  [$done_count/$total] $key on GPU $GPU"
  echo "  Branch: $branch"
  echo "  $(date)"
  echo "========================================"

  # Switch branch
  git checkout "$branch" 2>/dev/null || {
    echo "[error] Branch $branch not found. Skipping."
    continue
  }

  # Clear old log
  > "$log"

  # Run
  if CUDA_VISIBLE_DEVICES="$GPU" python run.py train_and_eval \
      --config "$config" \
      --output_parquet "$parquet" \
      >> "$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] [done] $key SUCCESS"
  else
    echo "[$(date +%H:%M:%S)] [error] $key FAILED (exit $?)"
  fi

  # Collect result using fixed copy in gitignored experiments/ dir
  python experiments/collect_result.py "$exp" "$task" 2>/dev/null || true

  # Mark done
  echo "$key" >> "$STATE_FILE"
done

echo ""
echo "========================================"
echo "  All experiments done. Aggregating..."
echo "========================================"
cd "$REPO_DIR"
git checkout master 2>/dev/null
python scripts/aggregate_results.py

echo "COMPLETE at $(date)"
