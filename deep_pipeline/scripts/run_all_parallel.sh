#!/bin/bash
# Run all A/B/C experiments using 3 GPUs in parallel via git worktrees.
# Each GPU runs one experiment at a time. When one finishes, the next starts.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
STATE_FILE="$REPO_DIR/experiments/.state"
WORKTREE_BASE="/tmp/ehr_exp_worktrees"
mkdir -p "$REPO_DIR/experiments/logs" "$REPO_DIR/experiments/results" "$WORKTREE_BASE"
touch "$STATE_FILE"

# All experiment/task pairs
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

GPUS=(0 1 2)

run_one() {
  local exp="$1"
  local task="$2"
  local gpu="$3"
  local key="${exp}_${task}"

  # Skip if done
  if grep -q "^${key}$" "$STATE_FILE" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] [skip] $key already done"
    return 0
  fi

  local branch="exp/$exp"
  local config="experiments/configs/${exp}_${task}_debug.json"
  local log="$REPO_DIR/experiments/logs/${exp}_${task}.log"
  local parquet="$REPO_DIR/experiments/results/${exp}_${task}.parquet"

  echo "[$(date +%H:%M:%S)] [start] $key on GPU $gpu"

  # Create worktree
  local worktree="${WORKTREE_BASE}/${exp}_${task}"
  rm -rf "$worktree" 2>/dev/null || true
  git -C "$REPO_DIR" worktree add "$worktree" "$branch" --quiet 2>/dev/null || {
    echo "[$(date +%H:%M:%S)] [error] $key: failed to create worktree for $branch"
    return 1
  }

  # Copy experiment configs to worktree (they live in experiments/ which is gitignored)
  mkdir -p "$worktree/experiments/configs" "$worktree/experiments/results" "$worktree/experiments/logs"
  cp "$REPO_DIR/experiments/configs/${exp}_${task}_debug.json" "$worktree/experiments/configs/" 2>/dev/null || true

  # Run from worktree
  cd "$worktree"
  if CUDA_VISIBLE_DEVICES="$gpu" python run.py train_and_eval \
      --config "$config" \
      --output_parquet "$parquet" \
      >> "$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] [done] $key SUCCESS"
  else
    echo "[$(date +%H:%M:%S)] [error] $key FAILED (exit $?)"
  fi
  cd "$REPO_DIR"

  # Collect result
  python scripts/collect_result.py "$exp" "$task" 2>/dev/null || true

  # Cleanup worktree
  git -C "$REPO_DIR" worktree remove "$worktree" --force 2>/dev/null || rm -rf "$worktree"

  echo "$key" >> "$STATE_FILE"
}

# Run jobs in parallel batches of 3 (one per GPU)
job_idx=0
total=${#JOBS[@]}
echo "Starting $total jobs across ${#GPUS[@]} GPUs"

while [ $job_idx -lt $total ]; do
  pids=()
  gpu_idx=0
  batch_start=$job_idx

  # Assign up to 3 jobs (one per GPU)
  while [ $gpu_idx -lt ${#GPUS[@]} ] && [ $job_idx -lt $total ]; do
    IFS=':' read -r exp task <<< "${JOBS[$job_idx]}"
    key="${exp}_${task}"

    # Skip completed ones
    if grep -q "^${key}$" "$STATE_FILE" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] [skip] $key already done"
      job_idx=$((job_idx + 1))
      continue
    fi

    run_one "$exp" "$task" "${GPUS[$gpu_idx]}" &
    pids+=($!)
    gpu_idx=$((gpu_idx + 1))
    job_idx=$((job_idx + 1))
  done

  # Wait for this batch
  if [ ${#pids[@]} -gt 0 ]; then
    echo "[$(date +%H:%M:%S)] Waiting for batch (${#pids[@]} jobs)..."
    for pid in "${pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
    echo "[$(date +%H:%M:%S)] Batch complete"
  fi
done

echo ""
echo "========================================"
echo "  All experiments done. Aggregating..."
echo "========================================"
cd "$REPO_DIR"
python scripts/aggregate_results.py

echo "COMPLETE at $(date)"
