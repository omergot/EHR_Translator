#!/bin/bash
# Run remaining A/B/C experiments on full mortality data.
# Each experiment runs on its own exp/ branch via git worktree,
# merged with latest master before running.
# GPU 1: c3, b3, b2, c1 (4 experiments, sequential)
# GPU 2: b1, b5, a1 (3 experiments, sequential)
set -eo pipefail

DEEP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GIT_ROOT="$(cd "$DEEP_DIR/.." && pwd)"
cd "$DEEP_DIR"
STATE_FILE="$DEEP_DIR/experiments/.state_full"
WORKTREE_BASE="/tmp/ehr_exp_full_worktrees"
mkdir -p "$DEEP_DIR/experiments/logs" "$DEEP_DIR/experiments/results" "$WORKTREE_BASE"
touch "$STATE_FILE"

# GPU assignment
GPU1_EXPS=("c3_cosine_fid" "b3_knn" "b2_shared_enc" "c1_focal")
GPU2_EXPS=("b1_hidden_mmd" "b5_ot" "a1_var_batching")

run_experiment() {
  local exp="$1"
  local gpu="$2"
  local branch="exp/$exp"
  local worktree="${WORKTREE_BASE}/${exp}"
  local wt_deep="${worktree}/deep_pipeline"
  local key="${exp}_full_mortality"
  local config_name="${exp}_full_mortality.json"

  # Skip if already done
  if grep -q "^${key}$" "$STATE_FILE" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] [GPU $gpu] [skip] $key already done"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] [GPU $gpu] Setting up $exp (branch: $branch)"

  # Create worktree from git root
  rm -rf "$worktree" 2>/dev/null || true
  git -C "$GIT_ROOT" worktree add "$worktree" "$branch" --quiet 2>/dev/null || {
    echo "[$(date +%H:%M:%S)] [GPU $gpu] [error] Failed to create worktree for $branch"
    return 1
  }

  # Merge latest deep_pipeline into the worktree branch
  echo "[$(date +%H:%M:%S)] [GPU $gpu] Merging master into $branch..."
  cd "$worktree"
  git merge master --no-edit --quiet 2>&1 || {
    echo "[$(date +%H:%M:%S)] [GPU $gpu] [warn] Merge conflict for $branch, attempting auto-resolution..."
    git checkout --theirs . 2>/dev/null || true
    git add -A 2>/dev/null || true
    git commit --no-edit -m "Auto-merge master into $branch" 2>/dev/null || true
  }
  cd "$DEEP_DIR"

  # Copy full-data config and collector into worktree
  mkdir -p "$wt_deep/experiments/configs" "$wt_deep/experiments/results" "$wt_deep/experiments/logs"
  cp "$DEEP_DIR/experiments/configs/${config_name}" "$wt_deep/experiments/configs/" 2>/dev/null || true
  cp "$DEEP_DIR/experiments/collect_result.py" "$wt_deep/experiments/" 2>/dev/null || true

  local log="$DEEP_DIR/experiments/logs/${key}.log"
  local parquet="$DEEP_DIR/experiments/results/${key}.parquet"

  echo "[$(date +%H:%M:%S)] [GPU $gpu] [start] $key"
  > "$log"

  # Run from worktree's deep_pipeline directory
  cd "$wt_deep"
  if CUDA_VISIBLE_DEVICES="$gpu" python run.py train_and_eval \
      --config "experiments/configs/${config_name}" \
      --output_parquet "$parquet" \
      >> "$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] [GPU $gpu] [done] $key SUCCESS"
  else
    echo "[$(date +%H:%M:%S)] [GPU $gpu] [error] $key FAILED (exit $?)"
  fi
  cd "$DEEP_DIR"

  # Collect result
  python experiments/collect_result.py "${exp}_full" "mortality" 2>/dev/null || true

  # Mark done
  echo "$key" >> "$STATE_FILE"

  # Cleanup worktree
  git -C "$GIT_ROOT" worktree remove "$worktree" --force 2>/dev/null || rm -rf "$worktree"
  echo "[$(date +%H:%M:%S)] [GPU $gpu] Finished $exp"
}

run_gpu_queue() {
  local gpu="$1"
  shift
  local exps=("$@")

  for exp in "${exps[@]}"; do
    run_experiment "$exp" "$gpu"
  done
}

echo "======================================================="
echo "  Full-data Mortality: Remaining A/B/C Experiments"
echo "  GPU 1: ${GPU1_EXPS[*]} (4 experiments)"
echo "  GPU 2: ${GPU2_EXPS[*]} (3 experiments)"
echo "  Started: $(date)"
echo "======================================================="

# Launch both GPU queues in parallel
run_gpu_queue 1 "${GPU1_EXPS[@]}" &
pid1=$!
run_gpu_queue 2 "${GPU2_EXPS[@]}" &
pid2=$!

# Wait for both
wait "$pid1" 2>/dev/null || true
wait "$pid2" 2>/dev/null || true

echo ""
echo "======================================================="
echo "  All mortality experiments complete!"
echo "  Finished: $(date)"
echo "======================================================="

# Print results
echo ""
echo "Results:"
for exp in c3_cosine_fid b1_hidden_mmd b3_knn b5_ot b2_shared_enc a1_var_batching c1_focal; do
  result_file="$DEEP_DIR/experiments/results/${exp}_full_mortality.json"
  if [ -f "$result_file" ]; then
    echo "  ${exp}_full / mortality:"
    python3 -c "
import json
with open('$result_file') as f:
    r = json.load(f)
d = r.get('difference', {})
print(f'    AUCROC: {d.get(\"AUCROC\", \"N/A\"):+.4f}  AUCPR: {d.get(\"AUCPR\", \"N/A\"):+.4f}  status: {r[\"status\"]}')
" 2>/dev/null || echo "    (parse error)"
  fi
done
