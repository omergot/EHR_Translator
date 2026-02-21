#!/bin/bash
# Run top-3 experiments (A3, C2, A4) on full data using 3 GPUs in parallel.
# Each GPU handles one experiment branch, running sepsis then mortality sequentially.
set -eo pipefail

DEEP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GIT_ROOT="$(cd "$DEEP_DIR/.." && pwd)"
cd "$DEEP_DIR"
STATE_FILE="$DEEP_DIR/experiments/.state_full"
WORKTREE_BASE="/tmp/ehr_exp_full_worktrees"
mkdir -p "$DEEP_DIR/experiments/logs" "$DEEP_DIR/experiments/results" "$WORKTREE_BASE"
touch "$STATE_FILE"

# Map: branch -> GPU
declare -A BRANCH_MAP=(
  ["a3_padding_fid"]="0"
  ["c2_gradnorm"]="1"
  ["a4_truncate"]="2"
)

run_experiment() {
  local exp="$1"
  local gpu="$2"
  local branch="exp/$exp"
  local worktree="${WORKTREE_BASE}/${exp}"
  local wt_deep="${worktree}/deep_pipeline"
  local exp_full="${exp}_full"

  echo "[$(date +%H:%M:%S)] [GPU $gpu] Starting $exp (branch: $branch)"

  # Create worktree from git root
  rm -rf "$worktree" 2>/dev/null || true
  git -C "$GIT_ROOT" worktree add "$worktree" "$branch" --quiet 2>/dev/null || {
    echo "[$(date +%H:%M:%S)] [GPU $gpu] [error] Failed to create worktree for $branch"
    return 1
  }

  # Copy full-data configs and collector into worktree's deep_pipeline
  mkdir -p "$wt_deep/experiments/configs" "$wt_deep/experiments/results" "$wt_deep/experiments/logs"
  cp "$DEEP_DIR/experiments/configs/${exp_full}_"*.json "$wt_deep/experiments/configs/" 2>/dev/null || true
  cp "$DEEP_DIR/experiments/collect_result.py" "$wt_deep/experiments/" 2>/dev/null || true

  for task in sepsis mortality; do
    local key="${exp_full}_${task}"

    # Skip if done
    if grep -q "^${key}$" "$STATE_FILE" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] [GPU $gpu] [skip] $key already done"
      continue
    fi

    local config="experiments/configs/${exp_full}_${task}.json"
    local log="$DEEP_DIR/experiments/logs/${exp_full}_${task}.log"
    local parquet="$DEEP_DIR/experiments/results/${exp_full}_${task}.parquet"

    echo "[$(date +%H:%M:%S)] [GPU $gpu] [start] $key"

    # Clear old log
    > "$log"

    # Run from worktree's deep_pipeline directory
    cd "$wt_deep"
    if CUDA_VISIBLE_DEVICES="$gpu" python run.py train_and_eval \
        --config "$config" \
        --output_parquet "$parquet" \
        >> "$log" 2>&1; then
      echo "[$(date +%H:%M:%S)] [GPU $gpu] [done] $key SUCCESS"
    else
      echo "[$(date +%H:%M:%S)] [GPU $gpu] [error] $key FAILED (exit $?)"
    fi
    cd "$DEEP_DIR"

    # Collect result
    python experiments/collect_result.py "${exp_full}" "$task" 2>/dev/null || true

    # Mark done
    echo "$key" >> "$STATE_FILE"
  done

  # Cleanup worktree
  git -C "$GIT_ROOT" worktree remove "$worktree" --force 2>/dev/null || rm -rf "$worktree"
  echo "[$(date +%H:%M:%S)] [GPU $gpu] Finished $exp"
}

echo "========================================"
echo "  Full-data experiments: A3, C2, A4"
echo "  3 experiments x 2 tasks = 6 runs"
echo "  Started: $(date)"
echo "========================================"

# Launch all 3 experiments in parallel
pids=()
for exp in "${!BRANCH_MAP[@]}"; do
  gpu="${BRANCH_MAP[$exp]}"
  run_experiment "$exp" "$gpu" &
  pids+=($!)
done

# Wait for all
for pid in "${pids[@]}"; do
  wait "$pid" 2>/dev/null || true
done

echo ""
echo "========================================"
echo "  All full-data experiments done."
echo "  Completed: $(date)"
echo "========================================"

# Print results
echo ""
echo "Results:"
for exp in a3_padding_fid c2_gradnorm a4_truncate; do
  for task in sepsis mortality; do
    result_file="$DEEP_DIR/experiments/results/${exp}_full_${task}.json"
    if [ -f "$result_file" ]; then
      echo "  ${exp}_full / $task:"
      python3 -c "
import json
with open('$result_file') as f:
    r = json.load(f)
d = r.get('difference', {})
print(f'    AUCROC: {d.get(\"AUCROC\", \"N/A\"):+.4f}  AUCPR: {d.get(\"AUCPR\", \"N/A\"):+.4f}  status: {r[\"status\"]}')
" 2>/dev/null || echo "    (parse error)"
    fi
  done
done
