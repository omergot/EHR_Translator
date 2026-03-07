#!/bin/bash
# Remote experiment launcher — called by gpu_scheduler.py via SSH
# Usage: remote_launch.sh <gpu> <log_file> <config> <output_parquet> <conda_env> <repo_path>
# Outputs PID of the launched process to stdout.
set -e

GPU="$1"
LOG_FILE="$2"
CONFIG="$3"
OUTPUT="$4"
CONDA_ENV="${5:-yaib}"
REPO_PATH="${6:-/home/omerg/Thesis/EHR_Translator/deep_pipeline}"

# Activate conda
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Ensure directories exist
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$OUTPUT")"

cd "$REPO_PATH"

# Launch in a fully detached session
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPU EHR_LOG_FILE=$LOG_FILE python run.py train_and_eval --config $CONFIG --output_parquet $OUTPUT > $LOG_FILE 2>&1" < /dev/null &
PID=$!

# Wait briefly to ensure process started
sleep 0.5
if kill -0 "$PID" 2>/dev/null; then
    echo "$PID"
else
    echo "LAUNCH_FAILED" >&2
    exit 1
fi
