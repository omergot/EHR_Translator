#!/bin/bash
# Sync code + data + dependencies to Athena GPU cluster.
#
# Usage:
#   ./scripts/athena_sync.sh          # sync everything (code + data + models)
#   ./scripts/athena_sync.sh code     # sync only code (fast)
#   ./scripts/athena_sync.sh data     # sync only cohort data + pretrained models
#
# Modeled on sync_remote.sh but adapted for Athena (SLURM cluster).
# Path mapping: /bigdata/omerg/Thesis → ~/Thesis (same as a6000/3090)
set -euo pipefail

ATHENA="omer.gotfrid@athena-login"
TARGET="${1:-all}"

sync_code() {
    echo "=== Syncing deep_pipeline code ==="
    rsync -avz --delete \
      --exclude='runs/' \
      --exclude='.git/' \
      --exclude='*.pptx' \
      --exclude='*.excalidraw' \
      --exclude='*.pdf' \
      --exclude='__pycache__/' \
      --exclude='*.egg-info/' \
      --exclude='experiments/logs/' \
      --exclude='experiments/results/' \
      --exclude='experiments/.remote_configs/' \
      --exclude='experiments/.athena_configs/' \
      --exclude='experiments/.athena_scripts/' \
      /bigdata/omerg/Thesis/EHR_Translator/deep_pipeline/ \
      "$ATHENA:~/Thesis/EHR_Translator/deep_pipeline/"

    echo ""
    echo "--- Syncing YAIB source ---"
    rsync -avz \
      --exclude='.git/' \
      --exclude='eicu-crd*' \
      --exclude='__pycache__/' \
      /bigdata/omerg/Thesis/YAIB/ \
      "$ATHENA:~/Thesis/YAIB/"

    echo ""
    echo "--- Fixing gin config paths on Athena ---"
    ssh "$ATHENA" 'find ~/Thesis/pretrained_models/ -name "train_config.gin" \
      -exec grep -l "/bigdata/omerg/Thesis" {} \; | \
      xargs -r sed -i "s|/bigdata/omerg/Thesis|$HOME/Thesis|g" && \
      echo "Done (gin paths fixed)."'

    echo ""
    echo "--- Reinstalling packages on Athena ---"
    ssh "$ATHENA" 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate yaib && \
      cd ~/Thesis/EHR_Translator/deep_pipeline && pip install -e . -q && \
      cd ~/Thesis/YAIB && pip install -e . -q && \
      echo "Done."'
}

sync_data() {
    echo "=== Syncing cohort data ==="
    rsync -avz --exclude='*cache*' \
      /bigdata/omerg/Thesis/cohort_data/ \
      "$ATHENA:~/Thesis/cohort_data/"

    echo ""
    echo "=== Syncing pretrained models ==="
    rsync -avz \
      /bigdata/omerg/Thesis/pretrained_models/ \
      "$ATHENA:~/Thesis/pretrained_models/"
}

case "$TARGET" in
    all)
        sync_code
        echo ""
        sync_data
        ;;
    code)
        sync_code
        ;;
    data)
        sync_data
        ;;
    *)
        echo "Unknown target: $TARGET. Available: all, code, data"
        exit 1
        ;;
esac

echo ""
echo "=== Athena sync complete ==="
