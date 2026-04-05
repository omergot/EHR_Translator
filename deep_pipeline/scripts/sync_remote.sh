#!/bin/bash
# Sync code + dependencies to remote servers.
#
# Usage:
#   ./scripts/sync_remote.sh              # sync to all servers (maria + 3090 + athena)
#   ./scripts/sync_remote.sh maria        # sync to A6000 server only
#   ./scripts/sync_remote.sh 3090         # sync to 3090 server only
#   ./scripts/sync_remote.sh athena       # sync to Athena cluster only
#   ./scripts/sync_remote.sh all          # sync to all servers
#
# This script handles items NOT managed by git worktrees:
#   - deep_pipeline code (main branch only — other branches use git worktrees)
#   - YAIB source code
#   - Gin config path fixes (pretrained_models/)
#   - Package reinstallation (pip install -e .)
#
# For branch-aware experiments, the scheduler creates git worktrees on the
# remote server automatically. This script is still needed for YAIB, pretrained
# models, and the initial deep_pipeline code sync.
set -euo pipefail

MARIA="omerg@132.68.39.40"
MARIA_BASE="/home/omerg/Thesis"

VISTA="omerg@132.68.35.177"
VISTA_BASE="/home/omerg/Thesis"

RSYNC_EXCLUDES=(
    --exclude='runs/'
    --exclude='.git/'
    --exclude='*.pptx'
    --exclude='*.excalidraw'
    --exclude='__pycache__/'
    --exclude='*.egg-info/'
    --exclude='experiments/logs/'
    --exclude='experiments/results/'
    --exclude='experiments/.remote_configs/'
)

sync_code_to() {
    local REMOTE="$1"
    local REMOTE_BASE="$2"
    local NAME="$3"

    echo "=== Syncing deep_pipeline code to $NAME ==="
    rsync -avz --delete "${RSYNC_EXCLUDES[@]}" \
      /bigdata/omerg/Thesis/EHR_Translator/deep_pipeline/ \
      "$REMOTE:$REMOTE_BASE/EHR_Translator/deep_pipeline/"

    echo ""
    echo "=== Syncing YAIB source to $NAME ==="
    rsync -avz \
      --exclude='.git/' \
      --exclude='eicu-crd*' \
      --exclude='__pycache__/' \
      /bigdata/omerg/Thesis/YAIB/ \
      "$REMOTE:$REMOTE_BASE/YAIB/"

    echo ""
    echo "=== Fixing gin config paths on $NAME ==="
    ssh "$REMOTE" "find $REMOTE_BASE/pretrained_models/ -name 'train_config.gin' \
      -exec grep -l '/bigdata/omerg/Thesis' {} \; | \
      xargs -r sed -i 's|/bigdata/omerg/Thesis|$REMOTE_BASE|g' && \
      echo 'Done (gin paths fixed).'"

    echo ""
    echo "=== Reinstalling packages on $NAME ==="
    ssh "$REMOTE" "source ~/miniforge3/etc/profile.d/conda.sh && conda activate yaib && \
      cd $REMOTE_BASE/EHR_Translator/deep_pipeline && pip install -e . -q && \
      cd $REMOTE_BASE/YAIB && pip install -e . -q && \
      echo 'Done.'"
}

sync_maria() {
    sync_code_to "$MARIA" "$MARIA_BASE" "maria (A6000)"
}

sync_3090() {
    sync_code_to "$VISTA" "$VISTA_BASE" "3090 (vista-pc15)"
}

sync_athena() {
    echo "=== Syncing to Athena (delegating to athena_sync.sh) ==="
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    "$SCRIPT_DIR/athena_sync.sh" code
}

TARGET="${1:-all}"

case "$TARGET" in
    maria)
        sync_maria
        ;;
    3090)
        sync_3090
        ;;
    athena)
        sync_athena
        ;;
    all)
        sync_maria
        echo ""
        echo "============================================"
        echo ""
        sync_3090
        echo ""
        echo "============================================"
        echo ""
        sync_athena
        ;;
    *)
        echo "Unknown target: $TARGET. Available: maria, 3090, athena, all"
        exit 1
        ;;
esac

echo ""
echo "Sync complete ($TARGET)."
