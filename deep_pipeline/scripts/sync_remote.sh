#!/bin/bash
# Sync code + dependencies to remote A6000 server (maria).
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

REMOTE="omerg@132.68.39.40"
REMOTE_BASE="/home/omerg/Thesis"

echo "=== Syncing deep_pipeline code ==="
rsync -avz --delete \
  --exclude='runs/' \
  --exclude='.git/' \
  --exclude='*.pptx' \
  --exclude='*.excalidraw' \
  --exclude='__pycache__/' \
  --exclude='*.egg-info/' \
  --exclude='experiments/logs/' \
  --exclude='experiments/results/' \
  --exclude='experiments/.remote_configs/' \
  /bigdata/omerg/Thesis/EHR_Translator/deep_pipeline/ \
  $REMOTE:$REMOTE_BASE/EHR_Translator/deep_pipeline/

echo ""
echo "=== Syncing YAIB source ==="
rsync -avz \
  --exclude='.git/' \
  --exclude='eicu-crd*' \
  --exclude='__pycache__/' \
  /bigdata/omerg/Thesis/YAIB/ \
  $REMOTE:$REMOTE_BASE/YAIB/

echo ""
echo "=== Fixing gin config paths on remote ==="
ssh $REMOTE 'find /home/omerg/Thesis/pretrained_models/ -name "train_config.gin" \
  -exec grep -l "/bigdata/omerg/Thesis" {} \; | \
  xargs -r sed -i "s|/bigdata/omerg/Thesis|/home/omerg/Thesis|g" && \
  echo "Done (gin paths fixed)."'

echo ""
echo "=== Reinstalling packages on remote ==="
ssh $REMOTE 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate yaib && \
  cd /home/omerg/Thesis/EHR_Translator/deep_pipeline && pip install -e . -q && \
  cd /home/omerg/Thesis/YAIB && pip install -e . -q && \
  echo "Done."'

echo ""
echo "Sync complete."
