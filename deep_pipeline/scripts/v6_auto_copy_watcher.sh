#!/bin/bash
# V6 Wave 0 → Wave 1 pretrain checkpoint auto-copier
# Watches for Wave 0 pretrain completions and copies to Wave 1 run dirs.
# Run in background: nohup bash scripts/v6_auto_copy_watcher.sh &

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

WAVE1_CONFIGS=(
    configs/v6/sepsis_v6_sr_only.json
    configs/v6/sepsis_v6_sr_cosine.json
    configs/v6/sepsis_v6_sr_cosine_accum.json
    configs/v6/mortality_v6_sr_cosine.json
)

echo "[v6-watcher] Started at $(date). Watching for Wave 0 pretrain completions..."

while true; do
    all_done=true
    for cfg in "${WAVE1_CONFIGS[@]}"; do
        # Extract run_dir from config to check if pretrain already copied
        run_dir=$(python3 -c "import json; c=json.load(open('$cfg')); print(c['output']['run_dir'])")
        if [ -f "$run_dir/pretrain_checkpoint.pt" ]; then
            continue  # Already has pretrain
        fi
        all_done=false
        # Try to find and copy
        result=$(python scripts/manage_pretrain.py --auto-copy "$cfg" 2>&1)
        if echo "$result" | grep -q "Copied"; then
            echo "[v6-watcher] $(date): Copied pretrain for $cfg"
        fi
    done

    if $all_done; then
        echo "[v6-watcher] $(date): All Wave 1 pretrain checkpoints in place. Done!"
        break
    fi

    sleep 300  # Check every 5 minutes
done
