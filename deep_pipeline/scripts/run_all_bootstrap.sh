#!/bin/bash
# Run bootstrap CIs on all key experiments for NeurIPS paper
# CPU-only, ~minutes total

set -e
cd "$(dirname "$0")/.."

SCRIPT="python scripts/bootstrap_ci.py --n-replicates 500"
OUT_DIR="experiments/results/bootstrap_cis"
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Bootstrap CIs for NeurIPS 2026 paper"
echo "========================================"

# --- Best retrieval translators ---
echo ""
echo ">>> Best Retrieval: AKI (aki_v5_cross3)"
$SCRIPT runs/aki_v5_cross3/eval.predictions.npz \
    --original runs/aki_v5_cross3/eval.original.predictions.npz \
    2>&1 | tee "$OUT_DIR/ci_aki_v5_cross3.txt"

echo ""
echo ">>> Best Retrieval: Sepsis (sepsis_retr_v4_mmd)"
$SCRIPT runs/sepsis_retr_v4_mmd/eval.predictions.npz \
    --original runs/sepsis_retr_v4_mmd/eval.original.predictions.npz \
    2>&1 | tee "$OUT_DIR/ci_sepsis_retr_v4_mmd.txt"

echo ""
echo ">>> Best Retrieval: Mortality (mortality_retr_v5_cross3)"
$SCRIPT runs/mortality_retr_v5_cross3/eval.predictions.npz \
    2>&1 | tee "$OUT_DIR/ci_mortality_retr_v5_cross3.txt"

# --- LoS and KF (regression — check if bootstrap_ci.py handles regression) ---
if [ -f "runs/los_retr_v5_cross3/eval.predictions.npz" ]; then
    echo ""
    echo ">>> Best Retrieval: LoS (los_retr_v5_cross3)"
    $SCRIPT runs/los_retr_v5_cross3/eval.predictions.npz \
        2>&1 | tee "$OUT_DIR/ci_los_retr_v5_cross3.txt" || echo "  (may fail for regression)"
fi

if [ -f "runs/kf_retr_v5_cross3/eval.predictions.npz" ]; then
    echo ""
    echo ">>> Best Retrieval: KF (kf_retr_v5_cross3)"
    $SCRIPT runs/kf_retr_v5_cross3/eval.predictions.npz \
        2>&1 | tee "$OUT_DIR/ci_kf_retr_v5_cross3.txt" || echo "  (may fail for regression)"
fi

# --- DANN baselines ---
for task in mortality aki sepsis; do
    echo ""
    echo ">>> DANN: $task"
    $SCRIPT "runs/dann_$task/eval.predictions.npz" \
        2>&1 | tee "$OUT_DIR/ci_dann_$task.txt"
done

# --- CORAL baselines ---
for task in mortality aki sepsis; do
    echo ""
    echo ">>> CORAL: $task"
    $SCRIPT "runs/coral_$task/eval.predictions.npz" \
        2>&1 | tee "$OUT_DIR/ci_coral_$task.txt"
done

# --- CoDATS baselines ---
for task in mortality aki sepsis; do
    echo ""
    echo ">>> CoDATS: $task"
    $SCRIPT "runs/codats_$task/eval.predictions.npz" \
        2>&1 | tee "$OUT_DIR/ci_codats_$task.txt"
done

# --- Paired comparisons: best retrieval vs best DA baseline ---
echo ""
echo "========================================"
echo "Paired comparisons (Ours vs DANN)"
echo "========================================"

echo ""
echo ">>> AKI: aki_v5_cross3 vs dann_aki"
$SCRIPT runs/aki_v5_cross3/eval.predictions.npz \
    --compare runs/dann_aki/eval.predictions.npz \
    2>&1 | tee "$OUT_DIR/ci_paired_aki_ours_vs_dann.txt"

echo ""
echo ">>> Sepsis: sepsis_retr_v4_mmd vs dann_sepsis"
$SCRIPT runs/sepsis_retr_v4_mmd/eval.predictions.npz \
    --compare runs/dann_sepsis/eval.predictions.npz \
    2>&1 | tee "$OUT_DIR/ci_paired_sepsis_ours_vs_dann.txt"

echo ""
echo ">>> Mortality: mortality_retr_v5_cross3 vs dann_mortality"
$SCRIPT runs/mortality_retr_v5_cross3/eval.predictions.npz \
    --compare runs/dann_mortality/eval.predictions.npz \
    2>&1 | tee "$OUT_DIR/ci_paired_mortality_ours_vs_dann.txt"

echo ""
echo "========================================"
echo "Done! Results in: $OUT_DIR/"
echo "========================================"
