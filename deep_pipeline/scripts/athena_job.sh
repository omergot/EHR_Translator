#!/bin/bash
# SLURM job template for Athena GPU cluster.
# This file is a TEMPLATE — athena_submit.py generates per-experiment scripts
# from this by replacing placeholders. Do not submit this directly.
#
# Placeholders (replaced by athena_submit.py):
#   __EXPNAME__    - experiment name
#   __WALLTIME__   - wall time (e.g., 1-00:00:00)
#   __QOS__        - QoS tier (e.g., 24h_1g)
#   __CONFIGPATH__ - path to remapped config on Athena
#   __OUTPUTPATH__ - path to output parquet on Athena
#   __ACCOUNT__    - SLURM account (e.g., aran_prj)
#   __PARTITIONS__ - SLURM partition(s), comma-separated (e.g., rtx6k-shared,l40s-shared)
#   __COMMAND__    - run.py subcommand (default: train_and_eval)
#
# Note: SBATCH directives use relative paths (resolved from submission dir).
# athena_submit.py submits with: cd $REPO && sbatch script.sh

#SBATCH --job-name=ehr___EXPNAME__
#SBATCH --account=__ACCOUNT__
#SBATCH --partition=__PARTITIONS__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=__WALLTIME__
#SBATCH --qos=__QOS__
#SBATCH --output=experiments/logs/athena_%x_%j.out
#SBATCH --error=experiments/logs/athena_%x_%j.err

# Activate environment — auto-detect based on GPU type
source ~/miniforge3/etc/profile.d/conda.sh

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if echo "$GPU_NAME" | grep -qi "Blackwell\|RTX PRO 6000\|H200\|H100"; then
    CONDA_ENV="yaib-cu128"
else
    CONDA_ENV="yaib"
fi
conda activate "$CONDA_ENV"

# Use conda's libstdc++ (compute nodes may have older system libs)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd ~/Thesis/EHR_Translator/deep_pipeline

# Ensure log directory exists
mkdir -p experiments/logs

echo "=== Athena Job Start ==="
echo "Experiment: __EXPNAME__"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Conda env: $CONDA_ENV"
echo "GPU: $GPU_NAME"
echo "PyTorch: $(python -c 'import torch; print(f"{torch.__version__}, CUDA {torch.version.cuda}")' 2>/dev/null)"
echo "Date: $(date)"
echo "Config: __CONFIGPATH__"
echo "========================"

# Run experiment (checkpoint resume is automatic via latest_checkpoint.pt)
python run.py __COMMAND__ \
  --config __CONFIGPATH__ \
  --output_parquet __OUTPUTPATH__

echo "=== Athena Job Complete ==="
echo "Date: $(date)"
