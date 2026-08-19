#!/bin/bash
#SBATCH --job-name=support-generalized
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=2-00:00:00
#SBATCH --output=support-%x-%j.out

set -euo pipefail

# Adjust these paths for your Biowulf account.
REPO_DIR="/data/${USER}/SUPPORT"
INPUT_DIR="/data/${USER}/support_training"
OUTPUT_DIR="/data/${USER}/support_models"
START_EPOCH="${START_EPOCH:-0}"

module purge
source "/data/${USER}/conda/etc/profile.d/conda.sh"
conda activate SUPPORT

cd "${REPO_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Memory mapping is safer for a large multi-file training directory. Remove
# --no-load-to-memory below if the complete dataset comfortably fits in RAM.
python -u run.py \
  --in-path "${INPUT_DIR}" \
  --out-dir "${OUTPUT_DIR}" \
  --exp-name generalized \
  --dimensionality "2+1D" \
  --n-epochs 100 \
  --epoch "${START_EPOCH}" \
  --n-cpu "${SLURM_CPUS_PER_TASK:-2}" \
  --prefetch-factor 2 \
  --no-load-to-memory

# To resume after model_36.pth was saved:
# sbatch --export=ALL,START_EPOCH=37 scripts/biowulf_train.sbatch
