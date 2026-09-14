#!/bin/bash
#SBATCH --job-name=support-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch:200
#SBATCH --cpus-per-task=16
#SBATCH --mem=240g
#SBATCH --time=1-00:00:00
#SBATCH --output=support-train-%j.out

set -euo pipefail

# Submit from the SUPPORT repository, or override these locations with:
#   SUPPORT_REPO_DIR, SUPPORT_CONDA_SH, and SUPPORT_CONDA_ENV.
REPO_DIR="${SUPPORT_REPO_DIR:-${SLURM_SUBMIT_DIR:-/data/${USER}/SUPPORT}}"
CONDA_SH="${SUPPORT_CONDA_SH:-/data/${USER}/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${SUPPORT_CONDA_ENV:-SUPPORT}"

if [[ ! -f "${REPO_DIR}/scripts/biowulf_train.py" ]]; then
    echo "SUPPORT runner not found at ${REPO_DIR}/scripts/biowulf_train.py" >&2
    echo "Submit from the repository root or set SUPPORT_REPO_DIR." >&2
    exit 2
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda initialization script not found at ${CONDA_SH}" >&2
    echo "Set SUPPORT_CONDA_SH to your conda/etc/profile.d/conda.sh path." >&2
    exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM_JOB_ID is not set; run this script through sbatch." >&2
    exit 2
fi

JOB_SCRATCH="/lscratch/${SLURM_JOB_ID}"
if [[ ! -d "${JOB_SCRATCH}" || ! -w "${JOB_SCRATCH}" ]]; then
    echo "Allocated local scratch is unavailable or not writable: ${JOB_SCRATCH}" >&2
    echo "Request lscratch together with the GPU in one --gres option." >&2
    exit 2
fi

TRAINING_CACHE_DIR="${JOB_SCRATCH}/support-training"
install -d -m 700 "${TRAINING_CACHE_DIR}"
export TMPDIR="${TRAINING_CACHE_DIR}"
export SUPPORT_TRAINING_CACHE_DIR="${TRAINING_CACHE_DIR}"

module purge
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python -u scripts/biowulf_train.py "$@"
