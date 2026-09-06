#!/bin/bash
#SBATCH --job-name=support-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240g
#SBATCH --time=1-00:00:00
#SBATCH --output=support-test-%j.out

set -euo pipefail

# Submit from the SUPPORT repository, or override these locations with:
#   SUPPORT_REPO_DIR, SUPPORT_CONDA_SH, and SUPPORT_CONDA_ENV.
REPO_DIR="${SUPPORT_REPO_DIR:-${SLURM_SUBMIT_DIR:-/data/${USER}/SUPPORT}}"
CONDA_SH="${SUPPORT_CONDA_SH:-/data/${USER}/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${SUPPORT_CONDA_ENV:-SUPPORT}"

if [[ ! -f "${REPO_DIR}/scripts/biowulf_test.py" ]]; then
    echo "SUPPORT runner not found at ${REPO_DIR}/scripts/biowulf_test.py" >&2
    echo "Submit from the repository root or set SUPPORT_REPO_DIR." >&2
    exit 2
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda initialization script not found at ${CONDA_SH}" >&2
    echo "Set SUPPORT_CONDA_SH to your conda/etc/profile.d/conda.sh path." >&2
    exit 2
fi

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

exec python -u scripts/biowulf_test.py "$@"
