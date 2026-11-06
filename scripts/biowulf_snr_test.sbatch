#!/bin/bash
#SBATCH --job-name=support-snr-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch:200
#SBATCH --cpus-per-task=6
#SBATCH --mem=164g
#SBATCH --time=24:00:00
#SBATCH --output=support-snr-test-%j.out

set -euo pipefail

# Submit from the SUPPORT repository, or override these locations with:
#   SUPPORT_REPO_DIR, SUPPORT_CONDA_SH, and SUPPORT_CONDA_ENV.
REPO_DIR="${SUPPORT_REPO_DIR:-${SLURM_SUBMIT_DIR:-/data/${USER}/SUPPORT}}"
CONDA_SH="${SUPPORT_CONDA_SH:-/data/${USER}/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${SUPPORT_CONDA_ENV:-SUPPORT}"

if [[ ! -f "${REPO_DIR}/notebooks/snr_test.py" ]]; then
    echo "SUPPORT SNR runner not found at ${REPO_DIR}/notebooks/snr_test.py" >&2
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

SNR_SCRATCH="${JOB_SCRATCH}/support-snr-test"
install -d -m 700 "${SNR_SCRATCH}"
export TMPDIR="${SNR_SCRATCH}"
export SUPPORT_SNR_SCRATCH_DIR="${SNR_SCRATCH}"

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

exec python -u notebooks/snr_test.py \
    --repository "${REPO_DIR}" \
    --n-cpu "${SLURM_CPUS_PER_TASK}" \
    --no-train-pin-memory \
    --no-test-pin-memory \
    --slack-webhook-file /data/garfinkeljb/.secrets/support-slack-webhook \
    "$@"
