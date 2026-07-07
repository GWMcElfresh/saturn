#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --signal=USR2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Headless batch run for impac_tb_saturn.py (SATURN ImpacTB workflow).
# Uses marimo script mode: python notebook.py (executes all cells and exits).
# Do NOT use marimo run here — that starts a web server and waits for a browser.
# Submit: bash submit_run.sh  (creates logs/ and writes logs/batch-<timestamp>-<jobid>.out)

export PROJECT_DIR="${PROJECT_DIR:-/home/exacloud/gscratch/prime-seq/Bimber/GW/saturn}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/pipeline_env.sh"
saturn_slurm_launch batch "${PROJECT_DIR}/submit_run.sh"

NOTEBOOK="${NOTEBOOK:-impac_tb_saturn.py}"
WORKING_DIR="${WORKING_DIR:-${PROJECT_DIR}/work}"
HARMONIZED_DIR="${HARMONIZED_DIR:-${PROJECT_DIR}/outputs/harmonized/harmonized_outputs}"
SATURN_OUTPUT_DIR="${SATURN_OUTPUT_DIR:-${PROJECT_DIR}/saturn_outputs}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

# Writable temp: compute-node scratch when available, else project gscratch
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  tmpdir="${SLURM_TMPDIR}/ood-marimo-${SLURM_JOB_ID}"
else
  tmpdir="${PROJECT_DIR}/tmp/ood-marimo-${USER}-${SLURM_JOB_ID}"
fi
mkdir -p -m 700 "${tmpdir}/tmp"

export TMPDIR="${tmpdir}/tmp"
export WORKING_DIR HARMONIZED_DIR SATURN_OUTPUT_DIR
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_JOB_CPUS_PER_NODE:-8}}"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${PROJECT_DIR}"
mkdir -p "${WORKING_DIR}"
if [[ "${SATURN_OUTPUT_DIR}" == /* ]]; then
  mkdir -p "${SATURN_OUTPUT_DIR}"
else
  mkdir -p "${WORKING_DIR}/${SATURN_OUTPUT_DIR#./}"
fi

# Sync and activate Python environment (.venv from uv.lock)
saturn_sync_venv "${PROJECT_DIR}" || exit 1
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "ERROR: No Python venv found at ${VENV_DIR} after uv sync." >&2
  exit 1
fi

saturn_log_banner "SATURN_BATCH"
echo "SATURN_BATCH: mode=script (python ${NOTEBOOK})"
echo "SATURN_BATCH: SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SATURN_BATCH: SATURN_STEP=${SATURN_STEP:-batch}"
echo "SATURN_BATCH: PROJECT_DIR=${PROJECT_DIR}"
echo "SATURN_BATCH: NOTEBOOK=${NOTEBOOK}"
echo "SATURN_BATCH: WORKING_DIR=${WORKING_DIR}"
echo "SATURN_BATCH: SATURN_OUTPUT_DIR=${SATURN_OUTPUT_DIR}"
echo "SATURN_BATCH: HARMONIZED_DIR=${HARMONIZED_DIR}"
echo "SATURN_BATCH: expect SATURN_IMPACTB: lines in log; artifacts under resolved SATURN_OUTPUT_DIR"
echo "SATURN_BATCH: post-train outputs: umap_species.png cell_clusters.tsv macrogene_weights.tsv"

exec python "${NOTEBOOK}"
