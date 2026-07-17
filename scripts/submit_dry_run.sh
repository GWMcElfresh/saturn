#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --signal=USR2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --partition=batch

# CPU dry-run for impac_tb_saturn.py (validate remap + gene–embedding overlap).
# Does not request a GPU. Sets SATURN_DRY_RUN=1.
# Submit from login node: bash scripts/submit_dry_run.sh
# Expect in log: gene_remap …, gene_overlap matched>0, dry_run_ok

export PROJECT_DIR="${PROJECT_DIR:-/home/exacloud/gscratch/prime-seq/Bimber/GW/saturn}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/pipeline_env.sh"
saturn_slurm_launch dry-run "${PROJECT_DIR}/scripts/submit_dry_run.sh"

NOTEBOOK="${NOTEBOOK:-impac_tb_saturn.py}"
WORKING_DIR="${WORKING_DIR:-${PROJECT_DIR}/work}"
HARMONIZED_DIR="${HARMONIZED_DIR:-${PROJECT_DIR}/outputs/harmonized/harmonized_outputs}"
SATURN_OUTPUT_DIR="${SATURN_OUTPUT_DIR:-${PROJECT_DIR}/saturn_outputs}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-${PROJECT_DIR}/data/protein_embeddings_export/ESM2}"
SATURN_EMBEDDING_MODEL="${SATURN_EMBEDDING_MODEL:-ESM2}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  tmpdir="${SLURM_TMPDIR}/ood-marimo-${SLURM_JOB_ID}"
else
  tmpdir="${PROJECT_DIR}/tmp/ood-marimo-${USER}-${SLURM_JOB_ID}"
fi
mkdir -p -m 700 "${tmpdir}/tmp"

export TMPDIR="${tmpdir}/tmp"
export WORKING_DIR HARMONIZED_DIR SATURN_OUTPUT_DIR EMBEDDINGS_DIR SATURN_EMBEDDING_MODEL
export SATURN_DRY_RUN=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_JOB_CPUS_PER_NODE:-4}}"
export MPLBACKEND=Agg

cd "${PROJECT_DIR}"
mkdir -p "${WORKING_DIR}"
if [[ "${SATURN_OUTPUT_DIR}" == /* ]]; then
  mkdir -p "${SATURN_OUTPUT_DIR}"
else
  mkdir -p "${WORKING_DIR}/${SATURN_OUTPUT_DIR#./}"
fi

saturn_sync_venv "${PROJECT_DIR}" || exit 1
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "ERROR: No Python venv found at ${VENV_DIR} after uv sync." >&2
  exit 1
fi

saturn_log_banner "SATURN_DRY_RUN"
echo "SATURN_DRY_RUN: mode=script (python ${NOTEBOOK})"
echo "SATURN_DRY_RUN: SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SATURN_DRY_RUN: SATURN_STEP=${SATURN_STEP:-dry-run}"
echo "SATURN_DRY_RUN: PROJECT_DIR=${PROJECT_DIR}"
echo "SATURN_DRY_RUN: NOTEBOOK=${NOTEBOOK}"
echo "SATURN_DRY_RUN: WORKING_DIR=${WORKING_DIR}"
echo "SATURN_DRY_RUN: SATURN_OUTPUT_DIR=${SATURN_OUTPUT_DIR}"
echo "SATURN_DRY_RUN: HARMONIZED_DIR=${HARMONIZED_DIR}"
echo "SATURN_DRY_RUN: EMBEDDINGS_DIR=${EMBEDDINGS_DIR}"
echo "SATURN_DRY_RUN: SATURN_EMBEDDING_MODEL=${SATURN_EMBEDDING_MODEL}"
echo "SATURN_DRY_RUN: expect gene_remap, gene_overlap matched>0, dry_run_ok"

exec python "${NOTEBOOK}"
