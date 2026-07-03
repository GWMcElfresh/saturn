#!/bin/bash
# Shared paths and SLURM helpers for saturn launch scripts.
# Source via: source "${PROJECT_DIR}/scripts/pipeline_env.sh"
# (Never use BASH_SOURCE for this path — SLURM copies job scripts to /var/spool/slurmd/.)

: "${PROJECT_DIR:=/home/exacloud/gscratch/prime-seq/Bimber/GW/saturn}"
export PROJECT_DIR

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

prepare_saturn_slurm_logs() {
  local step="$1"
  local project_dir="${2:-${PROJECT_DIR}}"

  SATURN_STEP="${step}"
  LOG_DIR="${LOG_DIR:-${project_dir}/logs}"
  mkdir -p "${LOG_DIR}"
  SUBMIT_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
  SLURM_LOG_OUT="${LOG_DIR}/${SATURN_STEP}-${SUBMIT_TIMESTAMP}-%j.out"
  SLURM_LOG_ERR="${LOG_DIR}/${SATURN_STEP}-${SUBMIT_TIMESTAMP}-%j.err"
}

# On a login node, sbatch the repo script with annotated log paths; on a compute node, no-op.
saturn_slurm_launch() {
  local step="$1"
  local script_path="$2"
  shift 2

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi

  prepare_saturn_slurm_logs "${step}" "${PROJECT_DIR}"

  export PROJECT_DIR SATURN_STEP LOG_DIR SUBMIT_TIMESTAMP

  echo "Submitting saturn-${step}; logs -> ${LOG_DIR}/${SATURN_STEP}-${SUBMIT_TIMESTAMP}-<jobid>.{out,err}"
  exec sbatch \
    --job-name="saturn-${step}" \
    --output="${SLURM_LOG_OUT}" \
    --error="${SLURM_LOG_ERR}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    "$@" \
    "${script_path}"
}

saturn_log_banner() {
  local prefix="$1"
  if [[ -n "${SUBMIT_TIMESTAMP:-}" && -n "${SATURN_STEP:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "${prefix}: log=${LOG_DIR}/${SATURN_STEP}-${SUBMIT_TIMESTAMP}-${SLURM_JOB_ID}.out"
    echo "${prefix}: err=${LOG_DIR}/${SATURN_STEP}-${SUBMIT_TIMESTAMP}-${SLURM_JOB_ID}.err"
  fi
}

# Install/update .venv from pyproject.toml + uv.lock before each job run.
saturn_sync_venv() {
  local project_dir="${1:-${PROJECT_DIR}}"

  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found on PATH. Install uv or add it to PATH before submitting jobs." >&2
    return 1
  fi

  echo "SATURN: syncing Python environment (uv sync) in ${project_dir}..."
  (cd "${project_dir}" && uv sync) || {
    echo "ERROR: uv sync failed in ${project_dir}" >&2
    return 1
  }
}
