#!/bin/bash
# Shared SLURM log helpers for saturn launch scripts.

prepare_saturn_slurm_logs() {
  local step="$1"
  local script_dir="$2"

  SATURN_STEP="${step}"
  LOG_DIR="${LOG_DIR:-${script_dir}/logs}"
  mkdir -p "${LOG_DIR}"
  SUBMIT_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
  SLURM_LOG_OUT="${LOG_DIR}/${SATURN_STEP}-${SUBMIT_TIMESTAMP}-%j.out"
  SLURM_LOG_ERR="${LOG_DIR}/${SATURN_STEP}-${SUBMIT_TIMESTAMP}-%j.err"
}

# On a login node, sbatch this script with annotated log paths; on a compute node, no-op.
saturn_slurm_launch() {
  local step="$1"
  local script_path="$2"
  shift 2

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi

  local script_dir
  script_dir="$(cd "$(dirname "${script_path}")" && pwd)"
  # shellcheck disable=SC1091
  source "${script_dir}/pipeline_logs.sh"
  prepare_saturn_slurm_logs "${step}" "${script_dir}"

  export SATURN_STEP LOG_DIR SUBMIT_TIMESTAMP

  echo "Submitting saturn-${step}; logs -> ${LOG_DIR}/${SATURN_STEP}-${SUBMIT_TIMESTAMP}-<jobid>.{out,err}"
  exec sbatch \
    --job-name="saturn-${step}" \
    --output="${SLURM_LOG_OUT}" \
    --error="${SLURM_LOG_ERR}" \
    --export=ALL \
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
