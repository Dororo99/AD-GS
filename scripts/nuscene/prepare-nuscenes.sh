#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RAW_ROOT="${1:-${REPO_ROOT}/data/nuscenes}"
PROCESSED_ROOT="${2:-${REPO_ROOT}/data/processed/nuscenes}"
PYTHON_BIN="${PYTHON_BIN:-/venv/ad-gs/bin/python}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
VALIDATE_FIRST="${VALIDATE_FIRST:-1}"
USE_COLOR="${USE_COLOR:-1}"
USE_DEPTH="${USE_DEPTH:-0}"
OVERWRITE="${OVERWRITE:-0}"
DOWNSAMPLE_RATIO="${DOWNSAMPLE_RATIO:-1.0}"

SCENES=(
  scene-0101
  scene-0689
  scene-0716
  scene-1096
  scene-0683
  scene-0758
  scene-1017
  scene-0100
  scene-0235
  scene-0252
)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${RAW_ROOT}/v1.0-trainval" ]]; then
  echo "nuScenes v1.0-trainval metadata not found under: ${RAW_ROOT}" >&2
  exit 1
fi

run_validate() {
  local scene="$1"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/nuscene.py" \
    "${RAW_ROOT}" "${PROCESSED_ROOT}" "${scene}" \
    --first_frame 0 \
    --last_frame -1 \
    --train_split_fraction 0.5 \
    --validate_only
}

if [[ "${VALIDATE_ONLY}" == "1" ]]; then
  for scene in "${SCENES[@]}"; do
    run_validate "${scene}"
  done
  exit 0
fi

mkdir -p "${PROCESSED_ROOT}"
for scene in "${SCENES[@]}"; do
  if [[ "${VALIDATE_FIRST}" == "1" ]]; then
    run_validate "${scene}"
  fi

  cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/nuscene.py"
    "${RAW_ROOT}" "${PROCESSED_ROOT}" "${scene}"
    --first_frame 0
    --last_frame -1
    --train_split_fraction 0.5
    --downsample_ratio "${DOWNSAMPLE_RATIO}"
  )
  if [[ "${USE_COLOR}" == "1" ]]; then
    cmd+=(--use_color)
  fi
  if [[ "${USE_DEPTH}" == "1" ]]; then
    cmd+=(--use_depth)
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"
done
