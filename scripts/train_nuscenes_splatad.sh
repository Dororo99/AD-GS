#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

GPU=${NUSCENES_TRAIN_GPU:-4}
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
    echo "NUSCENES_TRAIN_GPU must be a non-negative integer: ${GPU}" >&2
    exit 2
fi
export ADGS_MIN_FREE_GIB="${ADGS_MIN_FREE_GIB:-${NUSCENES_MIN_FREE_GIB:-100}}"

SCENES=(
    scene-0101 scene-0689 scene-0716 scene-1096 scene-0683
    scene-0758 scene-1017 scene-0100 scene-0235 scene-0252
)

exec "${SCRIPT_DIR}/train_splatad_split.sh" \
    nuscenes arguments/nuscenes.py \
    "${NUSCENES_PROCESSED_ROOT:-data/processed/nuscenes}" \
    "${NUSCENES_OUTPUT_ROOT:-output/nuscenes_splatad}" \
    "${GPU}" "${GPU}" "${SCENES[@]}"
