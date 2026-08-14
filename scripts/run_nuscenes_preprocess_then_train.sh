#!/usr/bin/env bash

# nuScenes: build priors and train all selected scenes sequentially on GPU 4.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

PROCESSED_ROOT=${NUSCENES_PROCESSED_ROOT:-${REPO_ROOT}/data/processed/nuscenes}
OUTPUT_ROOT=${NUSCENES_OUTPUT_ROOT:-${REPO_ROOT}/output/nuscenes_splatad}
PHYSICAL_GPU=${NUSCENES_PIPELINE_GPU:-4}

SCENES=(
    scene-0101 scene-0689 scene-0716 scene-1096 scene-0683
    scene-0758 scene-1017 scene-0100 scene-0235 scene-0252
)

exec env \
    "PIPELINE_DRY_RUN=${NUSCENES_PIPELINE_DRY_RUN:-${PIPELINE_DRY_RUN:-${DRY_RUN:-0}}}" \
    "PIPELINE_LOG=${NUSCENES_PIPELINE_LOG:-${OUTPUT_ROOT}/pipeline.log}" \
    "PIPELINE_LOCK=${NUSCENES_PIPELINE_LOCK:-${OUTPUT_ROOT}/pipeline.lock}" \
    "PIPELINE_MIN_FREE_GIB=${NUSCENES_MIN_FREE_GIB:-100}" \
    "KEEP_PRIOR_WORK=${NUSCENES_KEEP_PRIOR_WORK:-0}" \
    "ADGS_PRIOR_BUILDER_LOCK=${ADGS_PRIOR_BUILDER_LOCK:-${REPO_ROOT}/output/.adgs-prior-builder.gpu-${PHYSICAL_GPU}.lock}" \
    "${SCRIPT_DIR}/run_splatad_preprocess_then_train.sh" \
    nuscenes "${REPO_ROOT}/arguments/nuscenes.py" \
    "${PROCESSED_ROOT}" "${OUTPUT_ROOT}" "${PHYSICAL_GPU}" \
    "${SCENES[@]}"
