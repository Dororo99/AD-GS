#!/usr/bin/env bash

# AV2: build priors and train all selected scenes sequentially on GPU 5.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

PROCESSED_ROOT=${AV2_PROCESSED_ROOT:-${REPO_ROOT}/data/processed/av2}
OUTPUT_ROOT=${AV2_OUTPUT_ROOT:-${REPO_ROOT}/output/av2_splatad}
PHYSICAL_GPU=${AV2_PIPELINE_GPU:-5}

SCENES=(
    a7bcdabb-f9b7-3c16-806d-3ddf1c2d49a2
    76c3f58f-9003-3bdb-90a3-b87cfbfa1c3b
    5f2b8881-3447-3905-99f8-def9d72aae42
    d201af7e-48c8-34ad-be1c-e649af2cb5c2
    4d9e3bdf-7216-3161-8281-72863f3c2bf6
    38f30522-2d43-3ff3-a94b-84887ab1671d
    756f4ed0-5352-31e4-b3c6-2841b9e779d7
    91cded81-9f72-3930-bab7-5d3e3fa0a220
    511b93af-f16e-3195-8628-fbb972a17f74
    f5a3ee79-a131-3f8a-91e9-a6475d778149
)

exec env \
    "PIPELINE_DRY_RUN=${AV2_PIPELINE_DRY_RUN:-${PIPELINE_DRY_RUN:-${DRY_RUN:-0}}}" \
    "PIPELINE_LOG=${AV2_PIPELINE_LOG:-${OUTPUT_ROOT}/pipeline.log}" \
    "PIPELINE_LOCK=${AV2_PIPELINE_LOCK:-${OUTPUT_ROOT}/pipeline.lock}" \
    "PIPELINE_MIN_FREE_GIB=${AV2_MIN_FREE_GIB:-100}" \
    "KEEP_PRIOR_WORK=${AV2_KEEP_PRIOR_WORK:-0}" \
    "ADGS_PRIOR_BUILDER_LOCK=${ADGS_PRIOR_BUILDER_LOCK:-${REPO_ROOT}/output/.adgs-prior-builder.gpu-${PHYSICAL_GPU}.lock}" \
    "${SCRIPT_DIR}/run_splatad_preprocess_then_train.sh" \
    av2 "${REPO_ROOT}/arguments/av2.py" \
    "${PROCESSED_ROOT}" "${OUTPUT_ROOT}" "${PHYSICAL_GPU}" \
    "${SCENES[@]}"
