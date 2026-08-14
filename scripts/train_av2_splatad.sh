#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

GPU=${AV2_TRAIN_GPU:-5}
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
    echo "AV2_TRAIN_GPU must be a non-negative integer: ${GPU}" >&2
    exit 2
fi
export ADGS_MIN_FREE_GIB="${ADGS_MIN_FREE_GIB:-${AV2_MIN_FREE_GIB:-100}}"

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

exec "${SCRIPT_DIR}/train_splatad_split.sh" \
    av2 arguments/av2.py \
    "${AV2_PROCESSED_ROOT:-data/processed/av2}" \
    "${AV2_OUTPUT_ROOT:-output/av2_splatad}" \
    "${GPU}" "${GPU}" "${SCENES[@]}"
