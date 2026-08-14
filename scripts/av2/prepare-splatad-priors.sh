#!/usr/bin/env bash

# Prepare all selected Argoverse 2 scenes sequentially for the SplatAD protocol.
# The scenes must already have been converted under data/processed/av2.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

if (( $# > 2 )); then
    echo "Usage: $0 [PROCESSED_ROOT] [PHYSICAL_GPU]" >&2
    exit 2
fi

PROCESSED_ROOT=${1:-${AV2_PROCESSED_ROOT:-data/processed/av2}}
PHYSICAL_GPU=${2:-${AV2_PRIOR_GPU:-2}}

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

if [[ -n "${ONLY_SCENE:-}" ]]; then
    found=0
    for scene in "${SCENES[@]}"; do
        if [[ "${scene}" == "${ONLY_SCENE}" ]]; then
            SCENES=("${scene}")
            found=1
            break
        fi
    done
    if (( ! found )); then
        echo "ONLY_SCENE is not in the configured AV2 scene list: ${ONLY_SCENE}" >&2
        exit 2
    fi
fi

for scene in "${SCENES[@]}"; do
    scene_path="${PROCESSED_ROOT}/${scene}"
    echo "[AV2 prior] scene=${scene} gpu=${PHYSICAL_GPU}"
    bash "${REPO_ROOT}/scripts/prepare_splatad_priors.sh" \
        av2 "${scene_path}" "${PHYSICAL_GPU}"
done

echo "[AV2 prior] completed ${#SCENES[@]} scene(s)"
