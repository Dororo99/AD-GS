#!/usr/bin/env bash

# Prepare all selected Waymo scenes sequentially for the SplatAD protocol.
# The scenes must already have been converted under data/processed/waymo.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

if (( $# > 2 )); then
    echo "Usage: $0 [PROCESSED_ROOT] [PHYSICAL_GPU]" >&2
    exit 2
fi

PROCESSED_ROOT=${1:-${WAYMO_PROCESSED_ROOT:-data/processed/waymo}}
PHYSICAL_GPU=${2:-${WAYMO_PRIOR_GPU:-6}}

SCENES=(
    4986495627634617319_2980_000_3000_000
    4672649953433758614_2700_000_2720_000
    6791933003490312185_2607_000_2627_000
    17364342162691622478_780_000_800_000
    3385534893506316900_4252_000_4272_000
    9747453753779078631_940_000_960_000
    14940138913070850675_5755_330_5775_330
    204421859195625800_1080_000_1100_000
    7566697458525030390_1440_000_1460_000
    17159836069183024120_640_000_660_000
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
        echo "ONLY_SCENE is not in the configured Waymo scene list: ${ONLY_SCENE}" >&2
        exit 2
    fi
fi

for scene in "${SCENES[@]}"; do
    scene_path="${PROCESSED_ROOT}/${scene}"
    echo "[Waymo prior] scene=${scene} gpu=${PHYSICAL_GPU}"
    bash "${REPO_ROOT}/scripts/prepare_splatad_priors.sh" \
        waymo "${scene_path}" "${PHYSICAL_GPU}"
done

echo "[Waymo prior] completed ${#SCENES[@]} scene(s)"
