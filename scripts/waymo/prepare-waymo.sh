#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

RAW_ROOT=${1:-data/waymo}
PROCESSED_ROOT=${2:-data/processed/waymo}
PYTHON_BIN=${WAYMO_PYTHON:-/venv/ad-gs-waymo/bin/python}

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

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Waymo Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
mkdir -p "${PROCESSED_ROOT}"

for scene in "${SCENES[@]}"; do
    if [[ -n "${ONLY_SCENE:-}" && "${scene}" != "${ONLY_SCENE}" ]]; then
        continue
    fi
    record="${RAW_ROOT}/training/segment-${scene}_with_camera_labels.tfrecord"
    destination="${PROCESSED_ROOT}/${scene}"
    if [[ ! -f "${record}" ]]; then
        echo "Missing Waymo TFRecord: ${record}" >&2
        exit 1
    fi
    if [[ -d "${destination}" && -n "$(find "${destination}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing non-empty destination: ${destination}" >&2
        exit 1
    fi
    echo "[Waymo] ${scene}"
    "${PYTHON_BIN}" scripts/waymo/waymo.py \
        "${record}" "${destination}" \
        --part training --first_frame 0 --last_frame -1 \
        --select_camera 0 1 2 --train_split_fraction 0.5 --use_color
done
