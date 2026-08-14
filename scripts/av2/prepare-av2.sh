#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

VALIDATE_ONLY=0
if [[ "${1:-}" == "--validate_only" ]]; then
    VALIDATE_ONLY=1
    shift
fi

SRC_ROOT=${1:-./data/av2}
DST_ROOT=${2:-./data/processed/av2}
ONLY_SCENE=${3:-}
PYTHON_BIN=${AV2_PYTHON:-/venv/camosplat/bin/python}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "AV2 Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi

SCENES=(
    "a7bcdabb-f9b7-3c16-806d-3ddf1c2d49a2"
    "76c3f58f-9003-3bdb-90a3-b87cfbfa1c3b"
    "5f2b8881-3447-3905-99f8-def9d72aae42"
    "d201af7e-48c8-34ad-be1c-e649af2cb5c2"
    "4d9e3bdf-7216-3161-8281-72863f3c2bf6"
    "38f30522-2d43-3ff3-a94b-84887ab1671d"
    "756f4ed0-5352-31e4-b3c6-2841b9e779d7"
    "91cded81-9f72-3930-bab7-5d3e3fa0a220"
    "511b93af-f16e-3195-8628-fbb972a17f74"
    "f5a3ee79-a131-3f8a-91e9-a6475d778149"
)

if [[ -n "${ONLY_SCENE}" ]]; then
    SCENES=("${ONLY_SCENE}")
fi

for scene in "${SCENES[@]}"; do
    args=(
        "scripts/av2/av2.py"
        "${SRC_ROOT}"
        "${DST_ROOT}"
        "${scene}"
        "--split"
        "train"
        "--train_fraction"
        "0.5"
    )
    if (( VALIDATE_ONLY )); then
        args+=("--validate_only")
    fi
    echo "[AV2] scene=${scene} validate_only=${VALIDATE_ONLY}"
    "${PYTHON_BIN}" "${args[@]}"
done
