#!/usr/bin/env bash

# Finish the selected incomplete Waymo priors, then train all configured scenes.
# Every GPU stage is serialized on one physical GPU and training starts only
# after strict validation succeeds for all scenes.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PROCESSED_ROOT=${WAYMO_PROCESSED_ROOT:-data/processed/waymo}
OUTPUT_ROOT=${WAYMO_OUTPUT_ROOT:-output/waymo_splatad}
PHYSICAL_GPU=${WAYMO_PIPELINE_GPU:-0}
PYTHON_BIN=${ADGS_PYTHON:-/venv/ad-gs/bin/python}
PIPELINE_DRY_RUN=${PIPELINE_DRY_RUN:-0}
PIPELINE_LOG=${WAYMO_PIPELINE_LOG:-${OUTPUT_ROOT}/pipeline.log}
LOCK_FILE=${WAYMO_PIPELINE_LOCK:-${OUTPUT_ROOT}/pipeline.lock}

if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "WAYMO_PIPELINE_GPU must be a non-negative integer: ${PHYSICAL_GPU}" >&2
    exit 2
fi
if [[ ! "${PIPELINE_DRY_RUN}" =~ ^[01]$ ]]; then
    echo "PIPELINE_DRY_RUN must be 0 or 1: ${PIPELINE_DRY_RUN}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "AD-GS Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to prevent duplicate Waymo pipelines" >&2
    exit 2
fi

PRIOR_SCENES=(
    204421859195625800_1080_000_1100_000
    7566697458525030390_1440_000_1460_000
    17159836069183024120_640_000_660_000
)

print_command() {
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
}

scene_is_ready() {
    local scene_path=$1
    "${PYTHON_BIN}" scripts/validate_splatad_scene.py \
        "${scene_path}" --dataset waymo >/dev/null 2>&1
}

if (( PIPELINE_DRY_RUN )); then
    echo "[Waymo pipeline] dry-run on physical GPU ${PHYSICAL_GPU}"
    for scene in "${PRIOR_SCENES[@]}"; do
        scene_path="${PROCESSED_ROOT}/${scene}"
        if scene_is_ready "${scene_path}"; then
            echo "[Waymo pipeline][PRIOR READY] ${scene}"
        elif [[ -d "${scene_path}/.adgs-priors-work" ]]; then
            print_command env RESUME=1 bash scripts/prepare_splatad_priors.sh \
                waymo "${scene_path}" "${PHYSICAL_GPU}"
        else
            print_command bash scripts/prepare_splatad_priors.sh \
                waymo "${scene_path}" "${PHYSICAL_GPU}"
        fi
    done
    print_command env \
        "WAYMO_PROCESSED_ROOT=${PROCESSED_ROOT}" \
        "WAYMO_OUTPUT_ROOT=${OUTPUT_ROOT}" \
        "WAYMO_TRAIN_GPU=${PHYSICAL_GPU}" \
        bash scripts/train_waymo_splatad.sh
    exit 0
fi

mkdir -p "$(dirname -- "${PIPELINE_LOG}")" "$(dirname -- "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another Waymo pipeline holds ${LOCK_FILE}" >&2
    exit 1
fi
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

pipeline_start=${SECONDS}
on_exit() {
    local status=$?
    local elapsed=$((SECONDS - pipeline_start))
    trap - EXIT
    if (( status == 0 )); then
        echo "[Waymo pipeline][DONE] elapsed=${elapsed}s"
    else
        echo "[Waymo pipeline][FAILED] status=${status} elapsed=${elapsed}s log=${PIPELINE_LOG}" >&2
    fi
    exit "${status}"
}
trap on_exit EXIT

echo "============================================================"
echo "[Waymo pipeline][START] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "PID:          $$"
echo "GPU:          ${PHYSICAL_GPU} (all stages sequential)"
echo "Processed:    ${PROCESSED_ROOT}"
echo "Output:       ${OUTPUT_ROOT}"
echo "Log:          ${PIPELINE_LOG}"
echo "============================================================"

for ((index=0; index<${#PRIOR_SCENES[@]}; index++)); do
    scene=${PRIOR_SCENES[index]}
    scene_path="${PROCESSED_ROOT}/${scene}"
    echo "[Waymo pipeline][PRIOR $((index + 1))/${#PRIOR_SCENES[@]}][CHECK] ${scene}"
    if scene_is_ready "${scene_path}"; then
        echo "[Waymo pipeline][PRIOR $((index + 1))/${#PRIOR_SCENES[@]}][SKIP READY] ${scene}"
        continue
    fi

    if [[ -d "${scene_path}/.adgs-priors-work" ]]; then
        staged_size=$(du -sh "${scene_path}/.adgs-priors-work" | awk '{print $1}')
        echo "[Waymo pipeline][PRIOR $((index + 1))/${#PRIOR_SCENES[@]}][RESUME] ${scene} staged=${staged_size}"
        RESUME=1 bash scripts/prepare_splatad_priors.sh \
            waymo "${scene_path}" "${PHYSICAL_GPU}"
    else
        echo "[Waymo pipeline][PRIOR $((index + 1))/${#PRIOR_SCENES[@]}][START] ${scene}"
        bash scripts/prepare_splatad_priors.sh \
            waymo "${scene_path}" "${PHYSICAL_GPU}"
    fi

    if ! scene_is_ready "${scene_path}"; then
        echo "[Waymo pipeline][PRIOR $((index + 1))/${#PRIOR_SCENES[@]}][FAILED VALIDATION] ${scene}" >&2
        exit 1
    fi
    echo "[Waymo pipeline][PRIOR $((index + 1))/${#PRIOR_SCENES[@]}][DONE] ${scene}"
done

echo "[Waymo pipeline][PRIORS DONE] All three target scenes are training-ready."
echo "[Waymo pipeline][TRAIN START] Launching all 10 scenes sequentially on GPU ${PHYSICAL_GPU}."

WAYMO_PROCESSED_ROOT="${PROCESSED_ROOT}" \
WAYMO_OUTPUT_ROOT="${OUTPUT_ROOT}" \
WAYMO_TRAIN_GPU="${PHYSICAL_GPU}" \
bash scripts/train_waymo_splatad.sh
