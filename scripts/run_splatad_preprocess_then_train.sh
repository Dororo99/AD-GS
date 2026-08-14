#!/usr/bin/env bash

# Build all AD-GS priors for one selected-scene dataset and then train every
# scene on one physical GPU. A GPU-scoped lock prevents duplicate builders on
# the same physical GPU while allowing independent GPUs to preprocess in
# parallel.

set -Eeuo pipefail

if (( $# < 6 )); then
    echo "Usage: $0 {nuscenes|av2} CONFIG PROCESSED_ROOT OUTPUT_ROOT PHYSICAL_GPU SCENE..." >&2
    exit 2
fi

DATASET=$1
CONFIG=$2
PROCESSED_ROOT=$3
OUTPUT_ROOT=$4
PHYSICAL_GPU=$5
shift 5
SCENES=("$@")

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

case "${DATASET}" in
    nuscenes) PIPELINE_LABEL=nuScenes ;;
    av2) PIPELINE_LABEL=AV2 ;;
    *) echo "Unsupported pipeline dataset: ${DATASET}" >&2; exit 2 ;;
esac

PYTHON_BIN=${ADGS_PYTHON:-/venv/ad-gs/bin/python}
PIPELINE_DRY_RUN=${PIPELINE_DRY_RUN:-0}
KEEP_PRIOR_WORK=${KEEP_PRIOR_WORK:-0}
PIPELINE_LOG=${PIPELINE_LOG:-${OUTPUT_ROOT}/pipeline.log}
DATASET_LOCK=${PIPELINE_LOCK:-${OUTPUT_ROOT}/pipeline.lock}
PRIOR_BUILDER_LOCK=${ADGS_PRIOR_BUILDER_LOCK:-${REPO_ROOT}/output/.adgs-prior-builder.gpu-${PHYSICAL_GPU}.lock}
MIN_FREE_GIB=${PIPELINE_MIN_FREE_GIB:-100}
PRIOR_LOCK_WAIT_SECONDS=${PRIOR_LOCK_WAIT_SECONDS:-30}

if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "PHYSICAL_GPU must be a non-negative integer: ${PHYSICAL_GPU}" >&2
    exit 2
fi
for boolean_value in "${PIPELINE_DRY_RUN}" "${KEEP_PRIOR_WORK}"; do
    if [[ ! "${boolean_value}" =~ ^[01]$ ]]; then
        echo "PIPELINE_DRY_RUN and KEEP_PRIOR_WORK must be 0 or 1: ${boolean_value}" >&2
        exit 2
    fi
done
for integer_value in "${MIN_FREE_GIB}" "${PRIOR_LOCK_WAIT_SECONDS}"; do
    if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
        echo "Disk floor and lock wait must be non-negative integers: ${integer_value}" >&2
        exit 2
    fi
done
if (( PRIOR_LOCK_WAIT_SECONDS < 1 )); then
    echo "PRIOR_LOCK_WAIT_SECONDS must be at least 1" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "AD-GS Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 2
fi
if [[ ! -d "${PROCESSED_ROOT}" ]]; then
    echo "Missing processed root: ${PROCESSED_ROOT}" >&2
    exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to prevent duplicate pipelines" >&2
    exit 2
fi
for scene in "${SCENES[@]}"; do
    if [[ ! -d "${PROCESSED_ROOT}/${scene}" ]]; then
        echo "Missing converted scene: ${PROCESSED_ROOT}/${scene}" >&2
        exit 2
    fi
done

print_command() {
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
}

scene_is_ready() {
    local scene_path=$1
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_splatad_scene.py" \
        "${scene_path}" --dataset "${DATASET}" >/dev/null
}

scene_likely_ready() {
    local scene_path=$1
    [[ -d "${scene_path}/depth" &&
       -d "${scene_path}/semantic" &&
       -d "${scene_path}/sky" &&
       -d "${scene_path}/flow" &&
       -f "${scene_path}/colmap.ply" &&
       -f "${scene_path}/points3d.ply" ]]
}

available_space_gib() {
    local available_kib
    available_kib=$(df -Pk -- "${REPO_ROOT}" | awk 'NR == 2 {print $4}')
    if [[ ! "${available_kib}" =~ ^[0-9]+$ ]]; then
        echo "[${PIPELINE_LABEL} pipeline][DISK FAILED] Unable to read free space for ${REPO_ROOT}" >&2
        return 1
    fi
    echo $((available_kib / 1024 / 1024))
}

check_disk_free() {
    local phase=$1
    local free_gib
    free_gib=$(available_space_gib)
    echo "[${PIPELINE_LABEL} pipeline][DISK] phase=${phase} available=${free_gib}GiB minimum=${MIN_FREE_GIB}GiB"
    if (( free_gib < MIN_FREE_GIB )); then
        echo "[${PIPELINE_LABEL} pipeline][STOP LOW DISK] phase=${phase} available=${free_gib}GiB minimum=${MIN_FREE_GIB}GiB" >&2
        return 1
    fi
}

print_prepare_command() {
    local scene_path=$1
    local resume=$2
    print_command env \
        "ADGS_PYTHON=${PYTHON_BIN}" \
        "PRIOR_WORK_ROOT=${scene_path}/.adgs-priors-work" \
        OVERWRITE=0 "RESUME=${resume}" DRY_RUN=0 \
        "KEEP_WORK=${KEEP_PRIOR_WORK}" \
        bash "${SCRIPT_DIR}/prepare_splatad_priors.sh" \
        "${DATASET}" "${scene_path}" "${PHYSICAL_GPU}"
}

if (( PIPELINE_DRY_RUN )); then
    echo "============================================================"
    echo "[${PIPELINE_LABEL} pipeline][DRY-RUN]"
    echo "GPU:          ${PHYSICAL_GPU} (all stages sequential)"
    echo "Scenes:       ${#SCENES[@]}"
    echo "Processed:    ${PROCESSED_ROOT}"
    echo "Output:       ${OUTPUT_ROOT}"
    echo "Dataset lock: ${DATASET_LOCK} (not acquired)"
    echo "GPU lock:     ${PRIOR_BUILDER_LOCK} (not acquired)"
    echo "Disk floor:   ${MIN_FREE_GIB}GiB"
    echo "============================================================"
    for scene in "${SCENES[@]}"; do
        scene_path="${PROCESSED_ROOT}/${scene}"
        print_command "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_splatad_scene.py" \
            "${scene_path}" --dataset "${DATASET}"
        if scene_likely_ready "${scene_path}"; then
            echo "[${PIPELINE_LABEL} pipeline][DRY-RUN][LIKELY READY] ${scene}; runtime strict validation decides the skip"
        elif [[ -d "${scene_path}/.adgs-priors-work" ]]; then
            print_prepare_command "${scene_path}" 1
        else
            print_prepare_command "${scene_path}" 0
        fi
    done
    print_command env \
        "ADGS_PYTHON=${PYTHON_BIN}" \
        "ADGS_MIN_FREE_GIB=${MIN_FREE_GIB}" \
        DRY_RUN=1 ALLOW_EXISTING_OUTPUT=0 ONLY_SCENE= \
        bash "${SCRIPT_DIR}/train_splatad_split.sh" \
        "${DATASET}" "${CONFIG}" "${PROCESSED_ROOT}" "${OUTPUT_ROOT}" \
        "${PHYSICAL_GPU}" "${PHYSICAL_GPU}" "${SCENES[@]}"
    exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
        echo "Physical GPU ${PHYSICAL_GPU} is not available" >&2
        exit 2
    fi
fi

mkdir -p \
    "$(dirname -- "${PIPELINE_LOG}")" \
    "$(dirname -- "${DATASET_LOCK}")" \
    "$(dirname -- "${PRIOR_BUILDER_LOCK}")"

exec 9>"${DATASET_LOCK}"
if ! flock -n 9; then
    echo "Another ${PIPELINE_LABEL} pipeline holds ${DATASET_LOCK}" >&2
    exit 1
fi

exec > >(tee -a "${PIPELINE_LOG}") 2>&1
pipeline_start=$SECONDS

on_exit() {
    local status=$?
    local elapsed=$((SECONDS - pipeline_start))
    trap - EXIT INT TERM
    flock -u 9 || true

    if (( status == 0 )); then
        echo "[${PIPELINE_LABEL} pipeline][DONE] elapsed=${elapsed}s"
    else
        echo "[${PIPELINE_LABEL} pipeline][FAILED] status=${status} elapsed=${elapsed}s log=${PIPELINE_LOG}" >&2
    fi
    exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "============================================================"
echo "[${PIPELINE_LABEL} pipeline][START] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "PID:          $$"
echo "GPU:          ${PHYSICAL_GPU} (one scene at a time)"
echo "Scenes:       ${#SCENES[@]}"
echo "Processed:    ${PROCESSED_ROOT}"
echo "Output:       ${OUTPUT_ROOT}"
echo "Dataset lock: ${DATASET_LOCK}"
echo "GPU lock:     ${PRIOR_BUILDER_LOCK} (scene-scoped)"
echo "Disk floor:   ${MIN_FREE_GIB}GiB"
echo "Log:          ${PIPELINE_LOG}"
echo "============================================================"

check_disk_free pipeline-start

process_scene() {
    local index=$1
    local scene=$2
    local position=$((index + 1))
    local scene_path="${PROCESSED_ROOT}/${scene}"
    local work_root="${scene_path}/.adgs-priors-work"
    local staged_size

    echo "[${PIPELINE_LABEL} pipeline][PRIOR ${position}/${#SCENES[@]}][STRICT CHECK] ${scene}"
    if scene_is_ready "${scene_path}"; then
        echo "[${PIPELINE_LABEL} pipeline][PRIOR ${position}/${#SCENES[@]}][SKIP READY] ${scene}"
        if [[ -d "${work_root}" ]]; then
            echo "[${PIPELINE_LABEL} pipeline][WARNING] Ready scene retains staging; left untouched: ${work_root}"
        fi
        return 0
    fi

    (
        exec 8>"${PRIOR_BUILDER_LOCK}"
        echo "[${PIPELINE_LABEL} pipeline][PRIOR ${position}/${#SCENES[@]}][LOCK WAIT] ${scene}"
        while ! flock -w "${PRIOR_LOCK_WAIT_SECONDS}" 8; do
            echo "[${PIPELINE_LABEL} pipeline][PRIOR ${position}/${#SCENES[@]}][LOCK WAITING] ${scene} time=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        done
        echo "[${PIPELINE_LABEL} pipeline][PRIOR ${position}/${#SCENES[@]}][LOCK ACQUIRED] ${scene}"

        check_disk_free "prior:${scene}"
        if [[ -d "${work_root}" ]]; then
            staged_size=$(du -sh "${work_root}" 2>/dev/null | awk 'NR == 1 {print $1}') || staged_size=unknown
            staged_size=${staged_size:-unknown}
            echo "[${PIPELINE_LABEL} pipeline][PRIOR ${position}/${#SCENES[@]}][RESUME] ${scene} staged=${staged_size}"
            env \
                "ADGS_PYTHON=${PYTHON_BIN}" \
                "PRIOR_WORK_ROOT=${work_root}" \
                OVERWRITE=0 RESUME=1 DRY_RUN=0 \
                "KEEP_WORK=${KEEP_PRIOR_WORK}" \
                bash "${SCRIPT_DIR}/prepare_splatad_priors.sh" \
                "${DATASET}" "${scene_path}" "${PHYSICAL_GPU}"
        else
            echo "[${PIPELINE_LABEL} pipeline][PRIOR ${position}/${#SCENES[@]}][START] ${scene}"
            env \
                "ADGS_PYTHON=${PYTHON_BIN}" \
                "PRIOR_WORK_ROOT=${work_root}" \
                OVERWRITE=0 RESUME=0 DRY_RUN=0 \
                "KEEP_WORK=${KEEP_PRIOR_WORK}" \
                bash "${SCRIPT_DIR}/prepare_splatad_priors.sh" \
                "${DATASET}" "${scene_path}" "${PHYSICAL_GPU}"
        fi

        echo "[${PIPELINE_LABEL} pipeline][PRIOR ${position}/${#SCENES[@]}][DONE] ${scene} (builder strict validation passed)"
        echo "[${PIPELINE_LABEL} pipeline][PRIOR ${position}/${#SCENES[@]}][LOCK RELEASE] ${scene}"
    )
}

for ((index=0; index<${#SCENES[@]}; index++)); do
    process_scene "${index}" "${SCENES[index]}"
done

check_disk_free training
echo "[${PIPELINE_LABEL} pipeline][PRIORS DONE] All ${#SCENES[@]} scenes passed a strict validation path."
echo "[${PIPELINE_LABEL} pipeline][TRAIN START] GPU ${PHYSICAL_GPU}, one scene at a time."

env \
    "ADGS_PYTHON=${PYTHON_BIN}" \
    "ADGS_MIN_FREE_GIB=${MIN_FREE_GIB}" \
    DRY_RUN=0 ALLOW_EXISTING_OUTPUT=0 ONLY_SCENE= \
    bash "${SCRIPT_DIR}/train_splatad_split.sh" \
    "${DATASET}" "${CONFIG}" "${PROCESSED_ROOT}" "${OUTPUT_ROOT}" \
    "${PHYSICAL_GPU}" "${PHYSICAL_GPU}" "${SCENES[@]}"
