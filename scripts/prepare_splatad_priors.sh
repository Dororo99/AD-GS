#!/usr/bin/env bash

# Prepare every AD-GS prior for one already-converted SplatAD-protocol scene.
# Grounded-SAM-2 is run once per metadata-defined camera stream, never across
# cameras.  Heavy outputs remain under WORK_ROOT until all validations pass.

set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: $0 {waymo|nuscenes|av2} SCENE_DIR PHYSICAL_GPU" >&2
    echo "Example: $0 nuscenes data/processed/nuscenes/scene-0101 4" >&2
    exit 2
fi

DATASET=$1
SCENE_INPUT=$2
PHYSICAL_GPU=$3

case "${DATASET}" in
    waymo) CAMERA_COUNT=3; OBJECT_TEXT='car.bus.truck.van.human.' ;;
    nuscenes|av2) CAMERA_COUNT=$([[ "${DATASET}" == nuscenes ]] && echo 6 || echo 7); OBJECT_TEXT='car.bus.truck.van.human.bike.' ;;
    *) echo "Unsupported dataset: ${DATASET}" >&2; exit 2 ;;
esac
if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "PHYSICAL_GPU must be a non-negative integer: ${PHYSICAL_GPU}" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
if [[ ! -d "${SCENE_INPUT}" ]]; then
    echo "Scene directory does not exist: ${SCENE_INPUT}" >&2
    exit 2
fi
SCENE=$(cd -- "${SCENE_INPUT}" && pwd)

ADGS_PYTHON=${ADGS_PYTHON:-/venv/ad-gs/bin/python}
DPT_PYTHON=${DPT_PYTHON:-/venv/ad-gs-dpt/bin/python}
SAM_PYTHON=${SAM_PYTHON:-/venv/ad-gs-sam/bin/python}
DPT_ROOT=${DPT_ROOT:-${REPO_ROOT}/Depth-Anything-V2}
SAM_ROOT=${SAM_ROOT:-${REPO_ROOT}/Grounded-SAM-2}
SAM_CHECKPOINT=${SAM_CHECKPOINT:-${SAM_ROOT}/checkpoints/sam2.1_hiera_large.pt}
SAM_CONFIG=${SAM_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}
HF_HOME=${HF_HOME:-/workspace/.hf_home}
TORCH_HOME=${TORCH_HOME:-/root/.cache/torch}
COLMAP_BIN=${COLMAP_BIN:-/venv/ad-gs/bin/colmap}
WORK_ROOT=${PRIOR_WORK_ROOT:-${SCENE}/.adgs-priors-work}

DRY_RUN=${DRY_RUN:-0}
OVERWRITE=${OVERWRITE:-0}
RESUME=${RESUME:-0}
KEEP_WORK=${KEEP_WORK:-0}
SAM_STEP=${SAM_STEP:-1}
FLOW_STEP=${FLOW_STEP:-4}
FLOW_DOWNSAMPLE=${FLOW_DOWNSAMPLE:-1}
# The bundled /venv/ad-gs COLMAP binary is built without CUDA.  Opt in only
# when COLMAP_BIN points at a CUDA-enabled build.
COLMAP_USE_GPU=${COLMAP_USE_GPU:-0}

for integer_value in "${DRY_RUN}" "${OVERWRITE}" "${RESUME}" "${KEEP_WORK}" "${SAM_STEP}" "${FLOW_STEP}" "${FLOW_DOWNSAMPLE}" "${COLMAP_USE_GPU}"; do
    if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
        echo "Boolean/count options must be non-negative integers: ${integer_value}" >&2
        exit 2
    fi
done
if (( SAM_STEP < 1 || FLOW_STEP < 1 || FLOW_DOWNSAMPLE < 1 )); then
    echo "SAM_STEP, FLOW_STEP, and FLOW_DOWNSAMPLE must be positive" >&2
    exit 2
fi

HELPER=("${ADGS_PYTHON}" "${SCRIPT_DIR}/camera_safe_priors.py")
OVERWRITE_ARG=()
if (( OVERWRITE )); then
    OVERWRITE_ARG=(--overwrite)
fi
FLOW_SANDBOX_ARG=()
FLOW_RESUME_ARG=()
RESTART_SANDBOX_ARG=()
if (( RESUME )); then
    FLOW_SANDBOX_ARG=(--reuse)
    FLOW_RESUME_ARG=(--resume)
    RESTART_SANDBOX_ARG=(--overwrite)
fi

print_command() {
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
}

print_command_in_dir() {
    local directory=$1
    shift
    printf '[DRY-RUN] (cd %q &&' "${directory}"
    printf ' %q' "$@"
    printf ')\n'
}

run_in_dir() {
    local directory=$1
    shift
    echo "+ (cd ${directory} && $*)"
    (
        cd -- "${directory}"
        "$@"
    )
}

COMMON_OFFLINE_ENV=(
    "CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}"
    "CC=${CC:-/usr/bin/gcc-11}"
    "CXX=${CXX:-/usr/bin/g++-11}"
    "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}"
    "HF_HOME=${HF_HOME}"
    "HF_HUB_OFFLINE=1"
    "TRANSFORMERS_OFFLINE=1"
    "HF_DATASETS_OFFLINE=1"
    "TORCH_HOME=${TORCH_HOME}"
    "QT_QPA_PLATFORM=offscreen"
    "PYTHONUNBUFFERED=1"
)

ensure_camera_mask() {
    local camera_id=$1
    local kind=$2
    local text_prompt=$3
    local camera_root
    local collect_output
    local reason
    camera_root=$(printf '%s/cameras/camera_%03d' "${WORK_ROOT}" "${camera_id}")

    if (( RESUME )); then
        if collect_output=$("${HELPER[@]}" collect-camera-mask "${SCENE}" \
            --dataset "${DATASET}" --work "${WORK_ROOT}" \
            --kind "${kind}" --camera-id "${camera_id}" 2>&1); then
            printf '%s\n' "${collect_output}"
            echo "[RESUME][MASK READY] camera=${camera_id} kind=${kind}"
            return 0
        fi
        reason=${collect_output##*$'\n'}
        echo "[RESUME][MASK GENERATE] camera=${camera_id} kind=${kind} reason=${reason}"
    fi

    run_in_dir "${camera_root}" env "${COMMON_OFFLINE_ENV[@]}" \
        "PYTHONPATH=${SAM_ROOT}" "${SAM_PYTHON}" "${SAM_ROOT}/semantic.py" \
        "${camera_root}" --sam "${SAM_CHECKPOINT}" --sam_cfg "${SAM_CONFIG}" \
        --device cuda:0 --text "${text_prompt}" --name "${kind}" --step "${SAM_STEP}"
    "${HELPER[@]}" collect-camera-mask "${SCENE}" --dataset "${DATASET}" \
        --work "${WORK_ROOT}" --kind "${kind}" --camera-id "${camera_id}"
}

if (( DRY_RUN )); then
    "${HELPER[@]}" stage "${SCENE}" --dataset "${DATASET}" \
        --work "${WORK_ROOT}" --dry-run
    print_command "${HELPER[@]}" preflight "${SCENE}" --dataset "${DATASET}" "${OVERWRITE_ARG[@]}"
    print_command "${HELPER[@]}" stage "${SCENE}" --dataset "${DATASET}" \
        --work "${WORK_ROOT}" "${OVERWRITE_ARG[@]}"
    print_command_in_dir "${DPT_ROOT}" env "${COMMON_OFFLINE_ENV[@]}" \
        "${DPT_PYTHON}" run-dpt.py --img-path "${SCENE}/image" \
        --outdir "${WORK_ROOT}/depth" --encoder vitl --pred-only --grayscale
    print_command "${HELPER[@]}" compact-depth "${SCENE}" \
        --dataset "${DATASET}" --work "${WORK_ROOT}"
    for (( camera_id=0; camera_id<CAMERA_COUNT; camera_id++ )); do
        camera_root=$(printf '%s/cameras/camera_%03d' "${WORK_ROOT}" "${camera_id}")
        print_command_in_dir "${camera_root}" env "${COMMON_OFFLINE_ENV[@]}" \
            "PYTHONPATH=${SAM_ROOT}" "${SAM_PYTHON}" "${SAM_ROOT}/semantic.py" \
            "${camera_root}" --sam "${SAM_CHECKPOINT}" --sam_cfg "${SAM_CONFIG}" \
            --device cuda:0 --text sky. --name sky --step "${SAM_STEP}"
        print_command "${HELPER[@]}" collect-camera-mask "${SCENE}" \
            --dataset "${DATASET}" --work "${WORK_ROOT}" \
            --kind sky --camera-id "${camera_id}"
        print_command_in_dir "${camera_root}" env "${COMMON_OFFLINE_ENV[@]}" \
            "PYTHONPATH=${SAM_ROOT}" "${SAM_PYTHON}" "${SAM_ROOT}/semantic.py" \
            "${camera_root}" --sam "${SAM_CHECKPOINT}" --sam_cfg "${SAM_CONFIG}" \
            --device cuda:0 --text "${OBJECT_TEXT}" --name semantic --step "${SAM_STEP}"
        print_command "${HELPER[@]}" collect-camera-mask "${SCENE}" \
            --dataset "${DATASET}" --work "${WORK_ROOT}" \
            --kind semantic --camera-id "${camera_id}"
    done
    for sandbox_kind in flow segment colmap; do
        print_command "${HELPER[@]}" prepare-sandbox "${SCENE}" --dataset "${DATASET}" --work "${WORK_ROOT}" --kind "${sandbox_kind}"
    done
    print_command_in_dir "${REPO_ROOT}" env "${COMMON_OFFLINE_ENV[@]}" \
        "${ADGS_PYTHON}" scripts/flow.py "${WORK_ROOT}/flow_scene" \
        --device cuda:0 --downsample "${FLOW_DOWNSAMPLE}" --step "${FLOW_STEP}"
    print_command_in_dir "${REPO_ROOT}" env "${COMMON_OFFLINE_ENV[@]}" \
        "${ADGS_PYTHON}" scripts/segment_pcd.py "${WORK_ROOT}/segment_scene"
    COLMAP_DRY_ARGS=("${ADGS_PYTHON}" scripts/colmap.py "${WORK_ROOT}/colmap_scene" --cmd "${COLMAP_BIN}")
    if (( COLMAP_USE_GPU )); then COLMAP_DRY_ARGS+=(--use_gpu); fi
    print_command_in_dir "${REPO_ROOT}" env "${COMMON_OFFLINE_ENV[@]}" "${COLMAP_DRY_ARGS[@]}"
    print_command "${HELPER[@]}" verify-work "${SCENE}" --dataset "${DATASET}" --work "${WORK_ROOT}"
    print_command "${HELPER[@]}" commit-all "${SCENE}" --dataset "${DATASET}" --work "${WORK_ROOT}" "${OVERWRITE_ARG[@]}"
    print_command "${ADGS_PYTHON}" "${SCRIPT_DIR}/validate_splatad_scene.py" "${SCENE}" --dataset "${DATASET}"
    if (( ! KEEP_WORK )); then
        print_command "${HELPER[@]}" cleanup "${SCENE}" --work "${WORK_ROOT}"
    fi
    exit 0
fi

require_executable() {
    if [[ ! -x "$1" ]]; then
        echo "Executable not found: $1" >&2
        exit 2
    fi
}
require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 2
    fi
}

require_executable "${ADGS_PYTHON}"
require_executable "${DPT_PYTHON}"
require_executable "${SAM_PYTHON}"
require_executable "${COLMAP_BIN}"
require_file "${DPT_ROOT}/run-dpt.py"
require_file "${DPT_ROOT}/checkpoints/depth_anything_v2_vitl.pth"
require_file "${SAM_ROOT}/semantic.py"
require_file "${SAM_CHECKPOINT}"
require_file "${SAM_ROOT}/sam2/${SAM_CONFIG}"
require_file "${TORCH_HOME}/hub/facebookresearch_co-tracker_main/hubconf.py"
require_file "${TORCH_HOME}/hub/checkpoints/scaled_offline.pth"

GROUNDING_MODEL_ROOT="${HF_HOME}/hub/models--IDEA-Research--grounding-dino-base"
require_file "${GROUNDING_MODEL_ROOT}/refs/main"
GROUNDING_REVISION=$(tr -d '\r\n' < "${GROUNDING_MODEL_ROOT}/refs/main")
require_file "${GROUNDING_MODEL_ROOT}/snapshots/${GROUNDING_REVISION}/config.json"
require_file "${GROUNDING_MODEL_ROOT}/snapshots/${GROUNDING_REVISION}/model.safetensors"

if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
        echo "Physical GPU ${PHYSICAL_GPU} is not available" >&2
        exit 2
    fi
fi

"${HELPER[@]}" preflight "${SCENE}" --dataset "${DATASET}" "${OVERWRITE_ARG[@]}"
if (( RESUME )); then
    echo "[RESUME] Validating reusable staging: ${WORK_ROOT}"
    "${HELPER[@]}" validate-stage "${SCENE}" --dataset "${DATASET}" --work "${WORK_ROOT}"
    "${HELPER[@]}" compact-depth "${SCENE}" --dataset "${DATASET}" --work "${WORK_ROOT}"
    echo "[RESUME][DEPTH READY] Reusing validated staged depth."
else
    "${HELPER[@]}" stage "${SCENE}" --dataset "${DATASET}" \
        --work "${WORK_ROOT}" "${OVERWRITE_ARG[@]}"
    run_in_dir "${DPT_ROOT}" env "${COMMON_OFFLINE_ENV[@]}" \
        "${DPT_PYTHON}" run-dpt.py --img-path "${SCENE}/image" \
        --outdir "${WORK_ROOT}/depth" --encoder vitl --pred-only --grayscale
    "${HELPER[@]}" compact-depth "${SCENE}" --dataset "${DATASET}" \
        --work "${WORK_ROOT}"
fi

for (( camera_id=0; camera_id<CAMERA_COUNT; camera_id++ )); do
    camera_root=$(printf '%s/cameras/camera_%03d' "${WORK_ROOT}" "${camera_id}")
    if [[ ! -d "${camera_root}/image" ]]; then
        echo "Missing staged camera directory: ${camera_root}/image" >&2
        exit 1
    fi
    ensure_camera_mask "${camera_id}" sky sky.
    ensure_camera_mask "${camera_id}" semantic "${OBJECT_TEXT}"
done

"${HELPER[@]}" prepare-sandbox "${SCENE}" --dataset "${DATASET}" --work "${WORK_ROOT}" --kind flow "${FLOW_SANDBOX_ARG[@]}"
run_in_dir "${REPO_ROOT}" env "${COMMON_OFFLINE_ENV[@]}" \
    "${ADGS_PYTHON}" scripts/flow.py "${WORK_ROOT}/flow_scene" \
    --device cuda:0 --downsample "${FLOW_DOWNSAMPLE}" --step "${FLOW_STEP}" "${FLOW_RESUME_ARG[@]}"

"${HELPER[@]}" prepare-sandbox "${SCENE}" --dataset "${DATASET}" --work "${WORK_ROOT}" --kind segment "${RESTART_SANDBOX_ARG[@]}"
run_in_dir "${REPO_ROOT}" env "${COMMON_OFFLINE_ENV[@]}" \
    "${ADGS_PYTHON}" scripts/segment_pcd.py "${WORK_ROOT}/segment_scene"

"${HELPER[@]}" prepare-sandbox "${SCENE}" --dataset "${DATASET}" --work "${WORK_ROOT}" --kind colmap "${RESTART_SANDBOX_ARG[@]}"
COLMAP_ARGS=("${ADGS_PYTHON}" scripts/colmap.py "${WORK_ROOT}/colmap_scene" --cmd "${COLMAP_BIN}")
if (( COLMAP_USE_GPU )); then COLMAP_ARGS+=(--use_gpu); fi
run_in_dir "${REPO_ROOT}" env "${COMMON_OFFLINE_ENV[@]}" "${COLMAP_ARGS[@]}"

"${HELPER[@]}" verify-work "${SCENE}" --dataset "${DATASET}" --work "${WORK_ROOT}"
"${HELPER[@]}" commit-all "${SCENE}" --dataset "${DATASET}" --work "${WORK_ROOT}" "${OVERWRITE_ARG[@]}"
"${ADGS_PYTHON}" "${SCRIPT_DIR}/validate_splatad_scene.py" "${SCENE}" --dataset "${DATASET}"

if (( ! KEEP_WORK )); then
    "${HELPER[@]}" cleanup "${SCENE}" --work "${WORK_ROOT}"
else
    echo "Keeping staged work: ${WORK_ROOT}"
fi

echo "Prepared and validated ${DATASET} scene on physical GPU ${PHYSICAL_GPU}: ${SCENE}"
