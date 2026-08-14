#!/usr/bin/env bash

set -euo pipefail

if (( $# < 7 )); then
    echo "Usage: $0 DATASET CONFIG PROCESSED_ROOT OUTPUT_ROOT GPU_A GPU_B SCENE..." >&2
    exit 2
fi

DATASET=$1
CONFIG=$2
PROCESSED_ROOT=$3
OUTPUT_ROOT=$4
GPU_A=$5
GPU_B=$6
shift 6
SCENES=("$@")

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN=${ADGS_PYTHON:-/venv/ad-gs/bin/python}
RUN_RENDER=${RUN_RENDER:-1}
DRY_RUN=${DRY_RUN:-0}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}
WANDB_ENABLED=${WANDB_ENABLED:-1}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-CamoSplat_ICLR_2027}
WANDB_SPLIT_TYPE=${WANDB_SPLIT_TYPE:-SplatAD}
WANDB_MODEL_NAME=${WANDB_MODEL_NAME:-AD-GS}
WANDB_EVAL_INTERVAL=${WANDB_EVAL_INTERVAL:-500}
WANDB_EVAL_IMAGE_COUNT=${WANDB_EVAL_IMAGE_COUNT:-3}
WANDB_EVAL_CAMERA_ID=${WANDB_EVAL_CAMERA_ID:-0}
WANDB_EVAL_LPIPS=${WANDB_EVAL_LPIPS:-1}
WANDB_SCALAR_LOG_INTERVAL=${WANDB_SCALAR_LOG_INTERVAL:-10}
ADGS_CONSOLE_LOG_INTERVAL=${ADGS_CONSOLE_LOG_INTERVAL:-100}
ADGS_MIN_FREE_GIB=${ADGS_MIN_FREE_GIB:-0}

case "${DATASET}" in
    nuscenes) DEFAULT_WANDB_DATASET_TYPE=nuScenes ;;
    waymo) DEFAULT_WANDB_DATASET_TYPE=Waymo ;;
    av2) DEFAULT_WANDB_DATASET_TYPE=Argoverse2 ;;
    *) echo "Unsupported W&B dataset type: ${DATASET}" >&2; exit 2 ;;
esac
WANDB_DATASET_TYPE=${WANDB_DATASET_TYPE:-${DEFAULT_WANDB_DATASET_TYPE}}
WANDB_PROJECT=${WANDB_PROJECT:-${WANDB_SPLIT_TYPE}_${WANDB_DATASET_TYPE}_${WANDB_MODEL_NAME}}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${WANDB_PROJECT}}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "AD-GS Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! "${ADGS_MIN_FREE_GIB}" =~ ^[0-9]+$ ]]; then
    echo "ADGS_MIN_FREE_GIB must be a non-negative integer: ${ADGS_MIN_FREE_GIB}" >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 2
fi
if [[ "${GPU_A}" == "${GPU_B}" ]]; then
    WORKER_COUNT=1
    GPU_SUMMARY="${GPU_A} (sequential)"
else
    WORKER_COUNT=2
    GPU_SUMMARY="${GPU_A},${GPU_B} (parallel)"
fi

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
        echo "ONLY_SCENE is not in the configured ${DATASET} scene list" >&2
        exit 2
    fi
fi

TOTAL_SCENES=${#SCENES[@]}
echo "============================================"
echo "[${DATASET}] SplatAD training batch"
echo "Scenes:       ${TOTAL_SCENES}"
echo "GPUs:         ${GPU_SUMMARY}"
echo "Output:       ${OUTPUT_ROOT}"
echo "W&B:          ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_MODE})"
echo "W&B preview:  every ${WANDB_EVAL_INTERVAL} steps, ${WANDB_EVAL_IMAGE_COUNT} GT|Render pairs"
echo "Console log:  every ${ADGS_CONSOLE_LOG_INTERVAL} steps"
echo "Disk floor:   ${ADGS_MIN_FREE_GIB} GiB (0 disables the guard)"
echo "============================================"

EXTRA_TRAIN_ARGS=()
if [[ -n "${ADGS_EXTRA_TRAIN_ARGS:-}" ]]; then
    read -r -a EXTRA_TRAIN_ARGS <<< "${ADGS_EXTRA_TRAIN_ARGS}"
fi

print_command() {
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
}

available_space_gib() {
    local available_kib
    available_kib=$(df -Pk -- "${REPO_ROOT}" | awk 'NR == 2 {print $4}')
    if [[ ! "${available_kib}" =~ ^[0-9]+$ ]]; then
        echo "Unable to determine free disk space for ${REPO_ROOT}" >&2
        return 1
    fi
    echo $((available_kib / 1024 / 1024))
}

ensure_free_space() {
    local phase=$1
    local scene=$2
    local free_gib
    if (( ADGS_MIN_FREE_GIB == 0 )); then
        return 0
    fi
    free_gib=$(available_space_gib)
    echo "[${DATASET}][SPACE] phase=${phase} scene=${scene} available=${free_gib}GiB minimum=${ADGS_MIN_FREE_GIB}GiB"
    if (( free_gib < ADGS_MIN_FREE_GIB )); then
        echo "[${DATASET}][STOP LOW DISK] phase=${phase} scene=${scene} available=${free_gib}GiB minimum=${ADGS_MIN_FREE_GIB}GiB" >&2
        return 1
    fi
}

if (( DRY_RUN )); then
    echo "[${DATASET}] dry-run: strict scene validation is intentionally skipped"
else
    validation_failures=()
    for ((index=0; index<${#SCENES[@]}; index++)); do
        scene=${SCENES[index]}
        scene_path="${PROCESSED_ROOT}/${scene}"
        echo "[${DATASET}][VALIDATE $((index + 1))/${TOTAL_SCENES}] ${scene}"
        if validation_output=$("${PYTHON_BIN}" scripts/validate_splatad_scene.py \
            "${scene_path}" --dataset "${DATASET}" 2>&1)
        then
            echo "[${DATASET}][READY $((index + 1))/${TOTAL_SCENES}] ${scene}"
        else
            validation_failures+=("${scene}")
            echo "[${DATASET}][NEEDS_PRIORS $((index + 1))/${TOTAL_SCENES}] ${scene}" >&2
            while IFS= read -r validation_line; do
                printf '  %s\n' "${validation_line}" >&2
            done <<< "${validation_output}"
        fi
    done
    if (( ${#validation_failures[@]} > 0 )); then
        echo "[${DATASET}][VALIDATION FAILED] ${#validation_failures[@]}/${TOTAL_SCENES} scene(s) are not training-ready." >&2
        echo "Complete the full prior pipeline; do not bypass validation or synthesize obj=0." >&2
        for scene in "${validation_failures[@]}"; do
            scene_path="${PROCESSED_ROOT}/${scene}"
            if [[ -d "${scene_path}/.adgs-priors-work" ]]; then
                echo "  ${scene}: validated interrupted staging should be resumed before considering a restart." >&2
                printf '  RESUME=1 bash scripts/prepare_splatad_priors.sh %q %q %q\n' \
                    "${DATASET}" "${scene_path}" "${GPU_A}" >&2
            else
                printf '  bash scripts/prepare_splatad_priors.sh %q %q %q\n' \
                    "${DATASET}" "${scene_path}" "${GPU_A}" >&2
            fi
        done
        echo "RESUME=1 validates and reuses staging; this launcher never deletes it or enables OVERWRITE automatically." >&2
        exit 1
    fi
fi

train_worker() {
    local gpu=$1
    local offset=$2
    local stride=$3
    local index scene source model log_file wandb_name position
    local scene_start elapsed worker_status
    local -a train_env
    worker_status=0
    for ((index=offset; index<${#SCENES[@]}; index+=stride)); do
        scene=${SCENES[index]}
        position=$((index + 1))
        source="${PROCESSED_ROOT}/${scene}"
        model="${OUTPUT_ROOT}/${scene}"
        log_file="${model}/launcher.log"
        wandb_name="${WANDB_RUN_NAME_PREFIX:-}${scene}"
        train_env=(
            env
            "CUDA_VISIBLE_DEVICES=${gpu}"
            "ADGS_SCENE_NAME=${scene}"
            "ADGS_PHYSICAL_GPU=${gpu}"
            PYTHONUNBUFFERED=1
            "WANDB_ENABLED=${WANDB_ENABLED}"
            "WANDB_MODE=${WANDB_MODE}"
            "WANDB_ENTITY=${WANDB_ENTITY}"
            "WANDB_PROJECT=${WANDB_PROJECT}"
            "WANDB_NAME=${wandb_name}"
            "WANDB_RUN_GROUP=${WANDB_RUN_GROUP}"
            "WANDB_SPLIT_TYPE=${WANDB_SPLIT_TYPE}"
            "WANDB_DATASET_TYPE=${WANDB_DATASET_TYPE}"
            "WANDB_MODEL_NAME=${WANDB_MODEL_NAME}"
            "WANDB_EVAL_INTERVAL=${WANDB_EVAL_INTERVAL}"
            "WANDB_EVAL_IMAGE_COUNT=${WANDB_EVAL_IMAGE_COUNT}"
            "WANDB_EVAL_CAMERA_ID=${WANDB_EVAL_CAMERA_ID}"
            "WANDB_EVAL_LPIPS=${WANDB_EVAL_LPIPS}"
            "WANDB_SCALAR_LOG_INTERVAL=${WANDB_SCALAR_LOG_INTERVAL}"
            "ADGS_CONSOLE_LOG_INTERVAL=${ADGS_CONSOLE_LOG_INTERVAL}"
            WANDB_DISABLE_CODE=true
        )

        if (( DRY_RUN )); then
            print_command "${train_env[@]}" \
                "${PYTHON_BIN}" train.py -c "${CONFIG}" -s "${source}" \
                -m "${model}" --data_device cuda "${EXTRA_TRAIN_ARGS[@]}"
            if (( RUN_RENDER )); then
                print_command env "CUDA_VISIBLE_DEVICES=${gpu}" PYTHONUNBUFFERED=1 \
                    "${PYTHON_BIN}" render.py -c "${CONFIG}" -s "${source}" \
                    -m "${model}" --data_device cuda --skip_train
            fi
            continue
        fi

        if (( ! ALLOW_EXISTING_OUTPUT )) && \
            [[ -e "${model}/cfg_args" || -d "${model}/point_cloud" ]]; then
            echo "[${DATASET}][GPU ${gpu}][${position}/${TOTAL_SCENES}][FAILED] Existing output: ${model}" >&2
            echo "Set ALLOW_EXISTING_OUTPUT=1 only to rerun there intentionally; AD-GS does not auto-resume." >&2
            worker_status=1
            continue
        fi

        if ! ensure_free_space train "${scene}"; then
            worker_status=1
            break
        fi


        mkdir -p "${model}"
        scene_start=${SECONDS}
        {
            echo "============================================"
            echo "[${DATASET}][GPU ${gpu}][${position}/${TOTAL_SCENES}][TRAIN START] ${scene}"
            echo "Time:      $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
            echo "W&B:       ${WANDB_ENTITY}/${WANDB_PROJECT}/${wandb_name}"
            echo "Live log:  ${log_file}"
            echo "============================================"
        } | tee "${log_file}"

        if "${train_env[@]}" \
            "${PYTHON_BIN}" train.py -c "${CONFIG}" -s "${source}" \
            -m "${model}" --data_device cuda "${EXTRA_TRAIN_ARGS[@]}" \
            2>&1 | tee -a "${log_file}"
        then
            elapsed=$((SECONDS - scene_start))
            echo "[${DATASET}][GPU ${gpu}][${position}/${TOTAL_SCENES}][TRAIN DONE] ${scene} elapsed=${elapsed}s" | tee -a "${log_file}"
        else
            elapsed=$((SECONDS - scene_start))
            echo "[${DATASET}][GPU ${gpu}][${position}/${TOTAL_SCENES}][TRAIN FAILED] ${scene} elapsed=${elapsed}s log=${log_file}" | tee -a "${log_file}" >&2
            worker_status=1
            continue
        fi

        if (( RUN_RENDER )); then
            if ! ensure_free_space render "${scene}"; then
                worker_status=1
                break
            fi
            echo "[${DATASET}][GPU ${gpu}][${position}/${TOTAL_SCENES}][RENDER START] ${scene}" | tee -a "${log_file}"
            if env "CUDA_VISIBLE_DEVICES=${gpu}" PYTHONUNBUFFERED=1 \
                "${PYTHON_BIN}" render.py -c "${CONFIG}" -s "${source}" \
                -m "${model}" --data_device cuda --skip_train \
                2>&1 | tee -a "${log_file}"
            then
                echo "[${DATASET}][GPU ${gpu}][${position}/${TOTAL_SCENES}][RENDER DONE] ${scene}" | tee -a "${log_file}"
            else
                elapsed=$((SECONDS - scene_start))
                echo "[${DATASET}][GPU ${gpu}][${position}/${TOTAL_SCENES}][RENDER FAILED] ${scene} elapsed=${elapsed}s log=${log_file}" | tee -a "${log_file}" >&2
                worker_status=1
                continue
            fi
        fi

        elapsed=$((SECONDS - scene_start))
        echo "[${DATASET}][GPU ${gpu}][${position}/${TOTAL_SCENES}][SCENE DONE] ${scene} elapsed=${elapsed}s" | tee -a "${log_file}"
    done
    return "${worker_status}"
}

if (( DRY_RUN )); then
    train_worker "${GPU_A}" 0 "${WORKER_COUNT}"
    if (( WORKER_COUNT == 2 )); then
        train_worker "${GPU_B}" 1 "${WORKER_COUNT}"
    fi
    exit 0
fi

status=0
if (( WORKER_COUNT == 1 )); then
    if ! train_worker "${GPU_A}" 0 1; then
        status=1
    fi
else
    worker_pids=()
    train_worker "${GPU_A}" 0 2 &
    worker_pids+=("$!")
    train_worker "${GPU_B}" 1 2 &
    worker_pids+=("$!")

    for pid in "${worker_pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
fi
if (( status == 0 )); then
    echo "[${DATASET}][BATCH DONE] All ${TOTAL_SCENES} scenes completed successfully."
else
    echo "[${DATASET}][BATCH FAILED] One or more scenes failed; inspect each launcher.log." >&2
fi
exit "${status}"
