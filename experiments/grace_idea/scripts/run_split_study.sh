#!/usr/bin/env bash
set -euo pipefail

DATASETS=${DATASETS:-Texas}
SPLITS=${SPLITS:-0}
SEEDS=${SEEDS:-0}
METHODS=${METHODS:-"grace es_weighted"}
ES_CONTROLS=${ES_CONTROLS:-normal}
EPOCHS=${EPOCHS:-100}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-20}
GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-0}
SAVE_DIR=${SAVE_DIR:-runs/split_study}
MANIFEST_PATH=${MANIFEST_PATH:-"${SAVE_DIR}/run_manifest.csv"}
OVERWRITE=${OVERWRITE:-0}
LOG_EVERY=${LOG_EVERY:-100}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}

mkdir -p "$(dirname "${MANIFEST_PATH}")"
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  printf 'dataset,split_index,model_seed,method,control,status\n' > "${MANIFEST_PATH}"
fi

run_train() {
  local dataset="$1"
  local split_index="$2"
  local model_seed="$3"
  local method="$4"
  local control="$5"
  local reported_control="${control}"
  if [[ "${reported_control}" == "random" ]]; then
    reported_control="uniform_random"
  fi

  local args=(
    python train.py
    --dataset "${dataset}"
    --method "${method}"
    --seed "${model_seed}"
    --split-index "${split_index}"
    --epochs "${EPOCHS}"
    --gpu_id "${GPU_ID}"
    --save-dir "${SAVE_DIR}"
    --log-every "${LOG_EVERY}"
  )

  if [[ "${BATCH_SIZE}" != "0" ]]; then
    args+=(--batch-size "${BATCH_SIZE}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ "${method}" == "es_weighted" || "${method}" == "sgfn" || "${method}" == "pbcl" || "${method}" == "pccl" || "${method}" == "rr_gcl" || "${method}" == "hybrid_rr_gcl" || "${method}" == "cbr_gcl" ]]; then
    args+=(--warmup-epochs "${WARMUP_EPOCHS}")
    if [[ "${control}" == "shuffled" ]]; then
      args+=(--shuffle-weights)
    elif [[ "${control}" == "random" || "${control}" == "uniform_random" ]]; then
      args+=(--random-weights)
    elif [[ "${control}" != "normal" ]]; then
      echo "Unsupported ES control: ${control}" >&2
      return 2
    fi
  fi
  if [[ -n "${TRAIN_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=(${TRAIN_EXTRA_ARGS})
    args+=("${extra_args[@]}")
  fi

  echo "[run] dataset=${dataset} split=${split_index} seed=${model_seed} method=${method} control=${reported_control}"
  if "${args[@]}"; then
    printf '%s,%s,%s,%s,%s,completed\n' \
      "${dataset}" "${split_index}" "${model_seed}" "${method}" "${reported_control}" >> "${MANIFEST_PATH}"
  else
    printf '%s,%s,%s,%s,%s,failed\n' \
      "${dataset}" "${split_index}" "${model_seed}" "${method}" "${reported_control}" >> "${MANIFEST_PATH}"
    return 1
  fi
}

for dataset in ${DATASETS}; do
  for split_index in ${SPLITS}; do
    for model_seed in ${SEEDS}; do
      for method in ${METHODS}; do
        if [[ "${method}" == "es_weighted" || "${method}" == "sgfn" || "${method}" == "pbcl" || "${method}" == "pccl" || "${method}" == "rr_gcl" || "${method}" == "hybrid_rr_gcl" || "${method}" == "cbr_gcl" ]]; then
          for control in ${ES_CONTROLS}; do
            run_train "${dataset}" "${split_index}" "${model_seed}" "${method}" "${control}"
          done
        else
          run_train "${dataset}" "${split_index}" "${model_seed}" "${method}" "baseline"
        fi
      done
    done
  done
done
