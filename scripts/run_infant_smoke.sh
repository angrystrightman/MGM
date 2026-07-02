#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-mgm-repro}"
CONDA_BIN="${CONDA_BIN:-/home/sunyirong/miniforge3/bin/conda}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/mgm_repro/infant_smoke}"
MAX_SAMPLES="${MAX_SAMPLES:-96}"
SEED="${SEED:-0}"
VAL_SPLIT="${VAL_SPLIT:-0.25}"
MGM_CMD="${MGM_CMD:-mgm}"
CLEAN_RUN="${CLEAN_RUN:-1}"
SMOKE_CUDA_VISIBLE_DEVICES="${SMOKE_CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SMOKE_CUDA_VISIBLE_DEVICES}"
fi
if [[ "${CLEAN_RUN}" == "1" ]]; then
  case "${RUN_ROOT}" in
    "${REPO_ROOT}"/runs/*)
      rm -rf "${RUN_ROOT}"
      ;;
    *)
      echo "Refusing to clean RUN_ROOT outside ${REPO_ROOT}/runs: ${RUN_ROOT}" >&2
      exit 2
      ;;
  esac
fi
mkdir -p "${RUN_ROOT}"/{data,logs,model,predictions,representations}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty>}"

run_py() {
  "${CONDA_BIN}" run -n "${ENV_NAME}" python "$@"
}

run_mgm() {
  "${CONDA_BIN}" run -n "${ENV_NAME}" "${MGM_CMD}" "$@"
}

echo "[1/6] Checking packaged MGM resources"
run_py scripts/check_mgm_resources.py \
  --json-output "${RUN_ROOT}/resource_check.json"

echo "[2/6] Preparing infant smoke subset"
run_py scripts/prepare_infant_smoke_data.py \
  --abundance infant_data/abundance.csv \
  --metadata infant_data/meta_withbirth.csv \
  --output-dir "${RUN_ROOT}/data" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED}"

echo "[3/6] Constructing MicroCorpus"
run_mgm construct \
  -c configs/mgm_smoke.ini \
  -i "${RUN_ROOT}/data/abundance.csv" \
  -o "${RUN_ROOT}/data/corpus.pkl" \
  --seed "${SEED}"

echo "[4/6] Finetuning MGM on C/V labels"
run_mgm finetune \
  -c configs/mgm_smoke.ini \
  -i "${RUN_ROOT}/data/corpus.pkl" \
  -l "${RUN_ROOT}/data/labels_delivery_cv.csv" \
  -o "${RUN_ROOT}/model" \
  -H "${RUN_ROOT}/logs" \
  -s "${VAL_SPLIT}" \
  --seed "${SEED}"

echo "[5/6] Predicting and evaluating"
run_mgm predict \
  -E \
  -c configs/mgm_smoke.ini \
  -i "${RUN_ROOT}/data/corpus.pkl" \
  -l "${RUN_ROOT}/data/labels_delivery_cv.csv" \
  -m "${RUN_ROOT}/model" \
  -o "${RUN_ROOT}/predictions" \
  --seed "${SEED}"

echo "[6/6] Extracting embeddings and attention top-k"
run_py scripts/extract_mgm_representations.py \
  --corpus "${RUN_ROOT}/data/corpus.pkl" \
  --model "${RUN_ROOT}/model" \
  --output-dir "${RUN_ROOT}/representations" \
  --max-samples 16 \
  --top-k 10

echo "Infant smoke run complete: ${RUN_ROOT}"
