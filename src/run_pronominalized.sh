#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
MODEL_NAME="${MODEL_NAME:-gpt2}"
CORPUS="${CORPUS:-all}"
RESULTS_DIR="${RESULTS_DIR:-deepseek_pronoun_baseline_results}"
PRONOUN_CONTEXT_MODE="${PRONOUN_CONTEXT_MODE:-additional_context}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-1024}"
CONTEXT_SIZES="${CONTEXT_SIZES:-2 3}"

"${PYTHON_BIN}" src/gpt2_inference_pronominalized.py \
  --model_name "${MODEL_NAME}" \
  --results_dir "${RESULTS_DIR}" \
  --corpus "${CORPUS}" \
  --context_sizes \
    ${CONTEXT_SIZES} \
  --device "${DEVICE}" \
  --max_batch_tokens "${MAX_BATCH_TOKENS}" \
  --pronoun_context_mode "${PRONOUN_CONTEXT_MODE}" \
  "$@"
