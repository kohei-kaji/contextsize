#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
MODEL_NAME="${MODEL_NAME:-gpt2}"
CORPUS="${CORPUS:-all}"
PRONOUN_CONTEXT_MODE="${PRONOUN_CONTEXT_MODE:-additional_context}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-1024}"

"${PYTHON_BIN}" src/gpt2_inference_pronominalized.py \
  --model_name "${MODEL_NAME}" \
  --corpus "${CORPUS}" \
  --context_sizes \
    2 3 4 5 6 7 8 11 13 17 21 26 33 41 51 65 81 101 129 161 201 257 321 401 513 641 801 1024 \
  --device "${DEVICE}" \
  --max_batch_tokens "${MAX_BATCH_TOKENS}" \
  --pronoun_context_mode "${PRONOUN_CONTEXT_MODE}"
