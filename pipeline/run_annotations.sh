#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <corpus> <repeated|singleton> <word-info.tsv>" >&2
  exit 2
fi

CORPUS="$1"
MODE="$2"
WORD_INFO="$3"

case "${CORPUS}" in
  ns) SOURCE_NAME="stories.txt" ;;
  brown|onestop|provo) SOURCE_NAME="${CORPUS}.txt" ;;
  *) echo "Unsupported corpus: ${CORPUS}" >&2; exit 2 ;;
esac

case "${MODE}" in
  repeated)
    CONDITION="repeated_coref"
    DEFAULT_OCCURRENCE_POLICY="first"
    ;;
  singleton)
    CONDITION="singleton_baseline"
    DEFAULT_OCCURRENCE_POLICY="last-unoccupied"
    ;;
  *) echo "Mode must be 'repeated' or 'singleton'." >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RAW_TEXT="outputs/raw_texts/${SOURCE_NAME}"
STANZA_DIR="outputs/stanza_annotations"
DOCUMENT_DIR="outputs/conditions/${CONDITION}/documents/${CORPUS}"
WORD_OUTPUT="outputs/conditions/${CONDITION}/word_annotations/${CORPUS}.tsv"
TOKEN_OUTPUT="outputs/conditions/${CONDITION}/token_annotations/${CORPUS}.tsv"
OCCURRENCE_POLICY="${OCCURRENCE_POLICY:-${DEFAULT_OCCURRENCE_POLICY}}"

uv run python pipeline/01_coreference/build_stanza_word_coref.py \
  "${RAW_TEXT}" "${STANZA_DIR}" --corpus "${CORPUS}"

if [[ "${QUERY_DEEPSEEK:-0}" == "1" ]]; then
  MAPPING_ACTION=(--query-deepseek)
else
  MAPPING_SOURCE="${MAPPING_SOURCE:-${DOCUMENT_DIR}}"
  MAPPING_ACTION=(--reuse-mappings "${MAPPING_SOURCE}")
fi

uv run python pipeline/02_pronoun_mapping/pronominalize.py \
  "${STANZA_DIR}" "${DOCUMENT_DIR}" \
  --corpus "${CORPUS}" --mode "${MODE}" "${MAPPING_ACTION[@]}"

uv run python pipeline/03_token_alignment/add_pronoun_mappings_to_word_info.py \
  --word-info "${WORD_INFO}" \
  --mappings "${DOCUMENT_DIR}" \
  --output "${WORD_OUTPUT}" \
  --occurrence-policy "${OCCURRENCE_POLICY}"

uv run python pipeline/03_token_alignment/expand_word_info_pronouns.py \
  --word-info "${WORD_OUTPUT}" \
  --output "${TOKEN_OUTPUT}"

echo "Wrote ${WORD_OUTPUT}"
echo "Wrote ${TOKEN_OUTPUT}"
