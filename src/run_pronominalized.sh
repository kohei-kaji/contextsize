#!/usr/bin/env bash
set -euo pipefail

uv run gpt2_inference_pronominalized.py \
    --model_name gpt2 \
    --input_file ./stories.txt \
    --output_dir ns_pronominalized \
    --context_sizes \
        2 3 4 5 6 7 8 11 13 17 21 26 33 41 51 65 81 101 129 161 201 257 321 401 513 641 801 1024 \
    --device cuda:3 \
    --pronoun_tsv ./all_story_pronouns.tsv\
    --pronoun_context_mode additional_context