#!/usr/bin/env python3
"""
Run GPT-2 word surprisal inference on the original stories, using
all_story_pronouns.tsv to collapse annotated context spans to pronouns.

The input file is expected to contain one original story per line.
By default, this script runs every story represented in all_story_pronouns.tsv.

Output matches the existing surprisal format:
  word_info.tsv
  context_N.txt

Usage:
    python src/gpt2_inference_pronominalized.py
    python src/gpt2_inference_pronominalized.py --story_ids 0 1 2
    python src/gpt2_inference_pronominalized.py --context_sizes 3 10 50 100
    python src/gpt2_inference_pronominalized.py --pronoun_context_mode additional_context
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from word_surprisal import (
    DEFAULT_ALL_STORY_PRONOUNS_TSV,
    aggregate_wt_surprisal,
    build_per_token_windows_with_pronoun_replacements,
    compute_token_level_quantities,
    get_space_token_ids,
    load_pronoun_replacements_by_story,
    pretokenize_documents,
    pronoun_replacements_with_same_context,
    setup_tokenizer_and_model,
    validate_pronoun_token_counts,
)



def load_all_stories(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_word_info(path: str, story_ids, doc_wspans, doc_words) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("story\tzone\tword\tnum_tokens\n")
        for story_id, wspans, words in zip(story_ids, doc_wspans, doc_words):
            for word_i, ((tok_start, tok_end), word) in enumerate(zip(wspans, words)):
                f.write(f"{story_id}\t{word_i}\t{word}\t{tok_end - tok_start}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-2 word surprisal inference for original stories with pronoun-aware windows")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--input_file", required=True)
    parser.add_argument(
        "--story_ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Zero-based story ids to run. Defaults to every story represented in "
            "the pronoun TSV."
        ),
    )
    parser.add_argument(
        "--story_number",
        type=int,
        default=None,
        help=(
            "Backward-compatible 1-based single story selector. Prefer --story_ids. "
            "If omitted, all pronoun TSV stories are run."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_sizes", type=int, nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--max_batch_tokens", type=int, default=1024)
    parser.add_argument(
        "--pronoun_context_mode",
        choices=("window_only", "additional_context"),
        default="window_only",
        help=(
            "How to build pronoun-aware windows. window_only collapses full "
            "pronoun spans inside the original context window. "
            "additional_context backfills earlier tokens after collapsing so "
            "the realized window stays at the requested context size when "
            "possible."
        ),
    )
    parser.add_argument(
        "--pronoun_tsv",
        default=DEFAULT_ALL_STORY_PRONOUNS_TSV,
        help=(
            "Per-token pronoun replacement TSV with story, token_index, and "
            "pronoun_replacement columns. Defaults to src/all_story_pronouns.tsv."
        ),
    )
    args = parser.parse_args()

    if args.story_ids is not None and args.story_number is not None:
        raise ValueError("Use either --story_ids or --story_number, not both.")

    device = torch.device(args.device)
    dtype = torch.float16

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}  dtype: {dtype}  max_batch_tokens: {args.max_batch_tokens}")

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.model_max_length = sys.maxsize
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype).to(device)
    tokenizer, model = setup_tokenizer_and_model(tokenizer, model)
    model.eval()

    # GPT-2 supports 1024 positions. Keep the script conservative if another
    # compatible checkpoint is passed in.
    max_ctx = int(getattr(model.config, "n_positions", 1024))
    context_sizes = sorted({ctx for ctx in args.context_sizes if ctx <= max_ctx})
    skipped = sorted({ctx for ctx in args.context_sizes if ctx > max_ctx})
    if skipped:
        print(f"Skipping context sizes > {max_ctx}: {skipped}")
    print(f"Context sizes: {context_sizes}")

    all_stories = load_all_stories(args.input_file)
    replacements_by_story = load_pronoun_replacements_by_story(args.pronoun_tsv)

    if args.story_ids is not None:
        story_ids = args.story_ids
    elif args.story_number is not None:
        if args.story_number < 1:
            raise ValueError(
                f"story_number must be 1-based and positive, got {args.story_number}."
            )
        story_ids = [args.story_number - 1]
    else:
        story_ids = sorted(replacements_by_story)

    missing_pronouns = [story_id for story_id in story_ids if story_id not in replacements_by_story]
    if missing_pronouns:
        raise ValueError(
            f"Requested story ids missing from {args.pronoun_tsv}: {missing_pronouns}"
        )

    missing_stories = [story_id for story_id in story_ids if story_id >= len(all_stories)]
    if missing_stories:
        raise ValueError(
            f"Requested story ids missing from {args.input_file}: {missing_stories}. "
            f"{args.input_file} has {len(all_stories)} non-empty stories."
        )

    documents = [all_stories[story_id] for story_id in story_ids]
    print(
        f"Loaded {len(documents)} original stories from {args.input_file}: "
        f"{story_ids}"
    )

    doc_pronoun_replacements = validate_pronoun_token_counts(
        story_ids,
        documents,
        tokenizer,
        replacements_by_story,
        args.pronoun_tsv,
    )
    total_pronoun_rows = sum(len(replacements_by_story[story_id]) for story_id in story_ids)
    print(
        f"Validated {total_pronoun_rows} token-level pronoun entries "
        f"from {args.pronoun_tsv} against {len(story_ids)} stories"
    )

    print("Pre-tokenizing...")
    doc_tids, doc_wspans, doc_words, total_tokens, total_words = pretokenize_documents(
        documents, tokenizer
    )
    print(f"  {total_tokens} tokens, {total_words} words")

    info_path = os.path.join(args.output_dir, "word_info.tsv")
    write_word_info(info_path, story_ids, doc_wspans, doc_words)
    print(f"  -> {info_path}")

    pad_id = tokenizer.pad_token_id
    space_ids = get_space_token_ids(tokenizer)
    use_position_ids = getattr(model.config, "model_type", "").lower() != "opt"
    if args.pronoun_context_mode == "additional_context":
        build_pronoun_windows = pronoun_replacements_with_same_context
    else:
        build_pronoun_windows = build_per_token_windows_with_pronoun_replacements
    print(f"Pronoun context mode: {args.pronoun_context_mode}")

    for ctx_size in context_sizes:
        print(f"\n{'=' * 60}")
        print(f"  context_size={ctx_size}  ({ctx_size - 1} context + 1 target)")
        print(f"{'=' * 60}")
        t0 = time.time()

        all_win, all_mask, all_tgt, doc_off = build_pronoun_windows(
            doc_tids,
            ctx_size,
            pad_id,
            tokenizer,
            doc_pronoun_replacements,
        )
        batch_size = max(1, args.max_batch_tokens // ctx_size)
        print(f"  batch_size={batch_size}")

        surprisals, space_lp = compute_token_level_quantities(
            all_win,
            all_mask,
            all_tgt,
            model,
            space_ids,
            device,
            batch_size,
            use_position_ids=use_position_ids,
        )
        word_surps = aggregate_wt_surprisal(
            surprisals,
            space_lp,
            doc_tids,
            doc_wspans,
            doc_off,
        )

        out_path = os.path.join(args.output_dir, f"context_{ctx_size}.txt")
        np.savetxt(out_path, word_surps, fmt="%.8f")
        print(f"  -> {out_path}  ({total_words} words, {time.time() - t0:.1f}s)")

        del all_win, all_mask, all_tgt, doc_off, surprisals, space_lp, word_surps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
