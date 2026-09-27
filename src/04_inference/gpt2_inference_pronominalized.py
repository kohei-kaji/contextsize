#!/usr/bin/env python3
"""
Run GPT-2 word surprisal inference on the original stories, using a per-token
pronoun TSV to collapse annotated context spans to pronouns.

The input file is expected to contain one original story per line.
By default, explicit-input mode uses the repeated-coreference Natural Stories
token annotations under ``outputs/pre_inference/conditions``.

Output matches the existing surprisal format:
  word_info.tsv
  context_N.txt

Usage:
    python src/04_inference/gpt2_inference_pronominalized.py --corpus provo --context_sizes 3 10 50 100
    python src/04_inference/gpt2_inference_pronominalized.py --corpus all --context_sizes 3 10 50 100
    python src/04_inference/gpt2_inference_pronominalized.py --annotations-dir outputs/pre_inference/conditions/singleton_baseline/token_annotations --corpus all --context_sizes 3 10 50 100
    python src/04_inference/gpt2_inference_pronominalized.py --input_file outputs/pre_inference/raw_texts/provo.txt --pronoun_tsv outputs/pre_inference/conditions/repeated_coref/token_annotations/provo.tsv --output_dir outputs/surprisal/repeated_coref/provo/gpt2 --context_sizes 3 10 50 100
    python src/04_inference/gpt2_inference_pronominalized.py --pronoun_context_mode additional_context
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS_DIR = PROJECT_ROOT / "outputs" / "pre_inference" / "conditions" / "repeated_coref" / "token_annotations"
SUPPORTED_CORPORA = ("provo", "brown", "onestop", "ns")
DEFAULT_CONTEXT_SIZES = [3, 5, 10, 20, 50, 100]


def input_file_for_corpus(corpus: str) -> Path:
    source_name = "stories.txt" if corpus == "ns" else f"{corpus}.txt"
    return PROJECT_ROOT / "outputs" / "pre_inference" / "raw_texts" / source_name


def default_corpus_paths(annotations_dir: Path, corpus: str, model_name: str) -> tuple[str, str, str]:
    input_file = input_file_for_corpus(corpus)
    pronoun_tsv = annotations_dir / f"{corpus}.tsv"
    missing = [path for path in (input_file, pronoun_tsv) if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required corpus files for {corpus}: {missing_paths}")

    return (
        str(input_file),
        str(pronoun_tsv),
        str(
            PROJECT_ROOT
            / "outputs"
            / "surprisal"
            / annotations_dir.parent.name
            / corpus
            / model_name
        ),
    )


def corpora_for_mode(annotations_dir: Path, corpus_mode: str) -> tuple[str, ...]:
    if corpus_mode != "all":
        return (corpus_mode,)

    corpora = []
    for path in sorted(annotations_dir.glob("*.tsv")):
        corpus = path.stem
        if corpus in SUPPORTED_CORPORA:
            corpora.append(corpus)
    if not corpora:
        raise FileNotFoundError(f"No <corpus>.tsv files found in {annotations_dir}")
    return tuple(corpora)


def resolve_annotations_dir(path: Path) -> Path:
    if path.exists() or path.is_absolute():
        return path

    root_relative = PROJECT_ROOT / path
    if root_relative.exists():
        return root_relative

    return path


def load_all_stories(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_word_info(path: str, story_ids, doc_wspans, doc_words) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("story\tzone\tword\tnum_tokens\n")
        for story_id, wspans, words in zip(story_ids, doc_wspans, doc_words):
            for word_i, ((tok_start, tok_end), word) in enumerate(zip(wspans, words)):
                f.write(f"{story_id}\t{word_i}\t{word}\t{tok_end - tok_start}\n")


def selected_story_ids(args, replacements_by_story: dict[int, list[str | None]]) -> list[int]:
    if args.story_ids is not None:
        return args.story_ids
    if args.story_number is not None:
        if args.story_number < 1:
            raise ValueError(
                f"story_number must be 1-based and positive, got {args.story_number}."
            )
        return [args.story_number - 1]
    return sorted(replacements_by_story)


def run_inference_for_inputs(
    *,
    input_file: str,
    pronoun_tsv: str,
    output_dir: str,
    args,
    tokenizer,
    model,
    context_sizes: list[int],
    device,
    dtype,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nInput: {input_file}")
    print(f"Pronouns: {pronoun_tsv}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}  dtype: {dtype}  max_batch_tokens: {args.max_batch_tokens}")

    all_stories = load_all_stories(input_file)
    replacements_by_story = load_pronoun_replacements_by_story(pronoun_tsv)
    story_ids = selected_story_ids(args, replacements_by_story)

    missing_pronouns = [story_id for story_id in story_ids if story_id not in replacements_by_story]
    if missing_pronouns:
        raise ValueError(
            f"Requested story ids missing from {pronoun_tsv}: {missing_pronouns}"
        )

    missing_stories = [story_id for story_id in story_ids if story_id >= len(all_stories)]
    if missing_stories:
        raise ValueError(
            f"Requested story ids missing from {input_file}: {missing_stories}. "
            f"{input_file} has {len(all_stories)} non-empty stories."
        )

    documents = [all_stories[story_id] for story_id in story_ids]
    print(f"Loaded {len(documents)} original stories from {input_file}: {story_ids}")

    doc_pronoun_replacements = validate_pronoun_token_counts(
        story_ids,
        documents,
        tokenizer,
        replacements_by_story,
        pronoun_tsv,
    )
    total_pronoun_rows = sum(len(replacements_by_story[story_id]) for story_id in story_ids)
    print(
        f"Validated {total_pronoun_rows} token-level pronoun entries "
        f"from {pronoun_tsv} against {len(story_ids)} stories"
    )

    print("Pre-tokenizing...")
    doc_tids, doc_wspans, doc_words, total_tokens, total_words = pretokenize_documents(
        documents, tokenizer
    )
    print(f"  {total_tokens} tokens, {total_words} words")

    info_path = os.path.join(output_dir, "word_info.tsv")
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

        out_path = os.path.join(output_dir, f"context_{ctx_size}.txt")
        np.savetxt(out_path, word_surps, fmt="%.8f")
        print(f"  -> {out_path}  ({total_words} words, {time.time() - t0:.1f}s)")

        del all_win, all_mask, all_tgt, doc_off, surprisals, space_lp, word_surps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-2 word surprisal inference for original stories with pronoun-aware windows")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="directory containing token-level <corpus>.tsv annotation files",
    )
    parser.add_argument(
        "--corpus",
        choices=[*SUPPORTED_CORPORA, "all"],
        help=(
            "Use built-in paths for <annotations-dir>/<corpus>.tsv and the "
            "matching raw corpus. Use 'all' for every corpus available in the "
            "annotations directory."
        ),
    )
    parser.add_argument("--input_file")
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
            "One-based single story selector. Prefer --story_ids for multiple stories. "
            "If omitted, all pronoun TSV stories are run."
        ),
    )
    parser.add_argument("--output_dir")
    parser.add_argument("--context_sizes", type=int, nargs="+", default=DEFAULT_CONTEXT_SIZES)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
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
        default=None,
        help=(
            "Per-token pronoun replacement TSV with story, token_index, and "
            "pronoun_replacement columns. Defaults to "
            "outputs/pre_inference/conditions/repeated_coref/token_annotations/ns.tsv."
        ),
    )
    args = parser.parse_args()
    args.annotations_dir = resolve_annotations_dir(args.annotations_dir)

    if args.story_ids is not None and args.story_number is not None:
        raise ValueError("Use either --story_ids or --story_number, not both.")

    if args.corpus is not None and any(
        value is not None for value in (args.input_file, args.pronoun_tsv)
    ):
        raise ValueError("Use either --corpus or explicit --input_file/--pronoun_tsv.")

    if args.corpus is None:
        if args.input_file is None or args.output_dir is None:
            raise ValueError("--input_file and --output_dir are required unless --corpus is used.")
        run_specs = [
            (
                args.input_file,
                args.pronoun_tsv or DEFAULT_ALL_STORY_PRONOUNS_TSV,
                args.output_dir,
            )
        ]
    else:
        corpora = corpora_for_mode(args.annotations_dir, args.corpus)
        if args.output_dir is not None and len(corpora) > 1:
            raise ValueError("--output_dir can only be used with one --corpus at a time.")
        run_specs = [
            default_corpus_paths(args.annotations_dir, corpus, args.model_name)
            for corpus in corpora
        ]
        if args.output_dir is not None:
            input_file, pronoun_tsv, _ = run_specs[0]
            run_specs = [(input_file, pronoun_tsv, args.output_dir)]

    device = torch.device(args.device)
    dtype = torch.float16

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

    for input_file, pronoun_tsv, output_dir in run_specs:
        run_inference_for_inputs(
            input_file=input_file,
            pronoun_tsv=pronoun_tsv,
            output_dir=output_dir,
            args=args,
            tokenizer=tokenizer,
            model=model,
            context_sizes=context_sizes,
            device=device,
            dtype=dtype,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
