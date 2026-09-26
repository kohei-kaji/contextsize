#!/usr/bin/env python3
"""
Word-by-word surprisal with Whitespace-Trailing (WT) decoding (Oh & Schuler, 2024; EMNLP) for GPT-2, Pythia, OPT, and GPT-Neo model families.

Window convention (n-gram style):
  context_size = N means an N-token window: (N-1) context tokens + 1 target token.
  e.g. context_size=5 → 4 context tokens predict 1 target token (= 5-gram).

Algorithm:
  1. For each token position t in each document, build a fixed-length window of N tokens: tokens[max(0, t+1-N) : t+1], left-padded to length N.
  2. Forward pass → from each window extract:
       - surprisal[t]        = -log2 P(tok_t | logits[-2])   [if context exists]
       - space_lp_after[t]   = log P(space | logits[-1])
  3. Aggregate into whitespace-delimited words with WT correction:
       S_WT(w_i) = sum_surp(tokens of w_i) - a_bits + b_bits
       where a_bits = -space_lp_after[t_start - 1] / ln2
             b_bits = -space_lp_after[t_end - 1]   / ln2
     For the first word of a document, a_bits is omitted.

Notes:
  - BOS token is NOT added (add_special_tokens=False) following Kuribayashi et al. (2022).
  - OPT models do not accept position_ids; positions are inferred from attention_mask inside the model's own forward pass.

Usage:
  python word_surprisal.py \
    --model_name gpt2-large \
    --input_file documents.txt \
    --context_sizes 5 10 20 50 100 \
    --add_full_context \
    --output_dir output/gpt2-large \
    --gpus 0 1 2 3 \
    --max_batch_tokens 131072 \
    --dtype float16
"""

import argparse
import csv
import gc
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


LN2 = np.log(2.0)
TSV_OPTIONS = {
    "delimiter": "\t",
    "quoting": csv.QUOTE_NONE,
    "quotechar": None,
    "escapechar": "\\",
}

DEFAULT_STORY0_PRONOUNS_TSV = os.path.join(
    os.path.dirname(__file__), "story0_pronouns.tsv"
)
DEFAULT_ALL_STORY_PRONOUNS_TSV = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "deepseek_pronoun_results",
        "pronouns_ns.tsv",
    )
)


MODEL_REGISTRY = {
    # GPT-2
    "gpt2": "gpt2", "gpt2-medium": "gpt2-medium", "gpt2-large": "gpt2-large", "gpt2-xl": "gpt2-xl",
    # Pythia
    "pythia-14m": "EleutherAI/pythia-14m", "pythia-31m": "EleutherAI/pythia-31m",
    "pythia-70m": "EleutherAI/pythia-70m", "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m", "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b", "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b", "pythia-12b": "EleutherAI/pythia-12b",
    # OPT
    "opt-125m": "facebook/opt-125m", "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b", "opt-2.7b": "facebook/opt-2.7b",
    "opt-6.7b": "facebook/opt-6.7b", "opt-13b": "facebook/opt-13b",
    "opt-30b": "facebook/opt-30b",
    "opt-66b": "facebook/opt-66b",
    # GPT-Neo / GPT-J
    "gpt-neo-125m": "EleutherAI/gpt-neo-125M",
    "gpt-neo-1.3b": "EleutherAI/gpt-neo-1.3B",
    "gpt-neo-2.7b": "EleutherAI/gpt-neo-2.7B",
    "gpt-j-6b":     "EleutherAI/gpt-j-6B",
}

_MODEL_MAX_CTX_FALLBACK = {
    "gpt2": 1025, "gpt2-medium": 1025, "gpt2-large": 1025, "gpt2-xl": 1025,
    "gpt-neo-125m": 2048, "gpt-neo-1.3b": 2048, "gpt-neo-2.7b": 2048, "gpt-j-6b": 2048,
    "opt-125m": 2048, "opt-350m": 2048, "opt-1.3b": 2048, "opt-2.7b": 2048,
    "opt-6.7b": 2048, "opt-13b": 2048, "opt-30b": 2048, "opt-66b": 2048,
}


def get_space_token_ids(tokenizer) -> torch.Tensor:
    vocab = tokenizer.get_vocab()
    ids = sorted(
        idx for tok, idx in vocab.items()
        if tok.startswith("Ġ")
    )
    if not ids:
        raise ValueError("No space-prefixed tokens found in tokenizer vocabulary.")
    return torch.tensor(ids, dtype=torch.long)


def setup_tokenizer_and_model(tokenizer, model):
    eos = tokenizer.eos_token or "<|endoftext|>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = eos
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    return tokenizer, model


def pretokenize_documents(documents: list[str], tokenizer, causal_analysis=False):
    """Tokenize all documents and build word-token alignment.

    Returns:
        doc_token_ids:  list[list[int]]
        doc_word_spans: list[list[tuple[int,int]]]  -- (tok_start, tok_end) per word
        doc_words:      list[list[str]]
        total_tokens:   int
        total_words:    int
    """
    doc_token_ids = []
    doc_word_spans = []
    doc_words = []
    total_tokens = 0
    total_words = 0

    for text in documents:
        text = text.strip()
        if not text:
            doc_token_ids.append([])
            doc_word_spans.append([])
            doc_words.append([])
            continue

        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tids: list[int] = enc["input_ids"]
        offsets: list[tuple[int, int]] = enc["offset_mapping"]

        words = text.split()
        word_char_spans = []
        pos = 0
        for w in words:
            idx = text.index(w, pos)
            word_char_spans.append((idx, idx + len(w)))
            pos = idx + len(w)

        word_tok_spans = []
        tok_ptr = 0
        for wc_start, wc_end in word_char_spans:
            while tok_ptr < len(offsets) and offsets[tok_ptr][1] <= wc_start:
                tok_ptr += 1
            t_start = tok_ptr
            while tok_ptr < len(offsets) and offsets[tok_ptr][0] < wc_end:
                tok_ptr += 1
            word_tok_spans.append((t_start, tok_ptr))

        doc_token_ids.append(tids)
        doc_word_spans.append(word_tok_spans)
        doc_words.append(words)
        total_tokens += len(tids)
        total_words += len(words)

    return doc_token_ids, doc_word_spans, doc_words, total_tokens, total_words

import glob, re

def load_mask_coref_tokens(coref_dir: str, num_docs: int) -> list[set[int]]:
    result = []
    for doc_i in range(1, num_docs + 1):
        path = os.path.join(coref_dir, f"gpt2_coref_doc{doc_i}.conllu")
        coref_positions = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                token_idx = int(parts[0]) - 1  # convert 1-based → 0-based
                if parts[4] != "_":            # col 5 = coref chain IDs
                    coref_positions.add(token_idx)
        result.append(coref_positions)
    return result


def build_per_token_windows_with_causal_mask(
    doc_token_ids: list[list[int]],
    window_size: int,
    pad_token_id: int,
    mask_coref_tokens: list[set[int]],
):
    """ masking out co-ref tokens in the context window, so that the model can only attend to non-coref tokens during inference. """
    all_windows, all_attn_masks, all_targets, doc_offsets = build_per_token_windows(doc_token_ids, window_size, pad_token_id)

    row = 0
    for doc_i, tids in enumerate(doc_token_ids):
        coref_tokens = mask_coref_tokens[doc_i]
        for t in range(len(tids)):
            win_start = max(0, t + 1 - window_size)
            for local_i, doc_pos in enumerate(range(win_start, t)):
                if doc_pos in coref_tokens:
                    col = window_size - (t + 1 - win_start) + local_i
                    all_attn_masks[row, col] = 0
            row += 1

    return all_windows, all_attn_masks, all_targets, doc_offsets


def build_per_token_windows(
    doc_token_ids: list[list[int]],
    window_size: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build fixed-length left-padded windows for every token in every document.

    For token at document position t, the window is:
        tokens[max(0, t+1-window_size) : t+1]
    left-padded to window_size.

    Returns:
        all_windows:    LongTensor [total_tokens, window_size]
        all_attn_masks: LongTensor [total_tokens, window_size]
        all_targets:    LongTensor [total_tokens]
        doc_offsets:    list[int]  (doc d's tokens at indices [doc_offsets[d], doc_offsets[d+1]))
    """
    total = sum(len(tids) for tids in doc_token_ids)

    all_windows = torch.full((total, window_size), pad_token_id, dtype=torch.long)
    all_attn_masks = torch.zeros((total, window_size), dtype=torch.long)
    all_targets = torch.zeros(total, dtype=torch.long)

    doc_offsets = [0]
    row = 0

    for tids in doc_token_ids:
        L = len(tids)
        for t in range(L):
            win_start = max(0, t + 1 - window_size)
            win_tokens = tids[win_start: t + 1]
            win_len = len(win_tokens)

            # Left-pad: real tokens at end of window
            all_windows[row, window_size - win_len:] = torch.tensor(
                win_tokens, dtype=torch.long
            )
            all_attn_masks[row, window_size - win_len:] = 1
            all_targets[row] = tids[t]
            row += 1

        doc_offsets.append(row)

    assert row == total
    return all_windows, all_attn_masks, all_targets, doc_offsets


def load_pronoun_replacements_by_story(
    pronoun_tsv_path: str | None = None,
) -> dict[int, list[str | None]]:
    """Load per-token pronoun replacements grouped by zero-based story id.

    The preferred TSV shape is the all-story format:

        story, token_index, word_index, token, pronoun_replacement

    The returned dictionary is keyed by the TSV's zero-based story id. Each
    value is indexed by token position and contains a replacement pronoun, or
    None when the TSV has "NA". Legacy single-story files without a story
    column are treated as story 0.
    """
    if pronoun_tsv_path is None:
        pronoun_tsv_path = DEFAULT_ALL_STORY_PRONOUNS_TSV

    replacements_by_story: dict[int, list[str | None]] = {}
    expected_token_index_by_story: dict[int, int] = {}

    with open(pronoun_tsv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, **TSV_OPTIONS)
        required = {"token_index", "pronoun_replacement"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{pronoun_tsv_path} must contain columns: {sorted(required)}"
            )
        has_story_column = "story" in reader.fieldnames

        for row_number, row in enumerate(reader, start=2):
            try:
                story_id = int(row["story"]) if has_story_column else 0
                token_index = int(row["token_index"])
            except ValueError as exc:
                raise ValueError(
                    f"{pronoun_tsv_path} has invalid story/token_index at row "
                    f"{row_number}."
                ) from exc

            expected_token_index = expected_token_index_by_story.setdefault(story_id, 0)
            if token_index != expected_token_index:
                raise ValueError(
                    f"{pronoun_tsv_path} story {story_id} has "
                    f"token_index={token_index} at row {row_number}; expected "
                    f"{expected_token_index}."
                )

            replacement = row["pronoun_replacement"].strip()
            replacements_by_story.setdefault(story_id, []).append(
                None if replacement == "NA" else replacement
            )
            expected_token_index_by_story[story_id] = expected_token_index + 1

    if not replacements_by_story:
        raise ValueError(f"{pronoun_tsv_path} did not contain any pronoun rows.")

    return replacements_by_story


def validate_pronoun_token_counts(
    story_ids: list[int],
    documents: list[str],
    tokenizer,
    replacements_by_story: dict[int, list[str | None]] | None = None,
    pronoun_tsv_path: str | None = None,
) -> list[list[str | None]]:
    """Validate selected documents against per-story pronoun replacement rows.

    story_ids must line up with documents. The return value has the same order
    as documents and can be passed directly to
    build_per_token_windows_with_pronoun_replacements(...).
    """
    if len(story_ids) != len(documents):
        raise ValueError(
            "story_ids and documents must have the same length "
            f"({len(story_ids)} != {len(documents)})."
        )
    if replacements_by_story is None:
        replacements_by_story = load_pronoun_replacements_by_story(pronoun_tsv_path)
    if pronoun_tsv_path is None:
        pronoun_tsv_path = DEFAULT_ALL_STORY_PRONOUNS_TSV

    doc_pronoun_replacements = []
    for story_id, document in zip(story_ids, documents):
        if story_id not in replacements_by_story:
            raise ValueError(f"{pronoun_tsv_path} does not contain story {story_id}.")

        token_ids = tokenizer(document.strip(), add_special_tokens=False)["input_ids"]
        replacements = replacements_by_story[story_id]
        if len(token_ids) != len(replacements):
            raise ValueError(
                f"Token count mismatch for story {story_id}: tokenizer produced "
                f"{len(token_ids)} tokens, but {pronoun_tsv_path} contains "
                f"{len(replacements)} token rows."
            )
        doc_pronoun_replacements.append(replacements)

    return doc_pronoun_replacements

def _validate_pronoun_replacement_inputs(
    doc_token_ids: list[list[int]],
    doc_pronoun_replacements: list[list[str | None]],
) -> None:
    if len(doc_token_ids) != len(doc_pronoun_replacements):
        raise ValueError(
            "doc_token_ids and doc_pronoun_replacements must have the same "
            f"number of documents ({len(doc_token_ids)} != "
            f"{len(doc_pronoun_replacements)})."
        )

    for doc_i, (tids, replacements) in enumerate(
        zip(doc_token_ids, doc_pronoun_replacements)
    ):
        if len(tids) != len(replacements):
            raise ValueError(
                f"Document {doc_i} has {len(tids)} tokens but "
                f"{len(replacements)} pronoun replacement entries."
            )


def _build_replacement_spans(
    replacements: list[str | None],
) -> dict[int, tuple[int, int, str]]:
    replacement_spans: dict[int, tuple[int, int, str]] = {}
    doc_pos = 0
    while doc_pos < len(replacements):
        replacement = replacements[doc_pos]
        if replacement is None:
            doc_pos += 1
            continue

        span_start = doc_pos
        span_end = doc_pos + 1
        while span_end < len(replacements) and replacements[span_end] == replacement:
            span_end += 1

        for span_pos in range(span_start, span_end):
            replacement_spans[span_pos] = (span_start, span_end, replacement)
        doc_pos = span_end

    return replacement_spans


def _make_pronoun_token_id_getter(tokenizer):
    pronoun_token_cache: dict[tuple[str, bool, bool], list[int]] = {}

    def format_pronoun_for_position(pronoun: str, starts_sentence: bool) -> str:
        if pronoun == "i":
            return "I"
        if starts_sentence:
            return pronoun[:1].upper() + pronoun[1:]
        return pronoun

    def pronoun_token_ids(
        pronoun: str,
        starts_sentence: bool,
        at_doc_start: bool,
    ) -> list[int]:
        cache_key = (pronoun, starts_sentence, at_doc_start)
        if cache_key not in pronoun_token_cache:
            formatted_pronoun = format_pronoun_for_position(pronoun, starts_sentence)
            text = formatted_pronoun if at_doc_start else " " + formatted_pronoun
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not token_ids:
                raise ValueError(f"Replacement pronoun {pronoun!r} produced no tokens.")
            pronoun_token_cache[cache_key] = token_ids
        return pronoun_token_cache[cache_key]

    return pronoun_token_ids


def _span_starts_sentence(tokenizer, tids: list[int], span_start: int) -> bool:
    if span_start == 0:
        return True

    prefix_text = tokenizer.decode(tids[max(0, span_start - 16):span_start]).rstrip()
    if not prefix_text:
        return True

    opening_marks = set("\"'“‘([{:")
    while prefix_text and prefix_text[-1] in opening_marks:
        prefix_text = prefix_text[:-1].rstrip()
    if not prefix_text:
        return True

    return prefix_text[-1] in ".!?"


def _collapse_pronoun_replacement_window(
    tids: list[int],
    replacement_spans: dict[int, tuple[int, int, str]],
    win_start: int,
    target_pos: int,
    tokenizer,
    pronoun_token_ids,
) -> list[int]:
    collapsed_tokens: list[int] = []
    doc_pos = win_start

    while doc_pos <= target_pos:
        span = replacement_spans.get(doc_pos)
        if span is None:
            collapsed_tokens.append(tids[doc_pos])
            doc_pos += 1
            continue

        span_start, span_end, replacement = span
        collapse_span = (
            doc_pos == span_start
            and win_start <= span_start
            and span_end <= target_pos
        )

        if collapse_span:
            collapsed_tokens.extend(
                pronoun_token_ids(
                    replacement,
                    starts_sentence=_span_starts_sentence(tokenizer, tids, span_start),
                    at_doc_start=(span_start == 0),
                )
            )
            doc_pos = span_end
        else:
            collapsed_tokens.append(tids[doc_pos])
            doc_pos += 1

    return collapsed_tokens


def pronoun_replacements_with_same_context(
    doc_token_ids: list[list[int]],
    window_size: int,
    pad_token_id: int,
    tokenizer,
    doc_pronoun_replacements: list[list[str | None]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build pronoun-collapsed windows with extra left context when needed.

    This is like build_per_token_windows_with_pronoun_replacements, except if
    collapsing spans makes the window shorter than window_size, earlier document
    tokens are pulled in until the collapsed window is full or the document
    start is reached. Spans in that extra context are collapsed only when the
    complete span is contained in the expanded window.
    """
    _validate_pronoun_replacement_inputs(doc_token_ids, doc_pronoun_replacements)

    total = sum(len(tids) for tids in doc_token_ids)
    all_windows = torch.full((total, window_size), pad_token_id, dtype=torch.long)
    all_attn_masks = torch.zeros((total, window_size), dtype=torch.long)
    all_targets = torch.zeros(total, dtype=torch.long)

    doc_offsets = [0]
    row = 0
    pronoun_token_ids = _make_pronoun_token_id_getter(tokenizer)

    for tids, replacements in zip(doc_token_ids, doc_pronoun_replacements):
        replacement_spans = _build_replacement_spans(replacements)

        for t in range(len(tids)):
            win_start = max(0, t + 1 - window_size)
            collapsed_tokens = _collapse_pronoun_replacement_window(
                tids,
                replacement_spans,
                win_start,
                t,
                tokenizer,
                pronoun_token_ids,
            )

            while len(collapsed_tokens) < window_size and win_start > 0:
                win_start -= 1
                collapsed_tokens = _collapse_pronoun_replacement_window(
                    tids,
                    replacement_spans,
                    win_start,
                    t,
                    tokenizer,
                    pronoun_token_ids,
                )

            if len(collapsed_tokens) > window_size:
                collapsed_tokens = collapsed_tokens[-window_size:]

            win_len = len(collapsed_tokens)
            all_windows[row, window_size - win_len:] = torch.tensor(
                collapsed_tokens, dtype=torch.long
            )
            all_attn_masks[row, window_size - win_len:] = 1
            all_targets[row] = tids[t]
            row += 1

        doc_offsets.append(row)

    assert row == total
    return all_windows, all_attn_masks, all_targets, doc_offsets


def build_per_token_windows_with_pronoun_replacements(
    doc_token_ids: list[list[int]],
    window_size: int,
    pad_token_id: int,
    tokenizer,
    doc_pronoun_replacements: list[list[str | None]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build windows while collapsing annotated token spans to pronouns.

    This follows build_per_token_windows, except each window is first scanned
    for full contiguous document spans with the same non-NA
    pronoun_replacement. Each such span is replaced by a single occurrence of
    that pronoun only when the complete span is inside the context portion of
    the window, excluding the current target token.

    doc_pronoun_replacements must have the same nested shape as doc_token_ids.
    For an all-story pronoun TSV such as pronouns_ns.tsv, call
    validate_pronoun_token_counts(...) with the selected story ids and
    documents, then pass the returned nested list here.
    """
    _validate_pronoun_replacement_inputs(doc_token_ids, doc_pronoun_replacements)

    total = sum(len(tids) for tids in doc_token_ids)
    all_windows = torch.full((total, window_size), pad_token_id, dtype=torch.long)
    all_attn_masks = torch.zeros((total, window_size), dtype=torch.long)
    all_targets = torch.zeros(total, dtype=torch.long)

    doc_offsets = [0]
    row = 0
    pronoun_token_ids = _make_pronoun_token_id_getter(tokenizer)

    for tids, replacements in zip(doc_token_ids, doc_pronoun_replacements):
        replacement_spans = _build_replacement_spans(replacements)

        for t in range(len(tids)):
            win_start = max(0, t + 1 - window_size)
            collapsed_tokens = _collapse_pronoun_replacement_window(
                tids,
                replacement_spans,
                win_start,
                t,
                tokenizer,
                pronoun_token_ids,
            )

            if len(collapsed_tokens) > window_size:
                collapsed_tokens = collapsed_tokens[-window_size:]

            win_len = len(collapsed_tokens)
            all_windows[row, window_size - win_len:] = torch.tensor(
                collapsed_tokens, dtype=torch.long
            )
            all_attn_masks[row, window_size - win_len:] = 1
            all_targets[row] = tids[t]
            row += 1

        doc_offsets.append(row)

    assert row == total
    return all_windows, all_attn_masks, all_targets, doc_offsets


def _get_model_max_ctx(model_name: str, model) -> int:
    """Return the maximum usable context length for this model."""
    # 1. Check fallback table first (most reliable)
    if model_name in _MODEL_MAX_CTX_FALLBACK:
        return _MODEL_MAX_CTX_FALLBACK[model_name]
    # 2. Try model config attributes in priority order
    cfg = model.config
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        val = getattr(cfg, attr, None)
        if val is not None:
            # OPT's max_position_embeddings includes a 2-token offset
            if getattr(cfg, "model_type", "") == "opt":
                return val - 2
            return val
    # 3. Tokenizer model_max_length as last resort
    return 2048


def _model_accepts_position_ids(model) -> bool:
    """Return False for models that compute positions from attention_mask internally.

    OPT's forward() does not accept position_ids and raises TypeError if passed.
    All other supported families (GPT-2, GPT-Neo, GPT-J, Pythia) do accept it.
    """
    m = model.module if hasattr(model, "module") else model
    model_type = getattr(m.config, "model_type", "").lower()
    return model_type != "opt"


@torch.inference_mode()
def compute_token_level_quantities(
    all_windows: torch.Tensor,       # [N, W]
    all_attn_masks: torch.Tensor,    # [N, W]
    all_targets: torch.Tensor,       # [N]
    model: torch.nn.Module,
    space_ids: torch.Tensor,         # [S] on CPU
    device: torch.device,
    batch_size: int,
    use_position_ids: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and extract per-token surprisal & space probability.

    Args:
        use_position_ids: Pass explicit position_ids to the model.
            Set False for OPT, which computes positions from attention_mask
            internally and rejects the position_ids kwarg.

    Returns:
        surprisals:      np.ndarray [N]  -- surprisal in bits (0.0 if no context)
        space_lp_after:  np.ndarray [N]  -- log P(space | up to and including this token)
    """
    N, W = all_windows.shape
    surprisals = np.zeros(N, dtype=np.float64)
    space_lp_after = np.zeros(N, dtype=np.float64)

    space_ids_dev = space_ids.to(device)
    num_batches = (N + batch_size - 1) // batch_size

    for b_idx in range(num_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, N)

        input_ids = all_windows[start:end].to(device)      # [B, W]
        attn_mask = all_attn_masks[start:end].to(device)    # [B, W]
        targets = all_targets[start:end].to(device)         # [B]

        # Position IDs for left-padded sequences: cumsum of attn_mask - 1.
        # Not passed to OPT because it derives positions from attention_mask itself.
        forward_kwargs = dict(input_ids=input_ids, attention_mask=attn_mask)
        if use_position_ids:
            position_ids = attn_mask.long().cumsum(dim=-1) - 1
            position_ids.clamp_(min=0)
            forward_kwargs["position_ids"] = position_ids

        logits = model(**forward_kwargs).logits  # [B, W, V]

        # ---- surprisal: logits[:, -2, :] predicts target at position -1 ----
        if W >= 2:
            pred_logits = logits[:, -2, :]                              # [B, V]
            log_probs = F.log_softmax(pred_logits, dim=-1)              # [B, V]
            token_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

            # Valid only if position -2 has a real token (not padding)
            has_context = attn_mask[:, -2].bool()                       # [B]
            surp_bits = torch.where(
                has_context,
                -token_lp / LN2,
                torch.zeros_like(token_lp),
            )
            surprisals[start:end] = surp_bits.cpu().double().numpy()

        # ---- space_lp_after: logits[:, -1, :] predicts next position ----
        last_logits = logits[:, -1, :]                                  # [B, V]
        log_Z = torch.logsumexp(last_logits, dim=-1, keepdim=True)      # [B, 1]
        space_logits = last_logits[:, space_ids_dev]                     # [B, S]
        space_lp = torch.logsumexp(space_logits, dim=-1) - log_Z.squeeze(1)  # [B]
        space_lp_after[start:end] = space_lp.cpu().double().numpy()

        del logits, input_ids, attn_mask

        if (b_idx + 1) % 100 == 0 or (b_idx + 1) == num_batches:
            print(f"    [{end}/{N}] tokens processed", flush=True)

    return surprisals, space_lp_after


def aggregate_sum_surprisal(
    surprisals: np.ndarray,
    doc_token_ids: list[list[int]],
    doc_word_spans: list[list[tuple[int, int]]],
    doc_offsets: list[int],
) -> np.ndarray:
    """Aggregate per-token surprisals into per-word surprisals by simple summation."""
    total_words = sum(len(ws) for ws in doc_word_spans)
    result = np.zeros(total_words, dtype=np.float64)

    wi = 0
    for doc_i, (tids, wspans) in enumerate(zip(doc_token_ids, doc_word_spans)):
        if not tids:
            continue
        off = doc_offsets[doc_i]
        for t_start, t_end in wspans:
            result[wi] = surprisals[off + t_start: off + t_end].sum()
            wi += 1

    assert wi == total_words
    return result


def aggregate_wt_surprisal(
    surprisals: np.ndarray,
    space_lp_after: np.ndarray,
    doc_token_ids: list[list[int]],
    doc_word_spans: list[list[tuple[int, int]]],
    doc_offsets: list[int],
) -> np.ndarray:
    """Aggregate per-token surprisals into per-word WT-corrected surprisals.

    S_WT(w_i) = sum(surprisal[t] for t in word tokens)
                - a_bits + b_bits

    a_bits = -space_lp_after[t_start - 1] / ln2   (omitted for first word)
    b_bits = -space_lp_after[t_end - 1]   / ln2
    """
    total_words = sum(len(ws) for ws in doc_word_spans)
    result = np.zeros(total_words, dtype=np.float64)

    wi = 0
    for doc_i, (tids, wspans) in enumerate(zip(doc_token_ids, doc_word_spans)):
        if not tids:
            continue
        off = doc_offsets[doc_i]

        for w_i, (t_start, t_end) in enumerate(wspans):
            base_surp = surprisals[off + t_start: off + t_end].sum()

            b_lp = space_lp_after[off + t_end - 1]
            b_bits = -b_lp / LN2

            if w_i == 0:
                result[wi] = base_surp + b_bits
            else:
                a_lp = space_lp_after[off + t_start - 1]
                a_bits = -a_lp / LN2
                result[wi] = base_surp - a_bits + b_bits

            wi += 1

    assert wi == total_words
    return result


def process_all_context_sizes(
    doc_token_ids, doc_word_spans, total_tokens, total_words,
    context_sizes, model, tokenizer, space_ids, device,
    max_batch_tokens, output_dir, use_position_ids=True,
):
    pad_id = tokenizer.pad_token_id

    for ctx_size in sorted(context_sizes):
        print(f"\n{'='*60}")
        print(f"  context_size={ctx_size}  ({ctx_size-1} context + 1 target)")
        print(f"{'='*60}")
        t0 = time.time()

        print(f"  Building {total_tokens} windows of length {ctx_size}...")
        all_win, all_mask, all_tgt, doc_off = build_per_token_windows(doc_token_ids, ctx_size, pad_id)

        batch_size = max(1, max_batch_tokens // ctx_size)
        num_batches = (total_tokens + batch_size - 1) // batch_size
        print(f"  batch_size={batch_size}, {num_batches} batches")

        surprisals, space_lp = compute_token_level_quantities(
            all_win, all_mask, all_tgt, model, space_ids, device, batch_size,
            use_position_ids=use_position_ids,
        )

        word_surps = aggregate_wt_surprisal(
            surprisals, space_lp, doc_token_ids, doc_word_spans, doc_off,
        )

        out_path = os.path.join(output_dir, f"context_{ctx_size}.txt")
        np.savetxt(out_path, word_surps, fmt="%.8f")

        elapsed = time.time() - t0
        print(f"  ✓ {out_path}  ({total_words} words, {elapsed:.1f}s)")

        del all_win, all_mask, all_tgt, doc_off, surprisals, space_lp, word_surps
        gc.collect()
        torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser(
        description="Word-by-word surprisal with WT decoding (Oh & Schuler, 2024)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Context size convention (n-gram style):
  context_size = N  →  window of N tokens = (N-1) context + 1 target
  e.g. --context_sizes 5 10 20 50 100 --add_full_context
       5    = 4 context tokens + 1 target (≈ 5-gram)
       max  = model's actual max context window (added by --add_full_context)

Notes:
  - BOS token is NOT added during tokenisation (context-limitation setting).
  - Context sizes larger than the model's max are automatically skipped.
  - --add_full_context appends the model's actual max context to the list.
""",
    )
    parser.add_argument("--model_name", type=str, required=True,
        help=f"Model name: {', '.join(sorted(MODEL_REGISTRY.keys()))}; or HF path.")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--context_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--add_full_context", action="store_true",
        help="Automatically append the model's actual max context length to --context_sizes.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--max_batch_tokens", type=int, default=131072)
    parser.add_argument("--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use_device_map", action="store_true")
    parser.add_argument("--offload_folder", type=str, default=None,
        help="Directory for disk-offloaded weights when the model exceeds GPU memory "
             "(required for very large models like opt-66b on limited VRAM).")
    args = parser.parse_args()

    model_path = MODEL_REGISTRY.get(args.model_name, args.model_name)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {model_path}" + (f" (rev={args.revision})" if args.revision else ""))
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=args.revision, use_fast=True)

    use_device_map = args.use_device_map or len(args.gpus) > 1
    large_keys = ["6.9b", "12b", "6.7b", "13b", "30b", "66b"]
    if any(k in args.model_name.lower() for k in large_keys):
        use_device_map = True

    if use_device_map:
        max_memory = {
            g: f"{torch.cuda.get_device_properties(g).total_memory // (1024**3) - 2}GiB"
            for g in args.gpus
        }
        load_kwargs = dict(
            revision=args.revision,
            torch_dtype=torch_dtype,
            device_map="auto",
            max_memory=max_memory,
        )
        if args.offload_folder:
            os.makedirs(args.offload_folder, exist_ok=True)
            load_kwargs["offload_folder"] = args.offload_folder
            print(f"  Disk offload folder: {args.offload_folder}")
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        device = next(model.parameters()).device
    else:
        device = torch.device(f"cuda:{args.gpus[0]}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=args.revision, torch_dtype=torch_dtype,
        ).to(device)
        if len(args.gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=args.gpus)

    tokenizer, model = setup_tokenizer_and_model(tokenizer, model)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s  dtype={torch_dtype}")

    max_ctx = _get_model_max_ctx(args.model_name, model)
    use_pos_ids = _model_accepts_position_ids(model)
    print(f"  Max context: {max_ctx} tokens")
    print(f"  Use position_ids: {use_pos_ids}")

    context_sizes = sorted(set(args.context_sizes))
    if args.add_full_context and max_ctx not in context_sizes:
        context_sizes.append(max_ctx)
        context_sizes.sort()
        print(f"  Added full context ({max_ctx}) via --add_full_context")
    invalid = [c for c in context_sizes if c > max_ctx]
    if invalid:
        print(f"  Skipping context sizes > {max_ctx}: {invalid}")
    context_sizes = [c for c in context_sizes if c <= max_ctx]
    print(f"  Context sizes to compute: {context_sizes}")

    space_ids = get_space_token_ids(tokenizer)
    print(f"  Space-prefixed tokens: {len(space_ids)}")

    with open(args.input_file, "r", encoding="utf-8") as f:
        documents = [line.rstrip("\n") for line in f if line.strip()]
    print(f"  {len(documents)} documents")

    print("Pre-tokenizing...")
    t_tok = time.time()
    doc_tids, doc_wspans, doc_words, total_tokens, total_words = \
        pretokenize_documents(documents, tokenizer)
    print(f"  {total_tokens} tokens, {total_words} words ({time.time() - t_tok:.1f}s)")

    info_path = os.path.join(args.output_dir, "word_info.tsv")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("story\tzone\tword\tnum_tokens\n")
        for di, (wspans, ws) in enumerate(zip(doc_wspans, doc_words)):
            for wi, ((ts, te), w) in enumerate(zip(wspans, ws)):
                f.write(f"{di}\t{wi}\t{w}\t{te - ts}\n")
    print(f"  → {info_path}")

    t_main = time.time()
    process_all_context_sizes(
        doc_tids, doc_wspans, total_tokens, total_words,
        context_sizes, model, tokenizer, space_ids, device,
        args.max_batch_tokens, args.output_dir,
        use_position_ids=use_pos_ids,
    )
    print(f"\nDone. Total: {time.time() - t_main:.1f}s → {args.output_dir}/")


if __name__ == "__main__":
    main()
