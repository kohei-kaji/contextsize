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
import gc
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


LN2 = np.log(2.0)


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


def pretokenize_documents(documents: list[str], tokenizer):
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