#!/usr/bin/env python3
"""
Launch word_surprisal.py for multiple models, distributing across GPUs.

Strategy:
  - Small models (≤ 1.4B): assign one model per GPU, run in parallel
  - Large models (> 1.4B): use multiple GPUs per model via device_map

Usage:
  # Process all GPT-2 + Pythia models on 4 GPUs:
  python run_all.py \
    --input_file documents.txt \
    --output_base output \
    --context_sizes 5 10 20 50 100 1024 \
    --gpus 0 1 2 3 \
    --families gpt2 pythia \
    --max_batch_tokens 131072

  # Single model:
  python run_all.py \
    --input_file documents.txt \
    --output_base output \
    --context_sizes 5 10 20 50 100 \
    --gpus 0 \
    --models gpt2-large
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


# Model families grouped by approximate parameter count
FAMILIES = {
    "gpt2": [
        ("gpt2",        "small",  124_000_000),
        ("gpt2-medium", "medium", 355_000_000),
        ("gpt2-large",  "large",  774_000_000),
        ("gpt2-xl",     "xl",     1_500_000_000),
    ],
    "pythia": [
        ("pythia-14m",   "14m",  14_000_000),
        ("pythia-31m",   "31m",  31_000_000),
        ("pythia-70m",   "70m",  70_000_000),
        ("pythia-160m",  "160m", 160_000_000),
        ("pythia-410m",  "410m", 410_000_000),
        ("pythia-1b",    "1b",   1_000_000_000),
        ("pythia-1.4b",  "1.4b", 1_400_000_000),
        ("pythia-2.8b",  "2.8b", 2_800_000_000),
        ("pythia-6.9b",  "6.9b", 6_900_000_000),
        ("pythia-12b",   "12b",  12_000_000_000),
    ],
    "opt": [
        ("opt-125m",  "125m",  125_000_000),
        ("opt-350m",  "350m",  350_000_000),
        ("opt-1.3b",  "1.3b",  1_300_000_000),
        ("opt-2.7b",  "2.7b",  2_700_000_000),
        ("opt-6.7b",  "6.7b",  6_700_000_000),
        ("opt-13b",   "13b",   13_000_000_000),
        ("opt-30b",   "30b",   30_000_000_000),
        ("opt-66b",   "66b",   66_000_000_000),
    ],
    # GPT-Neo/J: same BPE tokenizer as GPT-2, max context 2048
    "gpt-neo": [
        ("gpt-neo-125m", "125m", 125_000_000),
        ("gpt-neo-1.3b", "1.3b", 1_300_000_000),
        ("gpt-neo-2.7b", "2.7b", 2_700_000_000),
        ("gpt-j-6b",     "6b",   6_000_000_000),
    ],
}

SMALL_THRESHOLD = 1_500_000_000  # ≤ 1.5B → fits on one GPU


def run_model(model_name, input_file, output_dir, context_sizes, gpus,
              max_batch_tokens, dtype, revision=None, add_full_context=False,
              offload_folder=None):
    """Run word_surprisal.py for a single model."""
    # CUDA_VISIBLE_DEVICES remaps physical GPUs to logical indices 0,1,2,...
    # So we pass logical indices to --gpus, not the physical GPU numbers.
    logical_gpus = list(range(len(gpus)))
    cmd = [
        sys.executable, "word_surprisal.py",
        "--model_name", model_name,
        "--input_file", input_file,
        "--output_dir", output_dir,
        "--context_sizes", *[str(c) for c in context_sizes],
        "--gpus", *[str(g) for g in logical_gpus],
        "--max_batch_tokens", str(max_batch_tokens),
        "--dtype", dtype,
    ]
    if revision:
        cmd.extend(["--revision", revision])
    if len(gpus) > 1:
        cmd.append("--use_device_map")
    if add_full_context:
        cmd.append("--add_full_context")
    if offload_folder:
        # Use a per-model subdirectory to avoid conflicts when running in parallel
        cmd.extend(["--offload_folder", os.path.join(offload_folder, model_name)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

    print(f"[LAUNCH] {model_name} on GPU(s) {gpus}")
    t0 = time.time()

    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
    )

    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"[{status}] {model_name} ({elapsed:.0f}s)")

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")

    return model_name, result.returncode, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_base", type=str, required=True)
    parser.add_argument("--context_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--families", type=str, nargs="+",
                        default=["gpt2", "pythia", "opt", "gpt-neo"],
                        choices=["gpt2", "pythia", "opt", "gpt-neo"])
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Specific model names (overrides --families)")
    parser.add_argument("--revision", type=str, default=None,
                        help="Pythia revision (e.g. step143000)")
    parser.add_argument("--max_batch_tokens", type=int, default=131072)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max_params", type=float, default=None,
                        help="Max model params in billions (e.g. 2.8 to skip 6.9b+)")
    parser.add_argument("--add_full_context", action="store_true",
                        help="Append each model's actual max context to context_sizes.")
    parser.add_argument("--offload_folder", type=str, default=None,
                        help="Root directory for disk-offloaded weights "
                             "(a per-model subdirectory is created automatically). "
                             "Required when a model exceeds total GPU VRAM.")
    args = parser.parse_args()

    # Build a flat lookup so --models can still find the correct param count
    _all_family_models = {
        name: params
        for fam in FAMILIES.values()
        for name, _, params in fam
    }

    # Collect models to process
    if args.models:
        # Look up params from FAMILIES so large models are classified correctly.
        # Unknown models default to SMALL_THRESHOLD+1 (= treated as large / multi-GPU).
        models_to_run = [
            (m, m, _all_family_models.get(m, SMALL_THRESHOLD + 1))
            for m in args.models
        ]
    else:
        models_to_run = []
        for fam in args.families:
            for name, label, params in FAMILIES[fam]:
                if args.max_params and params > args.max_params * 1e9:
                    continue
                models_to_run.append((name, label, params))

    print(f"Models to process: {len(models_to_run)}")
    for name, label, params in models_to_run:
        gb = params / 1e9
        print(f"  {name} ({gb:.1f}B params)")
    print(f"GPUs: {args.gpus}")
    print(f"Context sizes: {args.context_sizes}")
    print()

    num_gpus = len(args.gpus)

    # Separate into small (1 GPU) and large (multi-GPU) models
    small_models = [(n, l, p) for n, l, p in models_to_run if p <= SMALL_THRESHOLD]
    large_models = [(n, l, p) for n, l, p in models_to_run if p > SMALL_THRESHOLD]

    # ---- Phase 1: Process small models in parallel (one per GPU) ----
    if small_models:
        print(f"=== Phase 1: {len(small_models)} small models (parallel, 1 GPU each) ===")
        futures = {}
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            gpu_queue = list(args.gpus)
            pending = list(small_models)

            # Submit initial batch
            while pending and gpu_queue:
                name, label, params = pending.pop(0)
                gpu = gpu_queue.pop(0)
                out_dir = os.path.join(args.output_base, name)
                fut = executor.submit(
                    run_model, name, args.input_file, out_dir,
                    args.context_sizes, [gpu], args.max_batch_tokens,
                    args.dtype, args.revision, args.add_full_context,
                    args.offload_folder,
                )
                futures[fut] = (name, gpu)

            # As jobs finish, submit more
            for fut in as_completed(futures):
                name, gpu = futures[fut]
                gpu_queue.append(gpu)
                if pending:
                    next_name, next_label, next_params = pending.pop(0)
                    out_dir = os.path.join(args.output_base, next_name)
                    new_gpu = gpu_queue.pop(0)
                    new_fut = executor.submit(
                        run_model, next_name, args.input_file, out_dir,
                        args.context_sizes, [new_gpu], args.max_batch_tokens,
                        args.dtype, args.revision, args.add_full_context,
                        args.offload_folder,
                    )
                    futures[new_fut] = (next_name, new_gpu)

    # ---- Phase 2: Process large models sequentially (all GPUs) ----
    if large_models:
        print(f"\n=== Phase 2: {len(large_models)} large models (sequential, all GPUs) ===")
        for name, label, params in large_models:
            out_dir = os.path.join(args.output_base, name)
            # Scale max_batch_tokens down based on model size.
            # Large models leave little VRAM for activations; FFN intermediate tensors
            # (hidden_size × ffn_multiplier × batch_tokens) can easily cause OOM.
            #   > 30B params: 1024  (~1 window per batch for ctx≥1024; a few hundred for small ctx)
            #   > 6B  params: 8192
            #   otherwise:    65536
            if params > 30_000_000_000:
                safe_limit = 1024
            elif params > 6_000_000_000:
                safe_limit = 8192
            else:
                safe_limit = 65536
            batch_tokens = min(args.max_batch_tokens, safe_limit)
            print(f"  {name}: max_batch_tokens capped at {batch_tokens} "
                  f"(model={params/1e9:.0f}B, limit={safe_limit})")
            run_model(
                name, args.input_file, out_dir,
                args.context_sizes, args.gpus, batch_tokens,
                args.dtype, args.revision, args.add_full_context,
                args.offload_folder,
            )

    print("\n=== All models processed ===")


if __name__ == "__main__":
    main()