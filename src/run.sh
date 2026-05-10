uv run run.py \
  --input_file ./stories.txt \
  --output_base ns \
  --context_sizes \
    2 3 4 5 6 7 8 11 13 17 21 26 33 41 51 65 81 101 129 161 201 257 321 401 513 641 801 1025 \
  --add_full_context \
  --gpus 1 2 3 \
  --families gpt2 \
  --max_batch_tokens 1024 \
  --offload_folder /tmp/hf_offload


uv run run.py \
  --input_file ./brown.txt \
  --output_base brown \
  --context_sizes \
    2 3 4 5 6 7 8 11 13 17 21 26 33 41 51 65 81 101 129 161 201 257 321 401 513 641 801 1025 \
  --add_full_context \
  --gpus 1 2 3 \
  --families gpt2 \
  --max_batch_tokens 1024 \
  --offload_folder /tmp/hf_offload

uv run run.py \
  --input_file ./provo.txt \
  --output_base provo \
  --context_sizes \
    2 3 4 5 6 7 8 11 13 17 21 26 33 41 51 65 81 101 129 161 201 257 321 401 513 641 801 1025 \
  --add_full_context \
  --gpus 1 2 3 \
  --families gpt2 \
  --max_batch_tokens 1024 \
  --offload_folder /tmp/hf_offload

uv run run.py \
  --input_file ./onestop.txt \
  --output_base os \
  --context_sizes \
    2 3 4 5 6 7 8 11 13 17 21 26 33 41 51 65 81 101 129 161 201 257 321 401 513 641 801 1025 \
  --add_full_context \
  --gpus 1 2 3 \
  --families gpt2 \
  --max_batch_tokens 1024 \
  --offload_folder /tmp/hf_offload
