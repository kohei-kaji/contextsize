uv run run.py \
  --input_file ../data/stories.txt \
  --output_base ../data/ns_surp \
  --context_sizes \
   2 3 4 5 6 7 8 10 12 16 20 25 32 40 50 64 80 100 128 160 200 256 320 400 512 640 800 1024 \
  --add_full_context \
  --gpus 0 1 2 3 \
  --max_batch_tokens 1024 \
  --offload_folder /tmp/hf_offload
