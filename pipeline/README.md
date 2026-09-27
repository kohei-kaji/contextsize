# Reproducible pipeline

The pipeline has four stages. Each stage reads from the repository's
`outputs/` interface and writes the artifact consumed by the next stage.

```text
01_coreference
  raw text -> word-level Stanza coreference annotations
02_pronoun_mapping
  coreference annotations -> placeholder/pronoun mappings and readable text
03_token_alignment
  mappings -> word annotations -> token annotations
04_inference
  raw text + token annotations -> word surprisal at each context size
```

Install the pinned Python environment with `uv sync`. Then run stages 1–3 for
one corpus and condition with:

```shell
pipeline/run_annotations.sh ns repeated data/ns_surp/gpt2/word_info.tsv
```

The runner reuses the checked-in mapping JSONLs by default, which makes the
annotation build deterministic. Set `QUERY_DEEPSEEK=1` to request new mappings
instead; this requires `DEEPSEEK_API_KEY` in the environment or `local.env`.

Supported condition modes are `repeated` and `singleton`. Outputs are written
to `outputs/conditions/repeated_coref/` and
`outputs/conditions/singleton_baseline/`, respectively. Set
`OCCURRENCE_POLICY=first` or `OCCURRENCE_POLICY=last-unoccupied` to control how
duplicate phrases within a sentence are aligned.

Run inference for a generated condition with:

```shell
uv run python pipeline/04_inference/gpt2_inference_pronominalized.py \
  --annotations-dir outputs/conditions/repeated_coref/token_annotations \
  --corpus ns \
  --model_name gpt2 \
  --context_sizes 2 3 5 10 20 50 100
```

All scripts resolve defaults from the repository root and may be invoked from
any working directory.
