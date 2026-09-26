# Natural Stories mapping files

The original main-condition Natural Stories mapping JSONLs were not retained.
The files in this directory were reconstructed from:

- `stanza_word_results/ns_stanza_word_doc*.conllu` for sentences and target
  phrases;
- `deepseek_pronoun_results/pronouns_ns.tsv` for the saved pronouns.

They use the same record fields and filename layout as the `brown/`,
`onestop/`, and `provo/` result directories. All 850 target mappings receive a
pronoun found in the saved TSV; no API call or fallback pronoun was needed.
Four target strings occur more than once in a sentence with different archived
case-sensitive replacements. Since one JSONL placeholder can hold only one
pronoun, the first archived occurrence is used, matching the historical
single-placeholder representation.

The reconstruction is reproducible with:

```shell
uv run python src/reconstruct_pronoun_mappings_from_tsv.py \
  stanza_word_results \
  deepseek_pronoun_results/pronouns_ns.tsv \
  deepseek_pronoun_results/ns \
  --corpus ns
```

Using `--occurrence-policy first` when aligning these JSONLs back to
`word_info.tsv` reproduces all 12,373 rows of the saved `pronouns_ns.tsv`.
The TSV remains the canonical input to inference.
