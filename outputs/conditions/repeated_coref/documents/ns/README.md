# Natural Stories mapping files

This directory contains the sentence-level placeholder and pronoun mappings
for the repeated-coreference condition, along with readable pronominalized
text. All 850 target mappings have a saved pronoun.

The word-level and token-level annotations produced from these mappings are:

- `outputs/conditions/repeated_coref/word_annotations/ns.tsv`
- `outputs/conditions/repeated_coref/token_annotations/ns.tsv`

Use `--occurrence-policy first` in the token-alignment stage to reproduce the
saved Natural Stories annotations.
