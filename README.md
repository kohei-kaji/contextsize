# Contextsize

This repository contains the code for the EMNLP 2026 paper Using LMs to Model the Effects of Context and Coreference during Sentence Comprehension.


This repository contains the data and code for the context-size and pronominalization experiments.
Reusable inputs, annotations, and saved results are exposed under `outputs/`; executable stages are grouped under `pipeline/`.

## Data dependencies

- [Natural Stories](https://github.com/languageMIT/naturalstories)
- [Provo](https://osf.io/sjefs/overview)
- [OneStop](https://osf.io/zn9sq/overview)
- Brown
- [`the_pile_16k_unigrams.npy`](https://github.com/byungdoh/llm_surprisal/tree/eacl24/data)

## Data creation pipeline (before inference)

The input is one document per line. Natural Stories uses
`outputs/raw_texts/stories.txt`; `coref_destroyed.txt` and
`singleton_destroyed.txt` are previews, not source data.

```text
raw corpus text
  -> Stanza word tokens and coreference chains (.conllu)
  -> select and mask mentions
  -> DeepSeek placeholder-to-pronoun mappings (.jsonl)
  -> align mappings to GPT-2 word rows
  -> expand word rows to GPT-2 token positions
  -> token_annotations/<corpus>.tsv
```

Use `--mode repeated` for mentions in repeated chains and `--mode singleton`
for non-nested singleton mentions. The latter reconstructs the historical
baseline rule; use the archived baseline JSONLs for exact replication.

```shell
uv run python pipeline/01_coreference/build_stanza_word_coref.py \
  outputs/raw_texts/stories.txt work/stanza --corpus ns

uv run python pipeline/02_pronoun_mapping/pronominalize.py \
  work/stanza work/repeated/documents \
  --corpus ns --mode repeated --query-deepseek

uv run python pipeline/03_token_alignment/add_pronoun_mappings_to_word_info.py \
  --word-info data/ns_surp/gpt2/word_info.tsv \
  --mappings work/repeated/documents \
  --output work/repeated/word_annotations/ns.tsv \
  --occurrence-policy first

uv run python pipeline/03_token_alignment/expand_word_info_pronouns.py \
  --word-info work/repeated/word_annotations/ns.tsv \
  --output work/repeated/token_annotations/ns.tsv
```

The DeepSeek step is the only paid network request and requires
`DEEPSEEK_API_KEY`. The final token TSV is the pre-inference artifact. Mapping
JSONLs and readable pronominalized text are not read during inference.


## Reusable outputs

```text
outputs/
  raw_texts/                    canonical corpus text
  stanza_annotations/           historical Stanza-word annotations
  conditions/
    repeated_coref/
      documents/                mapping JSONLs and readable previews
      word_annotations/         word-level intermediate TSVs
      token_annotations/        final inference TSVs
    singleton_baseline/         same organization
  inference_results/            saved surprisal results
```

The repeated-coreference Natural Stories word-level TSV was not archived, but
its final token TSV is available. The main-condition Natural Stories JSONLs in
this checkout are reconstructions; the final archived token TSV is the exact
artifact used by inference.

## Reproducibility limits

- Fresh DeepSeek responses may differ even at temperature zero.
- Newly generated Stanza annotations may differ because the historical Stanza
  model files were not committed.
- The reconstructed singleton selector disagrees with 15 archived sentences
  on historical nested-span edge cases; use archived JSONLs for exact replay.
- Historical masking used a Python `set` to break equal-length ties. The
  recovered implementation adds a deterministic position/text tie-breaker.


## Statistical analyses and plots
```shell
cd analysis

# main analysis and plot
Rscript rt-analysis.R
Rscript plot.R


# follow-up analysis1: test the DLL increase
Rscript bayes_dll.R

# follow-up analysis2: POS-tag analysis
Rscript pos_analysis.R

# follow-up analysis3: DLL degradation analysis
Rscript dll_degradation.R
```
