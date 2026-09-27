# Contextsize

This repository contains the code for the EMNLP 2026 paper Using LMs to Model the Effects of Context and Coreference during Sentence Comprehension.


This repository contains the data and code for the context-size and pronominalization experiments.
Reusable inputs, annotations, and saved results are exposed under `outputs/`; executable stages are grouped under `src/`.

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

Use `repeated` for mentions in repeated chains and `singleton` for non-nested
singleton mentions. The checked-in mapping JSONLs provide deterministic replay;
new DeepSeek requests are opt-in.

```shell
src/run_annotations.sh ns repeated data/ns_surp/gpt2/word_info.tsv
```

See [`src/README.md`](src/README.md) for stage-by-stage commands,
condition options, and inference usage.

## Reusable outputs

```text
outputs/
  raw_texts/                    canonical corpus text
  stanza_annotations/           Stanza word-level annotations
  conditions/
    repeated_coref/
      documents/                mapping JSONLs and readable previews
      word_annotations/         word-level intermediate TSVs
      token_annotations/        final inference TSVs
    singleton_baseline/         same organization
  inference_results/            saved surprisal results
```


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
