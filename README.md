# Contextsize

This repository contains the code for the EMNLP 2026 paper Using LMs to Model the Effects of Context and Coreference during Sentence Comprehension.


This repository contains the data and code for the context-size and pronominalization experiments.
Reusable inputs, annotations, and saved results are exposed under `outputs/`,
split into pre-inference data and surprisal data. Executable stages are grouped
under `src/`.

## Data dependencies

- [Natural Stories](https://github.com/languageMIT/naturalstories)
- [Provo](https://osf.io/sjefs/overview)
- [OneStop](https://osf.io/zn9sq/overview)
- Brown
- [`the_pile_16k_unigrams.npy`](https://github.com/byungdoh/llm_surprisal/tree/eacl24/data)

## Pipeline

```text
raw corpus text
  -> Stanza coreference annotations
  -> repeated-coreference and singleton-baseline conditions
  -> pronoun mapping and GPT-2 token alignment
  -> surprisal inference across models and context sizes
  + behavioral and lexical data
  -> statistical analyses, tables, and figures
```

Checked-in pronoun mappings make annotation deterministic; new DeepSeek requests
are opt-in. See [`src/README.md`](src/README.md) for commands and options.

## Reusable outputs

```text
outputs/
  pre_inference/
    raw_texts/                  canonical corpus text
    stanza_annotations/         Stanza word-level annotations
    conditions/
      repeated_coref/
        documents/              mapping JSONLs and readable previews
        word_annotations/       word-level intermediate TSVs
        token_annotations/      final inference inputs
      singleton_baseline/       same organization
  surprisal/                    saved word-level surprisal results
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
