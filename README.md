```shell
cd data
git clone https://github.com/languageMIT/naturalstories.git
```

[the_pile_16k_unigrams.npy](https://github.com/byungdoh/llm_surprisal/tree/eacl24/data)


```shell
cd src
uv run baseline_preds_ns.py
source run.sh  # takes a few days
```

```shell
cd ..
Rscript src/analysis_ns.R
```
