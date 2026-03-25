import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

device = "cpu"

def extract_bos_eos_punct_flags_from_ud(ud_file: str) -> pd.DataFrame:
    token_rows = []
    sent_start_next = True
    prev_was_token = False
    last_idx = None

    tokenid_re = re.compile(r"TokenId=([^\s;|]+)")
    digits_re = re.compile(r"^\d+")

    with open(ud_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if prev_was_token and last_idx is not None:
                    token_rows[last_idx]["eos"] = 1
                sent_start_next = True
                prev_was_token = False
                last_idx = None
                continue
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                raise ValueError(f"Invalid CoNLL-U line: {line}")

            pos = parts[3]
            deprel = parts[7]
            misc = parts[9]

            m = tokenid_re.search(misc)
            if not m:
                raise ValueError(f"TokenId not found in MISC field: {misc}")
            tokenid_val = m.group(1)
            comps = tokenid_val.split(".")
            if len(comps) < 2:
                continue

            m_story = digits_re.match(comps[0])
            m_zone  = digits_re.match(comps[1])
            if not (m_story and m_zone):
                continue

            story = int(m_story.group(0))
            zone  = int(m_zone.group(0))

            is_punct = int(deprel == "punct")
            bos = 1 if sent_start_next else 0

            token_rows.append({
                "story": story,
                "zone": zone,
                "bos": bos,
                "eos": 0,
                "is_punct": is_punct,
                "pos": pos,
            })

            last_idx = len(token_rows) - 1
            prev_was_token = True
            sent_start_next = False

    if prev_was_token and last_idx is not None:
        token_rows[last_idx]["eos"] = 1

    df_tok = pd.DataFrame(token_rows, columns=["story", "zone", "bos", "eos", "is_punct", "pos"]).astype({
        "story":"int32","zone":"int32","bos":"int8","eos":"int8","is_punct":"int8","pos":"string"
    })

    agg = df_tok.groupby(["story", "zone"], as_index=False).agg(
            bos=("bos", "max"), 
            eos=("eos", "max"), 
            is_punct=("is_punct", lambda s: int((s > 0).any())),
            pos=("pos", lambda x: "_".join(x.astype(str)))
        )
    agg[["bos","eos","is_punct"]] = agg[["bos","eos","is_punct"]].astype("int8")
    agg = agg.sort_values(["story","zone"]).reset_index(drop=True)
    agg["sent_id"] = agg.groupby("story")["bos"].cumsum().astype("int32")

    agg["position"] = agg.groupby(["story", "sent_id"]).cumcount() + 1
    agg["position"] = agg["position"].astype("int32")

    agg = agg[["story","zone","bos","eos","is_punct","position","pos"]]
    return agg


def calc_unisurp(input_path: str) -> pd.DataFrame:
    """
    from Byung-Doh's code: https://github.com/byungdoh/llm_surprisal/blob/eacl24/get_unigram_surprisal.py
    Calculates unigram surprisal based on counts from 16k training batches (~33B tokens) of the Pile,
    saved in an array of length |V| under data/the_pile_16k_unigrams.npy
    """
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision="step143000")

    counts = np.load("./data/the_pile_16k_unigrams.npy").squeeze()
    log_total_counts = np.log2(np.sum(counts))

    with open(input_path, "r") as f:
        batches = []
        story_ids = []
        words_ids = []
        words = []
        unisurps = []
        for i, text in enumerate(f, start=1):
            text = text.strip()
            words_in_text = text.split(" ")
            words.extend(words_in_text)
            tokenizer_output = tokenizer(text)
            batches.append(tokenizer_output.input_ids)
            story_ids.extend([i] * len(words_in_text))
            words_ids.extend(list(range(1, len(words_in_text) + 1)))

        curr_word_ix = 0
        for batch in batches:
            toks = tokenizer.convert_ids_to_tokens(batch)
            unisurp = log_total_counts - np.log2(counts[batch])

            curr_word_unisurp = []
            curr_toks = []
            for i in range(len(toks)):
                curr_word_unisurp.append(unisurp[i])
                curr_toks += [toks[i]]
                curr_toks_str = tokenizer.convert_tokens_to_string(curr_toks)
                if words[curr_word_ix] == curr_toks_str.strip():
                    unisurps.append(sum(curr_word_unisurp))
                    curr_word_unisurp = []
                    curr_toks = []
                    curr_word_ix += 1

    df = pd.DataFrame({"story": story_ids, "zone": words_ids, "word": words, "unisurp": unisurps})
    return df


def main(input_path: str, output_path: str, ud_path: str) -> None:
    df_unisurp = calc_unisurp(input_path)
    df_position = extract_bos_eos_punct_flags_from_ud(ud_path)

    common_cols = ["story", "zone"]
    assert len(df_unisurp) == len(df_position), f"{len(df_unisurp)} != {len(df_position)}"

    df = pd.concat([df_unisurp, df_position.drop(columns=common_cols)], axis=1)
    df["wlen"] = df["word"].apply(len)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_path = "./data/stories.txt"
    output_path = "./data/baselines_ns.csv"
    ud_path = "./data/naturalstories/parses/ud/stories-aligned.conllx"

    main(input_path, output_path, ud_path)
