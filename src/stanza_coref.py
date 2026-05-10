import stanza 
import os 
from transformers import GPT2TokenizerFast


def save_gpt2_conllu_with_coref(doc, text, filename):
    # 1. Use the "Fast" tokenizer to get access to offset_mapping
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # 2. Tokenize and get character offsets
    encodings = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'])
    offsets = encodings['offset_mapping']
    token_ids = encodings['input_ids']

    # Flatten the doc words into a list for easier character matching
    # This assumes your 'doc' variable is from Stanza/spaCy
    all_words = [word for sent in doc.sentences for word in sent.words]

    with open(filename, "w", encoding="utf-8") as f:
        
        for i, (token, (start, end), t_id) in enumerate(zip(tokens, offsets, token_ids)):
            # Skip special tokens like <|endoftext|> if present
            if start == end == 0 and i != 0: continue

            id_string = "_" 
            # 3. Find which Stanza word overlaps with this GPT-2 token's offsets
            for word in all_words:
                # Check if the subword is within the character bounds of the original word
                # stanza's _end_char matches GPT2's offset end (also consider the case where the token is fully contained within the word)
                if word._end_char == end or start < word._start_char < end or word._start_char <= end < word._end_char:
                    upos = word._upos
                    if word.coref_chains:
                        ids = [f"{c.chain.index}" for c in word.coref_chains]
                        id_string = "|".join([str(i) for i in ids])
                    break

            # 4. Build the row
            clean_token = token.replace('Ġ', '')
            row = [
                str(i + 1),
                token,
                clean_token,
                upos, # UPOS
                id_string, # coref_label
                f"GPT2_ID={t_id}" # Combined MISC Column
            ]
            f.write("\t".join(row) + "\n")

if __name__ == "__main__":
    # Initialize the pipeline
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse,coref')

    os.makedirs("../gpt2_coref_results", exist_ok=True)
    
    os.makedirs("../gpt2_coref_results/ns", exist_ok=True)
    with open("../data/stories.txt", 'r', encoding='utf-8') as file:
        for line_id, line in enumerate(file, start=1):
            text = line.strip()
            line_doc = nlp(text)
            save_gpt2_conllu_with_coref(line_doc, text, f"../gpt2_coref_results/ns/gpt2_coref_doc{line_id}.conllu")

    os.makedirs("../gpt2_coref_results/brown", exist_ok=True)
    with open("../data/brown.txt", 'r', encoding='utf-8') as file:
        for line_id, line in enumerate(file, start=1):
            text = line.strip()
            line_doc = nlp(text)
            save_gpt2_conllu_with_coref(line_doc, text, f"../gpt2_coref_results/brown/gpt2_coref_doc{line_id}.conllu")

    os.makedirs("../gpt2_coref_results/provo", exist_ok=True)
    with open("../data/provo.txt", 'r', encoding='utf-8') as file:
        for line_id, line in enumerate(file, start=1):
            text = line.strip()
            line_doc = nlp(text)
            save_gpt2_conllu_with_coref(line_doc, text, f"../gpt2_coref_results/provo/gpt2_coref_doc{line_id}.conllu")

    os.makedirs("../gpt2_coref_results/os", exist_ok=True)
    with open("../data/onestop.txt", 'r', encoding='utf-8') as file:
        for line_id, line in enumerate(file, start=1):
            text = line.strip()
            line_doc = nlp(text)
            save_gpt2_conllu_with_coref(line_doc, text, f"../gpt2_coref_results/os/gpt2_coref_doc{line_id}.conllu")
