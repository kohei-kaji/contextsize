import collections
import os
import re
import openai
import json
from pathlib import Path

def load_local_env(path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

load_local_env(Path(__file__).resolve().parent.parent / "local.env")
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
BATCH_SIZE = 10
ALLOWED_PRONOUNS = {
    "he", "she", "it", "they",
    "you", "we", "i", "me", "us",
    "him", "her", "them",
    "my", "mine", "your", "yours", "our", "ours",
    "his", "hers", "its", "their", "theirs",
    "myself", "yourself", "ourselves",
    "himself", "herself", "itself", "themselves",
}

def rank_entities_by_id(file_path):
    # Global counter for ID occurrences
    id_frequency = collections.Counter()
    
    # Track which IDs were active in the PREVIOUS line 
    # to avoid double-counting multi-token phrases
    last_line_ids = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            misc = parts[7]
            
            if misc == "_":
                current_line_ids = set()
            else:
                # Get set of IDs on current line (handles '1|2')
                current_line_ids = set(misc.split('|'))

            # If an ID is in 'current' but NOT in 'last', it's a NEW mention
            new_mentions = current_line_ids - last_line_ids
            for eid in new_mentions:
                id_frequency[eid] += 1
            
            # Update last_line_ids for the next iteration
            last_line_ids = current_line_ids
    
    most_common = id_frequency.most_common()

    # we only want the entities with count > 1
    most_common = [eid for eid, count in most_common if count > 1]

    return most_common

def parse_conll_with_targets(file_path, target_ids):
    """
    Parses the file into sentences and identifies target noun phrases.
    target_ids should be a set of strings, e.g., {'56', '24', '40'}
    """
    all_sentences = []
    current_tokens = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            word = parts[1]
            misc = parts[7]
            upos = parts[3]
            current_tokens.append({'word': word, 'misc': misc, 'upos': upos})

            # Boundary check
            if word == ".":
                # 1. Reconstruct full sentence text
                sentence_text = " ".join([t['word'] for t in current_tokens])

                # Use an ordered dictionary of lists to track words by their ID
                # This handles multi-word phrases correctly
                id_to_words = collections.defaultdict(list)
                
                # To ensure we separate two occurrences of the same phrase,
                # we track when an ID stops and starts again.
                ordered_targets = []
                last_ids = set()
                
                # Buffer for phrases currently being "built"
                active_phrases = {} # {id: [words]}
                active_pos = {} # {id: [upos tags]} 

                for t in current_tokens:
                    current_ids = set(t['misc'].split('|')) if t['misc'] != "_" else set()
                    
                    # 1. Check for IDs that just ended
                    ended_ids = last_ids - current_ids
                    for eid in ended_ids:
                        if eid in target_ids and eid in active_phrases:
                            # if the upos is only a pronoun, we skip it
                            if not all(pos == "PRON" for pos in active_pos[eid]): 
                                if not all(pos == "ADV" for pos in active_pos[eid]):          
                                    ordered_targets.append(" ".join(active_phrases[eid]))
                            del active_phrases[eid]

                    # 2. Check for IDs that are starting or continuing
                    for eid in current_ids:
                        if eid in target_ids:
                            if eid not in active_phrases:
                                active_phrases[eid] = []
                                active_pos[eid] = []
                            active_phrases[eid].append(t['word'])
                            active_pos[eid].append(t['upos'])
                    
                    last_ids = current_ids

                # Cleanup any remaining active phrases at the end of sentence
                for eid, words in active_phrases.items():
                    ordered_targets.append(" ".join(words))

                all_sentences.append({
                    "sentence": sentence_text,
                    "noun_phrases": ordered_targets # This is now an ordered list
                })
                current_tokens = []

    return all_sentences

def mask_and_prepare(sentence, targets):
    """
    Replaces phrases with placeholders [[TARGET_N]] starting from the longest.
    Returns the masked sentence and the mapping of placeholders to original phrases.
    """
    # Sort targets by length descending to handle nested entities (outermost first)
    sorted_targets = sorted(list(set(targets)), key=len, reverse=True)
    mapping = {}
    masked_text = sentence
    
    for i, phrase in enumerate(sorted_targets):
        placeholder = f"[[TARGET_{i}]]"
        # We use escape to handle special characters in the phrase
        pattern = re.escape(phrase)
        # Check if the phrase is actually in the sentence (handles duplicates safely)
        if phrase in masked_text:
            masked_text = masked_text.replace(phrase, placeholder)
            mapping[placeholder] = phrase
            
    return masked_text, mapping

def apply_pronoun_mapping(masked_sentence, pronoun_mapping):
    """
    Replaces [[TARGET_N]] placeholders using a model-returned pronoun mapping.
    Pronouns are capitalized when the placeholder starts a sentence.
    """
    if not pronoun_mapping:
        return masked_sentence
    pattern = re.compile("|".join(re.escape(placeholder) for placeholder in pronoun_mapping))
    return pattern.sub(
        lambda match: format_pronoun_for_position(
            pronoun_mapping[match.group(0)],
            masked_sentence,
            match.start(),
        ),
        masked_sentence,
    )

def format_pronoun_for_position(pronoun, text, start_index):
    """
    Returns pronoun with orthographic capitalization appropriate for its position.
    """
    if pronoun == "i":
        return "I"
    if starts_sentence(text, start_index):
        return pronoun[:1].upper() + pronoun[1:]
    return pronoun

def starts_sentence(text, start_index):
    """
    True if text[start_index:] begins a sentence, allowing opening quotes/brackets.
    """
    prefix = text[:start_index].rstrip()
    if not prefix:
        return True

    opening_marks = set("\"'“‘([{:")
    while prefix and prefix[-1] in opening_marks:
        prefix = prefix[:-1].rstrip()
    if not prefix:
        return True

    return prefix[-1] in ".!?"

def extract_json_from_response(content):
    """
    Parses JSON even if the model wraps it in a markdown code fence.
    Expected shape:
    {"results": [
      {"sentence_index": 0, "pronoun_mapping": {"[[TARGET_0]]": "it"}}
    ]}
    """
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)

def result_items_from_response(raw_results):
    """
    Accepts the requested {"results": [...]} shape plus the older raw-list shape.
    """
    if isinstance(raw_results, dict):
        raw_results = raw_results.get("results", [])
    if not isinstance(raw_results, list):
        return []
    return [item for item in raw_results if isinstance(item, dict)]

def placeholder_variants(placeholder):
    """
    Models sometimes normalize [[TARGET_0]] to TARGET_0 or target_0.
    """
    bare = placeholder.strip("[]")
    return {
        placeholder,
        bare,
        bare.lower(),
        f"[[{bare.lower()}]]",
    }

def coerce_pronoun(value):
    """
    Returns one lowercase pronoun if the model produced exactly one allowed word.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if re.fullmatch(r"[a-z]+", candidate) and candidate in ALLOWED_PRONOUNS:
        return candidate
    return None

def normalize_pronoun_mapping(raw_mapping, expected_placeholders):
    """
    Keeps only expected placeholders and plain pronoun values.
    Missing or invalid entries are left out so callers can decide how to handle them.
    """
    normalized = {}
    if not isinstance(raw_mapping, dict):
        return normalized
    for placeholder in expected_placeholders:
        for variant in placeholder_variants(placeholder):
            pronoun = coerce_pronoun(raw_mapping.get(variant))
            if pronoun:
                normalized[placeholder] = pronoun
                break
    return normalized

def is_likely_plural(phrase):
    words = re.findall(r"[A-Za-z]+", phrase.lower())
    if not words:
        return False
    if any(word in {"they", "them", "their", "theirs", "themselves", "people", "children", "men", "women"} for word in words):
        return True
    head = words[-1]
    return head.endswith("s") and not head.endswith("ss")

def fallback_pronoun(masked_sentence, placeholder, original_phrase):
    """
    Last-resort replacement so every placeholder has exactly one pronoun.
    The model still gets first choice; this only prevents unmapped placeholders.
    """
    return "it"

def fill_missing_pronouns(item, pronoun_mapping):
    complete_mapping = dict(pronoun_mapping)
    for placeholder, original_phrase in item["target_mapping"].items():
        if placeholder not in complete_mapping:
            complete_mapping[placeholder] = fallback_pronoun(
                item["masked_sentence"],
                placeholder,
                original_phrase,
            )
    return complete_mapping

def chat_json(messages):
    return client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

def repair_missing_pronouns(item, pronoun_mapping):
    """
    Asks the model again only for placeholders that were omitted or invalid.
    """
    missing_placeholders = [
        placeholder
        for placeholder in item["target_mapping"]
        if placeholder not in pronoun_mapping
    ]
    if not missing_placeholders:
        return pronoun_mapping

    missing_context = {
        placeholder: item["target_mapping"][placeholder]
        for placeholder in missing_placeholders
    }
    prompt = (
        "Return ONLY valid JSON with this exact shape: "\
        "{\"pronoun_mapping\": {\"[[TARGET_0]]\": \"it\"}}.\n"
        "Choose exactly one lowercase, one-word English pronoun for every placeholder. "
        "Do not omit any placeholder. Do not return noun phrases.\n\n"
        f"Sentence: {item['masked_sentence']}\n"
        f"Missing Context: {json.dumps(missing_context, ensure_ascii=False)}"
    )

    try:
        response = chat_json([
            {"role": "system", "content": "You choose one pronoun for each placeholder."},
            {"role": "user", "content": prompt},
        ])
        raw_result = extract_json_from_response(response.choices[0].message.content)
        repaired = normalize_pronoun_mapping(
            raw_result.get("pronoun_mapping", {}) if isinstance(raw_result, dict) else {},
            set(missing_placeholders),
        )
        return {**pronoun_mapping, **repaired}
    except Exception as e:
        print(f"Pronoun repair API Error for sentence {item['sentence_index']}: {e}")
        return pronoun_mapping

def query_deepseek_batch(batch_data):
    """
    Sends a batch of masked sentences to the API and returns explicit
    [[TARGET_N]] -> pronoun mappings.
    """
    system_prompt = (
        "You are a linguistic expert. You will receive sentences with placeholders like [[TARGET_N]]. "
        "Your task is to choose exactly one appropriate English pronoun for each placeholder. "
        "Use sentence position to choose case: subject, object, possessive, or reflexive. "
        "Refer to the provided Context only to determine number, animacy, and likely gender. "
        "Never return the original Context text as a value. "
        "Return ONLY a valid JSON object. It must have this shape: "
        "{\"results\": [{\"sentence_index\": 0, \"pronoun_mapping\": {\"[[TARGET_0]]\": \"it\"}}]}. "
        "Every placeholder in Context must appear exactly once in pronoun_mapping."
    )
    
    prepared_items = []
    passthrough_items = []
    user_prompt = ""
    for i, item in enumerate(batch_data):
        masked_sent, mapping = mask_and_prepare(item['sentence'], item['noun_phrases'])
        sentence_index = item.get("sentence_index", i)
        if not mapping:
            passthrough_items.append({
                "sentence_index": sentence_index,
                "original_sentence": item["sentence"],
                "masked_sentence": item["sentence"],
                "target_mapping": {},
                "pronoun_mapping": {},
            })
            continue # No target to ask the model about.
        prepared_items.append({
            "sentence_index": sentence_index,
            "original_sentence": item["sentence"],
            "masked_sentence": masked_sent,
            "target_mapping": mapping,
        })
        # We provide the mapping in the prompt to help the model pick the right gender/number
        mapping_str = ", ".join([f"{k}: {v}" for k, v in mapping.items()])
        user_prompt += f"Sentence {sentence_index}: {masked_sent}\nContext: {mapping_str}\n\n"

    if not prepared_items:
        return passthrough_items

    try:
        response = chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw_results = extract_json_from_response(response.choices[0].message.content)
        results_by_index = {
            item.get("sentence_index"): item.get("pronoun_mapping", {})
            for item in result_items_from_response(raw_results)
        }

        results = []
        for item in prepared_items:
            expected_placeholders = set(item["target_mapping"])
            pronoun_mapping = normalize_pronoun_mapping(
                results_by_index.get(item["sentence_index"], {}),
                expected_placeholders,
            )
            pronoun_mapping = repair_missing_pronouns(item, pronoun_mapping)
            if set(pronoun_mapping) != expected_placeholders:
                missing = sorted(expected_placeholders - set(pronoun_mapping))
                print(f"Using fallback pronouns for sentence {item['sentence_index']}: {missing}")
                pronoun_mapping = fill_missing_pronouns(item, pronoun_mapping)

            results.append({
                **item,
                "pronoun_mapping": pronoun_mapping,
            })
        return sorted(results + passthrough_items, key=lambda item: item["sentence_index"])
    except Exception as e:
        print(f"API Error: {e}")
        return [
            {
                "sentence_index": item.get("sentence_index", i),
                "original_sentence": item["sentence"],
                "masked_sentence": item["sentence"],
                "target_mapping": {},
                "pronoun_mapping": {},
            }
            for i, item in enumerate(batch_data)
        ] # Fallback to original

def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]

def process_file(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {input_path.name}...")

    # 1. Get IDs mentioned at least twice
    top_ids = rank_entities_by_id(input_path)

    # 2. Parse sentences and record target phrases
    parsed_sentences = parse_conll_with_targets(input_path, set(top_ids))
    for sentence_index, item in enumerate(parsed_sentences):
        item["sentence_index"] = sentence_index

    # 3. Process in batches to save costs
    final_pronominalized_text = []
    final_mapping_records = []

    for i in range(0, len(parsed_sentences), BATCH_SIZE):
        batch = parsed_sentences[i : i + BATCH_SIZE]
        print(f"  Processing sentences {i} to {i + len(batch)}...")
    
        results = query_deepseek_batch(batch)
        final_mapping_records.extend(results)
        final_pronominalized_text.extend(
            apply_pronoun_mapping(item["masked_sentence"], item["pronoun_mapping"])
            for item in results
        )

    # 4. Save one set of outputs per input file
    text_output_path = output_dir / f"{input_path.stem}_pronominalized.txt"
    mapping_output_path = output_dir / f"{input_path.stem}_pronoun_mappings.jsonl"

    with open(text_output_path, "w", encoding="utf-8") as f:
        for line in final_pronominalized_text:
            f.write(line + "\n")

    with open(mapping_output_path, "w", encoding="utf-8") as f:
        for record in final_mapping_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Wrote {text_output_path}")
    print(f"  Wrote {mapping_output_path}")

def process_directory(input_dir, output_dir):
    input_dir = Path(input_dir)
    conllu_files = sorted(input_dir.glob("*.conllu"), key=natural_sort_key)
    if not conllu_files:
        raise FileNotFoundError(f"No .conllu files found in {input_dir}")

    for input_path in conllu_files:
        process_file(input_path, output_dir)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    process_directory(
        project_root / "stanza_word_results",
        project_root / "deepseek_pronoun_results",
    )
