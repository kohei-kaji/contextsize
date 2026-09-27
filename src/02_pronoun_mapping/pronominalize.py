#!/usr/bin/env python3
"""Generate pronominalized text and sentence-level pronoun-mapping JSONL files.

The ``repeated`` mode replaces coreference chains with more than one mention.
The ``singleton`` mode selects one-mention chains unless their span is nested
inside another coreference mention.

Use ``--prepare-only`` to inspect selection and masking without making an API
request.  API generation is opt-in via ``--query-deepseek``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ALLOWED_PRONOUNS = {
    "he", "she", "it", "they", "you", "we", "i", "me", "us", "him",
    "her", "them", "my", "mine", "your", "yours", "our", "ours", "his",
    "hers", "its", "their", "theirs", "myself", "yourself", "ourselves",
    "himself", "herself", "itself", "themselves",
}
PLACEHOLDER_RE = re.compile(r"\[\[TARGET_\d+\]\]")
CONTRACTIONS = {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m"}
CLOSE_PUNCTUATION = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "…"}
OPEN_PUNCTUATION = {"(", "[", "{", "“", "‘"}
JOIN_BOTH_SIDES = {"-", "/"}
CURRENCY = {"$", "£", "€", "¥"}


def starts_sentence(text: str, start_index: int) -> bool:
    prefix = text[:start_index].rstrip()
    if not prefix:
        return True
    while prefix and prefix[-1] in "\"'“‘([{:":
        prefix = prefix[:-1].rstrip()
    return not prefix or prefix[-1] in ".!?"


def format_pronoun(pronoun: str, text: str, start_index: int) -> str:
    if pronoun == "i":
        return "I"
    if starts_sentence(text, start_index):
        return pronoun[:1].upper() + pronoun[1:]
    return pronoun


def apply_pronoun_mapping(masked_sentence: str, mapping: dict[str, str]) -> str:
    """Replace placeholders and capitalize sentence-initial pronouns."""

    return PLACEHOLDER_RE.sub(
        lambda match: format_pronoun(
            mapping.get(match.group(0), match.group(0)),
            masked_sentence,
            match.start(),
        ),
        masked_sentence,
    )


def detokenize_stanza_text(text: str) -> str:
    """Undo spaces introduced by joining Stanza tokens with ``" "``."""

    tokens = text.split()
    if not tokens:
        return ""

    pieces: list[str] = []
    attach_next = False
    attach_next_from_previous = False
    straight_double_quote_open = False
    straight_single_quote_open = False

    for index, token in enumerate(tokens):
        attach_left = token in CLOSE_PUNCTUATION or token in CONTRACTIONS

        if token == '"':
            if straight_double_quote_open:
                attach_left = True
                straight_double_quote_open = False
            else:
                straight_double_quote_open = True
                attach_next = True
        elif token == "'":
            if straight_single_quote_open:
                attach_left = True
                straight_single_quote_open = False
            elif index > 0 and tokens[index - 1].lower().endswith("s"):
                attach_left = True
            else:
                straight_single_quote_open = True
                attach_next = True
        elif token == "‘":
            straight_single_quote_open = True
            attach_next = True
        elif token == "’":
            straight_single_quote_open = False
            attach_left = True
        elif token == "”":
            attach_left = True
        elif token in OPEN_PUNCTUATION or token in CURRENCY:
            attach_next = True
        elif token in JOIN_BOTH_SIDES:
            attach_left = True
            attach_next = True

        if not pieces:
            pieces.append(token)
        elif attach_left or attach_next_from_previous:
            pieces.append(token)
        else:
            pieces.extend((" ", token))

        attach_next_from_previous = attach_next
        attach_next = False

    return "".join(pieces)


@dataclass(frozen=True)
class Token:
    word: str
    upos: str
    coref_ids: frozenset[str]


@dataclass(frozen=True)
class Mention:
    chain_id: str
    sentence_index: int
    start: int
    end: int  # exclusive, relative to sentence


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def parse_stanza_word_conllu(path: Path) -> list[list[Token]]:
    """Read the pipeline's custom eight-column Stanza word format."""

    sentences: list[list[Token]] = []
    current: list[Token] = []

    def flush() -> None:
        if current:
            sentences.append(current.copy())
            current.clear()

    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 8 tab-separated columns"
                )
            ids = frozenset() if parts[7] == "_" else frozenset(parts[7].split("|"))
            current.append(Token(parts[1], parts[3], ids))
            # Documents use periods as explicit sentence boundaries.
            if parts[1] == ".":
                flush()
    # Ignore an unterminated fragment at EOF so every emitted sentence has an
    # explicit boundary.
    return sentences


def rank_chain_mentions(path: Path) -> Counter[str]:
    """Count coreference-chain mention starts."""

    counts: Counter[str] = Counter()
    previous_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for raw_line in source:
            parts = raw_line.strip().split("\t")
            if len(parts) < 8:
                continue
            current_ids = set() if parts[7] == "_" else set(parts[7].split("|"))
            counts.update(current_ids - previous_ids)
            previous_ids = current_ids
    return counts


def extract_mentions(sentences: list[list[Token]]) -> list[Mention]:
    mentions: list[Mention] = []
    for sentence_index, tokens in enumerate(sentences):
        active: dict[str, int] = {}
        for token_index, token in enumerate(tokens):
            ended = set(active) - token.coref_ids
            for chain_id in ended:
                mentions.append(
                    Mention(chain_id, sentence_index, active.pop(chain_id), token_index)
                )
            for chain_id in token.coref_ids:
                active.setdefault(chain_id, token_index)
        for chain_id, start in active.items():
            mentions.append(Mention(chain_id, sentence_index, start, len(tokens)))
    return mentions


def is_nested(candidate: Mention, all_mentions: Iterable[Mention]) -> bool:
    for other in all_mentions:
        if other is candidate or other.sentence_index != candidate.sentence_index:
            continue
        contains = other.start <= candidate.start and other.end >= candidate.end
        strictly_larger = other.start < candidate.start or other.end > candidate.end
        if contains and strictly_larger:
            return True
    return False


def select_mentions(
    mentions: list[Mention],
    mode: str,
    chain_counts: Counter[str] | None = None,
) -> tuple[list[Mention], dict[str, int]]:
    counts = chain_counts or Counter(mention.chain_id for mention in mentions)
    if mode == "repeated":
        selected = [mention for mention in mentions if counts[mention.chain_id] > 1]
        return selected, {
            "chains": len(counts),
            "selected_chains": sum(count > 1 for count in counts.values()),
            "selected_mentions": len(selected),
        }

    raw = [mention for mention in mentions if counts[mention.chain_id] == 1]
    selected = [mention for mention in raw if not is_nested(mention, mentions)]
    return selected, {
        "chains": len(counts),
        "raw_singleton_mentions": len(raw),
        "eligible_non_nested_singleton_mentions": len(selected),
        "nested_singleton_mentions_excluded": len(raw) - len(selected),
    }


def sentence_targets(
    sentences: list[list[Token]], selected: Iterable[Mention]
) -> dict[int, list[str]]:
    by_sentence: dict[int, list[Mention]] = defaultdict(list)
    for mention in selected:
        by_sentence[mention.sentence_index].append(mention)

    result: dict[int, list[str]] = defaultdict(list)
    for sentence_index, mentions in by_sentence.items():
        tokens = sentences[sentence_index]
        for mention in sorted(mentions, key=lambda item: (item.start, item.end)):
            span = tokens[mention.start : mention.end]
            pos = [token.upos for token in span]
            if pos and (all(tag == "PRON" for tag in pos) or all(tag == "ADV" for tag in pos)):
                continue
            result[sentence_index].append(" ".join(token.word for token in span))
    return result


def mask_and_prepare(sentence: str, targets: list[str]) -> tuple[str, dict[str, str]]:
    """Mask unique target strings, longest first, with deterministic tie-breaking."""

    first_position = {target: sentence.find(target) for target in targets}
    ordered = sorted(set(targets), key=lambda item: (-len(item), first_position[item], item))
    mapping: dict[str, str] = {}
    masked = sentence
    for placeholder_index, phrase in enumerate(ordered):
        placeholder = f"[[TARGET_{placeholder_index}]]"
        if phrase not in masked:
            continue
        masked = masked.replace(phrase, placeholder)
        mapping[placeholder] = phrase
    return masked, mapping


def prepare_records(
    sentences: list[list[Token]], selected: Iterable[Mention], mode: str
) -> list[dict[str, object]]:
    targets = sentence_targets(sentences, selected)
    records: list[dict[str, object]] = []
    for index, tokens in enumerate(sentences):
        original = " ".join(token.word for token in tokens)
        masked, target_mapping = mask_and_prepare(original, targets.get(index, []))
        record: dict[str, object] = {
            "sentence_index": index,
            "original_sentence": original,
            "masked_sentence": masked,
            "target_mapping": target_mapping,
            "pronoun_mapping": {},
        }
        if mode == "singleton":
            record["baseline_type"] = "all_non_nested_singletons"
        records.append(record)
    return records


def normalize_mapping(raw: object, expected: set[str]) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, str] = {}
    for placeholder in expected:
        candidates = (placeholder, placeholder.strip("[]"), placeholder.lower())
        for candidate in candidates:
            value = raw.get(candidate)
            if isinstance(value, str) and value.strip().lower() in ALLOWED_PRONOUNS:
                normalized[placeholder] = value.strip().lower()
                break
    return normalized


def deepseek_request(records: list[dict[str, object]], api_key: str) -> None:
    items = [record for record in records if record["target_mapping"]]
    if not items:
        return
    contexts = []
    for record in items:
        contexts.append(
            f"Sentence {record['sentence_index']}: {record['masked_sentence']}\n"
            f"Context: {json.dumps(record['target_mapping'], ensure_ascii=False)}"
        )
    prompt = (
        "Choose exactly one lowercase, one-word English pronoun for every placeholder. "
        "Return only JSON of the form {\"results\": [{\"sentence_index\": 0, "
        "\"pronoun_mapping\": {\"[[TARGET_0]]\": \"it\"}}]}.\n\n"
        + "\n\n".join(contexts)
    )
    body = json.dumps(
        {
            "model": "deepseek-chat",
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You choose one pronoun per placeholder."},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode()
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            envelope = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek request failed ({exc.code}): {detail}") from exc
    content = envelope["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    result = json.loads(content)
    by_index = {
        item.get("sentence_index"): item.get("pronoun_mapping", {})
        for item in result.get("results", [])
        if isinstance(item, dict)
    }
    for record in items:
        expected = set(record["target_mapping"])
        mapping = normalize_mapping(by_index.get(record["sentence_index"]), expected)
        missing = expected - set(mapping)
        if missing:
            raise ValueError(
                f"DeepSeek omitted or returned invalid pronouns for sentence "
                f"{record['sentence_index']}: {sorted(missing)}"
            )
        record["pronoun_mapping"] = mapping


def reuse_saved_mappings(records: list[dict[str, object]], path: Path) -> None:
    saved = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                item = json.loads(line)
                saved[item["sentence_index"]] = item
    for record in records:
        old = saved.get(record["sentence_index"])
        if old is None:
            raise ValueError(f"No saved record for sentence {record['sentence_index']}")
        if old.get("masked_sentence") != record["masked_sentence"]:
            raise ValueError(
                f"Saved masking differs at sentence {record['sentence_index']}; "
                "the source/model version is not identical"
            )
        expected = set(record["target_mapping"])
        mapping = normalize_mapping(old.get("pronoun_mapping"), expected)
        if set(mapping) != expected:
            raise ValueError(f"Saved mapping is incomplete at sentence {record['sentence_index']}")
        record["pronoun_mapping"] = mapping


def write_outputs(
    input_path: Path,
    output_dir: Path,
    records: list[dict[str, object]],
    prepare_only: bool,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_suffix = "_prepared.jsonl" if prepare_only else "_pronoun_mappings.jsonl"
    mapping_path = output_dir / f"{input_path.stem}{mapping_suffix}"
    with mapping_path.open("w", encoding="utf-8", newline="") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    if prepare_only:
        return mapping_path, None

    text_path = output_dir / f"{input_path.stem}_pronominalized.txt"
    with text_path.open("w", encoding="utf-8", newline="") as output:
        for record in records:
            rendered = apply_pronoun_mapping(
                str(record["masked_sentence"]), record["pronoun_mapping"]
            )
            output.write(detokenize_stanza_text(rendered) + "\n")
    return mapping_path, text_path


def process_file(args: argparse.Namespace, input_path: Path) -> dict[str, int]:
    sentences = parse_stanza_word_conllu(input_path)
    mentions = extract_mentions(sentences)
    selected, stats = select_mentions(
        mentions, args.mode, chain_counts=rank_chain_mentions(input_path)
    )
    records = prepare_records(sentences, selected, args.mode)

    if args.reuse_mappings:
        reuse_path = args.reuse_mappings
        if reuse_path.is_dir():
            reuse_path = reuse_path / f"{input_path.stem}_pronoun_mappings.jsonl"
        reuse_saved_mappings(records, reuse_path)
    elif args.query_deepseek:
        load_local_env(Path(__file__).resolve().parents[2] / "local.env")
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY or add it to local.env")
        for start in range(0, len(records), args.batch_size):
            deepseek_request(records[start : start + args.batch_size], api_key)
    elif not args.prepare_only:
        raise ValueError("Use --prepare-only, --reuse-mappings, or --query-deepseek")

    mapping_path, text_path = write_outputs(
        input_path, args.output_dir, records, args.prepare_only
    )
    print(f"{input_path.name}: {stats}; wrote {mapping_path}")
    if text_path:
        print(f"  wrote {text_path}")
    return stats


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A .conllu file or directory")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mode", choices=("repeated", "singleton"), default="repeated")
    parser.add_argument(
        "--corpus",
        help="when INPUT is a directory, only process <corpus>_stanza_word_doc*.conllu",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--reuse-mappings", type=Path)
    action.add_argument("--query-deepseek", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    pattern = f"{args.corpus}_stanza_word_doc*.conllu" if args.corpus else "*.conllu"
    inputs = (
        [args.input]
        if args.input.is_file()
        else sorted(args.input.glob(pattern), key=natural_key)
    )
    if not inputs:
        parser.error(f"no .conllu files found at {args.input}")
    for input_path in inputs:
        process_file(args, input_path)


if __name__ == "__main__":
    main()
