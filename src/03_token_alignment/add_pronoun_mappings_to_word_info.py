#!/usr/bin/env python3
"""
Add DeepSeek pronoun replacements to word_info.tsv.

The DeepSeek mapping JSONL files contain original sentences, masked sentences,
target noun phrases, and replacement pronouns. This script aligns each original
noun-phrase span to the corresponding story rows in word_info.tsv and populates
the existing pronoun_replacement column. It does not add columns to word_info.tsv.
"""

import argparse
import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPINGS = PROJECT_ROOT / "outputs" / "pre_inference" / "conditions" / "repeated_coref" / "documents" / "ns"
DEFAULT_WORD_INFO = PROJECT_ROOT / "data" / "ns_surp" / "gpt2" / "word_info.tsv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "pre_inference" / "conditions" / "repeated_coref" / "word_annotations" / "ns.tsv"
REPLACEMENT_COLUMN = "pronoun_replacement"
EMPTY_VALUE = "NA"
OPENING_PUNCTUATION = {"'", '"', "(", "[", "{", "“", "‘"}

TOKEN_RE = re.compile(r"\[\[TARGET_\d+\]\]|\w+|[^\w\s]", re.UNICODE)
CONTRACTION_REPLACEMENTS = [
    (re.compile(r"\b[Cc]annot\b"), "can not"),
    (re.compile(r"\b[Gg]onna\b"), "gon na"),
    (re.compile(r"\b[Cc]an't\b"), "ca n't"),
    (re.compile(r"\b[Ww]on't\b"), "wo n't"),
    (re.compile(r"\b([A-Za-z]+)n't\b"), r"\1 n't"),
    (re.compile(r"\b([A-Za-z]+)'(m|re|ve|ll|d|s)\b", re.IGNORECASE), r"\1 '\2"),
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def normalize(token: str) -> str:
    return token.lower()


def split_stanza_contractions(text: str) -> str:
    normalized = text.replace("’", "'").replace("‘", "'")
    for pattern, replacement in CONTRACTION_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def normalized_tokens(text: str) -> list[str]:
    return [normalize(tok) for tok in tokenize(split_stanza_contractions(text))]


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        # Quotes are ordinary corpus characters, not TSV quoting. Treating a
        # leading dialogue quote as CSV syntax can merge many physical rows.
        reader = csv.DictReader(
            f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None
        )
        return list(reader), list(reader.fieldnames or [])


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=fieldnames,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            quotechar=None,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_mappings(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def story_id_from_mapping_path(path: Path) -> str:
    match = re.search(r"doc(\d+)_pronoun_mappings\.jsonl$", path.name)
    if not match:
        raise ValueError(
            f"Could not infer story id from {path.name}; expected '*docN_pronoun_mappings.jsonl'"
        )
    return str(int(match.group(1)) - 1)


def load_mapping_files(path: Path) -> list[tuple[str, Path, list[dict]]]:
    if path.is_file():
        return [(story_id_from_mapping_path(path), path, load_mappings(path))]

    files = sorted(path.glob("*_pronoun_mappings.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *_pronoun_mappings.jsonl files found in {path}")
    return [
        (story_id_from_mapping_path(file_path), file_path, load_mappings(file_path))
        for file_path in files
    ]


def word_tokens(word: str) -> list[str]:
    return normalized_tokens(word)


def build_token_index(rows: list[dict[str, str]], story: str) -> tuple[list[str], list[int]]:
    tokens = []
    row_for_token = []
    for row_idx, row in enumerate(rows):
        if row["story"] != story:
            continue
        for token in word_tokens(row["word"]):
            tokens.append(token)
            row_for_token.append(row_idx)
    return tokens, row_for_token


def find_subsequence(haystack: list[str], needle: list[str], start: int) -> int:
    if not needle:
        return -1
    max_start = len(haystack) - len(needle)
    for idx in range(start, max_start + 1):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return -1


def find_last_unoccupied_subsequence(
    haystack: list[str], needle: list[str], occupied: set[int]
) -> int:
    """Find the last matching span not already claimed by a longer target.

    Resolve repeated phrases from right to left while reserving spans already
    claimed by longer targets in ``target_mapping``.
    """
    if not needle:
        return -1
    for idx in range(len(haystack) - len(needle), -1, -1):
        span = set(range(idx, idx + len(needle)))
        if not span.intersection(occupied) and haystack[idx : idx + len(needle)] == needle:
            return idx
    return -1


def ordered_unique(values: list[int]) -> list[int]:
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def starts_sentence_at(sentence_tokens: list[str], phrase_start: int) -> bool:
    preceding_tokens = sentence_tokens[:phrase_start]
    return all(token in OPENING_PUNCTUATION for token in preceding_tokens)


def leading_punctuation(text: str) -> str:
    match = re.match(r"^[^\w\s]+", text)
    return match.group(0) if match else ""


def trailing_punctuation(text: str) -> str:
    match = re.search(r"[^\w\s]+$", text)
    return match.group(0) if match else ""


def capitalize_pronoun(pronoun: str) -> str:
    if pronoun.lower() == "i":
        return "I"
    return pronoun[:1].upper() + pronoun[1:]


def format_replacement(
    replacement: str,
    rows: list[dict[str, str]],
    row_indices: list[int],
    sentence_tokens: list[str],
    phrase_start: int,
) -> str:
    if not row_indices:
        return replacement

    first_word = rows[row_indices[0]]["word"]
    last_word = rows[row_indices[-1]]["word"]
    formatted = replacement
    if starts_sentence_at(sentence_tokens, phrase_start) or formatted.lower() == "i":
        formatted = capitalize_pronoun(formatted)
    return f"{leading_punctuation(first_word)}{formatted}{trailing_punctuation(last_word)}"


def add_value(row: dict[str, str], column: str, value: str) -> None:
    if not value:
        return
    current = row.get(column, "")
    values = [] if current in {"", EMPTY_VALUE} else current.split("|")
    if value not in values:
        values.append(value)
    row[column] = "|".join(values)


def annotate_rows(
    rows: list[dict[str, str]],
    row_indices: list[int],
    replacement: str,
) -> None:
    for row_idx in row_indices:
        add_value(rows[row_idx], REPLACEMENT_COLUMN, replacement)


def annotate_story(
    rows: list[dict[str, str]],
    mappings: list[dict],
    story: str,
    unreplaced_label: str,
    occurrence_policy: str = "last-unoccupied",
) -> tuple[int, int, int]:
    tokens, row_for_token = build_token_index(rows, story)
    cursor = 0
    matched = 0
    flagged = 0
    missed = 0

    for item in mappings:
        sentence_tokens = normalized_tokens(item["original_sentence"])
        sent_start = find_subsequence(tokens, sentence_tokens, cursor)
        if sent_start < 0:
            missed += len(item.get("target_mapping", {}))
            continue

        pronoun_mapping = item.get("pronoun_mapping", {})
        occupied_sentence_tokens: set[int] = set()
        for placeholder, phrase in item.get("target_mapping", {}).items():
            replacement = pronoun_mapping.get(placeholder, unreplaced_label)
            if not phrase:
                missed += 1
                continue
            phrase_tokens = normalized_tokens(phrase)
            if occurrence_policy == "first":
                phrase_start = find_subsequence(sentence_tokens, phrase_tokens, 0)
            else:
                phrase_start = find_last_unoccupied_subsequence(
                    sentence_tokens, phrase_tokens, occupied_sentence_tokens
                )
            if phrase_start < 0:
                missed += 1
                continue
            occupied_sentence_tokens.update(
                range(phrase_start, phrase_start + len(phrase_tokens))
            )

            absolute = sent_start + phrase_start
            row_indices = ordered_unique(row_for_token[absolute : absolute + len(phrase_tokens)])
            if placeholder in pronoun_mapping:
                replacement = format_replacement(
                    replacement,
                    rows,
                    row_indices,
                    sentence_tokens,
                    phrase_start,
                )
            annotate_rows(
                rows,
                row_indices,
                replacement,
            )
            if placeholder in pronoun_mapping:
                matched += 1
            else:
                flagged += 1

        cursor = sent_start + len(sentence_tokens)

    return matched, flagged, missed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate word_info.tsv pronoun_replacement from DeepSeek pronoun mappings."
    )
    parser.add_argument("--word-info", type=Path, default=DEFAULT_WORD_INFO)
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="word-level annotation TSV",
    )
    parser.add_argument(
        "--unreplaced-label",
        default="NOT_REPLACED",
        help=f"value written to {REPLACEMENT_COLUMN} when DeepSeek found a target but did not replace it",
    )
    parser.add_argument(
        "--occurrence-policy",
        choices=("first", "last-unoccupied"),
        default="last-unoccupied",
        help=(
            "How to align a phrase repeated within one sentence. 'first' uses "
            "the first match; 'last-unoccupied' searches right-to-left while "
            "preventing overlapping assignments."
        ),
    )
    args = parser.parse_args()

    rows, fieldnames = read_tsv(args.word_info)
    if REPLACEMENT_COLUMN not in fieldnames:
        fieldnames.append(REPLACEMENT_COLUMN)
    for row in rows:
        row[REPLACEMENT_COLUMN] = EMPTY_VALUE

    total_matched = 0
    total_flagged = 0
    total_missed = 0
    for story, mapping_path, mappings in load_mapping_files(args.mappings):
        matched, flagged, missed = annotate_story(
            rows,
            mappings,
            story,
            args.unreplaced_label,
            occurrence_policy=args.occurrence_policy,
        )
        total_matched += matched
        total_flagged += flagged
        total_missed += missed
        print(
            f"story {story} ({mapping_path.name}): "
            f"replaced {matched}, unreplaced flagged {flagged}, missed {missed}"
        )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(output, rows, fieldnames)
    print(
        f"total replaced mappings: {total_matched}, "
        f"unreplaced targets flagged: {total_flagged}, missed: {total_missed}"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
