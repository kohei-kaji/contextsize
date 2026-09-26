#!/usr/bin/env python3
"""Build mapping JSONL/text files from Stanza inputs and a saved pronoun TSV.

This is intended for result sets where the token-level pronoun TSV survives but
the earlier sentence-level mapping JSONLs do not.  Target phrases are prepared
from the Stanza word/coreference files.  Pronouns are read back from the TSV by
aligning each original sentence and phrase to its story/word rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from .add_pronoun_mappings_to_word_info import (
        find_subsequence,
        normalized_tokens,
        ordered_unique,
    )
    from .pronominalize import (
        ALLOWED_PRONOUNS,
        extract_mentions,
        parse_stanza_word_conllu,
        prepare_records,
        rank_chain_mentions,
        select_mentions,
    )
    from .regenerate_pronominalized_text import (
        apply_pronoun_mapping,
        detokenize_stanza_text,
    )
except ImportError:  # Direct execution: python src/reconstruct_....py
    from add_pronoun_mappings_to_word_info import (
        find_subsequence,
        normalized_tokens,
        ordered_unique,
    )
    from pronominalize import (
        ALLOWED_PRONOUNS,
        extract_mentions,
        parse_stanza_word_conllu,
        prepare_records,
        rank_chain_mentions,
        select_mentions,
    )
    from regenerate_pronominalized_text import (
        apply_pronoun_mapping,
        detokenize_stanza_text,
    )


MAPPING_SUFFIX = "_pronoun_mappings.jsonl"


def natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def normalized_pronouns(value: str) -> list[str]:
    """Return allowed pronouns in a formatted TSV replacement, in order."""

    result: list[str] = []
    for part in value.split("|"):
        for word in re.findall(r"[A-Za-z]+", part):
            candidate = word.lower()
            if candidate in ALLOWED_PRONOUNS:
                result.append(candidate)
                break
    return result


def load_story_words(path: Path) -> dict[int, list[tuple[str, str]]]:
    """Collapse repeated GPT-2-token rows to one row per original word."""

    words: dict[int, dict[int, tuple[str, str]]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(
            source, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None
        )
        required = {"story", "word_index", "token", "pronoun_replacement"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            story = int(row["story"])
            word_index = int(row["word_index"])
            value = (row["token"], row["pronoun_replacement"])
            previous = words[story].setdefault(word_index, value)
            if previous != value:
                raise ValueError(
                    f"Inconsistent token rows for story {story}, word {word_index}"
                )
    return {
        story: [indexed[index] for index in sorted(indexed)]
        for story, indexed in words.items()
    }


def story_token_index(
    words: list[tuple[str, str]],
) -> tuple[list[str], list[int]]:
    tokens: list[str] = []
    row_for_token: list[int] = []
    for row_index, (word, _) in enumerate(words):
        for token in normalized_tokens(word):
            tokens.append(token)
            row_for_token.append(row_index)
    return tokens, row_for_token


def occurrence_pronoun(
    row_indices: list[int], words: list[tuple[str, str]]
) -> str | None:
    replacement_lists = [
        normalized_pronouns(words[index][1])
        for index in row_indices
        if words[index][1] != "NA"
    ]
    replacement_lists = [values for values in replacement_lists if values]
    if not replacement_lists:
        return None

    common = set(replacement_lists[0])
    for values in replacement_lists[1:]:
        common.intersection_update(values)
    if len(common) == 1:
        return next(iter(common))

    # Pipe-separated values record overlapping annotations in insertion order.
    # The current (usually shorter) target was added last.
    return replacement_lists[0][-1]


def phrase_pronouns(
    sentence_tokens: list[str],
    phrase: str,
    sentence_start: int,
    row_for_token: list[int],
    words: list[tuple[str, str]],
) -> list[str]:
    phrase_tokens = normalized_tokens(phrase)
    candidates: list[str] = []
    search_from = 0
    while True:
        phrase_start = find_subsequence(sentence_tokens, phrase_tokens, search_from)
        if phrase_start < 0:
            break
        absolute_start = sentence_start + phrase_start
        row_indices = ordered_unique(
            row_for_token[absolute_start : absolute_start + len(phrase_tokens)]
        )
        pronoun = occurrence_pronoun(row_indices, words)
        if pronoun:
            candidates.append(pronoun)
        search_from = phrase_start + 1
    return candidates


def prepare_file(input_path: Path) -> list[dict[str, object]]:
    sentences = parse_stanza_word_conllu(input_path)
    selected, _ = select_mentions(
        extract_mentions(sentences),
        "repeated",
        chain_counts=rank_chain_mentions(input_path),
    )
    return prepare_records(sentences, selected, "repeated")


def reconstruct_file(
    input_path: Path,
    story_id: int,
    words: list[tuple[str, str]],
    output_dir: Path,
) -> tuple[int, int, int]:
    records = prepare_file(input_path)
    story_tokens, row_for_token = story_token_index(words)
    cursor = 0
    resolved = 0
    ambiguous = 0
    fallback = 0

    for record in records:
        sentence_tokens = normalized_tokens(str(record["original_sentence"]))
        sentence_start = find_subsequence(story_tokens, sentence_tokens, cursor)
        if sentence_start < 0:
            raise ValueError(
                f"Could not align story {story_id}, sentence {record['sentence_index']}"
            )

        pronoun_mapping: dict[str, str] = {}
        target_mapping = record["target_mapping"]
        for placeholder, phrase in target_mapping.items():
            candidates = phrase_pronouns(
                sentence_tokens,
                phrase,
                sentence_start,
                row_for_token,
                words,
            )
            unique = list(dict.fromkeys(candidates))
            if not unique:
                pronoun = "it"
                fallback += 1
            else:
                pronoun = unique[0]
                resolved += 1
                if len(unique) > 1:
                    ambiguous += 1
            pronoun_mapping[placeholder] = pronoun
        record["pronoun_mapping"] = pronoun_mapping
        cursor = sentence_start + len(sentence_tokens)

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / f"{input_path.stem}{MAPPING_SUFFIX}"
    text_path = output_dir / f"{input_path.stem}_pronominalized.txt"
    with mapping_path.open("w", encoding="utf-8", newline="") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    with text_path.open("w", encoding="utf-8", newline="") as destination:
        for record in records:
            rendered = apply_pronoun_mapping(
                str(record["masked_sentence"]), record["pronoun_mapping"]
            )
            destination.write(detokenize_stanza_text(rendered) + "\n")
    return resolved, ambiguous, fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stanza_dir", type=Path)
    parser.add_argument("pronoun_tsv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--corpus", default="ns")
    args = parser.parse_args()

    story_words = load_story_words(args.pronoun_tsv)
    inputs = sorted(
        args.stanza_dir.glob(f"{args.corpus}_stanza_word_doc*.conllu"),
        key=natural_key,
    )
    if not inputs:
        parser.error(f"No {args.corpus}_stanza_word_doc*.conllu files found")

    totals = [0, 0, 0]
    for story_id, input_path in enumerate(inputs):
        if story_id not in story_words:
            raise ValueError(f"No TSV rows for story {story_id}")
        counts = reconstruct_file(
            input_path, story_id, story_words[story_id], args.output_dir
        )
        totals = [left + right for left, right in zip(totals, counts)]
        print(
            f"{input_path.name}: resolved={counts[0]}, "
            f"ambiguous={counts[1]}, fallback={counts[2]}"
        )
    print(
        f"Total targets={sum((totals[0], totals[2]))}; resolved={totals[0]}; "
        f"ambiguous={totals[1]}; fallback={totals[2]}"
    )


if __name__ == "__main__":
    main()
