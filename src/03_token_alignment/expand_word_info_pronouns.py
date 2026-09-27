#!/usr/bin/env python3
"""
Expand word_info.tsv to a token-level pronoun TSV.

The input is expected to contain the word-level columns written by the
surprisal scripts:

    story, zone, word, num_tokens, pronoun_replacement

By default this writes all stories with a leading story column:

    story, token_index, word_index, token, pronoun_replacement

The output always includes the story id so token indices remain unambiguous.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "pre_inference" / "conditions" / "repeated_coref" / "word_annotations" / "ns.tsv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "pre_inference" / "conditions" / "repeated_coref" / "token_annotations" / "ns.tsv"

REQUIRED_COLUMNS = {"story", "zone", "word", "num_tokens", "pronoun_replacement"}
EMPTY_VALUE = "NA"


def clean_pronoun_replacement(value: str, normalize_replacement: bool) -> str:
    value = (value or "").strip()
    if not value:
        return EMPTY_VALUE
    if value == EMPTY_VALUE:
        return value
    if not normalize_replacement:
        return value

    cleaned_parts = []
    for part in value.split("|"):
        cleaned = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", part.strip())
        cleaned = cleaned or part.strip()
        if cleaned != cleaned.upper():
            cleaned = cleaned.lower()
        cleaned_parts.append(cleaned)
    return "|".join(cleaned_parts)


def parse_num_tokens(row: dict[str, str], row_number: int) -> int:
    raw_value = row["num_tokens"]
    try:
        num_tokens = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid num_tokens={raw_value!r} at input row {row_number}."
        ) from exc

    if num_tokens < 1:
        raise ValueError(
            f"Invalid num_tokens={num_tokens} at input row {row_number}; expected >= 1."
        )
    return num_tokens


def read_word_info(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(
            f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None
        )
        fieldnames = list(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader), fieldnames


def expanded_rows(
    rows: list[dict[str, str]],
    story_filter: str | None,
    normalize_replacement: bool,
) -> list[dict[str, str]]:
    output_rows = []
    token_index_by_story: dict[str, int] = {}

    for input_idx, row in enumerate(rows, start=2):
        story = row["story"]
        if story_filter is not None and story != story_filter:
            continue

        token_index = token_index_by_story.setdefault(story, 0)
        word_index = row["zone"]
        token = row["word"]
        replacement = clean_pronoun_replacement(
            row["pronoun_replacement"],
            normalize_replacement=normalize_replacement,
        )

        for _ in range(parse_num_tokens(row, input_idx)):
            output_rows.append(
                {
                    "story": story,
                    "token_index": str(token_index),
                    "word_index": word_index,
                    "token": token,
                    "pronoun_replacement": replacement,
                }
            )
            token_index += 1

        token_index_by_story[story] = token_index

    return output_rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["story", "token_index", "word_index", "token", "pronoun_replacement"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            quotechar=None,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand word-level word_info.tsv rows to token-level pronoun rows."
    )
    parser.add_argument("--word-info", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--story",
        help="Only expand one story id, e.g. --story 0. By default all stories are expanded.",
    )
    parser.add_argument(
        "--normalize-replacements",
        action="store_true",
        help="Strip punctuation and lowercase pronoun_replacement values to match story0_pronouns.tsv style.",
    )
    args = parser.parse_args()

    rows, _ = read_word_info(args.word_info)
    output_rows = expanded_rows(
        rows,
        story_filter=args.story,
        normalize_replacement=args.normalize_replacements,
    )
    write_rows(args.output, output_rows)
    print(f"wrote {len(output_rows)} token rows to {args.output}")


if __name__ == "__main__":
    main()
