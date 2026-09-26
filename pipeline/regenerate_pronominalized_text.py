#!/usr/bin/env python3
"""Regenerate readable pronominalized text from saved DeepSeek mappings.

The mapping JSONL files intentionally keep Stanza's space-separated token
representation for reliable span matching.  The human-readable
``*_pronominalized.txt`` files should be detokenized after placeholders have
been replaced.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


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
    """Replace placeholders, matching the capitalization of the original run."""

    return PLACEHOLDER_RE.sub(
        lambda match: format_pronoun(
            mapping.get(match.group(0), match.group(0)),
            masked_sentence,
            match.start(),
        ),
        masked_sentence,
    )


def detokenize_stanza_text(text: str) -> str:
    """Undo the spaces introduced by joining Stanza tokens with ``" "``.

    This handles punctuation, English contractions, paired quotes, and the
    ASCII hyphen/slash forms found in the generated corpora.  En/em dashes are
    deliberately left spaced because the original corpora use them as dashes.
    """

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
                # Plural possessive: "researchers ' suggestions".
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


def output_path_for_mapping(mapping_path: Path) -> Path:
    suffix = "_pronoun_mappings.jsonl"
    if not mapping_path.name.endswith(suffix):
        raise ValueError(f"Unexpected mapping filename: {mapping_path}")
    stem = mapping_path.name[: -len(suffix)]
    return mapping_path.with_name(f"{stem}_pronominalized.txt")


def regenerate_mapping_file(mapping_path: Path) -> tuple[Path, int]:
    output_path = output_path_for_mapping(mapping_path)
    lines = []
    with mapping_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                rendered = apply_pronoun_mapping(
                    record["masked_sentence"], record.get("pronoun_mapping", {})
                )
            except KeyError as exc:
                raise ValueError(
                    f"{mapping_path}:{line_number} is missing {exc.args[0]!r}"
                ) from exc
            lines.append(detokenize_stanza_text(rendered))

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        for line in lines:
            destination.write(line + "\n")
    return output_path, len(lines)


def mapping_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*_pronoun_mappings.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dirs",
        nargs="+",
        type=Path,
        help="Result directories containing *_pronoun_mappings.jsonl files.",
    )
    args = parser.parse_args()

    total_files = 0
    total_lines = 0
    for results_dir in args.results_dirs:
        files = mapping_files(results_dir)
        if not files:
            raise FileNotFoundError(f"No mapping JSONL files found under {results_dir}")
        for mapping_path in files:
            _, line_count = regenerate_mapping_file(mapping_path)
            total_files += 1
            total_lines += line_count

    print(f"Regenerated {total_files} files containing {total_lines} lines.")


if __name__ == "__main__":
    main()
