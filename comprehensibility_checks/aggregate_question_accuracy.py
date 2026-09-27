#!/usr/bin/env python3
"""Aggregate the checked-in DeepSeek comprehensibility-check results."""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path
from typing import Iterable


CHECKS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CHECKS_DIR / "results"
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "customized_deepseek_question_results.csv"


def is_correct(row: dict[str, str]) -> bool:
    return row["is_correct"].strip().lower() == "true"


def sort_key(row: dict[str, object]) -> tuple:
    key = []
    for name in ("condition", "story_index", "question_index"):
        if name not in row:
            continue
        value = row[name]
        try:
            key.append(int(value))
        except (TypeError, ValueError):
            key.append(value)
    return tuple(key)


def summarize(rows: Iterable[dict[str, str]], keys: list[str]) -> list[dict[str, object]]:
    groups = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    for row in rows:
        group_key = tuple(row[key] for key in keys)
        groups[group_key]["n"] += 1
        groups[group_key]["correct"] += int(is_correct(row))

    records = []
    for group_key, counts in groups.items():
        n = counts["n"]
        correct = counts["correct"]
        record = {key: value for key, value in zip(keys, group_key)}
        record.update(
            {
                "n": n,
                "correct": correct,
                "accuracy": correct / n if n else None,
            }
        )
        records.append(record)
    return sorted(records, key=sort_key)


def missed_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    records = []
    for row in rows:
        if is_correct(row):
            continue
        chosen_answer = row["option_a"] if row["generated_label"] == "A" else row["option_b"]
        records.append(
            {
                "condition": row["condition"],
                "story_index": row["story_index"],
                "question_index": row["question_index"],
                "question": row["question"],
                "generated_label": row["generated_label"],
                "chosen_answer": chosen_answer,
                "correct_label": row["correct_label"],
                "correct_answer": row["correct_answer"],
            }
        )
    return sorted(records, key=sort_key)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total_correct = sum(is_correct(row) for row in rows)
    overall = [
        {
            "n": len(rows),
            "correct": total_correct,
            "accuracy": total_correct / len(rows) if rows else None,
        }
    ]

    outputs = {
        "customized_deepseek_question_accuracy_overall.csv": overall,
        "customized_deepseek_question_accuracy_by_condition.csv": summarize(rows, ["condition"]),
        "customized_deepseek_question_accuracy_by_condition_story.csv": summarize(
            rows, ["condition", "story_index"]
        ),
        "customized_deepseek_question_accuracy_by_condition_question.csv": summarize(
            rows, ["condition", "question_index"]
        ),
        "customized_deepseek_question_accuracy_missed.csv": missed_rows(rows),
    }

    for filename, records in outputs.items():
        write_csv(args.output_dir / filename, records)

    print(
        f"Overall accuracy: {total_correct}/{len(rows)} "
        f"({total_correct / len(rows):.3f})"
    )
    for row in outputs["customized_deepseek_question_accuracy_by_condition.csv"]:
        print(
            f"{row['condition']}: {row['correct']}/{row['n']} "
            f"({row['accuracy']:.3f})"
        )


if __name__ == "__main__":
    main()
