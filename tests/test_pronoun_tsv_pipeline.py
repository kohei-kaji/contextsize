import unittest
from importlib import import_module

alignment = import_module(
    "pipeline.03_token_alignment.add_pronoun_mappings_to_word_info"
)
expansion = import_module("pipeline.03_token_alignment.expand_word_info_pronouns")
annotate_story = alignment.annotate_story
expanded_rows = expansion.expanded_rows


class PronounTsvPipelineTests(unittest.TestCase):
    def test_mapping_alignment_then_gpt2_token_expansion(self):
        rows = [
            {"story": "0", "zone": "0", "word": "Alice", "num_tokens": "1", "pronoun_replacement": "NA"},
            {"story": "0", "zone": "1", "word": "saw", "num_tokens": "1", "pronoun_replacement": "NA"},
            {"story": "0", "zone": "2", "word": "Bob.", "num_tokens": "2", "pronoun_replacement": "NA"},
        ]
        mappings = [
            {
                "original_sentence": "Alice saw Bob .",
                "target_mapping": {"[[TARGET_0]]": "Bob"},
                "pronoun_mapping": {"[[TARGET_0]]": "him"},
            }
        ]

        matched, flagged, missed = annotate_story(rows, mappings, "0", "NOT_REPLACED")
        self.assertEqual((matched, flagged, missed), (1, 0, 0))
        self.assertEqual(rows[2]["pronoun_replacement"], "him.")

        tokens = expanded_rows(rows, story_filter=None, normalize_replacement=True)
        self.assertEqual(len(tokens), 4)
        self.assertEqual(
            [row["pronoun_replacement"] for row in tokens],
            ["NA", "NA", "him", "him"],
        )
        self.assertEqual([row["token_index"] for row in tokens], ["0", "1", "2", "3"])

    def test_duplicate_phrase_uses_last_span_as_archived_artifacts_do(self):
        rows = [
            {"story": "0", "zone": str(i), "word": word, "num_tokens": "1", "pronoun_replacement": "NA"}
            for i, word in enumerate(("ear", "to", "ear."))
        ]
        mappings = [
            {
                "original_sentence": "ear to ear .",
                "target_mapping": {"[[TARGET_0]]": "ear"},
                "pronoun_mapping": {"[[TARGET_0]]": "it"},
            }
        ]

        self.assertEqual(
            annotate_story(rows, mappings, "0", "NOT_REPLACED"), (1, 0, 0)
        )
        self.assertEqual(
            [row["pronoun_replacement"] for row in rows], ["NA", "NA", "it."]
        )

    def test_first_occurrence_policy_matches_main_condition_alignment(self):
        rows = [
            {"story": "0", "zone": str(i), "word": word, "num_tokens": "1", "pronoun_replacement": "NA"}
            for i, word in enumerate(("ear", "to", "ear."))
        ]
        mappings = [
            {
                "original_sentence": "ear to ear .",
                "target_mapping": {"[[TARGET_0]]": "ear"},
                "pronoun_mapping": {"[[TARGET_0]]": "it"},
            }
        ]

        annotate_story(
            rows,
            mappings,
            "0",
            "NOT_REPLACED",
            occurrence_policy="first",
        )
        self.assertEqual(
            [row["pronoun_replacement"] for row in rows], ["It", "NA", "NA"]
        )


if __name__ == "__main__":
    unittest.main()
