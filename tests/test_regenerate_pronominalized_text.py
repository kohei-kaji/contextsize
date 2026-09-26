import unittest

from pipeline.regenerate_pronominalized_text import (
    apply_pronoun_mapping,
    detokenize_stanza_text,
)


class DetokenizationTests(unittest.TestCase):
    def test_punctuation_contractions_and_hyphens(self):
        tokenized = "But do n't tell Ireland 's burr - headed veteran , please ."
        self.assertEqual(
            detokenize_stanza_text(tokenized),
            "But don't tell Ireland's burr-headed veteran, please.",
        )

    def test_quotes_and_parentheses(self):
        tokenized = 'The cry " Garryowen ! " ( loudly ) .'
        self.assertEqual(
            detokenize_stanza_text(tokenized),
            'The cry "Garryowen!" (loudly).',
        )

    def test_spaced_dash_is_preserved(self):
        self.assertEqual(
            detokenize_stanza_text("the world – but now"),
            "the world – but now",
        )

    def test_plural_possessive_apostrophe(self):
        self.assertEqual(
            detokenize_stanza_text("the researchers ' suggestions"),
            "the researchers' suggestions",
        )

    def test_curly_single_quotes(self):
        self.assertEqual(
            detokenize_stanza_text("the ‘ Love hormone ' oxytocin"),
            "the ‘Love hormone' oxytocin",
        )

    def test_placeholder_replacement_capitalization(self):
        self.assertEqual(
            apply_pronoun_mapping(
                "[[TARGET_0]] saw [[TARGET_1]] .",
                {"[[TARGET_0]]": "she", "[[TARGET_1]]": "him"},
            ),
            "She saw him .",
        )


if __name__ == "__main__":
    unittest.main()
