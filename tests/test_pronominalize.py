import json
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

pronominalize = import_module("pipeline.02_pronoun_mapping.pronominalize")
extract_mentions = pronominalize.extract_mentions
mask_and_prepare = pronominalize.mask_and_prepare
parse_stanza_word_conllu = pronominalize.parse_stanza_word_conllu
prepare_records = pronominalize.prepare_records
select_mentions = pronominalize.select_mentions
sentence_targets = pronominalize.sentence_targets


FIXTURE = """\
1\tAlice\tAlice\tPROPN\t_\t_\t2\t1
2\tsaw\tsee\tVERB\t_\t_\t0\t_
3\tthe\tthe\tDET\t_\t_\t5\t2
4\tred\tred\tADJ\t_\t_\t5\t2|3
5\tcar\tcar\tNOUN\t_\t_\t2\t2|3
6\t.\t.\tPUNCT\t_\t_\t2\t_

1\tAlice\tAlice\tPROPN\t_\t_\t2\t1
2\tsmiled\tsmile\tVERB\t_\t_\t0\t_
3\the\the\tPRON\t_\t_\t2\t4
4\t.\t.\tPUNCT\t_\t_\t2\t_
"""


class PronominalizationTests(unittest.TestCase):
    def parse_fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "sample.conllu"
        path.write_text(FIXTURE, encoding="utf-8")
        return parse_stanza_word_conllu(path)

    def test_repeated_mode_selects_every_mention_of_repeated_chain(self):
        sentences = self.parse_fixture()
        mentions = extract_mentions(sentences)
        selected, stats = select_mentions(mentions, "repeated")

        self.assertEqual([mention.chain_id for mention in selected], ["1", "1"])
        self.assertEqual(stats["selected_chains"], 1)
        self.assertEqual(
            sentence_targets(sentences, selected), {0: ["Alice"], 1: ["Alice"]}
        )

    def test_singleton_mode_excludes_nested_and_pronoun_targets(self):
        sentences = self.parse_fixture()
        mentions = extract_mentions(sentences)
        selected, stats = select_mentions(mentions, "singleton")

        self.assertEqual(stats["raw_singleton_mentions"], 3)
        self.assertEqual(stats["nested_singleton_mentions_excluded"], 1)
        # Chain 4 is selected structurally but filtered because it is all PRON.
        self.assertEqual(sentence_targets(sentences, selected), {0: ["the red car"]})

        records = prepare_records(sentences, selected, "singleton")
        self.assertEqual(records[0]["masked_sentence"], "Alice saw [[TARGET_0]] .")
        self.assertEqual(records[0]["target_mapping"], {"[[TARGET_0]]": "the red car"})
        self.assertEqual(records[0]["baseline_type"], "all_non_nested_singletons")

    def test_masking_is_deterministic_and_longest_first(self):
        expected = (
            "[[TARGET_0]] passed [[TARGET_1]] .",
            {"[[TARGET_0]]": "the red car", "[[TARGET_1]]": "car"},
        )
        self.assertEqual(
            mask_and_prepare("the red car passed car .", ["car", "the red car", "car"]),
            expected,
        )

    def test_prepared_records_are_json_serializable(self):
        sentences = self.parse_fixture()
        selected, _ = select_mentions(extract_mentions(sentences), "repeated")
        json.dumps(prepare_records(sentences, selected, "repeated"))


if __name__ == "__main__":
    unittest.main()
