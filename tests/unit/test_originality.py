"""Tests for the originality corpus and similarity math.

No network, no models, no audio. Every fixture is inline text.
"""

import tempfile
import unittest
from pathlib import Path

from core.originality import (
    add_to_corpus,
    load_corpus,
    recent_angles,
)


class CorpusTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_corpus_loads_as_empty_list(self):
        self.assertEqual(load_corpus(corpus_dir=self.dir), [])

    def test_added_script_round_trips(self):
        add_to_corpus("the tide withdraws", genre="sleep", angle="tidal",
                      corpus_dir=self.dir)
        entries = load_corpus(corpus_dir=self.dir)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].text, "the tide withdraws")
        self.assertEqual(entries[0].genre, "sleep")
        self.assertEqual(entries[0].angle, "tidal")

    def test_ids_are_unique_even_within_one_second(self):
        ids = {
            add_to_corpus(f"script {i}", genre="sleep", corpus_dir=self.dir)
            for i in range(25)
        }
        self.assertEqual(len(ids), 25)

    def test_genre_filter_partitions_the_corpus(self):
        add_to_corpus("a", genre="sleep", corpus_dir=self.dir)
        add_to_corpus("b", genre="focus", corpus_dir=self.dir)
        add_to_corpus("c", genre="sleep", corpus_dir=self.dir)
        self.assertEqual(len(load_corpus(genre="sleep", corpus_dir=self.dir)), 2)
        self.assertEqual(len(load_corpus(genre="focus", corpus_dir=self.dir)), 1)

    def test_entries_come_back_most_recent_first(self):
        for i in range(5):
            add_to_corpus(f"script {i}", genre="sleep", corpus_dir=self.dir)
        texts = [e.text for e in load_corpus(genre="sleep", corpus_dir=self.dir)]
        self.assertEqual(texts[0], "script 4")

    def test_limit_takes_the_most_recent(self):
        for i in range(10):
            add_to_corpus(f"script {i}", genre="sleep", corpus_dir=self.dir)
        entries = load_corpus(genre="sleep", limit=3, corpus_dir=self.dir)
        self.assertEqual([e.text for e in entries],
                         ["script 9", "script 8", "script 7"])

    def test_recent_angles_are_most_recent_first_and_deduplicated(self):
        for angle in ["tidal", "staircase", "tidal", "meadow"]:
            add_to_corpus("x", genre="sleep", angle=angle, corpus_dir=self.dir)
        self.assertEqual(
            recent_angles("sleep", limit=3, corpus_dir=self.dir),
            ["meadow", "tidal", "staircase"],
        )

    def test_a_missing_script_file_is_skipped_not_fatal(self):
        script_id = add_to_corpus("gone", genre="sleep", corpus_dir=self.dir)
        (self.dir / "scripts" / f"{script_id}.txt").unlink()
        self.assertEqual(load_corpus(genre="sleep", corpus_dir=self.dir), [])


if __name__ == "__main__":
    unittest.main()
