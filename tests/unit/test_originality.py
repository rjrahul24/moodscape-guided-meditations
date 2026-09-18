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

    def test_a_corrupt_index_is_treated_as_empty_not_fatal(self):
        """The corpus is a cache; a truncated index must not stop a run."""
        add_to_corpus("something", genre="sleep", corpus_dir=self.dir)
        (self.dir / "index.json").write_text("{not json at all", encoding="utf-8")
        self.assertEqual(load_corpus(corpus_dir=self.dir), [])


from core.originality import (
    build_idf,
    cosine,
    document_frequencies,
    longest_rare_run,
    ngrams,
    tfidf_vector,
    tokenize,
)

# Two scripts that share only stock meditation language but say different
# things. The check MUST NOT flag this pair -- it is the whole point.
GENERIC_A = (
    "Settle in and let your shoulders drop. Notice your breath without "
    "changing it. When you are ready, let your eyes close. Feel the weight "
    "of your hands resting in your lap. There is nowhere else to be."
)
GENERIC_B = (
    "Settle in and let your shoulders drop. Notice your breath without "
    "changing it. Picture a narrow copper staircase descending into warm "
    "lamplight. Each step takes you further from the noise of the day."
)
# A near-duplicate of GENERIC_B: same storyline, lightly reworded.
NEAR_DUPLICATE_B = (
    "Settle in and let your shoulders relax. Notice your breathing without "
    "changing it. Imagine a narrow copper staircase descending into warm "
    "lamplight. Every step takes you further from the noise of the day."
)


class TokenizeTest(unittest.TestCase):
    def test_lowercases_and_drops_punctuation(self):
        self.assertEqual(tokenize("Breathe In, Slowly."), ["breathe", "in", "slowly"])

    def test_strips_bracket_markers(self):
        self.assertEqual(
            tokenize("Rest. [pause:5s] [breath] Now rise."),
            ["rest", "now", "rise"],
        )

    def test_empty_text_is_empty(self):
        self.assertEqual(tokenize("   "), [])


class NgramTest(unittest.TestCase):
    def test_produces_overlapping_tuples(self):
        self.assertEqual(
            ngrams(["a", "b", "c"], 2), [("a", "b"), ("b", "c")]
        )

    def test_too_few_tokens_yields_nothing(self):
        self.assertEqual(ngrams(["a"], 2), [])


class IdfTest(unittest.TestCase):
    def test_a_term_in_every_document_weighs_less_than_a_rare_one(self):
        docs = [tokenize(t) for t in ["breath calm", "breath storm", "breath river"]]
        idf, _default = build_idf(docs, max_n=1)
        self.assertLess(idf[("breath",)], idf[("calm",)])

    def test_unseen_terms_get_the_maximum_weight(self):
        docs = [tokenize("breath calm"), tokenize("breath storm")]
        idf, default = build_idf(docs, max_n=1)
        self.assertGreaterEqual(default, max(idf.values()))

    def test_document_frequency_counts_documents_not_occurrences(self):
        docs = [tokenize("breath breath breath"), tokenize("river")]
        df = document_frequencies(docs, max_n=1)
        self.assertEqual(df[("breath",)], 1)


class CosineTest(unittest.TestCase):
    def _vectors(self, texts, target_a, target_b):
        docs = [tokenize(t) for t in texts]
        idf, default = build_idf(docs)
        return (
            tfidf_vector(tokenize(target_a), idf, default),
            tfidf_vector(tokenize(target_b), idf, default),
        )

    def test_identical_text_scores_one(self):
        a, b = self._vectors([GENERIC_A, GENERIC_B], GENERIC_A, GENERIC_A)
        self.assertAlmostEqual(cosine(a, b), 1.0, places=6)

    def test_disjoint_text_scores_zero(self):
        a, b = self._vectors(["alpha beta", "gamma delta"], "alpha beta", "gamma delta")
        self.assertAlmostEqual(cosine(a, b), 0.0, places=6)

    def test_empty_vector_scores_zero_without_dividing_by_zero(self):
        self.assertEqual(cosine({}, {("a",): 1.0}), 0.0)

    def test_near_duplicates_score_higher_than_generic_overlap(self):
        """The central requirement, as a single ordering assertion.

        Two scripts sharing only stock meditation phrasing must be measurably
        less similar than a pair that shares an actual storyline.
        """
        corpus = [GENERIC_A, GENERIC_B, NEAR_DUPLICATE_B]
        generic_pair = self._vectors(corpus, GENERIC_A, GENERIC_B)
        duplicate_pair = self._vectors(corpus, GENERIC_B, NEAR_DUPLICATE_B)
        self.assertLess(cosine(*generic_pair), cosine(*duplicate_pair))


LIFTED = (
    "A different opening entirely, about morning light on a kitchen floor. "
    "Picture a narrow copper staircase descending into warm lamplight. "
    "Then something else again, about the sound of a kettle."
)


class LongestRareRunTest(unittest.TestCase):
    def test_no_shared_text_returns_zero(self):
        a, b = tokenize("alpha beta gamma delta epsilon zeta"), tokenize(
            "one two three four five six"
        )
        df = document_frequencies([a, b], max_n=5)
        span, text = longest_rare_run(a, b, df)
        self.assertEqual((span, text), (0, ""))

    def test_a_lifted_passage_is_found_inside_different_surroundings(self):
        a, b = tokenize(LIFTED), tokenize(GENERIC_B)
        df = document_frequencies([a, b], max_n=5)
        span, text = longest_rare_run(a, b, df)
        self.assertGreaterEqual(span, 9)
        self.assertIn("copper staircase descending", text)

    def test_common_phrases_are_excluded_by_the_df_filter(self):
        """A phrase appearing in many documents is not a lifted passage."""
        shared = tokenize("notice your breath without changing it at all today")
        corpus = [shared for _ in range(6)]
        df = document_frequencies(corpus, max_n=5)
        span, _text = longest_rare_run(shared, shared, df, max_df=2)
        self.assertEqual(span, 0)

    def test_identical_documents_return_their_full_length(self):
        tokens = tokenize(GENERIC_B)
        df = document_frequencies([tokens], max_n=5)
        span, _text = longest_rare_run(tokens, tokens, df)
        self.assertEqual(span, len(tokens))

    def test_too_short_inputs_do_not_raise(self):
        a = tokenize("only three words")
        df = document_frequencies([a], max_n=5)
        self.assertEqual(longest_rare_run(a, a, df), (0, ""))


if __name__ == "__main__":
    unittest.main()
