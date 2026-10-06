import unittest

from core.eval.slop import SlopHit, slop_density, slop_hits


class TestSlopHits(unittest.TestCase):
    def test_hard_phrase_found_case_insensitive(self):
        hits = slop_hits("Let the warmth WASH OVER YOU like a tide.")
        self.assertIn(SlopHit("wash over you", "hard"), hits)

    def test_rate_limited_word_boundary(self):
        hits = slop_hits("Breathe gently. The gentleman waits.")
        terms = [h.term for h in hits]
        self.assertEqual(terms.count("gently"), 1)  # "gentleman" must not match

    def test_tags_are_ignored(self):
        self.assertEqual(slop_hits("[pause:3s] [breath]"), [])

    def test_curly_apostrophe_pattern(self):
        hits = slop_hits("It’s not a task, it’s a rest.")
        self.assertIn("not_x_its_y", [h.term for h in hits])

    def test_triple_soft_adjectives(self):
        hits = slop_hits("A warm, soft, gentle light.")
        self.assertIn("triple_soft_adjectives", [h.term for h in hits])

    def test_five_senses_list_is_not_triple_adjective(self):
        hits = slop_hits("Notice what you see, hear, and feel.")
        self.assertNotIn("triple_soft_adjectives", [h.term for h in hits])


class TestSlopDensity(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(slop_density(""), 0.0)

    def test_density_per_150_words(self):
        text = "gently " + "word " * 149  # 150 words, 1 hit
        self.assertAlmostEqual(slop_density(text), 1.0)


if __name__ == "__main__":
    unittest.main()
