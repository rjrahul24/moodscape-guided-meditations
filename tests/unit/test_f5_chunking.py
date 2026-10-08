"""Tests for the F5 breath-group chunker (pure functions)."""
import unittest

from f5_tts.infer.utils_infer import chunk_text

from core.f5_tts.chunking import (
    SAFETY,
    BreathGroup,
    breath_groups,
    f5_char_budget,
    gap_for,
)

TEXT = ("Let your shoulders soften, and let the jaw unclench. Notice the breath; it moves on its own. "
        "There is nothing to fix here, nothing to change, nothing to reach for.")


class TestBudget(unittest.TestCase):
    def test_mirrors_upstream_formula_with_safety(self):
        ref = "x" * 90  # 90 bytes over 10 s of reference
        self.assertEqual(f5_char_budget(ref, 10.0, 0.8), int(90 / 10.0 * 12.0 * 0.8 * SAFETY))

    def test_minimum_floor(self):
        self.assertEqual(f5_char_budget("x", 10.0, 0.1), 20)


class TestGroups(unittest.TestCase):
    def test_upstream_never_resplits(self):
        budget = 60
        for g in breath_groups(TEXT, budget):
            self.assertEqual(chunk_text(g.text, max_chars=int(budget / SAFETY)), [g.text], g.text)

    def test_text_is_preserved(self):
        groups = breath_groups(TEXT, 60)
        self.assertEqual(" ".join(g.text for g in groups).split(), TEXT.split())

    def test_gaps_follow_punctuation_and_last_is_none(self):
        groups = breath_groups(TEXT, 60)
        self.assertIsNone(groups[-1].gap_after_sec)
        for g in groups[:-1]:
            self.assertEqual(g.gap_after_sec, gap_for(g.text))
        self.assertEqual(gap_for("the jaw unclench."), 0.45)
        self.assertEqual(gap_for("the breath;"), 0.35)
        self.assertEqual(gap_for("soften,"), 0.20)
        self.assertEqual(gap_for('"rest."'), 0.45)
        self.assertEqual(gap_for("and the"), 0.0)
        self.assertEqual(gap_for("rest…"), 0.45)
        self.assertEqual(gap_for("rest —"), 0.20)

    def test_env_overrides_gap(self):
        self.assertEqual(gap_for("x.", environ={"MOODSCAPE_F5_GAP_SENTENCE": "0.6"}), 0.6)

    def test_fits_in_one_group(self):
        self.assertEqual(breath_groups("Breathe in.", 60), [BreathGroup("Breathe in.", None)])

    def test_giant_token_returns_single_group(self):
        token = "a" * 300
        self.assertEqual([g.text for g in breath_groups(token, 60)], [token])

    def test_long_unit_without_punctuation_splits_at_spaces(self):
        text = " ".join(["word"] * 40)  # 199 chars, no punctuation
        groups = breath_groups(text, 60)
        self.assertTrue(all(len(g.text.encode()) <= 60 for g in groups))
        self.assertEqual(" ".join(g.text for g in groups), text)

    def test_dash_is_clause_end_only_before_whitespace(self):
        groups = breath_groups("Rest — and soften, well-being stays", 10)
        self.assertEqual(groups[0].text, "Rest —")
        self.assertEqual(" ".join(g.text for g in groups), "Rest — and soften, well-being stays")

    def test_empty_text(self):
        self.assertEqual(breath_groups("   ", 60), [])


if __name__ == "__main__":
    unittest.main()
