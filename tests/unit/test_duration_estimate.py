"""Tests for script runtime estimation.

Pure arithmetic over the engine's own segment parse — no model, no audio.
"""

import unittest

from core.script_gen.duration import (
    DEFAULT_WPM,
    estimate_duration_sec,
    log_estimate_accuracy,
)


class TestDurationEstimate(unittest.TestCase):
    def test_explicit_pauses_are_summed(self):
        # Speech between the pauses is required: the preprocessor MERGES
        # adjacent pauses and keeps only the longest, so "[pause:10s]
        # [pause:20s]" back-to-back yields 20s, not 30s.
        script = "One.\n\n[pause:10s]\n\nTwo.\n\n[pause:20s]\n\nThree."
        estimate = estimate_duration_sec(script, engine="f5")
        self.assertGreaterEqual(estimate, 30.0)

    def test_adjacent_pauses_merge_to_the_longest(self):
        # Guards the merge behaviour itself, so the estimator can never drift
        # into naive addition.
        merged = estimate_duration_sec("[pause:10s]\n\n[pause:20s]", engine="f5")
        self.assertLess(merged, 30.0)

    def test_speech_scales_with_word_count(self):
        short = estimate_duration_sec("one two three four five.", engine="f5")
        longer = estimate_duration_sec(
            " ".join(["word"] * 100) + ".", engine="f5"
        )
        self.assertGreater(longer, short)

    def test_wpm_override_is_honoured(self):
        script = " ".join(["word"] * 100) + "."
        fast = estimate_duration_sec(script, engine="f5", wpm=200.0)
        slow = estimate_duration_sec(script, engine="f5", wpm=50.0)
        self.assertGreater(slow, fast)

    def test_hundred_words_at_hundred_wpm_is_about_a_minute(self):
        script = " ".join(["word"] * 100) + "."
        estimate = estimate_duration_sec(script, engine="f5", wpm=100.0)
        # 60s of speech, plus no pauses and a single sentence (no gaps).
        self.assertAlmostEqual(estimate, 60.0, delta=1.0)

    def test_inter_sentence_gaps_are_counted(self):
        # Both are exactly 8 words; only the sentence count differs, so the
        # delta is purely the three inter-sentence gaps (3 x 0.8s).
        one = estimate_duration_sec(
            "word word word word word word word word.", engine="f5", wpm=100.0
        )
        four = estimate_duration_sec(
            "word word. word word. word word. word word.", engine="f5", wpm=100.0
        )
        self.assertGreater(four, one + 1.5)

    def test_both_engines_supported(self):
        script = "Breathe in and let go.\n\n[pause:5s]\n\nBreathe out."
        self.assertGreater(estimate_duration_sec(script, engine="f5"), 0)
        self.assertGreater(estimate_duration_sec(script, engine="kokoro"), 0)

    def test_unknown_engine_raises(self):
        with self.assertRaises(ValueError):
            estimate_duration_sec("Breathe in.", engine="nope")

    def test_empty_script_is_zero(self):
        self.assertEqual(estimate_duration_sec("", engine="f5"), 0.0)

    def test_default_wpm_defined_for_both_engines(self):
        self.assertIn("f5", DEFAULT_WPM)
        self.assertIn("kokoro", DEFAULT_WPM)

    def test_breath_cues_add_their_measured_sample_durations(self):
        # Breath markers must not be zero-cost: [breath]/[inhale]/[exhale] are
        # each placed after an identical sentence run (not interleaved with
        # the sentences) so removing them changes nothing about how the
        # speech text is split into segments/sentences — the only difference
        # is the three breath cues themselves. Assert the delta, not an
        # absolute total, so this stays robust if WPM changes.
        with_breaths = (
            "Breathe in. Hold. Release. Rest. [breath] [inhale] [exhale]"
        )
        without_breaths = "Breathe in. Hold. Release. Rest."
        with_estimate = estimate_duration_sec(with_breaths, engine="f5")
        without_estimate = estimate_duration_sec(without_breaths, engine="f5")
        self.assertGreaterEqual(
            with_estimate - without_estimate, 1.2 + 1.5 + 1.8
        )

    def test_breath_marker_is_not_zero_cost(self):
        # Regression guard for the bug where "breath" segments were silently
        # ignored (only "pause" and "speech" were handled).
        with_breath = estimate_duration_sec("Breathe deeply. [breath]", engine="f5")
        without_breath = estimate_duration_sec("Breathe deeply.", engine="f5")
        self.assertGreater(with_breath, without_breath)

    def test_sleep_story_uses_shorter_paragraph_pauses(self):
        script = "Once there was a lantern.\n\nIt glowed softly."
        meditation = estimate_duration_sec(
            script, engine="f5", content_type="meditation"
        )
        sleep_story = estimate_duration_sec(
            script, engine="f5", content_type="sleep_story"
        )
        self.assertLess(sleep_story, meditation)


class TestAccuracyLogging(unittest.TestCase):
    def test_log_reports_signed_error(self):
        line = log_estimate_accuracy(300.0, 330.0, "f5")
        self.assertIn("f5", line)
        self.assertIn("300", line)
        self.assertIn("330", line)

    def test_log_handles_zero_actual(self):
        line = log_estimate_accuracy(300.0, 0.0, "f5")
        self.assertIsInstance(line, str)


if __name__ == "__main__":
    unittest.main()
