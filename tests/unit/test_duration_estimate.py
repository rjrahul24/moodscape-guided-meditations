"""Tests for script runtime estimation.

Pure arithmetic over the engine's own segment parse — no model, no audio.
"""

import unittest

from core.script_gen.duration import (
    DEFAULT_WPM,
    estimate_duration_sec,
    PAUSE_SCALE_BOUNDS,
    log_estimate_accuracy,
    scale_pauses,
    total_pause_sec,
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

    def test_inter_sentence_gaps_are_counted_for_kokoro(self):
        # Both are exactly 8 words; only the sentence count differs, so the
        # delta is purely the three inter-sentence gaps (3 x 0.8s). This is
        # Kokoro-specific behaviour -- see test_f5_does_not_add_* below for
        # why f5 must NOT show this delta.
        one = estimate_duration_sec(
            "word word word word word word word word.",
            engine="kokoro",
            wpm=100.0,
        )
        four = estimate_duration_sec(
            "word word. word word. word word. word word.",
            engine="kokoro",
            wpm=100.0,
        )
        self.assertGreater(four, one + 1.5)

    def test_f5_does_not_add_per_sentence_gaps_within_one_chunk(self):
        # F5's engine (core/f5_tts/engine.py) synthesizes every sentence
        # within one <=250-char chunk as continuous prose with no inserted
        # gap -- it only gaps between CHUNKS (core/f5_tts/preprocessor.py's
        # split_into_chunks). A 3-sentence paragraph and the same words as
        # one sentence stay in a single chunk here, so they must estimate
        # nearly the same duration. Applying Kokoro's per-sentence gap model
        # to f5 (the pre-fix bug) would make the 3-sentence version ~1.6s
        # longer for no reason.
        one_sentence = estimate_duration_sec(
            "word word word word word word word word word.",
            engine="f5",
            wpm=100.0,
        )
        three_sentences = estimate_duration_sec(
            "word word word. word word word. word word word.",
            engine="f5",
            wpm=100.0,
        )
        self.assertAlmostEqual(three_sentences, one_sentence, delta=0.05)

    def test_kokoro_gap_matches_engine_constant(self):
        from core.kokoro_tts.engine import INTER_SENTENCE_PAUSE_SEC

        one_sentence = estimate_duration_sec(
            "word word word word word word word word word.",
            engine="kokoro",
            wpm=100.0,
        )
        three_sentences = estimate_duration_sec(
            "word word word. word word word. word word word.",
            engine="kokoro",
            wpm=100.0,
        )
        self.assertAlmostEqual(
            three_sentences - one_sentence,
            2 * INTER_SENTENCE_PAUSE_SEC,
            delta=0.05,
        )

    def test_f5_adds_a_gap_only_between_chunk_boundary_segments(self):
        # A single long paragraph forces core/f5_tts/preprocessor.py's
        # split_into_chunks() to split it into multiple "speech" segments at
        # MAX_CHUNK_CHARS -- exactly the chunk boundary the real F5 engine
        # gaps between. Derive the expected chunk count from prepare_segments
        # itself so this doesn't hardcode MAX_CHUNK_CHARS or exact wrapping.
        from core.f5_tts.preprocessor import prepare_segments
        from core.script_gen.duration import _F5_CHUNK_GAP_SEC

        sentence = "The gentle tide moves in and out with the breath. "
        script = sentence * 20
        segments = prepare_segments(script, content_type="meditation")
        speech_segments = [s for s in segments if s["type"] == "speech"]
        self.assertGreater(
            len(speech_segments),
            1,
            "test script must actually split into multiple f5 chunks",
        )

        total_words = sum(len(s["text"].split()) for s in speech_segments)
        expected_speech = total_words / 100.0 * 60.0
        expected_gaps = (len(speech_segments) - 1) * _F5_CHUNK_GAP_SEC

        estimate = estimate_duration_sec(script, engine="f5", wpm=100.0)
        self.assertAlmostEqual(
            estimate, expected_speech + expected_gaps, delta=0.05
        )

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


class TestScalePauses(unittest.TestCase):
    def test_bounds_constant(self):
        self.assertEqual(PAUSE_SCALE_BOUNDS, (0.85, 1.15))

    def test_factor_one_is_identity(self):
        script = "Breathe in. [pause:4s] And out. [pause:2.5s] Rest."
        self.assertEqual(scale_pauses(script, 1.0), script)

    def test_shrinks_and_rounds_to_half_second(self):
        self.assertEqual(scale_pauses("A. [pause:4s] B.", 0.85), "A. [pause:3.5s] B.")

    def test_one_second_floor(self):
        self.assertEqual(scale_pauses("A. [pause:1s] B.", 0.85), "A. [pause:1s] B.")

    def test_tiny_pauses_floor_at_one_second(self):
        out = scale_pauses("A. [pause:0.5s] B. [pause:1.2s] C.", 0.85)
        self.assertEqual(out, "A. [pause:1s] B. [pause:1s] C.")

    def test_factor_above_bound_is_clamped(self):
        # 10 * 1.15 = 11.5 (not 20)
        self.assertEqual(scale_pauses("[pause:10s]", 2.0), "[pause:11.5s]")

    def test_factor_below_bound_is_clamped(self):
        # 10 * 0.85 = 8.5 (not 1)
        self.assertEqual(scale_pauses("[pause:10s]", 0.1), "[pause:8.5s]")

    def test_decimal_input_and_integer_output_format(self):
        # 2.5 * 1.15 = 2.875 -> round(5.75)/2 = 3.0 -> "3s"
        self.assertEqual(scale_pauses("[pause:2.5s]", 1.15), "[pause:3s]")

    def test_no_pauses_unchanged(self):
        script = "Just speech, nothing else.\n\nAnother paragraph."
        self.assertEqual(scale_pauses(script, 0.85), script)

    def test_other_bracket_tags_untouched(self):
        script = "[breath] In. [inhale] [pause:6s] [exhale] [voice:calm] Out."
        self.assertEqual(
            scale_pauses(script, 0.85),
            "[breath] In. [inhale] [pause:5s] [exhale] [voice:calm] Out.",
        )


class TestTotalPauseSec(unittest.TestCase):
    def test_sums_explicit_markers(self):
        self.assertAlmostEqual(
            total_pause_sec("A. [pause:4s] B. [pause:2.5s] C. [breath]"), 6.5
        )

    def test_no_pauses_is_zero(self):
        self.assertEqual(total_pause_sec("No pauses here."), 0.0)


class TestElevenLabsEstimate(unittest.TestCase):
    def test_faster_than_f5_and_breaths_counted(self):
        script = "Rest here now. [pause:5s]\n\nLet go. [exhale] Soften."
        el = estimate_duration_sec(script, engine="elevenlabs")
        f5 = estimate_duration_sec(script, engine="f5")
        self.assertGreater(el, 5.0 + 1.8)   # pause + [exhale] silence still counted
        self.assertLess(el, f5)

    def test_elevenlabs_wpm_is_measured_value(self):
        self.assertEqual(DEFAULT_WPM["elevenlabs"], 130.0)


if __name__ == "__main__":
    unittest.main()
