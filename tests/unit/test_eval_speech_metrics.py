import unittest

import numpy as np

from core.eval.speech_metrics import (
    Transcript,
    Word,
    cer,
    internal_gaps,
    normalize_text,
    rate_spread,
    speech_metrics,
    wer,
)

SR = 16000


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeTranscriber:
    def __init__(self, transcript: Transcript):
        self._t = transcript

    def transcribe(self, audio, sr):
        return self._t


class TestNormalize(unittest.TestCase):
    def test_normalize_handles_tags_digits_apostrophes(self):
        self.assertEqual(
            normalize_text("Breathe in for 4. [pause:3s] Don’t RUSH!"),
            "breathe in for four don't rush",
        )


class TestErrorRates(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(cer("let it be", "let it be"), 0.0)
        self.assertEqual(wer("let it be", "let it be"), 0.0)

    def test_one_word_dropped(self):
        self.assertAlmostEqual(wer("rest your hands now", "rest your now"), 0.25)

    def test_empty_reference(self):
        self.assertEqual(cer("", ""), 0.0)
        self.assertEqual(cer("", "extra"), 1.0)


class TestInternalGaps(unittest.TestCase):
    def test_planted_gap_inside_word_is_counted(self):
        audio = np.concatenate([_tone(0.5), np.zeros(int(SR * 0.4), np.float32), _tone(0.5)])
        words = [Word("relax", 0.0, 1.4)]
        self.assertEqual(internal_gaps(audio, SR, words), 1)

    def test_no_gap_in_continuous_speech(self):
        words = [Word("relax", 0.0, 1.0)]
        self.assertEqual(internal_gaps(_tone(1.0), SR, words), 0)

    def test_long_gap_between_words_mid_sentence_is_counted(self):
        words = [Word("rest", 0.0, 0.4), Word("here", 1.0, 1.4)]
        self.assertEqual(internal_gaps(_tone(1.4), SR, words), 1)

    def test_gap_after_sentence_end_is_intended(self):
        words = [Word("rest.", 0.0, 0.4), Word("Now", 3.0, 3.4)]
        self.assertEqual(internal_gaps(_tone(3.4), SR, words), 0)


class TestRateSpread(unittest.TestCase):
    def test_uniform_rate_has_zero_iqr(self):
        words = [Word("abc.", 0.0, 0.3), Word("abc.", 1.0, 1.3), Word("abc.", 2.0, 2.3)]
        spread = rate_spread(words)
        self.assertAlmostEqual(spread["sec_per_char_p50"], 0.1)
        self.assertAlmostEqual(spread["sec_per_char_iqr"], 0.0)

    def test_no_words(self):
        self.assertEqual(rate_spread([]), {"sec_per_char_p50": None, "sec_per_char_iqr": None})


class TestSpeechMetrics(unittest.TestCase):
    def test_uses_injected_transcriber(self):
        t = Transcript("Rest your hands.", [Word("Rest", 0.0, 0.3), Word("your", 0.3, 0.6), Word("hands.", 0.6, 1.0)])
        m = speech_metrics(_tone(1.0), SR, "Rest your hands. [pause:2s]", transcriber=FakeTranscriber(t))
        self.assertEqual(m["cer"], 0.0)
        self.assertEqual(m["internal_gaps"], 0)
        self.assertEqual(m["transcript"], "Rest your hands.")


if __name__ == "__main__":
    unittest.main()
