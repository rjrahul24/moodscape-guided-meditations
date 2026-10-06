import unittest

import numpy as np

from core.eval.speech_metrics import (
    Transcript,
    Word,
    WhisperTranscriber,
    _words_from_chunks,
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
        audio = np.concatenate([_tone(0.4), np.zeros(int(SR * 0.6), np.float32), _tone(0.4)])
        words = [Word("rest", 0.0, 0.4), Word("here", 1.0, 1.4)]
        self.assertEqual(internal_gaps(audio, SR, words), 1)

    def test_between_word_gap_with_energy_is_not_counted(self):
        words = [Word("rest", 0.0, 0.4), Word("here", 1.0, 1.4)]
        self.assertEqual(internal_gaps(_tone(1.4), SR, words), 0)

    def test_silence_touching_word_edge_is_ignored(self):
        audio = np.concatenate([_tone(0.5), np.zeros(int(SR * 0.5), np.float32)])
        self.assertEqual(internal_gaps(audio, SR, [Word("relax", 0.0, 1.0)]), 0)
        audio = np.concatenate([np.zeros(int(SR * 0.5), np.float32), _tone(0.5)])
        self.assertEqual(internal_gaps(audio, SR, [Word("relax", 0.0, 1.0)]), 0)

    def test_curly_quote_and_ellipsis_end_clause(self):
        audio = np.concatenate([_tone(0.4), np.zeros(int(SR * 0.6), np.float32), _tone(0.4)])
        for end in ("rest\u2026", "rest.\u201d", "rest\u2014", "rest,\u2019"):
            words = [Word(end, 0.0, 0.4), Word("here", 1.0, 1.4)]
            self.assertEqual(internal_gaps(audio, SR, words), 0, end)

    def test_gap_after_sentence_end_is_intended(self):
        words = [Word("rest.", 0.0, 0.4), Word("Now", 3.0, 3.4)]
        self.assertEqual(internal_gaps(_tone(3.4), SR, words), 0)


class TestRateSpread(unittest.TestCase):
    def test_uniform_rate_has_zero_iqr(self):
        words = [Word("abc.", 0.0, 0.3), Word("abc.", 1.0, 1.3), Word("abc.", 2.0, 2.3)]
        spread = rate_spread(words)
        self.assertAlmostEqual(spread["sec_per_char_p50"], 0.1)
        self.assertAlmostEqual(spread["sec_per_char_iqr"], 0.0)

    def test_curly_quote_and_ellipsis_end_sentences(self):
        words = [Word("abc\u2026", 0.0, 0.3), Word("abc.\u201d", 1.0, 1.3), Word("abc!\u2019", 2.0, 2.3)]
        self.assertAlmostEqual(rate_spread(words)["sec_per_char_p50"], 0.1)

    def test_no_words(self):
        self.assertEqual(rate_spread([]), {"sec_per_char_p50": None, "sec_per_char_iqr": None})


class TestWhisperTranscriber(unittest.TestCase):
    def test_missing_end_falls_back_and_zero_is_kept(self):
        chunks = [
            {"text": " a", "timestamp": (0.0, 0.0)},
            {"text": " b", "timestamp": (0.5, None)},
            {"text": " c", "timestamp": (1.2, None)},
        ]
        words = _words_from_chunks(chunks, 2.0)
        self.assertEqual([(w.start, w.end) for w in words], [(0.0, 0.0), (0.5, 1.2), (1.2, 2.0)])

    def test_torchcodec_disabled_only_during_pipeline_call(self):
        from transformers.pipelines import automatic_speech_recognition as asr

        seen = {}

        def fake_pipe(inputs, **kwargs):
            seen["during"] = asr.is_torchcodec_available()
            return {"text": " hi", "chunks": [{"text": " hi", "timestamp": (0.0, None)}]}

        before = asr.is_torchcodec_available
        t = WhisperTranscriber()
        t._pipe = fake_pipe
        out = t.transcribe(_tone(1.0), SR)
        self.assertFalse(seen["during"])
        self.assertIs(asr.is_torchcodec_available, before)
        self.assertEqual(out.words[0].end, 1.0)


class TestSpeechMetrics(unittest.TestCase):
    def test_uses_injected_transcriber(self):
        t = Transcript("Rest your hands.", [Word("Rest", 0.0, 0.3), Word("your", 0.3, 0.6), Word("hands.", 0.6, 1.0)])
        m = speech_metrics(_tone(1.0), SR, "Rest your hands. [pause:2s]", transcriber=FakeTranscriber(t))
        self.assertEqual(m["cer"], 0.0)
        self.assertEqual(m["internal_gaps"], 0)
        self.assertEqual(m["transcript"], "Rest your hands.")


if __name__ == "__main__":
    unittest.main()
