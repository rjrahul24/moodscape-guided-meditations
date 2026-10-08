import numpy as np

from core.eval.speech_metrics import Transcript, WhisperTranscriber
from core.speech_qa import (
    HIGH_CER,
    INTERNAL_GAP,
    TRUNCATED,
    chunk_signal_checks,
    rate_outliers,
    score_candidate,
    sec_per_char,
    transcript_check,
)

SR = 24000


def _tone(sec):
    t = np.arange(int(SR * sec)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _sil(sec):
    return np.zeros(int(SR * sec), dtype=np.float32)


def test_internal_gap_flagged():
    audio = np.concatenate([_tone(1.0), _sil(0.8), _tone(1.0)])
    assert INTERNAL_GAP in chunk_signal_checks(audio, SR, "hello", median_sec_per_char=None)


def test_leading_trailing_silence_not_gap():
    audio = np.concatenate([_sil(1.0), _tone(1.0), _sil(1.0)])
    assert INTERNAL_GAP not in chunk_signal_checks(audio, SR, "hello", median_sec_per_char=None)


def test_truncated_flagged_and_short_text_never():
    audio = _tone(0.2)
    text = "a" * 40
    assert TRUNCATED in chunk_signal_checks(audio, SR, text, median_sec_per_char=0.08)
    assert TRUNCATED not in chunk_signal_checks(audio, SR, "hello", median_sec_per_char=0.08)
    assert TRUNCATED not in chunk_signal_checks(audio, SR, text, median_sec_per_char=None)


def test_sec_per_char():
    assert abs(sec_per_char(_tone(1.0), SR, "ab cd") - 0.25) < 1e-6
    assert sec_per_char(_tone(1.0), SR, "  ") == 0.0


def test_rate_outliers():
    rates = [(0.08, 40)] * 5 + [(0.2, 40)]
    assert rate_outliers(rates) == {5}
    assert rate_outliers([(0.08, 40)] * 5 + [(0.3, 10)]) == set()
    assert rate_outliers([(0.08, 40), (0.3, 40)]) == set()


class _Fake:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio, sr):
        return Transcript(self.text, [])


def test_transcript_check():
    audio = _tone(1.0)
    assert transcript_check(audio, SR, "Breathe in slowly.", _Fake("breathe in slowly")) == (None, 0.0)
    code, value = transcript_check(audio, SR, "Breathe in slowly.", _Fake("completely different sentence here"))
    assert code == HIGH_CER and value > 0.2


def test_score_candidate_ordering():
    assert score_candidate([], 0.5, 1.0) < score_candidate(["X"], 0.0, 0.0)
    assert score_candidate(["X"], 0.1, 1.0) < score_candidate(["X"], 0.2, 0.0)
    assert score_candidate([], 0.1, 0.1) < score_candidate([], 0.1, 0.2)


def test_whisper_device_stored_without_load():
    t = WhisperTranscriber(device="cpu")
    assert t.device == "cpu" and t._pipe is None
