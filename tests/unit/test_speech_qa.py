import numpy as np

from core.eval.speech_metrics import Transcript, WhisperTranscriber, Word
from core.speech_qa import (
    HIGH_CER,
    INTERNAL_GAP,
    TRUNCATED,
    chunk_signal_checks,
    rate_outliers,
    score_candidate,
    sec_per_char,
    transcript_checks,
)

SR = 24000


def _tone(sec):
    t = np.arange(int(SR * sec)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _sil(sec):
    return np.zeros(int(SR * sec), dtype=np.float32)


def test_signal_checks_never_emit_internal_gap():
    # Raw silence is not a defect on its own: F5's natural comma/sentence
    # pauses run 0.6-1.7 s. Gaps are judged by Whisper word timings instead.
    audio = np.concatenate([_tone(1.0), _sil(1.5), _tone(1.0)])
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
    def __init__(self, text, words=None):
        self.text = text
        self.words = words or []
        self.calls = 0

    def transcribe(self, audio, sr):
        self.calls += 1
        return Transcript(self.text, self.words)


def test_transcript_checks_cer():
    audio = _tone(1.0)
    fake = _Fake("breathe in slowly")
    assert transcript_checks(audio, SR, "Breathe in slowly.", fake) == ([], 0.0)
    assert fake.calls == 1
    codes, value = transcript_checks(
        audio, SR, "Breathe in slowly.", _Fake("completely different sentence here")
    )
    assert codes == [HIGH_CER] and value > 0.2


def _gap_audio():
    # tone 0-1 s, silence 1.0-1.6 s, tone 1.6-2.6 s
    return np.concatenate([_tone(1.0), _sil(0.6), _tone(1.0)])


def test_silence_between_words_at_comma_not_flagged():
    words = [Word("Breathe,", 0.1, 1.0), Word("slowly.", 1.6, 2.5)]
    codes, _ = transcript_checks(_gap_audio(), SR, "Breathe, slowly.", _Fake("Breathe, slowly.", words))
    assert INTERNAL_GAP not in codes


def test_silence_between_words_mid_clause_flagged():
    words = [Word("Breathe", 0.1, 1.0), Word("slowly.", 1.6, 2.5)]
    codes, _ = transcript_checks(_gap_audio(), SR, "Breathe slowly.", _Fake("Breathe slowly.", words))
    assert codes == [INTERNAL_GAP]


def test_silence_inside_word_span_flagged():
    words = [Word("Breathe", 0.1, 0.5), Word("slowly.", 0.6, 2.5)]
    codes, _ = transcript_checks(_gap_audio(), SR, "Breathe slowly.", _Fake("Breathe slowly.", words))
    assert codes == [INTERNAL_GAP]


def test_leading_lead_in_and_silence_not_flagged():
    # F5 chunks open with a ~0.3 s ref-pad noise lead-in at about -50 dB, then
    # ~1.5 s of near-silence before speech; that is not a stall.
    rng = np.random.default_rng(0)
    lead_in = (rng.standard_normal(int(SR * 0.3)) * 10 ** (-50 / 20)).astype(np.float32)
    audio = np.concatenate([lead_in, _sil(1.5), _tone(1.0), _tone(1.0)])
    words = [Word("Breathe", 1.8, 2.8), Word("slowly.", 2.8, 3.8)]
    codes, _ = transcript_checks(audio, SR, "Breathe slowly.", _Fake("Breathe slowly.", words))
    assert codes == []
    assert chunk_signal_checks(audio, SR, "Breathe slowly.", median_sec_per_char=None) == []


def test_score_candidate_ordering():
    assert score_candidate([], 0.5, 1.0) < score_candidate(["X"], 0.0, 0.0)
    assert score_candidate(["X"], 0.1, 1.0) < score_candidate(["X"], 0.2, 0.0)
    assert score_candidate([], 0.1, 0.1) < score_candidate([], 0.1, 0.2)


def test_whisper_device_stored_without_load():
    t = WhisperTranscriber(device="cpu")
    assert t.device == "cpu" and t._pipe is None
