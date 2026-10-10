"""MOODSCAPE_F5_TRIM_LEADIN: trims the reference-pad lead-in of each chunk."""

import numpy as np
import pytest

from core.f5_tts import engine as eng
from core.speech_engine import SAMPLE_RATE

REF_TEXT = "This is a calm reference transcript that the voice reads aloud for cloning."


def _db(x):
    return 10 ** (x / 20.0)


def _noise_then_tone(noise_s=1.5, tone_s=1.0):
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(noise_s * SAMPLE_RATE)).astype(np.float32)
    noise *= _db(-55) / noise.std()
    t = np.arange(int(tone_s * SAMPLE_RATE)) / SAMPLE_RATE
    tone = (_db(-10) * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.concatenate([noise, tone])


def test_trims_leadin_keeps_lead_ms():
    arr = _noise_then_tone()
    out = eng._trim_leading_silence(arr, SAMPLE_RATE, lead_ms=60, floor_db=-35)
    assert abs(len(out) / SAMPLE_RATE - 1.06) <= 0.010
    start = len(arr) - len(out)
    assert abs(out[0]) <= abs(arr[start])


def test_tone_from_zero_unchanged():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    arr = (_db(-10) * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = eng._trim_leading_silence(arr, SAMPLE_RATE, lead_ms=60, floor_db=-35)
    np.testing.assert_array_equal(out, arr)


def test_all_zeros_unchanged():
    arr = np.zeros(SAMPLE_RATE, dtype=np.float32)
    out = eng._trim_leading_silence(arr, SAMPLE_RATE, lead_ms=60, floor_db=-35)
    np.testing.assert_array_equal(out, arr)


def test_too_short_unchanged():
    arr = np.ones(100, dtype=np.float32)
    out = eng._trim_leading_silence(arr, SAMPLE_RATE, lead_ms=60, floor_db=-35)
    np.testing.assert_array_equal(out, arr)


class FakeModel:
    def infer(self, **kw):
        return _noise_then_tone(), SAMPLE_RATE, None


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(eng, "_apply_silero_vad", lambda a, sr: a)
    monkeypatch.setenv("MOODSCAPE_F5_SHORT_PHRASE_PACING", "0")
    e = eng.F5Engine.__new__(eng.F5Engine)
    e._model = FakeModel()
    assets = {"audio": "ref.wav", "text": REF_TEXT, "duration_sec": 11.44}
    return e, assets


def _chunk(e, assets):
    return e._infer_chunk("Hello there.", assets, 0.88, 1, None)


def test_flag_off_matches_old_path(engine, monkeypatch):
    e, assets = engine
    monkeypatch.delenv("MOODSCAPE_F5_TRIM_LEADIN", raising=False)
    out = _chunk(e, assets)
    expected = eng._trim_trailing_silence(_noise_then_tone(), SAMPLE_RATE)
    np.testing.assert_array_equal(out, expected)


def test_flag_on_removes_leadin(engine, monkeypatch):
    e, assets = engine
    monkeypatch.delenv("MOODSCAPE_F5_TRIM_LEADIN", raising=False)
    off = _chunk(e, assets)
    monkeypatch.setenv("MOODSCAPE_F5_TRIM_LEADIN", "1")
    on = _chunk(e, assets)
    assert 1.4 < (len(off) - len(on)) / SAMPLE_RATE < 1.5
