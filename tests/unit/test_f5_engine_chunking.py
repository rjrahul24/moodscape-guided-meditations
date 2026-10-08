"""F5Engine MOODSCAPE_F5_CHUNKING integration, with a fake model (no weights)."""

import numpy as np
import pytest

from core.f5_tts import engine as eng
from core.speech_engine import SAMPLE_RATE

REF_TEXT = "This is a calm reference transcript that the voice reads aloud for cloning."
REF_DUR = 11.44
SPEED = 0.88
LONG = (
    "Settle into your seat and let your shoulders drop away from your ears. "
    "Notice the gentle rise and fall of your chest as you breathe. "
    "Allow every thought to drift past like a slow cloud in a quiet sky."
)


class FakeModel:
    def __init__(self):
        self.calls = []

    def infer(self, **kw):
        self.calls.append(kw)
        n = int(0.1 * len(kw["gen_text"]) * SAMPLE_RATE / 10)
        return np.ones(n, dtype=np.float32), SAMPLE_RATE, None


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(eng, "_apply_silero_vad", lambda a, sr: a)
    monkeypatch.setattr(eng, "_trim_trailing_silence", lambda a, sr: a)
    monkeypatch.setenv("MOODSCAPE_F5_SHORT_PHRASE_PACING", "0")
    e = eng.F5Engine.__new__(eng.F5Engine)
    e._model = FakeModel()
    e._phase_assets = {
        "default": {"audio": "ref.wav", "text": REF_TEXT, "duration_sec": REF_DUR}
    }
    e.last_chunks = []
    return e


def _run(e, segs, **kw):
    return e.synthesize(segs, speed=SPEED, seed=100, target_wpm=None, **kw)


def test_flag_off_one_call_per_segment_with_today_seeds(engine, monkeypatch):
    monkeypatch.delenv("MOODSCAPE_F5_CHUNKING", raising=False)
    _run(engine, [{"type": "speech", "text": LONG}, {"type": "speech", "text": "Hello there."}])
    assert [c["gen_text"] for c in engine._model.calls] == [LONG, "Hello there."]
    assert [c["seed"] for c in engine._model.calls] == [100, 101]
    assert engine.last_chunks == []


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_flag_on_splits_without_upstream_resplit(engine, monkeypatch):
    from f5_tts.infer.utils_infer import chunk_text

    monkeypatch.setenv("MOODSCAPE_F5_CHUNKING", "1")
    _run(engine, [{"type": "speech", "text": LONG}])
    calls = engine._model.calls
    assert len(calls) > 1
    upstream = len(REF_TEXT.encode("utf-8")) / REF_DUR * (22 - REF_DUR) * SPEED
    for c in calls:
        assert len(chunk_text(c["gen_text"], max_chars=int(upstream))) == 1
    assert [c["seed"] for c in calls] == list(range(100, 100 + len(calls)))
    assert len(engine.last_chunks) == len(calls)
    assert [r.seed for r in engine.last_chunks] == [c["seed"] for c in calls]


def test_flag_on_gap_between_groups(engine, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_F5_CHUNKING", "1")
    audio, _ = _run(engine, [{"type": "speech", "text": LONG}])
    assert len(engine._model.calls) >= 2
    zero_run = best = 0
    for v in audio:
        zero_run = zero_run + 1 if v == 0 else 0
        best = max(best, zero_run)
    assert best >= int((0.45 - 0.03) * SAMPLE_RATE)


def test_flag_on_single_short_sentence_matches_flag_off(engine, monkeypatch):
    segs = [{"type": "speech", "text": "Breathe in slowly."}]
    monkeypatch.delenv("MOODSCAPE_F5_CHUNKING", raising=False)
    off, _ = _run(engine, segs)
    engine._model.calls.clear()
    monkeypatch.setenv("MOODSCAPE_F5_CHUNKING", "1")
    on, _ = _run(engine, segs)
    assert len(engine._model.calls) == 1
    np.testing.assert_array_equal(on, off)

