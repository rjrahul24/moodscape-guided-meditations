"""F5Engine MOODSCAPE_F5_CHUNK_QA best-of-3 re-render, with fake model + transcriber."""

import numpy as np
import pytest

from core.eval.speech_metrics import Transcript
from core.f5_tts import engine as eng
from core.speech_engine import SAMPLE_RATE

REF_TEXT = "This is a calm reference transcript that the voice reads aloud for cloning."
REF_DUR = 11.44
SPEED = 0.88
BASE_SEED = 100
LONG = (
    "Settle into your seat and let your shoulders drop away from your ears. "
    "Notice the gentle rise and fall of your chest as you breathe. "
    "Allow every thought to drift past like a slow cloud in a quiet sky."
)


def _amp(seed: int) -> float:
    return np.float32(0.1 + seed * 1e-6)


def _seed_of(audio: np.ndarray) -> int:
    return int(round((float(audio[0]) - 0.1) / 1e-6))


class SeedModel:
    """Constant-amplitude output whose level encodes the seed; length ∝ chars."""

    def __init__(self):
        self.calls = []
        self.text_by_seed = {}

    def infer(self, **kw):
        self.calls.append(kw)
        self.text_by_seed[kw["seed"]] = kw["gen_text"]
        n = int(0.01 * len(kw["gen_text"]) * SAMPLE_RATE)
        return np.full(n, _amp(kw["seed"]), dtype=np.float32), SAMPLE_RATE, None


class FakeTranscriber:
    """Returns the right text unless the seed is in `wrong` (seed -> transcript)."""

    def __init__(self, model, wrong=None, raise_on_transcribe=False):
        self.model = model
        self.wrong = wrong or {}
        self.raise_on_transcribe = raise_on_transcribe
        self.unloaded = False
        self.seen = []

    def transcribe(self, audio, sr):
        if self.raise_on_transcribe:
            raise RuntimeError("whisper exploded")
        seed = _seed_of(audio)
        self.seen.append(seed)
        if seed in self.wrong:
            return Transcript(self.wrong[seed])
        return Transcript(self.model.text_by_seed[seed])

    def unload(self):
        self.unloaded = True


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(eng, "_apply_silero_vad", lambda a, sr: a)
    monkeypatch.setattr(eng, "_trim_trailing_silence", lambda a, sr: a)
    monkeypatch.setenv("MOODSCAPE_F5_SHORT_PHRASE_PACING", "0")
    monkeypatch.setenv("MOODSCAPE_F5_CHUNKING", "1")
    monkeypatch.setenv("MOODSCAPE_F5_CHUNK_QA", "1")
    monkeypatch.delenv("MOODSCAPE_F5_QA_DEVICE", raising=False)
    e = eng.F5Engine.__new__(eng.F5Engine)
    e._model = SeedModel()
    e._phase_assets = {
        "default": {"audio": "ref.wav", "text": REF_TEXT, "duration_sec": REF_DUR}
    }
    e.last_chunks = []
    e.last_qa = None
    return e


def _install(monkeypatch, transcriber):
    devices = []

    def factory(device):
        devices.append(device)
        return transcriber

    monkeypatch.setattr(eng, "_make_qa_transcriber", factory)
    return devices


def _run(e, segs):
    return e.synthesize(segs, speed=SPEED, seed=BASE_SEED, target_wpm=None)


def test_fail_then_pass_keeps_second_attempt(engine, monkeypatch):
    t = FakeTranscriber(engine._model, wrong={BASE_SEED + 1: "completely different words here"})
    devices = _install(monkeypatch, t)
    audio, _ = _run(engine, [{"type": "speech", "text": LONG}])
    n = len(engine.last_chunks)
    assert n >= 3
    qa = engine.last_qa
    assert qa["chunks"] == n
    assert qa["failed_first_pass"] == 1
    assert qa["rerendered"] == 1
    assert qa["still_failing"] == 0
    assert qa["codes"] == {"HIGH_CER": 1}
    assert qa["device"] == "mps"
    assert devices == ["mps"]
    # Early stop: only the +10_000 candidate was rendered.
    assert [c["seed"] for c in engine._model.calls[n:]] == [BASE_SEED + 1 + 10_000]
    rec = engine.last_chunks[1]
    assert rec.seed == BASE_SEED + 1 + 10_000
    assert _seed_of(rec.audio) == BASE_SEED + 1 + 10_000
    assert np.sum(audio == _amp(BASE_SEED + 1 + 10_000)) > 0
    assert np.sum(audio == _amp(BASE_SEED + 1)) == 0
    assert t.unloaded


def test_all_three_fail_keeps_best_and_bounds_attempts(engine, monkeypatch):
    s = BASE_SEED + 1
    t = FakeTranscriber(engine._model)
    _install(monkeypatch, t)

    class Wrong(dict):
        def __contains__(self, seed):
            return seed in (s, s + 10_000, s + 20_000)

        def __getitem__(self, seed):
            text = engine._model.text_by_seed[seed]
            # +10_000 is the least wrong (30% cut, CER ~0.3); others far worse.
            if seed == s + 10_000:
                return text[: int(len(text) * 0.7)]
            return text[: int(len(text) * 0.2)]

    t.wrong = Wrong()
    audio, _ = _run(engine, [{"type": "speech", "text": LONG}])
    n = len(engine.last_chunks)
    qa = engine.last_qa
    assert qa["failed_first_pass"] == 1
    assert qa["rerendered"] == 1
    assert qa["still_failing"] == 1
    extra = engine._model.calls[n:]
    assert [c["seed"] for c in extra] == [s + 10_000, s + 20_000]
    assert len(extra) <= 2 * qa["failed_first_pass"]
    # Same text and speed as the original chunk.
    assert all(c["gen_text"] == engine.last_chunks[1].text for c in extra)
    assert all(c["speed"] == engine.last_chunks[1].speed for c in extra)
    assert engine.last_chunks[1].seed == s + 10_000
    assert np.sum(audio == _amp(s + 10_000)) > 0


def test_all_pass_no_rerender(engine, monkeypatch):
    t = FakeTranscriber(engine._model)
    _install(monkeypatch, t)
    _run(engine, [{"type": "speech", "text": LONG}])
    n = len(engine.last_chunks)
    assert len(engine._model.calls) == n
    assert engine.last_qa == {
        "chunks": n, "failed_first_pass": 0, "rerendered": 0, "still_failing": 0,
        "codes": {}, "device": "mps",
    }
    assert t.unloaded


def test_device_env_override(engine, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_F5_QA_DEVICE", "cpu")
    devices = _install(monkeypatch, FakeTranscriber(engine._model))
    _run(engine, [{"type": "speech", "text": LONG}])
    assert devices == ["cpu"]
    assert engine.last_qa["device"] == "cpu"


def test_chunk_qa_without_chunking_is_noop(engine, monkeypatch):
    monkeypatch.delenv("MOODSCAPE_F5_CHUNKING", raising=False)
    devices = _install(monkeypatch, FakeTranscriber(engine._model))
    _run(engine, [{"type": "speech", "text": LONG}])
    assert devices == []
    assert engine.last_qa is None
    assert len(engine._model.calls) == 1


def test_both_flags_off_is_noop(engine, monkeypatch):
    monkeypatch.delenv("MOODSCAPE_F5_CHUNKING", raising=False)
    monkeypatch.delenv("MOODSCAPE_F5_CHUNK_QA", raising=False)
    devices = _install(monkeypatch, FakeTranscriber(engine._model))
    _run(engine, [{"type": "speech", "text": LONG}])
    assert devices == []
    assert engine.last_qa is None


def test_chunking_without_chunk_qa_is_noop(engine, monkeypatch):
    monkeypatch.delenv("MOODSCAPE_F5_CHUNK_QA", raising=False)
    devices = _install(monkeypatch, FakeTranscriber(engine._model))
    _run(engine, [{"type": "speech", "text": LONG}])
    assert devices == []
    assert engine.last_qa is None


def test_transcriber_error_keeps_first_pass_and_unloads(engine, monkeypatch):
    t = FakeTranscriber(engine._model, raise_on_transcribe=True)
    _install(monkeypatch, t)
    audio, act = _run(engine, [{"type": "speech", "text": LONG}])
    n = len(engine.last_chunks)
    assert len(audio) > 0 and len(audio) == len(act)
    assert engine.last_qa == {"error": "RuntimeError: whisper exploded"}
    assert t.unloaded
    assert len(engine._model.calls) == n
    assert np.sum(audio == _amp(BASE_SEED + 1)) > 0


def test_factory_error_is_swallowed(engine, monkeypatch):
    def boom(device):
        raise OSError("no weights")

    monkeypatch.setattr(eng, "_make_qa_transcriber", boom)
    audio, _ = _run(engine, [{"type": "speech", "text": LONG}])
    assert len(audio) > 0
    assert engine.last_qa == {"error": "OSError: no weights"}


def test_chunk_index_points_past_leading_pause(engine, monkeypatch):
    """ChunkRecord.index addresses the chunks list, which includes pauses."""
    pause_sec = 1.0
    segs = [
        {"type": "pause", "duration_sec": pause_sec},
        {"type": "speech", "text": "Breathe in slowly and fully now."},
    ]
    t = FakeTranscriber(engine._model, wrong={BASE_SEED: "nothing like it"})
    _install(monkeypatch, t)
    audio, _ = _run(engine, segs)
    assert len(engine.last_chunks) == 1
    rec = engine.last_chunks[0]
    assert rec.index == 1
    new = _amp(BASE_SEED + 10_000)
    pause_n = int(pause_sec * SAMPLE_RATE)
    # Replacement lands right after the pause, and the pause is untouched.
    assert np.all(audio[pause_n:] == new)
    assert not np.any(audio[:pause_n] == new)
    assert len(audio) == pause_n + len(rec.audio)
