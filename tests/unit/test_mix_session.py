import os
import time

import numpy as np
import pytest

from core import mix_session as ms


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path))


def _save(run_id):
    v = np.ones(10, np.float32); m = np.ones(10, bool); mu = np.zeros(12, np.float32)
    return ms.save_session(run_id, v, m, mu, {"mix_sr": 48000, "phrases": [(0.0, 1.0)]})


def test_round_trip_and_update():
    _save("r1")
    ms.update_params("r1", {"output_path": "/tmp/x.wav"})
    s = ms.load_session("r1")
    assert s.voice_audio.dtype == np.float32 and s.voice_activity.dtype == bool
    assert len(s.music_audio) == 12
    assert s.params["output_path"] == "/tmp/x.wav" and s.params["phrases"] == [[0.0, 1.0]]


def test_missing_raises_clear_error():
    with pytest.raises(ms.MixSessionNotFound, match="generate again"):
        ms.load_session("nope")


def test_prune_keeps_newest(monkeypatch):
    for i in range(7):
        p = _save(f"r{i}")
        os.utime(p, (time.time() + i, time.time() + i))
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_KEEP", "5")
    assert ms.prune_sessions() == 2
    with pytest.raises(ms.MixSessionNotFound):
        ms.load_session("r0")
    ms.load_session("r6")


def test_run_ids_unique():
    assert ms.new_run_id() != ms.new_run_id()


def test_pipeline_saves_session(tmp_path, monkeypatch):
    from core.pipeline import MeditationPipeline
    from tests.unit.pipeline_fakes import FakeMusic, MockTTS24
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts"))
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", FakeMusic)
    pipe = MeditationPipeline()
    out, _ = pipe.generate(script="Rest.", music_prompt="", tts_engine="mock",
                           custom_tts_engine=MockTTS24(), music_model="upload",
                           uploaded_music_path="dummy.wav")
    s = ms.load_session(pipe.last_run_id)
    assert s.params["output_path"] == out and s.params["mix_sr"] == 48000
    assert len(s.voice_audio) == len(s.voice_activity)
