"""Pipeline ElevenLabs branch: native 48 kHz, no upsample/DF/mastering, clean chain."""

import pytest

from core.pipeline import MeditationPipeline
from tests.unit.pipeline_fakes import FakeEL, FakeMusic, MockTTS24


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts"))
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", FakeMusic)
    calls = {"deess": 0}
    def no(*a, **k):
        raise AssertionError("must not run for ElevenLabs")
    monkeypatch.setattr("core.deepfilter_enhancer.enhance_voice_deepfilter", no)
    monkeypatch.setattr("core.audio_processor.upsample_audio", no)
    def spy(audio, sr):
        calls["deess"] += 1
        return audio
    monkeypatch.setattr("core.elevenlabs_tts.postprocessor.deess_elevenlabs", spy)
    return calls


def _run(pipe, eng, **kw):
    return pipe.generate(script="Close your eyes. [pause:2s]\n\nRest.", music_prompt="",
                         tts_engine="elevenlabs", custom_tts_engine=eng, music_model="upload",
                         uploaded_music_path="dummy.wav", elevenlabs_voice_id="v", **kw)


def test_elevenlabs_branch_skips_upsample_and_denoise(env):
    pipe, eng = MeditationPipeline(), FakeEL()
    out, status = _run(pipe, eng)
    assert out and eng.synth == 1 and env["deess"] == 1
    assert pipe.last_tts_manifest["chars_billed"] == 42


def test_cache_hit_restores_48k_and_manifest(env):
    from core.elevenlabs_tts.preprocessor import prepare_segments
    pipe, eng = MeditationPipeline(), FakeEL()
    _run(pipe, eng)
    first = dict(pipe.last_tts_manifest)
    assert first["chars_billed"] == 42 and "tts_cache_hit" not in first
    _run(pipe, eng)
    assert eng.synth == 1 and pipe.last_tts_cache_hit is True
    m = pipe.last_tts_manifest
    # A free re-run reports this run's billing, not the original render's.
    assert m["chars_billed"] == 0 and m["cache_misses"] == 0
    assert m["chars_cached"] == 42 + 8
    n_speech = sum(1 for s in prepare_segments("Close your eyes. [pause:2s]\n\nRest.",
                                               direction=first["direction"])
                   if s["type"] == "speech")
    assert m["cache_hits"] == n_speech
    assert m["tts_cache_hit"] is True
    # Settings, request ids and direction survive.
    for k in ("model_id", "voice_id", "stability", "similarity", "seed", "request_ids", "direction"):
        assert m[k] == first[k]


def test_injected_elevenlabs_engine_uses_pipeline_params(env, tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CACHE_DIR", str(tmp_path / "el_chunks"))
    import numpy as np
    from core.elevenlabs_tts.engine import ElevenLabsEngine, resolve_params

    class FakeClient:
        def __init__(self):
            self.calls = []
        def convert(self, text, voice_id, **kw):
            self.calls.append(dict(text=text, voice_id=voice_id, **kw))
            t = np.arange(24000) / 48000
            return (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16).tobytes(), "rid"

    client = FakeClient()
    eng = ElevenLabsEngine(params=resolve_params(voice_id="v", stability=0.5), client=client)
    _run(MeditationPipeline(), eng, elevenlabs_stability=0.6)
    assert client.calls and all(c["stability"] == 0.6 for c in client.calls)


def test_speed_change_does_not_miss_elevenlabs_cache(env):
    pipe, eng = MeditationPipeline(), FakeEL()
    _run(pipe, eng, speed=0.8)
    _run(pipe, eng, speed=0.75)
    assert eng.synth == 1 and pipe.last_tts_cache_hit is True


def test_changed_elevenlabs_param_misses_cache(env):
    pipe, eng = MeditationPipeline(), FakeEL()
    _run(pipe, eng, elevenlabs_stability=0.5)
    _run(pipe, eng, elevenlabs_stability=0.6)
    _run(pipe, eng, elevenlabs_direction="[warm, low voice]")
    assert eng.synth == 3


def test_invalid_direction_fails_before_tts(env):
    eng = FakeEL()
    with pytest.raises(ValueError, match="voice quality"):
        _run(MeditationPipeline(), eng, elevenlabs_direction="[sighs]")
    assert eng.synth == 0


def test_empty_direction_falls_back_to_default_and_manifest_records_it(env):
    from core.elevenlabs_tts.direction import default_direction
    pipe, eng = MeditationPipeline(), FakeEL()
    _run(pipe, eng, elevenlabs_direction="")
    assert pipe.last_tts_manifest["direction"] == default_direction()
    # Empty textbox and None resolve to the same direction → same cache entry.
    _run(pipe, eng)
    assert eng.synth == 1 and pipe.last_tts_cache_hit is True
    assert pipe.last_tts_manifest["direction"] == default_direction()


def test_manifest_records_explicit_direction(env):
    pipe, eng = MeditationPipeline(), FakeEL()
    _run(pipe, eng, elevenlabs_direction="[warm, low voice]")
    assert pipe.last_tts_manifest["direction"] == "[warm, low voice]"
    assert "direction" not in eng.last_manifest  # engine's own dict not mutated


def test_local_engine_cache_key_ignores_elevenlabs_params(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts"))
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", FakeMusic)
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_STABILITY", "0.6")
    from core import tts_cache
    seen = []
    real = tts_cache.compute_cache_key
    monkeypatch.setattr(tts_cache, "compute_cache_key",
                        lambda **kw: seen.append(kw) or real(**kw))
    MeditationPipeline().generate(script="Rest.", music_prompt="", tts_engine="mock",
                                  custom_tts_engine=MockTTS24(), music_model="upload",
                                  uploaded_music_path="dummy.wav",
                                  elevenlabs_stability=0.9, elevenlabs_voice_id="zzz")
    assert seen and seen[0].get("extra") is None
    assert "MOODSCAPE_ELEVENLABS_" not in seen[0]["env_fingerprint"]


@pytest.fixture
def reuse_env(tmp_path, monkeypatch):
    """Like `env` but lets local runs upsample (both engines run in one test)."""
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts"))
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", FakeMusic)


def _run_local(pipe, eng, **kw):
    return pipe.generate(script="Close your eyes. [pause:2s]\n\nRest.", music_prompt="",
                         tts_engine="mock", custom_tts_engine=eng, music_model="upload",
                         uploaded_music_path="dummy.wav", **kw)


def test_reuse_last_voice_ignores_elevenlabs_take_for_local_engine(reuse_env):
    _run(MeditationPipeline(), FakeEL())                       # latest = 48 kHz EL take
    local = MockTTS24()
    out, _ = _run_local(MeditationPipeline(), local, reuse_last_voice=True)
    assert out and local.synth == 1


def test_reuse_last_voice_ignores_local_take_for_elevenlabs(reuse_env):
    _run_local(MeditationPipeline(), MockTTS24())              # latest = 24 kHz local take
    el = FakeEL()
    out, _ = _run(MeditationPipeline(), el, reuse_last_voice=True)
    assert out and el.synth == 1


def test_reuse_last_voice_same_engine_still_reuses(reuse_env):
    _run_local(MeditationPipeline(), MockTTS24())
    local = MockTTS24()
    _run_local(MeditationPipeline(), local, reuse_last_voice=True)
    assert local.synth == 0
