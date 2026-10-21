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
    pipe, eng = MeditationPipeline(), FakeEL()
    _run(pipe, eng)
    _run(pipe, eng)
    assert eng.synth == 1 and pipe.last_tts_cache_hit is True
    assert pipe.last_tts_manifest["chars_billed"] == 42


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
