"""Auto-generate on the ElevenLabs provider (no network: fakes only)."""

import json

import pytest

from core.auto_generate import AutoConfig, ScriptGenerationError, render_script, tts_preflight
from core.genres import load_pack


class _Client:
    def __init__(self, exists=True):
        self.exists = exists

    def voice_exists(self, voice_id):
        return self.exists


class _Pipe:
    def __init__(self, path):
        self.path, self.kwargs = path, None
        self.last_tts_manifest = {"chars_billed": 10, "cache_hits": 1}

    def generate(self, **kw):
        self.kwargs = kw
        self.path.write_bytes(b"RIFF")
        return str(self.path), "ok"


def test_from_genre_picks_elevenlabs_default_voice():
    cfg = AutoConfig.from_genre(load_pack("stress_relief"), tts_engine="elevenlabs")
    assert cfg.elevenlabs_voice_id == "pjcYQlDFKMbcOUp6F5GD"
    assert cfg.voice_slug is None


def test_preflight_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVEN_LABS_API_KEY", raising=False)
    with pytest.raises(ScriptGenerationError, match="ELEVENLABS_API_KEY"):
        tts_preflight(AutoConfig(tts_engine="elevenlabs", elevenlabs_voice_id="v"))


def test_preflight_unknown_voice(monkeypatch):
    with pytest.raises(ScriptGenerationError, match="Voice Library"):
        tts_preflight(AutoConfig(tts_engine="elevenlabs", elevenlabs_voice_id="v"),
                      client_factory=lambda: _Client(exists=False))


def test_preflight_bad_env_direction(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION", "[sighs]")
    with pytest.raises(ScriptGenerationError, match="voice quality"):
        tts_preflight(AutoConfig(tts_engine="elevenlabs", elevenlabs_voice_id="v"),
                      client_factory=lambda: _Client())


def test_preflight_noop_for_local():
    tts_preflight(AutoConfig(tts_engine="f5"))


def test_render_script_passes_voice_and_records_meta(tmp_path, monkeypatch):
    monkeypatch.setattr("core.auto_generate._measure_actual_duration_sec", lambda p: 60.0)
    pipe = _Pipe(tmp_path / "out.wav")
    cfg = AutoConfig(tts_engine="elevenlabs", elevenlabs_voice_id="v")
    res = render_script("Rest here. [pause:3s]\n\nLet go.", config=cfg,
                        background_label="bg", background_path="bg.mp3", pipeline=pipe)
    assert pipe.kwargs["elevenlabs_voice_id"] == "v"
    assert "f5_voice_slug" not in pipe.kwargs
    meta = json.loads(open(res.meta_path).read())
    assert meta["voice"] == "v" and meta["elevenlabs"]["chars_billed"] == 10
    assert meta["elevenlabs"]["estimated_chars"] > 0
    assert res.voice == "v"
