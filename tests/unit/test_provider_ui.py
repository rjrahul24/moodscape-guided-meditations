import pytest

from core import provider_ui as ui


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ui, "_VALIDATION_CACHE", None, raising=False)


def test_resolve_engine():
    assert ui.resolve_tts_engine("ElevenLabs", "F5-TTS") == "elevenlabs"
    assert ui.resolve_tts_engine("Local Models", "F5-TTS") == "f5"
    assert ui.resolve_tts_engine("Local Models", "Kokoro") == "kokoro"
    assert ui.resolve_tts_engine("Local Models", "Chatterbox TTS") == "chatterbox"


def test_visibility_cloud_vs_local():
    cloud = ui.provider_visibility("ElevenLabs", "Instrumental + Vocal")
    assert cloud["elevenlabs_settings"] and not cloud["tts_engine_radio"]
    assert not cloud["speed_slider"] and not cloud["f5_settings"]
    local = ui.provider_visibility("Local Models", "Instrumental + Vocal")
    assert local["tts_engine_radio"] and not local["elevenlabs_settings"]
    assert local["speed_slider"]
    inst = ui.provider_visibility("ElevenLabs", "Instrumental Only")
    assert not inst["elevenlabs_settings"]


def test_voice_choices_mark_missing():
    choices = dict((v, label) for label, v in ui.voice_choices({"pMsXgVXv3BLzUgSXRplE": False}))
    assert choices["pMsXgVXv3BLzUgSXRplE"].endswith("(not in account)")
    assert not choices["pjcYQlDFKMbcOUp6F5GD"].endswith("(not in account)")


def test_validate_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVEN_LABS_API_KEY", raising=False)
    assert ui.validate_account_voices() is None
    assert "ELEVENLABS_API_KEY" in ui.api_key_banner()


def test_cost_line_counts_and_cache():
    from core.elevenlabs_tts import chunk_cache
    from core.elevenlabs_tts.engine import resolve_params
    from core.elevenlabs_tts.preprocessor import prepare_segments
    line = ui.cost_line("Rest here.\n\nLet go.", "meditation", None, "v", 0.5, 0.75)
    assert "2 chunks" in line and "0/2 cached" in line
    p = resolve_params(voice_id="v", stability=0.5, similarity=0.75)
    first = prepare_segments("Rest here.\n\nLet go.")[0]["text"]
    chunk_cache.store(chunk_cache.chunk_payload(first, "v", p.model_id, 0.5, 0.75, p.seed), b"\0\0", None)
    assert "1/2 cached" in ui.cost_line("Rest here.\n\nLet go.", "meditation", None, "v", 0.5, 0.75)


def test_cost_line_reports_bad_direction():
    assert "voice quality" in ui.cost_line("Hi.", "meditation", "[sighs]", "v", 0.5, 0.75)


def test_remix_ui_without_run():
    path, msg = ui.remix_ui(None, 1.0, 0.0)
    assert path is None and "Generate" in msg


def test_remix_ui_session_without_output_path_returns_message(tmp_path, monkeypatch):
    import numpy as np
    from core import mix_session as ms
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path / "s"))
    sr = 48000
    n = sr * 2
    ms.save_session("noout", np.full(n, 0.1, np.float32), np.ones(n, bool),
                    np.full(n, 0.05, np.float32), {
        "mix_sr": sr, "duck_amount_db": -16.0, "music_volume_db": -16.0,
        "fade_in_sec": 0.2, "fade_out_sec": 0.2, "stereo_output": False,
        "phrases": None, "bed_overrides": None, "target_lufs": -16.0, "mono_offset": -3.01,
        "output_format": "wav", "export_sr": sr, "use_upload": True,
        "content_type": "meditation", "env": {}})
    path, msg = ui.remix_ui("noout", 0.0, 0.0)
    assert path is None and "output_path" in msg


def test_remix_ui_os_error_returns_message(monkeypatch):
    import core.remix as remix_mod
    def boom(*a, **k):
        raise PermissionError("read-only output dir")
    monkeypatch.setattr(remix_mod, "remix", boom)
    path, msg = ui.remix_ui("r1", 0.0, 0.0)
    assert path is None and "read-only" in msg


def test_restore_provider_whitelists():
    assert ui.restore_provider("ElevenLabs") == "ElevenLabs"
    assert ui.restore_provider("Local Models") == "Local Models"
    for bad in (None, "", "evil<script>", 3):
        assert ui.restore_provider(bad) == ui.DEFAULT_PROVIDER


def test_generate_enabled(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVEN_LABS_API_KEY", raising=False)
    assert ui.generate_enabled("ElevenLabs", "Instrumental + Vocal") is False
    assert ui.generate_enabled("ElevenLabs", "Vocals Only") is False
    assert ui.generate_enabled("ElevenLabs", "Instrumental Only") is True
    assert ui.generate_enabled("Local Models", "Instrumental + Vocal") is True
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    assert ui.generate_enabled("ElevenLabs", "Instrumental + Vocal") is True
