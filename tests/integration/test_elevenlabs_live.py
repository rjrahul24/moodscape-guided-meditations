"""Live ElevenLabs render (bills ~60 characters). Skipped without a key.

Run with the key exported, e.g. ``set -a; source .env; set +a`` first.
"""

import os

import pytest
import soundfile as sf

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_LABS_API_KEY")),
    reason="ELEVENLABS_API_KEY / ELEVEN_LABS_API_KEY not set",
)


def test_one_paragraph_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts"))
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path / "sessions"))
    from core.pipeline import MeditationPipeline

    pipe = MeditationPipeline()
    bg = "assets/backgrounds/freemusicforvideo-meditation-background-409198.mp3"
    out, _ = pipe.generate(
        script="Let your shoulders soften. [pause:2s]\n\nRest here.",
        music_prompt="", tts_engine="elevenlabs", music_model="upload",
        uploaded_music_path=bg,
    )
    audio, sr = sf.read(out)
    assert sr == 48000 and len(audio) > sr * 5
    m = pipe.last_tts_manifest
    assert m["model_id"] == "eleven_v4"
    assert m["chars_billed"] + m["chars_cached"] > 0
    assert m["request_ids"], "request-id header not captured"
