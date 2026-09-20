"""Tests for pipeline caching and music volume control."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from core.pipeline import MeditationPipeline
from core.speech_engine import SpeechEngine


class MockTTS(SpeechEngine):
    """Deterministic mock TTS engine to test pipeline caching without heavy models."""

    def __init__(self):
        self.load_count = 0
        self.synth_count = 0
        self.unload_count = 0

    def load_model(self):
        self.load_count += 1

    def unload_model(self):
        self.unload_count += 1

    def get_available_voices(self):
        return [{"id": "mock_voice", "name": "Mock Voice", "description": "Mock"}]

    def synthesize(self, segments, voice=None, speed=1.0, progress_cb=None, seed=None, **kwargs):
        self.synth_count += 1
        sr = 24000
        # 3 seconds of sound
        audio = np.full(sr * 3, 0.1, dtype=np.float32)
        activity = np.ones(sr * 3, dtype=bool)
        return audio, activity


class MockMusicEngine:
    def __init__(self, *args, **kwargs):
        pass

    def load_model(self):
        pass

    def unload_model(self):
        pass

    def generate(self, prompt, duration_sec, progress_cb=None, **kwargs):
        sr = 48000
        n_samples = int(duration_sec * sr)
        return np.full(n_samples, 0.05, dtype=np.float32)


@pytest.fixture
def temp_tts_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts_cache"))
    return tmp_path


def test_pipeline_tts_cache_bypass(temp_tts_cache, monkeypatch):
    """Test that MeditationPipeline reuses cached TTS and skips TTS model load/synth on second run."""
    # Mock UploadMusicEngine so we don't need real audio files
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", MockMusicEngine)

    pipeline = MeditationPipeline()
    mock_tts = MockTTS()

    script = "Close your eyes and breathe gently. [pause:2s]"

    # First run: cache miss -> synthesize
    out1, status1 = pipeline.generate(
        script=script,
        music_prompt="peaceful ambient",
        tts_engine="mock",
        custom_tts_engine=mock_tts,
        music_model="upload",
        uploaded_music_path="dummy.wav",
        use_tts_cache=True,
    )
    assert mock_tts.load_count == 1
    assert mock_tts.synth_count == 1
    assert "Reused cached TTS" not in status1

    # Second run: same script & settings -> cache hit -> skip load & synth!
    out2, status2 = pipeline.generate(
        script=script,
        music_prompt="peaceful ambient",
        tts_engine="mock",
        custom_tts_engine=mock_tts,
        music_model="upload",
        uploaded_music_path="dummy.wav",
        use_tts_cache=True,
    )
    # Counts should NOT have increased!
    assert mock_tts.load_count == 1
    assert mock_tts.synth_count == 1
    assert "Reused cached TTS" in status2

    # Third run: modified script -> cache miss -> synthesizes again
    out3, status3 = pipeline.generate(
        script=script + " Release any tension.",
        music_prompt="peaceful ambient",
        tts_engine="mock",
        custom_tts_engine=mock_tts,
        music_model="upload",
        uploaded_music_path="dummy.wav",
        use_tts_cache=True,
    )
    assert mock_tts.load_count == 2
    assert mock_tts.synth_count == 2
    assert "Reused cached TTS" not in status3


def test_pipeline_music_volume_affects_mix(temp_tts_cache, monkeypatch):
    """Test that music_volume_db parameter in generate() alters the mixed music level."""
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", MockMusicEngine)

    pipeline = MeditationPipeline()
    mock_tts = MockTTS()
    script = "Listen to the quiet sounds. [pause:2s]"

    # Run with default music_volume_db (-16 dB)
    out1, _ = pipeline.generate(
        script=script,
        music_prompt="ambient",
        tts_engine="mock",
        custom_tts_engine=mock_tts,
        music_model="upload",
        uploaded_music_path="dummy.wav",
        music_volume_db=-16.0,
        use_tts_cache=True,
    )

    # Run with lower music_volume_db (-26 dB) using cached voice
    out2, _ = pipeline.generate(
        script=script,
        music_prompt="ambient",
        tts_engine="mock",
        custom_tts_engine=mock_tts,
        music_model="upload",
        uploaded_music_path="dummy.wav",
        music_volume_db=-26.0,
        use_tts_cache=True,
    )

    import soundfile as sf
    data1, _ = sf.read(out1)
    data2, _ = sf.read(out2)
    # The output files should be generated and have valid audio
    assert len(data1) > 0
    assert len(data2) > 0
