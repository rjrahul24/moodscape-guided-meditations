"""Unit tests for core/tts_cache.py."""

import numpy as np
import pytest
from pathlib import Path

from core.tts_cache import (
    compute_cache_key,
    has_cache,
    get_cached_tts,
    save_cached_tts,
    get_latest_tts,
    clear_cache,
)


@pytest.fixture
def temp_cache_dir(tmp_path, monkeypatch):
    """Point MOODSCAPE_TTS_CACHE_DIR to a temporary directory."""
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path))
    return tmp_path


class TestComputeCacheKey:
    def test_deterministic_and_whitespace_normalized(self):
        k1 = compute_cache_key(
            script="  Take a deep breath. [pause:3s]\n\nRelax.  ",
            content_type="meditation",
            tts_engine="f5",
            voice="delilah",
            speed=0.8,
        )
        k2 = compute_cache_key(
            script="Take a deep breath. [pause:3s]\n\nRelax.",
            content_type="meditation",
            tts_engine="f5",
            voice="delilah",
            speed=0.8,
        )
        assert k1 == k2

    def test_sensitive_to_script_changes(self):
        k1 = compute_cache_key(script="Take a deep breath.", tts_engine="f5", voice="delilah")
        k2 = compute_cache_key(script="Take another breath.", tts_engine="f5", voice="delilah")
        assert k1 != k2

    def test_sensitive_to_engine_and_voice(self):
        k1 = compute_cache_key(script="Relax.", tts_engine="f5", voice="delilah")
        k2 = compute_cache_key(script="Relax.", tts_engine="kokoro", voice="balanced_calm")
        k3 = compute_cache_key(script="Relax.", tts_engine="f5", voice="brittney")
        assert k1 != k2
        assert k1 != k3

    def test_sensitive_to_speed_and_pacing(self):
        k1 = compute_cache_key(script="Relax.", tts_engine="f5", voice="delilah", speed=0.80)
        k2 = compute_cache_key(script="Relax.", tts_engine="f5", voice="delilah", speed=0.85)
        k3 = compute_cache_key(script="Relax.", tts_engine="f5", voice="delilah", speed=0.80, f5_target_wpm=100)
        assert k1 != k2
        assert k1 != k3

    def test_auto_seed_maps_to_same_key(self):
        # When seed is None (auto-seed), different random seeds should not alter cache key
        k1 = compute_cache_key(script="Relax.", tts_engine="f5", voice="delilah", seed=None)
        k2 = compute_cache_key(script="Relax.", tts_engine="f5", voice="delilah", seed=None)
        assert k1 == k2

    def test_explicit_seeds_alter_key(self):
        k1 = compute_cache_key(script="Relax.", tts_engine="f5", voice="delilah", seed=42)
        k2 = compute_cache_key(script="Relax.", tts_engine="f5", voice="delilah", seed=99)
        assert k1 != k2


class TestCacheStorage:
    def test_roundtrip_save_and_get(self, temp_cache_dir):
        key = "test_key_12345"
        assert not has_cache(key)
        assert get_cached_tts(key) is None

        sr = 24000
        voice_audio = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        voice_activity = np.array([True, True, True, False], dtype=bool)
        meta = {"engine": "f5", "voice": "delilah"}

        saved_key = save_cached_tts(key, voice_audio, voice_activity, sample_rate=sr, metadata=meta)
        assert saved_key == key
        assert has_cache(key)

        loaded = get_cached_tts(key)
        assert loaded is not None
        np.testing.assert_allclose(loaded["voice_audio"], voice_audio, atol=1e-6)
        np.testing.assert_array_equal(loaded["voice_activity"], voice_activity)
        assert loaded["sample_rate"] == sr
        assert loaded["metadata"]["engine"] == "f5"

    def test_get_latest_tts(self, temp_cache_dir):
        key1 = "first_key"
        key2 = "second_key"

        audio1 = np.array([1.0], dtype=np.float32)
        audio2 = np.array([2.0], dtype=np.float32)
        act = np.array([True], dtype=bool)

        save_cached_tts(key1, audio1, act)
        latest1 = get_latest_tts()
        assert latest1 is not None
        np.testing.assert_allclose(latest1["voice_audio"], audio1)

        save_cached_tts(key2, audio2, act)
        latest2 = get_latest_tts()
        assert latest2 is not None
        np.testing.assert_allclose(latest2["voice_audio"], audio2)

    def test_clear_cache(self, temp_cache_dir):
        key1 = "k1"
        key2 = "k2"
        audio = np.array([0.5], dtype=np.float32)
        act = np.array([True], dtype=bool)

        save_cached_tts(key1, audio, act)
        save_cached_tts(key2, audio, act)

        assert has_cache(key1)
        assert has_cache(key2)

        # Clear specific
        deleted = clear_cache(key1)
        assert deleted == 2  # npz + json
        assert not has_cache(key1)
        assert has_cache(key2)

        # Clear all
        clear_cache()
        assert not has_cache(key2)
        assert get_latest_tts() is None
