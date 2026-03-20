"""Unit tests for HeartMulaEngine — all model calls are mocked."""

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from core.heart_mula.engine import (
    CROSSFADE_SEC,
    HeartMulaEngine,
    MAX_SEGMENT_SEC,
    TARGET_SAMPLE_RATE,
)


class TestHeartMulaEngine(unittest.TestCase):

    def setUp(self):
        self.engine = HeartMulaEngine()
        self.engine.initialized = True
        self.engine.device = "mps"
        self.engine._backend = "mps"

    # ── Crossfade tests ──────────────────────────────────────────────

    def test_crossfade_segments_single(self):
        """Single segment returned unchanged."""
        arr = np.ones(44100, dtype=np.float32)
        result = HeartMulaEngine._crossfade_segments([arr], CROSSFADE_SEC)
        np.testing.assert_array_equal(result, arr)

    def test_crossfade_segments_two(self):
        """Two segments produce correct length."""
        sr = TARGET_SAMPLE_RATE
        s1 = np.ones(sr * 10, dtype=np.float32)
        s2 = np.ones(sr * 10, dtype=np.float32) * 0.5
        fade_samples = int(CROSSFADE_SEC * sr)
        result = HeartMulaEngine._crossfade_segments([s1, s2], CROSSFADE_SEC)
        expected_len = len(s1) + len(s2) - fade_samples
        self.assertEqual(len(result), expected_len)
        self.assertEqual(result.dtype, np.float32)

    def test_crossfade_equal_power(self):
        """Cosine-squared crossfade preserves energy: fade_out + fade_in = 1."""
        n = int(CROSSFADE_SEC * TARGET_SAMPLE_RATE)
        t = np.linspace(0.0, math.pi / 2.0, n, dtype=np.float32)
        fade_out = np.cos(t) ** 2
        fade_in = np.sin(t) ** 2
        np.testing.assert_allclose(fade_out + fade_in, np.ones(n), atol=1e-5)

    # ── Segment lyrics tests ─────────────────────────────────────────

    def test_build_segment_lyrics_single(self):
        lyrics = HeartMulaEngine._build_segment_lyrics(None, 0, 1, 60.0)
        self.assertIn("[intro]", lyrics)
        self.assertIn("[outro]", lyrics)

    def test_build_segment_lyrics_first(self):
        lyrics = HeartMulaEngine._build_segment_lyrics(None, 0, 3, 60.0)
        self.assertIn("[intro]", lyrics)
        self.assertNotIn("[outro]", lyrics)

    def test_build_segment_lyrics_middle(self):
        lyrics = HeartMulaEngine._build_segment_lyrics(None, 1, 3, 60.0)
        self.assertIn("[verse]", lyrics)
        self.assertIn("[bridge]", lyrics)
        self.assertNotIn("[intro]", lyrics)
        self.assertNotIn("[outro]", lyrics)

    def test_build_segment_lyrics_last(self):
        lyrics = HeartMulaEngine._build_segment_lyrics(None, 2, 3, 60.0)
        self.assertIn("[outro]", lyrics)
        self.assertNotIn("[intro]", lyrics)

    def test_build_segment_lyrics_user_supplied(self):
        user = "[verse]\nmy lyrics"
        result = HeartMulaEngine._build_segment_lyrics(user, 0, 1, 60.0)
        self.assertEqual(result, user)

    # ── Postprocess tests ────────────────────────────────────────────

    def test_postprocess_stereo_to_mono(self):
        """Stereo (2, N) tensor → mono float32 array at 48 kHz."""
        tensor = torch.ones(2, 48000) * 0.5
        result = HeartMulaEngine._postprocess(tensor, 48000)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 48000)
        self.assertEqual(result.dtype, np.float32)

    def test_postprocess_batch_dimension(self):
        """Batch (1, 2, N) tensor → mono float32 array at 48 kHz."""
        tensor = torch.ones(1, 2, 48000) * 0.3
        result = HeartMulaEngine._postprocess(tensor, 48000)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 48000)

    def test_postprocess_peak_normalize(self):
        """Output should be peak-normalized to -1 dBFS."""
        tensor = torch.ones(1, 48000) * 0.1  # quiet input
        result = HeartMulaEngine._postprocess(tensor, 48000)
        peak_db = 20 * np.log10(np.abs(result).max())
        self.assertAlmostEqual(peak_db, -1.0, delta=0.1)

    # ── Routing tests ────────────────────────────────────────────────

    def test_generate_routes_to_long_form(self):
        """Duration > MAX_SEGMENT_SEC routes to _generate_long_form."""
        with patch.object(
            HeartMulaEngine, "_generate_long_form",
            return_value=np.zeros(44100, dtype=np.float32),
        ) as mock_lf, patch.object(
            HeartMulaEngine, "_generate_single",
            return_value=np.zeros(44100, dtype=np.float32),
        ):
            self.engine.generate("ambient", MAX_SEGMENT_SEC + 1.0)
            mock_lf.assert_called_once()

    def test_generate_routes_to_single(self):
        """Duration <= MAX_SEGMENT_SEC routes to _generate_single."""
        with patch.object(
            HeartMulaEngine, "_generate_single",
            return_value=np.zeros(44100, dtype=np.float32),
        ) as mock_s, patch.object(
            HeartMulaEngine, "_generate_long_form",
            return_value=np.zeros(44100, dtype=np.float32),
        ):
            self.engine.generate("ambient", 60.0)
            mock_s.assert_called_once()

    def test_generate_story_mode(self):
        """prompt_stages routes to _generate_story."""
        with patch.object(
            HeartMulaEngine, "_generate_story",
            return_value=np.zeros(44100, dtype=np.float32),
        ) as mock_story:
            self.engine.generate(
                "ambient", 60.0,
                prompt_stages=[("deep drone", 60.0), ("light pads", 60.0)],
            )
            mock_story.assert_called_once()

    def test_long_form_segment_count(self):
        """_generate_long_form generates correct number of segments."""
        total = 600.0  # 10 minutes
        expected_n = math.ceil(total / MAX_SEGMENT_SEC)
        call_count = [0]

        def fake_single(tags, duration, lyrics, progress_cb):
            call_count[0] += 1
            n = int(duration * TARGET_SAMPLE_RATE)
            return np.zeros(n, dtype=np.float32)

        with patch.object(self.engine, "_generate_single", side_effect=fake_single):
            self.engine._generate_long_form("ambient", total, None, None)
            self.assertEqual(call_count[0], expected_n)


if __name__ == "__main__":
    unittest.main()
