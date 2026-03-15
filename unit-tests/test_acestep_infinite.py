import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from core.acestep_engine import AceStepEngine, TARGET_SAMPLE_RATE, NATIVE_SAMPLE_RATE


class TestAceStepInfinite(unittest.TestCase):
    def setUp(self):
        self.engine = AceStepEngine()
        self.engine.initialized = True
        self.engine.model_type = "sft"
        self.engine._dit = MagicMock()
        self.engine._llm = MagicMock()
        self.engine._dit.initialize_service.return_value = ("Success", True)

    @patch("core.qa_monitor.compute_composite_score", return_value=0.9)
    @patch("core.acestep_engine.AceStepEngine._smooth_boundary")
    @patch("core.acestep_engine.AceStepEngine._generate_cover_continuation")
    @patch("core.acestep_engine.AceStepEngine._generate_single_raw")
    @patch("core.acestep_engine.AceStepEngine._validate_output")
    @patch("soundfile.write")
    def test_generate_infinite_reference_audio_propagation(
        self, mock_sf_write, mock_validate, mock_raw, mock_cover, mock_smooth, mock_score,
    ):
        """Verify reference_audio_path is propagated to all phases."""
        mock_raw.return_value = (torch.ones(2, int(60 * NATIVE_SAMPLE_RATE)), NATIVE_SAMPLE_RATE)
        mock_cover.return_value = (torch.ones(2, int(60 * NATIVE_SAMPLE_RATE)), NATIVE_SAMPLE_RATE)
        mock_validate.return_value = (True, "OK")
        mock_smooth.side_effect = lambda tensor, *a, **kw: tensor

        ref_path = "/tmp/test_ref.wav"
        with patch.object(AceStepEngine, "_prepare_reference_audio", return_value=ref_path):
            self.engine.generate("test prompt", 120.0, melody_audio=np.zeros(100), melody_sample_rate=24000)

        # Check genesis
        self.assertEqual(mock_raw.call_args[1]["reference_audio_path"], ref_path)
        # Check continuation
        self.assertEqual(mock_cover.call_args[1]["reference_audio_path"], ref_path)
        # Check smoothing
        self.assertEqual(mock_smooth.call_args[1]["reference_audio_path"], ref_path)

    @patch("core.qa_monitor.compute_composite_score", return_value=0.9)
    @patch("core.acestep_engine.AceStepEngine._smooth_boundary")
    @patch("core.acestep_engine.AceStepEngine._generate_cover_continuation")
    @patch("core.acestep_engine.AceStepEngine._generate_single_raw")
    @patch("core.acestep_engine.AceStepEngine._validate_output")
    @patch("soundfile.write")
    def test_generate_infinite_retry_logic(
        self, mock_sf_write, mock_validate, mock_raw, mock_cover, mock_smooth, mock_score,
    ):
        """Verify that _generate_infinite retries failed segments."""
        mock_raw.return_value = (torch.ones(2, int(60 * NATIVE_SAMPLE_RATE)), NATIVE_SAMPLE_RATE)
        mock_cover.return_value = (torch.ones(2, int(60 * NATIVE_SAMPLE_RATE)), NATIVE_SAMPLE_RATE)
        mock_smooth.side_effect = lambda tensor, *a, **kw: tensor

        # Genesis (60s) + 1 Continuation (52s - 2s crossfade = 50s) = 110s
        # First attempt of continuation fails, second succeeds
        mock_validate.side_effect = [(False, "Too quiet"), (True, "OK")]

        self.engine.generate("test prompt", 110.0)

        # 1 genesis call (mock_validate NOT called for genesis in this test)
        # 2 cover calls (failed attempt + successful retry)
        self.assertEqual(mock_cover.call_count, 2)

    @patch("core.qa_monitor.compute_composite_score", return_value=0.9)
    def test_generate_single_switch(self, mock_score):
        """Verify routing: <=90s -> single, >90s -> infinite."""
        with patch.object(AceStepEngine, "_generate_single") as mock_single, \
             patch.object(AceStepEngine, "_generate_infinite") as mock_infinite, \
             patch.object(AceStepEngine, "_validate_output", return_value=(True, "OK")):

            mock_single.return_value = np.zeros(100, dtype=np.float32)
            mock_infinite.return_value = np.zeros(100, dtype=np.float32)

            # Case 1: 60s -> single (via generate which calls _generate_single)
            self.engine.generate("test", 60.0)
            mock_single.assert_called_once()
            mock_infinite.assert_not_called()

            mock_single.reset_mock()
            mock_infinite.reset_mock()

            # Case 2: 120s -> infinite
            self.engine.generate("test", 120.0)
            mock_single.assert_not_called()
            mock_infinite.assert_called_once()

    def test_validate_output(self):
        """Verify output validation catches bad audio."""
        # Valid audio
        good = np.random.randn(24000 * 30).astype(np.float32) * 0.5
        valid, reason = AceStepEngine._validate_output(good, 30.0)
        self.assertTrue(valid)

        # NaN audio
        bad_nan = np.full(24000 * 30, np.nan, dtype=np.float32)
        valid, reason = AceStepEngine._validate_output(bad_nan, 30.0)
        self.assertFalse(valid)
        self.assertIn("NaN", reason)

        # Silent audio
        silent = np.zeros(24000 * 30, dtype=np.float32)
        valid, reason = AceStepEngine._validate_output(silent, 30.0)
        self.assertFalse(valid)
        self.assertIn("silent", reason.lower())

        # Too short
        short = np.random.randn(24000 * 5).astype(np.float32) * 0.5
        valid, reason = AceStepEngine._validate_output(short, 30.0)
        self.assertFalse(valid)
        self.assertIn("short", reason.lower())

    def test_enhance_prompt_mesa(self):
        """Verify MESA prompt framework produces valid caption and lyrics."""
        caption, lyrics = AceStepEngine._enhance_prompt("soft piano, warm pads", duration_hint=60.0)

        # Caption should include user prompt and meditation tags
        self.assertIn("soft piano, warm pads", caption)
        self.assertIn("no vocals", caption)
        # Should NOT contain contradictions
        self.assertNotIn("no melody", caption)
        self.assertNotIn("no chord changes", caption)

        # Lyrics should use standard section tags
        self.assertIn("[Instrumental]", lyrics)
        self.assertIn("[Intro", lyrics)
        self.assertIn("[Outro", lyrics)
        # Should NOT contain non-standard tags
        self.assertNotIn("[Drone]", lyrics)
        self.assertNotIn("[Static Pad]", lyrics)

    def test_structural_lyrics_scale_with_duration(self):
        """Verify lyrics structure scales with duration."""
        short_lyrics = AceStepEngine._build_structural_lyrics("test", 30.0)
        medium_lyrics = AceStepEngine._build_structural_lyrics("test", 180.0)
        long_lyrics = AceStepEngine._build_structural_lyrics("test", 600.0)

        # Short should have fewer sections than long
        self.assertLess(short_lyrics.count("["), medium_lyrics.count("["))
        self.assertLess(medium_lyrics.count("["), long_lyrics.count("["))

        # All should start with [Instrumental]
        self.assertTrue(short_lyrics.startswith("[Instrumental]"))
        self.assertTrue(medium_lyrics.startswith("[Instrumental]"))
        self.assertTrue(long_lyrics.startswith("[Instrumental]"))


if __name__ == "__main__":
    unittest.main()
