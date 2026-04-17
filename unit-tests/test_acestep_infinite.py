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

    @patch("core.acestep_engine.AceStepEngine._generate_single_repaint")
    @patch("core.acestep_engine.AceStepEngine._generate_single_raw")
    @patch("soundfile.write")
    def test_generate_infinite_reference_audio_propagation(
        self, mock_sf_write, mock_raw, mock_repaint,
    ):
        """Verify reference_audio_path is propagated to genesis and repaint continuation."""
        # Genesis returns 90s of stereo audio (GENESIS_LEN = 90.0)
        genesis_samples = int(90 * NATIVE_SAMPLE_RATE)
        mock_raw.return_value = (torch.ones(2, genesis_samples), NATIVE_SAMPLE_RATE)

        # Repaint continuation returns 80s of mono audio (20s overlap + 60s new)
        repaint_samples = int(80 * NATIVE_SAMPLE_RATE)
        mock_repaint.return_value = np.zeros(repaint_samples, dtype=np.float32)

        ref_path = "/tmp/test_ref.wav"
        with patch("tempfile.NamedTemporaryFile") as mock_tmpfile, \
             patch("os.unlink"):
            mock_tmpfile.return_value.__enter__ = lambda s: s
            mock_tmpfile.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmpfile.return_value.name = "/tmp/fake_overlap.wav"
            with patch.object(AceStepEngine, "_prepare_reference_audio", return_value=ref_path):
                self.engine.generate("test prompt", 120.0, melody_audio=np.zeros(100), melody_sample_rate=24000)

        # Check genesis received reference_audio_path
        self.assertEqual(mock_raw.call_args[1]["reference_audio_path"], ref_path)
        # Check repaint continuation received reference_audio_path
        self.assertEqual(mock_repaint.call_args[1]["reference_audio_path"], ref_path)

    @patch("core.acestep_engine.AceStepEngine._generate_single_repaint")
    @patch("core.acestep_engine.AceStepEngine._generate_single_raw")
    @patch("soundfile.write")
    def test_generate_infinite_retry_logic(
        self, mock_sf_write, mock_raw, mock_repaint,
    ):
        """Verify that _generate_infinite stops early when repaint returns None."""
        # Genesis returns 90s of stereo audio (GENESIS_LEN = 90.0)
        genesis_samples = int(90 * NATIVE_SAMPLE_RATE)
        mock_raw.return_value = (torch.ones(2, genesis_samples), NATIVE_SAMPLE_RATE)

        # Repaint raises an exception on first call → continuation_audio stays None → early stop
        mock_repaint.side_effect = RuntimeError("Repaint failed")

        with patch("tempfile.NamedTemporaryFile") as mock_tmpfile, \
             patch("os.unlink"):
            mock_tmpfile.return_value.__enter__ = lambda s: s
            mock_tmpfile.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmpfile.return_value.name = "/tmp/fake_overlap.wav"
            result = self.engine.generate("test prompt", 150.0)

        # Should return the genesis audio (90s) without crashing
        self.assertIsNotNone(result)
        # Repaint was called once (then stopped early)
        self.assertEqual(mock_repaint.call_count, 1)

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
