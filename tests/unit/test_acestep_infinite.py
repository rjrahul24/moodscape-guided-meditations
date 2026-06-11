import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from core.acestep.engine import AceStepEngine, TARGET_SAMPLE_RATE, NATIVE_SAMPLE_RATE


class TestAceStepInfinite(unittest.TestCase):
    def setUp(self):
        self.engine = AceStepEngine()
        self.engine.initialized = True
        self.engine.model_type = "sft"
        self.engine._dit = MagicMock()
        self.engine._llm = MagicMock()
        self.engine._dit.initialize_service.return_value = ("Success", True)

    @patch("core.acestep.engine.AceStepEngine._generate_single_repaint")
    @patch("core.acestep.engine.AceStepEngine._generate_single_raw")
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

    @patch("core.acestep.engine.AceStepEngine._generate_single_repaint")
    @patch("core.acestep.engine.AceStepEngine._generate_single_raw")
    @patch("soundfile.write")
    def test_generate_infinite_retry_logic(
        self, mock_sf_write, mock_raw, mock_repaint,
    ):
        """Verify that _generate_infinite stops early after both repaint attempts fail."""
        # Genesis returns 90s of stereo audio (GENESIS_LEN = 90.0)
        genesis_samples = int(90 * NATIVE_SAMPLE_RATE)
        mock_raw.return_value = (torch.ones(2, genesis_samples), NATIVE_SAMPLE_RATE)

        # Repaint raises every time → both attempts fail → early stop
        mock_repaint.side_effect = RuntimeError("Repaint failed")

        with patch("tempfile.NamedTemporaryFile") as mock_tmpfile, \
             patch("os.unlink"):
            mock_tmpfile.return_value.__enter__ = lambda s: s
            mock_tmpfile.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmpfile.return_value.name = "/tmp/fake_overlap.wav"
            result = self.engine.generate("test prompt", 150.0)

        # Should return the genesis audio (90s) without crashing
        self.assertIsNotNone(result)
        # Repaint was attempted twice (initial + one retry), then early stop
        self.assertEqual(mock_repaint.call_count, 2)

    @patch("core.acestep.engine.AceStepEngine._generate_single_repaint")
    @patch("core.acestep.engine.AceStepEngine._generate_single_raw")
    @patch("soundfile.write")
    def test_generate_infinite_seed_pinning(
        self, mock_sf_write, mock_raw, mock_repaint,
    ):
        """seed must reach genesis unchanged and each repaint as seed + seg_num."""
        genesis_samples = int(90 * NATIVE_SAMPLE_RATE)
        mock_raw.return_value = (torch.ones(2, genesis_samples), NATIVE_SAMPLE_RATE)
        repaint_samples = int(80 * NATIVE_SAMPLE_RATE)
        # Non-silent tail so QA passes first try (one call per segment).
        mock_repaint.return_value = (
            np.random.default_rng(0).standard_normal(repaint_samples).astype(np.float32) * 0.3
        )

        with patch("tempfile.NamedTemporaryFile") as mock_tmpfile, \
             patch("os.unlink"), \
             patch("core.qa_monitor.compute_composite_score", return_value=0.9):
            mock_tmpfile.return_value.__enter__ = lambda s: s
            mock_tmpfile.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmpfile.return_value.name = "/tmp/fake_overlap.wav"
            self.engine.generate("test prompt", 150.0, seed=1234)

        self.assertEqual(mock_raw.call_args[1]["seed"], 1234)
        # First continuation is seg_num=2 → seed + 2
        self.assertEqual(mock_repaint.call_args_list[0][1]["seed"], 1236)

    @patch("core.acestep.engine.AceStepEngine._generate_single_repaint")
    @patch("core.acestep.engine.AceStepEngine._generate_single_raw")
    @patch("soundfile.write")
    def test_generate_infinite_segment_qa_retry(
        self, mock_sf_write, mock_raw, mock_repaint,
    ):
        """A low composite score on a continuation triggers one offset-seed retry."""
        genesis_samples = int(90 * NATIVE_SAMPLE_RATE)
        mock_raw.return_value = (torch.ones(2, genesis_samples), NATIVE_SAMPLE_RATE)
        repaint_samples = int(80 * NATIVE_SAMPLE_RATE)
        mock_repaint.return_value = np.full(repaint_samples, 0.1, dtype=np.float32)

        with patch("tempfile.NamedTemporaryFile") as mock_tmpfile, \
             patch("os.unlink"), \
             patch("core.qa_monitor.compute_composite_score", return_value=0.2), \
             patch.object(AceStepEngine, "_seam_discontinuity_db", return_value=0.0):
            mock_tmpfile.return_value.__enter__ = lambda s: s
            mock_tmpfile.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmpfile.return_value.name = "/tmp/fake_overlap.wav"
            self.engine.generate("test prompt", 150.0, seed=1000)

        # One segment needed (150-90=60s), scored 0.2 < 0.6 → retried once.
        self.assertEqual(mock_repaint.call_count, 2)
        self.assertEqual(mock_repaint.call_args_list[0][1]["seed"], 1002)
        self.assertEqual(mock_repaint.call_args_list[1][1]["seed"], 2002)

    def test_seam_discontinuity_metric(self):
        """Identical material across the junction ≈ 0 dB; a timbral jump is large."""
        sr = NATIVE_SAMPLE_RATE
        rng = np.random.default_rng(7)
        t = np.arange(sr * 2, dtype=np.float32) / sr
        drone = np.sin(2 * np.pi * 110 * t).astype(np.float32) * 0.3

        same = AceStepEngine._seam_discontinuity_db(drone, drone, sr)
        self.assertLess(same, 1.0)

        bright = (rng.standard_normal(sr * 2).astype(np.float32) * 0.3)
        jump = AceStepEngine._seam_discontinuity_db(drone, bright, sr)
        self.assertGreater(jump, 6.0)


class TestAceStepLoopMode(unittest.TestCase):
    def setUp(self):
        self.engine = AceStepEngine()
        self.engine.initialized = True
        self.engine.model_type = "sft"
        self.engine._dit = MagicMock()
        self.engine._llm = MagicMock()

    def _patch_base(self, base_audio):
        return patch.object(AceStepEngine, "_generate_infinite", return_value=base_audio)

    @patch("core.qa_monitor.compute_composite_score", return_value=0.9)
    def test_loop_mode_returns_exact_target_length(self, mock_score):
        base = np.random.default_rng(1).standard_normal(
            TARGET_SAMPLE_RATE * 240).astype(np.float32) * 0.3
        with self._patch_base(base):
            out = self.engine._generate_looped("test", 600.0, seed=5)
        self.assertEqual(out.shape[-1], int(600.0 * TARGET_SAMPLE_RATE))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ndim, 1)

    @patch("core.qa_monitor.compute_composite_score", return_value=0.2)
    def test_loop_mode_low_qa_retries_base_once(self, mock_score):
        base = np.random.default_rng(2).standard_normal(
            TARGET_SAMPLE_RATE * 240).astype(np.float32) * 0.3
        with self._patch_base(base) as mock_inf:
            self.engine._generate_looped("test", 600.0, seed=5)
        self.assertEqual(mock_inf.call_count, 2)
        self.assertEqual(mock_inf.call_args_list[0][1]["seed"], 5)
        self.assertEqual(mock_inf.call_args_list[1][1]["seed"], 5 + 7919)

    @patch("core.qa_monitor.compute_composite_score", return_value=0.9)
    def test_long_form_routing(self, mock_score):
        """auto → loop above 300s, evolve at 90-300s; explicit modes win."""
        out = np.zeros(TARGET_SAMPLE_RATE * 10, dtype=np.float32)
        cases = [
            (600.0, "auto", "loop"),
            (150.0, "auto", "evolve"),
            (150.0, "loop", "loop"),
            (600.0, "evolve", "evolve"),
        ]
        for duration, mode, expected in cases:
            with patch.object(AceStepEngine, "_generate_looped", return_value=out) as m_loop, \
                 patch.object(AceStepEngine, "_generate_infinite", return_value=out) as m_inf:
                self.engine.generate("test", duration, long_form_mode=mode)
                if expected == "loop":
                    m_loop.assert_called_once()
                    m_inf.assert_not_called()
                else:
                    m_inf.assert_called_once()
                    m_loop.assert_not_called()

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


class TestInlineGenerationPatch(unittest.TestCase):
    def test_inline_thread_patch_runs_target_synchronously(self):
        """The patched Thread must run its target inline (MLX graphs cannot
        cross threads) while everything else still delegates to threading."""
        import sys
        import types

        # Build a fake generate_music_execute module so the patch can be
        # exercised without the heavy acestep package.
        fake = types.ModuleType("acestep.core.generation.handler.generate_music_execute")
        import threading as real_threading
        fake.threading = real_threading
        pkgs = {
            "acestep": types.ModuleType("acestep"),
            "acestep.core": types.ModuleType("acestep.core"),
            "acestep.core.generation": types.ModuleType("acestep.core.generation"),
            "acestep.core.generation.handler": types.ModuleType("acestep.core.generation.handler"),
        }
        saved = {k: sys.modules.get(k) for k in list(pkgs) + [fake.__name__]}
        try:
            sys.modules.update(pkgs)
            sys.modules[fake.__name__] = fake
            pkgs["acestep.core.generation.handler"].generate_music_execute = fake

            AceStepEngine._patch_inline_generation_thread()

            ran_in = {}
            t = fake.threading.Thread(target=lambda: ran_in.setdefault(
                "thread", real_threading.current_thread().name), name="service-generate", daemon=True)
            t.start()
            t.join(timeout=600)
            self.assertEqual(ran_in["thread"], real_threading.current_thread().name)
            self.assertFalse(t.is_alive())
            # Non-Thread attributes still delegate to the real module
            self.assertIs(fake.threading.Event, real_threading.Event)
            # Idempotent
            AceStepEngine._patch_inline_generation_thread()
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
