import unittest
import numpy as np
from core.qa_monitor import check_silence_gaps, check_clipping, run_qa_checks


class TestQAMonitor(unittest.TestCase):
    def test_check_silence_gaps(self):
        sr = 24000
        # 2 seconds of silence
        audio = np.zeros(2 * sr, dtype=np.float32)

        # Max silence is 15s. Should be empty
        issues = check_silence_gaps(audio, sr, max_silence_sec=15.0)
        self.assertEqual(len(issues), 0)

        # If max silence is 1.0s, should detect
        issues = check_silence_gaps(audio, sr, max_silence_sec=1.0)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["type"], "long_silence")
        self.assertAlmostEqual(issues[0]["duration_sec"], 2.0, delta=0.1)

    def test_check_clipping(self):
        # Normal audio
        audio_ok = np.array([0.1, -0.5, 0.8, -0.8], dtype=np.float32)
        res_ok = check_clipping(audio_ok, threshold=0.99)
        self.assertTrue(res_ok["passed"])
        self.assertEqual(res_ok["clipped_samples"], 0)

        # Clipped audio
        audio_clip = np.array([0.1, 1.0, -1.0, 0.8], dtype=np.float32)
        res_clip = check_clipping(audio_clip, threshold=0.99)
        self.assertFalse(res_clip["passed"])
        self.assertEqual(res_clip["clipped_samples"], 2)

    def test_run_qa_checks_integration(self):
        # A simple pulse that isn't clipped and isn't silent
        sr = 24000
        audio = np.random.uniform(-0.5, 0.5, sr).astype(np.float32)
        # Using a higher threshold to avoid false positive for silence
        results = run_qa_checks(audio, sr, log_results=False)
        self.assertTrue(results["clipping"]["passed"])
        self.assertEqual(len(results["silence"]), 0)
        self.assertIn("lufs", results)

    # ── New tests for QA warning fixes ────────────────────────────────────────

    def test_silence_ratio_between_60_and_70_now_passes(self):
        """Silence ratios in 60-70% range should now pass (long-pause meditations)."""
        from core.qa_monitor import check_silence_ratio

        sr = 24000
        n_total = 10 * sr  # 10 seconds
        # Build audio that is ~68% silent: 32% active windows, 68% zero
        audio = np.zeros(n_total, dtype=np.float32)
        window = int(0.05 * sr)  # 50 ms — matches check_silence_ratio internals
        n_windows = n_total // window
        active_windows = int(n_windows * 0.32)
        for i in range(active_windows):
            start = i * window
            audio[start:start + window] = 0.5  # above silence threshold

        result = check_silence_ratio(audio, sr)
        self.assertGreater(result["silence_ratio"], 0.60,
                           "Test setup error: ratio not in 60-70% range")
        self.assertLessEqual(result["silence_ratio"], 0.70,
                             "Test setup error: ratio unexpectedly above 70%")
        self.assertTrue(result["passed"],
                        f"Expected passed=True for silence_ratio={result['silence_ratio']:.3f} "
                        f"with 70% upper bound")

    def test_silence_ratio_still_fails_above_70_percent(self):
        """Silence ratios above 70% should still fail (over-padded audio)."""
        from core.qa_monitor import check_silence_ratio

        sr = 24000
        n_total = 10 * sr
        # ~90% silent: only 10% of windows have signal
        audio = np.zeros(n_total, dtype=np.float32)
        window = int(0.05 * sr)
        n_windows = n_total // window
        active_windows = int(n_windows * 0.10)
        for i in range(active_windows):
            start = i * window
            audio[start:start + window] = 0.5

        result = check_silence_ratio(audio, sr)
        self.assertFalse(result["passed"],
                         f"Expected passed=False for silence_ratio={result['silence_ratio']:.3f}")

    def test_vad_uses_16khz_resampled_audio(self):
        """_apply_silero_vad must call get_speech_timestamps with sampling_rate=16000."""
        from unittest.mock import MagicMock, patch

        sr = 24000
        audio = np.random.uniform(-0.1, 0.1, sr * 2).astype(np.float32)

        captured = {}

        def fake_get_speech_timestamps(audio_tensor, model, sampling_rate):
            captured["sampling_rate"] = sampling_rate
            n = audio_tensor.shape[-1]
            return [{"start": 0, "end": n}]

        mock_model = MagicMock()
        mock_utils = (fake_get_speech_timestamps, None, None, None, None)

        with patch("torch.hub.load", return_value=(mock_model, mock_utils)):
            from core.f5_tts.engine import _apply_silero_vad
            _apply_silero_vad(audio, sr)

        self.assertEqual(
            captured.get("sampling_rate"), 16000,
            f"Silero VAD must receive sampling_rate=16000 (got {captured.get('sampling_rate')}). "
            "24 kHz is unsupported and triggers the fallback warning."
        )


if __name__ == "__main__":
    unittest.main()
