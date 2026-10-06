# tests/unit/test_eval_mix_metrics.py
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from core.eval.mix_metrics import loudness, mix_metrics, true_peak_dbtp

SR = 48000


def _pink(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(SR * seconds))
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white), 1 / SR)
    spectrum[1:] /= np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum, n=len(white))
    return (0.1 * pink / np.max(np.abs(pink))).astype(np.float32)


class TestLoudness(unittest.TestCase):
    def test_dual_mono_is_about_3db_louder(self):
        m = loudness(_pink(10.0), SR)
        self.assertAlmostEqual(m["integrated_lufs_dual_mono"] - m["integrated_lufs"], 3.01, delta=0.05)

    def test_silent_file_yields_none_loudness(self):
        m = loudness(np.zeros(SR * 5, dtype=np.float32), SR)
        self.assertIsNone(m["integrated_lufs"])
        self.assertIsNone(m["integrated_lufs_dual_mono"])


class TestTruePeak(unittest.TestCase):
    def test_intersample_peak_exceeds_sample_peak(self):
        n = np.arange(SR)
        x = np.sin(2 * np.pi * (SR / 4) * n / SR + np.pi / 4).astype(np.float32)  # samples at ±0.707
        self.assertGreater(true_peak_dbtp(x, SR), 20 * np.log10(np.max(np.abs(x))) + 2.0)

    def test_silence_is_none(self):
        self.assertIsNone(true_peak_dbtp(np.zeros(1000, dtype=np.float32), SR))


class TestMixMetricsFile(unittest.TestCase):
    def _write(self, audio: np.ndarray) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, audio, SR)
        self.addCleanup(os.remove, path)
        return path

    def test_mono_file_metrics(self):
        m = mix_metrics(self._write(_pink(10.0)), estimated_sec=8.0)
        self.assertEqual(m["channels"], 1)
        self.assertAlmostEqual(m["actual_sec"], 10.0, places=2)
        self.assertAlmostEqual(m["duration_ratio"], 1.25, places=2)
        self.assertIsNotNone(m["lra"])

    def test_stereo_file_metrics(self):
        mono = _pink(10.0)
        m = mix_metrics(self._write(np.stack([mono, mono], axis=1)))
        self.assertEqual(m["channels"], 2)
        self.assertAlmostEqual(m["integrated_lufs"], m["integrated_lufs_dual_mono"], places=3)
        self.assertIsNone(m["duration_ratio"])


if __name__ == "__main__":
    unittest.main()
