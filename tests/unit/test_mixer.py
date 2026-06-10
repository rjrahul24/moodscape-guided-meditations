import unittest
import numpy as np
from core.mixer import (
    overlay_tracks,
    apply_fades,
    normalize_loudness,
    resample_for_export,
)


class TestMixer(unittest.TestCase):
    def test_overlay_tracks(self):
        sr = 24000
        voice = np.zeros(sr * 2, dtype=np.float32)  # 2 seconds
        music = np.ones(sr * 1, dtype=np.float32)  # 1 second
        aligned_v, aligned_m = overlay_tracks(
            voice, music, music_pre_roll_sec=2.0, sample_rate=sr,
        )
        # Total: 2s pre_roll + 2s voice + 8s default post_roll = 12s
        expected = sr * 12
        self.assertEqual(aligned_v.shape[0], expected)
        self.assertEqual(aligned_v.shape, aligned_m.shape)

    def test_apply_fades_exponential(self):
        """Exponential fade (default) should start at 0 and end at 0."""
        sr = 24000
        audio = np.ones(sr * 4, dtype=np.float32)
        out = apply_fades(audio, sr, fade_in_sec=1.0, fade_out_sec=1.0)
        self.assertEqual(out.shape, audio.shape)
        self.assertAlmostEqual(out[0], 0.0, places=5)
        self.assertAlmostEqual(out[-1], 0.0, places=5)
        # Mid-point should be close to 1.0 (untouched)
        self.assertAlmostEqual(out[sr * 2], 1.0, places=5)

    def test_apply_fades_linear(self):
        """Linear fade should still work when explicitly requested."""
        sr = 24000
        audio = np.ones(sr * 2, dtype=np.float32)
        out = apply_fades(audio, sr, fade_in_sec=1.0, fade_out_sec=1.0, curve="linear")
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[-1], 0.0)

    def test_apply_fades_exponential_curve_shape(self):
        """Exponential fade-in should be concave-up (slower start than linear)."""
        sr = 48000
        audio = np.ones(sr * 2, dtype=np.float32)
        out_exp = apply_fades(audio, sr, fade_in_sec=1.0, fade_out_sec=0.0, curve="exponential")
        out_lin = apply_fades(audio, sr, fade_in_sec=1.0, fade_out_sec=0.0, curve="linear")

        # At 25% through the fade, exponential should be lower than linear
        quarter = sr // 4
        self.assertLess(out_exp[quarter], out_lin[quarter],
                        "Exponential fade should start slower than linear")

    def test_normalize_loudness(self):
        audio = np.ones(24000, dtype=np.float32)
        out = normalize_loudness(audio, target_lufs=-14.0)
        self.assertEqual(out.shape, audio.shape)

    def test_resample_for_export(self):
        audio = np.zeros(24000, dtype=np.float32)
        out = resample_for_export(audio, source_rate=24000, target_rate=44100)
        self.assertEqual(out.shape[0], 44100)


if __name__ == "__main__":
    unittest.main()
