import unittest
import numpy as np
from core.mixer import (
    apply_rms_ducking,
    apply_envelope_ducking,
    apply_multiband_ducking,
    overlay_tracks,
    apply_fades,
    normalize_loudness,
    resample_for_export,
)


class TestMixer(unittest.TestCase):
    def test_apply_rms_ducking(self):
        sr = 24000
        voice = np.random.uniform(-0.1, 0.1, sr).astype(np.float32)
        music = np.ones(sr, dtype=np.float32)
        out = apply_rms_ducking(voice, music, sample_rate=sr)
        self.assertEqual(out.shape, music.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_apply_envelope_ducking_basic(self):
        sr = 24000
        voice = np.random.uniform(-0.1, 0.1, sr).astype(np.float32)
        music = np.ones(sr, dtype=np.float32)
        out = apply_envelope_ducking(voice, music, sample_rate=sr)
        self.assertEqual(out.shape, music.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_envelope_ducking_hold_bridges_gaps(self):
        """Hold time should keep music ducked across short silence gaps."""
        sr = 48000
        duration_sec = 4.0
        n = int(sr * duration_sec)
        voice = np.zeros(n, dtype=np.float32)
        music = np.ones(n, dtype=np.float32) * 0.5

        # Two spoken phrases with a 0.5s gap (shorter than 1.2s hold)
        phrase1_start, phrase1_end = int(0.5 * sr), int(1.5 * sr)
        phrase2_start, phrase2_end = int(2.0 * sr), int(3.0 * sr)
        voice[phrase1_start:phrase1_end] = 0.3
        voice[phrase2_start:phrase2_end] = 0.3

        # With hold=0 (no hold), the gap should recover
        out_no_hold = apply_envelope_ducking(
            voice, music, sample_rate=sr, duck_amount_db=-12.0,
            hold_ms=0.0, attack_ms=20.0, release_ms=200.0,
        )
        # With hold=1200ms, the gap should stay ducked
        out_held = apply_envelope_ducking(
            voice, music, sample_rate=sr, duck_amount_db=-12.0,
            hold_ms=1200.0, attack_ms=20.0, release_ms=200.0,
        )

        # Measure music level in the gap between phrases
        gap_mid = int(1.75 * sr)
        gap_slice = slice(gap_mid - 500, gap_mid + 500)
        gap_no_hold = np.mean(np.abs(out_no_hold[gap_slice]))
        gap_held = np.mean(np.abs(out_held[gap_slice]))

        # With hold, music in the gap should be lower (still ducked)
        self.assertLess(gap_held, gap_no_hold,
                        "Hold time should keep music ducked across short gaps")

    def test_multiband_ducking_preserves_bass(self):
        """Multiband ducking should attenuate mid more than low frequencies."""
        sr = 48000
        n = sr * 2
        voice = np.zeros(n, dtype=np.float32)
        voice[sr // 2: sr] = 0.3  # speech in middle

        # Music: mix of 100Hz sine (bass) and 1kHz sine (mid)
        t = np.arange(n, dtype=np.float32) / sr
        bass = np.sin(2 * np.pi * 100 * t).astype(np.float32) * 0.3
        mid = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.3
        music = bass + mid

        out = apply_multiband_ducking(
            voice, music, sample_rate=sr, duck_amount_db=-12.0,
            low_duck_ratio=0.25, high_duck_ratio=0.5,
        )

        # During speech, bass should be preserved more than mid
        speech_slice = slice(int(0.6 * sr), int(0.9 * sr))

        # Fullband ducking for comparison
        out_full = apply_envelope_ducking(
            voice, music, sample_rate=sr, duck_amount_db=-12.0,
        )

        # The multiband output should have higher overall energy during speech
        # (because bass is preserved) compared to fullband
        mb_rms = np.sqrt(np.mean(out[speech_slice] ** 2))
        fb_rms = np.sqrt(np.mean(out_full[speech_slice] ** 2))
        self.assertGreater(mb_rms, fb_rms,
                           "Multiband should preserve more energy (bass) than fullband")

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
