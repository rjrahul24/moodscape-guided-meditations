import unittest
import numpy as np
from core.mixer import (
    adaptive_vad_threshold,
    calibrate_music_bed,
    detect_phrases,
    mix,
    overlay_tracks,
    apply_fades,
    normalize_loudness,
    resample_for_export,
)


def _tone(freq_hz: float, duration_s: float, sr: int, rms_db: float) -> np.ndarray:
    """1 kHz-style sine at an exact RMS level (dBFS). At ~1 kHz, K-weighted
    LUFS of a steady sine reads ≈ its RMS dBFS, so these make analytic
    loudness fixtures."""
    t = np.arange(int(duration_s * sr), dtype=np.float32) / sr
    amplitude = (10.0 ** (rms_db / 20.0)) * np.sqrt(2.0)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _gated_voice(sr: int, rms_db: float = -21.0, n_phrases: int = 3,
                 on_s: float = 6.0, off_s: float = 4.0) -> np.ndarray:
    """Synthetic narration: n_phrases tone bursts separated by silence."""
    parts = []
    for _ in range(n_phrases):
        parts.append(_tone(997.0, on_s, sr, rms_db))
        parts.append(np.zeros(int(off_s * sr), dtype=np.float32))
    return np.concatenate(parts)


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


class TestAdaptiveVadThreshold(unittest.TestCase):
    def test_threshold_tracks_speech_level(self):
        """Threshold should land ~22 dB below the speech envelope level."""
        sr = 48000
        voice = _gated_voice(sr, rms_db=-21.0)
        thr = adaptive_vad_threshold(voice, sr)
        # Envelope p95 of a -21 dBFS RMS tone is ~-21 dB → threshold ~-43,
        # inside the clamp range.
        self.assertGreaterEqual(thr, -55.0)
        self.assertLessEqual(thr, -35.0)
        self.assertAlmostEqual(thr, -43.0, delta=3.0)

    def test_threshold_shifts_with_voice_level(self):
        """A quieter voice should get a proportionally lower threshold."""
        sr = 48000
        thr_nominal = adaptive_vad_threshold(_gated_voice(sr, rms_db=-21.0), sr)
        thr_quiet = adaptive_vad_threshold(_gated_voice(sr, rms_db=-30.0), sr)
        self.assertAlmostEqual(thr_nominal - thr_quiet, 9.0, delta=2.0)

    def test_degenerate_short_audio_falls_back(self):
        sr = 48000
        short = _tone(997.0, 1.0, sr, -21.0)
        self.assertEqual(adaptive_vad_threshold(short, sr), -40.0)

    def test_degenerate_flat_audio_falls_back(self):
        """Constant-level audio (speech/floor spread < 12 dB) → fallback."""
        sr = 48000
        flat = _tone(997.0, 10.0, sr, -30.0)
        self.assertEqual(adaptive_vad_threshold(flat, sr), -40.0)


class TestDetectPhrasesAdaptive(unittest.TestCase):
    def test_adaptive_matches_fixed_on_nominal_voice(self):
        """On a voice at the pipeline's nominal level, threshold_db=None must
        reproduce the legacy fixed -40 dB phrase boundaries."""
        sr = 48000
        voice = _gated_voice(sr, rms_db=-21.0, n_phrases=3)
        fixed = detect_phrases(voice, sr, threshold_db=-40.0)
        adaptive = detect_phrases(voice, sr, threshold_db=None)
        self.assertEqual(len(fixed), len(adaptive))
        for (fs, fe), (as_, ae) in zip(fixed, adaptive):
            self.assertAlmostEqual(fs, as_, delta=0.05)
            self.assertAlmostEqual(fe, ae, delta=0.05)


class TestCalibrateMusicBed(unittest.TestCase):
    SR = 48000

    def test_invariance_at_golden_levels(self):
        """Stems at today's nominal pre-mix levels must calibrate back to the
        legacy constants (within rounding) — the golden path must not move."""
        voice = _gated_voice(self.SR, rms_db=-21.2)
        music = _tone(997.0, len(voice) / self.SR, self.SR, rms_db=-19.7)
        vol_db, duck_db = calibrate_music_bed(voice, music, self.SR)
        self.assertAlmostEqual(vol_db, -16.0, delta=0.75)
        self.assertAlmostEqual(duck_db, -16.0, delta=0.01)

    def test_hot_music_gets_attenuated_more(self):
        """An abnormally hot upload should receive a lower bed gain."""
        voice = _gated_voice(self.SR, rms_db=-21.2)
        music = _tone(997.0, len(voice) / self.SR, self.SR, rms_db=-10.0)
        vol_db, _ = calibrate_music_bed(voice, music, self.SR)
        self.assertLess(vol_db, -22.0)

    def test_quiet_music_gets_boost_within_clamp(self):
        """A whisper-quiet upload should get more gain, clamped at -8 dB."""
        voice = _gated_voice(self.SR, rms_db=-21.2)
        music = _tone(997.0, len(voice) / self.SR, self.SR, rms_db=-45.0)
        vol_db, _ = calibrate_music_bed(voice, music, self.SR)
        self.assertEqual(vol_db, -8.0)

    def test_analytic_offset(self):
        """music_volume_db must equal (V - pause_offset) - M within tolerance."""
        voice = _gated_voice(self.SR, rms_db=-21.2)
        music = _tone(997.0, len(voice) / self.SR, self.SR, rms_db=-22.0)
        vol_db, duck_db = calibrate_music_bed(voice, music, self.SR)
        # V ≈ -21.2, M ≈ -22.0 → (V - 14.5) - M = -13.7
        self.assertAlmostEqual(vol_db, -13.7, delta=0.75)
        self.assertAlmostEqual(duck_db, -(30.5 - 14.5), delta=0.01)

    def test_degenerate_silent_voice_returns_legacy(self):
        voice = np.zeros(self.SR * 20, dtype=np.float32)
        music = _tone(997.0, 20.0, self.SR, rms_db=-20.0)
        self.assertEqual(calibrate_music_bed(voice, music, self.SR), (-16.0, -16.0))

    def test_degenerate_short_input_returns_legacy(self):
        voice = _tone(997.0, 2.0, self.SR, -21.0)
        music = _tone(997.0, 2.0, self.SR, -20.0)
        self.assertEqual(calibrate_music_bed(voice, music, self.SR), (-16.0, -16.0))


class TestMixPhrasesPassthrough(unittest.TestCase):
    def test_mix_accepts_precomputed_phrases(self):
        """mix() with explicit phrases must match a phrase-free mix when the
        phrases equal what VAD would detect (pre-roll shift correctness)."""
        sr = 48000
        voice = _gated_voice(sr, rms_db=-21.0, n_phrases=2, on_s=4.0, off_s=3.0)
        music = _tone(440.0, len(voice) / sr + 30.0, sr, rms_db=-20.0)
        activity = np.abs(voice) > 0

        phrases = detect_phrases(voice, sr, threshold_db=-40.0)
        self.assertTrue(phrases)

        mixed_auto = mix(voice, activity, music, sample_rate=sr,
                         fade_in_sec=0.0, fade_out_sec=0.0)
        mixed_pre = mix(voice, activity, music, sample_rate=sr,
                        fade_in_sec=0.0, fade_out_sec=0.0, phrases=phrases)
        self.assertEqual(mixed_auto.shape, mixed_pre.shape)
        # The ducked beds should be near-identical (same phrase timeline).
        diff = np.max(np.abs(mixed_auto - mixed_pre))
        self.assertLess(diff, 0.02)


if __name__ == "__main__":
    unittest.main()
