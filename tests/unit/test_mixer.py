import os
import unittest
from unittest.mock import patch
import numpy as np
from core.mixer import (
    _slew_limit_rise,
    compute_breathing_gain_db,
    adaptive_vad_threshold,
    bed_offset_overrides,
    calibrate_music_bed,
    detect_phrases,
    export_audio,
    export_lufs_target,
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

    def test_golden_levels_sit_3db_under_legacy(self):
        """Stems at today's nominal pre-mix levels calibrate to a bed 3 dB under
        the legacy -16 dB constant (R3 listening: music read as too loud in every
        test), with the same -16 dB duck depth."""
        voice = _gated_voice(self.SR, rms_db=-21.2)
        music = _tone(997.0, len(voice) / self.SR, self.SR, rms_db=-19.7)
        vol_db, duck_db = calibrate_music_bed(voice, music, self.SR)
        self.assertAlmostEqual(vol_db, -19.0, delta=0.75)
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
        # V ≈ -21.2, M ≈ -22.0 → (V - 17.5) - M = -16.7
        self.assertAlmostEqual(vol_db, -16.7, delta=0.75)
        self.assertAlmostEqual(duck_db, -(33.5 - 17.5), delta=0.01)

    def test_degenerate_silent_voice_returns_legacy(self):
        voice = np.zeros(self.SR * 20, dtype=np.float32)
        music = _tone(997.0, 20.0, self.SR, rms_db=-20.0)
        self.assertEqual(calibrate_music_bed(voice, music, self.SR), (-19.0, -16.0))

    def test_degenerate_short_input_returns_legacy(self):
        voice = _tone(997.0, 2.0, self.SR, -21.0)
        music = _tone(997.0, 2.0, self.SR, -20.0)
        self.assertEqual(calibrate_music_bed(voice, music, self.SR), (-19.0, -16.0))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


class TestFadeIn(unittest.TestCase):
    """The intro fade must not read as multiple seconds of silence."""

    def test_fade_in_audible_within_three_quarters_second(self):
        sr = 48000
        audio = np.ones(sr * 5, dtype=np.float32)
        out = apply_fades(audio, sr, fade_in_sec=1.5, fade_out_sec=0.0)
        # Still a gentle fade from near-zero...
        self.assertLess(out[0], 0.05)
        # ...but clearly audible (≥ -12 dB) within 0.75 s, not 2-3 s.
        self.assertGreaterEqual(out[int(0.75 * sr)], 0.25)
        # Full level once the fade completes (first un-faded sample).
        self.assertGreater(out[int(1.5 * sr)], 0.99)


class TestSpectralDucking(unittest.TestCase):
    """Multiband spectral ducking: B1 of the research plan."""

    def test_lr_crossover_perfect_reconstruction(self):
        """low + mid + high must reconstruct the input bit-transparently
        (subtractive mid) so the bed is untouched when no ducking occurs."""
        from core.mixer import _lr_crossover_3way
        sr = 48000
        rng = np.random.default_rng(0)
        music = rng.standard_normal(sr * 2).astype(np.float32) * 0.1
        low, mid, high = _lr_crossover_3way(music, sr, lo_hz=250.0, hi_hz=4000.0)
        recon = low + mid + high
        self.assertEqual(recon.shape, music.shape)
        self.assertLess(float(np.max(np.abs(recon - music))), 1e-5)

    def test_lr_crossover_reconstruction_stereo(self):
        from core.mixer import _lr_crossover_3way
        sr = 48000
        rng = np.random.default_rng(1)
        music = (rng.standard_normal((2, sr)) * 0.1).astype(np.float32)
        low, mid, high = _lr_crossover_3way(music, sr)
        self.assertLess(float(np.max(np.abs((low + mid + high) - music))), 1e-5)

    def test_transparent_when_no_speech(self):
        """With no detected phrases the gain is unity everywhere, so the
        multiband duck must return the music essentially unchanged."""
        from core.mixer import apply_breathing_duck_multiband
        sr = 48000
        music = _tone(1000.0, 4.0, sr, rms_db=-20.0)
        voice = np.zeros_like(music)
        out = apply_breathing_duck_multiband(
            voice, music, sr, duck_depth_db=-12.0, phrases=[],
        )
        self.assertEqual(out.shape, music.shape)
        self.assertLess(float(np.max(np.abs(out - music))), 1e-3)

    def test_mid_band_ducked_low_and_high_preserved(self):
        """During speech, a mid-band tone (1 kHz) is attenuated while a
        low-band (100 Hz) and high-band (8 kHz) tone are left alone."""
        from core.mixer import apply_breathing_duck_multiband
        sr = 48000
        phrases = [(1.0, 3.0)]
        a, b = int(1.5 * sr), int(2.5 * sr)   # window inside the phrase
        q, r = int(0.0 * sr), int(0.5 * sr)   # window before the phrase

        def duck(freq):
            music = _tone(freq, 4.0, sr, rms_db=-20.0)
            out = apply_breathing_duck_multiband(
                np.zeros_like(music), music, sr,
                duck_depth_db=-12.0, phrases=phrases,
            )
            return music, out

        mid_in, mid_out = duck(1000.0)
        low_in, low_out = duck(100.0)
        high_in, high_out = duck(8000.0)

        # Mid band: clearly attenuated during the phrase (~-12 dB → <0.5x).
        self.assertLess(_rms(mid_out[a:b]), 0.6 * _rms(mid_in[a:b]))
        # Mid band: untouched before the phrase.
        self.assertGreater(_rms(mid_out[q:r]), 0.95 * _rms(mid_in[q:r]))
        # Low and high bands: preserved even during the phrase.
        self.assertGreater(_rms(low_out[a:b]), 0.9 * _rms(low_in[a:b]))
        self.assertGreater(_rms(high_out[a:b]), 0.9 * _rms(high_in[a:b]))

    def test_mix_branches_to_multiband_via_env(self):
        """MOODSCAPE_SPECTRAL_DUCK=1 makes mix() preserve a sub-bass bed that
        the fullband duck would pull down — proving the branch is taken."""
        import os
        sr = 48000
        voice = _gated_voice(sr, rms_db=-21.0, n_phrases=2, on_s=4.0, off_s=3.0)
        activity = np.abs(voice) > 0
        music = _tone(80.0, len(voice) / sr + 40.0, sr, rms_db=-18.0)
        phrases = detect_phrases(voice, sr, threshold_db=-40.0)

        full = mix(voice, activity, music, sample_rate=sr,
                   fade_in_sec=0.0, fade_out_sec=0.0, phrases=phrases)
        os.environ["MOODSCAPE_SPECTRAL_DUCK"] = "1"
        try:
            multi = mix(voice, activity, music, sample_rate=sr,
                        fade_in_sec=0.0, fade_out_sec=0.0, phrases=phrases)
        finally:
            os.environ.pop("MOODSCAPE_SPECTRAL_DUCK", None)

        self.assertEqual(full.shape, multi.shape)
        # The 80 Hz bed lives in the low band: ducked fullband, preserved
        # multiband → the multiband mix retains more total energy.
        self.assertGreater(_rms(multi), _rms(full))


class TestSharedReverb(unittest.TestCase):
    """Shared convolution-reverb send: B2 of the research plan."""

    def test_send_preserves_length_and_adds_energy(self):
        from core.mixer import add_shared_reverb
        sr = 48000
        music = _tone(440.0, 2.0, sr, rms_db=-18.0)
        send = add_shared_reverb(music, sr, send_db=-20.0)
        self.assertEqual(send.shape, music.shape)
        self.assertTrue(np.all(np.isfinite(send)))
        self.assertGreater(_rms(send), 0.0)

    def test_send_level_tracks_send_db(self):
        """A lower send_db must produce a quieter send."""
        from core.mixer import add_shared_reverb
        sr = 48000
        music = _tone(440.0, 2.0, sr, rms_db=-18.0)
        loud = add_shared_reverb(music, sr, send_db=-18.0)
        quiet = add_shared_reverb(music, sr, send_db=-30.0)
        self.assertGreater(_rms(loud), _rms(quiet))

    def test_mix_shared_reverb_changes_output_via_env(self):
        import os
        sr = 48000
        voice = _gated_voice(sr, rms_db=-21.0, n_phrases=2, on_s=4.0, off_s=3.0)
        activity = np.abs(voice) > 0
        music = _tone(440.0, len(voice) / sr + 40.0, sr, rms_db=-20.0)
        phrases = detect_phrases(voice, sr, threshold_db=-40.0)

        base = mix(voice, activity, music, sample_rate=sr,
                   fade_in_sec=0.0, fade_out_sec=0.0, phrases=phrases)
        os.environ["MOODSCAPE_SHARED_REVERB"] = "1"
        try:
            verb = mix(voice, activity, music, sample_rate=sr,
                       fade_in_sec=0.0, fade_out_sec=0.0, phrases=phrases)
        finally:
            os.environ.pop("MOODSCAPE_SHARED_REVERB", None)
        self.assertEqual(base.shape, verb.shape)
        self.assertGreater(float(np.max(np.abs(verb - base))), 1e-4)


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


class TestBedRiseLimit(unittest.TestCase):
    sr = 16000

    def _voice(self, spans, total):
        v = np.zeros(int(total * self.sr), np.float32)
        t = np.arange(v.size) / self.sr
        for s, e in spans:
            m = (t >= s) & (t < e)
            v[m] = 0.3 * np.sin(2 * np.pi * 200 * t[m])
        return v

    def test_limiter_not_applied_when_flag_off(self):
        v = self._voice([(1, 3), (20, 22)], 25)
        ph = [(1, 3), (20, 22)]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOODSCAPE_BED_RISE_LIMIT", None)
            off = compute_breathing_gain_db(v, self.sr, duck_depth_db=-16.0, phrases=ph)
        with patch.dict(os.environ, {"MOODSCAPE_BED_RISE_LIMIT": "1"}):
            on = compute_breathing_gain_db(v, self.sr, duck_depth_db=-16.0, phrases=ph)
        sl = slice(int(10 * self.sr), int(18 * self.sr))
        self.assertGreaterEqual(off[sl].max(), -1.0)
        self.assertLessEqual(on[sl].max(), -4.0 + 0.05)

    def test_short_pause_rises_at_most_rate_times_pause(self):
        v = self._voice([(1, 3), (5, 7)], 10)
        with patch.dict(os.environ, {"MOODSCAPE_BED_RISE_LIMIT": "1"}):
            g = compute_breathing_gain_db(v, self.sr, duck_depth_db=-16.0, phrases=[(1, 3), (5, 7)])
        peak_in_pause = g[int(3 * self.sr):int(5 * self.sr)].max()
        self.assertLessEqual(peak_in_pause, -16.0 + 3.0 * 2.0 + 0.5)

    def test_long_pause_capped_below_today(self):
        v = self._voice([(1, 3), (20, 22)], 25)
        with patch.dict(os.environ, {"MOODSCAPE_BED_RISE_LIMIT": "1"}):
            g = compute_breathing_gain_db(v, self.sr, duck_depth_db=-16.0, phrases=[(1, 3), (20, 22)])
        self.assertLessEqual(g[int(10 * self.sr):int(18 * self.sr)].max(), -16.0 + 12.0 + 0.05)

    def test_falls_are_not_limited(self):
        g = np.concatenate([np.zeros(800, np.float32), np.full(800, -16.0, np.float32)])
        out = _slew_limit_rise(g, self.sr, 3.0)
        self.assertLessEqual(out[900], -15.0)

    def test_no_pause_equals_input_and_silence_has_no_nan(self):
        g = np.full(16000, -16.0, np.float32)
        np.testing.assert_allclose(_slew_limit_rise(g, self.sr, 3.0), g)
        v = np.zeros(16000, np.float32)
        with patch.dict(os.environ, {"MOODSCAPE_BED_RISE_LIMIT": "1"}):
            self.assertFalse(np.isnan(compute_breathing_gain_db(v, self.sr, duck_depth_db=-16.0)).any())


class TestBedOffsetOverrides(unittest.TestCase):
    SR = 48000

    def test_unset_returns_empty(self):
        self.assertEqual(bed_offset_overrides(environ={}), {})

    def test_both_set_widens_clamp(self):
        out = bed_offset_overrides(environ={
            "MOODSCAPE_BED_SPEECH_OFFSET_LU": "28",
            "MOODSCAPE_BED_PAUSE_OFFSET_LU": "20",
        })
        self.assertEqual(out, {
            "speech_offset_lu": 28.0,
            "pause_offset_lu": 20.0,
            "duck_clamp_db": (-20.0, -4.0),
        })

    def test_only_pause_set(self):
        out = bed_offset_overrides(environ={"MOODSCAPE_BED_PAUSE_OFFSET_LU": "20"})
        self.assertEqual(out, {"pause_offset_lu": 20.0, "duck_clamp_db": (-20.0, -4.0)})
        self.assertNotIn("speech_offset_lu", out)

    def test_non_numeric_names_variable(self):
        with self.assertRaises(ValueError) as ctx:
            bed_offset_overrides(environ={"MOODSCAPE_BED_PAUSE_OFFSET_LU": "abc"})
        self.assertIn("MOODSCAPE_BED_PAUSE_OFFSET_LU", str(ctx.exception))

    def test_defaults_to_os_environ(self):
        with patch.dict(os.environ, {"MOODSCAPE_BED_SPEECH_OFFSET_LU": "28",
                                     "MOODSCAPE_BED_PAUSE_OFFSET_LU": "20"}):
            self.assertEqual(bed_offset_overrides()["speech_offset_lu"], 28.0)

    def test_shallow_duck_behaviour(self):
        """With 28/20 overrides the calibrated duck is 8 dB (clamped by -4)."""
        voice = _gated_voice(self.SR, rms_db=-21.2)
        music = _tone(997.0, len(voice) / self.SR, self.SR, rms_db=-19.7)
        overrides = bed_offset_overrides(environ={
            "MOODSCAPE_BED_SPEECH_OFFSET_LU": "28",
            "MOODSCAPE_BED_PAUSE_OFFSET_LU": "20",
        })
        _, duck_db = calibrate_music_bed(voice, music, self.SR, **overrides)
        self.assertAlmostEqual(duck_db, -8.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()


class TestExportLoudness(unittest.TestCase):
    def _tone(self, sr=48000, sec=5.0):
        t = np.arange(int(sr * sec)) / sr
        return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def _measure(self, path):
        import soundfile as sf
        import pyloudnorm as pyln
        y, sr = sf.read(path, dtype="float32")
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)  # dual-mono playback
        return pyln.Meter(sr).integrated_loudness(y)

    def test_mono_meditation_plays_at_minus_16(self):
        target, off = export_lufs_target("meditation", environ={})
        path = export_audio(self._tone(), 48000, target_sample_rate=48000,
                            target_lufs=target, mono_playback_offset_db=off)
        self.assertAlmostEqual(self._measure(path), -16.0, delta=0.5)

    def test_mono_sleep_plays_at_minus_18(self):
        target, off = export_lufs_target("sleep_story", environ={})
        path = export_audio(self._tone(), 48000, target_sample_rate=48000,
                            target_lufs=target, mono_playback_offset_db=off)
        self.assertAlmostEqual(self._measure(path), -18.0, delta=0.5)

    def test_kill_switch_restores_legacy(self):
        self.assertEqual(
            export_lufs_target("sleep_story", environ={"MOODSCAPE_LOUDNESS_DUAL_MONO": "0"}),
            (-16.0, 0.0))

    def test_stereo_ignores_offset(self):
        st = np.stack([self._tone(), self._tone()])
        path = export_audio(st, 48000, target_sample_rate=48000,
                            target_lufs=-16.0, mono_playback_offset_db=3.01)
        self.assertAlmostEqual(self._measure(path), -16.0, delta=0.5)
