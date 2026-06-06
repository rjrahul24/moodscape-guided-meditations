"""Regression tests for the music/vocal mixing quality fixes.

Covers the three defects found in the uploaded-instrumental + voice mix:
  1. Broadband "static" distortion from the pedalboard Limiter in the music
     and master chains.
  2. Ducking that is too shallow / not gradual / does not rise during pauses.
  3. Over-loud baseline music level.
"""

import numpy as np
import unittest

SR = 48000


def _noise_floor_db(x, tones=(110.0, 220.0, 440.0), sr=SR):
    """Broadband energy NOT at the known tones/harmonics, relative to signal.

    A clean linear process leaves this very low (< -90 dB); nonlinear
    distortion (the buggy limiter) raises it into the audible range.
    """
    x = np.asarray(x, dtype=np.float64)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    mask = np.ones_like(X, dtype=bool)
    for t in tones:
        for h in range(1, 8):
            mask &= np.abs(f - t * h) > 8.0
    sig = np.sqrt(np.sum(X[~mask] ** 2)) + 1e-12
    noise = np.sqrt(np.sum(X[mask] ** 2)) + 1e-12
    return 20.0 * np.log10(noise / sig)


def _tones(seconds=8.0, amp=0.5, sr=SR):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    x = 0.4 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    return (x / np.max(np.abs(x)) * amp).astype(np.float32)


class TestChainsAreClean(unittest.TestCase):
    """Issue 1: the music + master chains must not add broadband distortion."""

    def test_upload_music_chain_clean(self):
        from core.audio_processor import make_upload_music_chain, apply_fx
        x = _tones(amp=0.4)
        out = apply_fx(x, make_upload_music_chain(), SR)
        self.assertLess(_noise_floor_db(out), -70.0,
                        "upload music chain adds audible broadband distortion")
        self.assertLessEqual(np.max(np.abs(out)), np.max(np.abs(x)) * 1.15,
                             "music chain unexpectedly inflates level")

    def test_master_chain_clean(self):
        from core.audio_processor import make_master_chain, apply_fx
        # Below the glue compressor's -12 dB threshold, so we isolate the
        # limiter-type broadband distortion bug (the compressor doing normal
        # gain-riding on louder material is expected and not "static").
        x = _tones(amp=0.15)
        out = apply_fx(x, make_master_chain(), SR)
        self.assertLess(_noise_floor_db(out), -70.0,
                        "master chain adds audible broadband distortion")

    def test_no_limiter_in_music_or_master_chains(self):
        from pedalboard import Limiter
        from core.audio_processor import (
            make_upload_music_chain, make_acestep_music_chain,
            make_lyria_music_chain, make_master_chain,
        )
        for chain in (make_upload_music_chain(), make_acestep_music_chain(),
                      make_lyria_music_chain(), make_master_chain()):
            self.assertFalse(any(isinstance(p, Limiter) for p in chain),
                             "pedalboard Limiter must be removed (it distorts)")


class TestTruePeakLimiter(unittest.TestCase):
    """Clean true-peak limiting replaces the broken pedalboard Limiter."""

    def test_holds_true_peak_ceiling(self):
        from core.mixer import true_peak_limit
        from scipy.signal import resample_poly
        x = _tones(amp=1.0) * 2.0  # hot: peak ~2.0
        out = true_peak_limit(x, SR, threshold_db=-1.0)
        up = resample_poly(out, 4, 1, axis=-1)
        tp_db = 20.0 * np.log10(np.max(np.abs(up)) + 1e-12)
        self.assertLessEqual(tp_db, -1.0 + 0.05, f"true peak {tp_db:.2f} exceeds -1 dBTP")

    def test_transparent_below_threshold(self):
        from core.mixer import true_peak_limit
        x = _tones(amp=0.2)  # peak 0.2, far below -1 dBTP (0.891)
        out = true_peak_limit(x, SR, threshold_db=-1.0)
        # Must NOT inflate a sub-threshold signal (the pedalboard bug did +4.75 dB).
        self.assertAlmostEqual(np.max(np.abs(out)), np.max(np.abs(x)), delta=0.01)
        self.assertLess(_noise_floor_db(out), -70.0)


class TestBreathingDuck(unittest.TestCase):
    """Issue 2: deep, gradual, breathing duck (drops low under speech,
    rises gradually during pauses)."""

    def _voice_with_phrases(self, total_s=16.0):
        n = int(total_s * SR)
        v = np.zeros(n, dtype=np.float32)
        # phrase A [2,5], long pause [5,11], phrase B [11,14]
        for a, b in [(2.0, 5.0), (11.0, 14.0)]:
            t = np.linspace(0, b - a, int((b - a) * SR), endpoint=False, dtype=np.float32)
            v[int(a * SR):int(a * SR) + len(t)] = 0.4 * np.sin(2 * np.pi * 300 * t)
        return v, n

    def test_gain_curve_deep_during_speech_and_rises_in_pause(self):
        from core.mixer import compute_breathing_gain_db
        v, n = self._voice_with_phrases()
        g = compute_breathing_gain_db(v, SR, duck_depth_db=-15.0)
        self.assertEqual(g.shape[0], n)
        speech = g[int(3.5 * SR)]          # mid phrase A
        pause = g[int(8.0 * SR)]           # mid long pause
        self.assertLessEqual(speech, -12.0, f"duck too shallow during speech ({speech:.1f} dB)")
        self.assertGreater(pause, -3.0, f"music did not rise during pause ({pause:.1f} dB)")
        self.assertGreater(pause - speech, 9.0, "insufficient breathing range")

    def test_gain_curve_is_gradual(self):
        from core.mixer import compute_breathing_gain_db
        v, n = self._voice_with_phrases()
        g = compute_breathing_gain_db(v, SR, duck_depth_db=-15.0)
        max_step = float(np.max(np.abs(np.diff(g))))
        self.assertLess(max_step, 0.02, f"gain moves too abruptly ({max_step:.4f} dB/sample)")

    def test_apply_breathing_duck_attenuates_music_under_speech(self):
        from core.mixer import apply_breathing_duck
        v, n = self._voice_with_phrases()
        music = (0.5 * np.sin(2 * np.pi * 200 * np.arange(n) / SR)).astype(np.float32)
        ducked = apply_breathing_duck(v, music, SR, duck_depth_db=-15.0)
        self.assertEqual(ducked.shape, music.shape)

        def rms(x, c):  # 0.4 s window
            return float(np.sqrt(np.mean(x[int(c * SR) - 9600:int(c * SR) + 9600] ** 2)))
        speech_rms = rms(ducked, 3.5)
        pause_rms = rms(ducked, 8.0)
        self.assertGreater(pause_rms, speech_rms * 3.0,
                           "music not clearly ducked under speech vs pause")


if __name__ == "__main__":
    unittest.main()
