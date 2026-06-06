"""Unit tests for the uploaded-instrumental music source.

Covers the length-fitting helper (loop / trim / used-as-is), the
UploadMusicEngine drop-in contract (mono float32 @ 48 kHz, exact length),
and the dedicated FX chain.
"""

import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from core.upload_music import UploadMusicEngine, TARGET_SAMPLE_RATE
from core.upload_music.arrange import fit_to_length


SR = 48000


def _sine(seconds: float, freq: float = 220.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


class TestFitToLength(unittest.TestCase):
    def test_equal_length_used_as_is(self):
        bg = _sine(2.0)
        target = bg.shape[-1]
        out, report = fit_to_length(bg, SR, target)
        self.assertEqual(out.shape[-1], target)
        self.assertEqual(report.mode, "used_as_is")
        np.testing.assert_allclose(np.squeeze(out), bg, atol=0.0)

    def test_longer_source_is_trimmed(self):
        bg = _sine(5.0)
        target = int(3.0 * SR)
        out, report = fit_to_length(bg, SR, target)
        self.assertEqual(out.shape[-1], target)
        self.assertEqual(report.mode, "trimmed")

    def test_shorter_source_is_looped(self):
        bg = _sine(1.3)  # non-integer relationship → real seam
        target = int(5.0 * SR)
        out, report = fit_to_length(bg, SR, target)
        self.assertEqual(out.shape[-1], target)
        self.assertEqual(report.mode, "looped")
        self.assertGreaterEqual(report.loops, 2)
        out1d = np.squeeze(out)
        # No hard discontinuity (click) at the crossfaded seams.
        self.assertLess(float(np.max(np.abs(np.diff(out1d)))), 0.1)
        # Equal-power crossfade must not introduce clipping.
        self.assertLessEqual(float(np.max(np.abs(out1d))), 1.0)
        self.assertTrue(np.all(np.isfinite(out1d)))

    def test_tiny_source_falls_back_to_tiling(self):
        bg = _sine(0.002)  # < 4 ms → no meaningful crossfade
        target = int(0.05 * SR)
        out, report = fit_to_length(bg, SR, target)
        self.assertEqual(out.shape[-1], target)
        self.assertEqual(report.mode, "tiled_no_xfade")


class TestUploadMusicEngine(unittest.TestCase):
    def _write_temp(self, audio: np.ndarray, sr: int = SR, suffix: str = ".wav") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        sf.write(path, audio, sr)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_target_sample_rate_constant(self):
        self.assertEqual(TARGET_SAMPLE_RATE, 48000)

    def test_generate_returns_mono_float32_48k_exact_length(self):
        path = self._write_temp(_sine(5.0))
        engine = UploadMusicEngine(path)
        engine.load_model()
        out = engine.generate("ignored prompt", 20.0)
        engine.unload_model()

        self.assertEqual(out.ndim, 1, "engine must return mono 1-D audio")
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape[0], int(round(20.0 * 48000)))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_generate_downmixes_stereo_to_mono(self):
        stereo = np.stack([_sine(3.0), _sine(3.0, freq=330.0)], axis=-1)  # (n, 2)
        path = self._write_temp(stereo)
        engine = UploadMusicEngine(path)
        engine.load_model()
        out = engine.generate("", 3.0)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape[0], int(round(3.0 * 48000)))

    def test_generate_resamples_to_48k(self):
        # Source at 22.05 kHz must be resampled so 2 s == 96000 samples @ 48k.
        path = self._write_temp(_sine(2.0, sr=22050), sr=22050)
        engine = UploadMusicEngine(path)
        engine.load_model()
        out = engine.generate("", 2.0)
        self.assertEqual(out.shape[0], int(round(2.0 * 48000)))

    def test_generate_exposes_fit_report(self):
        path = self._write_temp(_sine(2.0))
        engine = UploadMusicEngine(path)
        engine.load_model()
        engine.generate("", 10.0)  # shorter source → looped
        self.assertIsNotNone(engine.fit_report)
        self.assertEqual(engine.fit_report.mode, "looped")


class TestUploadMusicChain(unittest.TestCase):
    def test_chain_processes_without_changing_length(self):
        from core.audio_processor import make_upload_music_chain

        chain = make_upload_music_chain()
        audio = _sine(2.0)
        out = chain(audio, SR)
        out = np.squeeze(out)
        self.assertEqual(out.shape[-1], audio.shape[-1])
        self.assertTrue(np.all(np.isfinite(out)))


class TestUploadFlowsThroughMaster(unittest.TestCase):
    """An upload-shaped music array + voice must master to spec via the
    project's existing mix/master/export path (no upload-specific mixing)."""

    def test_master_lands_near_minus16_lufs_under_peak_ceiling(self):
        import pyloudnorm as pyln
        import soundfile as sf_read
        from core.mixer import mix, export_audio, normalize_loudness
        from core.audio_processor import make_master_chain

        # Synthetic narration: 20 s with three voiced phrases.
        n = int(20.0 * SR)
        voice = np.zeros(n, dtype=np.float32)
        activity = np.zeros(n, dtype=bool)
        for start_s, end_s in [(3.0, 6.0), (8.0, 12.0), (14.0, 18.0)]:
            sl = slice(int(start_s * SR), int(end_s * SR))
            voice[sl] = _sine(end_s - start_s, freq=180.0)
            activity[sl] = True
        voice = normalize_loudness(voice, SR, target_lufs=-18.0)

        # Uploaded instrumental fitted to the same length.
        engine = UploadMusicEngine.__new__(UploadMusicEngine)  # skip path check
        engine.uploaded_path = None
        engine.fit_report = None
        from core.upload_music.arrange import fit_to_length
        music, _ = fit_to_length(_sine(5.0, freq=110.0), SR, n)
        music = normalize_loudness(np.squeeze(music), SR, target_lufs=-16.0)

        mixed = mix(voice, activity, music, sample_rate=SR,
                    duck_amount_db=-12.0, fade_in_sec=3.0, fade_out_sec=5.0)

        out_path = export_audio(
            mixed, sample_rate=SR, output_format="wav",
            target_sample_rate=48000, master_chain=make_master_chain(),
            target_lufs=-16.0,
        )
        self.addCleanup(lambda: os.path.exists(out_path) and os.remove(out_path))

        data, file_sr = sf_read.read(out_path)
        meter = pyln.Meter(file_sr)
        lufs = meter.integrated_loudness(data)
        peak = float(np.max(np.abs(data)))

        self.assertAlmostEqual(lufs, -16.0, delta=1.0,
                               msg=f"master LUFS {lufs:.2f} not within -16 ±1")
        self.assertLessEqual(peak, 10 ** (-1.5 / 20.0) + 1e-3,
                             msg=f"peak {peak:.4f} exceeds -1.5 dBTP export ceiling")


if __name__ == "__main__":
    unittest.main()
