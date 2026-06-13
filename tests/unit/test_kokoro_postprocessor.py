import unittest
import numpy as np
from unittest.mock import patch

from core.kokoro_tts.postprocessor import (
    build_voice_chain,
    generate_room_tone,
    humanize_voice,
    process_chunk,
    crossfade_chunks,
    apply_segment_fades,
    apply_fx,
    split_band_deess,
    apply_parallel_compression,
    normalize_chunk_rms,
    SAMPLE_RATE,
)


class TestGenerateRoomTone(unittest.TestCase):
    def test_correct_length(self):
        tone = generate_room_tone(1.0, sr=SAMPLE_RATE)
        expected = SAMPLE_RATE
        self.assertEqual(len(tone), expected)

    def test_short_duration(self):
        tone = generate_room_tone(0.05, sr=SAMPLE_RATE)
        self.assertEqual(len(tone), int(0.05 * SAMPLE_RATE))

    def test_zero_duration(self):
        tone = generate_room_tone(0.0, sr=SAMPLE_RATE)
        self.assertEqual(len(tone), 0)

    def test_float32_output(self):
        tone = generate_room_tone(0.5, sr=SAMPLE_RATE)
        self.assertEqual(tone.dtype, np.float32)

    def test_level_approximately_correct(self):
        tone = generate_room_tone(2.0, sr=SAMPLE_RATE, level_db=-55.0)
        rms = np.sqrt(np.mean(tone ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        # Should be approximately -55 dBFS (within 5 dB tolerance)
        self.assertAlmostEqual(rms_db, -55.0, delta=5.0)

    def test_no_clipping(self):
        tone = generate_room_tone(1.0, sr=SAMPLE_RATE, level_db=-30.0)
        self.assertLessEqual(np.max(np.abs(tone)), 1.0)


class TestHumanizeVoice(unittest.TestCase):
    def test_returns_unchanged_when_too_short(self):
        short = np.random.randn(100).astype(np.float32) * 0.1
        result = humanize_voice(short, sr=SAMPLE_RATE)
        np.testing.assert_array_equal(result, short)

    def test_output_length_matches_input(self):
        # Generate a 1-second sine wave (a simple voiced signal)
        t = np.linspace(0, 1.0, SAMPLE_RATE, dtype=np.float64)
        audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        result = humanize_voice(audio, sr=SAMPLE_RATE)
        self.assertEqual(len(result), len(audio))

    def test_output_is_float32(self):
        t = np.linspace(0, 1.0, SAMPLE_RATE, dtype=np.float64)
        audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        result = humanize_voice(audio, sr=SAMPLE_RATE)
        self.assertEqual(result.dtype, np.float32)

    def test_graceful_degradation_without_pyworld(self):
        """If pyworld is not available, should return audio unchanged."""
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.3
        with patch('core.kokoro_tts.postprocessor._PYWORLD_AVAILABLE', False):
            result = humanize_voice(audio, sr=SAMPLE_RATE)
            np.testing.assert_array_equal(result, audio)


class TestProcessChunk(unittest.TestCase):
    def test_dc_offset_removed(self):
        audio = np.ones(1000, dtype=np.float32) * 0.5
        result = process_chunk(audio)
        self.assertAlmostEqual(np.mean(result), 0.0, places=3)

    def test_output_within_range(self):
        audio = np.random.randn(5000).astype(np.float32) * 2.0
        result = process_chunk(audio)
        self.assertLessEqual(np.max(np.abs(result)), 1.0)


class TestCrossfadeChunks(unittest.TestCase):
    def test_single_chunk_passthrough(self):
        chunk = np.ones(1000, dtype=np.float32) * 0.5
        result = crossfade_chunks([chunk])
        np.testing.assert_array_equal(result, chunk)

    def test_two_chunks_shorter_than_sum(self):
        c1 = np.ones(2000, dtype=np.float32) * 0.3
        c2 = np.ones(2000, dtype=np.float32) * 0.3
        result = crossfade_chunks([c1, c2])
        # Crossfaded result should be shorter than concatenation
        self.assertLess(len(result), 4000)

    def test_empty_list(self):
        result = crossfade_chunks([])
        self.assertEqual(len(result), 0)


class TestNormalizeChunkRms(unittest.TestCase):
    def test_normalizes_to_target(self):
        # Use a signal with RMS near the target so the gain clamp doesn't limit
        target_rms = 10 ** (-23.0 / 20)  # ~0.0708
        np.random.seed(42)
        audio = np.random.randn(10000).astype(np.float32) * target_rms * 2
        audio -= np.mean(audio)
        result = normalize_chunk_rms(audio, target_db=-23.0)
        rms = np.sqrt(np.mean(result ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        self.assertAlmostEqual(rms_db, -23.0, delta=1.5)

    def test_silent_chunk_unchanged(self):
        silent = np.zeros(1000, dtype=np.float32)
        result = normalize_chunk_rms(silent)
        np.testing.assert_array_equal(result, silent)


class TestBuildVoiceChainParameters(unittest.TestCase):
    """Verify the tuned FX chain parameters match the spec."""

    def test_compressor_threshold_is_minus_28(self):
        from pedalboard import Compressor
        chain = build_voice_chain()
        compressor = next((p for p in chain if isinstance(p, Compressor)), None)
        self.assertIsNotNone(compressor, "Expected a Compressor plugin in build_voice_chain()")
        self.assertAlmostEqual(
            compressor.threshold_db, -28.0, places=1,
            msg=f"Compressor threshold should be -28 dB, got {compressor.threshold_db}"
        )

    def test_compressor_ratio_unchanged(self):
        from pedalboard import Compressor
        chain = build_voice_chain()
        compressor = next((p for p in chain if isinstance(p, Compressor)), None)
        self.assertAlmostEqual(compressor.ratio, 2.5, places=1)

    def test_default_reverb_mix_is_0_18(self):
        from pedalboard import Convolution
        chain = build_voice_chain()
        convolution = next((p for p in chain if isinstance(p, Convolution)), None)
        self.assertIsNotNone(convolution, "Expected a Convolution plugin in build_voice_chain()")
        self.assertAlmostEqual(
            convolution.mix, 0.18, places=2,
            msg=f"Default reverb mix should be 0.18, got {convolution.mix}"
        )

    def test_custom_reverb_amount_respected(self):
        from pedalboard import Convolution
        chain = build_voice_chain(reverb_amount=0.10)
        convolution = next((p for p in chain if isinstance(p, Convolution)), None)
        self.assertAlmostEqual(convolution.mix, 0.10, places=2)


class TestSplitBandDeess(unittest.TestCase):
    def _tone(self, freq, dur=0.5, amp=0.5):
        t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_attenuates_sibilant_band(self):
        # A loud 6.5 kHz tone sits inside the de-esser band and should be reduced.
        sib = self._tone(6500, amp=0.7)
        out = split_band_deess(sib, sr=SAMPLE_RATE)
        self.assertLess(
            np.sqrt(np.mean(out ** 2)),
            np.sqrt(np.mean(sib ** 2)),
            "6.5 kHz energy should be attenuated by the de-esser",
        )

    def test_preserves_low_band(self):
        # A 200 Hz tone is well outside the band and should pass ~unchanged.
        low = self._tone(200, amp=0.5)
        out = split_band_deess(low, sr=SAMPLE_RATE)
        rms_in = np.sqrt(np.mean(low ** 2))
        rms_out = np.sqrt(np.mean(out ** 2))
        self.assertAlmostEqual(rms_out, rms_in, delta=rms_in * 0.1)

    def test_float32_output(self):
        out = split_band_deess(self._tone(6500), sr=SAMPLE_RATE)
        self.assertEqual(out.dtype, np.float32)


class TestParallelCompression(unittest.TestCase):
    def test_float32_and_finite(self):
        t = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
        sig = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        out = apply_parallel_compression(sig, sr=SAMPLE_RATE)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(len(out), len(sig))


class TestApplyFxKokoroFlags(unittest.TestCase):
    def setUp(self):
        t = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
        # Mix of a mid tone + a strong sibilant tone.
        self.sig = (
            0.3 * np.sin(2 * np.pi * 300 * t)
            + 0.4 * np.sin(2 * np.pi * 6500 * t)
        ).astype(np.float32)
        self.chain = build_voice_chain()

    def test_deess_off_is_passthrough_vs_on(self):
        with patch.dict('os.environ', {'MOODSCAPE_KOKORO_DEESS': '0'}):
            off = apply_fx(self.sig, build_voice_chain(), SAMPLE_RATE, engine='kokoro')
        with patch.dict('os.environ', {'MOODSCAPE_KOKORO_DEESS': '1'}):
            on = apply_fx(self.sig, build_voice_chain(), SAMPLE_RATE, engine='kokoro')
        n = min(len(off), len(on))
        # De-essing should change the output (reduce sibilant energy).
        self.assertFalse(np.allclose(off[:n], on[:n], atol=1e-6))

    def test_non_kokoro_engine_ignores_kokoro_flags(self):
        # With engine != "kokoro", de-essing/parallel comp must not run even
        # if flags are set — F5 handles de-essing in its mastering.
        with patch.dict('os.environ', {
            'MOODSCAPE_KOKORO_DEESS': '1',
            'MOODSCAPE_KOKORO_PARALLEL_COMP': '1',
        }):
            f5 = apply_fx(self.sig, build_voice_chain(), SAMPLE_RATE, engine='f5')
            none = apply_fx(self.sig, build_voice_chain(), SAMPLE_RATE, engine=None)
        n = min(len(f5), len(none))
        self.assertTrue(np.allclose(f5[:n], none[:n], atol=1e-6))

    def test_output_within_peak_ceiling(self):
        with patch.dict('os.environ', {'MOODSCAPE_KOKORO_PARALLEL_COMP': '1'}):
            out = apply_fx(self.sig, build_voice_chain(), SAMPLE_RATE, engine='kokoro')
        self.assertLessEqual(np.max(np.abs(out)), 1.0)

    def test_saturation_reorder_runs(self):
        with patch.dict('os.environ', {'MOODSCAPE_KOKORO_SAT_PLACEMENT': 'pre_reverb'}):
            out = apply_fx(self.sig, build_voice_chain(), SAMPLE_RATE, engine='kokoro')
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(out)))


class TestProximityEqEnv(unittest.TestCase):
    def test_proximity_gain_env_respected(self):
        from pedalboard import LowShelfFilter
        with patch.dict('os.environ', {
            'MOODSCAPE_KOKORO_PROXIMITY_DB': '3.0',
            'MOODSCAPE_KOKORO_PROXIMITY_HZ': '160',
        }):
            chain = build_voice_chain()
        shelf = next((p for p in chain if isinstance(p, LowShelfFilter)), None)
        self.assertIsNotNone(shelf)
        self.assertAlmostEqual(shelf.gain_db, 3.0, places=2)
        self.assertAlmostEqual(shelf.cutoff_frequency_hz, 160.0, places=1)


if __name__ == '__main__':
    unittest.main()
