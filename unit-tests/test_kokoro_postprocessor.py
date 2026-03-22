import unittest
import numpy as np
from unittest.mock import patch

from core.kokoro_tts.postprocessor import (
    generate_room_tone,
    humanize_voice,
    process_chunk,
    crossfade_chunks,
    apply_segment_fades,
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


if __name__ == '__main__':
    unittest.main()
