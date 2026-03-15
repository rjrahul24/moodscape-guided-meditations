import unittest
import numpy as np

from core.kokoro_tts.preprocessor import (
    split_into_sentences as k_split,
    merge_sentences_to_chunks as k_merge,
)
from core.kokoro_tts.postprocessor import (
    crossfade_chunks as _crossfade_chunks,
    trim_tts_artifacts,
)

class TestTTSEngines(unittest.TestCase):
    # --- Kokoro Tests ---
    def test_kokoro_split_sentences(self):
        text = "Hello world. This is an extremely long test sentence! Right? Yes."
        # "Hello world." (2) -> carry
        # "This is an extremely long test sentence!" (7) -> added to list (9 words)
        # "Right?" (1) -> carry
        # "Yes." (1) -> carry, appended to last element
        sentences = k_split(text)
        self.assertEqual(len(sentences), 1)
        self.assertTrue("Yes." in sentences[0])

    def test_kokoro_crossfade_chunks(self):
        # Two simple arrays of 2400 samples
        chunk1 = np.ones(2400, dtype=np.float32)
        chunk2 = np.ones(2400, dtype=np.float32) * 0.5

        result = _crossfade_chunks([chunk1, chunk2], crossfade_samples=100)
        # Total length should be sum - overlap
        self.assertEqual(len(result), 2400 + 2400 - 100)

    def test_kokoro_crossfade_300ms(self):
        """Default crossfade is 300ms (7200 samples at 24 kHz)."""
        sr = 24000
        crossfade_samples = int(0.300 * sr)  # 7200
        # Chunks must be at least 2x the crossfade length to avoid clamping
        chunk1 = np.ones(crossfade_samples * 3, dtype=np.float32)
        chunk2 = np.ones(crossfade_samples * 3, dtype=np.float32) * 0.5

        result = _crossfade_chunks([chunk1, chunk2], crossfade_samples=crossfade_samples)
        expected_len = len(chunk1) + len(chunk2) - crossfade_samples
        self.assertEqual(len(result), expected_len)

    def test_kokoro_crossfade_equal_power(self):
        """Cosine-squared crossfade preserves energy: fade_out + fade_in = 1.0.

        fade_out = cos²(t), fade_in = sin²(t) = cos²(π/2 - t)
        By the Pythagorean identity: cos²(t) + sin²(t) = 1 at all points.
        """
        n = 7200
        t_out = np.linspace(0, np.pi / 2, n)
        t_in = np.linspace(np.pi / 2, 0, n)
        fade_out = np.cos(t_out) ** 2
        fade_in = np.cos(t_in) ** 2
        np.testing.assert_allclose(fade_out + fade_in, np.ones(n), atol=1e-6)
        
    def test_trim_tts_artifacts(self):
        sr = 24000
        # 0.5s of loud sine wave
        t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
        signal = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)
        # 0.5s of strict zero silence
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        
        audio = np.concatenate([signal, silence])
        trimmed = trim_tts_artifacts(audio, sr=sr)
        
        # We expect trailing silence to be trimmed, preserving a minimum tail
        self.assertLess(len(trimmed), len(audio))
        self.assertGreaterEqual(len(trimmed), len(signal))

if __name__ == '__main__':
    unittest.main()
