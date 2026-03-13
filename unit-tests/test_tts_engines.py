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
