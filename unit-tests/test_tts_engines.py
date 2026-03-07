import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.kokoro_tts.preprocessor import (
    split_into_sentences as k_split,
    merge_sentences_to_chunks as k_merge,
)
from core.kokoro_tts.postprocessor import (
    crossfade_chunks as _crossfade_chunks,
    trim_tts_artifacts,
)

from core.parler_tts.preprocessor import (
    split_into_sentences as p_split,
)
from core.parler_tts.engine import (
    VOICE_PRESETS,
    ParlerTTSEngine,
    adjust_description_for_speed as p_adjust,
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

    # --- Parler Tests ---
    def test_parler_split_sentences(self):
        text = "Hello world. This is a somewhat longer sentence for testing! Okay?"
        # Parler merges <6 words instead of 4.
        sentences = p_split(text)
        self.assertGreater(len(sentences), 0)

    def test_parler_adjust_speed(self):
        desc = "Warm voice. Speaks normally."
        adj1 = p_adjust(desc, 0.6)
        self.assertIn("extremely slowly", adj1.lower())
        adj2 = p_adjust(desc, 0.8)
        self.assertIn("slowly", adj2.lower())
        adj3 = p_adjust(desc, 1.0)
        self.assertIn("natural", adj3.lower())

    # --- Parler Engine Execution Tests (mocked model) ---

    def _make_engine_with_mock_model(self, native_sr=44100):
        """Return a ParlerTTSEngine with a fully mocked model and tokenizer."""
        from core.speech_engine import SAMPLE_RATE

        engine = ParlerTTSEngine()
        engine.device = "cpu"
        engine._native_sr = native_sr

        # Tokenizer returns simple tensors
        import torch
        tok = MagicMock()
        tok.return_value = MagicMock(
            input_ids=torch.zeros(1, 5, dtype=torch.long),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
        )
        engine.tokenizer = tok

        # Model.generate returns a 1-D int tensor of audio samples (float values
        # are expected after .float() conversion, so use float-compatible values).
        num_samples = native_sr  # 1 second of fake audio
        fake_audio = torch.sin(torch.linspace(0, 6.28, num_samples)).unsqueeze(0)
        model = MagicMock()
        model.generate.return_value = fake_audio
        engine.model = model

        return engine

    def test_parler_generate_speech_chunk_calls_model(self):
        """_generate_speech_chunk should call model.generate and return float32 audio."""
        from core.speech_engine import SAMPLE_RATE
        engine = self._make_engine_with_mock_model()
        audio = engine._generate_speech_chunk("Breathe slowly.", "A calm voice.")

        engine.model.generate.assert_called_once()
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        self.assertGreater(len(audio), 0)

    def test_parler_generate_speech_chunk_passes_correct_kwargs(self):
        """model.generate must be called with the required keyword arguments."""
        engine = self._make_engine_with_mock_model()
        engine._generate_speech_chunk("Relax.", "Soft voice.")

        _, kwargs = engine.model.generate.call_args
        self.assertIn("input_ids", kwargs)
        self.assertIn("attention_mask", kwargs)
        self.assertIn("prompt_input_ids", kwargs)
        self.assertIn("prompt_attention_mask", kwargs)
        self.assertTrue(kwargs.get("do_sample"))

    def test_parler_synthesize_produces_audio_and_activity(self):
        """synthesize() should return matching-length voice_audio and voice_activity."""
        engine = self._make_engine_with_mock_model()
        segments = [
            {"type": "speech", "text": "Close your eyes and breathe deeply."},
            {"type": "pause", "duration_sec": 1.0},
            {"type": "speech", "text": "Feel the calm wash over you."},
        ]
        voice_audio, voice_activity = engine.synthesize(segments, voice="")

        self.assertIsInstance(voice_audio, np.ndarray)
        self.assertIsInstance(voice_activity, np.ndarray)
        self.assertEqual(len(voice_audio), len(voice_activity))
        self.assertGreater(len(voice_audio), 0)
        # model.generate should have been called at least once per speech segment
        self.assertGreaterEqual(engine.model.generate.call_count, 2)

    def test_parler_resolve_voice_description_preset(self):
        """_resolve_voice_description should map preset names to their descriptions."""
        engine = ParlerTTSEngine()
        label, desc = VOICE_PRESETS[0]
        resolved = engine._resolve_voice_description(label)
        self.assertEqual(resolved, desc)

    def test_parler_resolve_voice_description_raw_string(self):
        """_resolve_voice_description should pass through raw description strings."""
        engine = ParlerTTSEngine()
        raw = "A deep, resonant voice with slow pacing."
        self.assertEqual(engine._resolve_voice_description(raw), raw)

    def test_parler_unloaded_model_raises(self):
        """Calling synthesize before load_model should raise RuntimeError."""
        engine = ParlerTTSEngine()
        with self.assertRaises(RuntimeError):
            engine.synthesize([{"type": "speech", "text": "Hello."}])


if __name__ == '__main__':
    unittest.main()
