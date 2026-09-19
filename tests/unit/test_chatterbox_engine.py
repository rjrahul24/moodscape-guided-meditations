"""Tests for Chatterbox TTS engine integration."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from core.chatterbox_tts.engine import ChatterboxEngine, _resolve_reference_audio
from core.speech_engine import SAMPLE_RATE


class TestChatterboxEngine(unittest.TestCase):
    def test_init_defaults(self):
        engine = ChatterboxEngine()
        self.assertEqual(engine.exaggeration, 0.30)
        self.assertEqual(engine.cfg_weight, 0.50)
        self.assertIn(engine.device, ("mps", "cuda", "cpu"))
        self.assertFalse(engine._loaded)

    def test_resolve_reference_audio(self):
        # Known existing reference files in assets/speakers/reference_audio
        self.assertIsNone(_resolve_reference_audio(None))
        self.assertIsNone(_resolve_reference_audio("default"))
        brittney_path = _resolve_reference_audio("Brittney")
        self.assertIsNotNone(brittney_path)
        self.assertTrue(brittney_path.endswith(".wav"))

    def test_get_available_voices(self):
        engine = ChatterboxEngine()
        voices = engine.get_available_voices()
        voice_ids = [v["id"] for v in voices]
        self.assertIn("default", voice_ids)
        self.assertIn("Brittney", voice_ids)

    def test_synthesize_with_pause_and_speech(self):
        engine = ChatterboxEngine(device="cpu")
        fake_model = MagicMock()
        # Mock 1 second of audio at 24kHz
        fake_wav = torch.zeros((1, 24000), dtype=torch.float32)
        fake_model.generate.return_value = fake_wav
        engine.model = fake_model
        engine._loaded = True

        segments = [
            {"type": "pause", "duration_sec": 0.5},
            {"type": "speech", "text": "Take a slow breath."},
        ]

        voice_audio, voice_activity = engine.synthesize(segments, speed=1.0)
        self.assertIsInstance(voice_audio, np.ndarray)
        self.assertIsInstance(voice_activity, np.ndarray)
        self.assertEqual(voice_audio.dtype, np.float32)
        self.assertEqual(voice_activity.dtype, bool)

        # 0.5s pause = 12000 samples, 1.0s speech = 24000 samples -> 36000 total
        self.assertEqual(len(voice_audio), 36000)
        self.assertEqual(len(voice_activity), 36000)
        # First 12000 samples should be False in activity mask
        self.assertFalse(np.any(voice_activity[:12000]))
        # Next 24000 samples should be True in activity mask
        self.assertTrue(np.all(voice_activity[12000:]))

    def test_unload_model(self):
        engine = ChatterboxEngine(device="cpu")
        engine.model = MagicMock()
        engine._loaded = True
        engine.unload_model()
        self.assertIsNone(engine.model)
        self.assertFalse(engine._loaded)
