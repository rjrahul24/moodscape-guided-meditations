"""Tests for Chatterbox TTS engine and postprocessor integration."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from core.chatterbox_tts.engine import ChatterboxEngine, _resolve_reference_audio
from core.chatterbox_tts.postprocessor import ChatterboxMasteringEngine, build_chatterbox_voice_chain
from core.speech_engine import SAMPLE_RATE


class TestChatterboxEngine(unittest.TestCase):
    def test_init_defaults(self):
        engine = ChatterboxEngine()
        self.assertEqual(engine.exaggeration, 0.28)
        self.assertEqual(engine.cfg_weight, 0.35)
        self.assertEqual(engine.temperature, 0.55)
        self.assertEqual(engine.voice_slug, "Brittney")
        self.assertIn(engine.device, ("mps", "cuda", "cpu"))
        self.assertFalse(engine._loaded)

    def test_resolve_reference_audio(self):
        # Known existing reference files in assets/speakers/reference_audio
        # None or default should resolve to Brittney.wav
        default_path = _resolve_reference_audio(None)
        self.assertIsNotNone(default_path)
        self.assertTrue(default_path.endswith("Brittney.wav"))

        auto_path = _resolve_reference_audio("default")
        self.assertIsNotNone(auto_path)
        self.assertTrue(auto_path.endswith("Brittney.wav"))

        # Explicit resemblance baseline
        self.assertIsNone(_resolve_reference_audio("resemble_default"))

        # Specific library voices
        brittney_path = _resolve_reference_audio("Brittney")
        self.assertIsNotNone(brittney_path)
        self.assertTrue(brittney_path.endswith("Brittney.wav"))

        clara_path = _resolve_reference_audio("Clara")
        self.assertIsNotNone(clara_path)
        self.assertTrue(clara_path.endswith("Clara.wav"))

    def test_get_available_voices(self):
        engine = ChatterboxEngine()
        voices = engine.get_available_voices()
        voice_ids = [v["id"] for v in voices]
        self.assertIn("Brittney", voice_ids)
        self.assertIn("Clara", voice_ids)
        self.assertIn("Delilah", voice_ids)
        self.assertIn("Eryn", voice_ids)
        self.assertIn("resemble_default", voice_ids)

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
        self.assertFalse(np.any(voice_activity[:12000]))
        self.assertTrue(np.all(voice_activity[12000:]))

    def test_unload_model(self):
        engine = ChatterboxEngine(device="cpu")
        engine.model = MagicMock()
        engine._loaded = True
        engine.unload_model()
        self.assertIsNone(engine.model)
        self.assertFalse(engine._loaded)


class TestChatterboxPostprocessor(unittest.TestCase):
    def test_mastering_engine(self):
        master = ChatterboxMasteringEngine()
        # 1 second of audio at 48 kHz
        audio = np.random.uniform(-0.5, 0.5, 48000).astype(np.float32)
        out = master.master_vocals(audio, sr=48000)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), 48000)
        self.assertTrue(np.all(np.abs(out) <= 1.0))
        self.assertFalse(np.isnan(out).any())

    def test_dry_voice_chain(self):
        # reverb_amount = 0 should return dry limiter only without reverb
        chain = build_chatterbox_voice_chain(reverb_amount=0.0)
        audio = np.random.uniform(-0.5, 0.5, 48000).astype(np.float32)
        audio_2d = audio.reshape(1, -1)
        out = chain(audio_2d, 48000).squeeze(0)
        self.assertEqual(len(out), 48000)
        self.assertTrue(np.all(np.abs(out) <= 1.0))

    def test_abbey_road_reverb_chain(self):
        chain = build_chatterbox_voice_chain(reverb_amount=0.05, ir_name="warm_studio")
        audio = np.random.uniform(-0.5, 0.5, 48000).astype(np.float32)
        audio_2d = audio.reshape(1, -1)
        out = chain(audio_2d, 48000).squeeze(0)
        self.assertEqual(len(out), 48000)
        self.assertTrue(np.all(np.abs(out) <= 1.0))
