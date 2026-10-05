"""Tests for Chatterbox TTS engine, preprocessor, and postprocessor integration."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from pedalboard import Limiter

from core.chatterbox_tts.engine import (
    ChatterboxEngine,
    _condition_reference_audio,
    _resolve_reference_audio,
    _trim_vocoder_silence,
)
from core.chatterbox_tts.postprocessor import (
    ChatterboxMasteringEngine,
    build_chatterbox_voice_chain,
)
from core.chatterbox_tts.preprocessor import (
    normalize_for_chatterbox,
    prepare_segments,
    split_into_chunks,
)
from core.speech_engine import SAMPLE_RATE


class TestChatterboxPreprocessor(unittest.TestCase):
    def test_quote_stripping(self):
        text = 'She whispered: "Take a deep breath," and \'relax\'.'
        norm = normalize_for_chatterbox(text)
        self.assertNotIn('"', norm)
        self.assertNotIn("'", norm)
        self.assertIn("take a deep breath", norm.lower())

    def test_case_softening(self):
        text = "PLEASE BREATHE IN deeply AND EXHALE slowly."
        norm = normalize_for_chatterbox(text)
        # All-caps words should be downcased to avoid aggressive volume projection
        self.assertNotIn("PLEASE", norm)
        self.assertNotIn("BREATHE", norm)
        self.assertNotIn("EXHALE", norm)
        self.assertIn("please breathe in", norm.lower())

    def test_preserve_ellipses_and_em_dashes(self):
        text = "Letting go... of all thoughts—feeling calm."
        norm = normalize_for_chatterbox(text)
        self.assertIn("...", norm)
        self.assertIn("—", norm)

    def test_chunking_bounds(self):
        long_text = (
            "Allow your entire body to settle down into the surface beneath you, "
            "noticing how the weight of the day slowly dissolves into nothingness, "
            "as each inhale brings soothing energy and every exhale releases tension."
        )
        chunks = split_into_chunks(long_text, max_chars=160)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 180)
            self.assertTrue(c.endswith((".", "...", "—", "?", "!", ",")))

    def test_prepare_segments(self):
        script = (
            "Welcome to this peaceful meditation.\n\n"
            "[pause:4.0s]\n\n"
            "Gently close your eyes [breath] and begin to unwind."
        )
        segments = prepare_segments(script, content_type="meditation")
        types = [s["type"] for s in segments]
        self.assertIn("speech", types)
        self.assertIn("pause", types)
        self.assertIn("breath", types)


class TestChatterboxEngine(unittest.TestCase):
    def test_init_defaults(self):
        engine = ChatterboxEngine()
        # Broadcast-stable meditation parameters
        self.assertEqual(engine.exaggeration, 0.40)
        self.assertEqual(engine.cfg_weight, 0.50)
        self.assertEqual(engine.temperature, 0.75)
        self.assertEqual(engine.min_p, 0.05)
        self.assertEqual(engine.top_p, 1.0)
        self.assertEqual(engine.repetition_penalty, 1.20)
        self.assertEqual(engine.voice_slug, "Brittney")
        self.assertIn(engine.device, ("mps", "cuda", "cpu"))
        self.assertFalse(engine._loaded)

    def test_resolve_reference_audio(self):
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

    def test_condition_reference_audio(self):
        brittney_path = _resolve_reference_audio("Brittney")
        self.assertIsNotNone(brittney_path)
        cond_path = _condition_reference_audio(brittney_path)
        self.assertIsNotNone(cond_path)
        self.assertTrue(cond_path.endswith(".wav"))

        # Check conditioned audio stats
        import soundfile as sf
        audio, sr = sf.read(cond_path)
        self.assertEqual(sr, 24000)
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        self.assertLessEqual(peak, 0.92)
        # RMS should be close to -20 dBFS (approx 0.10 linear)
        self.assertAlmostEqual(20 * np.log10(rms), -20.0, delta=1.5)

    def test_trim_vocoder_silence_taper(self):
        # 1 sec audio with 0.2s silence at ends
        sr = 24000
        speech = np.sin(2 * np.pi * 440 * np.linspace(0, 0.6, int(0.6 * sr))).astype(np.float32)
        silence = np.zeros(int(0.2 * sr), dtype=np.float32)
        full = np.concatenate([silence, speech, silence])
        trimmed = _trim_vocoder_silence(full)
        self.assertLess(len(trimmed), len(full))
        # Head and tail should be softly tapered (first and last samples near 0)
        self.assertAlmostEqual(trimmed[0], 0.0, delta=0.05)
        self.assertAlmostEqual(trimmed[-1], 0.0, delta=0.05)

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
        fake_wav = torch.zeros((1, 24000), dtype=torch.float32)
        fake_model.generate.return_value = fake_wav
        engine.model = fake_model
        engine._loaded = True

        segments = [
            {"type": "pause", "duration_sec": 0.5},
            {"type": "speech", "text": "Take a slow breath."},
        ]

        with patch("core.chatterbox_tts.engine._condition_reference_audio") as mock_cond:
            mock_cond.return_value = "dummy_cond.wav"
            voice_audio, voice_activity = engine.synthesize(segments, speed=1.0)

        self.assertIsInstance(voice_audio, np.ndarray)
        self.assertIsInstance(voice_activity, np.ndarray)
        self.assertEqual(voice_audio.dtype, np.float32)
        self.assertEqual(voice_activity.dtype, bool)
        self.assertFalse(np.isnan(voice_audio).any())

    def test_unload_model(self):
        engine = ChatterboxEngine(device="cpu")
        engine.model = MagicMock()
        engine._loaded = True
        engine.unload_model()
        self.assertIsNone(engine.model)
        self.assertFalse(engine._loaded)


class TestChatterboxPostprocessor(unittest.TestCase):
    def test_mastering_engine_no_limiter(self):
        master = ChatterboxMasteringEngine()
        # Ensure no Pedalboard Limiter is inside the master chain (avoids +4.75dB static bug)
        for plugin in master._master_chain or []:
            self.assertNotIsInstance(plugin, Limiter)

        audio = np.random.uniform(-0.5, 0.5, 48000).astype(np.float32)
        out = master.master_vocals(audio, sr=48000)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), 48000)
        self.assertTrue(np.all(np.abs(out) <= 1.0))
        self.assertFalse(np.isnan(out).any())

    def test_dry_voice_chain(self):
        # reverb_amount = 0 should return empty transparent chain (no limiter, 100% dry)
        chain = build_chatterbox_voice_chain(reverb_amount=0.0)
        for plugin in chain:
            self.assertNotIsInstance(plugin, Limiter)
        audio = np.random.uniform(-0.5, 0.5, 48000).astype(np.float32)
        audio_2d = audio.reshape(1, -1)
        out = chain(audio_2d, 48000).squeeze(0)
        self.assertEqual(len(out), 48000)
        self.assertTrue(np.all(np.abs(out) <= 1.0))

    def test_abbey_road_reverb_chain(self):
        chain = build_chatterbox_voice_chain(reverb_amount=0.05, ir_name="warm_studio")
        for plugin in chain:
            self.assertNotIsInstance(plugin, Limiter)
        audio = np.random.uniform(-0.5, 0.5, 48000).astype(np.float32)
        audio_2d = audio.reshape(1, -1)
        out = chain(audio_2d, 48000).squeeze(0)
        self.assertEqual(len(out), 48000)
        self.assertTrue(np.all(np.abs(out) <= 1.0))
