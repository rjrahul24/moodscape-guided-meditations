# tests/unit/test_qwen3_engine.py
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.qwen3_tts.engine import Qwen3Engine
from core.qwen3_tts.postprocessor import condition_chunk, time_stretch


def _result(seconds=1.0, sr=24000, amp=0.3):
    t = np.arange(int(sr * seconds)) / sr
    return SimpleNamespace(audio=(amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sample_rate=sr)


def _engine(mode, model):
    with patch.dict(os.environ, {"MOODSCAPE_QWEN3_MODE": mode}):
        eng = Qwen3Engine(voice_slug="Brittney", loader=lambda repo: model)
    eng.load_model()
    return eng


SEGS = [{"type": "speech", "text": "Rest here.", "voice": None},
        {"type": "pause", "duration_sec": 1.0},
        {"type": "speech", "text": "Feel your hands.", "voice": None}]


class TestPostprocessor(unittest.TestCase):
    def test_trims_silence_and_resamples(self):
        sr = 48000
        tone = _result(1.0, sr).audio
        padded = np.concatenate([np.zeros(sr // 2, np.float32), tone, np.zeros(sr // 2, np.float32)])
        out = condition_chunk(padded, sr)
        self.assertAlmostEqual(len(out) / 24000, 1.0, delta=0.05)

    def test_stretch_identity(self):
        x = _result().audio
        self.assertIs(time_stretch(x, 1.0), x)


class TestQwen3Engine(unittest.TestCase):
    def test_clone_mode_never_passes_instruct(self):
        model = MagicMock()
        model.generate.return_value = [_result()]
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_INSTRUCT": "Whisper."}):
            _engine("clone", model).synthesize(SEGS)
        kwargs = model.generate.call_args.kwargs
        self.assertNotIn("instruct", kwargs)
        self.assertIn("ref_audio", kwargs)
        self.assertTrue(kwargs["ref_text"])

    def test_custom_mode_uses_speaker_and_instruct(self):
        model = MagicMock()
        model.generate_custom_voice.return_value = [_result()]
        _engine("custom", model).synthesize(SEGS, content_type="sleep_story")
        kwargs = model.generate_custom_voice.call_args.kwargs
        self.assertIn("bedtime", kwargs["instruct"].lower())
        self.assertEqual(kwargs["language"], "English")
        self.assertIn("speaker", kwargs)

    def test_design_mode_uses_description(self):
        model = MagicMock()
        model.generate_voice_design.return_value = [_result()]
        _engine("design", model).synthesize(SEGS)
        self.assertIn("narrator", model.generate_voice_design.call_args.kwargs["instruct"].lower())

    def test_contract_and_pause_length(self):
        model = MagicMock()
        model.generate.return_value = [_result(1.0)]
        audio, activity = _engine("clone", model).synthesize(SEGS)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(activity.dtype, np.bool_)
        self.assertEqual(len(audio), len(activity))
        self.assertGreaterEqual(len(audio) / 24000, 2.9)  # two ~1 s chunks + 1 s pause (edge trim shaves <1 ms each)

    def test_multi_result_and_resample(self):
        model = MagicMock()
        model.generate.return_value = [_result(0.5, 48000), _result(0.5, 48000)]
        audio, _ = _engine("clone", model).synthesize(SEGS[:1])
        self.assertAlmostEqual(len(audio) / 24000, 1.0, delta=0.1)

    def test_seed_progression(self):
        model = MagicMock()
        model.generate.return_value = [_result()]
        with patch("core.qwen3_tts.engine._seed") as seed:
            _engine("clone", model).synthesize(SEGS, seed=10)
        self.assertEqual([c.args[0] for c in seed.call_args_list], [10, 11])

    def test_empty_output_retries_then_raises(self):
        model = MagicMock()
        model.generate.return_value = [SimpleNamespace(audio=np.zeros(10, np.float32), sample_rate=24000)]
        with self.assertRaisesRegex(RuntimeError, "Rest here"):
            _engine("clone", model).synthesize(SEGS[:1], seed=5)
        self.assertEqual(model.generate.call_count, 2)

    def test_missing_weights_names_repo(self):
        def broken(repo):
            raise OSError("not found")
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_MODE": "clone"}):
            eng = Qwen3Engine(voice_slug="Brittney", loader=broken)
        with self.assertRaisesRegex(RuntimeError, "Qwen3-TTS"):
            eng.load_model()

    def test_logs_rtf(self):
        model = MagicMock()
        model.generate.return_value = [_result()]
        eng = _engine("clone", model)
        with self.assertLogs("moodscape.qwen3_tts", "INFO") as cm:
            eng.synthesize(SEGS)
        self.assertTrue(any("RTF" in line for line in cm.output))

    def test_unload_clears_model(self):
        eng = _engine("clone", MagicMock())
        eng.unload_model()
        self.assertIsNone(eng._model)


if __name__ == "__main__":
    unittest.main()
