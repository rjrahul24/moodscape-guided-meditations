"""Unit tests for AceStepEngine — extracted from test_music_engines.py."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from core.acestep_engine import AceStepEngine, TARGET_SAMPLE_RATE


class TestAceStepEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AceStepEngine()
        self.engine.initialized = True
        self.engine.model_type = "sft"
        self.engine._dit = MagicMock()
        self.engine._llm = MagicMock()

    def test_acestep_model_selection_and_steps(self):
        # Initial state (from setUp) is SFT
        self.assertEqual(self.engine.model_type, "sft")
        self.assertEqual(self.engine._get_inference_steps(is_repaint=False), 50)
        self.assertEqual(self.engine._get_inference_steps(is_repaint=True), 50)

        # Test switching to Turbo
        self.engine.load_model = MagicMock()
        self.engine._generate_single = MagicMock(return_value=np.zeros(10))
        self.engine.generate("Calm", 10.0, acestep_model_type="turbo")

        self.engine.load_model.assert_called_with(model_type="turbo")
        # Note: In real operation, load_model sets self.model_type.
        # For the test after mock, we set it manually to check steps logic.
        self.engine.model_type = "turbo"
        self.assertEqual(self.engine._get_inference_steps(is_repaint=False), 8)
        self.assertEqual(self.engine._get_inference_steps(is_repaint=True), 8)

    def test_acestep_crossfade_stages(self):
        sr = TARGET_SAMPLE_RATE  # 48 kHz (native ACE-Step rate)
        # 3 second segments
        s1 = np.ones(sr * 3, dtype=np.float32)
        s2 = np.ones(sr * 3, dtype=np.float32) * 0.5

        # 1-second crossfade
        res = AceStepEngine._crossfade_stages([s1, s2], crossfade_sec=1.0)

        # Total should be 3s + 3s - 1s = 5s
        self.assertEqual(res.shape[0], sr * 5)
        self.assertEqual(res.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
