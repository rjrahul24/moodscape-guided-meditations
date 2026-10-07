# tests/integration/test_qwen3_live.py
import os
import unittest

import numpy as np

SCRIPT = (
    "Find a comfortable position, and let your body settle. [pause:3s] "
    "Notice the gentle weight of your hands. [pause:4s] "
    "There is nowhere else you need to be right now."
)


@unittest.skipUnless(os.environ.get("MOODSCAPE_QWEN3_LIVE") == "1", "set MOODSCAPE_QWEN3_LIVE=1 to run")
class TestQwen3Live(unittest.TestCase):
    def _render(self, mode: str):
        from unittest.mock import patch
        from core.qwen3_tts.engine import Qwen3Engine
        from core.qwen3_tts.preprocessor import prepare_segments
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_MODE": mode}):
            engine = Qwen3Engine(voice_slug="Brittney")
            engine.load_model()
            try:
                return engine.synthesize(prepare_segments(SCRIPT), seed=42)
            finally:
                engine.unload_model()

    def _check(self, audio, activity):
        self.assertFalse(np.isnan(audio).any())
        self.assertGreater(np.abs(audio).max(), 0.01)
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        self.assertLess(zcr, 0.4)  # same noise heuristic as the pipeline's vocal sanity check
        seconds = len(audio) / 24000
        self.assertGreater(seconds, 7.0 + 4.0)    # pauses alone are 7 s
        self.assertLess(seconds, 7.0 + 30.0)
        self.assertTrue(activity.any())

    def test_clone(self):
        self._check(*self._render("clone"))

    def test_custom(self):
        self._check(*self._render("custom"))

    def test_design(self):
        self._check(*self._render("design"))
