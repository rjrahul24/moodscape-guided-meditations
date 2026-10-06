import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("MOODSCAPE_EVAL_LIVE") == "1", "set MOODSCAPE_EVAL_LIVE=1 to run")
class TestWhisperLive(unittest.TestCase):
    def test_real_whisper_on_tts_fixture(self):
        from core.eval.speech_metrics import WhisperTranscriber, speech_metrics
        from core.kokoro_tts.engine import KokoroEngine
        from core.kokoro_tts.preprocessor import prepare_segments

        text = "Find a comfortable position and let your shoulders soften."
        engine = KokoroEngine()
        engine.load_model()
        audio, _ = engine.synthesize(prepare_segments(text), voice="balanced_calm", speed=0.9)
        engine.unload_model()
        transcriber = WhisperTranscriber()
        metrics = speech_metrics(audio, 24000, text, transcriber=transcriber)
        transcriber.unload()
        self.assertLess(metrics["cer"], 0.10, metrics["transcript"])
        self.assertTrue(np.isfinite(metrics["cer"]))
