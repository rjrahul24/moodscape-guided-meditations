import unittest
import os
import shutil
import tempfile
import sys
import numpy as np

from core.pipeline import MeditationPipeline

class TestIntegrationModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prevent MPS fault on teardown
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_instrumental_only(self):
        pipeline = MeditationPipeline()

        # Pick the first available background track for the upload engine.
        from core.upload_music import scan_backgrounds
        backgrounds = scan_backgrounds()
        if not backgrounds:
            self.skipTest("No background tracks in assets/backgrounds/")
        uploaded_path = backgrounds[0][1]

        # 0.1m = 6 seconds of music, keeps test relatively swift but real
        out_path, status = pipeline.generate(
            script="",
            music_prompt="Calm piano, test",
            generation_mode="Instrumental Only",
            instrumental_duration_m=0.1,
            music_model="upload",
            uploaded_music_path=uploaded_path,
            output_format="wav",
            stem_separation=False
        )
        self.assertTrue(os.path.exists(out_path))
        self.assertTrue(out_path.endswith(".wav"))
        
    def test_vocals_only(self):
        pipeline = MeditationPipeline()
        out_path, status = pipeline.generate(
            script="Hello.", 
            music_prompt="", 
            generation_mode="Vocals Only",
            tts_engine="kokoro",
            output_format="wav",
            stem_separation=False 
        )
        self.assertTrue(os.path.exists(out_path))
        self.assertTrue(out_path.endswith(".wav"))

if __name__ == "__main__":
    unittest.main()
