import unittest
import os
import tempfile
import shutil

from core.pipeline import MeditationPipeline

STRESS_SCRIPT = (
    "Welcome to this extended deep relaxation session. [pause:3s]\n\n"
    "Find a position that feels completely comfortable... [pause:5s]\n\n"
    "Take a slow breath in through your nose... [pause:4s] "
    "and release gently through your mouth. [pause:6s]\n\n"
    "Now simply rest in this complete stillness. [pause:15s]\n\n"
    "There is nothing to do. Nowhere to go. [pause:10s]\n\n"
    "When you are ready, gently begin to return. [pause:5s]\n\n"
    "Slowly open your eyes. Thank you for this practice. [pause:3s]\n"
)

class TestStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        cls.test_dir = tempfile.mkdtemp()
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    @unittest.skip("Stress tests take ~15 minutes and involve heavy generational load. Run manually.")
    def test_full_15_minute_generation(self):
        pipeline = MeditationPipeline()
        out_path, status = pipeline.generate(
            script=STRESS_SCRIPT,
            music_prompt="slow ambient pads, warm synths, no drums, peaceful",
            voice="golden_hour",
            speed=0.78,
            output_format="wav",
            seed=42,
            instrumental_duration_m=15.0 # Just to stress it
        )
        self.assertTrue(os.path.exists(out_path))
        
if __name__ == "__main__":
    unittest.main()
