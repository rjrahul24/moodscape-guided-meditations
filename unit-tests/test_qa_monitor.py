import unittest
import numpy as np
from core.qa_monitor import check_silence_gaps, check_clipping, run_qa_checks

class TestQAMonitor(unittest.TestCase):
    def test_check_silence_gaps(self):
        sr = 24000
        # 2 seconds of silence
        audio = np.zeros(2 * sr, dtype=np.float32)
        
        # Max silence is 15s. Should be empty
        issues = check_silence_gaps(audio, sr, max_silence_sec=15.0)
        self.assertEqual(len(issues), 0)
        
        # If max silence is 1.0s, should detect
        issues = check_silence_gaps(audio, sr, max_silence_sec=1.0)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["type"], "long_silence")
        self.assertAlmostEqual(issues[0]["duration_sec"], 2.0, delta=0.1)

    def test_check_clipping(self):
        # Normal audio
        audio_ok = np.array([0.1, -0.5, 0.8, -0.8], dtype=np.float32)
        res_ok = check_clipping(audio_ok, threshold=0.99)
        self.assertTrue(res_ok["passed"])
        self.assertEqual(res_ok["clipped_samples"], 0)

        # Clipped audio
        audio_clip = np.array([0.1, 1.0, -1.0, 0.8], dtype=np.float32)
        res_clip = check_clipping(audio_clip, threshold=0.99)
        self.assertFalse(res_clip["passed"])
        self.assertEqual(res_clip["clipped_samples"], 2)

    def test_run_qa_checks_integration(self):
        # A simple pulse that isn't clipped and isn't silent
        sr = 24000
        audio = np.random.uniform(-0.5, 0.5, sr).astype(np.float32)
        # Using a higher threshold to avoid false positive for silence
        results = run_qa_checks(audio, sr, log_results=False)
        self.assertTrue(results["clipping"]["passed"])
        self.assertEqual(len(results["silence"]), 0)
        self.assertIn("lufs", results)

if __name__ == '__main__':
    unittest.main()
