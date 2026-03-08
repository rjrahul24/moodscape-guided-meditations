import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from core.acestep_engine import AceStepEngine, TARGET_SAMPLE_RATE

class TestAceStepInfinite(unittest.TestCase):
    def setUp(self):
        self.engine = AceStepEngine()
        self.engine.initialized = True
        self.engine.model_type = "sft"
        self.engine._dit = MagicMock()
        self.engine._llm = MagicMock()
        self.engine._dit.initialize_service.return_value = ("Success", True)
    @patch("core.acestep_engine.AceStepEngine._generate_single")
    @patch("core.acestep_engine.AceStepEngine._generate_single_repaint")
    @patch("soundfile.write")
    def test_generate_infinite_logic(self, mock_sf_write, mock_repaint, mock_single):
        # Mock segments: 60s for first, then 30s extensions (actually repaint returns 60s)
        # We'll make them all ones/zeroes to distinguish
        first_seg = np.ones(int(60.0 * TARGET_SAMPLE_RATE), dtype=np.float32)
        second_seg = np.ones(int(60.0 * TARGET_SAMPLE_RATE), dtype=np.float32) * 0.5
        
        mock_single.return_value = first_seg
        mock_repaint.return_value = second_seg
        
        # Test 80s generation (Should take 1 base + 1 repaint)
        prompt = "test prompt"
        duration = 80.0
        
        # Wait, AceStepEngine.generate triggers infinite if > 90.0. 
        # Let's test with 120s to ensure it enters the loop.
        duration = 120.0 
        
        result = self.engine.generate(prompt, duration)
        
        # Verify calls
        self.assertEqual(mock_single.call_count, 1)
        # Remaining after 60s is 60s. Each loop adds 30s. So 2 loops.
        self.assertEqual(mock_repaint.call_count, 2)
        # Check that metadata was passed to single (default 70, Auto, None for ref)
        mock_single.assert_any_call(
            unittest.mock.ANY, unittest.mock.ANY, 
            lyrics=unittest.mock.ANY, bpm=70, keyscale="Auto", 
            reference_audio_path=None
        )
        
        # Total length should be approx 120s (minus tiny crossfade bits if overlaps were used)
        # Our implementation uses a tiny overlap (0.1s) per loop.
        # 60 + 30 + 30 - 0.1 - 0.1 = 119.8s
        expected_len = 120.0 * TARGET_SAMPLE_RATE - (2 * 0.1 * TARGET_SAMPLE_RATE)
        self.assertAlmostEqual(len(result) / TARGET_SAMPLE_RATE, 119.8, places=1)

    def test_generate_single_switch(self):
        # Verify it switches correctly based on duration
        with patch.object(AceStepEngine, "_generate_single") as mock_single, \
             patch.object(AceStepEngine, "_generate_infinite") as mock_infinite:
            
            mock_single.return_value = np.zeros(100)
            mock_infinite.return_value = np.zeros(100)
            
            # Case 1: 60s -> single
            self.engine.generate("test", 60.0)
            mock_single.assert_called_once()
            mock_infinite.assert_not_called()
            
            mock_single.reset_mock()
            mock_infinite.reset_mock()
            
            # Case 2: 120s -> infinite
            self.engine.generate("test", 120.0)
            mock_single.assert_not_called()
            mock_infinite.assert_called_once()

if __name__ == "__main__":
    unittest.main()
