import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from core.stem_separator import StemSeparator

class TestStemSeparator(unittest.TestCase):
    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_remove_drums_and_vocals(self, mock_apply, mock_get_model):
        
        # Mock the loaded model and its sources
        mock_model = MagicMock()
        # The typical htdemucs order: drums, bass, other, vocals
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model
        
        # Audio length
        length = 44100
        
        # Mock apply_model to return a predictable tensor
        # Shape: (batch=1, num_sources=4, channels=2, time=length)
        mock_stems = torch.zeros((1, 4, 2, length), dtype=torch.float32)
        # Drums are on index 0, bass on 1, other on 2, vocals on 3.
        # Let's make bass=1.0 and other=0.5.
        mock_stems[0, 1, :, :] = 1.0 # Bass
        mock_stems[0, 2, :, :] = 0.5 # Other
        # The rest are 0.0
        
        mock_apply.return_value = mock_stems
        
        separator = StemSeparator()
        # Input audio (mono, will be upmixed to stereo inside then downmixed)
        input_audio = np.random.uniform(-0.1, 0.1, length).astype(np.float32)
        
        out = separator.remove_drums_and_vocals(input_audio, sample_rate=44100)
        
        # We expect the output to be the sum of bass and other: 1.0 + 0.5 = 1.5
        # And downmixed to mono via mean across channels. 
        # But both channels were 1.5 so mean is 1.5.
        self.assertEqual(out.shape, (length,))
        self.assertAlmostEqual(out.mean().item(), 1.5, places=5)
        
if __name__ == '__main__':
    unittest.main()
