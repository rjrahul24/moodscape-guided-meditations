import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from core.stem_separator import StemSeparator

class TestStemSeparator(unittest.TestCase):
    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_remove_drums_and_vocals_internal(self, mock_apply, mock_get_model):
        """Test the in-process separation logic (called by subprocess worker)."""

        # Mock the loaded model and its sources
        mock_model = MagicMock()
        # The typical htdemucs order: drums, bass, other, vocals
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model

        # Audio length — use 44100 Hz so no resampling is needed
        length = 44100

        # Mock apply_model to return a predictable tensor
        # Shape: (batch=1, num_sources=4, channels=2, time=length)
        mock_stems = torch.zeros((1, 4, 2, length), dtype=torch.float32)
        # Drums are on index 0, bass on 1, other on 2, vocals on 3.
        mock_stems[0, 1, :, :] = 1.0  # Bass
        mock_stems[0, 2, :, :] = 0.5  # Other
        mock_apply.return_value = mock_stems

        separator = StemSeparator()
        input_audio = np.random.uniform(-0.1, 0.1, length).astype(np.float32)

        out = separator._remove_drums_and_vocals_internal(input_audio, sample_rate=44100)

        # We expect the output to be the sum of bass and other: 1.0 + 0.5 = 1.5
        # Downmixed to mono via mean across channels → still 1.5.
        self.assertEqual(out.shape, (length,))
        self.assertAlmostEqual(out.mean().item(), 1.5, places=5)

    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_apply_model_called_with_split(self, mock_apply, mock_get_model):
        """Verify apply_model is called with split=True and shifts=0 for memory efficiency."""

        mock_model = MagicMock()
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model

        length = 44100
        mock_stems = torch.zeros((1, 4, 2, length), dtype=torch.float32)
        mock_apply.return_value = mock_stems

        separator = StemSeparator()
        input_audio = np.zeros(length, dtype=np.float32)

        separator._remove_drums_and_vocals_internal(input_audio, sample_rate=44100)

        # Check that apply_model was called with memory-efficient parameters
        mock_apply.assert_called_once()
        call_kwargs = mock_apply.call_args
        self.assertTrue(call_kwargs.kwargs.get('split', False))
        self.assertEqual(call_kwargs.kwargs.get('shifts', 1), 0)
        self.assertEqual(call_kwargs.kwargs.get('device'), 'cpu')

if __name__ == '__main__':
    unittest.main()
