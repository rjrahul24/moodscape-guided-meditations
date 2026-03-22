import unittest
from unittest.mock import patch, MagicMock
import torch

from core.kokoro_tts.voice_manager import (
    _slerp,
    slerp_blend,
    add_voice_jitter,
    blend_voices,
    get_voice,
    MEDITATION_PRESETS,
    is_british_voice,
)


def _make_voice(seed: int = 0) -> torch.Tensor:
    """Create a deterministic fake voice tensor (511, 1, 256)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(511, 1, 256, generator=gen)


class TestSlerp(unittest.TestCase):
    def test_endpoints(self):
        v0 = _make_voice(0)
        v1 = _make_voice(1)
        # t=0 → v0, t=1 → v1
        result_0 = _slerp(v0, v1, 0.0)
        self.assertTrue(torch.allclose(result_0, v0, atol=1e-4))
        result_1 = _slerp(v0, v1, 1.0)
        self.assertTrue(torch.allclose(result_1, v1, atol=1e-4))

    def test_midpoint_norm_preserved(self):
        v0 = _make_voice(0)
        v1 = _make_voice(1)
        mid = _slerp(v0, v1, 0.5)
        # SLERP midpoint norm should be between v0 and v1 norms
        norm_mid = torch.norm(mid.flatten()).item()
        norm_v0 = torch.norm(v0.flatten()).item()
        norm_v1 = torch.norm(v1.flatten()).item()
        self.assertGreater(norm_mid, min(norm_v0, norm_v1) * 0.9)
        self.assertLess(norm_mid, max(norm_v0, norm_v1) * 1.1)

    def test_shape_preserved(self):
        v0 = _make_voice(0)
        v1 = _make_voice(1)
        result = _slerp(v0, v1, 0.3)
        self.assertEqual(result.shape, (511, 1, 256))

    def test_identical_vectors(self):
        v = _make_voice(42)
        result = _slerp(v, v.clone(), 0.5)
        self.assertTrue(torch.allclose(result, v, atol=1e-4))


class TestSlerpBlend(unittest.TestCase):
    @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
    def test_single_voice(self, mock_load):
        mock_load.return_value = _make_voice(0)
        result = slerp_blend({"af_heart": 1.0})
        self.assertEqual(result.shape, (511, 1, 256))
        mock_load.assert_called_once_with("af_heart")

    @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
    def test_two_voices(self, mock_load):
        mock_load.side_effect = lambda vid: _make_voice(0 if vid == "af_heart" else 1)
        result = slerp_blend({"af_heart": 0.6, "af_sky": 0.4})
        self.assertEqual(result.shape, (511, 1, 256))

    @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
    def test_three_voices(self, mock_load):
        def _side(vid):
            seeds = {"af_heart": 0, "af_bella": 1, "af_nicole": 2}
            return _make_voice(seeds.get(vid, 3))
        mock_load.side_effect = _side
        result = slerp_blend({"af_heart": 0.5, "af_bella": 0.3, "af_nicole": 0.2})
        self.assertEqual(result.shape, (511, 1, 256))


class TestVoiceJitter(unittest.TestCase):
    def test_output_shape_matches(self):
        v = _make_voice(0)
        result = add_voice_jitter(v, amount=0.003)
        self.assertEqual(result.shape, v.shape)

    def test_small_perturbation(self):
        v = _make_voice(0)
        result = add_voice_jitter(v, amount=0.003)
        diff = torch.abs(result - v)
        # Max diff should be small (within ~5 sigma of 0.003)
        self.assertLess(diff.max().item(), 0.05)

    def test_different_each_call(self):
        v = _make_voice(0)
        r1 = add_voice_jitter(v, amount=0.003)
        r2 = add_voice_jitter(v, amount=0.003)
        self.assertFalse(torch.equal(r1, r2))


class TestGetVoice(unittest.TestCase):
    @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
    def test_preset_returns_tensor(self, mock_load):
        mock_load.return_value = _make_voice(0)
        result = get_voice("balanced_calm")
        self.assertIsInstance(result, torch.Tensor)

    def test_single_id_returns_string(self):
        result = get_voice("af_heart")
        self.assertEqual(result, "af_heart")

    @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
    def test_csv_blend_returns_tensor(self, mock_load):
        mock_load.return_value = _make_voice(0)
        result = get_voice("af_heart,af_nicole")
        self.assertIsInstance(result, torch.Tensor)


class TestBritishVoice(unittest.TestCase):
    def test_british_detected(self):
        self.assertTrue(is_british_voice("bf_emma"))
        self.assertTrue(is_british_voice("bm_george"))

    def test_american_not_british(self):
        self.assertFalse(is_british_voice("af_heart"))
        self.assertFalse(is_british_voice("am_adam"))


if __name__ == '__main__':
    unittest.main()
