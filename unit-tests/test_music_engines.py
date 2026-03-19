import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

from core.music_engine import MusicEngine
from core.acestep_engine import AceStepEngine

class TestMusicEngines(unittest.TestCase):
    def setUp(self):
        self.engine = AceStepEngine()
        self.engine.initialized = True
        self.engine.model_type = "sft"
        self.engine._dit = MagicMock()
        self.engine._llm = MagicMock()

    # --- MusicGen Tests ---
    def test_musicgen_stage_prompt(self):
        stages = [
            ("Prompt 1", 30.0),
            ("Prompt 2", 30.0),
            ("Prompt 3", 30.0)
        ]
        self.assertEqual(MusicEngine._stage_prompt_for_time(0.0, stages), "Prompt 1")
        self.assertEqual(MusicEngine._stage_prompt_for_time(15.0, stages), "Prompt 1")
        self.assertEqual(MusicEngine._stage_prompt_for_time(45.0, stages), "Prompt 2")
        self.assertEqual(MusicEngine._stage_prompt_for_time(100.0, stages), "Prompt 3")

    def test_musicgen_num_segments(self):
        engine = MusicEngine()
        # SEGMENT_DURATION = 30
        # Default extend_stride = 12.0
        
        # 30s -> 1 seq
        self.assertEqual(engine._num_segments(30.0, extend_stride=12.0), 1)
        # 42s -> 30 + 12 -> 2 seq
        self.assertEqual(engine._num_segments(42.0, extend_stride=12.0), 2)
        # 50s -> 30 + 12 + 12(part) -> 3 seq
        self.assertEqual(engine._num_segments(50.0, extend_stride=12.0), 3)

    @patch("audiocraft.models.MusicGen.get_pretrained")
    def test_musicgen_parameter_passing(self, mock_get_pretrained):
        # Setup mock model
        mock_model = MagicMock()
        mock_get_pretrained.return_value = mock_model
        
        engine = MusicEngine()
        engine.load_model()
        
        # Mock generate methods to return blank audio
        mock_model.generate.return_value = torch.zeros((1, 1, 32000 * 30))
        
        # Call generate with custom parameters
        engine.generate(
            "test prompt", 
            10.0, 
            temperature=0.5, 
            top_k=50, 
            cfg_coef=5.0,
            top_p=0.1
        )
        
        # Verify set_generation_params was called with correct values
        mock_model.set_generation_params.assert_called_with(
            duration=30,
            use_sampling=True,
            top_k=50,
            top_p=0.1,
            temperature=0.5,
            cfg_coef=5.0
        )



    # --- _check_spectral_flux Tests ---

    def test_spectral_flux_rejects_silence(self):
        """A pure-silence tensor must be REJECTED (return False).

        This guards against the 'Silence Acceptance Bug': if a silent segment
        were accepted it would be fed as audio context to the next continuation
        window, cascading blank audio through the rest of the track.
        """
        silent = torch.zeros(1, 32000 * 30)  # 30s of silence at native SR
        result = MusicEngine._check_spectral_flux(silent)
        self.assertFalse(
            result,
            "_check_spectral_flux() must return False for a silent segment "
            "so the retry loop regenerates it instead of propagating silence.",
        )

    def test_spectral_flux_accepts_valid_audio(self):
        """Band-limited Gaussian noise (ambient music proxy) must be ACCEPTED (return True).

        A pure sine wave has a highly periodic STFT spectrum that creates an
        extremely large peak/median flux ratio and falsely triggers the hallucination
        guard.  Gaussian noise spreads energy broadly across all frequency bins,
        making the per-frame flux variation small relative to the median — a much
        better proxy for real ambient music output from MusicGen.
        """
        torch.manual_seed(42)
        # 30s of random Gaussian noise at the native sample rate, scaled to a
        # reasonable amplitude (similar to MusicGen's typical output level).
        noise = torch.randn(1, 32000 * 30) * 0.3
        result = MusicEngine._check_spectral_flux(noise)
        self.assertTrue(
            result,
            "_check_spectral_flux() must return True for clean non-silent audio.",
        )

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
        from core.acestep_engine import TARGET_SAMPLE_RATE
        sr = TARGET_SAMPLE_RATE  # 48 kHz (native ACE-Step rate)
        # 3 second segments
        s1 = np.ones(sr * 3, dtype=np.float32)
        s2 = np.ones(sr * 3, dtype=np.float32) * 0.5

        # 1-second crossfade
        res = AceStepEngine._crossfade_stages([s1, s2], crossfade_sec=1.0)

        # Total should be 3s + 3s - 1s = 5s
        self.assertEqual(res.shape[0], sr * 5)
        self.assertEqual(res.dtype, np.float32)

    def test_musicgen_default_params(self):
        """Default temperature and top_k must match the documented ambient meditation sweet spots.

        This test pins the defaults using inspect.signature so that any future
        accidental reversion (e.g. temperature=0.7, top_k=80) is immediately caught.
        Reference: docs/model_implementation_guides/musicgen.md §7 Tuning Guide.
        """
        import inspect
        sig = inspect.signature(MusicEngine.generate)
        self.assertEqual(
            sig.parameters["temperature"].default, 0.87,
            "temperature default must be 0.87 (ambient sweet spot per docs)",
        )
        self.assertEqual(
            sig.parameters["top_k"].default, 250,
            "top_k default must be 250 (full ambient vocabulary breadth per docs)",
        )

if __name__ == '__main__':
    unittest.main()
