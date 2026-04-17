import unittest
import numpy as np
from core.audio_processor import (
    make_heartmula_music_chain,
    make_acestep_music_chain,
    make_lyria_music_chain,
    make_vocal_pocket_chain,
    make_master_chain,
    apply_fx,
    resample_to_44100,
    upsample_audio,
)


class TestAudioProcessor(unittest.TestCase):
    def test_chains_creation(self):
        # Ensure all chains instantiate without error
        hmc = make_heartmula_music_chain()
        self.assertIsNotNone(hmc)
        amc = make_acestep_music_chain()
        self.assertIsNotNone(amc)
        lmc = make_lyria_music_chain()
        self.assertIsNotNone(lmc)
        vpc = make_vocal_pocket_chain()
        self.assertIsNotNone(vpc)
        mac = make_master_chain()
        self.assertIsNotNone(mac)

    def test_acestep_chain_has_expected_effects(self):
        """Verify the ACE-Step chain contains all expected effects."""
        import os
        from core.audio_processor import IR_CATALOG
        chain = make_acestep_music_chain()
        # 9 base effects: NoiseGate, HPF, LowShelf(200Hz), PeakFilter(2500Hz),
        #                 PeakFilter(4500Hz), HighShelf(8kHz), LowpassFilter(16kHz),
        #                 Compressor, Limiter
        # + 1 Convolution reverb when warm_studio IR file is present on disk
        ir_path = IR_CATALOG.get("warm_studio", {}).get("path", "")
        expected = 10 if os.path.isfile(ir_path) else 9
        self.assertEqual(len(chain), expected, f"Expected {expected} effects, got {len(chain)}")

    def test_acestep_chain_midrange_and_air_filters(self):
        """Verify 2.5 kHz mid-cut is correctly configured."""
        from pedalboard import PeakFilter
        chain = make_acestep_music_chain()

        # Find the 2.5 kHz PeakFilter (creates space for voice)
        peak_mid = [p for p in chain if isinstance(p, PeakFilter)
                    and abs(p.cutoff_frequency_hz - 2500) < 100]
        self.assertEqual(len(peak_mid), 1, "Missing PeakFilter near 2500 Hz")
        self.assertAlmostEqual(peak_mid[0].gain_db, -2.0, places=1)

    def test_acestep_chain_signal_path(self):
        """Smoke test: 48 kHz signal through ACE-Step chain produces valid output."""
        chain = make_acestep_music_chain()
        # 1 second of pink-ish noise at 48 kHz
        rng = np.random.default_rng(42)
        audio = rng.uniform(-0.3, 0.3, 48000).astype(np.float32)
        out = apply_fx(audio, chain, sample_rate=48000)
        self.assertEqual(out.shape, audio.shape)
        self.assertFalse(np.isnan(out).any(), "NaN in output")
        self.assertTrue(np.all(np.abs(out) <= 1.0), "Clipping in output")

    def test_apply_fx_1d(self):
        # A simple array 24000 samples (1 sec)
        audio = np.random.uniform(-0.5, 0.5, 24000).astype(np.float32)
        chain = make_heartmula_music_chain()
        out = apply_fx(audio, chain, sample_rate=24000)
        self.assertEqual(audio.shape, out.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_resample_to_44100(self):
        sr_in = 24000
        # 1 sec
        audio = np.ones(sr_in, dtype=np.float32)
        out = resample_to_44100(audio, sr_in)
        # Should be converted to 44100 shape
        self.assertEqual(out.shape[0], 44100)

        # Test pass-through if already 44100
        out_again = resample_to_44100(out, 44100)
        self.assertEqual(out.shape, out_again.shape)

    def test_upsample_audio(self):
        audio = np.zeros(24000, dtype=np.float32)
        out = upsample_audio(audio, from_sr=24000, to_sr=48000)
        self.assertEqual(out.shape[0], 48000)
        self.assertEqual(out.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
