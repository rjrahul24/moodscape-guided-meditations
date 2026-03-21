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
        """Verify the ACE-Step chain contains all 11 expected effects."""
        chain = make_acestep_music_chain()
        # 11 effects: NoiseGate, HPF, LowShelf, PeakFilter(3kHz), PeakFilter(4kHz),
        #             PeakFilter(6kHz), HighShelf(8kHz), HighShelf(10kHz),
        #             LowpassFilter(16kHz), Compressor, Limiter
        self.assertEqual(len(chain), 11, f"Expected 11 effects, got {len(chain)}")

    def test_acestep_chain_midrange_and_air_filters(self):
        """Verify 3 kHz midrange cut, 6 kHz gap fill, and 8 kHz air shelf filters."""
        from pedalboard import PeakFilter, HighShelfFilter
        chain = make_acestep_music_chain()

        # Find the 3 kHz PeakFilter (primary AI artifact zone)
        peak_3k = [p for p in chain if isinstance(p, PeakFilter)
                   and abs(p.cutoff_frequency_hz - 3000) < 1]
        self.assertEqual(len(peak_3k), 1, "Missing PeakFilter at 3000 Hz")
        self.assertAlmostEqual(peak_3k[0].gain_db, -4.5, places=1)

        # Find the 6 kHz PeakFilter (5-7 kHz gap fill)
        peak_6k = [p for p in chain if isinstance(p, PeakFilter)
                   and abs(p.cutoff_frequency_hz - 6000) < 1]
        self.assertEqual(len(peak_6k), 1, "Missing PeakFilter at 6000 Hz")
        self.assertAlmostEqual(peak_6k[0].gain_db, -2.0, places=1)

        # Find the 8 kHz HighShelfFilter (gentle air)
        shelf_8k = [p for p in chain if isinstance(p, HighShelfFilter)
                    and abs(p.cutoff_frequency_hz - 8000) < 1]
        self.assertEqual(len(shelf_8k), 1, "Missing HighShelfFilter at 8000 Hz")
        self.assertAlmostEqual(shelf_8k[0].gain_db, 0.5, places=1)

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
