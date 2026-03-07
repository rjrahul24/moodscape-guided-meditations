import unittest
import numpy as np
from core.audio_processor import (
    make_voice_chain,
    make_music_chain,
    make_acestep_music_chain,
    make_master_chain,
    apply_fx,
    resample_to_44100,
    upsample_audio
)

class TestAudioProcessor(unittest.TestCase):
    def test_chains_creation(self):
        # Ensure all chains instantiate without error
        vc = make_voice_chain()
        self.assertIsNotNone(vc)
        mc = make_music_chain()
        self.assertIsNotNone(mc)
        amc = make_acestep_music_chain()
        self.assertIsNotNone(amc)
        mac = make_master_chain()
        self.assertIsNotNone(mac)

    def test_apply_fx_1d(self):
        # A simple array 24000 samples (1 sec)
        audio = np.random.uniform(-0.5, 0.5, 24000).astype(np.float32)
        chain = make_voice_chain(reverb_amount=0.0)
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
