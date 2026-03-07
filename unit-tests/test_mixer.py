import unittest
import numpy as np
from core.mixer import (
    apply_rms_ducking,
    apply_mask_ducking,
    overlay_tracks,
    apply_fades,
    normalize_loudness,
    resample_for_export
)

class TestMixer(unittest.TestCase):
    def test_apply_rms_ducking(self):
        sr = 24000
        voice = np.random.uniform(-0.1, 0.1, sr).astype(np.float32)
        music = np.ones(sr, dtype=np.float32)
        out = apply_rms_ducking(voice, music, sample_rate=sr)
        self.assertEqual(out.shape, music.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_apply_mask_ducking(self):
        sr = 24000
        activity = np.zeros(sr, dtype=bool)
        activity[12000:15000] = True # Spoken segment
        music = np.ones(sr, dtype=np.float32)
        
        out = apply_mask_ducking(activity, music, sample_rate=sr, duck_amount_db=-10.0)
        self.assertEqual(out.shape, music.shape)
        # Check that ducking happened somewhere (audio < 1.0)
        self.assertTrue(np.any(out < 1.0))

    def test_overlay_tracks(self):
        sr = 24000
        voice = np.zeros(sr * 2, dtype=np.float32) # 2 seconds
        music = np.ones(sr * 1, dtype=np.float32) # 1 second
        # Will add 2 seconds of pre_roll silence to voice by default
        aligned_v, aligned_m = overlay_tracks(voice, music, music_pre_roll_sec=2.0, sample_rate=sr)
        
        # Total length should be 4 seconds (2s pre roll + 2s voice)
        self.assertEqual(aligned_v.shape[0], sr * 4)
        self.assertEqual(aligned_v.shape, aligned_m.shape)

    def test_apply_fades(self):
        sr = 24000
        audio = np.ones(sr * 2, dtype=np.float32) # 2 secs
        out = apply_fades(audio, sr, fade_in_sec=1.0, fade_out_sec=1.0)
        self.assertEqual(out.shape, audio.shape)
        self.assertEqual(out[0], 0.0) # start of fade in
        self.assertEqual(out[-1], 0.0) # end of fade out

    def test_normalize_loudness(self):
        # Just ensure array comes back right shape without error
        audio = np.ones(24000, dtype=np.float32)
        out = normalize_loudness(audio, target_lufs=-14.0)
        self.assertEqual(out.shape, audio.shape)

    def test_resample_for_export(self):
        audio = np.zeros(24000, dtype=np.float32)
        out = resample_for_export(audio, source_rate=24000, target_rate=44100)
        self.assertEqual(out.shape[0], 44100)

if __name__ == '__main__':
    unittest.main()
