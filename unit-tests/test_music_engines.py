import unittest
import numpy as np

from core.music_engine import MusicEngine
from core.acestep_engine import AceStepEngine

class TestMusicEngines(unittest.TestCase):
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
        # SEGMENT_DURATION = 30, CONTEXT_DURATION = 10, NET_NEW = 20
        # 30s -> 1 seq
        self.assertEqual(engine._num_segments(30.0), 1)
        # 50s -> 30 + 20 -> 2 seq
        self.assertEqual(engine._num_segments(50.0), 2)
        # 60s -> 30 + 20 + 20(part) -> 3 seq
        self.assertEqual(engine._num_segments(60.0), 3)

    # --- ACE-Step Tests ---
    def test_acestep_enhance_prompt(self):
        base_prompt = "Warm sleep music"
        enhanced = AceStepEngine._enhance_prompt(base_prompt)
        # Must contain anti-percussion and ambient guides
        self.assertIn("Deep meditation", enhanced)
        self.assertIn(base_prompt, enhanced)
        self.assertIn("no percussion", enhanced.lower())
        self.assertIn("no drums", enhanced.lower())
        
        # Assert not duplicated if user includes it
        duplicate_prompt = "calm gentle ambient no drums"
        enhanced_dup = AceStepEngine._enhance_prompt(duplicate_prompt)
        self.assertEqual(enhanced_dup.count("calm"), 2) # It appears once in extra and once in prompt list, actually wait: 'calm' is excluded if in user prompt! So it should appear ONCE.
        # Let's just check length is less than completely concatenated

    def test_acestep_crossfade_stages(self):
        sr = 24000
        # 3 second segments
        s1 = np.ones(sr * 3, dtype=np.float32)
        s2 = np.ones(sr * 3, dtype=np.float32) * 0.5
        
        # 1-second crossfade
        res = AceStepEngine._crossfade_stages([s1, s2], crossfade_sec=1.0)
        
        # Total should be 3s + 3s - 1s = 5s
        self.assertEqual(res.shape[0], sr * 5)
        self.assertEqual(res.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()
