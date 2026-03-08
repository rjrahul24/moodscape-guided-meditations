import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.music_engine import MusicEngine
from core.acestep_engine import AceStepEngine

class TestMusicEngines(unittest.TestCase):
    def setUp(self):
        self.engine = AceStepEngine()
        self.engine.initialized = True
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
        # SEGMENT_DURATION = 30, CONTEXT_DURATION = 10, NET_NEW = 20
        # 30s -> 1 seq
        self.assertEqual(engine._num_segments(30.0), 1)
        # 50s -> 30 + 20 -> 2 seq
        self.assertEqual(engine._num_segments(50.0), 2)
        # 60s -> 30 + 20 + 20(part) -> 3 seq
        self.assertEqual(engine._num_segments(60.0), 3)

    # --- ACE-Step Tests ---
    @patch("soundfile.write")
    @patch("tempfile.mkstemp")
    @patch("os.remove")
    @patch("os.path.exists")
    @patch("os.close")
    def test_acestep_reference_audio_handling(self, mock_close, mock_exists, mock_remove, mock_mkstemp, mock_sf_write):
        mock_mkstemp.return_value = (999, "/tmp/fake_ref.wav")
        mock_exists.return_value = True
        
        dummy_audio = np.zeros(100)
        self.engine._generate_single = MagicMock(return_value=np.zeros(200))
        
        self.engine.generate("Calm", 10.0, melody_audio=dummy_audio, melody_sample_rate=24000)
        
        # Verify temp file was "created"
        mock_sf_write.assert_called_once()
        self.assertEqual(mock_sf_write.call_args[0][0], "/tmp/fake_ref.wav")
        
        # Verify path was passed to _generate_single
        self.engine._generate_single.assert_called_once()
        kwargs = self.engine._generate_single.call_args[1]
        self.assertEqual(kwargs["reference_audio_path"], "/tmp/fake_ref.wav")
        
        # Verify cleanup
        mock_remove.assert_called_once_with("/tmp/fake_ref.wav")

    def test_acestep_enhance_prompt(self):
        base_prompt = "Warm sleep music"
        caption, lyrics = AceStepEngine._enhance_prompt(base_prompt)
        
        # Caption checks
        self.assertIn("Deep meditation", caption)
        self.assertIn(base_prompt.lower(), caption)
        self.assertIn("no percussion", caption.lower())
        # Anti-sawtooth check
        self.assertIn("rounded harmonic profile", caption)
        
        # Lyrics checks
        self.assertIn("[Instrumental]", lyrics)
        self.assertIn("[warm]", lyrics)

    def test_acestep_sanitize_prompt(self):
        dirty = "60 bpm sawtooth synth in C Major"
        sanitized = AceStepEngine._sanitize_prompt(dirty)
        self.assertNotIn("60 bpm", sanitized)
        self.assertNotIn("sawtooth", sanitized)
        self.assertNotIn("c major", sanitized)
        self.assertEqual(sanitized, "synth in") # "synth in" remains

    def test_acestep_extract_lyrics_tags(self):
        prompt = "dreamy ethereal music for sleep"
        tags = AceStepEngine._extract_lyrics_tags(prompt)
        self.assertIn("[Instrumental]", tags)
        self.assertIn("[dreamy]", tags)
        self.assertIn("[ethereal]", tags)
        
        prompt_simple = "pure silence"
        tags_simple = AceStepEngine._extract_lyrics_tags(prompt_simple)
        self.assertEqual(tags_simple, "[Instrumental]")

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
