import os
import unittest
from unittest.mock import patch

from core.qwen3_tts.preprocessor import normalize_for_qwen3, prepare_segments, split_into_chunks

LONG = " ".join(f"This is calm sentence number {i} for the listener." for i in range(12))


class TestQwen3Preprocessor(unittest.TestCase):
    def test_pauses_and_breaths_preserved(self):
        segs = prepare_segments("Rest here. [pause:4s] [breath] Feel your hands.")
        types = [s["type"] for s in segs]
        self.assertIn("pause", types)
        self.assertIn("breath", types)
        self.assertEqual(next(s for s in segs if s["type"] == "pause")["duration_sec"], 4.0)

    def test_paragraph_pause_follows_content_type(self):
        med = prepare_segments("One line.\n\nTwo line.", content_type="meditation")
        sleep = prepare_segments("One line.\n\nTwo line.", content_type="sleep_story")
        med_p = next(s for s in med if s["type"] == "pause")["duration_sec"]
        sleep_p = next(s for s in sleep if s["type"] == "pause")["duration_sec"]
        self.assertGreater(med_p, sleep_p)

    def test_chunks_respect_limit_at_sentence_boundaries(self):
        chunks = split_into_chunks(normalize_for_qwen3(LONG), max_chars=200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 200 for c in chunks))
        self.assertTrue(all(c.endswith(".") for c in chunks))

    def test_single_overlong_sentence_is_not_dropped(self):
        sentence = "word " * 80 + "end."
        chunks = split_into_chunks(sentence.strip(), max_chars=200)
        self.assertEqual(" ".join(chunks).split(), sentence.split())

    def test_env_chunk_size_used(self):
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_CHUNK_CHARS": "80"}):
            speech = [s for s in prepare_segments(LONG) if s["type"] == "speech"]
        self.assertTrue(all(len(s["text"]) <= 80 for s in speech))

    def test_normalization_matches_f5_rules(self):
        self.assertNotIn("...", normalize_for_qwen3("Breathe... and rest."))
        self.assertNotIn("RELAX", normalize_for_qwen3("Now RELAX fully."))


if __name__ == "__main__":
    unittest.main()
