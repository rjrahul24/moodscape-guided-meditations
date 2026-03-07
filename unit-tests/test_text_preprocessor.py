import unittest
from core.text_preprocessor import expand_for_tts, preprocess_for_meditation, validate_chunk_length

class TestTextPreprocessor(unittest.TestCase):
    def test_expand_abbreviations(self):
        text = "Wait 5 sec and 10 min."
        expanded = expand_for_tts(text)
        self.assertTrue("seconds" in expanded)
        self.assertTrue("minutes" in expanded)
        
        freq = "Target is 432 Hz."
        self.assertEqual("Target is four hundred and thirty-two hertz.", expand_for_tts(freq))

    def test_expand_numbers(self):
        self.assertEqual(expand_for_tts("I have 0 apples"), "I have zero apples")
        self.assertEqual(expand_for_tts("Count to 3"), "Count to three")
        self.assertEqual(expand_for_tts("Level 42"), "Level forty-two")
        self.assertEqual(expand_for_tts("Over 999"), "Over nine hundred and ninety-nine")

    def test_preprocess_for_meditation(self):
        text = "Breathe in  [pause:2s] exhale."
        processed = preprocess_for_meditation(text)
        # Verify ellipsis was added before pause and double spaces fixed
        self.assertIn("in... [pause:2s]", processed)
        self.assertNotIn("  ", processed)

    def test_validate_chunk_length(self):
        # A short chunk should remain a single chunk
        short_text = "This is a short sentence."
        chunks = validate_chunk_length(short_text, max_tokens=200)
        self.assertEqual(len(chunks), 1)

        # A long sequence that exceeds 200 tokens but has sentences
        long_sentence = "This is a test. " * 50
        chunks = validate_chunk_length(long_sentence, max_tokens=50)
        self.assertGreater(len(chunks), 1)
        
        # Verify no chunk exceeds a large threshold (hard ceiling logic)
        for chunk in chunks:
            estimated_tokens = int(len(chunk.split()) * 1.3)
            # The hard ceiling is 400. The test max_tokens is 50.
            # So it should be close to or under 50 per chunk.
            self.assertLessEqual(estimated_tokens, 50 * 2) # Account for single sentence potential overflow
            
if __name__ == '__main__':
    unittest.main()
