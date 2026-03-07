import unittest
from core.kokoro_tts.preprocessor import expand_for_tts, preprocess_for_meditation

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

if __name__ == '__main__':
    unittest.main()
