import unittest
from core.kokoro_tts.preprocessor import (
    expand_for_tts,
    preprocess_for_meditation,
    annotate_speed,
    clamp_speed,
    _convert_to_contractions,
    _inject_sensory_ellipses,
    _vary_sentence_lengths,
)


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


class TestContractions(unittest.TestCase):
    def test_basic_contractions(self):
        self.assertIn("you're", _convert_to_contractions("you are relaxing"))
        self.assertIn("don't", _convert_to_contractions("do not worry"))
        self.assertIn("it's", _convert_to_contractions("it is safe"))
        self.assertIn("let's", _convert_to_contractions("let us begin"))
        self.assertIn("can't", _convert_to_contractions("cannot hold"))
        self.assertIn("there's", _convert_to_contractions("there is peace"))

    def test_case_insensitive(self):
        result = _convert_to_contractions("You are welcome")
        self.assertIn("you're", result.lower())

    def test_no_partial_matches(self):
        # "dare" should not match "you are" boundary
        result = _convert_to_contractions("dare to breathe")
        self.assertEqual(result, "dare to breathe")


class TestSensoryEllipses(unittest.TestCase):
    def test_inserts_ellipsis_before_sensory_word(self):
        result = _inject_sensory_ellipses("Feel the warmth spreading.")
        self.assertIn("the... warmth", result)

    def test_determiner_your(self):
        result = _inject_sensory_ellipses("Notice your peace within.")
        self.assertIn("your... peace", result)

    def test_no_ellipsis_without_determiner(self):
        result = _inject_sensory_ellipses("Feel warmth spreading.")
        self.assertNotIn("...", result)

    def test_no_double_ellipsis(self):
        # Already has punctuation before — should not add another
        result = _inject_sensory_ellipses("Feel the, warmth spreading.")
        self.assertNotIn("... warmth", result)

    def test_multiple_sensory_words(self):
        result = _inject_sensory_ellipses("the calm and the stillness")
        self.assertIn("the... calm", result)
        self.assertIn("the... stillness", result)


class TestSentenceLengthVariation(unittest.TestCase):
    def test_breaks_long_clause(self):
        text = (
            "Take a very deep breath in through your nose slowly "
            "and with gentle attention, and let it flow out."
        )
        result = _vary_sentence_lengths(text)
        # Should have promoted the comma before "and" to a period
        self.assertIn(". And", result)

    def test_no_break_short_clause(self):
        text = "Breathe in, and exhale."
        result = _vary_sentence_lengths(text)
        # Clause is too short — no break
        self.assertNotIn(". And", result)

    def test_max_one_break(self):
        text = (
            "one two three four five six seven eight nine ten, and "
            "eleven twelve thirteen fourteen fifteen sixteen, and done."
        )
        result = _vary_sentence_lengths(text)
        # count=1 in the regex ensures only one break
        self.assertEqual(result.count(". And"), 1)


class TestAnnotateSpeed(unittest.TestCase):
    def test_short_phrase_slower(self):
        speed = annotate_speed("Be still.", 0.90)
        self.assertAlmostEqual(speed, 0.90 * 0.88, places=3)

    def test_question_slower(self):
        speed = annotate_speed("What do you notice in this moment?", 0.90)
        self.assertAlmostEqual(speed, 0.90 * 0.95, places=3)

    def test_ellipsis_slower(self):
        # Must be >= 6 words to avoid triggering short-phrase rule first
        speed = annotate_speed("Allow yourself to gently drift away...", 0.90)
        self.assertAlmostEqual(speed, 0.90 * 0.92, places=3)

    def test_default_unchanged(self):
        speed = annotate_speed("Breathe deeply and allow your body to relax.", 0.90)
        self.assertEqual(speed, 0.90)

    def test_clamps_to_floor(self):
        # Very slow base speed * 0.88 would be below 0.65
        speed = annotate_speed("Rest.", 0.70)
        self.assertGreaterEqual(speed, 0.65)

    def test_clamp_speed_floor(self):
        self.assertEqual(clamp_speed(0.50), 0.65)
        self.assertEqual(clamp_speed(0.65), 0.65)
        self.assertEqual(clamp_speed(0.90), 0.90)


if __name__ == '__main__':
    unittest.main()
