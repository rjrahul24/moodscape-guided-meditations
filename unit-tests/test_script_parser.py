import unittest
from core.script_parser import parse_script

class TestScriptParser(unittest.TestCase):
    def test_parse_empty(self):
        self.assertEqual(parse_script(""), [])
        self.assertEqual(parse_script("   \n "), [])

    def test_parse_only_text(self):
        script = "This is a simple script."
        segments = parse_script(script)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["type"], "speech")
        self.assertEqual(segments[0]["text"], "This is a simple script.")

    def test_parse_with_explicit_pause(self):
        script = "Hello. [pause:3s] Welcome."
        segments = parse_script(script)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["text"], "Hello.")
        self.assertEqual(segments[1]["type"], "pause")
        self.assertEqual(segments[1]["duration_sec"], 3.0)
        self.assertEqual(segments[2]["text"], "Welcome.")
        
    def test_parse_with_float_pause(self):
        script = "Softly... [pause:2.5s] breathe."
        segments = parse_script(script)
        self.assertEqual(segments[1]["duration_sec"], 2.5)

    def test_parse_double_newlines_as_pauses(self):
        script = "Line one.\n\nLine two."
        segments = parse_script(script)
        # Double newline is replaced by [pause:2.5s]
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[1]["type"], "pause")
        self.assertEqual(segments[1]["duration_sec"], 2.5)
        self.assertEqual(segments[2]["text"], "Line two.")

    def test_parse_consecutive_pauses(self):
        script = "Breathe. [pause:2s] [pause:3s]\n\nRelax."
        segments = parse_script(script)
        # Consecutive pauses should be merged.
        # [pause:2s] + [pause:3s] + [pause:2.5s] = 7.5s
        
        # Actually let's check exact behavior:
        # text "Breathe."
        # pause 2.0
        # pause 3.0 -> merged -> 5.0
        # \n\n is 2.5 -> merged -> 7.5
        # text "Relax."
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["text"], "Breathe.")
        self.assertEqual(segments[1]["type"], "pause")
        self.assertEqual(segments[1]["duration_sec"], 7.5)
        self.assertEqual(segments[2]["text"], "Relax.")

if __name__ == '__main__':
    unittest.main()
