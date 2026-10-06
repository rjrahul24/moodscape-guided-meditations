import tempfile
import unittest
from pathlib import Path

from core.eval.testset import DEFAULT_TESTSET, TestCase, load_testset


class TestLoadTestset(unittest.TestCase):
    def test_default_testset_has_24_unique_cases(self):
        cases = load_testset()
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({c.id for c in cases}), 24)
        self.assertTrue(DEFAULT_TESTSET.is_file())

    def test_case_fields(self):
        first = load_testset()[0]
        self.assertEqual(first, TestCase("anxiety_relief-short", "anxiety_relief", "short", 1100))

    def test_seed_offset_per_band(self):
        cases = {c.id: c for c in load_testset()}
        self.assertEqual(cases["anxiety_relief-medium"].seed, 1101)

    def test_unknown_band_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.toml"
            path.write_text('bands = ["huge"]\n[[cases]]\ngenre = "body_scan"\nseed = 1\n')
            with self.assertRaisesRegex(ValueError, "huge"):
                load_testset(path)

    def test_unknown_genre_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.toml"
            path.write_text('bands = ["short"]\n[[cases]]\ngenre = "no_such_genre"\nseed = 1\n')
            with self.assertRaises(Exception):
                load_testset(path)


if __name__ == "__main__":
    unittest.main()
