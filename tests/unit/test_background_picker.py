"""Tests for random background-instrumental selection.

Builds on core.upload_music.scan_backgrounds; a fake scan keeps these fast,
since the real scanner reads audio headers from disk.
"""

import random
import unittest

from core.background_picker import pick_background

FAKE_TRACKS = [
    ("Autumn Sky — 12:00", "/bg/autumn.mp3"),
    ("Healing Forest — 23:12", "/bg/forest.mp3"),
    ("Piano Dreamcloud — 18:40", "/bg/piano.mp3"),
]


def fake_scan(tracks=FAKE_TRACKS):
    return lambda: list(tracks)


class TestBackgroundPicker(unittest.TestCase):
    def test_returns_a_label_and_path_pair(self):
        label, path = pick_background(scan=fake_scan())
        self.assertIn((label, path), FAKE_TRACKS)

    def test_is_reproducible_under_a_seed(self):
        first = pick_background(scan=fake_scan(), rng=random.Random(42))
        second = pick_background(scan=fake_scan(), rng=random.Random(42))
        self.assertEqual(first, second)

    def test_exclude_is_honoured(self):
        _label, path = pick_background(
            scan=fake_scan(), exclude=["/bg/autumn.mp3", "/bg/forest.mp3"]
        )
        self.assertEqual(path, "/bg/piano.mp3")

    def test_excluding_everything_falls_back_to_full_pool(self):
        _label, path = pick_background(
            scan=fake_scan(), exclude=[p for _, p in FAKE_TRACKS]
        )
        self.assertIn(path, [p for _, p in FAKE_TRACKS])

    def test_empty_library_raises_with_a_helpful_message(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            pick_background(scan=lambda: [])
        self.assertIn("backgrounds", str(ctx.exception).lower())

    def test_defaults_to_the_real_scanner(self):
        # The real library has 20 tracks committed; this guards the wiring.
        label, path = pick_background()
        self.assertTrue(label)
        self.assertTrue(path.endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")))


if __name__ == "__main__":
    unittest.main()
