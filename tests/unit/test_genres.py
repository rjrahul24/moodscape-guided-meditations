"""Tests for the genre pack registry."""

import random
import tempfile
import unittest
from pathlib import Path

from core.genres import (
    GENRE_PACKS_DIR,
    Angle,
    GenrePack,
    GenrePackError,
    genre_choices,
    load_all_packs,
    load_pack,
    pick_angle,
)

VALID = """
label        = "Grief & Loss"
family       = "Emotional"
content_type = "meditation"
music_tags   = ["warm", "sparse"]
pause_ratio  = 0.34

technique = "RAIN, held loosely."
arc = ["arrival", "body", "memory", "kindness", "return"]
safety = "No stage models. Do not imply closure."
banned = ["time heals", "move on"]

[[angles]]
name    = "the empty chair"
imagery = ["a chair by a window", "afternoon light"]

[[angles]]
name    = "tidal"
imagery = ["a shoreline at dusk", "wet sand"]
"""


class LoadPackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, body: str) -> None:
        (self.dir / f"{name}.toml").write_text(body, encoding="utf-8")

    def test_a_valid_pack_loads(self):
        self._write("grief_and_loss", VALID)
        pack = load_pack("grief_and_loss", packs_dir=self.dir)
        self.assertIsInstance(pack, GenrePack)
        self.assertEqual(pack.slug, "grief_and_loss")
        self.assertEqual(pack.label, "Grief & Loss")
        self.assertEqual(pack.content_type, "meditation")
        self.assertEqual(pack.music_tags, ("warm", "sparse"))
        self.assertEqual(len(pack.angles), 2)
        self.assertIsInstance(pack.angles[0], Angle)

    def test_a_missing_pack_names_the_path(self):
        with self.assertRaises(GenrePackError) as ctx:
            load_pack("nope", packs_dir=self.dir)
        self.assertIn("nope", str(ctx.exception))

    def test_a_missing_required_field_is_rejected(self):
        self._write("bad", VALID.replace('family       = "Emotional"', ""))
        with self.assertRaises(GenrePackError) as ctx:
            load_pack("bad", packs_dir=self.dir)
        self.assertIn("family", str(ctx.exception))

    def test_an_unknown_content_type_is_rejected(self):
        self._write("bad", VALID.replace('"meditation"', '"podcast"'))
        with self.assertRaises(GenrePackError) as ctx:
            load_pack("bad", packs_dir=self.dir)
        self.assertIn("podcast", str(ctx.exception))

    def test_an_unknown_music_tag_is_rejected(self):
        self._write("bad", VALID.replace('["warm", "sparse"]', '["funky"]'))
        with self.assertRaises(GenrePackError) as ctx:
            load_pack("bad", packs_dir=self.dir)
        self.assertIn("funky", str(ctx.exception))

    def test_an_out_of_range_pause_ratio_is_rejected(self):
        self._write("bad", VALID.replace("0.34", "0.95"))
        with self.assertRaises(GenrePackError):
            load_pack("bad", packs_dir=self.dir)

    def test_fewer_than_two_angles_is_rejected(self):
        body = VALID.split("[[angles]]")[0] + """
[[angles]]
name    = "only one"
imagery = ["a thing"]
"""
        self._write("bad", body)
        with self.assertRaises(GenrePackError) as ctx:
            load_pack("bad", packs_dir=self.dir)
        self.assertIn("angles", str(ctx.exception))

    def test_duplicate_angle_names_are_rejected(self):
        self._write("bad", VALID.replace('"tidal"', '"the empty chair"'))
        with self.assertRaises(GenrePackError):
            load_pack("bad", packs_dir=self.dir)


class ShippedPacksTest(unittest.TestCase):
    """Every pack in the repository must load. A malformed pack cannot ship."""

    def test_all_shipped_packs_load_and_validate(self):
        packs = load_all_packs()
        self.assertEqual(len(packs), 46)
        for slug, pack in packs.items():
            self.assertEqual(pack.slug, slug)
            self.assertTrue(pack.technique.strip(), msg=slug)
            self.assertTrue(pack.safety.strip(), msg=slug)
            self.assertGreaterEqual(len(pack.arc), 3, msg=slug)
            self.assertGreaterEqual(len(pack.angles), 2, msg=slug)

    def test_labels_are_unique(self):
        labels = [pack.label for pack in load_all_packs().values()]
        self.assertEqual(len(labels), len(set(labels)))

    def test_genre_choices_group_by_family(self):
        choices = genre_choices()
        families = [family for family, _entries in choices]
        self.assertEqual(len(families), len(set(families)))
        total = sum(len(entries) for _family, entries in choices)
        self.assertEqual(total, 46)

    def test_packs_dir_exists(self):
        self.assertTrue(GENRE_PACKS_DIR.is_dir())


class PickAngleTest(unittest.TestCase):
    def _pack(self, names: list[str]) -> GenrePack:
        return GenrePack(
            slug="s", label="L", family="F", content_type="meditation",
            music_tags=("warm",), pause_ratio=0.3, technique="t",
            arc=("a", "b", "c"), safety="s", banned=(),
            angles=tuple(Angle(name=n, imagery=("x",)) for n in names),
        )

    def test_excludes_recent_angles(self):
        pack = self._pack(["one", "two", "three"])
        for _ in range(30):
            self.assertEqual(
                pick_angle(pack, recent=["one", "two"]).name, "three"
            )

    def test_falls_back_to_all_angles_when_everything_is_recent(self):
        pack = self._pack(["one", "two"])
        angle = pick_angle(pack, recent=["one", "two"])
        self.assertIn(angle.name, {"one", "two"})

    def test_is_reproducible_with_a_seeded_rng(self):
        pack = self._pack(["one", "two", "three", "four"])
        first = pick_angle(pack, rng=random.Random(7)).name
        second = pick_angle(pack, rng=random.Random(7)).name
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
