# tests/unit/test_eval_ab.py
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.eval.ab import build_pairs, cut_excerpts, find_case_audio, level_gain, level_match, pair_gain_scale, score, sign_test_p
from core.eval.mix_metrics import loudness

SR = 8000


def _noise(seconds: float, amp: float, seed: int) -> np.ndarray:
    return (amp * np.random.default_rng(seed).standard_normal(int(SR * seconds))).astype(np.float32)


def _make_run(root: Path, cases: dict[str, float]) -> Path:
    for case, amp in cases.items():
        d = root / case
        d.mkdir(parents=True)
        sf.write(d / "med_1.wav", _noise(300, amp, hash(case) % 1000), SR)
        (d / "med_1.meta.json").write_text("{}")
        sf.write(d / "narration_stem.wav", _noise(1, amp, 1), SR)  # must be ignored
    return root


class TestExcerpts(unittest.TestCase):
    def test_three_positions(self):
        parts = cut_excerpts(_noise(300, 0.1, 0), SR, excerpt_sec=75)
        self.assertEqual([p[0] for p in parts], ["start", "middle", "end"])
        self.assertTrue(all(len(p[1]) == 75 * SR and not p[2] for p in parts))

    def test_short_file_excerpt_truncated(self):
        parts = cut_excerpts(_noise(60, 0.1, 0), SR, excerpt_sec=75)
        self.assertTrue(all(p[2] for p in parts))
        self.assertTrue(all(0 < len(p[1]) <= 60 * SR for p in parts))

    def test_level_match(self):
        matched = level_match(_noise(20, 0.3, 0), SR, target_lufs=-20.0)
        self.assertAlmostEqual(loudness(matched, SR)["integrated_lufs_dual_mono"], -20.0, delta=0.1)


class TestPairs(unittest.TestCase):
    def test_find_case_audio_ignores_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp) / "a", {"body_scan-short": 0.1})
            self.assertEqual(find_case_audio(run)["body_scan-short"].name, "med_1.wav")

    def test_missing_case_raises_with_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _make_run(Path(tmp) / "a", {"x-short": 0.1, "y-short": 0.1})
            b = _make_run(Path(tmp) / "b", {"x-short": 0.1})
            with self.assertRaisesRegex(ValueError, "y-short"):
                build_pairs(a, b, Path(tmp) / "ab")

    def test_build_pairs_is_blind_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _make_run(Path(tmp) / "a", {"x-short": 0.1, "y-short": 0.2})
            b = _make_run(Path(tmp) / "b", {"x-short": 0.3, "y-short": 0.05})
            n = build_pairs(a, b, Path(tmp) / "ab1", seed=7, excerpt_sec=30)
            build_pairs(a, b, Path(tmp) / "ab2", seed=7, excerpt_sec=30)
            self.assertEqual(n, 6)
            key1 = json.loads((Path(tmp) / "ab1" / ".key.json").read_text())
            key2 = json.loads((Path(tmp) / "ab2" / ".key.json").read_text())
            self.assertEqual(key1, key2)
            self.assertEqual({k["A"] for k in key1["pairs"].values()} | {k["B"] for k in key1["pairs"].values()},
                             {"baseline", "challenger"})
            with open(Path(tmp) / "ab1" / "sheet.csv") as fh:
                header = next(csv.reader(fh))
            self.assertEqual(header, ["pair", "choice", "reason"])  # no arm/case leakage


class TestScore(unittest.TestCase):
    def test_sign_test_thresholds(self):
        self.assertLess(sign_test_p(17, 24), 0.05)
        self.assertGreater(sign_test_p(16, 24), 0.05)

    def _scored(self, choices: list[str]) -> dict:
        tmp = Path(tempfile.mkdtemp())
        pairs = {f"{i:02d}": {"A": "challenger", "B": "baseline", "case": "c", "position": "start"}
                 for i in range(len(choices))}
        (tmp / "key.json").write_text(json.dumps({"pairs": pairs}))
        with open(tmp / "sheet.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pair", "choice", "reason"])
            for i, c in enumerate(choices):
                w.writerow([f"{i:02d}", c, ""])
        return score(tmp)

    def test_ship(self):
        result = self._scored(["A"] * 17 + ["B"] * 7)
        self.assertEqual(result["verdict"], "SHIP")
        self.assertEqual(result["challenger_wins"], 17)

    def test_ties_excluded_and_inconclusive(self):
        result = self._scored(["A"] * 10 + ["B"] * 8 + ["same"] * 6)
        self.assertEqual(result["ties"], 6)
        self.assertEqual(result["verdict"], "INCONCLUSIVE")

    def test_no_ship_when_baseline_clearly_wins(self):
        self.assertEqual(self._scored(["B"] * 18 + ["A"] * 6)["verdict"], "NO-SHIP")


class TestLevelMatchNoClip(unittest.TestCase):
    def test_reduces_gain_instead_of_clipping(self):
        quiet_peaky = _noise(20, 0.01, 0)
        quiet_peaky[100] = 0.9  # high crest factor: loudness gain would push this past 1.0
        out = level_match(quiet_peaky, SR, target_lufs=-20.0)
        self.assertLessEqual(float(np.max(np.abs(out))), 0.98 + 1e-6)
        self.assertGreater(float(np.max(np.abs(out))), 0.97)  # scaled, not flattened

    def test_pair_keeps_relative_level(self):
        a = _noise(20, 0.01, 0)
        a[100] = 0.9
        b = _noise(20, 0.05, 1)
        scale = pair_gain_scale(a, b, SR)
        self.assertLess(scale, 1.0)
        ma = level_match(a, SR, gain_scale=scale)
        mb = level_match(b, SR, gain_scale=scale)
        want = level_gain(a, SR) / level_gain(b, SR)
        got = (np.max(np.abs(ma)) / np.max(np.abs(a))) / (np.max(np.abs(mb)) / np.max(np.abs(b)))
        self.assertAlmostEqual(float(got), want, places=3)
        self.assertLessEqual(float(np.max(np.abs(mb))), 0.98 + 1e-6)


class TestRandomSeedAndHiddenKey(unittest.TestCase):
    def test_default_seed_is_random_and_recorded_and_key_is_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _make_run(Path(tmp) / "a", {"x-short": 0.1})
            b = _make_run(Path(tmp) / "b", {"x-short": 0.2})
            build_pairs(a, b, Path(tmp) / "ab1", excerpt_sec=30)
            build_pairs(a, b, Path(tmp) / "ab2", excerpt_sec=30)
            seeds = [json.loads((Path(tmp) / d / ".key.json").read_text())["seed"] for d in ("ab1", "ab2")]
            self.assertNotEqual(seeds[0], seeds[1])
            self.assertFalse((Path(tmp) / "ab1" / "key.json").exists())


class TestScoreValidation(unittest.TestCase):
    _scored = TestScore._scored
    def test_invalid_choice_raises(self):
        with self.assertRaisesRegex(ValueError, "01.*maybe"):
            self._scored(["A", "maybe"])

    def test_blank_is_unanswered_and_incomplete(self):
        result = self._scored(["A"] * 20 + ["B"] * 3 + [""])
        self.assertEqual(result["unanswered"], 1)
        self.assertEqual(result["verdict"], "INCOMPLETE")

    def test_tie_spellings(self):
        result = self._scored(["A", "Tie", "=", " SAME "])
        self.assertEqual(result["ties"], 3)
        self.assertEqual(result["unanswered"], 0)

    def test_unknown_pair_raises(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "key.json").write_text(json.dumps({"pairs": {}}))
        (tmp / "sheet.csv").write_text("pair,choice,reason\n07,A,\n")
        with self.assertRaisesRegex(ValueError, "07"):
            score(tmp)


class TestSheetCompleteness(unittest.TestCase):
    def _dir(self, sheet: str, n_keys: int = 3, key_name: str = ".key.json") -> Path:
        tmp = Path(tempfile.mkdtemp())
        pairs = {f"{i:02d}": {"A": "challenger", "B": "baseline", "case": "c", "position": "start"}
                 for i in range(n_keys)}
        (tmp / key_name).write_text(json.dumps({"pairs": pairs}))
        (tmp / "sheet.csv").write_text(sheet)
        return tmp

    def test_missing_ids_count_as_unanswered(self):
        result = score(self._dir("pair,choice,reason\n00,A,\n"))
        self.assertEqual(result["unanswered"], 2)
        self.assertEqual(result["verdict"], "INCOMPLETE")

    def test_duplicate_ids_raise(self):
        with self.assertRaisesRegex(ValueError, "01"):
            score(self._dir("pair,choice,reason\n00,A,\n01,A,\n01,B,\n02,A,\n"))

    def test_missing_columns_raise(self):
        with self.assertRaisesRegex(ValueError, "choice"):
            score(self._dir("pair,reason\n00,\n"))

    def test_legacy_key_json_still_read(self):
        result = score(self._dir("pair,choice,reason\n00,A,\n01,A,\n02,B,\n", key_name="key.json"))
        self.assertEqual(result["challenger_wins"], 2)


class TestTruncatedPairs(unittest.TestCase):
    def test_short_case_single_pair_equal_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _make_run(Path(tmp) / "a", {"x-short": 0.1})
            b = _make_run(Path(tmp) / "b", {"x-short": 0.2})
            sf.write(b / "x-short" / "med_1.wav", _noise(50, 0.2, 3), SR)  # shorter than a (300 s)
            n = build_pairs(a, b, Path(tmp) / "ab", excerpt_sec=75)
            self.assertEqual(n, 1)
            key = json.loads((Path(tmp) / "ab" / ".key.json").read_text())["pairs"]
            self.assertEqual(key["00"]["position"], "whole")
            self.assertTrue(key["00"]["truncated"])
            la = len(sf.read(Path(tmp) / "ab" / "pairs" / "00_A.wav")[0])
            lb = len(sf.read(Path(tmp) / "ab" / "pairs" / "00_B.wav")[0])
            self.assertEqual(la, lb)
            self.assertEqual(la, 50 * SR)


class TestShuffleAndPositions(unittest.TestCase):
    def _runs(self, tmp):
        cases = {"c0": 0.1, "c1": 0.1, "c2": 0.1}
        a = _make_run(Path(tmp) / "a", cases)
        b = _make_run(Path(tmp) / "b", {k: 0.2 for k in cases})
        return a, b, Path(tmp) / "ab"

    def _key(self, out):
        return json.loads((out / ".key.json").read_text())["pairs"]

    def test_pair_order_is_shuffled_but_key_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, chal, out = self._runs(tmp)
            n = build_pairs(base, chal, out, seed=1, excerpt_sec=30)
            key = self._key(out)
            self.assertEqual(n, 9)
            self.assertEqual(sorted({v["case"] for v in key.values()}), ["c0", "c1", "c2"])
            self.assertEqual(sorted(key), [f"{i:02d}" for i in range(9)])
            first_three = [key[f"{i:02d}"]["case"] for i in range(3)]
            self.assertGreater(len(set(first_three)), 1)

    def test_positions_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, chal, out = self._runs(tmp)
            n = build_pairs(base, chal, out, seed=1, excerpt_sec=30, positions=["start", "middle"])
            self.assertEqual(n, 6)
            self.assertEqual({v["position"] for v in self._key(out).values()}, {"start", "middle"})

    def test_unknown_position_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, chal, out = self._runs(tmp)
            with self.assertRaises(ValueError):
                build_pairs(base, chal, out, seed=1, positions=["nope"])

    def test_positions_filter_keeps_whole_pair_for_short_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, chal, out = self._runs(tmp)
            n = build_pairs(base, chal, out, seed=1, excerpt_sec=400, positions=["start"])
            self.assertEqual(n, 3)
            self.assertEqual({v["position"] for v in self._key(out).values()}, {"whole"})


class TestLeanScoring(unittest.TestCase):
    def _ab(self, roles, choices):
        tmp = Path(tempfile.mkdtemp())
        pairs = {f"{i:02d}": {"A": r, "B": "baseline" if r == "challenger" else "challenger",
                              "case": "c", "position": "start"} for i, r in enumerate(roles)}
        (tmp / ".key.json").write_text(json.dumps({"pairs": pairs}))
        with open(tmp / "sheet.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pair", "choice", "reason"])
            for i, c in enumerate(choices):
                w.writerow([f"{i:02d}", c, ""])
        return tmp

    def test_leans_are_ties_for_verdict_and_reported(self):
        ab = self._ab(["challenger", "baseline", "challenger"], ["leanA", " lean B ", "A"])
        r = score(ab)
        self.assertEqual((r["challenger_wins"], r["baseline_wins"], r["ties"]), (1, 0, 2))
        self.assertEqual((r["challenger_leans"], r["baseline_leans"]), (2, 0))
        self.assertAlmostEqual(r["weighted_rate"], (1 + 1.0) / (1 + 0 + 1.0))
        self.assertEqual(r["lean_inclusive_p"], round(sign_test_p(3, 3), 4))
        self.assertEqual(r["by_position"]["start"]["challenger_leans"], 2)
        self.assertEqual(r["by_position"]["start"]["baseline_leans"], 0)

    def test_weighted_rate_zero_when_no_decisions(self):
        r = score(self._ab(["challenger"], ["same"]))
        self.assertEqual(r["weighted_rate"], 0.0)

    def test_invalid_choice_message_lists_leans(self):
        with self.assertRaisesRegex(ValueError, "leanA"):
            score(self._ab(["challenger"], ["maybe"]))


if __name__ == "__main__":
    unittest.main()
