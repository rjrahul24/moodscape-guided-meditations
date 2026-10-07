# core/eval/ab.py
"""Blind A/B listening: excerpt cutting, level matching, pairing, scoring.

Convention: run A is the baseline, run B the challenger. On disk the arms are
shuffled per pair into files labelled A/B; only the hidden .key.json knows which is which.
"""

import csv
import json
import math
import random
import secrets
from pathlib import Path

import numpy as np

from core.eval.mix_metrics import loudness

POSITIONS = (("start", 0.10), ("middle", 0.50), ("end", 0.85))
SHIP_RATE = 0.60
SHIP_P = 0.05
TIE_CHOICES = ("SAME", "TIE", "=")
KEY_NAME = ".key.json"
LEGACY_KEY_NAME = "key.json"
PEAK_LIMIT = 0.98


def find_case_audio(run_dir: Path) -> dict[str, Path]:
    cases: dict[str, Path] = {}
    for case_dir in sorted(p for p in Path(run_dir).iterdir() if p.is_dir()):
        mixes = [w for w in case_dir.glob("*.wav") if w.with_suffix(".meta.json").is_file()]
        if mixes:
            cases[case_dir.name] = mixes[0]
    return cases


def cut_excerpts(audio: np.ndarray, sr: int, *, excerpt_sec: float = 75.0) -> list[tuple[str, np.ndarray, bool]]:
    n = len(audio)
    length = int(excerpt_sec * sr)
    truncated = n < length
    out = []
    for name, frac in POSITIONS:
        if truncated:
            out.append((name, audio.copy(), True))
            continue
        start = min(max(0, int(frac * n) - length // 2), n - length)
        out.append((name, audio[start:start + length].copy(), False))
    return out


def level_gain(audio: np.ndarray, sr: int, target_lufs: float = -20.0) -> float:
    """Linear gain that brings audio to target_lufs (1.0 when loudness is unmeasurable)."""
    measured = loudness(audio, sr)["integrated_lufs_dual_mono"]
    if measured is None:
        return 1.0
    return 10.0 ** ((target_lufs - measured) / 20.0)


def level_match(audio: np.ndarray, sr: int, target_lufs: float = -20.0, *, gain_scale: float = 1.0) -> np.ndarray:
    """Scale audio to target_lufs without clipping.

    If the gain would push the peak past PEAK_LIMIT it is reduced instead of
    hard-clipping. gain_scale lets a caller apply the same extra reduction to
    both arms of a pair so their relative level is preserved.
    """
    gain = level_gain(audio, sr, target_lufs) * gain_scale
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak * gain > PEAK_LIMIT:
        gain = PEAK_LIMIT / peak
    return (audio * gain).astype(np.float32)


def pair_gain_scale(a: np.ndarray, b: np.ndarray, sr: int, target_lufs: float = -20.0) -> float:
    """Shared reduction (<= 1) so neither arm exceeds PEAK_LIMIT at its matched gain."""
    scale = 1.0
    for audio in (a, b):
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        needed = peak * level_gain(audio, sr, target_lufs)
        if needed > PEAK_LIMIT:
            scale = min(scale, PEAK_LIMIT / needed)
    return scale


def build_pairs(
    baseline_dir: Path,
    challenger_dir: Path,
    out_dir: Path,
    *,
    seed: int | None = None,
    excerpt_sec: float = 75.0,
) -> int:
    import soundfile as sf

    base, chal = find_case_audio(baseline_dir), find_case_audio(challenger_dir)
    missing = sorted(set(base) ^ set(chal))
    if missing:
        raise ValueError(f"Cases present in only one run: {', '.join(missing)}")

    if seed is None:
        seed = secrets.randbits(32)
    rng = random.Random(seed)
    pairs_dir = Path(out_dir) / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict] = {}
    index = 0
    for case in sorted(base):
        a_audio, sr_a = sf.read(base[case], dtype="float32")
        b_audio, sr_b = sf.read(chal[case], dtype="float32")
        if sr_a != sr_b:
            raise ValueError(f"{case}: sample rates differ ({sr_a} vs {sr_b})")
        common = min(len(a_audio), len(b_audio))
        a_audio, b_audio = a_audio[:common], b_audio[:common]
        a_parts = cut_excerpts(a_audio, sr_a, excerpt_sec=excerpt_sec)
        b_parts = cut_excerpts(b_audio, sr_b, excerpt_sec=excerpt_sec)
        if a_parts[0][2]:  # shorter than one excerpt: one whole-file pair, not three identical ones
            a_parts = [("whole", a_parts[0][1], True)]
            b_parts = [("whole", b_parts[0][1], True)]
        for (pos, a_ex, a_tr), (_pos, b_ex, b_tr) in zip(a_parts, b_parts):
            arms = [("baseline", a_ex), ("challenger", b_ex)]
            rng.shuffle(arms)
            pair_id = f"{index:02d}"
            scale = pair_gain_scale(arms[0][1], arms[1][1], sr_a)
            for label, (_arm, excerpt) in zip(("A", "B"), arms):
                sf.write(pairs_dir / f"{pair_id}_{label}.wav", level_match(excerpt, sr_a, gain_scale=scale), sr_a)
            key[pair_id] = {"A": arms[0][0], "B": arms[1][0], "case": case, "position": pos,
                            "truncated": bool(a_tr or b_tr)}
            index += 1

    (Path(out_dir) / KEY_NAME).write_text(json.dumps({"seed": seed, "pairs": key}, indent=2), encoding="utf-8")
    with open(Path(out_dir) / "sheet.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair", "choice", "reason"])
        for pair_id in key:
            writer.writerow([pair_id, "", ""])
    return index


def sign_test_p(wins: int, n: int) -> float:
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def _load_key(ab_dir: Path) -> dict:
    path = Path(ab_dir) / KEY_NAME
    if not path.is_file():
        path = Path(ab_dir) / LEGACY_KEY_NAME
    return json.loads(path.read_text(encoding="utf-8"))["pairs"]


def score(ab_dir: Path) -> dict:
    key = _load_key(ab_dir)
    challenger = baseline = ties = unanswered = 0
    by_position: dict[str, dict[str, int]] = {}
    with open(Path(ab_dir) / "sheet.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing_cols = [c for c in ("pair", "choice") if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(f"sheet.csv is missing required column(s): {', '.join(missing_cols)}")
        rows = list(reader)
    seen: set[str] = set()
    duplicates = sorted({r["pair"] for r in rows if r["pair"] in seen or seen.add(r["pair"])})
    if duplicates:
        raise ValueError(f"Duplicate pair id(s) in sheet.csv: {', '.join(duplicates)}")
    unanswered += len(set(key) - seen)  # pairs absent from the sheet were never answered
    for row in rows:
        choice = (row["choice"] or "").strip().upper()
        if row["pair"] not in key:
            raise ValueError(f"Pair {row['pair']!r} in sheet.csv is not in the answer key")
        entry = key[row["pair"]]
        bucket = by_position.setdefault(entry["position"], {"challenger": 0, "baseline": 0, "ties": 0})
        if choice in ("A", "B"):
            winner = entry[choice]
            if winner == "challenger":
                challenger += 1
            else:
                baseline += 1
            bucket[winner] += 1
        elif choice in TIE_CHOICES:
            ties += 1
            bucket["ties"] += 1
        elif choice:
            raise ValueError(f"Pair {row['pair']}: invalid choice {row['choice']!r} (use A, B or same)")
        else:
            unanswered += 1
    n = challenger + baseline
    rate = challenger / n if n else 0.0
    p = sign_test_p(challenger, n)
    if unanswered:
        verdict = "INCOMPLETE"
    elif n and rate >= SHIP_RATE and p < SHIP_P:
        verdict = "SHIP"
    elif n and (1 - rate) >= SHIP_RATE and sign_test_p(baseline, n) < SHIP_P:
        verdict = "NO-SHIP"
    else:
        verdict = "INCONCLUSIVE"
    return {"challenger_wins": challenger, "baseline_wins": baseline, "ties": ties, "unanswered": unanswered,
            "rate": round(rate, 3), "p": round(p, 4), "verdict": verdict, "by_position": by_position}
