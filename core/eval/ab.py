# core/eval/ab.py
"""Blind A/B listening: excerpt cutting, level matching, pairing, scoring.

Convention: run A is the baseline, run B the challenger. On disk the arms are
shuffled per pair into files labelled A/B; only key.json knows which is which.
"""

import csv
import json
import math
import random
from pathlib import Path

import numpy as np

from core.eval.mix_metrics import loudness

POSITIONS = (("start", 0.10), ("middle", 0.50), ("end", 0.85))
SHIP_RATE = 0.60
SHIP_P = 0.05


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


def level_match(audio: np.ndarray, sr: int, target_lufs: float = -20.0) -> np.ndarray:
    measured = loudness(audio, sr)["integrated_lufs_dual_mono"]
    if measured is None:
        return audio
    gain = 10.0 ** ((target_lufs - measured) / 20.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def build_pairs(
    baseline_dir: Path,
    challenger_dir: Path,
    out_dir: Path,
    *,
    seed: int = 0,
    excerpt_sec: float = 75.0,
) -> int:
    import soundfile as sf

    base, chal = find_case_audio(baseline_dir), find_case_audio(challenger_dir)
    missing = sorted(set(base) ^ set(chal))
    if missing:
        raise ValueError(f"Cases present in only one run: {', '.join(missing)}")

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
        for (pos, a_ex, a_tr), (_pos, b_ex, b_tr) in zip(
            cut_excerpts(a_audio, sr_a, excerpt_sec=excerpt_sec),
            cut_excerpts(b_audio, sr_b, excerpt_sec=excerpt_sec),
        ):
            arms = [("baseline", a_ex), ("challenger", b_ex)]
            rng.shuffle(arms)
            pair_id = f"{index:02d}"
            for label, (_arm, excerpt) in zip(("A", "B"), arms):
                sf.write(pairs_dir / f"{pair_id}_{label}.wav", level_match(excerpt, sr_a), sr_a)
            key[pair_id] = {"A": arms[0][0], "B": arms[1][0], "case": case, "position": pos,
                            "truncated": bool(a_tr or b_tr)}
            index += 1

    (Path(out_dir) / "key.json").write_text(json.dumps({"seed": seed, "pairs": key}, indent=2), encoding="utf-8")
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


def score(ab_dir: Path) -> dict:
    key = json.loads((Path(ab_dir) / "key.json").read_text(encoding="utf-8"))["pairs"]
    challenger = baseline = ties = 0
    by_position: dict[str, dict[str, int]] = {}
    with open(Path(ab_dir) / "sheet.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            choice = row["choice"].strip().upper()
            entry = key[row["pair"]]
            bucket = by_position.setdefault(entry["position"], {"challenger": 0, "baseline": 0, "ties": 0})
            if choice in ("A", "B"):
                winner = entry[choice]
                if winner == "challenger":
                    challenger += 1
                else:
                    baseline += 1
                bucket[winner] += 1
            elif choice:
                ties += 1
                bucket["ties"] += 1
    n = challenger + baseline
    rate = challenger / n if n else 0.0
    p = sign_test_p(challenger, n)
    if n and rate >= SHIP_RATE and p < SHIP_P:
        verdict = "SHIP"
    elif n and (1 - rate) >= SHIP_RATE and sign_test_p(baseline, n) < SHIP_P:
        verdict = "NO-SHIP"
    else:
        verdict = "INCONCLUSIVE"
    return {"challenger_wins": challenger, "baseline_wins": baseline, "ties": ties,
            "rate": round(rate, 3), "p": round(p, 4), "verdict": verdict, "by_position": by_position}
