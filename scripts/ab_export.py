# scripts/ab_export.py
"""Export blind A/B listening pairs from a baseline and a challenger eval run.

Usage:
    python scripts/ab_export.py --a var/eval/base/baseline --b var/eval/new/baseline --out var/eval/ab-001
Then listen to pairs/NN_A.wav vs NN_B.wav and fill sheet.csv (choice = A, B, leanA, leanB or same).
"""

import argparse
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval.ab import build_pairs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="baseline config dir (contains <case>/ subdirs)")
    parser.add_argument("--b", required=True, help="challenger config dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--excerpt-sec", type=float, default=75.0)
    parser.add_argument("--positions", default=None,
                        help="comma-separated excerpt positions to cut (start,middle,end); default all")
    parser.add_argument("--seed", type=int, default=None, help="shuffle seed (default: random, recorded in .key.json)")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    print(f"A/B shuffle seed: {seed}")
    positions = [p.strip() for p in args.positions.split(",") if p.strip()] if args.positions else None
    try:
        n = build_pairs(Path(args.a), Path(args.b), Path(args.out), seed=seed, excerpt_sec=args.excerpt_sec,
                        positions=positions)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {n} blind pairs to {Path(args.out) / 'pairs'}. Fill {Path(args.out) / 'sheet.csv'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
