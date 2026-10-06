# scripts/ab_score.py
"""Score a filled A/B sheet: wins, ties, sign test and SHIP verdict.

Usage:
    python scripts/ab_score.py --ab var/eval/ab-001
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval.ab import score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ab", required=True)
    args = parser.parse_args()
    result = score(Path(args.ab))
    print(json.dumps(result, indent=2))
    print(f"\nVerdict: {result['verdict']} (challenger {result['challenger_wins']}"
          f" / baseline {result['baseline_wins']} / ties {result['ties']}, p={result['p']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
