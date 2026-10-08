# scripts/ab_import_answers.py
"""Write answers downloaded from the listening page into each round's sheet.csv.

Usage:
    python scripts/ab_import_answers.py answers.json "Round A=var/eval/x/abA" ...
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval.listen import import_answers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("answers")
    parser.add_argument("rounds", nargs="+", help="'name=ab_dir'")
    args = parser.parse_args()
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    rounds = {}
    for spec in args.rounds:
        name, _, path = spec.rpartition("=")
        if not name:
            parser.error(f"Round spec must look like 'name=ab_dir': {spec!r}")
        rounds[name] = Path(path)
    for name, n in import_answers(answers, rounds).items():
        print(f"{name}: {n} answered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
