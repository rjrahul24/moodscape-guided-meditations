# scripts/ab_listen_page.py
"""Build a static blind-listening page for one or more A/B rounds.

Usage:
    python scripts/ab_listen_page.py --out page.html --title "R2 eval" "Round A|how text=var/eval/x/abA" ...
Audio is referenced by relative path, so open the HTML from a location that keeps those paths valid.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval.listen import build_page_data, render_page  # noqa: E402


def parse_round(spec: str) -> tuple[str, str, Path]:
    head, _, path = spec.rpartition("=")
    if not head:
        raise ValueError(f"Round spec must look like 'name|how=ab_dir': {spec!r}")
    name, _, how = head.partition("|")
    return name, how, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="MoodScape listening test")
    parser.add_argument("rounds", nargs="+", help="'name|how text=ab_dir'")
    args = parser.parse_args()
    out = Path(args.out)
    data = build_page_data([parse_round(s) for s in args.rounds], out)
    out.write_text(render_page(args.title, data), encoding="utf-8")
    print(f"wrote {out} ({sum(len(r['pairs']) for r in data)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
