# scripts/ab_review.py
"""Write REVIEW.md for an eval and inject the measurement card into listen.html.

Usage:
    python scripts/ab_review.py SPEC.toml
Arms, rounds, descriptions and cached-arm expectations all come from the spec.
Read-only except <out>/REVIEW.md and <out>/listen.html.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval.chain import EvalSpecError, load_eval_spec  # noqa: E402
from core.eval.review import write_review  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec")
    args = parser.parse_args(argv)
    try:
        spec = load_eval_spec(args.spec)
    except EvalSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not Path(spec.out).is_dir():
        print(f"error: out dir {spec.out} does not exist", file=sys.stderr)
        return 2
    print(f"wrote {write_review(spec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
