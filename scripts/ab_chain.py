# scripts/ab_chain.py
"""Generate the zsh eval chain from a TOML eval spec (prints it, or writes it with --write).

Usage:
    python scripts/ab_chain.py var/eval/r3/spec.toml [--write [PATH]]
Without a path, --write puts chain.sh in the spec's out dir and makes it executable.
The chain is never run by this script.
"""

import argparse
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval.chain import EvalSpecError, load_eval_spec, render_chain  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec")
    parser.add_argument("--write", nargs="?", const="", default=None, metavar="PATH",
                        help="write the chain (default <out>/chain.sh) and chmod +x instead of printing")
    args = parser.parse_args(argv)
    try:
        spec = load_eval_spec(args.spec)
    except EvalSpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = render_chain(spec)
    if args.write is None:
        print(text, end="")
        return 0
    dest = Path(args.write) if args.write else Path(spec.out) / "chain.sh"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
