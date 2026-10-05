"""CLI: benchmark script-generation model pairings.

Usage:
    python scripts/bench_script_models.py \
        --pair ollama:qwen3:30b ollama:gemma3:27b \
        --pair anthropic:claude-opus-5 anthropic:claude-sonnet-5 \
        --out bench_results.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.auto_generate import AutoConfig  # noqa: E402
from core.bench import (  # noqa: E402
    BENCH_PROMPTS,
    format_bench_table,
    run_bench,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("GENERATOR", "JUDGE"),
        required=True,
        help="A generator and judge spec, e.g. --pair ollama:qwen3:30b anthropic:claude-opus-5",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=len(BENCH_PROMPTS),
        help="Use only the first N benchmark prompts.",
    )
    parser.add_argument("--out", type=Path, default=Path("bench_results.md"))
    args = parser.parse_args()

    pairings = [tuple(p) for p in args.pair]
    rows = run_bench(
        pairings,
        prompts=BENCH_PROMPTS[: args.limit],
        config=AutoConfig.from_env(),
    )

    table = format_bench_table(rows)
    args.out.write_text(table + "\n", encoding="utf-8")
    print(table)

    failures = [r for r in rows if not r.passed]
    print(f"\n{len(rows) - len(failures)}/{len(rows)} passed.")
    for row in failures:
        print(f"  FAIL {row.generator} -> {row.judge}: {row.error[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
