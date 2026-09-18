#!/usr/bin/env python
"""Render a matrix of genres x model configurations for listening tests.

Benchmarks rank models on generic creative writing. They cannot tell you
which model writes the better MEDITATION, which is what this decides. Writes
every script, audio file and metric into one directory so configurations can
be compared by ear.

    python scripts/eval_genres.py --genres stress_relief,grief_and_loss \\
        --configs "qwen=ollama:qwen3.8:27b|ollama:gemma4:31b" \\
                  "muse=ollama:muse-glimmer:30b|ollama:gemma4:31b" \\
        --out /tmp/eval
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.auto_generate import AutoConfig, run  # noqa: E402
from core.genres import load_all_packs, load_pack  # noqa: E402


def _parse_config(spec: str) -> tuple[str, str, str]:
    """'name=writer_spec|judge_spec' -> (name, writer_spec, judge_spec)."""
    name, _, models = spec.partition("=")
    writer, _, judge = models.partition("|")
    if not (name and writer and judge):
        raise SystemExit(f"Malformed --configs entry: {spec!r}")
    return name, writer, judge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genres", default="", help="comma-separated slugs; default all")
    parser.add_argument("--band", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=1, help="runs per genre per config")
    args = parser.parse_args()

    slugs = (
        [s.strip() for s in args.genres.split(",") if s.strip()]
        or sorted(load_all_packs())
    )
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in args.configs:
        name, writer, judge = _parse_config(spec)
        os.environ["MOODSCAPE_SCRIPT_PLANNER"] = writer
        os.environ["MOODSCAPE_SCRIPT_GENERATOR"] = writer
        os.environ["MOODSCAPE_SCRIPT_JUDGE"] = judge

        for slug in slugs:
            for index in range(args.runs):
                out_dir = out_root / name / f"{slug}-{index}"
                out_dir.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                try:
                    result = run(
                        "",
                        genre=slug,
                        config=AutoConfig.from_genre(load_pack(slug), band=args.band),
                    )
                    # Move produced files into out_dir
                    shutil.copy2(result.audio_path, out_dir / Path(result.audio_path).name)
                    shutil.copy2(result.script_path, out_dir / Path(result.script_path).name)
                    shutil.copy2(result.meta_path, out_dir / Path(result.meta_path).name)

                    rows.append(
                        {
                            "config": name, "genre": slug, "run": index,
                            "ok": True,
                            "seconds": round(time.monotonic() - started, 1),
                            "angle": result.angle,
                            "estimated_sec": round(result.estimated_sec, 1),
                            "originality": round(result.originality, 3),
                            "advisories": [v.code for v in result.violations],
                            "audio": str(out_dir / Path(result.audio_path).name),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "config": name, "genre": slug, "run": index,
                            "ok": False,
                            "seconds": round(time.monotonic() - started, 1),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                print(json.dumps(rows[-1]), flush=True)

    (out_root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ok = sum(1 for row in rows if row["ok"])
    print(f"\n{ok}/{len(rows)} runs succeeded. Results: {out_root / 'results.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
