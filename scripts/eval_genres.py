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
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.auto_generate import AutoConfig, run  # noqa: E402
from core.eval.report import compute_run_metrics, summarize  # noqa: E402
from core.eval.testset import load_testset  # noqa: E402
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
    parser.add_argument("--testset", default="", help="path to a testset TOML (overrides --genres/--band/--runs)")
    parser.add_argument("--metrics", action="store_true", help="compute script/speech/mix metrics per run")
    parser.add_argument("--export-stems", action="store_true", help="export voice stem so speech metrics avoid the bed")
    parser.add_argument("--engine", default="f5", help="TTS engine for AutoConfig.tts_engine (f5, qwen3, ...)")
    args = parser.parse_args()

    slugs = (
        [s.strip() for s in args.genres.split(",") if s.strip()]
        or sorted(load_all_packs())
    )
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.testset:
        jobs = [(case.id, case.genre, case.band, case.seed) for case in load_testset(Path(args.testset))]
    else:
        jobs = [(f"{slug}-{index}", slug, args.band, None) for slug in slugs for index in range(args.runs)]

    transcriber = None
    if args.metrics:
        from core.eval.speech_metrics import WhisperTranscriber
        transcriber = WhisperTranscriber()

    for spec in args.configs:
        name, writer, judge = _parse_config(spec)
        os.environ["MOODSCAPE_SCRIPT_PLANNER"] = writer
        os.environ["MOODSCAPE_SCRIPT_GENERATOR"] = writer
        os.environ["MOODSCAPE_SCRIPT_JUDGE"] = judge

        for job_id, slug, band, seed in jobs:
            out_dir = out_root / name / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            try:
                from core.pipeline import MeditationPipeline
                pipeline = MeditationPipeline()
                extra = {}
                if seed is not None:
                    extra["seed"] = seed
                if args.export_stems:
                    extra["do_export_stems"] = True
                result = run(
                    "",
                    genre=slug,
                    config=AutoConfig.from_genre(load_pack(slug), band=band, tts_engine=args.engine),
                    pipeline=pipeline,
                    rng=random.Random(seed) if seed is not None else None,
                    **extra,
                )
                for src_path in (result.audio_path, result.script_path, result.meta_path):
                    if Path(src_path).is_file():
                        shutil.move(str(src_path), str(out_dir / Path(src_path).name))
                audio_out = out_dir / Path(result.audio_path).name
                row = {
                    "config": name, "genre": slug, "case": job_id, "band": band, "seed": seed,
                    "ok": True,
                    "seconds": round(time.monotonic() - started, 1),
                    "angle": result.angle,
                    "estimated_sec": round(result.estimated_sec, 1),
                    "originality": round(result.originality, 3),
                    "advisories": [v.code for v in result.violations],
                    "audio": str(audio_out),
                }
                if args.metrics:
                    stems = getattr(pipeline, "last_stems", None) or {}
                    voice_stem = stems.get("voice")
                    if voice_stem and Path(voice_stem).is_file():
                        moved = out_dir / "narration_stem.wav"
                        shutil.move(voice_stem, moved)
                        voice_stem = str(moved)
                    metrics = compute_run_metrics(
                        script=result.script,
                        draft=result.draft_script or None,
                        engine=args.engine,
                        content_type=load_pack(slug).content_type,
                        audio_path=str(audio_out),
                        voice_stem_path=voice_stem,
                        estimated_sec=result.estimated_sec,
                        transcriber=transcriber,
                    )
                    transcriber.unload()  # keep Whisper out of RAM while the next job's LLM loads
                    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    row["metrics"] = metrics
                rows.append(row)
            except Exception as exc:
                rows.append({
                    "config": name, "genre": slug, "case": job_id, "band": band, "seed": seed,
                    "ok": False,
                    "seconds": round(time.monotonic() - started, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                })
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "metrics"}), flush=True)

    if transcriber is not None:
        transcriber.unload()
    (out_root / "summary.md").write_text(summarize(rows), encoding="utf-8")
    print("\n" + summarize(rows))

    (out_root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ok = sum(1 for row in rows if row["ok"])
    print(f"\n{ok}/{len(rows)} runs succeeded. Results: {out_root / 'results.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
