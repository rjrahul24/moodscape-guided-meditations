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

from core.auto_generate import DURATION_BANDS, AutoConfig, render_script, run  # noqa: E402
from core.eval.runner_support import (  # noqa: E402
    fresh_job_corpus,
    prepare_snapshot,
    stale_case_dirs,
    write_results,
)
from core.eval.report import compute_run_metrics, summarize  # noqa: E402
from core.eval.render_only import load_source_cases  # noqa: E402
from core.eval.testset import load_testset  # noqa: E402
from core.genres import load_all_packs, load_pack  # noqa: E402
from core.originality import CORPUS_DIR  # noqa: E402


def _parse_config(spec: str) -> tuple[str, str, str]:
    """'name=writer_spec|judge_spec' -> (name, writer_spec, judge_spec)."""
    name, _, models = spec.partition("=")
    writer, _, judge = models.partition("|")
    if not (name and writer and judge):
        raise SystemExit(f"Malformed --configs entry: {spec!r}")
    return name, writer, judge


def _finish_job(result, *, out_dir, base_row, started):
    """Move a finished job's audio/script/meta into out_dir; return (base row + ok/voice/seconds, audio path)."""
    for src_path in (result.audio_path, result.script_path, result.meta_path):
        if Path(src_path).is_file():
            shutil.move(str(src_path), str(out_dir / Path(src_path).name))
    audio_out = out_dir / Path(result.audio_path).name
    row = dict(base_row)
    row.update({"ok": True, "voice": result.voice,
                "seconds": round(time.monotonic() - started, 1)})
    try:
        hit = json.loads((out_dir / Path(result.meta_path).name).read_text(encoding="utf-8")).get("tts_cache_hit")
        if isinstance(hit, bool):
            row["tts_cache_hit"] = hit
    except (OSError, ValueError):
        pass  # bookkeeping only; never fail a finished job
    return row, audio_out


def _attach_metrics(row, *, result_script, draft, pipeline, audio_out, out_dir, args,
                    transcriber, slug, estimated_sec):
    stems = getattr(pipeline, "last_stems", None) or {}
    voice_stem = stems.get("voice")
    if voice_stem and Path(voice_stem).is_file():
        moved = out_dir / "narration_stem.wav"
        shutil.move(voice_stem, moved)
        voice_stem = str(moved)
    try:
        metrics = compute_run_metrics(
            script=result_script,
            draft=draft,
            engine=args.engine,
            content_type=load_pack(slug).content_type,
            audio_path=str(audio_out),
            voice_stem_path=voice_stem,
            estimated_sec=estimated_sec,
            transcriber=transcriber,
        )
    finally:
        transcriber.unload()  # never leave Whisper resident, even after an exception
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    row["metrics"] = metrics


def _run_render_only(args, out_root: Path) -> int:
    name = args.name
    stale = stale_case_dirs(out_root / name)
    if stale:
        print(f"Refusing to start: {out_root / name} already contains case dirs "
              f"({', '.join(stale[:5])}{'...' if len(stale) > 5 else ''}). Use a new --out/--name "
              "or delete the directory.", file=sys.stderr)
        return 2
    cases, skipped = load_source_cases(Path(args.scripts_from))
    for msg in skipped:
        print(f"skipped: {msg}", file=sys.stderr)
    out_root.mkdir(parents=True, exist_ok=True)

    transcriber = None
    if args.metrics:
        from core.eval.speech_metrics import WhisperTranscriber
        transcriber = WhisperTranscriber()

    rows = []
    for case in cases:
        started = time.monotonic()
        base_row = {"config": name, "genre": case.genre, "case": case.case, "band": case.band,
                    "seed": case.seed, "rendered_from": str(case.source_dir)}
        if not Path(case.background_path).is_file():
            rows.append({**base_row, "ok": False, "seconds": 0.0,
                         "error": f"background missing: {case.background_path}"})
        else:
            out_dir = out_root / name / case.case
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                src_meta = json.loads(next(case.source_dir.glob("*.meta.json")).read_text(encoding="utf-8"))
                estimated_sec = src_meta.get("estimated_sec")
                from core.pipeline import MeditationPipeline
                pipeline = MeditationPipeline()
                overrides = {"voice_slug": case.voice} if case.voice else {}
                cfg = AutoConfig.from_genre(
                    load_pack(case.genre),
                    band=case.band if case.band in DURATION_BANDS else "short",
                    tts_engine=args.engine,
                    **overrides,
                )
                result = render_script(
                    case.script, config=cfg,
                    background_label=case.background_label, background_path=case.background_path,
                    pipeline=pipeline,
                    meta={"rendered_from": str(case.source_dir), "estimated_sec": estimated_sec},
                    seed=case.seed, do_export_stems=args.export_stems,
                )
                row, audio_out = _finish_job(
                    result, out_dir=out_dir, base_row=base_row, started=started)
                if estimated_sec is not None:
                    row["estimated_sec"] = round(estimated_sec, 1)
                row["audio"] = str(audio_out)
                if args.metrics:
                    _attach_metrics(row, result_script=case.script, draft=None, pipeline=pipeline,
                                    audio_out=audio_out, out_dir=out_dir, args=args,
                                    transcriber=transcriber, slug=case.genre,
                                    estimated_sec=estimated_sec)
                rows.append(row)
            except Exception as exc:
                rows.append({**base_row, "ok": False,
                             "seconds": round(time.monotonic() - started, 1),
                             "error": f"{type(exc).__name__}: {exc}"})
        write_results(out_root, rows, summarize(rows))
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "metrics"}), flush=True)

    if transcriber is not None:
        transcriber.unload()
    write_results(out_root, rows, summarize(rows))
    print("\n" + summarize(rows))
    ok = sum(1 for row in rows if row["ok"])
    print(f"\n{ok}/{len(rows)} renders succeeded. Results: {out_root / 'results.json'}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genres", default="", help="comma-separated slugs; default all")
    parser.add_argument("--band", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--scripts-from", default="",
                        help="config dir of an earlier run (<out>/<name>): re-render its exact scripts, "
                             "skipping all LLM stages and the corpus snapshot")
    parser.add_argument("--name", default="eval", help="output config dir name in --scripts-from mode")
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=1, help="runs per genre per config")
    parser.add_argument("--testset", default="", help="path to a testset TOML (overrides --genres/--band/--runs)")
    parser.add_argument("--metrics", action="store_true", help="compute script/speech/mix metrics per run")
    parser.add_argument("--export-stems", action="store_true", help="export voice stem so speech metrics avoid the bed")
    parser.add_argument("--engine", default="f5", help="TTS engine for AutoConfig.tts_engine (f5, elevenlabs, ...); elevenlabs bills per character — see meta.json 'elevenlabs'")
    parser.add_argument("--corpus-from", default="",
                        help="earlier eval --out dir; reuse its corpus_snapshot instead of snapshotting the live corpus")
    args = parser.parse_args()
    if not args.configs and not args.scripts_from:
        parser.error("one of --configs or --scripts-from is required")

    slugs = (
        [s.strip() for s in args.genres.split(",") if s.strip()]
        or sorted(load_all_packs())
    )
    out_root = Path(args.out)
    if args.scripts_from:
        return _run_render_only(args, out_root)
    parsed_configs = [_parse_config(spec) for spec in args.configs]
    for name, _w, _j in parsed_configs:
        stale = stale_case_dirs(out_root / name)
        if stale:
            print(f"Refusing to start: {out_root / name} already contains case dirs "
                  f"({', '.join(stale[:5])}{'...' if len(stale) > 5 else ''}). Stale mixes would confuse "
                  "A/B export; use a new --out or delete the directory.", file=sys.stderr)
            return 2
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = prepare_snapshot(out_root, CORPUS_DIR, Path(args.corpus_from) if args.corpus_from else None)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Originality corpus snapshot: {snapshot} (live corpus is never written)", flush=True)
    rows = []

    if args.testset:
        jobs = [(case.id, case.genre, case.band, case.seed) for case in load_testset(Path(args.testset))]
    else:
        jobs = [(f"{slug}-{index}", slug, args.band, None) for slug in slugs for index in range(args.runs)]

    transcriber = None
    if args.metrics:
        from core.eval.speech_metrics import WhisperTranscriber
        transcriber = WhisperTranscriber()

    for name, writer, judge in parsed_configs:
        os.environ["MOODSCAPE_SCRIPT_PLANNER"] = writer
        os.environ["MOODSCAPE_SCRIPT_GENERATOR"] = writer
        os.environ["MOODSCAPE_SCRIPT_JUDGE"] = judge

        for job_id, slug, band, seed in jobs:
            out_dir = out_root / name / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            job_corpus = fresh_job_corpus(snapshot, out_dir)
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
                    config=AutoConfig.from_genre(
                        load_pack(slug), band=band, tts_engine=args.engine, corpus_dir=job_corpus,
                    ),
                    pipeline=pipeline,
                    rng=random.Random(seed) if seed is not None else None,
                    **extra,
                )
                row, audio_out = _finish_job(
                    result, out_dir=out_dir, started=started,
                    base_row={"config": name, "genre": slug, "case": job_id, "band": band, "seed": seed})
                row.update({
                    "angle": result.angle,
                    "estimated_sec": round(result.estimated_sec, 1),
                    "originality": round(result.originality, 3),
                    "advisories": [v.code for v in result.violations],
                    "audio": str(audio_out),
                })
                if args.metrics:
                    _attach_metrics(row, result_script=result.script, draft=result.draft_script or None,
                                    pipeline=pipeline, audio_out=audio_out, out_dir=out_dir, args=args,
                                    transcriber=transcriber, slug=slug, estimated_sec=result.estimated_sec)
                rows.append(row)
            except Exception as exc:
                rows.append({
                    "config": name, "genre": slug, "case": job_id, "band": band, "seed": seed,
                    "ok": False,
                    "seconds": round(time.monotonic() - started, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                shutil.rmtree(job_corpus, ignore_errors=True)
            write_results(out_root, rows, summarize(rows))
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "metrics"}), flush=True)

    if transcriber is not None:
        transcriber.unload()
    write_results(out_root, rows, summarize(rows))
    print("\n" + summarize(rows))
    ok = sum(1 for row in rows if row["ok"])
    print(f"\n{ok}/{len(rows)} runs succeeded. Results: {out_root / 'results.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
