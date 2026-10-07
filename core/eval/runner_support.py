# core/eval/runner_support.py
"""Filesystem helpers for scripts/eval_genres.py: corpus isolation and run hygiene.

Eval runs must not read the live originality corpus (it drifts between runs and
confounds baseline vs challenger) nor write to it. The corpus is frozen once per
run into <out>/corpus_snapshot and copied fresh into each job.
"""

import json
import shutil
from pathlib import Path

SNAPSHOT_NAME = "corpus_snapshot"
JOB_CORPUS_NAME = ".corpus"


def snapshot_corpus(source: Path, dest: Path) -> Path:
    """Copy source (may not exist) to dest, replacing dest. Returns dest."""
    source, dest = Path(source), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    if source.is_dir():
        shutil.copytree(source, dest)
    else:
        dest.mkdir(parents=True)
    return dest


def prepare_snapshot(out_root: Path, live_corpus: Path, corpus_from: Path | None = None) -> Path:
    """Return <out>/corpus_snapshot, creating it if needed.

    --corpus-from <dir> copies <dir>/corpus_snapshot (an earlier run's frozen
    state); it must exist. Without it, an existing snapshot in out_root is kept
    (so adding a config later reuses the same frozen state), else the live
    corpus is copied.
    """
    dest = Path(out_root) / SNAPSHOT_NAME
    if corpus_from is not None:
        src = Path(corpus_from) / SNAPSHOT_NAME
        if not src.is_dir():
            raise FileNotFoundError(f"No {SNAPSHOT_NAME} in --corpus-from {corpus_from}")
        if src.resolve() == dest.resolve():
            return dest
        return snapshot_corpus(src, dest)
    if dest.is_dir():
        return dest
    return snapshot_corpus(live_corpus, dest)


def fresh_job_corpus(snapshot: Path, job_dir: Path) -> Path:
    """Copy the snapshot into a clean <job_dir>/.corpus and return it."""
    return snapshot_corpus(snapshot, Path(job_dir) / JOB_CORPUS_NAME)


def stale_case_dirs(config_dir: Path) -> list[str]:
    """Names of case directories already present under a config dir."""
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        return []
    return sorted(p.name for p in config_dir.iterdir() if p.is_dir())


def write_results(out_root: Path, rows: list[dict], summary: str) -> None:
    """Rewrite results.json and summary.md so a crash never loses finished jobs."""
    out_root = Path(out_root)
    (out_root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_root / "summary.md").write_text(summary, encoding="utf-8")
