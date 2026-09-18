"""Guarantee that no two generated meditations are the same.

Two layers cooperate. This module supplies the machinery for both:

  * Proactive -- ``recent_angles`` and ``avoid_terms`` feed the planner an
    explicit "do not reuse this" list before a word is written.
  * Reactive  -- ``assess`` measures a finished script against the corpus so
    ``linter.check_originality`` can turn the result into a Violation that the
    existing repair loop already knows how to act on.

The hard part is not flagging generic meditation language. Every script
legitimately says "notice your breath". TF-IDF self-calibrates against exactly
that: terms appearing across the whole corpus get near-zero weight while
distinctive ones keep full weight, so no hand-maintained stoplist is needed.

That dictates an asymmetry worth stating plainly: IDF is computed over the
ENTIRE corpus (all genres), because that is what learns which phrases are
generic; similarity is measured WITHIN one genre, because that is where
collisions actually happen.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "var" / "originality"

_INDEX_NAME = "index.json"
_SCRIPTS_DIRNAME = "scripts"


@dataclass(frozen=True)
class CorpusEntry:
    """One previously generated script."""

    script_id: str
    genre: str
    angle: str
    created: str
    text: str


def _resolve(corpus_dir: Path | None) -> Path:
    return corpus_dir if corpus_dir is not None else CORPUS_DIR


def _read_index(root: Path) -> list[dict]:
    path = root / _INDEX_NAME
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Corpus index at %s is unreadable; treating as empty.", path)
        return []
    return data if isinstance(data, list) else []


def _write_index(root: Path, records: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / _INDEX_NAME).write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


def add_to_corpus(
    script: str,
    *,
    genre: str,
    angle: str = "",
    corpus_dir: Path | None = None,
) -> str:
    """Record a rendered script so later runs can be checked against it.

    Returns the new entry's script_id.
    """
    root = _resolve(corpus_dir)
    scripts_dir = root / _SCRIPTS_DIRNAME
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Microsecond precision plus a random suffix: a fast test loop can add
    # many entries inside one microsecond tick on some platforms, and a
    # collision would silently overwrite a prior script.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    script_id = f"{stamp}-{os.urandom(3).hex()}"

    (scripts_dir / f"{script_id}.txt").write_text(script, encoding="utf-8")

    records = _read_index(root)
    records.append(
        {
            "script_id": script_id,
            "genre": genre,
            "angle": angle,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _write_index(root, records)
    return script_id


def load_corpus(
    *,
    genre: str | None = None,
    limit: int | None = None,
    corpus_dir: Path | None = None,
) -> list[CorpusEntry]:
    """Load corpus entries, most recent first.

    A record whose script file has been deleted is skipped rather than
    raising: the corpus is machine-local cache, and a missing file must not
    stop a generation run.
    """
    root = _resolve(corpus_dir)
    scripts_dir = root / _SCRIPTS_DIRNAME

    entries: list[CorpusEntry] = []
    for record in reversed(_read_index(root)):
        if genre is not None and record.get("genre") != genre:
            continue
        script_id = record.get("script_id", "")
        path = scripts_dir / f"{script_id}.txt"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            logger.debug("Corpus script %s is missing; skipping.", path)
            continue
        entries.append(
            CorpusEntry(
                script_id=script_id,
                genre=record.get("genre", ""),
                angle=record.get("angle", ""),
                created=record.get("created", ""),
                text=text,
            )
        )
        if limit is not None and len(entries) >= limit:
            break
    return entries


def recent_angles(
    genre: str, *, limit: int = 3, corpus_dir: Path | None = None
) -> list[str]:
    """Angles most recently used for a genre, most recent first, deduplicated.

    Reads the index only -- no script bodies -- so it stays cheap as the
    corpus grows.
    """
    root = _resolve(corpus_dir)
    seen: list[str] = []
    for record in reversed(_read_index(root)):
        if record.get("genre") != genre:
            continue
        angle = record.get("angle", "")
        if angle and angle not in seen:
            seen.append(angle)
        if len(seen) >= limit:
            break
    return seen
