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
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
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


# Bracket markers ([pause:5s], [breath]) are contract syntax, not prose, and
# appear in every script. Removing them keeps them out of the statistics.
_MARKER = re.compile(r"\[[^\]]*\]")
_WORD = re.compile(r"[a-z]+")

MAX_NGRAM = 3


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with bracket markers removed."""
    return _WORD.findall(_MARKER.sub(" ", text.lower()))


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """Overlapping n-grams. Empty when there are fewer than n tokens."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _all_terms(tokens: Sequence[str], max_n: int) -> list[tuple[str, ...]]:
    terms: list[tuple[str, ...]] = []
    for n in range(1, max_n + 1):
        terms.extend(ngrams(tokens, n))
    return terms


def document_frequencies(
    token_lists: Sequence[Sequence[str]], max_n: int = MAX_NGRAM
) -> dict[tuple[str, ...], int]:
    """How many documents each term appears in (not how many times)."""
    df: Counter[tuple[str, ...]] = Counter()
    for tokens in token_lists:
        df.update(set(_all_terms(tokens, max_n)))
    return dict(df)


def build_idf(
    token_lists: Sequence[Sequence[str]], max_n: int = MAX_NGRAM
) -> tuple[dict[tuple[str, ...], float], float]:
    """Smoothed inverse document frequency over the whole corpus.

    Returns (idf_map, default_idf). The default applies to terms absent from
    the corpus, which are by definition maximally distinctive, so it is the
    largest weight the formula can produce.
    """
    n_docs = len(token_lists)
    df = document_frequencies(token_lists, max_n)
    idf = {
        term: math.log((n_docs + 1) / (count + 1)) + 1.0
        for term, count in df.items()
    }
    default_idf = math.log(n_docs + 1) + 1.0
    return idf, default_idf


def tfidf_vector(
    tokens: Sequence[str],
    idf: dict[tuple[str, ...], float],
    default_idf: float,
    max_n: int = MAX_NGRAM,
) -> dict[tuple[str, ...], float]:
    """L2-normalised TF-IDF vector. Pre-normalising makes cosine a dot product."""
    counts = Counter(_all_terms(tokens, max_n))
    if not counts:
        return {}
    vector = {
        term: count * idf.get(term, default_idf) for term, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return {}
    return {term: value / norm for term, value in vector.items()}


def cosine(a: dict[tuple[str, ...], float], b: dict[tuple[str, ...], float]) -> float:
    """Cosine similarity of two L2-normalised vectors, in [0, 1]."""
    if not a or not b:
        return 0.0
    # Iterate the smaller vector; the result is symmetric.
    if len(b) < len(a):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


RUN_NGRAM = 5
RUN_MAX_DF = 2


def longest_rare_run(
    tokens_a: Sequence[str],
    tokens_b: Sequence[str],
    df: dict[tuple[str, ...], int],
    *,
    n: int = RUN_NGRAM,
    max_df: int = RUN_MAX_DF,
) -> tuple[int, str]:
    """Longest run of consecutive rare n-grams from A that also occur in B.

    Returns (shared_token_span, shared_text). A run of k consecutive
    overlapping n-grams spans k + n - 1 tokens.

    Only n-grams with document frequency <= max_df count, so stock meditation
    phrasing -- which appears across the whole corpus -- cannot trigger this.

    Approximation worth knowing: consecutive matching n-grams in A are not
    proven contiguous in B. In practice a run of overlapping rare n-grams that
    all appear in B is a lifted passage; the alternative (a full longest-common-
    substring over every corpus pair) is O(n*m) per pair and not worth the cost
    for the same answer.
    """
    rare_b = {
        gram for gram in ngrams(tokens_b, n) if df.get(gram, 0) <= max_df
    }
    if not rare_b:
        return 0, ""

    best_len = 0
    best_start = 0
    run = 0
    for index, gram in enumerate(ngrams(tokens_a, n)):
        if gram in rare_b and df.get(gram, 0) <= max_df:
            run += 1
            if run > best_len:
                best_len = run
                best_start = index - run + 1
        else:
            run = 0

    if best_len == 0:
        return 0, ""

    span = best_len + n - 1
    return span, " ".join(tokens_a[best_start : best_start + span])
