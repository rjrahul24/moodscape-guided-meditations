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
        # A corrupt index must not be silently overwritten: add_to_corpus()
        # calls this, then _write_index()s a single new record over whatever
        # this returns -- if that were [], every prior script would be
        # orphaned permanently the moment one write got cut short (which is
        # itself how the index goes corrupt in the first place, since the
        # old _write_index() was a non-atomic full rewrite). Rename the
        # unreadable file aside instead, so the corpus survives and a human
        # can recover it, and log exactly where it went.
        quarantine = path.with_name(
            f"{path.name}.corrupt-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        )
        try:
            path.rename(quarantine)
            logger.warning(
                "Corpus index at %s was unreadable; moved aside to %s and "
                "starting a fresh index. Prior scripts on disk are intact "
                "and can be recovered by inspecting the quarantined file.",
                path, quarantine,
            )
        except OSError:
            logger.warning(
                "Corpus index at %s is unreadable and could not be moved "
                "aside; treating as empty for this run.", path,
            )
        return []
    return data if isinstance(data, list) else []


def _write_index(root: Path, records: list[dict]) -> None:
    """Write the index atomically: a temp file plus os.replace().

    A plain write_text() truncates the file before writing, so a process
    killed mid-write (or a full disk) leaves a partial JSON file -- which is
    exactly the "corrupt index" case _read_index() has to quarantine. Writing
    to a temp file in the same directory and renaming it into place makes the
    replace atomic: the index is always either the old complete file or the
    new complete file, never a partial one.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / _INDEX_NAME
    tmp_path = path.with_name(f"{_INDEX_NAME}.tmp-{os.getpid()}-{os.urandom(3).hex()}")
    tmp_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


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

    CAUTION: `df` must have been built with `max_n >= n`. document_frequencies()
    defaults to max_n=3 while this function defaults to n=5, so a caller using
    both defaults would find no 5-gram in `df`, score every 5-gram as rare, and
    silently defeat the filter. assess() passes max_n=RUN_NGRAM for this reason.
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
        if gram in rare_b:
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


# Below this many documents, IDF carries no signal: with three scripts every
# phrase looks rare, so cosine would flag unrelated meditations as copies.
SMALL_CORPUS_BELOW = 10


@dataclass(frozen=True)
class OriginalityReport:
    """How much a candidate script resembles what has already been made."""

    max_cosine: float
    cosine_available: bool
    nearest_id: str
    shared_span: int
    shared_text: str
    corpus_size: int
    # How many SAME-GENRE scripts were actually compared against (the size of
    # `compare_against`). This is what determines whether the rare-run
    # document-frequency statistics are meaningful for THIS genre -- a genre
    # holding one or two prior scripts cannot tell rare from stock phrasing,
    # regardless of how large the all-genre corpus (`corpus_size`) is.
    compared_count: int = 0


def assess(
    script: str,
    *,
    compare_against: Sequence[CorpusEntry],
    idf_texts: Sequence[str],
) -> OriginalityReport:
    """Measure a script against prior work.

    Args:
        script: The candidate script.
        compare_against: Prior scripts to compare with -- SAME GENRE ONLY.
            That is where collisions actually happen.
        idf_texts: Every script in the corpus, ALL GENRES. This is what
            teaches the weighting which phrases are generic meditation
            vocabulary, so it must not be narrowed to one genre.
    """
    if not compare_against:
        return OriginalityReport(
            max_cosine=0.0,
            cosine_available=False,
            nearest_id="",
            shared_span=0,
            shared_text="",
            corpus_size=len(idf_texts),
            compared_count=0,
        )

    candidate_tokens = tokenize(script)
    other_tokens = {entry.script_id: tokenize(entry.text) for entry in compare_against}

    # Tokenize the whole (all-genre) corpus once and reuse it for both the
    # rare-run document-frequency filter and the cosine IDF weighting below,
    # rather than tokenizing idf_texts twice.
    idf_token_lists = [tokenize(text) for text in idf_texts]

    # df for the rare-run check MUST be built from the entire corpus (every
    # genre), not just compare_against (same genre only): max_df is meant to
    # exclude phrasing that is common ACROSS THE WHOLE CORPUS, and a genre
    # holding one or two prior scripts can never demonstrate that a phrase is
    # common. The candidate itself is included too, so a phrase the candidate
    # repeats internally is correctly seen as common rather than lifted.
    run_df = document_frequencies(
        [candidate_tokens, *idf_token_lists], max_n=RUN_NGRAM
    )

    best_span, best_text = 0, ""
    for tokens in other_tokens.values():
        span, text = longest_rare_run(candidate_tokens, tokens, run_df)
        if span > best_span:
            best_span, best_text = span, text

    cosine_available = len(idf_texts) >= SMALL_CORPUS_BELOW
    max_cosine, nearest_id = 0.0, ""

    if cosine_available:
        idf, default_idf = build_idf(idf_token_lists)
        candidate_vector = tfidf_vector(candidate_tokens, idf, default_idf)
        for script_id, tokens in other_tokens.items():
            score = cosine(
                candidate_vector, tfidf_vector(tokens, idf, default_idf)
            )
            if score > max_cosine:
                max_cosine, nearest_id = score, script_id

    return OriginalityReport(
        max_cosine=max_cosine,
        cosine_available=cosine_available,
        nearest_id=nearest_id,
        shared_span=best_span,
        shared_text=best_text,
        corpus_size=len(idf_texts),
        compared_count=len(compare_against),
    )


def avoid_terms(
    entries: Sequence[CorpusEntry],
    idf_texts: Sequence[str],
    *,
    top_n: int = 12,
) -> list[str]:
    """The most distinctive phrases in recent scripts, for the planner to avoid.

    Only multi-word terms are returned: single words ("staircase") over-
    constrain the writer, while phrases ("copper staircase descending") name
    the specific image that should not recur.
    """
    if not entries or not idf_texts:
        return []

    idf, default_idf = build_idf([tokenize(text) for text in idf_texts])
    weights: Counter[tuple[str, ...]] = Counter()
    for entry in entries:
        vector = tfidf_vector(tokenize(entry.text), idf, default_idf)
        for term, value in vector.items():
            if len(term) >= 2:
                weights[term] += value

    # TF-IDF weight alone ties constantly: every term unique to one entry
    # gets the same maximal weight, whether it is a content phrase ("copper
    # staircase") or a run of function words ("it picture a"). Break ties by
    # character length rather than insertion order -- a longer phrase is
    # disproportionately likely to be the specific, informative one, and
    # this needs no hand-maintained stoplist, just the term itself.
    ranked = sorted(
        weights.items(),
        key=lambda item: (item[1], sum(len(word) for word in item[0])),
        reverse=True,
    )
    return [" ".join(term) for term, _ in ranked[:top_n]]
