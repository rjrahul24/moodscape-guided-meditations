"""ElevenLabs preprocessing: shared script grammar -> API-ready chunks + pauses.

Scripts stay engine-neutral ([pause:Xs], [breath]/[inhale]/[exhale], blank-line
paragraphs). Pauses and breath markers become programmatic silence; speech
between them becomes chunks, each prefixed with the delivery direction. No
other tag is ever added, and every residual ``[...]`` group in user speech text
is stripped: only the sanitized direction prefix may contain brackets.
"""

from __future__ import annotations

import os
import re

from core.elevenlabs_tts.direction import default_direction, sanitize_direction
from core.f5_tts.preprocessor import parse_script
from core.speech_engine import breath_silence_sec
from core.text_utils import expand_text

HARD_MAX_CHARS = 9000  # < 10,000-char eleven_v4 request limit, incl. direction
DEFAULT_CHUNK_MAX_CHARS = 1500
_TAG_RE = re.compile(r"\[[^\]]*\]")
_SENTENCE_END = re.compile(r"[.!?…]+[\"')]?\s+")


def chunk_max_chars() -> int:
    raw = os.environ.get("MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS", str(DEFAULT_CHUNK_MAX_CHARS))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS must be an integer, got {raw!r}"
        ) from None
    return max(200, min(HARD_MAX_CHARS, value))


DIRECTION_MODES = ("every", "after_long_pause", "first")
DEFAULT_DIRECTION_PAUSE_SEC = 3.0
MAX_SENTENCE_GAP_MS = 5000


def sentence_gap_ms() -> int:
    """Experiment: silence inserted between sentences (0 = off, chunks stay packed)."""
    raw = os.environ.get("MOODSCAPE_ELEVENLABS_SENTENCE_GAP_MS", "0").strip() or "0"
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"MOODSCAPE_ELEVENLABS_SENTENCE_GAP_MS must be an integer, got {raw!r}"
        ) from None
    return max(0, min(MAX_SENTENCE_GAP_MS, value))


def direction_mode() -> str:
    """Experiment: which chunks get the direction prefix (default every)."""
    raw = (os.environ.get("MOODSCAPE_ELEVENLABS_DIRECTION_MODE") or "every").strip().lower()
    if raw not in DIRECTION_MODES:
        raise ValueError(
            f"MOODSCAPE_ELEVENLABS_DIRECTION_MODE must be one of {DIRECTION_MODES}, got {raw!r}"
        )
    return raw


def direction_pause_sec() -> float:
    raw = os.environ.get("MOODSCAPE_ELEVENLABS_DIRECTION_PAUSE_SEC", "")
    if not raw.strip():
        return DEFAULT_DIRECTION_PAUSE_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        raise ValueError(
            f"MOODSCAPE_ELEVENLABS_DIRECTION_PAUSE_SEC must be a number, got {raw!r}"
        ) from None


def _apply_direction(out: list[dict], tag: str, mode: str, min_pause: float) -> None:
    """Prefix ``tag`` onto speech chunks per ``mode`` (in place). The first chunk
    always gets it; ``after_long_pause`` re-applies it after >= min_pause of silence."""
    silence, first = 0.0, True
    for seg in out:
        if seg["type"] == "pause":
            silence += seg["duration_sec"]
            continue
        if mode == "every" or first or (mode == "after_long_pause" and silence >= min_pause):
            seg["text"] = f"{tag} {seg['text']}"
        first, silence = False, 0.0


def _split_sentences(text: str) -> list[str]:
    """Split after sentence punctuation, never inside [...]."""
    parts: list[str] = []
    depth, start, i = 0, 0, 0
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
        elif depth == 0:
            m = _SENTENCE_END.match(text, i)
            if m:
                parts.append(text[start:m.end()].strip())
                start = i = m.end()
                continue
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _split_words(sentence: str, limit: int) -> list[str]:
    out, buf = [], ""
    for word in sentence.split():
        if len(word) > limit:
            if buf:
                out.append(buf)
                buf = ""
            slices = [word[i:i + limit] for i in range(0, len(word), limit)]
            out.extend(slices[:-1])
            buf = slices[-1]
            continue
        if buf and len(buf) + 1 + len(word) > limit:
            out.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}".strip()
    if buf:
        out.append(buf)
    return out


def _pack(sentences: list[str], limit: int) -> list[str]:
    chunks, cur = [], ""
    for sentence in sentences:
        pieces = _split_words(sentence, limit) if len(sentence) > limit else [sentence]
        for piece in pieces:
            if cur and len(cur) + 1 + len(piece) > limit:
                chunks.append(cur)
                cur = piece
            else:
                cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def prepare_segments(script: str, content_type: str = "meditation",
                     direction: str | None = None) -> list[dict]:
    from core.content_profiles import get_profile

    tag = sanitize_direction(direction) if direction else default_direction()
    limit = chunk_max_chars() - len(tag) - 1
    gap_sec = sentence_gap_ms() / 1000.0
    mode, min_pause = direction_mode(), direction_pause_sec()
    raw = parse_script(
        script, paragraph_pause_sec=get_profile(content_type)["elevenlabs_paragraph_pause_sec"]
    )

    out: list[dict] = []
    speech_buf: list[str] = []

    def flush() -> None:
        if not speech_buf:
            return
        text = " ".join(speech_buf)
        speech_buf.clear()
        # Joining can reassemble tags across segments; strip again, then drop
        # any stray bracket so nothing bracketed but the direction reaches the API.
        text = _TAG_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", re.sub(r"[\[\]]", " ", text)).strip()
        if not text:
            return
        sentences = _split_sentences(text)
        if gap_sec > 0:
            # One chunk per sentence with programmatic silence between them.
            pieces = [p for s in sentences
                      for p in (_split_words(s, limit) if len(s) > limit else [s])]
        else:
            pieces = _pack(sentences, limit)
        for i, piece in enumerate(pieces):
            if i and gap_sec > 0:
                out.append({"type": "pause", "duration_sec": gap_sec})
            out.append({"type": "speech", "text": piece})

    for seg in raw:
        kind = seg["type"]
        if kind == "speech":
            text = re.sub(r"\s+", " ", _TAG_RE.sub(" ", expand_text(seg["text"]))).strip()
            if text:
                speech_buf.append(text)
        elif kind == "pause":
            flush()
            out.append({"type": "pause", "duration_sec": float(seg["duration_sec"])})
        elif kind == "breath":
            flush()
            out.append({"type": "pause", "duration_sec": breath_silence_sec(seg["subtype"])})
        # "voice" segments: one ElevenLabs voice per render — ignored.
    flush()
    _apply_direction(out, tag, mode, min_pause)
    return out


def estimate_billed_chars(segments: list[dict]) -> int:
    """Characters sent to the API (what ElevenLabs bills) for these segments."""
    return sum(len(s["text"]) for s in segments if s["type"] == "speech")
