"""ElevenLabs preprocessing: shared script grammar -> API-ready chunks + pauses.

Scripts stay engine-neutral ([pause:Xs], [breath]/[inhale]/[exhale], blank-line
paragraphs). Pauses and breath markers become programmatic silence; speech
between them becomes chunks, each prefixed with the delivery direction. No
other tag is ever added.
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
_SENTENCE_END = re.compile(r"[.!?…]+[\"')]?\s+")


def chunk_max_chars() -> int:
    raw = os.environ.get("MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS", str(DEFAULT_CHUNK_MAX_CHARS))
    return max(200, min(HARD_MAX_CHARS, int(raw)))


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

    tag = sanitize_direction(direction) if direction is not None else default_direction()
    limit = chunk_max_chars() - len(tag) - 1
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
        for piece in _pack(_split_sentences(text), limit):
            out.append({"type": "speech", "text": f"{tag} {piece}"})

    for seg in raw:
        kind = seg["type"]
        if kind == "speech":
            speech_buf.append(expand_text(seg["text"]).strip())
        elif kind == "pause":
            flush()
            out.append({"type": "pause", "duration_sec": float(seg["duration_sec"])})
        elif kind == "breath":
            flush()
            out.append({"type": "pause", "duration_sec": breath_silence_sec(seg["subtype"])})
        # "voice" segments: one ElevenLabs voice per render — ignored.
    flush()
    return out


def estimate_billed_chars(segments: list[dict]) -> int:
    """Characters sent to the API (what ElevenLabs bills) for these segments."""
    return sum(len(s["text"]) for s in segments if s["type"] == "speech")
