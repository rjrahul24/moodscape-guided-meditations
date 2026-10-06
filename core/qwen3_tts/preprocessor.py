"""Script -> segments for Qwen3-TTS.

Uses the shared F5 script grammar ([pause:Xs], [breath], paragraph pauses by
content type) and F5's text normalisation, wrapped so Qwen can diverge later.
Qwen3-TTS is autoregressive with long context, so chunks can be longer than
F5's; the size is env-tunable (MOODSCAPE_QWEN3_CHUNK_CHARS, default 200).
"""

import re

from core.f5_tts.preprocessor import normalize_for_f5, parse_script
from core.qwen3_tts.presets import chunk_chars_from_env

_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def normalize_for_qwen3(text: str) -> str:
    text = normalize_for_f5(text)
    # Lowercase all-caps words to avoid letter-by-letter pronunciation
    text = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group(0).lower(), text)
    return text


def _split_long(sentence: str, max_chars: int) -> list[str]:
    parts, current = [], ""
    for word in sentence.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    chunks, current = [], ""
    for sentence in (s.strip() for s in _SENTENCES.split(text) if s.strip()):
        pieces = [sentence] if len(sentence) <= max_chars else _split_long(sentence, max_chars)
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def prepare_segments(script: str, content_type: str = "meditation") -> list[dict]:
    from core.content_profiles import get_profile

    pause = get_profile(content_type)["f5_paragraph_pause_sec"]
    max_chars = chunk_chars_from_env()
    out: list[dict] = []
    voice = None
    for seg in parse_script(script, paragraph_pause_sec=pause):
        if seg["type"] == "voice":
            voice = seg["voice"]
        elif seg["type"] == "speech":
            for chunk in split_into_chunks(normalize_for_qwen3(seg["text"]), max_chars):
                out.append({"type": "speech", "text": chunk, "voice": voice})
        else:
            out.append(seg)
    return out
