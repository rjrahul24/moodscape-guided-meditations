"""F5-TTS preprocessing — character-aware chunking for F5-TTS's 30s context window.

F5-TTS infers ~7–10 seconds of audio per 200 characters at meditative speed (0.75).
The 200-character limit keeps every chunk safely within the model's context window.

Key difference from kokoro_tts/preprocessor.py:
- No IPA phoneme injection (F5-TTS does its own G2P from raw text)
- No text expansion (digits, abbreviations) — pass natural prose as-is
- Character-count chunking instead of token-count chunking
"""

import re

MAX_CHUNK_CHARS = 200
_PARAGRAPH_PAUSE_SEC = 6.5


def parse_script(script: str) -> list[dict]:
    """Parse a meditation script into speech and pause segments.

    Supports the same pause marker syntax as kokoro_tts/preprocessor.py:
        [pause:Xs]          — explicit pause in seconds
        [N second pause]    — alternate explicit pause format
        [breath] / [inhale] / [exhale]  — 1.2s breath pause
        double newline      — paragraph break (6.5s pause)

    Returns a list of dicts with keys:
        {"type": "speech", "text": str}
        {"type": "pause", "duration_sec": float}
    """
    if not script or not script.strip():
        return []

    # Normalise paragraph breaks to pause markers
    script = re.sub(r'\n\n+', f' [pause:{_PARAGRAPH_PAUSE_SEC}s] ', script)

    # Normalise "[N second pause]" variants → "[pause:Ns]"
    script = re.sub(
        r'\[(\d+(?:\.\d+)?)\s*(?:second|sec|s)\s*pause\]',
        lambda m: f'[pause:{m.group(1)}s]',
        script,
        flags=re.IGNORECASE,
    )

    # Normalise breath markers → short pause
    script = re.sub(r'\[(?:breath|inhale|exhale)\]', '[pause:1.2s]', script, flags=re.IGNORECASE)

    # Split on pause markers; odd-indexed groups are durations
    parts = re.split(r'\[pause:(\d+(?:\.\d+)?)s\]', script)
    segments = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            text = part.strip()
            if text:
                segments.append({'type': 'speech', 'text': text})
        else:
            duration = float(part)
            if duration > 0:
                segments.append({'type': 'pause', 'duration_sec': duration})
    return segments


def split_into_chunks(text: str) -> list[str]:
    """Split a speech block into chunks of at most MAX_CHUNK_CHARS characters.

    Splits at sentence boundaries (.!?…) to preserve natural prosody.
    Falls back to the full text if it cannot be split (never returns empty).
    """
    raw = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks: list[str] = []
    buf = ''
    for sent in raw:
        candidate = (buf + ' ' + sent).strip() if buf else sent
        if len(candidate) > MAX_CHUNK_CHARS and buf:
            chunks.append(buf.strip())
            buf = sent
        else:
            buf = candidate
    if buf:
        chunks.append(buf.strip())
    return chunks if chunks else [text]


def prepare_segments(script: str) -> list[dict]:
    """Full preprocessing pipeline for F5-TTS.

    Parses the script into pause/speech segments, then splits each speech
    block into ≤200-character chunks to stay within F5-TTS's 30s context window.

    Returns the same segment dict format as kokoro_tts/preprocessor.py so the
    pipeline's synthesize() call is engine-agnostic.
    """
    raw_segments = parse_script(script)
    expanded: list[dict] = []
    for seg in raw_segments:
        if seg['type'] == 'speech':
            chunks = split_into_chunks(seg['text'])
            expanded.extend({'type': 'speech', 'text': c} for c in chunks)
        else:
            expanded.append(seg)
    return expanded
