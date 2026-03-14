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
        {"type": "speech", "text": str, "voice": str|None}
        {"type": "pause", "duration_sec": float}
        {"type": "voice", "voice": str}
    """
    if not script or not script.strip():
        return []

    # Normalise "[N second pause]" variants → "[pause:Ns]"  (run before \n\n
    # conversion so explicit tags are already in canonical form when we check
    # whether a "paragraph" is pause-only below)
    script = re.sub(
        r'\[(\d+(?:\.\d+)?)\s*(?:second|sec|s)\s*pause\]',
        lambda m: f'[pause:{m.group(1)}s]',
        script,
        flags=re.IGNORECASE,
    )

    # Normalise breath markers → short pause (same reason: do before \n\n pass)
    script = re.sub(r'\[(?:breath|inhale|exhale)\]', '[pause:1.2s]', script, flags=re.IGNORECASE)

    # Convert paragraph breaks to pause markers — BUT only when both sides of
    # the break are actual speech.  If either the block before or after a \n\n
    # boundary contains only a pause marker (user put [pause:Xs] on its own
    # line for readability), those newlines are just formatting; skip the extra
    # paragraph-level pause so the explicit duration is honoured exactly.
    _PAUSE_ONLY = re.compile(r'^\[pause:\d+(?:\.\d+)?s\]$')
    blocks = re.split(r'\n\n+', script)
    parts_joined: list[str] = []
    for i, block in enumerate(blocks):
        parts_joined.append(block)
        if i < len(blocks) - 1:
            cur_is_pause = _PAUSE_ONLY.fullmatch(block.strip()) is not None
            nxt_is_pause = _PAUSE_ONLY.fullmatch(blocks[i + 1].strip()) is not None
            if cur_is_pause or nxt_is_pause:
                # At least one side is a pause tag — the \n\n is just formatting
                parts_joined.append(' ')
            else:
                parts_joined.append(f' [pause:{_PARAGRAPH_PAUSE_SEC}s] ')
    script = ''.join(parts_joined)

    # Split on pause AND voice markers. 
    # Group 1: duration, Group 2: voice name.
    parts = re.split(r'\[pause:(\d+(?:\.\d+)?)s\]|\[voice:([^\]]+)\]', script)
    segments = []
    # With two groups, re.split returns [text, dur1, voice1, text, dur2, voice2, ...]
    # So we step by 3.
    for i in range(0, len(parts), 3):
        text = parts[i].strip()
        if text:
            segments.append({'type': 'speech', 'text': text})
        
        if i + 1 < len(parts) and parts[i+1] is not None:
            duration = float(parts[i+1])
            if duration > 0:
                segments.append({'type': 'pause', 'duration_sec': duration})
        
        if i + 2 < len(parts) and parts[i+2] is not None:
            voice = parts[i+2].strip()
            segments.append({'type': 'voice', 'voice': voice})

    # Merge back-to-back pause segments (can still arise from edge cases such
    # as consecutive cue tags) by keeping only the longest of any run.
    merged: list[dict] = []
    for seg in segments:
        if seg['type'] == 'pause' and merged and merged[-1]['type'] == 'pause':
            merged[-1]['duration_sec'] = max(merged[-1]['duration_sec'], seg['duration_sec'])
        else:
            merged.append(seg)
    return merged


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
    current_voice = None
    
    for seg in raw_segments:
        if seg['type'] == 'voice':
            current_voice = seg['voice']
        elif seg['type'] == 'speech':
            chunks = split_into_chunks(seg['text'])
            for c in chunks:
                expanded.append({
                    'type': 'speech', 
                    'text': c, 
                    'voice': current_voice
                })
        else:
            expanded.append(seg)
    return expanded
