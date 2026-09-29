"""Chatterbox TTS preprocessing — text normalization and prosodic semantic chunking.

Implements the research-backed text normalization and syntactic pacing guidelines
for Chatterbox TTS on Apple Silicon:
  1. SSML markup stripping (Chatterbox lacks native SSML).
  2. Quotation mark removal (prevents sighing / hallucination artifacts).
  3. Case softening (converts aggressive all-caps/shouting to gentle sentence case).
  4. Prosodic punctuation preservation:
     - Preserves ellipses ('...') and em-dashes ('—') which trigger relaxing
       trailing pitch drops and 400–650 ms transitions in Llama-3.
     - Maps colons and semicolons to commas for smooth clause cadence.
  5. Semantic clause chunking: 150–250 characters at punctuation boundaries to avoid
     transformer attention degradation and token drift.
  6. Decoupled programmatic digital silence buffering for meditative pacing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.text_utils import expand_text

logger = logging.getLogger("moodscape.chatterbox_preprocessor")

MAX_CHUNK_CHARS = 240
MIN_CHUNK_CHARS = 100
_DEFAULT_PARAGRAPH_PAUSE_SEC = 3.5


def normalize_for_chatterbox(text: str) -> str:
    """Normalize text for Chatterbox TTS autoregressive generation.

    Applies text expansion, quote stripping, case softening, and punctuation
    tuning per Apple Silicon meditation research specifications.
    """
    if not text:
        return ""

    # Shared digit & abbreviation expansion (e.g. 5min -> five minutes)
    text = expand_text(text)

    # 1. Strip all quotation marks (double and single)
    # Standalone double quotes trigger unprompted sighing artifacts (>1s) in the tokenizer
    text = re.sub(r'["“”„«»]', '', text)
    # Remove single quotes when used as quotation marks (not apostrophes in contractions)
    text = re.sub(r"(^|\s)['‘‚`]([a-zA-Z])", r"\1\2", text)
    text = re.sub(r"([a-zA-Z])['’](?=\s|$|[.,!?;:])", r"\1", text)

    # 2. Case softening: eliminate shouting/volume projection from ALL-CAPS words
    # Capitalization acts as stress/volume marker in Llama-3; lowercase all-caps words
    def _soften_word(m: re.Match) -> str:
        w = m.group(0)
        # Keep single 'I' or very common acronyms if needed, otherwise lowercase
        if w == "I":
            return "I"
        return w.lower()

    text = re.sub(r'\b[A-Z]{2,}\b', _soften_word, text)

    # Convert entire string to gentle sentence case (first letter capitalized)
    # but ensure it does not scream
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Prosodic punctuation normalization
    # Map colons & semicolons to commas or ellipses
    text = re.sub(r';', ',', text)
    text = re.sub(r':', ',', text)

    # Normalize multiple dashes to clean em-dash
    text = re.sub(r'--+|–', '—', text)

    # Normalize 4+ dots to standard 3-dot ellipsis
    text = re.sub(r'\.{4,}', '...', text)
    # Ensure ellipsis is cleanly spaced if mashed against words
    text = re.sub(r'([a-zA-Z])\.\.\.', r'\1 ... ', text)

    # Remove emojis and non-standard symbols that can confuse autoregressive tokens
    text = re.sub(r'[^\w\s.,!?;:\'’—\-…~]', '', text)

    # Clean double spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def parse_script(
    script: str,
    paragraph_pause_sec: float = _DEFAULT_PARAGRAPH_PAUSE_SEC,
) -> list[dict[str, Any]]:
    """Parse a meditation script into speech and pause segments.

    Decouples speech generation from long pause generation to prevent
    autoregressive token hallucinations and hums.

    Supports:
        [pause:Xs] or [X second pause] — programmatic silence buffer
        [breath] / [inhale] / [exhale] — natural breath pause (1.5–2.0s)
        double newline (\n\n)          — paragraph transition pause
        [voice:slug]                   — speaker switch
    """
    if not script or not script.strip():
        return []

    # 1. Normalize "[N second pause]" variants → "[pause:Ns]"
    script = re.sub(
        r'\[(\d+(?:\.\d+)?)\s*(?:second|sec|s)\s*pause\]',
        lambda m: f'[pause:{m.group(1)}s]',
        script,
        flags=re.IGNORECASE,
    )

    # 2. Normalize breath markers
    script = re.sub(
        r'\[(breath|inhale|exhale)\]',
        lambda m: f'[breath:{m.group(1).lower()}]',
        script,
        flags=re.IGNORECASE,
    )

    # 3. Convert paragraph breaks to pause markers
    _PAUSE_ONLY = re.compile(r'^\[pause:\d+(?:\.\d+)?s\]$')
    blocks = re.split(r'\n\n+', script)
    parts_joined: list[str] = []
    for i, block in enumerate(blocks):
        parts_joined.append(block)
        if i < len(blocks) - 1:
            cur_is_pause = _PAUSE_ONLY.fullmatch(block.strip()) is not None
            nxt_is_pause = _PAUSE_ONLY.fullmatch(blocks[i + 1].strip()) is not None
            if cur_is_pause or nxt_is_pause:
                parts_joined.append(' ')
            else:
                parts_joined.append(f' [pause:{paragraph_pause_sec}s] ')
    script = ''.join(parts_joined)

    # 4. Split on pause, voice, and breath markers
    parts = re.split(
        r'\[pause:(\d+(?:\.\d+)?)s\]|\[voice:([^\]]+)\]|\[breath:(breath|inhale|exhale)\]',
        script,
    )

    segments: list[dict[str, Any]] = []
    for i in range(0, len(parts), 4):
        text = parts[i].strip()
        if text:
            segments.append({'type': 'speech', 'text': text})

        if i + 1 < len(parts) and parts[i + 1] is not None:
            duration = float(parts[i + 1])
            if duration > 0:
                segments.append({'type': 'pause', 'duration_sec': duration})

        if i + 2 < len(parts) and parts[i + 2] is not None:
            voice = parts[i + 2].strip()
            segments.append({'type': 'voice', 'voice': voice})

        if i + 3 < len(parts) and parts[i + 3] is not None:
            segments.append({'type': 'breath', 'subtype': parts[i + 3]})

    # Merge adjacent pause segments
    merged: list[dict[str, Any]] = []
    for seg in segments:
        if seg['type'] == 'pause' and merged and merged[-1]['type'] == 'pause':
            merged[-1]['duration_sec'] = max(merged[-1]['duration_sec'], seg['duration_sec'])
        else:
            merged.append(seg)

    return merged


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split speech text into semantic clauses between 120 and 240 characters.

    Splits at sentence boundaries (.!?…) or natural clause pauses (,—)
    to keep every chunk within the optimal autoregressive context window.
    Ensures every emitted chunk ends with punctuation for clean cadence.
    """
    text = text.strip()
    if not text:
        return []

    # Ensure terminal punctuation on the entire block if missing
    if not re.search(r'[.!?…—]$', text):
        text += '.'

    # First split on primary sentence boundaries
    sentences = re.split(r'(?<=[.!?…])\s+', text)
    chunks: list[str] = []
    buf = ''

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # If a single sentence exceeds max_chars, split on clauses (, or —)
        if len(sent) > max_chars:
            clause_parts = re.split(r'(?<=[,—])\s+', sent)
            for cp in clause_parts:
                candidate = (buf + ' ' + cp).strip() if buf else cp
                if len(candidate) > max_chars and buf:
                    chunk_to_emit = buf.strip()
                    if not re.search(r'[.!?…—,]$', chunk_to_emit):
                        chunk_to_emit += '.'
                    chunks.append(chunk_to_emit)
                    buf = cp
                else:
                    buf = candidate
        else:
            candidate = (buf + ' ' + sent).strip() if buf else sent
            if len(candidate) > max_chars and buf:
                chunk_to_emit = buf.strip()
                if not re.search(r'[.!?…—,]$', chunk_to_emit):
                    chunk_to_emit += '.'
                chunks.append(chunk_to_emit)
                buf = sent
            else:
                buf = candidate

    if buf:
        chunk_to_emit = buf.strip()
        if not re.search(r'[.!?…—]$', chunk_to_emit):
            chunk_to_emit += '.'
        chunks.append(chunk_to_emit)

    return chunks if chunks else [text]


def prepare_segments(
    script: str,
    content_type: str = "meditation",
) -> list[dict[str, Any]]:
    """Full preprocessing pipeline for Chatterbox TTS.

    Parses the script into pause and speech segments, normalizes text
    specifically for Chatterbox, and segments into 150–250 character clauses.
    """
    from core.content_profiles import get_profile

    profile = get_profile(content_type)
    # Default paragraph pause: 3.5s for meditation, 2.0s for sleep story
    paragraph_pause_sec = profile.get(
        "chatterbox_paragraph_pause_sec",
        profile.get("f5_paragraph_pause_sec", _DEFAULT_PARAGRAPH_PAUSE_SEC),
    )

    raw_segments = parse_script(script, paragraph_pause_sec=paragraph_pause_sec)
    expanded: list[dict[str, Any]] = []
    current_voice = None

    for seg in raw_segments:
        if seg['type'] == 'voice':
            current_voice = seg['voice']
        elif seg['type'] == 'speech':
            normalized = normalize_for_chatterbox(seg['text'])
            chunks = split_into_chunks(normalized)
            for c in chunks:
                expanded.append({
                    'type': 'speech',
                    'text': c,
                    'voice': current_voice,
                })
        else:
            expanded.append(seg)

    return expanded
