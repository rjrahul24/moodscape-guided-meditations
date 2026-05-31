"""IndexTTS-2 preprocessing — text normalization and character-aware chunking.

IndexTTS-2 is an autoregressive zero-shot TTS model using BigVGANv2 as vocoder.
It performs its own G2P from raw text (like F5-TTS), so no IPA injection is needed.

Key differences from F5-TTS preprocessor:
  - 250-char chunk limit (vs F5's 300) — autoregressive models are more prone to
    hallucination on longer inputs than diffusion-based models
  - 3.5s paragraph pause (vs F5's 3.0) — BigVGANv2's richer acoustic tails benefit
    from slightly more separation between paragraphs
  - Same shared text expansion (digits, abbreviations) via core.text_utils
  - Same punctuation normalization rules (colons, ellipses, em-dashes, hyphens)
"""

import logging
import re

from core.text_utils import expand_text

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 360  # ≈180 BPE tokens, matching the engine's max_text_tokens
                       # _per_segment. IndexTTS-2 has no context carry-forward
                       # between segments, so fewer, longer chunks reduce emotion
                       # drift across long meditation scripts. Sentence-boundary
                       # snapping below this cap keeps prosodic arcs intact.
_PARAGRAPH_PAUSE_SEC = 3.5

# Meditation-domain pronunciation lexicon — applied BEFORE expand_text() and
# punctuation normalization. SentencePiece BPE fractures these terms into
# multiple sub-word tokens, leading to fricative artifacts or letter-by-letter
# reads. Phonetic spellings yield clean, deliberate pronunciation.
INDEX_MEDITATION_LEXICON = {
    r"\bOm\b":         "ohm",
    r"\bAum\b":        "ohm",
    r"\bvipassana\b":  "vi-pah-sana",
    r"\bpranayama\b":  "prah-nah-yama",
    r"\bmetta\b":      "meh-tah",
    r"\bsamadhi\b":    "sah-mah-dee",
    r"\bnamaste\b":    "nah-mas-tay",
    r"\bchakra\b":     "chah-kra",
    r"\bkundalini\b":  "koon-da-lee-nee",
    r"\bmudra\b":      "moo-drah",
    r"\bsavasana\b":   "shah-vah-sana",
    r"\bshavasana\b":  "shah-vah-sana",
    r"\bkoan\b":       "koh-an",
    r"\bzazen\b":      "zah-zen",
    r"\bdharma\b":     "dar-ma",
    r"\bsangha\b":     "sang-ha",
    r"\btantra\b":     "tahn-tra",
    r"\bsutra\b":      "soo-tra",
}


def _apply_meditation_lexicon(text: str) -> str:
    """Replace meditation-domain terms with phonetic spellings (case-insensitive)."""
    for pattern, replacement in INDEX_MEDITATION_LEXICON.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_for_indextts(text: str) -> str:
    """Normalize text for IndexTTS-2's G2P and prosody model.

    Applies shared digit/abbreviation expansion, then IndexTTS-2-specific fixes:
    - Colons → commas (IndexTTS-2 ignores colons, producing no pause)
    - Ellipses → periods (autoregressive decoder does not reliably pause for ellipses)
    - Em-dashes → commas (not reliably handled by IndexTTS-2)
    - Hyphens in compounds removed: 'well-being' → 'wellbeing'
    """
    # Shared expansion: digits → words, abbreviations → full forms
    text = expand_text(text)

    # Punctuation normalization (same rules as F5-TTS)
    text = re.sub(r':', ',', text)              # colons → commas
    text = re.sub(r'\.{2,}', '.', text)         # ellipses → single period
    text = re.sub(r'—|–|--+', ',', text)        # em/en-dashes → commas
    # Remove hyphens in compound words (not between digits — those are already
    # handled by expand_text's hyphenated number expansion)
    text = re.sub(r'(?<=[a-zA-Z])-(?=[a-zA-Z])', '', text)

    # Meditation lexicon LAST — phonetic spellings use hyphens for syllabification
    # and must survive the compound-hyphen stripper above.
    text = _apply_meditation_lexicon(text)

    return text


def parse_script(script: str) -> list[dict]:
    """Parse a meditation script into speech and pause segments.

    Supports the same pause marker syntax as kokoro_tts/preprocessor.py and
    f5_tts/preprocessor.py:
        [pause:Xs]          — explicit pause in seconds
        [N second pause]    — alternate explicit pause format
        [breath] / [inhale] / [exhale]  — 1.2s breath pause
        double newline      — paragraph break (3.5s pause)
        [voice:phase_name]  — switch reference voice phase

    Returns a list of dicts with keys:
        {"type": "speech", "text": str, "voice": str|None}
        {"type": "pause", "duration_sec": float}
        {"type": "voice", "voice": str}
        {"type": "breath", "subtype": str}
    """
    if not script or not script.strip():
        return []

    # Normalise "[N second pause]" variants → "[pause:Ns]"
    script = re.sub(
        r'\[(\d+(?:\.\d+)?)\s*(?:second|sec|s)\s*pause\]',
        lambda m: f'[pause:{m.group(1)}s]',
        script,
        flags=re.IGNORECASE,
    )

    # Normalise breath markers → tagged breath markers
    script = re.sub(
        r'\[(breath|inhale|exhale)\]',
        lambda m: f'[breath:{m.group(1).lower()}]',
        script,
        flags=re.IGNORECASE,
    )

    # Convert paragraph breaks to pause markers — but only when both sides of
    # the break are actual speech. If either side contains only a pause marker,
    # the newlines are just formatting — skip the extra paragraph-level pause.
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
                parts_joined.append(f' [pause:{_PARAGRAPH_PAUSE_SEC}s] ')
    script = ''.join(parts_joined)

    # Split on pause, voice, AND breath markers.
    # Group 1: pause duration, Group 2: voice name, Group 3: breath subtype.
    parts = re.split(
        r'\[pause:(\d+(?:\.\d+)?)s\]|\[voice:([^\]]+)\]|\[breath:(breath|inhale|exhale)\]',
        script,
    )
    segments = []
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

    # Merge back-to-back pause segments (keep longest)
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
    """Full preprocessing pipeline for IndexTTS-2.

    Parses the script into pause/speech segments, then splits each speech
    block into ≤MAX_CHUNK_CHARS-character chunks to prevent autoregressive
    hallucination on long inputs.

    Returns the same segment dict format as kokoro_tts and f5_tts preprocessors
    so the pipeline's synthesize() call is engine-agnostic.
    """
    raw_segments = parse_script(script)
    expanded: list[dict] = []
    current_voice = None

    for seg in raw_segments:
        if seg['type'] == 'voice':
            current_voice = seg['voice']
        elif seg['type'] == 'speech':
            normalized = normalize_for_indextts(seg['text'])
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
