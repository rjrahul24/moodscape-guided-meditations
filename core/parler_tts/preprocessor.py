"""Parler TTS preprocessing pipeline — script parsing, text normalization, and chunking.

Consolidates all preprocessing steps tailored to Parler TTS:
  1. Script parsing: [pause:Xs] markers and paragraph breaks → structured segments
  2. Parler-specific text expansion
  3. Preprocessing for meditation
  4. Sentence splitting and token estimation
"""

import re

# We rely on an explicit paragraph pause for meditation spacing
_PARAGRAPH_PAUSE_SEC = 6.5

# Parler TTS Large v1 uses a DAC codec at 44100 Hz with ~86 tokens/second.
# We use 50 tokens/word for safety margin.
_TOKENS_PER_WORD = 50
_MIN_NEW_TOKENS = 256   # ~3 seconds minimum
_MAX_NEW_TOKENS = 2048  # ~24 seconds ceiling per chunk

# Voice-drift prevention: sentences longer than this are split at clause boundaries
_MAX_WORDS_PER_CHUNK = 25
MIN_SENTENCE_WORDS = 6  # Parler needs more context than Kokoro for good prosody


# ── Script parsing ───────────────────────────────────────────────────────

def parse_script(script: str) -> list[dict]:
    """Parse a meditation script into speech and pause segments.

    Supported markers:
        [pause:Xs]   — explicit pause of X seconds (int or float)
        \\n\\n         — paragraph break, treated as a 6.5s pause

    Returns a list of dicts:
        {"type": "speech", "text": "..."}
        {"type": "pause", "duration_sec": float}
    """
    if not script or not script.strip():
        return []

    # First, replace double newlines with a pause marker so we handle them uniformly
    script = re.sub(r'\n\n+', f' [pause:{_PARAGRAPH_PAUSE_SEC}s] ', script)

    # Split on pause markers, capturing the duration
    pause_pattern = r'\[pause:(\d+(?:\.\d+)?)s\]'
    parts = re.split(pause_pattern, script)

    # re.split with a capture group alternates: [text, duration, text, duration, ...]
    segments = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text chunk
            text = part.strip()
            if text:
                segments.append({"type": "speech", "text": text})
        else:
            # Captured pause duration
            duration = float(part)
            if duration > 0:
                # Collapse consecutive pauses: merge with previous if it's also a pause
                if segments and segments[-1]["type"] == "pause":
                    segments[-1]["duration_sec"] += duration
                else:
                    segments.append({"type": "pause", "duration_sec": duration})

    return segments


# ── Text expansion and prosody ───────────────────────────────────────────

_ABBREV_MAP = {
    r'\bsec\b': 'seconds',
    r'\bmin\b': 'minutes',
    r'\bhr\b': 'hours',
    r'\bvs\b': 'versus',
    r'\betc\b': 'et cetera',
    r'\be\.g\b': 'for example',
    r'\bi\.e\b': 'that is',
    r'\bHz\b': 'hertz',
    r'\bkHz\b': 'kilohertz',
}

_ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
         'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
         'sixteen', 'seventeen', 'eighteen', 'nineteen']
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
         'eighty', 'ninety']

def _int_to_words(n: int) -> str:
    """Convert an integer 0–999 to English words."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (('-' + _ONES[ones]) if ones else '')
    hundreds, remainder = divmod(n, 100)
    rest = (' and ' + _int_to_words(remainder)) if remainder else ''
    return _ONES[hundreds] + ' hundred' + rest

def _replace_number(match: re.Match) -> str:
    """Replace a matched digit sequence with English words."""
    try:
        n = int(match.group(0))
        if 0 <= n <= 999:
            return _int_to_words(n) if n != 0 else 'zero'
    except ValueError:
        pass
    return match.group(0)

def expand_for_tts(text: str) -> str:
    """Expand digits and abbreviations to their spoken equivalents."""
    for pattern, replacement in _ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,3}\b', _replace_number, text)
    return text

def preprocess_for_meditation(text: str) -> str:
    """Optimise text for calm, meditation-style Parler TTS delivery."""
    text = expand_for_tts(text)
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'(\w)\s*(\[pause:)', r'\1... \2', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()

def prepare_segments(script: str) -> list[dict]:
    """Parse script -> preprocess speech segments."""
    segments = parse_script(script)
    for seg in segments:
        if seg["type"] == "speech":
            seg["text"] = preprocess_for_meditation(seg["text"])
    return segments


# ── Sentence splitting and token estimation ──────────────────────────────

def estimate_max_tokens(text: str) -> int:
    """Estimate max_new_tokens for a text chunk based on word count.
    
    Heuristic: meditation speech at 0.85x speed ≈ 120 words/min ≈ 2 words/sec
    → 1 word ≈ 0.5s ≈ 43 tokens.  We use 50 tokens/word for safety margin
    and clamp to [_MIN_NEW_TOKENS, _MAX_NEW_TOKENS].
    """
    n_words = len(text.split())
    estimate = n_words * _TOKENS_PER_WORD
    return max(_MIN_NEW_TOKENS, min(estimate, _MAX_NEW_TOKENS))

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries.

    Handles standard endings (.!?) and ellipsis (...).
    Short sentences are merged for better Parler prosody.
    Long sentences are further split at clause boundaries
    (commas, semicolons, em-dashes) to limit autoregressive drift.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    raw = [p for p in parts if p.strip()]

    if len(raw) <= 1:
        return _cap_sentence_length(raw)

    merged: list[str] = []
    carry = ""
    for i, s in enumerate(raw):
        if carry:
            s = carry + " " + s
            carry = ""
        if len(s.split()) < MIN_SENTENCE_WORDS and i < len(raw) - 1:
            carry = s
        else:
            merged.append(s)
    if carry:
        if merged:
            merged[-1] = merged[-1] + " " + carry
        else:
            merged.append(carry)

    return _cap_sentence_length(merged)

def _cap_sentence_length(sentences: list[str]) -> list[str]:
    """Split sentences exceeding _MAX_WORDS_PER_CHUNK at clause boundaries.

    Tries commas, semicolons, and em-dashes first. Falls back to splitting
    at the word limit if no clause boundary is found.
    """
    result: list[str] = []
    for sent in sentences:
        words = sent.split()
        if len(words) <= _MAX_WORDS_PER_CHUNK:
            result.append(sent)
            continue

        # Try splitting at clause boundaries (comma, semicolon, em-dash)
        clause_parts = re.split(r"(?<=[,;—–])\s+", sent)
        if len(clause_parts) > 1:
            # Greedily merge clause parts up to the word limit
            chunk = ""
            for part in clause_parts:
                candidate = (chunk + " " + part).strip() if chunk else part
                if len(candidate.split()) > _MAX_WORDS_PER_CHUNK and chunk:
                    result.append(chunk)
                    chunk = part
                else:
                    chunk = candidate
            if chunk:
                result.append(chunk)
        else:
            # No clause boundaries — hard split at word limit
            for i in range(0, len(words), _MAX_WORDS_PER_CHUNK):
                chunk = " ".join(words[i:i + _MAX_WORDS_PER_CHUNK])
                result.append(chunk)

    return result

def adjust_description_for_speed(description: str, speed: float) -> str:
    """Modify the voice description to reflect the requested speed.

    Parler TTS doesn't have a numeric speed parameter like Kokoro.
    Instead, we inject pacing keywords into the description.
    """
    # Remove any existing speed-related phrases to avoid conflicts
    speed_phrases = [
        "very slow",
        "slow",
        "moderate speed",
        "slightly slow",
        "fast",
        "slightly fast",
        "very fast",
        "normal speed",
    ]
    cleaned = description
    for phrase in speed_phrases:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)

    # Map numeric speed to descriptive pacing with aggressive phrasing for meditation
    if speed <= 0.65:
        pace = "extremely slowly, with a very calm, monotone delivery, pausing frequently, whispery, and close-sounding"
    elif speed <= 0.75:
        pace = "very slowly, with a calm, monotone delivery, pausing frequently, and close-sounding"
    elif speed <= 0.85:
        pace = "slowly, with a gentle, calm delivery and comforting tone"
    elif speed <= 0.95:
        pace = "moderate, steady pacing with a soothing presence"
    else:
        pace = "natural, conversational pacing"

    return f"{cleaned.rstrip('. ')}. Speaking with {pace}."
