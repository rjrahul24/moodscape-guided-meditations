"""Text preprocessing to optimise scripts for Kokoro TTS meditation delivery."""

import re

# ── Abbreviation & number expansion tables ──────────────────────────

# Mapping for common abbreviations in meditation scripts
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

# Ordinal and cardinal number words
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
    """Expand digits and abbreviations to their spoken equivalents.

    Prevents Kokoro's G2P engine from mispronouncing or rushing through
    numeric and abbreviated tokens.  Must be called before any TTS inference.
    """
    # Expand abbreviations first (before number expansion)
    for pattern, replacement in _ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Replace standalone integers (0–999) with word equivalents.
    # Uses a word-boundary assertion to avoid partial matches inside
    # larger tokens (e.g. "432Hz" is handled by the Hz abbreviation above).
    text = re.sub(r'\b\d{1,3}\b', _replace_number, text)

    return text



def preprocess_for_meditation(text: str) -> str:
    """Optimise text for calm, meditation-style Kokoro TTS delivery.

    Rules derived from Kokoro TTS optimisation research:
    - Commas = micro-pauses
    - Ellipsis (...) = reflective breath (longer internal pause)
    - Periods = full stop + intonation reset
    - Em dashes = slight rhythmic break
    - Semicolons = brief pause with anticipation

    Applied to each speech segment's text *before* it is sent to the TTS
    engine.  Does NOT alter [pause:Xs] markers — those are handled by the
    script parser.
    """
    # Expand digits and abbreviations to spoken equivalents
    text = expand_for_tts(text)

    # Ensure ellipsis is properly formatted (not separate dots)
    text = re.sub(r'\.{2,}', '...', text)

    # Add breathing room: ensure there is always trailing punctuation
    # before [pause:Xs] markers for natural trailing prosody
    text = re.sub(r'(\w)\s*(\[pause:)', r'\1... \2', text)

    # Normalise excessive whitespace
    text = re.sub(r'  +', ' ', text)

    return text.strip()


def validate_chunk_length(text: str, max_tokens: int = 200) -> list[str]:
    """Split text into chunks that do not exceed Kokoro's sweet spot.

    Kokoro performs best with 100-200 tokens per chunk.
    Above ~400-510 tokens it starts rushing.  This function enforces a hard
    ceiling of 400 tokens per chunk.
    """
    HARD_CEILING = 400

    words = text.split()
    estimated_tokens = int(len(words) * 1.3)

    if estimated_tokens <= max_tokens:
        return [text]

    # Split at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = int(len(sentence.split()) * 1.3)

        # If a single sentence exceeds the hard ceiling, force-split by word count
        if sentence_tokens > HARD_CEILING:
            # Flush current chunk first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0

            s_words = sentence.split()
            max_words = int(HARD_CEILING / 1.3)
            for i in range(0, len(s_words), max_words):
                chunks.append(' '.join(s_words[i:i + max_words]))
            continue

        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
