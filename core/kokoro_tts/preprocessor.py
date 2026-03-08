"""Kokoro TTS preprocessing pipeline — script parsing, text normalization, and chunking.

Consolidates all preprocessing steps tailored to Kokoro-82M:
  1. Script parsing: [pause:Xs] markers and paragraph breaks → structured segments
  2. Text expansion: digits, abbreviations → spoken equivalents
  3. Meditation prosody: punctuation enhancement at natural phrasing boundaries
  4. Token-aware chunking: sentences merged into 100–150 token chunks
"""

import re

# ── Script parser constants ──────────────────────────────────────────────
# Paragraph breaks (double newline) → 6.5s pause for spacious meditation pacing
_PARAGRAPH_PAUSE_SEC = 6.5

# ── Text expansion tables ────────────────────────────────────────────────
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

# ── Token-aware chunking constants ───────────────────────────────────────
# Kokoro's sweet spot is 100–150 tokens; above ~200 it starts rushing.
MAX_CHUNK_TOKENS = 150
MIN_CHUNK_TOKENS = 15
HARD_CEILING_TOKENS = 400

# Kokoro produces poor output for very short utterances (<~20 tokens).
# Sentences shorter than this word count are merged with adjacent ones.
MIN_SENTENCE_WORDS = 4

# Minimum speed floor — Kokoro becomes distorted below 0.65
MIN_SPEED = 0.65

# Maximum consecutive words without a comma or period before a forced break
_BREATH_GROUP_MAX = 12

# Conjunctions used as insertion points for breath-group comma breaks
_BREAK_CONJUNCTIONS = frozenset({'and', 'or', 'but', 'as', 'while'})


# ── Script parsing ───────────────────────────────────────────────────────

def parse_script(script: str) -> list[dict]:
    """Parse a meditation script into speech and pause segments.

    Supported markers:
        [pause:Xs]         — explicit pause of X seconds (int or float)
        [N second pause]   — alias: [2 second pause] → [pause:2s]
        [N sec pause]      — alias: [30 sec pause] → [pause:30s]
        [breath]           — alias for a 1.2s breath pause
        [inhale]/[exhale]  — alias for a 1.2s breath pause
        \\n\\n               — paragraph break, treated as a 6.5s pause

    Returns a list of dicts:
        {"type": "speech", "text": "..."}
        {"type": "pause", "duration_sec": float}
    """
    if not script or not script.strip():
        return []

    script = re.sub(r'\n\n+', f' [pause:{_PARAGRAPH_PAUSE_SEC}s] ', script)

    # Normalize natural-language pause aliases → [pause:Xs]
    # Handles: [2 second pause], [0.5 second pause], [30 sec pause], [5s pause]
    script = re.sub(
        r'\[(\d+(?:\.\d+)?)\s*(?:second|sec|s)\s*pause\]',
        lambda m: f'[pause:{m.group(1)}s]',
        script,
        flags=re.IGNORECASE,
    )
    # Normalize breath/inhale/exhale → 1.2s pause (ellipsis-equivalent pacing)
    script = re.sub(r'\[(?:breath|inhale|exhale)\]', '[pause:1.2s]', script, flags=re.IGNORECASE)

    pause_pattern = r'\[pause:(\d+(?:\.\d+)?)s\]'
    parts = re.split(pause_pattern, script)

    segments = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            text = part.strip()
            if text:
                segments.append({"type": "speech", "text": text})
        else:
            duration = float(part)
            if duration > 0:
                if segments and segments[-1]["type"] == "pause":
                    segments[-1]["duration_sec"] += duration
                else:
                    segments.append({"type": "pause", "duration_sec": duration})

    return segments


# ── Text expansion ───────────────────────────────────────────────────────

def _int_to_words(n: int) -> str:
    """Convert a non-negative integer to English words (supports up to 999,999)."""
    if n == 0:
        return 'zero'
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (('-' + _ONES[ones]) if ones else '')
    if n < 1000:
        hundreds, remainder = divmod(n, 100)
        rest = (' and ' + _int_to_words(remainder)) if remainder else ''
        return _ONES[hundreds] + ' hundred' + rest
    if n < 1_000_000:
        thousands, remainder = divmod(n, 1000)
        rest = (' ' + _int_to_words(remainder)) if remainder else ''
        return _int_to_words(thousands) + ' thousand' + rest
    return str(n)  # fallback for very large numbers — leave as-is


def _replace_number(match: re.Match) -> str:
    """Replace a matched digit sequence with English words."""
    try:
        return _int_to_words(int(match.group(0)))
    except ValueError:
        return match.group(0)


def _replace_hyphenated_numbers(match: re.Match) -> str:
    """Convert hyphenated digit sequences to comma-separated words.

    Handles breathing ratios and similar patterns:
      '4-7-8'  → 'four, seven, eight'
      '10-20'  → 'ten, twenty'
    """
    parts = match.group(0).split('-')
    words = []
    for p in parts:
        try:
            words.append(_int_to_words(int(p)))
        except ValueError:
            return match.group(0)  # not all parts are digits — leave unchanged
    return ', '.join(words)


def expand_for_tts(text: str) -> str:
    """Expand digits and abbreviations to their spoken equivalents.

    Processing order:
    1. Abbreviation substitution.
    2. Hyphenated number patterns (e.g. '4-7-8') → comma-separated words.
    3. Standalone integers up to 999,999 → words.
    """
    for pattern, replacement in _ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Step 2: hyphenated patterns first — consumed before standalone digit pass
    text = re.sub(r'\b\d+(?:-\d+)+\b', _replace_hyphenated_numbers, text)
    # Step 3: standalone integers up to 999,999
    text = re.sub(r'\b\d{1,6}\b', _replace_number, text)
    return text


# ── Phoneme injection for Sanskrit/yoga terms ────────────────────────────
# Maps regex patterns to misaki IPA injection syntax [display](/IPA/).
# Kokoro's G2P engine mispronounces these loanwords without explicit guidance.
_PHONEME_MAP: list[tuple[str, str]] = [
    # ── existing entries ─────────────────────────────────────────────────
    (r'\bchakra\b',      '[chakra](/tʃɑːkɹə/)'),
    (r'\bchakras\b',     '[chakras](/tʃɑːkɹəz/)'),
    (r'\bpranayama\b',   '[pranayama](/pɹɑːnəˈjɑːmə/)'),
    (r'\bsavasana\b',    '[savasana](/ʃɑːˈvɑːsənə/)'),
    (r'\bujjayi\b',      '[ujjayi](/uːˈdʒɑːjiː/)'),
    (r'\bmantra\b',      '[mantra](/ˈmɑːntɹə/)'),
    (r'\bmantras\b',     '[mantras](/ˈmɑːntɹəz/)'),
    (r'\bnamaste\b',     '[namaste](/nɑːməˈsteɪ/)'),
    (r'\bsankalpa\b',    '[sankalpa](/sɑːŋˈkɑːlpə/)'),
    (r'\bnidra\b',       '[nidra](/ˈniːdɹə/)'),
    # ── yoga styles & postures ───────────────────────────────────────────
    (r'\bvipassana\b',   '[vipassana](/vɪˈpɑːsənə/)'),
    (r'\bashtanga\b',    '[ashtanga](/ɑːʃˈtɑːŋɡə/)'),
    (r'\bvinyasa\b',     '[vinyasa](/vɪnˈjɑːsə/)'),
    (r'\bshavasana\b',   '[shavasana](/ʃɑːˈvɑːsənə/)'),
    (r'\basana\b',       '[asana](/ˈɑːsənə/)'),
    (r'\basanas\b',      '[asanas](/ˈɑːsənəz/)'),
    # ── breath & energy ──────────────────────────────────────────────────
    (r'\bprana\b',       '[prana](/ˈpɹɑːnə/)'),
    (r'\bmudra\b',       '[mudra](/ˈmuːdɹə/)'),
    (r'\bmudras\b',      '[mudras](/ˈmuːdɹəz/)'),
    (r'\bbandha\b',      '[bandha](/ˈbɑːndə/)'),
    (r'\bkundalini\b',   '[kundalini](/ˌkʊndəˈliːni/)'),
    # ── philosophy & states ──────────────────────────────────────────────
    (r'\bsamsara\b',     '[samsara](/sɑːmˈsɑːɹə/)'),
    (r'\bnirvana\b',     '[nirvana](/nɪɹˈvɑːnə/)'),
    (r'\bdharma\b',      '[dharma](/ˈdɑːɹmə/)'),
    (r'\bahimsa\b',      '[ahimsa](/ɑːˈhɪmsə/)'),
    (r'\bsatya\b',       '[satya](/ˈsɑːtjə/)'),
    (r'\bsutra\b',       '[sutra](/ˈsuːtɹə/)'),
    (r'\bsutras\b',      '[sutras](/ˈsuːtɹəz/)'),
    (r'\bsamadhi\b',     '[samadhi](/səˈmɑːdi/)'),
    (r'\bprajna\b',      '[prajna](/ˈpɹɑːdʒnə/)'),
    (r'\btapas\b',       '[tapas](/ˈtɑːpəs/)'),
    (r'\bishvara\b',     '[ishvara](/ˈɪʃvəɹə/)'),
    # ── sacred sounds & greetings ────────────────────────────────────────
    (r'\bom\b',          '[om](/oʊm/)'),
    (r'\baum\b',         '[aum](/ɑːuːm/)'),
    (r'\bnamaskara\b',   '[namaskara](/nɑːməˈskɑːɹə/)'),
]


def inject_phonemes(text: str) -> str:
    """Replace known mispronounced Sanskrit/yoga words with misaki IPA injection."""
    for pattern, replacement in _PHONEME_MAP:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── Prosody punctuation constants ────────────────────────────────────────
# Meditation-specific verbs for "as you/we [verb]" comma insertion.
# Restricted to avoid false positives on generic "as you would expect" phrases.
_MEDITATION_VERBS = (
    r'breathe|exhale|inhale|relax|drift|settle|rest|soften|deepen|'
    r'release|let|allow|notice|feel|sink|melt|open|expand|quiet|'
    r'slow|ground|center|arrive|land|return|scan|observe|witness|'
    r'float|dissolve|unwind|surrender|drop|fall|flow|be|stay|sit'
)

# Nouns that form natural breath-group markers for "with each [noun]".
_BREATH_NOUNS = r'breath|exhale|inhale|heartbeat|moment|step|wave|rise|fall'

# Sentence boundary lookbehind — matches positions after ^, ". ", "! ", "? "
_SENT_START = r'(?:(?:^)|(?<=\. )|(?<=\! )|(?<=\? ))'


def _break_long_breath_groups(text: str) -> str:
    """Insert commas to prevent runs of > _BREATH_GROUP_MAX consecutive words
    without a comma or period.

    Tokenises on whitespace. Each token is classified as:
      - punctuation-bearing (contains , . ! ? \\n) → resets word count
      - plain English word (matches \\w[\\w'-]*) → increments count; if count
        has already reached _BREATH_GROUP_MAX and this word is a conjunction,
        a comma is inserted before it and the count resets
      - other (spaces, IPA markers like [word](/IPA/)) → appended unchanged,
        not counted

    The comma insertion happens on the first eligible conjunction *after* the
    12-word limit is crossed. If no conjunction appears, the run is left as-is.
    """
    tokens = re.split(r'(\s+)', text)
    word_count = 0
    result = []

    for token in tokens:
        if re.search(r'[,\.!?\n]', token):
            # Punctuation resets the breath-group counter
            word_count = 0
            result.append(token)
        elif re.fullmatch(r"\w[\w'-]*", token):
            # Plain English word — insert comma before conjunction if over limit
            if word_count >= _BREATH_GROUP_MAX and token.lower() in _BREAK_CONJUNCTIONS:
                result.append(', ')
                word_count = 0
            word_count += 1
            result.append(token)
        else:
            # Whitespace, IPA markers, or other non-word tokens — pass through
            result.append(token)

    return ''.join(result)


def enhance_prosody_punctuation(text: str) -> str:
    """Insert punctuation at natural phrasing boundaries to guide Kokoro's prosody.

    Kokoro is highly sensitive to commas and periods — they directly control
    micro-pause duration, intonation shape, and breath-group boundaries.
    This function targets high-frequency meditation text patterns where
    missing punctuation causes flat or rushed delivery.

    Applied rules (in order):
    1. Add trailing period if segment lacks terminal punctuation.
    2. ``Now [verb]`` at sentence start → ``Now, [verb]``.
    3. Introductory prepositional opener → append comma if missing.
    4. Mid-clause ``as you/we [meditation verb]`` → ``, as you [verb]``.
    5. Mid-clause gerund phrase (letting/allowing/releasing…) → ``, letting …``.
    6. Mid-clause ``with each [breath noun]`` → ``, with each [noun]``.
    7. Long clause (≥ 8 words) before ``and``/``but`` → ``, and``/``, but``.
    8. Breath-group limit: > 12 consecutive words → comma before next conjunction.
    9. Clean up any double-comma or comma-before-period artifacts.
    """
    # 1. Ensure terminal punctuation — Kokoro reads unpunctuated endings flat
    text = re.sub(r'([A-Za-z])\s*$', r'\1.', text)

    # 2. Sentence-initial "Now [verb]" without comma
    #    "Now breathe deeply." → "Now, breathe deeply."
    text = re.sub(
        rf'({_SENT_START}Now) ([a-z])',
        r'\1, \2',
        text,
    )

    # 3. Introductory prepositional phrases at sentence start, lacking a comma
    #    "With each breath allow…" → "With each breath, allow…"
    #    "In this moment feel…"   → "In this moment, feel…"
    text = re.sub(
        rf'({_SENT_START}'
        r'(?:With|In|At|On|Through|For) (?:each|this|every|a|the|your) \w+)'
        r'(?=[^,\.!\?])',
        r'\1,',
        text,
    )

    # 4. Mid-clause "as you/we [meditation verb]" — breath-group reset point
    #    "breathe deeply as you exhale" → "breathe deeply, as you exhale"
    #    Skipped when already preceded by a comma.
    text = re.sub(
        rf'(?<![,])\s+(as (?:you|we) (?:{_MEDITATION_VERBS}))',
        r', \1',
        text,
        flags=re.IGNORECASE,
    )

    # 5. Mid-clause gerund phrases — participial continuation markers
    #    "exhale slowly letting go of tension" → "exhale slowly, letting go of tension"
    text = re.sub(
        r'(?<![,])\s+((?:letting|allowing|releasing|melting|softening|sinking) \w)',
        r', \1',
        text,
        flags=re.IGNORECASE,
    )

    # 6. Mid-clause "with each [breath noun]"
    #    "feel lighter with each breath" → "feel lighter, with each breath"
    text = re.sub(
        rf'(?<![,])\s+(with each (?:{_BREATH_NOUNS}))',
        r', \1',
        text,
        flags=re.IGNORECASE,
    )

    # 7. Long clause (≥ 8 words since last punctuation) before "and"/"but"
    #    "breathe in slowly and with intention release" →
    #    "breathe in slowly, and with intention release"
    def _comma_before_conj(m: re.Match) -> str:
        prefix, conj, rest = m.group(1), m.group(2), m.group(3)
        last_boundary = max(
            prefix.rfind(','), prefix.rfind('.'),
            prefix.rfind('!'), prefix.rfind('?'),
        )
        clause = prefix[last_boundary + 1:].strip()
        if len(clause.split()) >= 8:
            return f'{prefix}, {conj} {rest}'
        return m.group(0)

    text = re.sub(
        r'([A-Za-z][^,\.\n]+?) (and|but|while) ([a-z])',
        _comma_before_conj,
        text,
    )

    # 8. Breath-group limit — force a comma before the next conjunction once
    #    a run of _BREATH_GROUP_MAX consecutive words without punctuation is hit.
    text = _break_long_breath_groups(text)

    # 9. Cleanup artifacts
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r',(\s*[\.!\?])', r'\1', text)
    text = re.sub(r'  +', ' ', text)

    return text.strip()


def preprocess_for_meditation(text: str) -> str:
    """Optimise text for calm, meditation-style Kokoro TTS delivery.

    Pipeline:
    1. Expand digits/abbreviations to spoken forms.
    2. Inject IPA phonemes for Sanskrit/yoga terms.
    3. Insert commas at natural phrasing boundaries (enhance_prosody_punctuation).
    4. Add ellipsis transition before any inline [pause:] markers.
    5. Normalise ellipsis to exactly three dots.
    """
    text = expand_for_tts(text)
    text = inject_phonemes(text)
    text = enhance_prosody_punctuation(text)
    # Soft ellipsis transition into an explicit pause marker (graceful handoff)
    text = re.sub(r'(\w)\s*(\[pause:)', r'\1... \2', text)
    # Normalise any multi-dot sequences to clean three-dot ellipsis
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


# ── Token-aware sentence splitting and chunking ──────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word for English."""
    return int(len(text.split()) * 1.3)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries.

    Handles standard sentence endings (.!?) and ellipsis (...).
    Very short sentences (< MIN_SENTENCE_WORDS words) are merged with the
    next sentence so Kokoro receives enough context for natural prosody.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    raw = [p for p in parts if p.strip()]

    if len(raw) <= 1:
        return raw

    merged: list[str] = []
    carry = ""
    for s in raw:
        if carry:
            s = carry + " " + s
            carry = ""
        if len(s.split()) < MIN_SENTENCE_WORDS and s is not raw[-1]:
            carry = s
        else:
            merged.append(s)
    if carry:
        if merged:
            merged[-1] = merged[-1] + " " + carry
        else:
            merged.append(carry)

    return merged


def merge_sentences_to_chunks(sentences: list[str]) -> list[str]:
    """Merge consecutive sentences into chunks targeting 100–150 tokens.

    Never exceeds HARD_CEILING_TOKENS per chunk. Very short sentences
    (below MIN_CHUNK_TOKENS) are merged with neighbouring text.
    """
    if not sentences:
        return sentences

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        if sentence_tokens > HARD_CEILING_TOKENS:
            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_tokens = 0

            words = sentence.split()
            max_words = int(HARD_CEILING_TOKENS / 1.3)
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue

        if current_tokens + sentence_tokens > MAX_CHUNK_TOKENS and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = [sentence]
            current_tokens = sentence_tokens
        else:
            current_parts.append(sentence)
            current_tokens += sentence_tokens

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def clamp_speed(speed: float) -> float:
    """Clamp speed to safe Kokoro range (0.65–1.0).

    Below 0.65, Kokoro produces distorted, robotic output.
    """
    return max(speed, MIN_SPEED)


def prepare_segments(script: str) -> list[dict]:
    """Full preprocessing pipeline: parse script → preprocess each speech segment.

    Returns structured segments ready for KokoroEngine.synthesize().
    """
    segments = parse_script(script)
    for seg in segments:
        if seg["type"] == "speech":
            seg["text"] = preprocess_for_meditation(seg["text"])
    return segments
