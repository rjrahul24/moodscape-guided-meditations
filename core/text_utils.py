"""Shared text expansion utilities for TTS preprocessing.

Provides digit-to-words conversion, abbreviation expansion, and hyphenated
number handling used by both Kokoro and F5-TTS preprocessors.
"""

import re

# ── Abbreviation table ────────────────────────────────────────────────────
ABBREV_MAP = {
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

# ── Number-to-words tables ────────────────────────────────────────────────
_ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
         'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
         'sixteen', 'seventeen', 'eighteen', 'nineteen']
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
         'eighty', 'ninety']


def int_to_words(n: int) -> str:
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
        rest = (' and ' + int_to_words(remainder)) if remainder else ''
        return _ONES[hundreds] + ' hundred' + rest
    if n < 1_000_000:
        thousands, remainder = divmod(n, 1000)
        rest = (' ' + int_to_words(remainder)) if remainder else ''
        return int_to_words(thousands) + ' thousand' + rest
    return str(n)  # fallback for very large numbers — leave as-is


def _replace_number(match: re.Match) -> str:
    """Replace a matched digit sequence with English words."""
    try:
        return int_to_words(int(match.group(0)))
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
            words.append(int_to_words(int(p)))
        except ValueError:
            return match.group(0)  # not all parts are digits — leave unchanged
    return ', '.join(words)


def expand_text(text: str) -> str:
    """Expand digits and abbreviations to their spoken equivalents.

    Processing order:
    1. Abbreviation substitution.
    2. Hyphenated number patterns (e.g. '4-7-8') → comma-separated words.
    3. Standalone integers up to 999,999 → words.
    """
    for pattern, replacement in ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Step 2: hyphenated patterns first — consumed before standalone digit pass
    text = re.sub(r'\b\d+(?:-\d+)+\b', _replace_hyphenated_numbers, text)
    # Step 3: standalone integers up to 999,999
    text = re.sub(r'\b\d{1,6}\b', _replace_number, text)
    return text
