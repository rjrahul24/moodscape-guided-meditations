"""Prompt engineering utilities for the Google Lyria RealTime API.

Lyria uses weighted text prompts rather than a single description string.
This module handles:
  - Parsing the user's music prompt (supports 'Label: weight, ...' syntax)
  - Merging with meditation-optimised base tags
  - Returning a list of WeightedPrompt objects ready for the Lyria session

Lyria-safe prompt notes:
  - Avoid artist names (trigger safety filters)
  - Use genre / texture / instrument descriptors
  - Weights can be any non-zero float; 1.0 is the neutral reference
"""

from __future__ import annotations

# Base tags that anchor the output in calm, meditative ambient territory.
# Applied at a modest weight (0.6) so the user's prompt takes precedence.
MEDITATION_BASE_TAGS: list[str] = [
    "Ambient",
    "Ethereal Ambience",
    "Minimalist Drone",
    "Deep Soundscape",
    "Static Texture",
    "Low-Frequency Wash",
]


def parse_weighted_prompt_string(s: str) -> list[tuple[str, float]]:
    """Parse a weighted prompt string into (text, weight) pairs.

    Supports the syntax ``'Label: weight, Label2: weight2, ...'``.
    If no colon separator is found, the whole string is treated as a single
    prompt with weight 1.0.

    Examples::

        parse_weighted_prompt_string("Hang Drum: 1.5, Piano: 0.8")
        # → [("Hang Drum", 1.5), ("Piano", 0.8)]

        parse_weighted_prompt_string("Soft ambient drone")
        # → [("Soft ambient drone", 1.0)]
    """
    pairs: list[tuple[str, float]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            text, weight_str = part.rsplit(":", 1)
            try:
                weight = float(weight_str.strip())
            except ValueError:
                weight = 1.0
            pairs.append((text.strip(), weight))
        else:
            pairs.append((part, 1.0))
    return pairs


def build_lyria_prompts(user_prompt: str):
    """Build a list of WeightedPrompt objects for a Lyria session.

    Merges meditation base tags with the user's prompt.  If the user wrote
    weighted syntax (e.g. ``'Hang Drum: 1.5, Piano: 0.8'``), that is parsed
    and preserved.  Otherwise the user's text is treated as a single prompt
    at weight 1.0.

    Base meditation tags that the user has already mentioned are skipped to
    avoid diluting the attention budget with duplicates.

    Args:
        user_prompt: Raw music style prompt from the Gradio UI.

    Returns:
        List of ``google.genai.types.WeightedPrompt`` objects.
    """
    from google.genai import types  # lazy import; only needed at generation time

    user_pairs = parse_weighted_prompt_string(user_prompt)
    user_text_lower = user_prompt.lower()

    # Prepend base tags that aren't already covered by the user's input
    base_pairs: list[tuple[str, float]] = [
        (tag, 0.6)
        for tag in MEDITATION_BASE_TAGS
        if tag.lower() not in user_text_lower
    ]

    all_pairs = base_pairs + user_pairs
    return [
        types.WeightedPrompt(text=text, weight=weight)
        for text, weight in all_pairs
    ]
