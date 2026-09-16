"""Estimate a script's spoken runtime without rendering it.

Pauses are summed exactly from the engine's own parse. Speech is estimated as
word_count / wpm * 60 — the same formula core/f5_tts/engine.py:463 uses when
fixed pacing is enabled. Inter-sentence room-tone gaps are added because the
engines insert them. Breath/inhale/exhale cues add their measured sample
duration (BREATH_SEC) since both preprocessors emit a distinct "breath"
segment type for these markers rather than a "pause".

Fades are deliberately NOT added: apply_fades shapes amplitude over audio that
already exists, so they do not extend runtime.
"""

import logging
import re

from core.kokoro_tts.engine import ELLIPSIS_PAUSE_SEC, INTER_SENTENCE_PAUSE_SEC

logger = logging.getLogger(__name__)

# Measured speaking rates. F5 at speed 0.88 runs ~95-100 WPM per
# docs/prompting_guides/vocal_meditation_f5_instructions.md. These are the
# calibration constants: log_estimate_accuracy() exists to refine them from
# real renders rather than leaving them a guess.
DEFAULT_WPM: dict[str, float] = {
    "f5": 97.0,
    "kokoro": 105.0,
}

# Measured from assets/breath_sounds/ (breath.wav 1.200s, inhale.wav 1.500s,
# exhale.wav 1.800s). Calibrate alongside DEFAULT_WPM if those samples change.
BREATH_SEC: dict[str, float] = {
    "breath": 1.2,
    "inhale": 1.5,
    "exhale": 1.8,
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _load_prepare_segments(engine: str):
    """Return the engine's prepare_segments, or raise for an unknown engine."""
    if engine == "f5":
        from core.f5_tts.preprocessor import prepare_segments
        return prepare_segments
    if engine == "kokoro":
        from core.kokoro_tts.preprocessor import prepare_segments
        return prepare_segments
    raise ValueError(
        f"Unknown engine {engine!r}. Expected 'f5' or 'kokoro'."
    )


def _speech_seconds(text: str, wpm: float) -> float:
    """Words at the target rate, plus the gaps the engine puts between sentences."""
    words = len(text.split())
    if words == 0:
        return 0.0

    speech = words / wpm * 60.0

    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
    gaps = 0.0
    # A gap follows every sentence except the last one in this segment.
    for sentence in sentences[:-1]:
        gaps += (
            ELLIPSIS_PAUSE_SEC
            if sentence.rstrip().endswith("...")
            else INTER_SENTENCE_PAUSE_SEC
        )

    return speech + gaps


def estimate_duration_sec(
    script: str,
    *,
    engine: str = "f5",
    content_type: str = "meditation",
    wpm: float | None = None,
) -> float:
    """Estimate total spoken runtime in seconds.

    Args:
        script: Raw script text with [pause:Xs] markers.
        engine: "f5" or "kokoro" — selects the preprocessor and default WPM.
        content_type: "meditation" or "sleep_story". Changes paragraph-break
            pause length via the content profile.
        wpm: Override the engine's default speaking rate.

    Returns:
        Estimated duration in seconds. 0.0 for an empty script.
    """
    if not script.strip():
        return 0.0

    prepare_segments = _load_prepare_segments(engine)
    rate = wpm if wpm is not None else DEFAULT_WPM[engine]

    segments = prepare_segments(script, content_type=content_type)

    total = 0.0
    for segment in segments:
        if segment["type"] == "pause":
            total += float(segment["duration_sec"])
        elif segment["type"] == "speech":
            total += _speech_seconds(segment["text"], rate)
        elif segment["type"] == "breath":
            # Unrecognised subtypes degrade to the plain "breath" duration
            # rather than raising — a bad tag shouldn't crash a generation run.
            total += BREATH_SEC.get(segment.get("subtype"), BREATH_SEC["breath"])

    return total


def log_estimate_accuracy(
    estimated_sec: float, actual_sec: float, engine: str
) -> str:
    """Log estimate-versus-actual so DEFAULT_WPM can be calibrated from data.

    Returns the log line so callers can also write it into run metadata.
    """
    error = actual_sec - estimated_sec
    ratio = (actual_sec / estimated_sec) if estimated_sec else float("nan")
    line = (
        f"duration[{engine}] estimated={estimated_sec:.1f}s "
        f"actual={actual_sec:.1f}s error={error:+.1f}s ratio={ratio:.3f}"
    )
    logger.info(line)
    return line
