"""Estimate a script's spoken runtime without rendering it.

Pauses are summed exactly from the engine's own parse. Speech is estimated as
word_count / wpm * 60 — the same formula core/f5_tts/engine.py:463 uses when
fixed pacing is enabled. Gap handling is engine-specific because the two
engines insert silence differently: Kokoro adds a room-tone gap after every
sentence, while F5 only gaps between ≤250-char CHUNKS (each usually several
sentences) — see _speech_seconds and _F5_CHUNK_GAP_SEC. Breath/inhale/exhale
cues add their measured sample duration (BREATH_SEC) since both
preprocessors emit a distinct "breath" segment type for these markers rather
than a "pause".

Fades are deliberately NOT added: apply_fades shapes amplitude over audio that
already exists, so they do not extend runtime.
"""

import logging
import re

from core.kokoro_tts.engine import ELLIPSIS_PAUSE_SEC, INTER_SENTENCE_PAUSE_SEC

logger = logging.getLogger(__name__)

# Measured speaking rates. These are the calibration constants:
# log_estimate_accuracy() exists to refine them from real renders rather than
# leaving them a guess.
#
# f5: measured 2026-09-17 from a real render of REALISTIC_SCRIPT (198 words,
# excluding [pause:Xs] markers) in
# tests/integration/test_auto_generate_e2e.py — at the old 97.0 WPM the
# estimate was 205.4s against an actual 226.0s (ratio 1.10, implied WPM
# 85.0). 85.0 reproduces the actual duration almost exactly (ratio 1.001).
# Caveat: F5 clones the pacing of its reference audio, so this number is
# voice-dependent — a markedly faster or slower reference voice will drift
# from it. A 22-word script measured separately gave a wildly different
# implied rate (36.8 WPM) because fixed per-chunk overhead (reference-audio
# padding, leading/trailing silence) dominates a 3-chunk script; that
# measurement was discarded as not representative of steady-state speech
# rate.
#
# kokoro: NOT remeasured — left at the prior estimate. No real-render data
# backs this number yet; do not change it on the strength of the f5
# measurement above.
DEFAULT_WPM: dict[str, float] = {
    "f5": 85.0,
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

# core/f5_tts/engine.py inserts a 0.4s room-tone gap plus a 300ms equal-power
# crossfade between consecutive "speech"-type chunks (the crossfade overlaps
# the gap, so the net silence added is GAP - FADE = 0.1s). This only happens
# at CHUNK boundaries — the ≤250-char splits core/f5_tts/preprocessor.py's
# split_into_chunks() makes per "speech" segment, which usually span several
# sentences. It does NOT happen between sentences within one chunk; those are
# synthesized as continuous prose with no gap at all.
_F5_CHUNK_GAP_SEC = 0.4 - 0.3


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


def _speech_seconds(text: str, wpm: float, *, engine: str) -> float:
    """Words at the target rate, plus this engine's own inter-sentence gaps.

    Kokoro (core/kokoro_tts/engine.py) inserts a room-tone gap after every
    sentence within a segment, so that model applies here. F5
    (core/f5_tts/engine.py) does not: sentences within one ≤250-char chunk
    are synthesized as continuous prose with no inserted gap at all — F5
    only gaps at chunk BOUNDARIES, which estimate_duration_sec accounts for
    separately via _F5_CHUNK_GAP_SEC, between consecutive "speech" segments.
    Applying Kokoro's per-sentence gap to F5 text overestimates duration by
    ~0.7s per extra sentence in a multi-sentence paragraph.
    """
    words = len(text.split())
    if words == 0:
        return 0.0

    speech = words / wpm * 60.0

    if engine != "kokoro":
        return speech

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
    prev_type = None
    for segment in segments:
        if segment["type"] == "pause":
            total += float(segment["duration_sec"])
        elif segment["type"] == "speech":
            if engine == "f5" and prev_type == "speech":
                # Mirrors the engine: a gap is only inserted between two
                # consecutive "speech" chunks, never before the first one or
                # after a pause/breath segment.
                total += _F5_CHUNK_GAP_SEC
            total += _speech_seconds(segment["text"], rate, engine=engine)
        elif segment["type"] == "breath":
            # Unrecognised subtypes degrade to the plain "breath" duration
            # rather than raising — a bad tag shouldn't crash a generation run.
            total += BREATH_SEC.get(segment.get("subtype"), BREATH_SEC["breath"])
        prev_type = segment["type"]

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
