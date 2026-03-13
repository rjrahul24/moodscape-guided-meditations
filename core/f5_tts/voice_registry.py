"""VoiceRegistry — scans and validates F5-TTS reference asset pairs.

Directory layout (relative to this module's location at core/f5_tts/):

    assets/reference_audio/      — 24 kHz, 16-bit PCM .wav files (one per slug)
    assets/reference_transcript/ — verbatim .txt transcripts (one per slug)

A voice slug is derived from the filename without extension, e.g.:
    assets/reference_audio/calm_brittney.wav
    assets/reference_transcript/calm_brittney.txt
    → slug: "calm_brittney"

A voice is only registered if both the .wav and the matching .txt file exist
and the transcript is non-empty.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Asset directories, anchored to this module's location (core/f5_tts/)
_ASSETS_DIR = Path(__file__).parent / "assets"
_AUDIO_DIR = _ASSETS_DIR / "reference_audio"
_TRANSCRIPT_DIR = _ASSETS_DIR / "reference_transcript"


def scan() -> dict[str, dict[str, Path]]:
    """Scan asset directories and return all complete voice pairs.

    A voice entry is included only when both files exist and the transcript
    is non-empty:
        assets/reference_audio/{slug}.wav
        assets/reference_transcript/{slug}.txt

    Returns:
        dict mapping slug → {"audio": Path (absolute), "transcript": Path (absolute)}.
        Returns an empty dict if either directory is missing or no pairs exist.
    """
    if not _AUDIO_DIR.is_dir():
        logger.warning(
            "F5-TTS reference_audio directory not found: %s — "
            "create it and add .wav files to register voices.",
            _AUDIO_DIR,
        )
        return {}

    registry: dict[str, dict[str, Path]] = {}

    for wav_path in sorted(_AUDIO_DIR.glob("*.wav")):
        slug = wav_path.stem
        txt_path = _TRANSCRIPT_DIR / f"{slug}.txt"

        if not _TRANSCRIPT_DIR.is_dir():
            logger.warning(
                "F5-TTS reference_transcript directory not found: %s", _TRANSCRIPT_DIR
            )
            break

        if not txt_path.is_file():
            logger.warning(
                "Voice '%s' skipped — transcript missing: %s", slug, txt_path
            )
            continue

        if txt_path.stat().st_size == 0:
            logger.warning(
                "Voice '%s' skipped — transcript is empty: %s", slug, txt_path
            )
            continue

        registry[slug] = {
            "audio": wav_path.resolve(),
            "transcript": txt_path.resolve(),
        }
        logger.debug("Registered F5-TTS voice: '%s'", slug)

    if not registry:
        logger.warning(
            "No complete F5-TTS voice pairs found. "
            "Add .wav files to '%s' and matching .txt files to '%s'.",
            _AUDIO_DIR,
            _TRANSCRIPT_DIR,
        )

    return registry


def get_voice(slug: str) -> dict[str, Path]:
    """Return validated asset paths for a specific voice slug.

    Args:
        slug: Voice identifier — filename stem without extension (e.g. "calm_brittney").

    Returns:
        dict with keys "audio" (Path) and "transcript" (Path), both absolute.

    Raises:
        FileNotFoundError: If the .wav, .txt, or both are missing or the
            transcript is empty.
    """
    audio_path = _AUDIO_DIR / f"{slug}.wav"
    txt_path = _TRANSCRIPT_DIR / f"{slug}.txt"

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"F5-TTS reference audio not found for voice '{slug}'.\n"
            f"Expected: {audio_path}\n"
            f"Place a 24 kHz mono WAV file there to register this voice."
        )

    if not txt_path.is_file():
        raise FileNotFoundError(
            f"F5-TTS transcript not found for voice '{slug}'.\n"
            f"Expected: {txt_path}\n"
            f"Place a verbatim .txt transcript there to complete the asset pair."
        )

    if txt_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"F5-TTS transcript for voice '{slug}' is empty.\n"
            f"File: {txt_path}\n"
            f"Add the verbatim spoken text to this file."
        )

    return {
        "audio": audio_path.resolve(),
        "transcript": txt_path.resolve(),
    }
