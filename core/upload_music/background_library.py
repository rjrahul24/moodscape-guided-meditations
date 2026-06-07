"""Background instrumental library — scans assets/backgrounds/ for tracks.

Replaces the old manual upload flow: instead of uploading an instrumental,
the user picks one of the curated tracks in ``assets/backgrounds/``.  This
module discovers those files and builds ``(label, path)`` pairs for the Gradio
dropdown, where ``label`` is a short human-readable name plus the track length
(e.g. ``"Healing Forest — 23:12"``).

The selected dropdown *value* is the absolute file path, which feeds the
existing :class:`~core.upload_music.engine.UploadMusicEngine` unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("moodscape.upload")

# Anchored to the project root (three levels up from this file).
_PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKGROUNDS_DIR = _PROJECT_ROOT / "assets" / "backgrounds"

# Formats the upload engine (pedalboard / libsndfile) can decode.
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


def _short_name(stem: str) -> str:
    """Derive a short, readable track name from a filename stem.

    Strips a leading author/source prefix and a trailing numeric ID, then
    title-cases the remainder.  Falls back to the cleaned full stem if that
    heuristic would leave nothing.

    ``light_music-healing-forest-187590`` -> ``Healing Forest``
    """
    tokens = [t for t in stem.split("-") if t]
    core = tokens[1:] if len(tokens) > 1 else tokens  # drop author prefix
    if core and core[-1].isdigit():
        core = core[:-1]                              # drop trailing ID
    if not core:                                      # heuristic emptied it
        core = tokens or [stem]
    return " ".join(core).replace("_", " ").strip().title()


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``MM:SS`` (e.g. 1152.0 -> ``19:12``)."""
    total = int(round(seconds))
    return f"{total // 60:d}:{total % 60:02d}"


def scan_backgrounds() -> list[tuple[str, str]]:
    """Scan ``assets/backgrounds/`` for instrumental tracks.

    Returns:
        ``[(label, abs_path), ...]`` sorted by label, where ``label`` is
        ``"<Short Name> — MM:SS"``.  Returns ``[]`` if the directory is missing
        or empty.  Per-file errors (undecodable / corrupt) are logged and the
        file is skipped, so one bad asset never breaks the whole list.
    """
    if not BACKGROUNDS_DIR.is_dir():
        logger.warning("Backgrounds dir not found: %s", BACKGROUNDS_DIR)
        return []

    import soundfile as sf

    choices: list[tuple[str, str]] = []
    for path in sorted(BACKGROUNDS_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _AUDIO_EXTS:
            continue
        try:
            duration = sf.info(str(path)).duration
            label = f"{_short_name(path.stem)} — {_format_duration(duration)}"
        except Exception as e:  # corrupt / unreadable — skip, don't crash scan
            logger.warning("Skipping unreadable background '%s': %s", path.name, e)
            continue
        choices.append((label, str(path)))

    choices.sort(key=lambda c: c[0].lower())
    return choices
