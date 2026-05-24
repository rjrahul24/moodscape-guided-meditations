"""VoiceRegistry for IndexTTS-2 — scans and validates speaker and emotion assets.

IndexTTS-2 uses two types of reference audio:
  1. Speaker references (voice cloning) — stored in reference_audio/vocals/
  2. Emotion references (emotion control) — stored in reference_audio/instrumental/

Unlike F5-TTS, IndexTTS-2 does NOT require verbatim transcripts — it performs
its own speech analysis from the reference audio. So a voice is registered
with just a .wav file (no matching .txt required).

Directory layout (relative to project root):
    reference_audio/vocals/         — 24 kHz, 16-bit PCM .wav files (one per voice)
    reference_audio/instrumental/   — 24 kHz, 16-bit PCM .wav files (one per emotion)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Asset directories, anchored to the project root (three levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_VOCALS_DIR = _PROJECT_ROOT / "reference_audio" / "vocals"
_EMOTIONS_DIR = _PROJECT_ROOT / "reference_audio" / "instrumental"

# Reference-audio hygiene thresholds (Conformer encoder + zero-shot timbre quality).
_MIN_SR_HZ = 16000          # below 16 kHz: insufficient spectral detail for cloning
_MIN_DURATION_SEC = 4.0     # too short: weak speaker embedding
_MAX_DURATION_SEC = 15.0    # too long: Conformer encoder averages out timbre
_PEAK_CEILING_DBFS = -1.0   # above this: likely clipped, distorts cloned voice
_MIN_RMS_DBFS = -45.0       # below this: essentially silent / unusable


def _validate_reference(path: Path, kind: str) -> None:
    """Log warnings for reference assets that violate IndexTTS-2 hygiene rules.

    Non-fatal: bad assets still register (engine downmixes / resamples on load),
    but the user gets actionable feedback about likely quality regressions.
    """
    try:
        import soundfile as sf
        import numpy as np

        info = sf.info(str(path))
        sr, duration, channels = info.samplerate, info.duration, info.channels

        if sr < _MIN_SR_HZ:
            logger.warning(
                "IndexTTS-2 %s '%s': sample rate %d Hz < %d Hz — clone fidelity will suffer.",
                kind, path.name, sr, _MIN_SR_HZ,
            )
        if duration < _MIN_DURATION_SEC:
            logger.warning(
                "IndexTTS-2 %s '%s': %.1fs is shorter than %.1fs minimum — weak embedding.",
                kind, path.name, duration, _MIN_DURATION_SEC,
            )
        elif duration > _MAX_DURATION_SEC:
            logger.warning(
                "IndexTTS-2 %s '%s': %.1fs exceeds %.1fs — Conformer encoder will average out timbre.",
                kind, path.name, duration, _MAX_DURATION_SEC,
            )
        if channels > 1:
            logger.warning(
                "IndexTTS-2 %s '%s': %d channels (stereo) — engine will downmix; mono is preferred.",
                kind, path.name, channels,
            )

        # Peak / RMS check (cheap — read once, downsample if huge)
        arr, _ = sf.read(str(path), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak >= 10 ** (_PEAK_CEILING_DBFS / 20.0):
            logger.warning(
                "IndexTTS-2 %s '%s': peak %.2f dBFS — likely clipped, will distort cloning.",
                kind, path.name, 20.0 * np.log10(max(peak, 1e-9)),
            )
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
        rms_db = 20.0 * np.log10(max(rms, 1e-9))
        if rms_db < _MIN_RMS_DBFS:
            logger.warning(
                "IndexTTS-2 %s '%s': RMS %.1f dBFS is essentially silent — unusable as reference.",
                kind, path.name, rms_db,
            )
    except Exception as e:
        logger.debug("Reference validation skipped for '%s': %s", path.name, e)


def scan_voices() -> dict[str, dict[str, Path]]:
    """Scan reference_audio/vocals/ for speaker reference audio files.

    Returns mapping:
        voice_slug -> {"audio": Path}

    Each .wav file in the vocals directory becomes a selectable voice.
    The slug is derived from the filename without extension:
        calm_meditation.wav -> slug: "calm_meditation"
    """
    registry: dict[str, dict[str, Path]] = {}

    if not _VOCALS_DIR.is_dir():
        logger.warning("IndexTTS-2 vocals directory not found: %s", _VOCALS_DIR)
        return {}

    for wav_path in sorted(_VOCALS_DIR.glob("*.wav")):
        slug = wav_path.stem
        registry[slug] = {
            "audio": wav_path.resolve(),
        }
        _validate_reference(wav_path, "voice")
        logger.debug("Registered IndexTTS-2 voice: '%s'", slug)

    if not registry:
        logger.warning(
            "No IndexTTS-2 voices found. "
            "Add .wav files (24 kHz mono, 5-10s) to '%s'.",
            _VOCALS_DIR,
        )

    return registry


def scan_emotions() -> dict[str, dict[str, Path]]:
    """Scan reference_audio/instrumental/ for emotion reference audio files.

    Returns mapping:
        emotion_slug -> {"audio": Path}

    Each .wav file in the instrumental directory becomes a selectable emotion.
    The slug is derived from the filename without extension:
        calm.wav -> slug: "calm"
    """
    registry: dict[str, dict[str, Path]] = {}

    if not _EMOTIONS_DIR.is_dir():
        logger.warning("IndexTTS-2 emotions directory not found: %s", _EMOTIONS_DIR)
        return {}

    for wav_path in sorted(_EMOTIONS_DIR.glob("*.wav")):
        slug = wav_path.stem
        registry[slug] = {
            "audio": wav_path.resolve(),
        }
        _validate_reference(wav_path, "emotion")
        logger.debug("Registered IndexTTS-2 emotion: '%s'", slug)

    return registry


def get_voice(slug: str) -> dict[str, Path]:
    """Return the asset dict for a specific voice slug.

    Returns:
        {"audio": Path} for the voice reference WAV.

    Raises:
        FileNotFoundError: If the slug is not found in the registry.
    """
    registry = scan_voices()
    if slug not in registry:
        raise FileNotFoundError(
            f"IndexTTS-2 voice '{slug}' not found. "
            f"Available voices: {sorted(registry.keys()) or '(none)'}. "
            f"Add .wav files to '{_VOCALS_DIR}'."
        )
    return registry[slug]


def get_emotion(slug: str) -> dict[str, Path]:
    """Return the asset dict for a specific emotion slug.

    Returns:
        {"audio": Path} for the emotion reference WAV.

    Raises:
        FileNotFoundError: If the slug is not found in the registry.
    """
    registry = scan_emotions()
    if slug not in registry:
        raise FileNotFoundError(
            f"IndexTTS-2 emotion '{slug}' not found. "
            f"Available emotions: {sorted(registry.keys()) or '(none)'}. "
            f"Add .wav files to '{_EMOTIONS_DIR}'."
        )
    return registry[slug]
