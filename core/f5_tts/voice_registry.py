"""VoiceRegistry — scans and validates F5-TTS reference asset pairs.

Directory layout (relative to project root):

    assets/speakers/reference_audio/  — 24 kHz, 16-bit PCM .wav files (shared
                                        pool; also used by IndexTTS-2)
    assets/speakers/reference_text/   — verbatim .txt transcripts (F5-only)

A voice slug is derived from the filename without extension, e.g.:
    assets/speakers/reference_audio/calm_brittney.wav
    assets/speakers/reference_text/calm_brittney.txt
    → slug: "calm_brittney"

A voice is only registered with F5 if both the .wav AND the matching .txt file
exist (and the transcript is non-empty). WAVs without transcripts are skipped
silently — IndexTTS-2 will still pick them up from its own scan.

Multi-phase support can be added via assets/speakers/voices.toml:
    [calm_brittney]
    default = { ref_audio = "guided.wav", ref_text = "..." }
    closing = { ref_audio = "closing.wav", ref_text = "..." }
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

# Asset directories, anchored to the project root (three levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SPEAKERS_DIR = _PROJECT_ROOT / "assets" / "speakers"
_AUDIO_DIR = _SPEAKERS_DIR / "reference_audio"
_TRANSCRIPT_DIR = _SPEAKERS_DIR / "reference_text"
_VOICES_TOML = _SPEAKERS_DIR / "voices.toml"


def scan() -> dict[str, dict[str, dict[str, Path | str]]]:
    """Scan asset directories and return all complete voice pairs and TOML-defined phases.

    Returns hierarchical mapping:
        narrator_slug -> phase_name -> {"audio": Path, "transcript": Path | str}

    Legacy .wav/.txt pairs are mapped to the "default" phase.
    """
    registry: dict[str, dict[str, dict[str, Path | str]]] = {}

    if not _AUDIO_DIR.is_dir():
        logger.warning("F5-TTS reference_audio directory not found: %s", _AUDIO_DIR)
        return {}

    for wav_path in sorted(_AUDIO_DIR.glob("*.wav")):
        slug = wav_path.stem
        txt_path = _TRANSCRIPT_DIR / f"{slug}.txt"

        if not _TRANSCRIPT_DIR.is_dir():
            logger.warning(
                "F5-TTS reference_text directory not found: %s", _TRANSCRIPT_DIR
            )
            break

        if not txt_path.is_file():
            # Expected: IndexTTS-only voices share assets/speakers/ but have no transcripts.
            logger.debug(
                "F5 voice '%s' skipped — no transcript at %s (IndexTTS-only voice?)",
                slug, txt_path,
            )
            continue

        if txt_path.stat().st_size == 0:
            logger.warning(
                "Voice '%s' skipped — transcript is empty: %s", slug, txt_path
            )
            continue

        registry[slug] = {
            "default": {
                "audio": wav_path.resolve(),
                "transcript": txt_path.resolve(),
            }
        }
        logger.debug("Registered legacy F5-TTS voice: '%s'", slug)

    # 2. Layer TOML definitions
    if _VOICES_TOML.is_file():
        try:
            with open(_VOICES_TOML, "rb") as f:
                data = tomllib.load(f)
            
            for narrator, phases in data.items():
                if not isinstance(phases, dict):
                    continue
                
                narrator_registry = registry.get(narrator, {})
                for phase_name, meta in phases.items():
                    if not isinstance(meta, dict):
                        continue
                    
                    audio_val = meta.get("ref_audio")
                    text_val = meta.get("ref_text")
                    
                    if audio_val and text_val:
                        # Resolve audio path relative to _AUDIO_DIR or as absolute
                        audio_path = Path(audio_val)
                        if not audio_path.is_absolute():
                            audio_path = _AUDIO_DIR / audio_path
                        
                        if audio_path.is_file():
                            narrator_registry[phase_name] = {
                                "audio": audio_path.resolve(),
                                "transcript": text_val
                            }
                        else:
                            logger.warning("Voice '%s' phase '%s' skipped — audio not found: %s", narrator, phase_name, audio_path)
                
                if narrator_registry:
                    registry[narrator] = narrator_registry
                    
        except Exception as e:
            logger.error("Failed to parse voices.toml: %s", e)

    if not registry:
        logger.warning(
            "No complete F5-TTS voice pairs found. "
            "Add .wav files to '%s' and matching .txt transcripts to '%s'.",
            _AUDIO_DIR,
            _TRANSCRIPT_DIR,
        )

    return registry


def get_voice(slug: str) -> dict[str, dict[str, Path | str]]:
    """Return dict of phases for a specific voice slug.

    Returns:
        dict mapping phase_name -> {"audio": Path, "transcript": Path | str}
    """
    registry = scan()
    if slug not in registry:
        raise FileNotFoundError(f"Voice '{slug}' not found in registry.")
    return registry[slug]
