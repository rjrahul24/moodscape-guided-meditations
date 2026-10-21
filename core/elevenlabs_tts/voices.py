"""ElevenLabs voice registry loaded from assets/elevenlabs/voices.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VOICES_PATH = Path(__file__).resolve().parents[2] / "assets" / "elevenlabs" / "voices.toml"


@dataclass(frozen=True)
class ElevenLabsVoice:
    id: str
    name: str
    notes: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} — {self.notes}" if self.notes else self.name


@dataclass(frozen=True)
class VoiceRegistry:
    voices: tuple[ElevenLabsVoice, ...]
    default_voice: str
    family_map: dict[str, str] = field(default_factory=dict)


def load_registry(path: Path | None = None) -> VoiceRegistry:
    """Parse and validate the registry. Raises ValueError on any bad reference."""
    data = tomllib.loads(Path(path or VOICES_PATH).read_text(encoding="utf-8"))
    voices = tuple(
        ElevenLabsVoice(id=v["id"], name=v["name"], notes=v.get("notes", ""))
        for v in data.get("voice", [])
    )
    if not voices:
        raise ValueError("voices.toml lists no [[voice]] entries.")
    known = sorted(v.id for v in voices)
    default = data.get("default_voice", "")
    if default not in known:
        raise ValueError(f"default_voice {default!r} is not a listed voice; known: {known}")
    family_map = dict(data.get("family_map", {}))
    for family, voice_id in family_map.items():
        if voice_id not in known:
            raise ValueError(
                f"family_map maps {family!r} to unknown voice {voice_id!r}; known: {known}"
            )
    return VoiceRegistry(voices=voices, default_voice=default, family_map=family_map)


def list_voices(path: Path | None = None) -> list[ElevenLabsVoice]:
    return list(load_registry(path).voices)


def default_voice_id(path: Path | None = None) -> str:
    return load_registry(path).default_voice


def voice_for_family(family: str, path: Path | None = None) -> str:
    """Mapped voice for a genre family, else the default voice."""
    registry = load_registry(path)
    return registry.family_map.get(family, registry.default_voice)
