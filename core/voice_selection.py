# core/voice_selection.py
"""Choose a narrator voice per genre family from measured reference-clip traits.

Each reference clip is measured once (speaking rate, median pitch, pitch range,
spectral brightness) and cached by (filename, size, mtime). Families map to a
profile -- calm, energy, expressive, steady -- and the voice scoring highest on
that profile (z-scored across the pool) narrates the family. An optional
assets/speakers/voice_map.toml overrides any family.
"""

import json
import logging
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = _ROOT / "var" / "voice_traits.json"
DEFAULT_OVERRIDES = _ROOT / "assets" / "speakers" / "voice_map.toml"

FAMILY_PROFILE: dict[str, str] = {
    "Sleep & Rest": "calm",
    "Stress & Anxiety": "calm",
    "Morning & Energy": "energy",
    "Body & Movement": "energy",
    "Emotional": "expressive",
    "Presence": "expressive",
    "Focus & Work": "steady",
    "Breathwork": "steady",
}
FALLBACK_PROFILE = "steady"


@dataclass(frozen=True)
class VoiceTraits:
    wpm: float
    f0_median: float
    f0_range: float
    centroid: float


def measure_voice(audio_path: Path, transcript: str) -> VoiceTraits:
    import librosa
    import soundfile as sf

    y, sr = sf.read(str(audio_path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    duration = len(y) / sr
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr, frame_length=1024)
    f0v = f0[voiced & ~np.isnan(f0)]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    loud = rms > 0.1 * rms.max() if rms.size and rms.max() > 0 else np.ones_like(centroid, bool)
    n = min(len(centroid), len(loud))
    return VoiceTraits(
        wpm=len(transcript.split()) / duration * 60.0 if duration else 0.0,
        f0_median=float(np.median(f0v)) if f0v.size else 0.0,
        f0_range=float(np.percentile(f0v, 90) - np.percentile(f0v, 10)) if f0v.size else 0.0,
        centroid=float(np.median(centroid[:n][loud[:n]])) if n else 0.0,
    )


def _transcript_text(value) -> str:
    return Path(value).read_text(encoding="utf-8").strip() if isinstance(value, Path) else str(value)


def voice_traits(registry: dict | None = None, cache_path: Path | None = None) -> dict[str, VoiceTraits]:
    if registry is None:
        from core.f5_tts import voice_registry
        registry = voice_registry.scan()
    cache_path = cache_path or DEFAULT_CACHE
    try:
        cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        cache = {}
    out: dict[str, VoiceTraits] = {}
    changed = False
    for slug, phases in sorted(registry.items()):
        phase = phases.get("default") or next(iter(phases.values()))
        audio = Path(phase["audio"])
        try:
            stat = audio.stat()
            key = [audio.name, stat.st_size, stat.st_mtime]
            entry = cache.get(slug)
            if entry and entry.get("key") == key:
                out[slug] = VoiceTraits(**entry["traits"])
                continue
            traits = measure_voice(audio, _transcript_text(phase["transcript"]))
        except Exception as exc:  # unreadable clip: skip, never fail a job
            logger.warning("voice_selection: could not measure %s (%s); skipping", audio, exc)
            continue
        out[slug] = traits
        cache[slug] = {"key": key, "traits": asdict(traits)}
        changed = True
    if changed:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2))
        except OSError as exc:
            logger.warning("voice_selection: could not write cache %s (%s)", cache_path, exc)
    return out


def _z(values: dict[str, float]) -> dict[str, float]:
    arr = np.array(list(values.values()), dtype=float)
    std = arr.std()
    return {k: 0.0 if std == 0 else float((v - arr.mean()) / std) for k, v in values.items()}


def profile_scores(traits: dict[str, VoiceTraits]) -> dict[str, dict[str, float]]:
    zr = _z({k: t.wpm for k, t in traits.items()})
    zc = _z({k: t.centroid for k, t in traits.items()})
    zf = _z({k: t.f0_median for k, t in traits.items()})
    zg = _z({k: t.f0_range for k, t in traits.items()})
    scores = {}
    for k in traits:
        calm = -zr[k] - zc[k] - zf[k]
        scores[k] = {"calm": calm, "energy": -calm, "expressive": zg[k], "steady": -zg[k]}
    return scores


def _overrides(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(tomllib.loads(path.read_text(encoding="utf-8")).get("families", {}))


def voice_for_family(family: str, *, traits: dict[str, VoiceTraits] | None = None,
                     overrides_path: Path | None = None) -> str:
    traits = traits if traits is not None else voice_traits()
    if not traits:
        raise RuntimeError("No measurable reference voices found in assets/speakers.")
    override = _overrides(overrides_path or DEFAULT_OVERRIDES).get(family)
    if override is not None:
        if override not in traits:
            raise ValueError(f"voice_map.toml maps {family!r} to unknown voice {override!r}; known: {sorted(traits)}")
        return override
    profile = FAMILY_PROFILE.get(family, FALLBACK_PROFILE)
    scores = profile_scores(traits)
    return sorted(traits, key=lambda k: (-round(scores[k][profile], 9), k))[0]
