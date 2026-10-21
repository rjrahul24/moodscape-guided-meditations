"""Testable logic behind the Voice Provider toggle and Remix panel.

app.py cannot be imported in tests, so every decision the UI makes lives here
and app.py only wires components to these functions.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("moodscape.provider_ui")

PROVIDERS = ("ElevenLabs", "Local Models")
DEFAULT_PROVIDER = "Local Models"
_LOCAL_LABELS = {"F5-TTS": "f5", "Kokoro": "kokoro", "Chatterbox TTS": "chatterbox"}


def is_cloud(provider: str) -> bool:
    return provider == "ElevenLabs"


def restore_provider(value) -> str:
    """Whitelist a provider restored from the browser (localStorage)."""
    return value if value in PROVIDERS else DEFAULT_PROVIDER


def generate_enabled(provider: str, mode: str) -> bool:
    """Generate is disabled only for a voiced ElevenLabs run with no API key."""
    if not is_cloud(provider) or mode == "Instrumental Only":
        return True
    from core.elevenlabs_tts.client import has_api_key
    return has_api_key()


def resolve_tts_engine(provider: str, local_choice: str) -> str:
    if is_cloud(provider):
        return "elevenlabs"
    if local_choice in _LOCAL_LABELS:
        return _LOCAL_LABELS[local_choice]
    from core.engine_registry import list_all_engines
    return {label: eid for label, eid in list_all_engines()}.get(local_choice, local_choice.lower())


def provider_visibility(provider: str, mode: str) -> dict[str, bool]:
    vocal = mode != "Instrumental Only"
    cloud = is_cloud(provider)
    return {
        "tts_engine_radio": vocal and not cloud,
        "kokoro_settings": False,   # recomputed by the local engine radio handler
        "f5_settings": vocal and not cloud,
        "speed_slider": vocal and not cloud,
        "f5_cfg_slider": not cloud,
        "microprosody_checkbox": not cloud,
        "elevenlabs_settings": vocal and cloud,
    }


def api_key_banner() -> str:
    from core.elevenlabs_tts.client import has_api_key
    if has_api_key():
        return ""
    return "⚠️ ELEVENLABS_API_KEY is not set in .env — add it to generate with ElevenLabs."


_VALIDATION_CACHE: dict[str, bool] | None = None


def validate_account_voices(client_factory=None) -> dict[str, bool] | None:
    """Which registry voices exist in the account. Memoized per process for the
    real client (one round of GET /v1/voices/{id}, not one per toggle)."""
    global _VALIDATION_CACHE
    from core.elevenlabs_tts.client import ElevenLabsClient, has_api_key
    from core.elevenlabs_tts.voices import list_voices
    if client_factory is None:
        if not has_api_key():
            return None
        if _VALIDATION_CACHE is not None:
            return _VALIDATION_CACHE
    try:
        client = (client_factory or ElevenLabsClient)()
        result = {v.id: client.voice_exists(v.id) for v in list_voices()}
    except Exception as exc:  # validation is advisory; never block the UI
        logger.warning("ElevenLabs voice validation failed: %s", exc)
        return None
    if client_factory is None:
        _VALIDATION_CACHE = result
    return result


def voice_choices(validation: dict[str, bool] | None = None) -> list[tuple[str, str]]:
    from core.elevenlabs_tts.voices import list_voices
    out = []
    for v in list_voices():
        label = v.label
        if validation is not None and validation.get(v.id) is False:
            label = f"{label} (not in account)"
        out.append((label, v.id))
    return out


def cost_line(script: str, content_type: str, direction: str | None, voice_id: str,
              stability: float, similarity: float) -> str:
    from core.elevenlabs_tts import chunk_cache
    from core.elevenlabs_tts.engine import resolve_params
    from core.elevenlabs_tts.preprocessor import estimate_billed_chars, prepare_segments
    if not (script or "").strip():
        return "≈ 0 characters"
    try:
        segs = prepare_segments(script, content_type=content_type, direction=direction or None)
    except ValueError as exc:
        return f"⚠️ {exc}"
    p = resolve_params(voice_id=voice_id, stability=stability, similarity=similarity)
    speech = [s["text"] for s in segs if s["type"] == "speech"]
    cached = sum(chunk_cache.is_cached(chunk_cache.chunk_payload(
        t, p.voice_id, p.model_id, p.stability, p.similarity, p.seed)) for t in speech)
    return (f"≈ {estimate_billed_chars(segs):,} characters · {len(speech)} chunks · "
            f"{cached}/{len(speech)} cached")


def remix_ui(run_id: str | None, voice_db: float, music_db: float) -> tuple[str | None, str]:
    from core.mix_session import MixSessionNotFound
    from core.remix import remix
    if not run_id:
        return None, "Generate a meditation first, then remix it."
    try:
        path = remix(run_id, float(voice_db), float(music_db))
    except KeyError as exc:  # session saved by an older/failed run lacks a field
        return None, f"Mix session is incomplete (missing {exc}) — generate again to remix."
    except (MixSessionNotFound, ValueError, OSError) as exc:
        return None, str(exc)
    return path, f"Remixed (voice {voice_db:+g} dB, music {music_db:+g} dB) → {path}"
