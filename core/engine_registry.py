"""Engine Registry — manages built-in, promoted, and experimental TTS engines.

Enables dynamic registration, sandbox experimentation, and one-click promotion
of fine-tuned TTS models into the active application flows.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from core.speech_engine import SpeechEngine

logger = logging.getLogger("moodscape.engine_registry")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VAR_DIR = _PROJECT_ROOT / "var"
_PROMOTED_MODELS_FILE = _VAR_DIR / "promoted_models.json"


# ── Built-in Engine Descriptors ──────────────────────────────────────────────

BUILTIN_ENGINES = {
    "kokoro": {
        "id": "kokoro",
        "name": "Kokoro",
        "description": "Kokoro-82M lightweight neural TTS with 54-blend voice manager.",
        "base_engine": "kokoro",
        "default_voice": "balanced_calm",
    },
    "f5": {
        "id": "f5",
        "name": "F5-TTS",
        "description": "Zero-shot voice cloning diffusion transformer (Flow Matching).",
        "base_engine": "f5",
        "default_voice": "calm_brittney",
    },
    "chatterbox": {
        "id": "chatterbox",
        "name": "Chatterbox TTS",
        "description": "Resemble AI Chatterbox 500M TTS with zero-shot cloning and emotion tuning.",
        "base_engine": "chatterbox",
        "default_voice": "Brittney",
        "presets": {
            "speed": 0.90,
            "reverb_amount": 0.05,
            "duck_amount_db": -16.0,
            "df_wet": 0.85,
            "exaggeration": 0.28,
            "cfg_weight": 0.35,
            "temperature": 0.55,
        },
    },
}


def _ensure_var_dir() -> Path:
    _VAR_DIR.mkdir(parents=True, exist_ok=True)
    return _VAR_DIR


def get_promoted_models_path() -> Path:
    """Return the absolute path to the promoted models JSON file."""
    return _PROMOTED_MODELS_FILE


def load_promoted_models() -> dict[str, dict[str, Any]]:
    """Load all promoted models from disk. Returns empty dict if absent/corrupted."""
    if not _PROMOTED_MODELS_FILE.is_file():
        return {}
    try:
        with open(_PROMOTED_MODELS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception as e:
        logger.warning("Could not read promoted models from %s: %s", _PROMOTED_MODELS_FILE, e)
        return {}


def save_promoted_models(models: dict[str, dict[str, Any]]) -> bool:
    """Save promoted models to disk safely using atomic write."""
    _ensure_var_dir()
    tmp_file = _PROMOTED_MODELS_FILE.with_suffix(".tmp")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(models, f, indent=2)
        tmp_file.replace(_PROMOTED_MODELS_FILE)
        return True
    except Exception as e:
        logger.error("Failed to save promoted models: %s", e)
        if tmp_file.exists():
            tmp_file.unlink()
        return False


def promote_model(
    model_id: str,
    display_name: str,
    base_engine: str,
    config: dict[str, Any] | None = None,
    presets: dict[str, Any] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Promote an experimental or fine-tuned model into the active app registry.

    Args:
        model_id: Unique slug for the model (e.g. 'f5_zen_sage' or 'kokoro_deep_warmth').
        display_name: Human-friendly name displayed in UI dropdowns.
        base_engine: 'f5', 'kokoro', or 'custom'.
        config: Engine parameters (e.g. voice_slug, checkpoint_path, ref_audio, ref_text).
        presets: Fine-tuned optimal settings discovered in the sandbox (speed, WPM, CFG, reverb, ducking).
        description: Notes on the model quality or target tone.

    Returns:
        The saved model record dict.
    """
    clean_id = model_id.strip().lower().replace(" ", "_")
    if not clean_id:
        raise ValueError("Model ID cannot be empty.")

    models = load_promoted_models()
    record = {
        "id": clean_id,
        "name": display_name.strip() or clean_id.title(),
        "base_engine": base_engine,
        "description": description or f"Promoted from Sandbox on {datetime.now().strftime('%Y-%m-%d')}",
        "config": config or {},
        "presets": presets or {},
        "promoted_at": datetime.now().isoformat(),
    }
    models[clean_id] = record
    save_promoted_models(models)
    logger.info("Promoted model '%s' (%s) to active registry", clean_id, record["name"])
    return record


def demote_model(model_id: str) -> bool:
    """Remove a previously promoted model from the active app registry."""
    clean_id = model_id.strip().lower()
    models = load_promoted_models()
    if clean_id in models:
        del models[clean_id]
        save_promoted_models(models)
        logger.info("Demoted model '%s' from active registry", clean_id)
        return True
    return False


def list_all_engines() -> list[tuple[str, str]]:
    """Return all available engines as (label, id) pairs for dropdowns.

    Built-in engines come first, followed by any promoted models.
    """
    choices: list[tuple[str, str]] = [
        ("F5-TTS", "f5"),
        ("Kokoro", "kokoro"),
        ("Chatterbox TTS", "chatterbox"),
    ]
    promoted = load_promoted_models()
    for model_id, info in sorted(promoted.items(), key=lambda kv: kv[1].get("name", kv[0])):
        label = f"{info.get('name', model_id)} [Promoted]"
        choices.append((label, model_id))
    return choices


def get_engine_info(engine_id: str) -> dict[str, Any] | None:
    """Get metadata and presets for an engine (built-in or promoted)."""
    norm_id = engine_id.strip().lower()
    if norm_id in BUILTIN_ENGINES:
        return BUILTIN_ENGINES[norm_id]
    promoted = load_promoted_models()
    return promoted.get(norm_id)


def get_model_presets(engine_id: str) -> dict[str, Any]:
    """Retrieve fine-tuned presets for a promoted model, or default empty dict."""
    info = get_engine_info(engine_id)
    if info and "presets" in info:
        return info["presets"]
    return {}


# ── Dynamic Engine Instantiation ─────────────────────────────────────────────

def get_engine(
    engine_id: str = "kokoro",
    voice_slug: str | None = None,
    **kwargs: Any,
) -> SpeechEngine:
    """Factory that instantiates and returns the appropriate SpeechEngine.

    Handles built-in engines ('kokoro', 'f5', 'chatterbox') as well as promoted models.
    """
    norm_id = engine_id.strip().lower()

    if norm_id == "kokoro":
        from core.kokoro_tts.engine import KokoroEngine
        return KokoroEngine()

    if norm_id == "f5":
        from core.f5_tts.engine import F5Engine
        return F5Engine(voice_slug=voice_slug)

    if norm_id == "chatterbox":
        from core.chatterbox_tts.engine import ChatterboxEngine
        return ChatterboxEngine(voice_slug=voice_slug)

    # Check promoted models
    promoted = load_promoted_models()
    if norm_id in promoted:
        record = promoted[norm_id]
        base = record.get("base_engine", "f5")
        config = record.get("config", {})

        if base == "kokoro":
            from core.kokoro_tts.engine import KokoroEngine
            return KokoroEngine()

        if base == "f5":
            from core.f5_tts.engine import F5Engine
            # Use configured voice_slug or reference voice if provided
            resolved_slug = voice_slug or config.get("voice_slug")
            return F5Engine(voice_slug=resolved_slug)

        if base == "chatterbox":
            from core.chatterbox_tts.engine import ChatterboxEngine
            resolved_slug = voice_slug or config.get("voice_slug")
            return ChatterboxEngine(voice_slug=resolved_slug)

        if base == "custom":
            return SandboxCustomEngine(
                name=record.get("name", norm_id),
                config=config,
            )

    # Fallback default
    logger.warning("Engine ID '%s' not recognized, falling back to Kokoro", engine_id)
    from core.kokoro_tts.engine import KokoroEngine
    return KokoroEngine()


# ── Sandbox Custom Engine Wrapper ───────────────────────────────────────────

class SandboxCustomEngine(SpeechEngine):
    """Configurable wrapper for experimenting with external or custom TTS models.

    Supports custom reference audio clips, HuggingFace pipeline adapters,
    or experimental voice synthesis passes while conforming strictly to the
    SpeechEngine ABC (mono float32 @ 24kHz + voice activity mask).
    """

    def __init__(self, name: str = "custom", config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self._inner_engine: SpeechEngine | None = None
        self._loaded = False

    def load_model(self) -> None:
        """Initialise the underlying model according to config."""
        adapter_type = self.config.get("adapter_type", "f5")
        if adapter_type == "f5":
            from core.f5_tts.engine import F5Engine
            voice_slug = self.config.get("voice_slug")
            self._inner_engine = F5Engine(voice_slug=voice_slug)
            self._inner_engine.load_model()
        elif adapter_type == "chatterbox":
            from core.chatterbox_tts.engine import ChatterboxEngine
            voice_slug = self.config.get("voice_slug")
            self._inner_engine = ChatterboxEngine(voice_slug=voice_slug)
            self._inner_engine.load_model()
        else:
            from core.kokoro_tts.engine import KokoroEngine
            self._inner_engine = KokoroEngine()
            self._inner_engine.load_model()
        self._loaded = True

    def unload_model(self) -> None:
        """Free model resources."""
        if self._inner_engine is not None:
            self._inner_engine.unload_model()
            self._inner_engine = None
        self._loaded = False

    def synthesize(
        self,
        segments: list[dict],
        voice: str = "default",
        speed: float = 0.90,
        progress_cb=None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Synthesize via inner engine."""
        if not self._loaded or self._inner_engine is None:
            self.load_model()
        return self._inner_engine.synthesize(
            segments,
            voice=voice,
            speed=speed,
            progress_cb=progress_cb,
            **kwargs,
        )

    def get_available_voices(self) -> list[dict]:
        """Return voices from inner engine or custom config."""
        if self._inner_engine is not None:
            return self._inner_engine.get_available_voices()
        return [{"id": "default", "name": self.name, "description": "Custom sandbox voice"}]
