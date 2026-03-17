"""Kokoro TTS — self-contained pipeline for Kokoro-82M meditation narration.

Public API:
    KokoroEngine        — TTS engine (load_model / synthesize / unload_model)
    prepare_segments    — Script parsing + Kokoro-specific text preprocessing
    build_voice_chain   — Unified Kokoro voice FX chain (single-pass processing)
    apply_fx            — Apply a Pedalboard chain to audio
    VOICES              — Available Kokoro voice IDs
    MEDITATION_PRESETS  — Curated voice blend presets
    SAMPLE_RATE         — Native output sample rate (24000 Hz)
"""

from core.kokoro_tts.engine import KokoroEngine, VOICES  # noqa: F401
from core.kokoro_tts.preprocessor import prepare_segments  # noqa: F401
from core.kokoro_tts.postprocessor import (  # noqa: F401
    apply_fx,
    build_voice_chain,
)
from core.kokoro_tts.voice_manager import MEDITATION_PRESETS  # noqa: F401
from core.speech_engine import SAMPLE_RATE  # noqa: F401
