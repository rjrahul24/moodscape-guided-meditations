"""Kokoro TTS — self-contained pipeline for Kokoro-82M meditation narration.

Public API:
    KokoroEngine        — TTS engine (load_model / synthesize / unload_model)
    KokoroMasteringEngine — Two-phase mastering (restore_vocals / master_vocals)
    prepare_segments    — Script parsing + Kokoro-specific text preprocessing
    build_voice_chain   — Kokoro-tailored Pedalboard voice FX chain
    build_master_chain  — Kokoro-tailored Phase B mastering EQ chain
    apply_fx            — Apply a Pedalboard chain to audio
    VOICES              — Available Kokoro voice IDs
    MEDITATION_PRESETS  — Curated voice blend presets
    SAMPLE_RATE         — Native output sample rate (24000 Hz)
"""

from core.kokoro_tts.engine import KokoroEngine, VOICES  # noqa: F401
from core.kokoro_tts.preprocessor import prepare_segments  # noqa: F401
from core.kokoro_tts.postprocessor import (  # noqa: F401
    KokoroMasteringEngine,
    apply_fx,
    build_master_chain,
    build_voice_chain,
)
from core.kokoro_tts.voice_manager import MEDITATION_PRESETS  # noqa: F401
from core.speech_engine import SAMPLE_RATE  # noqa: F401
