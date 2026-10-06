"""Qwen3-TTS engine (experimental, sandbox only) — Apache-2.0 alternative to F5."""

from core.qwen3_tts import presets
from core.qwen3_tts.preprocessor import prepare_segments

__all__ = ["presets", "prepare_segments"]
