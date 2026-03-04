"""Backward-compatibility shim — prefer importing from core.kokoro_engine."""

from core.kokoro_engine import KokoroEngine as TTSEngine  # noqa: F401
from core.kokoro_engine import SAMPLE_RATE, VOICES  # noqa: F401
