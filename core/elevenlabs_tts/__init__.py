"""ElevenLabs v4 cloud TTS engine."""

from core.elevenlabs_tts.client import (
    ElevenLabsAPIError,
    ElevenLabsClient,
    ElevenLabsConfigError,
    has_api_key,
)
from core.elevenlabs_tts.direction import (
    DEFAULT_DIRECTION,
    default_direction,
    sanitize_direction,
)
from core.elevenlabs_tts.preprocessor import estimate_billed_chars, prepare_segments
from core.elevenlabs_tts.voices import (
    ElevenLabsVoice,
    default_voice_id,
    list_voices,
    voice_for_family,
)

__all__ = [
    "DEFAULT_DIRECTION",
    "ElevenLabsAPIError",
    "ElevenLabsClient",
    "ElevenLabsConfigError",
    "ElevenLabsVoice",
    "default_direction",
    "default_voice_id",
    "estimate_billed_chars",
    "has_api_key",
    "list_voices",
    "prepare_segments",
    "sanitize_direction",
    "voice_for_family",
]
