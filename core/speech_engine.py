"""Abstract base class for TTS engines in the MoodScape pipeline."""

from abc import ABC, abstractmethod

import numpy as np

SAMPLE_RATE = 24000

# Breath markers ([breath]/[inhale]/[exhale]) render as SILENCE of these
# lengths in every engine. No model produces natural breath or sigh sounds,
# so no engine may ever emit one (user rule, 2026-10-12).
BREATH_SILENCE_SEC: dict[str, float] = {"breath": 1.2, "inhale": 1.5, "exhale": 1.8}


def breath_silence_sec(subtype: str) -> float:
    """Silent-beat length for a breath marker; unknown subtypes use 'breath'."""
    return BREATH_SILENCE_SEC.get(subtype, BREATH_SILENCE_SEC["breath"])


class SpeechEngine(ABC):
    """Interface that all TTS engines must implement.

    Every engine produces mono float32 audio at `sample_rate` Hz (24 000 unless the engine overrides it) together with a
    boolean voice-activity mask of the same length.  The rest of the pipeline
    (Pedalboard FX, mixer) is engine-agnostic — it only depends on
    this contract.
    """

    #: Native output rate. Local engines produce 24 kHz; an engine may override
    #: (ElevenLabs returns 48 kHz). The pipeline honours this attribute.
    sample_rate: int = SAMPLE_RATE

    @abstractmethod
    def load_model(self) -> None:
        """Load / initialise the TTS model or API client."""

    @abstractmethod
    def unload_model(self) -> None:
        """Release model resources and free memory."""

    @abstractmethod
    def synthesize(
        self,
        segments: list[dict],
        voice: str,
        speed: float,
        progress_cb=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate speech audio from parsed script segments.

        Args:
            segments: List from script_parser.parse_script()
                      (dicts with type "speech"/"pause").
            voice:    Voice identifier (engine-specific).
            speed:    Speaking speed (0.5–1.0).
            progress_cb: Called with (current_index, total_segments).

        Returns:
            voice_audio:    float32 mono numpy array at 24 000 Hz.
            voice_activity: bool array (same length), True where voice is active.
        """

    @abstractmethod
    def get_available_voices(self) -> list[dict]:
        """Return available voices as a list of dicts.

        Each dict has keys: "id", "name", "description".
        """
