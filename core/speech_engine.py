"""Abstract base class for TTS engines in the MoodScape pipeline."""

from abc import ABC, abstractmethod

import numpy as np

SAMPLE_RATE = 24000


class SpeechEngine(ABC):
    """Interface that all TTS engines must implement.

    Every engine produces mono float32 audio at 24 000 Hz together with a
    boolean voice-activity mask of the same length.  The rest of the pipeline
    (HeartMuLa, Pedalboard FX, mixer) is engine-agnostic — it only depends on
    this contract.
    """

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
