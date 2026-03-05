"""Session configuration for reproducible meditation generation."""

import json
import time
from dataclasses import dataclass, field


@dataclass
class SessionConfig:
    """Immutable configuration for a single meditation generation session.

    Centralises every generation parameter so that sessions can be logged
    and reproduced exactly.
    """

    # Model
    model_version: str = "hexgrad/Kokoro-82M"
    backend: str = "auto"  # "auto", "mlx", "pytorch"

    # Voice
    voice: str = "golden_hour"
    speed: float = 0.78
    lang_code: str = "a"

    # Chunking
    max_chunk_tokens: int = 200
    min_chunk_tokens: int = 15
    crossfade_ms: int = 75
    inter_sentence_pause_sec: float = 0.8
    ellipsis_pause_sec: float = 1.2

    # Audio
    sample_rate: int = 24000
    target_lufs: float = -18.0

    # Reproducibility
    seed: int = field(default_factory=lambda: int(time.time()) % (2**31))

    def to_json(self) -> str:
        """Serialise config to JSON for logging / reproducibility."""
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SessionConfig":
        """Reconstruct a SessionConfig from its JSON representation."""
        return cls(**json.loads(json_str))
