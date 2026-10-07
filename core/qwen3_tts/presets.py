"""Model ids, meditation delivery presets and env-driven settings for Qwen3-TTS."""

import os

MODES = ("clone", "custom", "design")

# Resolved in Task 1 Step 2 from the mlx-community listing; prefer 1.7B bf16.
MODEL_REPOS: dict[str, str] = {
    "clone": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "custom": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    "design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
}

DEFAULT_SPEAKER = "serena"

INSTRUCT_PRESETS: dict[str, str] = {
    "meditation": (
        "Speak slowly and softly in a warm, intimate, unhurried voice, with a "
        "gentle falling cadence at the end of each sentence, as a calm meditation guide."
    ),
    "sleep_story": (
        "Speak very slowly and quietly, soft and low, drowsy and soothing, like "
        "reading a bedtime story to someone who is falling asleep."
    ),
}

DESIGN_PRESETS: dict[str, str] = {
    "meditation": (
        "A warm adult female narrator in her thirties with a low, soft, fully voiced "
        "tone. Calm, unhurried and intimate, like a meditation teacher speaking close "
        "to the microphone."
    ),
    "sleep_story": (
        "A gentle adult female narrator with a low, hushed but fully voiced tone. "
        "Drowsy, soothing and very slow, like a bedtime storyteller."
    ),
}


def mode_from_env() -> str:
    mode = os.environ.get("MOODSCAPE_QWEN3_MODE", "clone").strip().lower()
    if mode not in MODES:
        raise ValueError(f"MOODSCAPE_QWEN3_MODE={mode!r} is not one of {MODES}.")
    return mode


def temperature_from_env() -> float:
    return float(os.environ.get("MOODSCAPE_QWEN3_TEMPERATURE", "0.7"))


def chunk_chars_from_env() -> int:
    return int(os.environ.get("MOODSCAPE_QWEN3_CHUNK_CHARS", "200"))


def stretch_from_env() -> float:
    return float(os.environ.get("MOODSCAPE_QWEN3_STRETCH", "1.0"))


def speaker_from_env() -> str:
    return (os.environ.get("MOODSCAPE_QWEN3_SPEAKER") or "").strip() or DEFAULT_SPEAKER


def instruct_for(mode: str, content_type: str) -> str | None:
    """Delivery instruction for custom mode, voice description for design mode."""
    if mode == "clone":
        return None
    override = os.environ.get("MOODSCAPE_QWEN3_INSTRUCT", "").strip()
    if override:
        return override
    key = content_type if content_type in INSTRUCT_PRESETS else "meditation"
    return (DESIGN_PRESETS if mode == "design" else INSTRUCT_PRESETS)[key]
