"""Voice management for Kokoro TTS — loading, blending, and session consistency.

Handles voice tensor loading from HuggingFace, weighted blending of multiple
voice embeddings, curated meditation presets, and British voice detection.
"""

import torch
from huggingface_hub import hf_hub_download

SAMPLE_RATE = 24000

# Curated premium meditation voice blends
MEDITATION_PRESETS = {
    "balanced_calm": {
        "description": "Warm, natural, very human (best all-purpose)",
        "blend": {"af_heart": 0.5, "af_nicole": 0.5},
    },
    "deep_rest": {
        "description": "Intimate, breathy, very soft (sleep/deep relaxation)",
        "blend": {"af_heart": 0.4, "af_nicole": 0.4, "af_nova": 0.2},
    },
    "soft_whisper": {
        "description": "ASMR-style relaxation, airy and smooth",
        "blend": {"af_nicole": 0.6, "af_nova": 0.4},
    },
    "golden_hour": {
        "description": "Slightly more airy warmth",
        "blend": {"af_heart": 0.6, "af_sky": 0.4},
    },
    "earth_root": {
        "description": "Male/female grounding blend",
        "blend": {"af_heart": 0.7, "am_adam": 0.3},
    },
}

# Voices that require the British English pipeline (lang_code='b')
BRITISH_VOICES = {"bf_emma", "bf_lily", "bm_george", "bm_daniel", "bm_fable", "bm_lewis"}


def load_voice_tensor(voice_id: str) -> torch.Tensor:
    """Load a single voice embedding from HuggingFace."""
    path = hf_hub_download(
        repo_id="hexgrad/Kokoro-82M",
        filename=f"voices/{voice_id}.pt",
    )
    return torch.load(path, weights_only=True)


def blend_voices(voice_weights: dict[str, float]) -> torch.Tensor:
    """Blend multiple voice tensors by weighted sum.

    Args:
        voice_weights: e.g. {"af_heart": 0.6, "af_sky": 0.4} — weights
                       should sum to ~1.0.
    """
    result = None
    for voice_id, weight in voice_weights.items():
        tensor = load_voice_tensor(voice_id)
        if result is None:
            result = tensor * weight
        else:
            result = result + tensor * weight
    return result


def get_voice(voice_spec: str):
    """Resolve a voice specification to either a string ID or a blended tensor.

    Accepts:
      - Single voice ID: "af_heart" → returns string
      - Comma-separated blend: "af_heart,af_nicole" → returns blended tensor (equal weight)
      - Preset name: "golden_hour" → returns blended tensor from preset

    Returns:
        str | torch.Tensor
    """
    # Check if it is a preset
    if voice_spec in MEDITATION_PRESETS:
        return blend_voices(MEDITATION_PRESETS[voice_spec]["blend"])

    # Check if it is a comma-separated blend
    if "," in voice_spec:
        voices = [v.strip() for v in voice_spec.split(",")]
        weight = 1.0 / len(voices)
        return blend_voices({v: weight for v in voices})

    # Single voice ID — return as string (Kokoro handles it)
    return voice_spec


def is_british_voice(voice_spec: str) -> bool:
    """Check whether any voice in the spec requires the British pipeline.

    Works for single IDs, comma-separated blends, and presets.
    """
    if voice_spec in MEDITATION_PRESETS:
        voice_ids = MEDITATION_PRESETS[voice_spec]["blend"].keys()
    elif "," in voice_spec:
        voice_ids = [v.strip() for v in voice_spec.split(",")]
    else:
        voice_ids = [voice_spec]

    return any(v in BRITISH_VOICES for v in voice_ids)
