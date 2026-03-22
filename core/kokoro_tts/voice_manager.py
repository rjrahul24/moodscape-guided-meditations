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
    "serene_sky": {
        "description": "Maximum breathiness, deep relaxation (research-validated)",
        "blend": {"af_sky": 0.65, "af_sarah": 0.35},
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


def _slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two tensors.

    Preserves embedding norms on the hypersphere, producing more natural
    intermediate voices than linear interpolation (which shrinks norms
    at midpoint by ~29% for orthogonal vectors).
    """
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()
    v0_norm = v0_flat / torch.norm(v0_flat)
    v1_norm = v1_flat / torch.norm(v1_flat)
    omega = torch.acos(torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0))
    sin_omega = torch.sin(omega)
    if sin_omega.abs() < 1e-6:
        # Vectors nearly parallel — fall back to linear
        result = (1.0 - t) * v0_flat + t * v1_flat
    else:
        result = (torch.sin((1.0 - t) * omega) / sin_omega) * v0_flat + \
                 (torch.sin(t * omega) / sin_omega) * v1_flat
    return result.reshape(v0.shape)


def slerp_blend(voice_weights: dict[str, float]) -> torch.Tensor:
    """Blend voice tensors using spherical linear interpolation (SLERP).

    For 2 voices: direct SLERP with weight as interpolation parameter.
    For 3+ voices: iterative pairwise SLERP (blend first two, then
    SLERP result with third, etc.).

    SLERP preserves the norm of voice embeddings on the style space
    hypersphere, producing smoother blends than linear interpolation.
    """
    items = list(voice_weights.items())
    if len(items) == 1:
        return load_voice_tensor(items[0][0])

    # Normalise weights to sum to 1.0
    total = sum(w for _, w in items)
    items = [(vid, w / total) for vid, w in items]

    # For 2 voices: direct SLERP
    if len(items) == 2:
        v0 = load_voice_tensor(items[0][0])
        v1 = load_voice_tensor(items[1][0])
        t = items[1][1]  # weight of second voice = interpolation parameter
        return _slerp(v0, v1, t)

    # For 3+ voices: iterative pairwise SLERP
    result = load_voice_tensor(items[0][0])
    accumulated_weight = items[0][1]
    for voice_id, weight in items[1:]:
        tensor = load_voice_tensor(voice_id)
        # t is the fraction of the new voice in the running blend
        t = weight / (accumulated_weight + weight)
        result = _slerp(result, tensor, t)
        accumulated_weight += weight

    return result


def add_voice_jitter(voice_tensor: torch.Tensor, amount: float = 0.001) -> torch.Tensor:
    """Add subtle random perturbation to a voice tensor for natural variation.

    Human speakers have micro-variation in timbre across utterances.
    This creates a "liveness" quality when applied per-sentence — each
    sentence gets a slightly different voice without sounding like a
    different speaker.

    Args:
        voice_tensor: Voice embedding tensor of shape (511, 1, 256).
        amount: Perturbation magnitude (~0.1% of typical embedding norm).
                Keep very low (≤0.001) to avoid audible timbre shifts at
                pause boundaries.
    """
    return voice_tensor + torch.randn_like(voice_tensor) * amount


def get_voice(voice_spec: str):
    """Resolve a voice specification to a blended tensor or string ID.

    Accepts:
      - Single voice ID: "af_heart" → returns string
      - Comma-separated blend: "af_heart,af_nicole" → returns blended tensor (equal weight)
      - Preset name: "golden_hour" → returns blended tensor from preset

    Blends use SLERP interpolation to preserve embedding norms.

    Returns:
        str | torch.Tensor
    """
    # Check if it is a preset
    if voice_spec in MEDITATION_PRESETS:
        return slerp_blend(MEDITATION_PRESETS[voice_spec]["blend"])

    # Check if it is a comma-separated blend
    if "," in voice_spec:
        voices = [v.strip() for v in voice_spec.split(",")]
        weight = 1.0 / len(voices)
        return slerp_blend({v: weight for v in voices})

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
