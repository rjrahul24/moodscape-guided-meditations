"""Neural speech denoising via DeepFilterNet (MLX port).

DeepFilterNet is a 2.1M-parameter speech enhancement model designed for
real-time noise suppression. The MLX port runs natively on Apple Silicon
with minimal memory overhead (~50 MB).

This module provides a drop-in replacement for noisereduce spectral gating
with dramatically better noise floor reduction. DeepFilterNet uses
learned spectral masks rather than statistical thresholds, preserving
soft consonants and breath sounds that spectral gating would damage.

Usage in the pipeline:
    voice_audio = enhance_voice_deepfilter(voice_audio, sr=48000)
    # Then apply Pedalboard FX chain

Requires:
    pip install mlx-audio
    # Model auto-downloads from HuggingFace: mlx-community/DeepFilterNet-mlx

Falls back gracefully to noisereduce if mlx-audio is not installed or
if running on non-Apple-Silicon hardware.
"""

import gc
import logging

import numpy as np

logger = logging.getLogger("moodscape.deepfilter")

# Module-level model cache (lazy-loaded on first call)
_model = None
_model_loaded = False


def _is_available() -> bool:
    """Check if DeepFilterNet MLX can run on this system."""
    try:
        import mlx.core  # noqa: F401
        from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model():
    """Lazy-load the DeepFilterNet MLX model (cached for session)."""
    global _model, _model_loaded
    if _model_loaded:
        return _model

    try:
        from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

        logger.info("[DeepFilterNet] Loading model from mlx-community/DeepFilterNet-mlx...")
        _model = DeepFilterNetModel.from_pretrained("mlx-community/DeepFilterNet-mlx")
        _model_loaded = True
        logger.info("[DeepFilterNet] Model loaded successfully (2.1M params).")
        return _model
    except Exception as e:
        logger.warning("[DeepFilterNet] Failed to load model: %s", e)
        _model_loaded = True  # don't retry
        _model = None
        return None


def unload_model():
    """Release the cached model to free memory."""
    global _model, _model_loaded
    if _model is not None:
        del _model
        _model = None
        _model_loaded = False
        gc.collect()
        logger.info("[DeepFilterNet] Model unloaded.")


def _blend_wet_dry(dry: np.ndarray, wet: np.ndarray, wet_amount: float) -> np.ndarray:
    """Linearly blend a processed (wet) signal back over the original (dry).

    ``wet_amount`` in [0, 1]: 0.0 returns the dry signal untouched, 1.0 returns
    the fully processed signal. Lengths are aligned to the shorter of the two
    (DeepFilterNet's STFT framing can shift length by a few samples).
    """
    n = min(len(dry), len(wet))
    blended = (1.0 - wet_amount) * dry[:n] + wet_amount * wet[:n]
    return np.clip(blended, -1.0, 1.0).astype(np.float32)


def enhance_voice_deepfilter(
    audio: np.ndarray,
    sr: int = 48000,
    wet: float = 1.0,
) -> np.ndarray:
    """Denoise voice audio using DeepFilterNet MLX.

    DeepFilterNet processes at 48 kHz natively, making it ideal for our
    pipeline which upsamples TTS output to 48 kHz before FX processing.

    If DeepFilterNet is unavailable (missing dependency, non-Apple-Silicon),
    falls back to noisereduce spectral gating with the research-spec params.

    Args:
        audio: Mono float32 array at ``sr``.
        sr:    Sample rate (should be 48000 for best results).
        wet:   Dry/wet blend in [0, 1]. The MLX wrapper exposes no denoising
               strength knob, so we attenuate by mixing the dry signal back in.
               Use a low value (~0.10) for already-clean engines like IndexTTS-2
               (BigVGANv2) — full-strength denoising strips natural breaths and
               produces the "AI voice" signature. 1.0 = fully processed.

    Returns:
        Enhanced mono float32 array at the same sample rate.
    """
    if wet <= 0.0:  # bypass entirely
        return audio

    if audio.shape[-1] < sr:  # skip very short clips (<1s)
        logger.debug("[DeepFilterNet] Audio too short (<1s), skipping.")
        return audio

    if not _is_available():
        logger.info(
            "[DeepFilterNet] Not available (install mlx-audio). "
            "Falling back to noisereduce spectral gating."
        )
        return _fallback_spectral_denoise(audio, sr, wet=wet)

    model = _load_model()
    if model is None:
        logger.info("[DeepFilterNet] Model failed to load. Falling back to spectral gating.")
        return _fallback_spectral_denoise(audio, sr, wet=wet)

    logger.info(
        "[DeepFilterNet] Enhancing %d samples (%.1fs) at %d Hz (wet=%.2f)...",
        len(audio), len(audio) / sr, sr, wet,
    )

    try:
        # enhance_array() processes a 1-D float32 numpy array at 48 kHz
        enhanced_audio = model.enhance_array(audio.astype(np.float32))
        enhanced_audio = np.asarray(enhanced_audio, dtype=np.float32)

        if enhanced_audio.ndim > 1:
            enhanced_audio = enhanced_audio.squeeze()

        # Preserve original peak level (DeepFilterNet may alter gain)
        orig_peak = np.abs(audio).max()
        new_peak = np.abs(enhanced_audio).max()
        if new_peak > 1e-8 and orig_peak > 1e-8:
            enhanced_audio = enhanced_audio * (orig_peak / new_peak)

        logger.info("[DeepFilterNet] Enhancement complete. Peak preserved at %.2f dBFS.",
                    20 * np.log10(max(np.abs(enhanced_audio).max(), 1e-8)))
        return _blend_wet_dry(audio, enhanced_audio, wet)

    except Exception as e:
        logger.warning("[DeepFilterNet] Enhancement failed (%s). Falling back to spectral gating.", e)
        return _fallback_spectral_denoise(audio, sr, wet=wet)


def _fallback_spectral_denoise(audio: np.ndarray, sr: int, wet: float = 1.0) -> np.ndarray:
    """Fallback to noisereduce spectral gating (research-spec params)."""
    try:
        import noisereduce as nr
        enhanced = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.7,
            n_fft=512,
            freq_mask_smooth_hz=300,
        ).astype(np.float32)
        return _blend_wet_dry(audio, enhanced, wet)
    except Exception as e:
        logger.warning("[DeepFilterNet-fallback] Spectral gating also failed: %s", e)
        return audio
