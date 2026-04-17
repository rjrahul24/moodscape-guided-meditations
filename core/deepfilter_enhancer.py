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
import tempfile
from pathlib import Path

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


def enhance_voice_deepfilter(
    audio: np.ndarray,
    sr: int = 48000,
) -> np.ndarray:
    """Denoise voice audio using DeepFilterNet MLX.

    DeepFilterNet processes at 48 kHz natively, making it ideal for our
    pipeline which upsamples TTS output to 48 kHz before FX processing.

    If DeepFilterNet is unavailable (missing dependency, non-Apple-Silicon),
    falls back to noisereduce spectral gating with the research-spec params.

    Args:
        audio: Mono float32 array at ``sr``.
        sr:    Sample rate (should be 48000 for best results).

    Returns:
        Enhanced mono float32 array at the same sample rate.
    """
    if audio.shape[-1] < sr:  # skip very short clips (<1s)
        logger.debug("[DeepFilterNet] Audio too short (<1s), skipping.")
        return audio

    if not _is_available():
        logger.info(
            "[DeepFilterNet] Not available (install mlx-audio). "
            "Falling back to noisereduce spectral gating."
        )
        return _fallback_spectral_denoise(audio, sr)

    model = _load_model()
    if model is None:
        logger.info("[DeepFilterNet] Model failed to load. Falling back to spectral gating.")
        return _fallback_spectral_denoise(audio, sr)

    logger.info(
        "[DeepFilterNet] Enhancing %d samples (%.1fs) at %d Hz...",
        len(audio), len(audio) / sr, sr,
    )

    try:
        import soundfile as sf

        # DeepFilterNet's enhance() expects a file path, so write to temp file
        tmp_in = tempfile.NamedTemporaryFile(
            delete=False, suffix="_dfnet_in.wav", dir=tempfile.gettempdir()
        )
        tmp_in.close()

        sf.write(tmp_in.name, audio, sr, subtype="FLOAT")

        # Run enhancement
        enhanced_audio = model.enhance(tmp_in.name)

        # enhanced_audio is an mlx array or numpy array
        if hasattr(enhanced_audio, "tolist"):
            # MLX array → convert to numpy
            enhanced_audio = np.array(enhanced_audio, dtype=np.float32)
        else:
            enhanced_audio = np.asarray(enhanced_audio, dtype=np.float32)

        # Squeeze to 1D if needed
        if enhanced_audio.ndim > 1:
            enhanced_audio = enhanced_audio.squeeze()

        # Preserve original peak level (DeepFilterNet may alter gain)
        orig_peak = np.abs(audio).max()
        new_peak = np.abs(enhanced_audio).max()
        if new_peak > 1e-8 and orig_peak > 1e-8:
            enhanced_audio = enhanced_audio * (orig_peak / new_peak)

        # Clean up temp file
        try:
            Path(tmp_in.name).unlink(missing_ok=True)
        except Exception:
            pass

        logger.info("[DeepFilterNet] Enhancement complete. Peak preserved at %.2f dBFS.",
                    20 * np.log10(max(np.abs(enhanced_audio).max(), 1e-8)))
        return np.clip(enhanced_audio, -1.0, 1.0).astype(np.float32)

    except Exception as e:
        logger.warning("[DeepFilterNet] Enhancement failed (%s). Falling back to spectral gating.", e)
        return _fallback_spectral_denoise(audio, sr)


def _fallback_spectral_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Fallback to noisereduce spectral gating (research-spec params)."""
    try:
        import noisereduce as nr
        return nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.7,
            n_fft=512,
            freq_mask_smooth_hz=300,
        ).astype(np.float32)
    except Exception as e:
        logger.warning("[DeepFilterNet-fallback] Spectral gating also failed: %s", e)
        return audio
