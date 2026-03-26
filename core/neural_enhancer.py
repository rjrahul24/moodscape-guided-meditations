"""Neural post-processing for codec artifact removal.

Apollo (ICASSP 2025) is a GAN-based music restoration model designed
specifically for removing compression and codec artifacts.  It uses
band-split processing, band-sequence modeling, and band-reconstruction
to target mid-to-high frequency degradation characteristic of RVQ codecs
like HeartCodec.

Memory: ~7 GB.  Loads AFTER HeartMuLa unloads (sequential in the pipeline).

Requires:
    git clone https://github.com/JusperLee/Apollo
    # Add Apollo/ to sys.path or install as package
"""

import gc
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("moodscape.neural_enhancer")

# Apollo repo expected at project root
_APOLLO_DIR = Path(__file__).parent.parent / "Apollo"
_APOLLO_CHECKPOINT = _APOLLO_DIR / "pretrained_models" / "apollo_model.ckpt"


def _is_apollo_available() -> bool:
    """Check if Apollo is cloned and checkpoint exists."""
    return _APOLLO_DIR.exists() and _APOLLO_CHECKPOINT.exists()


def enhance_with_apollo(
    audio: np.ndarray,
    sample_rate: int = 48000,
    target_sr: int = 44100,
) -> np.ndarray:
    """Remove codec artifacts from HeartMuLa output using Apollo.

    Apollo processes at 44.1 kHz, so we resample in/out if needed.
    Falls back gracefully if Apollo is not installed.

    Args:
        audio: Mono float32 array at ``sample_rate``.
        sample_rate: Input sample rate (typically 48000).
        target_sr: Apollo's native sample rate (44100).

    Returns:
        Enhanced mono float32 array at the original ``sample_rate``.
    """
    if not _is_apollo_available():
        logger.warning(
            "[Apollo] Not available (clone https://github.com/JusperLee/Apollo "
            "and download checkpoint).  Skipping neural enhancement."
        )
        return audio

    if audio.shape[-1] < sample_rate:
        logger.info("[Apollo] Audio too short (<1s), skipping.")
        return audio

    logger.info("[Apollo] Enhancing %d samples (%.1fs)...", len(audio), len(audio) / sample_rate)

    try:
        # Add Apollo to path
        if str(_APOLLO_DIR) not in sys.path:
            sys.path.insert(0, str(_APOLLO_DIR))

        import torch
        from apollo_model import ApolloModel  # Apollo's model class

        # Resample to Apollo's native 44.1 kHz if needed
        if sample_rate != target_sr:
            import torchaudio
            t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            t = torchaudio.functional.resample(t, sample_rate, target_sr)
            audio_44k = t.squeeze().numpy()
        else:
            audio_44k = audio

        # Load Apollo model
        device = torch.device("cpu")  # MPS may not be supported by Apollo
        model = ApolloModel.load_from_checkpoint(str(_APOLLO_CHECKPOINT))
        model = model.to(device).eval()

        # Process in chunks to manage memory (30s chunks)
        chunk_size = target_sr * 30
        enhanced_chunks = []

        with torch.no_grad():
            for start in range(0, len(audio_44k), chunk_size):
                chunk = audio_44k[start:start + chunk_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
                enhanced = model(chunk_tensor)
                enhanced_chunks.append(enhanced.squeeze().cpu().numpy())

        enhanced_audio = np.concatenate(enhanced_chunks).astype(np.float32)

        # Unload Apollo
        del model
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        # Resample back to original sample rate
        if sample_rate != target_sr:
            t = torch.from_numpy(enhanced_audio).unsqueeze(0).unsqueeze(0)
            t = torchaudio.functional.resample(t, target_sr, sample_rate)
            enhanced_audio = t.squeeze().numpy().astype(np.float32)

        # Preserve original peak level
        orig_peak = np.abs(audio).max()
        new_peak = np.abs(enhanced_audio).max()
        if new_peak > 1e-8 and orig_peak > 1e-8:
            enhanced_audio = enhanced_audio * (orig_peak / new_peak)

        logger.info("[Apollo] Enhancement complete.")
        return np.clip(enhanced_audio, -1.0, 1.0).astype(np.float32)

    except Exception as e:
        logger.warning("[Apollo] Enhancement failed (%s), returning original audio.", e)
        return audio
