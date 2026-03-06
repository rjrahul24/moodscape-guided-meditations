"""AI Source Separation via HT Demucs — removes drums and vocals from generated music.

Music generation models (MusicGen, ACE-Step) sometimes produce unwanted percussive
hits or vocal-like artefacts despite "no drums, no vocals" prompting.  Running the
generated audio through Meta's HT Demucs source separation model and discarding the
drums and vocals stems provides a robust safety net that guarantees a purely ambient,
percussion-free meditation background.

Model: htdemucs (4-source: drums, bass, vocals, other)
Device: CPU (Demucs is lightweight and fast on CPU; keeps GPU free)
Output: Mono float32 at the same sample rate as input — the bass + other stems
        summed together, with drums and vocals discarded.
"""

import gc
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Demucs native sample rate
_DEMUCS_SR = 44100


class StemSeparator:
    """Separates audio into stems via HT Demucs and removes drums/vocals."""

    def __init__(self):
        self._model = None

    def load_model(self):
        """Load the HT Demucs pretrained model."""
        from demucs.pretrained import get_model

        logger.info("[StemSeparator] Loading htdemucs...")
        self._model = get_model("htdemucs")
        self._model.eval()
        # Run on CPU — Demucs is fast enough and this keeps GPU free
        self._model.cpu()
        logger.info("[StemSeparator] htdemucs loaded (CPU)")

    def unload_model(self):
        """Release model memory."""
        del self._model
        self._model = None
        gc.collect()
        logger.info("[StemSeparator] htdemucs unloaded")

    def remove_drums_and_vocals(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Remove drums and vocals stems from audio, keeping only tonal content.

        Args:
            audio: Mono float32 numpy array at any sample rate.
            sample_rate: Sample rate of the input audio.

        Returns:
            Mono float32 numpy array at the same sample rate, with drums and
            vocals removed. Only bass + other (ambient pads, synths, etc.) remain.
        """
        if self._model is None:
            self.load_model()

        import torchaudio

        # Convert to torch tensor — Demucs expects (batch, channels, time)
        tensor = torch.from_numpy(audio.astype(np.float32))
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)  # (1, time) — mono

        # Demucs requires stereo at 44100 Hz
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(2, 1)  # mono → stereo

        needs_resample = sample_rate != _DEMUCS_SR
        if needs_resample:
            tensor = torchaudio.functional.resample(
                tensor, sample_rate, _DEMUCS_SR,
                lowpass_filter_width=64, rolloff=0.9475,
            )

        # Add batch dimension: (1, 2, time)
        tensor = tensor.unsqueeze(0)

        # Run separation
        from demucs.apply import apply_model

        logger.info("[StemSeparator] Running source separation...")
        with torch.no_grad():
            sources = apply_model(self._model, tensor, device="cpu")
        # sources shape: (1, num_sources, 2, time)
        # htdemucs sources order: drums, bass, other, vocals

        source_names = self._model.sources
        logger.info("[StemSeparator] Sources: %s", source_names)

        # Sum all stems EXCEPT drums and vocals
        keep_indices = [
            i for i, name in enumerate(source_names)
            if name not in ("drums", "vocals")
        ]
        kept = sources[0, keep_indices].sum(dim=0)  # (2, time) — stereo

        # Stereo → mono
        result = kept.mean(dim=0)  # (time,)

        # Resample back to original sample rate if needed
        if needs_resample:
            result = result.unsqueeze(0)
            result = torchaudio.functional.resample(
                result, _DEMUCS_SR, sample_rate,
                lowpass_filter_width=64, rolloff=0.9475,
            )
            result = result.squeeze(0)

        result = result.numpy().astype(np.float32)

        # Log what was removed
        removed = [name for name in source_names if name in ("drums", "vocals")]
        logger.info(
            "[StemSeparator] Removed stems: %s | Kept: %s",
            removed,
            [source_names[i] for i in keep_indices],
        )

        return result
