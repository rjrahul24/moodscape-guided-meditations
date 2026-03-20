"""AI Source Separation via HT Demucs — removes drums and vocals from generated music.

Music generation models (HeartMuLa, ACE-Step) sometimes produce unwanted percussive
hits or vocal-like artefacts despite "no drums, no vocals" prompting.  Running the
generated audio through Meta's HT Demucs source separation model and discarding the
drums and vocals stems provides a robust safety net that guarantees a purely ambient,
percussion-free meditation background.

Model: htdemucs (4-source: drums, bass, vocals, other) — single 42M-param model (~168 MB)
Device: CPU (Demucs is lightweight and fast on CPU; keeps GPU free)
Output: Mono float32 at the same sample rate as input — the bass + other stems
        summed together, with drums and vocals discarded.

Memory strategy: htdemucs is used instead of mdx_extra (334M params, 1.3 GB as 4
sub-models) to keep subprocess peak memory under ~800 MB.  apply_model's built-in
split=True with a small segment size handles chunking internally.
"""

import gc
import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)

# Demucs native sample rate
_DEMUCS_SR = 44100

# Segment size for apply_model's built-in chunking (seconds).
# Smaller = less peak memory per chunk.  htdemucs default is 7.8s;
# we use 5s to keep the subprocess well within memory budget.
_SEGMENT_SEC = 5.0


class StemSeparator:
    """Separates audio into stems via HT Demucs and removes drums/vocals."""

    def __init__(self):
        self._model = None

    def load_model(self):
        """Load the Meta HT Demucs pretrained model (42M params, ~168 MB)."""
        from demucs.pretrained import get_model

        model_name = "htdemucs"
        logger.info(f"[StemSeparator] Loading {model_name}...")
        self._model = get_model(model_name)
        self._model.eval()
        self._model.cpu()
        logger.info(f"[StemSeparator] {model_name} loaded (CPU, ~168 MB)")

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
        """Remove drums and vocals stems via isolated subprocess.

        Isolates the Demucs model into a separate process so its memory is
        fully reclaimed by the OS on exit.  Uses .npy files for IPC.
        """
        import subprocess
        import tempfile
        import os
        import time

        gc.collect()

        # Force release ALL cached MLX metal buffers so the subprocess has memory
        try:
            import mlx.core as mx
            mx.set_cache_limit(0)
            mx.clear_cache()
            active_mb = mx.get_active_memory() / 1e6
            cache_mb = mx.get_cache_memory() / 1e6
            logger.info(
                "[StemSeparator] MLX metal memory released — active: %.1f MB, cache: %.1f MB",
                active_mb, cache_mb,
            )
        except (ImportError, Exception):
            pass

        # Also release any MPS cached memory
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, Exception):
            pass

        # Give macOS memory compactor time to reclaim freed pages
        time.sleep(3)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as in_f:
            input_path = in_f.name
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as out_f:
            output_path = out_f.name

        try:
            logger.info(
                "[StemSeparator] Launching isolated separation subprocess (htdemucs) — "
                "audio shape=%s, sr=%d, duration=%.1fs",
                audio.shape, sample_rate, len(audio) / sample_rate,
            )
            np.save(input_path, audio)

            cmd = [sys.executable, "scripts/separate_worker.py", input_path, str(sample_rate), output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"[StemSeparator] Subprocess failed with code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(f"AI source separation failed in subprocess (code {result.returncode})")

            processed_audio = np.load(output_path)
            logger.info("[StemSeparator] Isolated separation complete")
            return processed_audio

        finally:
            for p in (input_path, output_path):
                if os.path.exists(p):
                    os.unlink(p)
                if os.path.exists(p + ".npy"):
                    os.unlink(p + ".npy")

    def _remove_drums_and_vocals_internal(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Run source separation in-process (called by separate_worker.py).

        Loads htdemucs once, calls apply_model with split=True and a small
        segment size so Demucs handles chunking internally with proper
        overlap blending.  No custom lazy-chunk logic needed.
        """
        import torch
        import torchaudio
        from demucs.apply import apply_model

        torch.set_num_threads(1)

        # Ensure writable 1D float32 (mmap'd arrays from np.load are read-only,
        # and some engines may output 2D arrays like (samples, 1))
        if not audio.flags.writeable or audio.dtype != np.float32:
            audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=0) if audio.shape[0] <= 8 else audio.mean(axis=1)

        self.load_model()

        try:
            # Convert to torch tensor — Demucs expects (batch, channels, time)
            tensor = torch.from_numpy(audio)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)  # (1, time) — mono

            # Demucs requires stereo
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

            logger.info(
                "[StemSeparator-Worker] Running htdemucs separation (%.1fs, segment=%.1fs, split=True, shifts=0)...",
                tensor.shape[-1] / _DEMUCS_SR, _SEGMENT_SEC,
            )

            with torch.no_grad():
                sources = apply_model(
                    self._model, tensor, device="cpu",
                    split=True, segment=_SEGMENT_SEC,
                    shifts=0, overlap=0.25,
                )
            # sources shape: (1, num_sources, 2, time)

            source_names = self._model.sources
            keep_indices = [
                i for i, name in enumerate(source_names)
                if name not in ("drums", "vocals")
            ]
            kept = sources[0, keep_indices].sum(dim=0)  # (2, time) — stereo
            del sources

            result = kept.mean(dim=0)  # (time,) — mono
            del kept

            # Resample back to original sample rate if needed
            if needs_resample:
                result = result.unsqueeze(0)
                result = torchaudio.functional.resample(
                    result, _DEMUCS_SR, sample_rate,
                    lowpass_filter_width=64, rolloff=0.9475,
                )
                result = result.squeeze(0)

            return result.numpy().astype(np.float32)

        finally:
            self.unload_model()
            gc.collect()
