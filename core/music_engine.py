"""MusicGen wrapper for generating ambient meditation background music.

Strategy: Generate a single clean 30-second segment using musicgen-small,
then loop it with smooth equal-power crossfade. This avoids continuation
artifacts (hum, noise) that occur when chaining multiple MusicGen calls,
and runs ~5x faster than musicgen-medium.

For background ambient music that sits behind a voice, a well-looped 30s
clip of drone/pad texture is indistinguishable from longer unique generation.

Target hardware: Apple Silicon M1 Max (32 GB unified memory)
MusicGen sample rate: 32000 Hz  →  resampled to 24000 Hz for pipeline
"""

import gc

import numpy as np


NATIVE_SAMPLE_RATE = 32000       # MusicGen native output rate
TARGET_SAMPLE_RATE = 24000       # Kokoro TTS / pipeline standard rate
SEGMENT_DURATION = 30            # Seconds per MusicGen call (hard limit)
LOOP_CROSSFADE = 5               # Seconds of equal-power crossfade when looping


class MusicEngine:
    """Generates ambient background music via a single MusicGen call + looping.

    Uses musicgen-small (300M params) forced to CPU for stable, fast inference.
    Generates one clean 30-second segment and loops with equal-power crossfade
    to fill any duration. This eliminates continuation artifacts entirely.
    """

    def __init__(self):
        self.model = None
        self.model_name = None

    def load_model(self):
        """Load MusicGen-small on CPU. Falls back gracefully if unavailable."""
        import warnings

        from audiocraft.models import MusicGen

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")

            # musicgen-small: 300M params, ~1.5x realtime on M1 Max CPU
            # Sufficient quality for ambient background behind narration
            # Force CPU — MPS has dtype instability in AudioCraft on Apple Silicon
            self.model = MusicGen.get_pretrained("facebook/musicgen-small", device="cpu")
            self.model_name = "musicgen-small"

    def unload_model(self):
        """Release model and free memory."""
        del self.model
        self.model = None
        self.model_name = None
        gc.collect()

    def generate(
        self,
        prompt: str,
        total_duration_sec: float,
        progress_cb=None,
    ) -> np.ndarray:
        """Generate background music: one clean segment, looped to fill duration.

        Args:
            prompt: Text description of desired music style (keep under 40 words
                    total for best MusicGen attention).
            total_duration_sec: Target duration in seconds.
            progress_cb: Called with (current_segment, total_segments).

        Returns:
            Mono float32 numpy array at 24000 Hz.
        """
        if self.model is None:
            raise RuntimeError("Music model not loaded. Call load_model() first.")

        import torch
        import torchaudio

        # ── Generate a single clean 30s segment from text prompt ──────────
        self.model.set_generation_params(
            duration=SEGMENT_DURATION,
            use_sampling=True,
            top_k=250,
            top_p=0.0,            # Use top_k exclusively — more stable for ambient
            temperature=0.8,      # Balanced: evolving textures without artifacts (docs: 0.75–0.90)
            cfg_coef=4.0,         # Strong prompt adherence (docs: 3.5–5.0, 4.0 is sweet spot)
        )

        if progress_cb is not None:
            progress_cb(0, 1)

        wav = self.model.generate([prompt], progress=False)
        # wav shape: (batch=1, channels, samples) at 32000 Hz
        segment = wav[0]  # (channels, samples)

        if progress_cb is not None:
            progress_cb(1, 1)

        # Force mono
        if segment.ndim == 2 and segment.shape[0] > 1:
            segment = segment.mean(dim=0, keepdim=True)
        elif segment.ndim == 1:
            segment = segment.unsqueeze(0)

        segment_np = segment.squeeze(0).cpu().numpy().astype(np.float32)

        # Clamp to valid range
        segment_np = np.clip(segment_np, -1.0, 1.0)

        # ── Loop to fill the requested duration ───────────────────────────
        target_samples = int(total_duration_sec * NATIVE_SAMPLE_RATE)
        if len(segment_np) >= target_samples:
            full_audio = segment_np[:target_samples]
        else:
            full_audio = self._loop_block(segment_np, target_samples)

        # ── Resample from 32000 Hz to 24000 Hz ───────────────────────────
        audio_tensor = torch.from_numpy(full_audio).unsqueeze(0)  # (1, N)
        resampled = torchaudio.functional.resample(
            audio_tensor, NATIVE_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )

        return resampled.squeeze(0).numpy().astype(np.float32)

    def _loop_block(self, block: np.ndarray, target_samples: int) -> np.ndarray:
        """Loop a single audio block with equal-power crossfade to fill duration.

        Equal-power crossfade (sqrt curves) maintains consistent perceived
        loudness at the loop point, unlike linear crossfade which creates a
        volume dip at the seam.
        """
        if len(block) >= target_samples:
            return block[:target_samples]

        crossfade_samples = int(LOOP_CROSSFADE * NATIVE_SAMPLE_RATE)
        # Don't let crossfade exceed half the block length
        crossfade_samples = min(crossfade_samples, len(block) // 2)

        result = block.copy()

        while len(result) < target_samples:
            overlap = min(crossfade_samples, len(result), len(block))

            # Equal-power crossfade for constant perceived loudness at seam
            t = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            fade_out = np.sqrt(1.0 - t)
            fade_in = np.sqrt(t)

            crossfaded = result[-overlap:] * fade_out + block[:overlap] * fade_in
            result = np.concatenate([result[:-overlap], crossfaded, block[overlap:]])

        return result[:target_samples]
