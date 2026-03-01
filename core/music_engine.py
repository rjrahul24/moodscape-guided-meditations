"""MusicGen wrapper for generating ambient background music."""

import gc
import math

import numpy as np


NATIVE_SAMPLE_RATE = 32000
TARGET_SAMPLE_RATE = 24000
SEGMENT_DURATION = 30       # Max seconds per MusicGen call
CONTEXT_DURATION = 10       # Seconds of overlap fed as audio prompt for continuation
CROSSFADE_DURATION = 2      # Seconds of crossfade at each segment seam
MAX_UNIQUE_DURATION = 50    # Generate up to this many seconds of unique music, then loop
LOOP_CROSSFADE = 5          # Seconds of crossfade when looping the stitched block


class MusicEngine:
    """Wraps Meta's MusicGen to generate instrumental background music.

    Uses musicgen-medium (1.5B params) for rich ambient textures, with a
    sliding-window continuation strategy to produce evolving, non-repetitive
    tracks. Caps unique generation at ~50 seconds (2 segments) to keep
    generation time reasonable on CPU, then loops with crossfade.
    """

    def __init__(self):
        self.model = None
        self.model_name = None

    def load_model(self):
        """Load MusicGen-medium (1.5B params — rich ambient textures for meditation)."""
        import warnings

        from audiocraft.models import MusicGen

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
            self.model = MusicGen.get_pretrained("facebook/musicgen-medium")
            self.model_name = "musicgen-medium"

    def unload_model(self):
        """Release model and free GPU memory."""
        del self.model
        self.model = None
        self.model_name = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(
        self,
        prompt: str,
        total_duration_sec: float,
        progress_cb=None,
    ) -> np.ndarray:
        """Generate background music using sliding-window continuation.

        Produces up to MAX_UNIQUE_DURATION seconds of evolving music by chaining
        MusicGen continuation calls (typically 2 segments), then loops the result
        with crossfade if more duration is needed.

        Args:
            prompt: Text description of desired music style (should already be
                    enhanced with meditation context by the caller).
            total_duration_sec: Target duration in seconds.
            progress_cb: Called with (current_segment, total_segments) after each segment.

        Returns:
            Mono float32 numpy array at 24000 Hz.
        """
        if self.model is None:
            raise RuntimeError("Music model not loaded. Call load_model() first.")

        import torch
        import torchaudio

        # Determine how much unique music to generate (capped at MAX_UNIQUE_DURATION)
        unique_duration = min(total_duration_sec, MAX_UNIQUE_DURATION)

        # Calculate number of segments needed
        if unique_duration <= SEGMENT_DURATION:
            num_segments = 1
        else:
            # First segment gives SEGMENT_DURATION seconds.
            # Each continuation adds (SEGMENT_DURATION - CONTEXT_DURATION) net seconds.
            net_per_continuation = SEGMENT_DURATION - CONTEXT_DURATION
            num_segments = 1 + math.ceil(
                (unique_duration - SEGMENT_DURATION) / net_per_continuation
            )

        # ── Segment 1: generate from text prompt ──────────────────────────
        self.model.set_generation_params(
            duration=SEGMENT_DURATION,
            use_sampling=True,
            top_k=250,
            top_p=0.0,
            temperature=0.8,
            cfg_coef=3.5,
        )

        wav = self.model.generate([prompt])
        # wav shape: (batch=1, channels, samples)
        segments = [wav[0]]  # keep as tensor until stitching

        if progress_cb is not None:
            progress_cb(1, num_segments)

        # ── Segments 2..N: continuation with audio context ────────────────
        for i in range(1, num_segments):
            prev = segments[-1]  # (channels, samples)

            # Extract last CONTEXT_DURATION seconds as audio prompt
            context_samples = int(CONTEXT_DURATION * NATIVE_SAMPLE_RATE)
            context = prev[:, -context_samples:]  # (channels, context_samples)
            # generate_continuation expects (batch, channels, samples)
            context_batch = context.unsqueeze(0)

            self.model.set_generation_params(
                duration=SEGMENT_DURATION,
                use_sampling=True,
                top_k=250,
                top_p=0.0,
                temperature=0.8,
                cfg_coef=3.5,
            )

            continuation = self.model.generate_continuation(
                prompt=context_batch,
                prompt_sample_rate=NATIVE_SAMPLE_RATE,
                descriptions=[prompt],
            )
            segments.append(continuation[0])  # (channels, samples)

            if progress_cb is not None:
                progress_cb(i + 1, num_segments)

        # ── Stitch segments with crossfade ────────────────────────────────
        unique_audio = self._stitch_segments(segments)

        # Convert to mono numpy
        if unique_audio.ndim == 2 and unique_audio.shape[0] > 1:
            unique_audio = unique_audio.mean(dim=0)
        elif unique_audio.ndim == 2:
            unique_audio = unique_audio[0]

        unique_np = unique_audio.cpu().numpy().astype(np.float32)

        # ── Loop if needed to fill total duration ─────────────────────────
        target_samples = int(total_duration_sec * NATIVE_SAMPLE_RATE)
        if len(unique_np) < target_samples:
            full_audio = self._loop_block(unique_np, target_samples)
        else:
            full_audio = unique_np[:target_samples]

        # ── Resample from 32000 Hz to 24000 Hz ───────────────────────────
        audio_tensor = torch.from_numpy(full_audio).unsqueeze(0)  # (1, N)
        resampler = torchaudio.transforms.Resample(
            orig_freq=NATIVE_SAMPLE_RATE, new_freq=TARGET_SAMPLE_RATE
        )
        resampled = resampler(audio_tensor).squeeze(0).numpy()

        return resampled.astype(np.float32)

    def _stitch_segments(self, segments: list) -> "torch.Tensor":
        """Stitch continuation segments by stripping context overlap and crossfading.

        Each continuation segment's first CONTEXT_DURATION seconds duplicate the
        end of the previous segment. We strip that overlap and apply a short
        crossfade at the seam for smooth transitions.
        """
        import torch

        if len(segments) == 1:
            return segments[0]

        crossfade_samples = int(CROSSFADE_DURATION * NATIVE_SAMPLE_RATE)
        context_samples = int(CONTEXT_DURATION * NATIVE_SAMPLE_RATE)

        # Start with the first segment in full
        result = segments[0]

        for seg in segments[1:]:
            # Strip the context region (first CONTEXT_DURATION seconds)
            new_audio = seg[:, context_samples:]

            if new_audio.shape[1] == 0:
                continue

            # Apply crossfade at the seam
            cf = min(crossfade_samples, result.shape[1], new_audio.shape[1])
            if cf > 0:
                fade_out = torch.linspace(1.0, 0.0, cf, device=result.device)
                fade_in = torch.linspace(0.0, 1.0, cf, device=result.device)

                # Crossfade the overlap region
                crossfaded = result[:, -cf:] * fade_out + new_audio[:, :cf] * fade_in
                result = torch.cat([result[:, :-cf], crossfaded, new_audio[:, cf:]], dim=1)
            else:
                result = torch.cat([result, new_audio], dim=1)

        return result

    def _loop_block(self, block: np.ndarray, target_samples: int) -> np.ndarray:
        """Loop a stitched audio block with crossfade to fill the target duration."""
        if len(block) >= target_samples:
            return block[:target_samples]

        crossfade_samples = int(LOOP_CROSSFADE * NATIVE_SAMPLE_RATE)
        result = block.copy()

        while len(result) < target_samples:
            overlap = min(crossfade_samples, len(result), len(block))

            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)

            crossfaded = result[-overlap:] * fade_out + block[:overlap] * fade_in
            result = np.concatenate([result[:-overlap], crossfaded, block[overlap:]])

        return result[:target_samples]
