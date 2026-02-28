"""MusicGen wrapper for generating ambient background music."""

import gc
import math

import numpy as np


NATIVE_SAMPLE_RATE = 32000
TARGET_SAMPLE_RATE = 24000
MAX_SEGMENT_DURATION = 30  # seconds
CROSSFADE_DURATION = 3     # seconds


class MusicEngine:
    """Wraps Meta's MusicGen to generate instrumental background music."""

    def __init__(self):
        self.model = None
        self.model_name = None

    def load_model(self):
        """Load MusicGen, falling back from melody to small on failure."""
        from audiocraft.models import MusicGen

        try:
            self.model = MusicGen.get_pretrained("facebook/musicgen-melody")
            self.model_name = "musicgen-melody"
        except Exception:
            self.model = MusicGen.get_pretrained("facebook/musicgen-small")
            self.model_name = "musicgen-small"

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
        """Generate background music of the requested duration.

        Args:
            prompt: Text description of desired music style.
            total_duration_sec: Target duration in seconds.
            progress_cb: Called with (current_segment, total_segments) after each segment.

        Returns:
            Mono float32 numpy array at 24000 Hz.
        """
        if self.model is None:
            raise RuntimeError("Music model not loaded. Call load_model() first.")

        import torch
        import torchaudio

        # Determine how many 30-second segments we need
        effective_segment = MAX_SEGMENT_DURATION - CROSSFADE_DURATION
        num_segments = max(1, math.ceil(total_duration_sec / effective_segment))

        # If total duration fits in a single segment, just generate it directly
        if total_duration_sec <= MAX_SEGMENT_DURATION:
            num_segments = 1
            segment_duration = total_duration_sec
        else:
            segment_duration = MAX_SEGMENT_DURATION

        segments: list[np.ndarray] = []

        for i in range(num_segments):
            self.model.set_generation_params(
                duration=segment_duration,
                use_sampling=True,
                top_k=250,
                temperature=1.0,
            )

            wav = self.model.generate([prompt])
            # wav shape: (batch, channels, samples)
            audio = wav[0].cpu().numpy()

            # Convert stereo to mono if needed
            if audio.ndim == 2 and audio.shape[0] > 1:
                audio = audio.mean(axis=0)
            elif audio.ndim == 2:
                audio = audio[0]

            segments.append(audio.astype(np.float32))

            if progress_cb is not None:
                progress_cb(i + 1, num_segments)

        # Crossfade segments together
        if len(segments) == 1:
            full_audio = segments[0]
        else:
            full_audio = self._crossfade_segments(segments)

        # Trim to exact requested duration (at native sample rate)
        target_samples = int(total_duration_sec * NATIVE_SAMPLE_RATE)
        if len(full_audio) > target_samples:
            full_audio = full_audio[:target_samples]

        # Resample from 32000 Hz to 24000 Hz
        audio_tensor = torch.from_numpy(full_audio).unsqueeze(0)  # (1, N)
        resampler = torchaudio.transforms.Resample(
            orig_freq=NATIVE_SAMPLE_RATE, new_freq=TARGET_SAMPLE_RATE
        )
        resampled = resampler(audio_tensor).squeeze(0).numpy()

        return resampled.astype(np.float32)

    def _crossfade_segments(self, segments: list[np.ndarray]) -> np.ndarray:
        """Join multiple audio segments with linear crossfade."""
        fade_len = int(CROSSFADE_DURATION * NATIVE_SAMPLE_RATE)
        result = segments[0]

        for seg in segments[1:]:
            # Ensure both have enough samples for the crossfade
            overlap = min(fade_len, len(result), len(seg))

            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)

            # Blend the overlap region
            crossfaded = result[-overlap:] * fade_out + seg[:overlap] * fade_in

            # Assemble: everything before overlap + crossfaded region + everything after overlap
            result = np.concatenate([result[:-overlap], crossfaded, seg[overlap:]])

        return result
