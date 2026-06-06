"""UploadMusicEngine — user-supplied instrumental as a music source.

This engine lets a user upload their own instrumental/background file instead
of generating one with ACE-Step or Lyria.  It implements the same public
contract as the generative engines so the pipeline can treat all music sources
uniformly::

    engine = UploadMusicEngine(path)
    engine.load_model()
    audio = engine.generate(prompt, duration_sec, ...)
    engine.unload_model()

The uploaded file is decoded, resampled to 48 kHz, downmixed to mono float32,
and fitted to exactly ``total_duration_sec`` (loop / trim / used-as-is) so the
returned array is indistinguishable from generative-engine output and flows
through the existing FX / ducking / mastering path unchanged.

``prompt`` and music-generation kwargs (bpm, keyscale, prompt_stages, melody
audio, …) are ignored — there is nothing to generate.
"""

from __future__ import annotations

import logging

import numpy as np
from pedalboard.io import AudioFile

from core.upload_music.arrange import FitReport, fit_to_length

logger = logging.getLogger("moodscape.upload")

# Uploaded audio is decoded to this rate; matches ACE-Step / Lyria output so
# the pipeline mixes everything at a single rate.
TARGET_SAMPLE_RATE = 48_000


class UploadMusicEngine:
    """Drop-in music engine backed by a user-uploaded instrumental file.

    Returns:
        Mono float32 numpy array at 48 kHz, exactly ``total_duration_sec`` long.
    """

    def __init__(self, uploaded_path: str | None) -> None:
        if not uploaded_path:
            raise ValueError(
                "UploadMusicEngine requires a path to an uploaded instrumental file."
            )
        self.uploaded_path = uploaded_path
        self.fit_report: FitReport | None = None

    # ── Lifecycle (no weights; present for contract symmetry) ────────────────
    def load_model(self) -> None:
        return None

    def unload_model(self) -> None:
        return None

    # ── Public generation entry point ────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        total_duration_sec: float,
        progress_cb=None,
        **kwargs,  # absorb unused generative kwargs (prompt_stages, bpm, …)
    ) -> np.ndarray:
        """Load, resample, downmix and fit the uploaded instrumental.

        Args:
            prompt: Ignored (no generation happens).
            total_duration_sec: Target duration in seconds.
            progress_cb: Optional ``(current, total)`` callback.
            **kwargs: Ignored; present for API compatibility with other engines.

        Returns:
            Mono float32 numpy array at 48 kHz with exactly
            ``round(total_duration_sec * 48000)`` samples.
        """
        if progress_cb:
            progress_cb(0, 1)

        # Decode + resample to 48 kHz. AudioFile handles wav/mp3/flac/ogg/m4a/
        # aiff via libsndfile and returns shape (channels, samples).
        with AudioFile(str(self.uploaded_path)).resampled_to(TARGET_SAMPLE_RATE) as f:
            audio = f.read(f.frames)

        # Downmix to mono float32.
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        if audio.shape[0] == 0:
            raise ValueError(
                f"Uploaded instrumental decoded to empty audio: {self.uploaded_path}"
            )

        target_samples = int(round(total_duration_sec * TARGET_SAMPLE_RATE))
        fitted, report = fit_to_length(audio, TARGET_SAMPLE_RATE, target_samples)
        self.fit_report = report

        logger.info(
            "[Upload] %s — %.1fs source → %.1fs target (%s, loops=%d)",
            self.uploaded_path, report.source_seconds, report.target_seconds,
            report.mode, report.loops,
        )

        if progress_cb:
            progress_cb(1, 1)

        # fit_to_length returns (1, n) for mono input → squeeze to 1-D.
        return np.ascontiguousarray(np.squeeze(fitted), dtype=np.float32)
