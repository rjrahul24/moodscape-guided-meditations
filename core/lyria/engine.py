"""LyriaEngine — Google Lyria RealTime API wrapper for MoodScape.

Lyria RealTime (models/lyria-realtime-exp) streams 48 kHz stereo 16-bit PCM
over a WebSocket session.  This engine wraps that streaming API and returns
a mono float32 numpy array at 48 kHz — the engine's native quality,
preserved without downsampling.

The pipeline (core/pipeline.py) handles the TTS-to-48 kHz upsampling and
mixes at 48 kHz when Lyria is selected, so audio fidelity is maintained
end-to-end.

Audio path:
    Lyria API → raw int16 48 kHz stereo PCM
              → deinterleave + average to mono float32
              → returned at 48 kHz (no resampling)

Session limits:
    Lyria sessions have a hard cap of ~10 minutes.  If the requested
    duration exceeds _MAX_SESSION_SEC (9.5 min), the engine splits the
    generation into multiple sequential sessions and crossfades them.

SynthID:
    All Lyria output is embedded with a SynthID watermark.  This engine
    does not strip or modify the watermark — do not add post-processing that
    would remove it.
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

logger = logging.getLogger("moodscape.lyria")

# ── Constants ──────────────────────────────────────────────────────────────────

_MODEL_ID = "models/lyria-realtime-exp"
_API_VERSION = "v1alpha"

# Lyria native output format
_NATIVE_SR = 48_000       # Hz
_BYTES_PER_SAMPLE = 2     # int16
_CHANNELS = 2             # stereo

# Maximum safe session duration (10-minute hard limit with 30s safety buffer)
_MAX_SESSION_SEC = 570.0  # 9.5 minutes

# Exported so pipeline.py can import the correct sample rate for this engine
TARGET_SAMPLE_RATE = _NATIVE_SR

# Generation defaults tuned for meditation
_DEFAULT_BPM = 70
_DEFAULT_DENSITY = 0.2    # sparse — avoids busy / distracting textures
_DEFAULT_BRIGHTNESS = 0.3  # warm / low-end weighted, no harsh treble
_DEFAULT_GUIDANCE = 4.0   # Lyria's recommended midpoint for prompt adherence

# Equal-power crossfade between multi-session chunks
_CROSSFADE_SEC = 3.0


# ── Engine ────────────────────────────────────────────────────────────────────

class LyriaEngine:
    """Drop-in music engine backed by the Google Lyria RealTime API.

    Public API mirrors HeartMulaEngine / AceStepEngine so the pipeline can
    treat all engines uniformly::

        engine = LyriaEngine()
        engine.load_model()
        audio = engine.generate(prompt, duration_sec, ...)
        engine.unload_model()

    Returns:
        Mono float32 numpy array at 48 kHz.
    """

    def __init__(self) -> None:
        self._client = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Validate the API key and initialise the Gemini client.

        Reads ``GOOGLE_API_KEY`` from the environment.  Raises ``ValueError``
        with a clear message if the key is absent so the error surfaces in
        the Gradio UI rather than crashing silently.
        """
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Add GOOGLE_API_KEY=<your_key> to the .env file at the project root "
                "and restart the app."
            )

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. Run: "
                "pip install 'google-genai>=1.16.0'"
            ) from exc

        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": _API_VERSION},
        )
        logger.info("Lyria RealTime client initialised (model=%s).", _MODEL_ID)

    def unload_model(self) -> None:
        """Release the client reference (no local GPU memory to free)."""
        self._client = None

    # ── Public generation entry point ─────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        total_duration_sec: float,
        progress_cb=None,
        prompt_stages: list[tuple[str, float]] | None = None,
        bpm: int = _DEFAULT_BPM,
        density: float = _DEFAULT_DENSITY,
        brightness: float = _DEFAULT_BRIGHTNESS,
        **kwargs,  # absorb unused kwargs (e.g. melody_audio, keyscale)
    ) -> np.ndarray:
        """Generate background music and return mono float32 at 48 kHz.

        Args:
            prompt: Music style description (plain text or weighted syntax
                    ``'Label: weight, Label2: weight2, ...'``).
            total_duration_sec: Target duration in seconds.
            progress_cb: Optional callback ``(current_step, total_steps)``.
            prompt_stages: Story-mode list of ``(prompt, duration_sec)`` pairs.
                When provided, one Lyria session is opened per stage and the
                results are crossfaded together.
            bpm: Beats per minute (60–200). Lower values suit meditation.
            density: Musical density (0.0–1.0). Low values produce sparse,
                meditative textures.
            brightness: Spectral brightness (0.0–1.0). Lower values keep the
                mix warm and avoid harsh treble.
            **kwargs: Ignored; present for API compatibility with other engines.

        Returns:
            Mono float32 numpy array at 48 000 Hz.
        """
        if self._client is None:
            raise RuntimeError(
                "LyriaEngine: call load_model() before generate()."
            )

        if prompt_stages is not None:
            return self._generate_story(
                prompt_stages, bpm, density, brightness, progress_cb
            )

        return self._generate_full(
            prompt, total_duration_sec, bpm, density, brightness, progress_cb
        )

    # ── Internal: single / multi-session generation ───────────────────────────

    def _generate_full(
        self,
        prompt: str,
        duration_sec: float,
        bpm: int,
        density: float,
        brightness: float,
        progress_cb,
    ) -> np.ndarray:
        """Generate a single continuous clip, splitting if longer than 9.5 min."""
        from core.lyria.prompts import build_lyria_prompts

        weighted_prompts = build_lyria_prompts(prompt)

        if duration_sec <= _MAX_SESSION_SEC:
            if progress_cb:
                progress_cb(0, 1)
            pcm = asyncio.run(
                self._run_session(weighted_prompts, duration_sec, bpm, density, brightness)
            )
            if progress_cb:
                progress_cb(1, 1)
            return self._pcm_to_numpy(pcm)

        # Long-form: split into ≤9.5-minute chunks
        chunks: list[np.ndarray] = []
        remaining = duration_sec
        chunk_idx = 0
        total_chunks = int(np.ceil(duration_sec / _MAX_SESSION_SEC))

        while remaining > 0.5:
            chunk_dur = min(remaining, _MAX_SESSION_SEC)
            if progress_cb:
                progress_cb(chunk_idx, total_chunks)
            logger.info(
                "Lyria: generating chunk %d/%d (%.1fs)",
                chunk_idx + 1, total_chunks, chunk_dur,
            )
            pcm = asyncio.run(
                self._run_session(weighted_prompts, chunk_dur, bpm, density, brightness)
            )
            chunks.append(self._pcm_to_numpy(pcm))
            remaining -= chunk_dur
            chunk_idx += 1

        if progress_cb:
            progress_cb(total_chunks, total_chunks)

        return self._crossfade_chunks(chunks)

    def _generate_story(
        self,
        prompt_stages: list[tuple[str, float]],
        bpm: int,
        density: float,
        brightness: float,
        progress_cb,
    ) -> np.ndarray:
        """Generate per-stage audio segments and crossfade them together."""
        from core.lyria.prompts import build_lyria_prompts

        total = len(prompt_stages)
        segments: list[np.ndarray] = []

        for i, (stage_prompt, stage_dur) in enumerate(prompt_stages):
            if progress_cb:
                progress_cb(i, total)
            logger.info(
                "Lyria story stage %d/%d: '%.40s' (%.1fs)",
                i + 1, total, stage_prompt, stage_dur,
            )
            weighted = build_lyria_prompts(stage_prompt)
            # Split each stage if it exceeds the session limit
            if stage_dur <= _MAX_SESSION_SEC:
                pcm = asyncio.run(
                    self._run_session(weighted, stage_dur, bpm, density, brightness)
                )
                segments.append(self._pcm_to_numpy(pcm))
            else:
                # Recursively chunk the stage
                stage_audio = self._generate_full(
                    stage_prompt, stage_dur, bpm, density, brightness, None
                )
                segments.append(stage_audio)

        if progress_cb:
            progress_cb(total, total)

        return self._crossfade_chunks(segments)

    # ── Async WebSocket session ────────────────────────────────────────────────

    async def _run_session(
        self,
        weighted_prompts,
        duration_sec: float,
        bpm: int,
        density: float,
        brightness: float,
    ) -> bytes:
        """Open one Lyria WebSocket session and collect PCM bytes.

        The session is closed as soon as ``target_bytes`` have been received,
        ensuring we never collect more audio than needed.

        Returns:
            Raw int16 48 kHz stereo PCM bytes, trimmed to exact target length.
        """
        from google.genai import types

        target_bytes = int(_NATIVE_SR * _BYTES_PER_SAMPLE * _CHANNELS * duration_sec)
        pcm_buffer = bytearray()

        logger.info(
            "Lyria: opening session — duration=%.1fs, bpm=%d, density=%.2f, brightness=%.2f",
            duration_sec, bpm, density, brightness,
        )

        async with self._client.aio.live.music.connect(model=_MODEL_ID) as session:
            await session.set_weighted_prompts(prompts=weighted_prompts)
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(
                    bpm=bpm,
                    density=density,
                    brightness=brightness,
                    guidance=_DEFAULT_GUIDANCE,
                )
            )
            await session.play()

            async for message in session.receive():
                if (
                    hasattr(message, "server_content")
                    and message.server_content
                    and message.server_content.audio_chunks
                ):
                    for chunk in message.server_content.audio_chunks:
                        pcm_buffer.extend(chunk.data)

                if len(pcm_buffer) >= target_bytes:
                    await session.stop()
                    break

        received = len(pcm_buffer)
        logger.info(
            "Lyria: session complete — received %d bytes (target %d, %.1fs)",
            received, target_bytes, duration_sec,
        )

        # Trim to exact target (int16 stereo alignment: multiples of 4 bytes)
        aligned = (target_bytes // (_BYTES_PER_SAMPLE * _CHANNELS)) * (_BYTES_PER_SAMPLE * _CHANNELS)
        return bytes(pcm_buffer[:aligned])

    # ── Audio conversion ───────────────────────────────────────────────────────

    def _pcm_to_numpy(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert raw 48 kHz stereo int16 PCM to mono float32 at 48 kHz.

        Audio path:
            bytes (int16, interleaved stereo)
            → float32 normalised to [-1, 1]
            → reshape to (samples, 2)
            → average channels to mono
            → mono float32 at 48 kHz (no resampling — native quality)
        """
        if not pcm_bytes:
            logger.warning("Lyria: received empty PCM buffer — returning silence.")
            return np.zeros(int(_NATIVE_SR * 1.0), dtype=np.float32)

        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        arr /= 32768.0  # int16 → float32 in [-1, 1]

        # Ensure even length for stereo deinterleaving
        if len(arr) % 2 != 0:
            arr = arr[:-1]

        stereo = arr.reshape(-1, 2)  # (num_samples, 2) — columns: [L, R]
        mono = stereo.mean(axis=1)   # equal-weight channel average
        return mono.astype(np.float32)

    # ── Crossfade ──────────────────────────────────────────────────────────────

    def _crossfade_chunks(self, chunks: list[np.ndarray]) -> np.ndarray:
        """Concatenate audio chunks with equal-power cosine crossfade.

        Uses a pi/2 cosine taper (cos/sin pair) so that
        ``fade_out² + fade_in² = 1`` at every sample — no energy dip at seams.

        Args:
            chunks: List of mono float32 arrays at _NATIVE_SR.

        Returns:
            Single mono float32 array.
        """
        if len(chunks) == 1:
            return chunks[0]

        xfade_samples = int(_CROSSFADE_SEC * _NATIVE_SR)
        result = chunks[0]

        for nxt in chunks[1:]:
            fade_len = min(xfade_samples, len(result), len(nxt))
            t = np.linspace(0.0, np.pi / 2.0, fade_len, dtype=np.float32)
            fade_out = np.cos(t)
            fade_in = np.sin(t)
            # Overlap-add: tail of result + head of nxt
            overlap = result[-fade_len:] * fade_out + nxt[:fade_len] * fade_in
            result = np.concatenate([result[:-fade_len], overlap, nxt[fade_len:]])

        return result
