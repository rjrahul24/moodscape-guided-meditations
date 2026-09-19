"""Chatterbox TTS engine wrapper implementing SpeechEngine contract.

Provides high-fidelity synthesis and zero-shot voice cloning for guided
meditations and sleep stories using Resemble AI's Chatterbox 500M model.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.speech_engine import SAMPLE_RATE, SpeechEngine

logger = logging.getLogger("moodscape.chatterbox")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REF_AUDIO_DIR = _PROJECT_ROOT / "assets" / "speakers" / "reference_audio"


def _resolve_reference_audio(voice_identifier: str | None) -> str | None:
    """Resolve a voice slug or filename to an existing .wav reference clip."""
    if not voice_identifier or voice_identifier in ("default", "none", "(none)"):
        return None

    # If direct existing file path
    p = Path(voice_identifier)
    if p.is_file():
        return str(p.resolve())

    # Check assets/speakers/reference_audio
    candidates = [
        _REF_AUDIO_DIR / f"{voice_identifier}.wav",
        _REF_AUDIO_DIR / f"{voice_identifier.title()}.wav",
        _REF_AUDIO_DIR / f"{voice_identifier.replace('calm_', '').title()}.wav",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())

    # Look for partial matches in directory
    if _REF_AUDIO_DIR.is_dir():
        for f in _REF_AUDIO_DIR.glob("*.wav"):
            if voice_identifier.lower() in f.stem.lower():
                return str(f.resolve())

    return None


class ChatterboxEngine(SpeechEngine):
    """SpeechEngine implementation for Chatterbox TTS (Resemble AI).

    Natively outputs mono float32 audio at 24 000 Hz, matching MoodScape's
    exact audio pipeline contract.
    """

    def __init__(
        self,
        voice_slug: str | None = None,
        exaggeration: float = 0.30,
        cfg_weight: float = 0.50,
        device: str | None = None,
    ) -> None:
        """Initialize the Chatterbox engine.

        Args:
            voice_slug: Optional reference speaker slug (e.g. 'Brittney', 'Clara')
                        for zero-shot voice cloning. If None, uses default voice.
            exaggeration: Emotional expressiveness (0.1–1.0). 0.25–0.35 is optimal
                          for calm, spacious meditation and sleep narration.
            cfg_weight: Classifier-free guidance weight (0.1–1.0).
            device: 'mps', 'cuda', or 'cpu'. Defaults to best available hardware.
        """
        self.voice_slug = voice_slug
        self.ref_audio_path = _resolve_reference_audio(voice_slug)
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight

        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the ChatterboxTTS weights onto target device."""
        if self._loaded and self.model is not None:
            return

        logger.info("Loading ChatterboxTTS onto %s...", self.device)
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self._loaded = True
        logger.info("ChatterboxTTS loaded successfully (sample rate: %s Hz)", getattr(self.model, "sr", 24000))

    def unload_model(self) -> None:
        """Unload model and free GPU / MPS memory buffers."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        logger.debug("ChatterboxTTS unloaded and memory freed.")

    def synthesize(
        self,
        segments: list[dict[str, Any]],
        voice: str | None = None,
        speed: float = 0.90,
        progress_cb=None,
        seed: int | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize meditation script segments into 24 kHz mono audio.

        Args:
            segments: List of dicts with 'type' ('speech' or 'pause') and
                      'text' or 'duration_sec'.
            voice: Voice slug or reference clip path (overrides init setting).
            speed: Playback / pacing speed (0.75–1.15).
            progress_cb: Optional callback (current_segment, total_segments).
            seed: Deterministic seed for reproducible generation.
            exaggeration: Emotional exaggeration (overrides default 0.30).
            cfg_weight: CFG weight (overrides default 0.50).

        Returns:
            Tuple of (voice_audio, voice_activity) where voice_audio is float32
            mono at 24 000 Hz and voice_activity is a parallel bool mask.
        """
        if not self._loaded or self.model is None:
            self.load_model()

        if seed is not None:
            torch.manual_seed(seed)

        # Resolve voice reference audio
        active_ref = _resolve_reference_audio(voice) if voice else self.ref_audio_path

        # Resolve expressiveness params
        env_exagg = os.environ.get("MOODSCAPE_CHATTERBOX_EXAGGERATION")
        active_exagg = float(env_exagg) if env_exagg else (exaggeration or self.exaggeration)

        env_cfg = os.environ.get("MOODSCAPE_CHATTERBOX_CFG")
        active_cfg = float(env_cfg) if env_cfg else (cfg_weight or self.cfg_weight)

        audio_pieces: list[np.ndarray] = []
        activity_pieces: list[np.ndarray] = []
        total_segments = len(segments)

        for idx, seg in enumerate(segments):
            if progress_cb:
                progress_cb(idx + 1, total_segments)

            seg_type = seg.get("type", "speech")

            if seg_type == "pause":
                duration = float(seg.get("duration_sec", 1.0))
                num_samples = int(duration * SAMPLE_RATE)
                if num_samples > 0:
                    pause_audio = np.zeros(num_samples, dtype=np.float32)
                    pause_activity = np.zeros(num_samples, dtype=bool)
                    audio_pieces.append(pause_audio)
                    activity_pieces.append(pause_activity)

            elif seg_type == "speech":
                text = seg.get("text", "").strip()
                if not text:
                    continue

                # Synthesize chunk via Chatterbox
                try:
                    wav_tensor = self.model.generate(
                        text=text,
                        audio_prompt_path=active_ref,
                        exaggeration=active_exagg,
                        cfg_weight=active_cfg,
                        temperature=0.75,
                        repetition_penalty=1.2,
                    )
                    # Convert to mono numpy float32
                    chunk = wav_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

                    # Normalize peak to avoid internal clipping before postprocessor
                    max_abs = np.max(np.abs(chunk)) if len(chunk) > 0 else 0.0
                    if max_abs > 0.95:
                        chunk = chunk * (0.95 / max_abs)

                    # Optional time stretching if speed != 1.0
                    if abs(speed - 1.0) > 0.03:
                        import librosa
                        chunk = librosa.effects.time_stretch(chunk, rate=speed)

                    activity = np.ones(len(chunk), dtype=bool)
                    audio_pieces.append(chunk)
                    activity_pieces.append(activity)

                except Exception as e:
                    logger.error("Chatterbox failed on segment '%s': %s", text[:40], e)
                    # Fallback short silence so the pipeline does not collapse
                    fallback = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
                    audio_pieces.append(fallback)
                    activity_pieces.append(np.zeros(len(fallback), dtype=bool))

        if not audio_pieces:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

        voice_audio = np.concatenate(audio_pieces).astype(np.float32)
        voice_activity = np.concatenate(activity_pieces).astype(bool)

        return voice_audio, voice_activity

    def get_available_voices(self) -> list[dict[str, str]]:
        """Return all available Chatterbox voice options."""
        voices = [
            {
                "id": "default",
                "name": "Chatterbox Default",
                "description": "Standard Resemble AI baseline voice.",
            }
        ]
        if _REF_AUDIO_DIR.is_dir():
            for f in sorted(_REF_AUDIO_DIR.glob("*.wav")):
                slug = f.stem
                voices.append({
                    "id": slug,
                    "name": f"{slug.replace('_', ' ').title()} (Zero-Shot Clone)",
                    "description": f"Cloned from {f.name}",
                })
        return voices
