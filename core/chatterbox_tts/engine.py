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
    """Resolve a voice slug or filename to an existing .wav reference clip in the voice library.

    If voice_identifier is None, 'default', or 'auto', defaults to the primary
    clean library voice ('Brittney.wav'). If explicitly requested as 'resemble_default',
    returns None to trigger the Resemble AI checkpoint baseline.
    """
    if voice_identifier in ("resemble_default", "none", "(none)"):
        return None

    # If direct existing file path provided
    if voice_identifier:
        p = Path(voice_identifier)
        if p.is_file():
            return str(p.resolve())

    # Try matching against F5 voice registry
    try:
        from core.f5_tts.voice_registry import scan as scan_voices
        registry = scan_voices()
        if voice_identifier:
            # Exact match
            if voice_identifier in registry:
                return str(registry[voice_identifier]["default"]["audio"])
            # Case-insensitive match
            for slug, entry in registry.items():
                if slug.lower() == voice_identifier.lower():
                    return str(entry["default"]["audio"])
    except Exception as e:
        logger.debug("Voice registry lookup error: %s", e)

    # Check assets/speakers/reference_audio directory
    if _REF_AUDIO_DIR.is_dir():
        if voice_identifier and voice_identifier not in ("default", "auto"):
            clean_id = voice_identifier.replace("calm_", "").strip()
            candidates = [
                _REF_AUDIO_DIR / f"{voice_identifier}.wav",
                _REF_AUDIO_DIR / f"{voice_identifier.title()}.wav",
                _REF_AUDIO_DIR / f"{voice_identifier.capitalize()}.wav",
                _REF_AUDIO_DIR / f"{clean_id.title()}.wav",
            ]
            for c in candidates:
                if c.is_file():
                    return str(c.resolve())

            for f in _REF_AUDIO_DIR.glob("*.wav"):
                if clean_id.lower() == f.stem.lower():
                    return str(f.resolve())

        # Default fallback: Brittney.wav (primary library reference)
        brittney_path = _REF_AUDIO_DIR / "Brittney.wav"
        if brittney_path.is_file():
            return str(brittney_path.resolve())

        # Any available wav in directory
        all_wavs = sorted(_REF_AUDIO_DIR.glob("*.wav"))
        if all_wavs:
            return str(all_wavs[0].resolve())

    return None


def _trim_vocoder_silence(
    audio: np.ndarray,
    threshold_db: float = -45.0,
    min_keep_samples: int = 1200,
) -> np.ndarray:
    """Trim sub-perceptual vocoder artifacts from chunk head and tail.

    HiFi-GAN/HiFTNet vocoders do not produce true digital silence at
    sentence boundaries — they emit low-level metallic hissing, DC offset
    drift, and faint breath hallucinations. This helper trims energy below
    ``threshold_db`` from both ends while preserving a minimum safety
    margin (``min_keep_samples`` ≈ 50 ms at 24 kHz).

    Only trims; never extends. Returns the original array if it's too
    short or entirely above threshold.
    """
    if len(audio) < min_keep_samples * 2:
        return audio

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    abs_audio = np.abs(audio)

    # Find first sample above threshold (head trim)
    above = np.where(abs_audio > threshold_linear)[0]
    if len(above) == 0:
        return audio  # entirely below threshold — don't destroy

    start = max(0, above[0] - min_keep_samples)
    end = min(len(audio), above[-1] + min_keep_samples + 1)

    return audio[start:end]


class ChatterboxEngine(SpeechEngine):
    """SpeechEngine implementation for Chatterbox TTS (Resemble AI).

    Natively outputs mono float32 audio at 24 000 Hz, matching MoodScape's
    exact audio pipeline contract.
    """

    def __init__(
        self,
        voice_slug: str | None = None,
        exaggeration: float = 0.28,
        cfg_weight: float = 0.35,
        temperature: float = 0.55,
        device: str | None = None,
    ) -> None:
        """Initialize the Chatterbox engine.

        Args:
            voice_slug: Reference speaker slug (e.g. 'Brittney', 'Clara')
                        for zero-shot voice cloning. Defaults to 'Brittney'.
            exaggeration: Emotional expressiveness (0.1–1.0). 0.25–0.30 is optimal
                          for calm, spacious meditation and sleep narration.
            cfg_weight: Classifier-free guidance weight (0.1–1.0). 0.35 is optimal
                        for meditation — produces drawn-out vowels and eliminates
                        phase-smear chorus echo caused by over-conditioning.
            temperature: T3 autoregressive sampling temperature (0.3–0.7). Lower
                         values yield cleaner, more stable output. 0.55 eliminates
                         the vocoder jitter and buzz caused by the default 0.8.
            device: 'mps', 'cuda', or 'cpu'. Defaults to best available hardware.
        """
        self.voice_slug = voice_slug or "Brittney"
        self.ref_audio_path = _resolve_reference_audio(self.voice_slug)
        self.exaggeration = float(exaggeration)
        self.cfg_weight = float(cfg_weight)
        self.temperature = float(temperature)

        env_dev = os.environ.get("MOODSCAPE_CHATTERBOX_DEVICE")
        if env_dev in ("cpu", "mps", "cuda"):
            self.device = env_dev
        elif device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            # S3Gen Flow Matching suffers from numerical drift on MPS (Metal),
            # causing phase-smear and comb filtering (echo). We force CPU on Mac
            # for bit-exact, artifact-free synthesis.
            self.device = "cpu"

        self.model = None
        self._loaded = False
        self._cached_conds_key: tuple[str, float] | None = None

    def load_model(self) -> None:
        """Load the ChatterboxTTS weights onto target device."""
        if self._loaded and self.model is not None:
            return

        logger.info("Loading ChatterboxTTS onto %s...", self.device)
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self._loaded = True
        self._cached_conds_key = None
        logger.info("ChatterboxTTS loaded successfully (sample rate: %s Hz)", getattr(self.model, "sr", 24000))

    def unload_model(self) -> None:
        """Unload model and free GPU / MPS memory buffers."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        self._cached_conds_key = None

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
            voice: Voice slug or reference clip path (defaults to Brittney).
            speed: Playback / pacing speed (0.75–1.15).
            progress_cb: Optional callback (current_segment, total_segments).
            seed: Deterministic seed for reproducible generation.
            exaggeration: Emotional exaggeration (overrides default 0.28).
            cfg_weight: CFG weight (overrides default 0.50).

        Returns:
            Tuple of (voice_audio, voice_activity) where voice_audio is float32
            mono at 24 000 Hz and voice_activity is a parallel bool mask.
        """
        if not self._loaded or self.model is None:
            self.load_model()

        if seed is not None:
            torch.manual_seed(seed)

        # Resolve voice reference audio (defaults to Brittney if None/default)
        active_ref = _resolve_reference_audio(voice) if voice else self.ref_audio_path
        if active_ref is None and voice not in ("resemble_default", "none"):
            active_ref = _resolve_reference_audio("Brittney")

        # Resolve expressiveness params
        env_exagg = os.environ.get("MOODSCAPE_CHATTERBOX_EXAGGERATION")
        active_exagg = float(env_exagg) if env_exagg else (exaggeration or self.exaggeration)

        env_cfg = os.environ.get("MOODSCAPE_CHATTERBOX_CFG")
        active_cfg = float(env_cfg) if env_cfg else (cfg_weight or self.cfg_weight)

        env_temp = os.environ.get("MOODSCAPE_CHATTERBOX_TEMP")
        active_temp = float(env_temp) if env_temp else self.temperature

        # Prepare and cache voice conditionals once for all segments
        if active_ref:
            cache_key = (active_ref, round(active_exagg, 3))
            if self._cached_conds_key != cache_key:
                try:
                    logger.info("Preparing Chatterbox voice conditionals for %s (device=%s)...", Path(active_ref).name, self.device)
                    self.model.prepare_conditionals(active_ref, exaggeration=active_exagg)
                    self._cached_conds_key = cache_key
                except Exception as e:
                    if "metal" in str(e).lower() and self.device == "mps":
                        logger.warning("MPS Metal compilation failed during prepare_conditionals: %s. Switching to CPU...", e)
                        self.device = "cpu"
                        self.unload_model()
                        self.load_model()
                        self.model.prepare_conditionals(active_ref, exaggeration=active_exagg)
                        self._cached_conds_key = cache_key
                    else:
                        raise

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

                # Synthesize chunk via Chatterbox using cached conditionals
                try:
                    wav_tensor = self.model.generate(
                        text=text,
                        audio_prompt_path=None if active_ref else None,
                        exaggeration=active_exagg,
                        cfg_weight=active_cfg,
                        temperature=active_temp,
                        repetition_penalty=1.2,
                        min_p=0.05,
                    )
                except Exception as e:
                    if "metal" in str(e).lower() and self.device == "mps":
                        logger.warning("MPS Metal error during synthesis: %s. Falling back to CPU...", e)
                        self.device = "cpu"
                        self.unload_model()
                        self.load_model()
                        if active_ref:
                            self.model.prepare_conditionals(active_ref, exaggeration=active_exagg)
                            self._cached_conds_key = (active_ref, round(active_exagg, 3))
                        wav_tensor = self.model.generate(
                            text=text,
                            audio_prompt_path=None,
                            exaggeration=active_exagg,
                            cfg_weight=active_cfg,
                            temperature=active_temp,
                            repetition_penalty=1.2,
                            min_p=0.05,
                        )
                    else:
                        logger.error("Chatterbox failed on segment '%s': %s", text[:40], e)
                        fallback = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
                        audio_pieces.append(fallback)
                        activity_pieces.append(np.zeros(len(fallback), dtype=bool))
                        continue

                # Convert to mono numpy float32
                chunk = wav_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

                # Normalize peak to avoid internal clipping before postprocessor
                max_abs = np.max(np.abs(chunk)) if len(chunk) > 0 else 0.0
                if max_abs > 0.95:
                    chunk = chunk * (0.95 / max_abs)

                # Trim sub-perceptual vocoder artifacts (metallic hiss,
                # DC drift) from chunk head/tail before concatenation
                chunk = _trim_vocoder_silence(chunk)

                # Optional time stretching if speed != 1.0
                if abs(speed - 1.0) > 0.03:
                    import librosa
                    chunk = librosa.effects.time_stretch(chunk, rate=speed)

                activity = np.ones(len(chunk), dtype=bool)
                audio_pieces.append(chunk)
                activity_pieces.append(activity)

        if not audio_pieces:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

        voice_audio = np.concatenate(audio_pieces).astype(np.float32)
        voice_activity = np.concatenate(activity_pieces).astype(bool)

        return voice_audio, voice_activity

    def get_available_voices(self) -> list[dict[str, str]]:
        """Return all available Chatterbox voice options matching the library."""
        voices: list[dict[str, str]] = []
        if _REF_AUDIO_DIR.is_dir():
            for f in sorted(_REF_AUDIO_DIR.glob("*.wav")):
                slug = f.stem
                voices.append({
                    "id": slug,
                    "name": slug.replace("_", " ").title(),
                    "description": f"Voice library reference clip ({f.name})",
                })

        voices.append({
            "id": "resemble_default",
            "name": "Resemble AI Male Baseline",
            "description": "Standard pre-baked checkpoint voice (conds.pt)",
        })
        return voices
