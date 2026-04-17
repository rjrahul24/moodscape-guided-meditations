"""Chatterbox TTS engine — wraps Chatterbox 0.5B for emotion-controlled meditation narration.

Chatterbox (Resemble AI, MIT license) is a 0.5B-parameter TTS model with:
  - Zero-shot voice cloning from a 5-15s reference clip
  - Emotion exaggeration dial (0.0=monotone → 1.0=dramatic)
  - Paralinguistic tags: [laugh], [chuckle], [cough]
  - cfg_weight for voice adherence and pacing control
  - 24 kHz output at 16-bit quality

For meditation, we use moderate exaggeration (~0.45) + low cfg_weight (~0.2)
for warm, deliberate delivery with natural expressiveness.

Device: MPS on Apple Silicon, CPU fallback elsewhere.
Memory: ~4-6 GB (loads AFTER music model unloads in the pipeline).
"""

import gc
import logging
import random
import re
from pathlib import Path

import numpy as np

from core.kokoro_tts.postprocessor import (
    apply_segment_fades,
    generate_room_tone,
    humanize_voice,
    normalize_chunk_rms,
    reduce_synthesis_noise,
)
from core.speech_engine import SAMPLE_RATE, SpeechEngine

logger = logging.getLogger(__name__)

# Default voice assets directory
_ASSETS_DIR = Path(__file__).parent / "assets"
_REF_AUDIO_DIR = _ASSETS_DIR / "reference_audio"

# Meditation-optimized defaults
_DEFAULT_EXAGGERATION = 0.45  # Moderate = warmth + care; 0.25 was near-monotone
_DEFAULT_CFG_WEIGHT = 0.2     # Lower cfg = slower, more deliberate pacing
_DEFAULT_SPEED = 0.90         # Meditation pace (ABC compliance; pacing via cfg_weight)

# Inter-sentence pauses for natural meditation pacing
INTER_SENTENCE_PAUSE_SEC = 1.0   # Breathing room between sentences (Chatterbox speaks faster)
ELLIPSIS_PAUSE_SEC = 1.8         # Contemplative pauses after "..."

# Trailing-silence trimmer (Chatterbox generates silence padding)
_TRIM_THRESHOLD_DB = -45.0
_TRIM_TAIL_MS = 50.0


def _trim_trailing_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove trailing silence from a Chatterbox speech chunk."""
    threshold = 10 ** (_TRIM_THRESHOLD_DB / 20.0)
    active = np.where(np.abs(audio) > threshold)[0]
    if len(active) == 0:
        return audio
    tail = int(_TRIM_TAIL_MS / 1000.0 * sr)
    cut = min(int(active[-1]) + tail + 1, len(audio))
    return audio[:cut]


def _split_text_into_sentences(text: str) -> list[str]:
    """Split text into sentences for per-sentence synthesis.

    Chatterbox handles short sentences better than long paragraphs.
    Splits on sentence boundaries (.!?) while preserving ellipses intact.
    """
    # Replace ellipses with placeholder
    text = text.replace("...", "〰ELLIPSIS〰")
    text = text.replace("…", "〰ELLIPSIS〰")

    # Split on sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Restore ellipses
    sentences = [s.replace("〰ELLIPSIS〰", "...") for s in sentences]

    # Filter out empty strings
    return [s.strip() for s in sentences if s.strip()]


class ChatterboxEngine(SpeechEngine):
    """Wraps Chatterbox TTS for emotion-controlled meditation narration.

    Implements the SpeechEngine interface — produces mono float32 audio at
    24 000 Hz with a parallel boolean voice-activity mask, matching the
    contract expected by the pipeline's mixing and FX stages.
    """

    def __init__(
        self,
        exaggeration: float = _DEFAULT_EXAGGERATION,
        cfg_weight: float = _DEFAULT_CFG_WEIGHT,
        reference_audio: str | None = None,
    ) -> None:
        """Initialise the engine with emotion and voice cloning parameters.

        Args:
            exaggeration: Emotion intensity dial (0.0-1.0). Moderate (~0.45) for
                warm meditation, lower for deep sleep, higher for energetic narration.
            cfg_weight: Voice adherence / pacing control (0.0-1.0). Lower
                values produce slower, more deliberate speech.
            reference_audio: Optional path to a reference .wav file for
                voice cloning. If None, uses Chatterbox's default voice.
        """
        self._model = None
        self._exaggeration = float(np.clip(exaggeration, 0.0, 1.0))
        self._cfg_weight = float(np.clip(cfg_weight, 0.0, 1.0))
        self._reference_audio = reference_audio

        # Validate reference audio if provided
        if self._reference_audio is not None:
            ref_path = Path(self._reference_audio)
            if not ref_path.exists():
                logger.warning(
                    "Reference audio not found: %s. Will use default voice.",
                    self._reference_audio,
                )
                self._reference_audio = None

    def load_model(self) -> None:
        """Load the Chatterbox TTS model onto MPS (Apple Silicon) or CPU.

        Uses the official `ChatterboxTTS.from_pretrained()` which downloads
        model weights from HuggingFace on first run (~2 GB).
        """
        if self._model is not None:
            return

        import torch

        # Detect device: MPS for Apple Silicon, CUDA if available, else CPU
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        logger.info("Loading Chatterbox TTS (0.5B) on %s...", device)

        # Patch torch.load for MPS compatibility (from official example_for_mac.py)
        map_location = torch.device(device)
        torch_load_original = torch.load

        def patched_torch_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = map_location
            return torch_load_original(*args, **kwargs)

        torch.load = patched_torch_load

        try:
            from chatterbox.tts import ChatterboxTTS
            self._model = ChatterboxTTS.from_pretrained(device=device)
            logger.info("Chatterbox TTS loaded successfully on %s.", device)
        finally:
            torch.load = torch_load_original  # restore original torch.load

    def unload_model(self) -> None:
        """Release model weights and free device memory."""
        if self._model is not None:
            del self._model
            self._model = None

        gc.collect()

        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Chatterbox TTS unloaded.")

    def synthesize(
        self,
        segments: list[dict],
        voice: str = "",
        speed: float = _DEFAULT_SPEED,
        progress_cb=None,
        seed: int | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        reference_audio: str | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize speech from parsed script segments.

        Each speech segment is split into individual sentences, synthesized
        with emotion control, then crossfaded together with inter-sentence
        pauses for natural meditation pacing.

        Args:
            segments:        Parsed script segments (from script_parser).
            voice:           Unused (ABC compliance). Voice is set via
                             reference_audio at init or per-call.
            speed:           Speaking speed (unused by Chatterbox directly;
                             pacing is controlled via cfg_weight).
            progress_cb:     Called with (current_index, total_segments).
            seed:            Optional deterministic seed.
            exaggeration:    Override emotion intensity for this call.
            cfg_weight:      Override voice adherence for this call.
            reference_audio: Override reference audio for this call.
            **kwargs:        Absorbs additional kwargs from pipeline.

        Returns:
            voice_audio:    float32 mono numpy array at 24 000 Hz.
            voice_activity: bool array (same length), True where voice is active.
        """
        if self._model is None:
            raise RuntimeError("Chatterbox model not loaded. Call load_model() first.")

        import torch
        import torchaudio as ta

        # Resolve parameters (per-call overrides > init defaults)
        exag = exaggeration if exaggeration is not None else self._exaggeration
        cfg = cfg_weight if cfg_weight is not None else self._cfg_weight
        ref_audio = reference_audio or self._reference_audio

        # Set deterministic seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        audio_chunks: list[np.ndarray] = []
        activity_chunks: list[np.ndarray] = []
        total = len(segments)

        for idx, segment in enumerate(segments):
            if segment["type"] == "speech":
                sentences = _split_text_into_sentences(segment["text"])
                if not sentences:
                    sentences = [segment["text"]]

                for s_idx, sentence in enumerate(sentences):
                    # Generate speech via Chatterbox
                    gen_kwargs = dict(
                        text=sentence,
                        exaggeration=exag,
                        cfg_weight=cfg,
                    )
                    if ref_audio is not None:
                        gen_kwargs["audio_prompt_path"] = ref_audio

                    wav = self._model.generate(**gen_kwargs)

                    # Convert to numpy float32
                    if hasattr(wav, "cpu"):
                        arr = wav.cpu().squeeze().numpy().astype(np.float32)
                    else:
                        arr = np.asarray(wav, dtype=np.float32).squeeze()

                    # Trim trailing silence
                    arr = _trim_trailing_silence(arr, SAMPLE_RATE)

                    # Resample if Chatterbox outputs at different rate
                    model_sr = getattr(self._model, 'sr', SAMPLE_RATE)
                    if model_sr != SAMPLE_RATE:
                        audio_tensor = torch.from_numpy(arr).unsqueeze(0)
                        audio_tensor = ta.functional.resample(audio_tensor, model_sr, SAMPLE_RATE)
                        arr = audio_tensor.squeeze().numpy().astype(np.float32)

                    # Per-chunk cleanup: normalize volume + fade edges to mask
                    # cold-start artifacts and prevent hard cuts between sentences
                    arr = normalize_chunk_rms(arr, target_db=-23.0)
                    arr = apply_segment_fades(arr)

                    audio_chunks.append(arr)
                    activity_chunks.append(np.ones(len(arr), dtype=bool))

                    # Inter-sentence pause (skip for last sentence of last segment)
                    is_last_sentence = s_idx == len(sentences) - 1
                    next_is_pause = (
                        is_last_sentence
                        and idx + 1 < total
                        and segments[idx + 1]["type"] == "pause"
                    )
                    if not is_last_sentence or (idx < total - 1 and not next_is_pause):
                        pause_sec = (
                            ELLIPSIS_PAUSE_SEC
                            if sentence.rstrip().endswith(("...", "\u2026"))
                            else INTER_SENTENCE_PAUSE_SEC
                        )
                        room_tone = generate_room_tone(pause_sec, sr=SAMPLE_RATE)
                        audio_chunks.append(room_tone)
                        activity_chunks.append(np.zeros(len(room_tone), dtype=bool))

            elif segment["type"] == "pause":
                room_tone = generate_room_tone(segment["duration_sec"], sr=SAMPLE_RATE)
                audio_chunks.append(room_tone)
                activity_chunks.append(np.zeros(len(room_tone), dtype=bool))

            elif segment["type"] == "breath":
                # Chatterbox generates naturalistic breathing via exaggeration;
                # replace explicit breath markers with short room-tone pauses
                # instead of full-volume breath WAVs that sound artificial.
                room_tone = generate_room_tone(0.6, sr=SAMPLE_RATE)
                audio_chunks.append(room_tone)
                activity_chunks.append(np.zeros(len(room_tone), dtype=bool))

            if progress_cb is not None:
                progress_cb(idx + 1, total)

        if not audio_chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

        voice_audio = np.concatenate(audio_chunks).astype(np.float32)
        voice_activity = np.concatenate(activity_chunks)

        # Spectral gating on assembled audio for stable noise profile
        voice_audio = reduce_synthesis_noise(voice_audio, sr=SAMPLE_RATE)

        # Pitch humanization + formant warmth via pyworld.
        # Adds micro-pitch drift, subtle vibrato, random jitter, and 3%
        # formant shift — transforms flat Chatterbox output into perceived
        # natural expressiveness matching Headspace/Calm quality.
        voice_audio = humanize_voice(voice_audio, sr=SAMPLE_RATE)

        return voice_audio, voice_activity

    def get_available_voices(self) -> list[dict]:
        """Return available reference voices."""
        voices = [
            {
                "id": "default",
                "name": "Default (Chatterbox)",
                "description": "Built-in Chatterbox voice, no reference needed.",
            }
        ]

        # Scan reference audio directory for available voice options
        if _REF_AUDIO_DIR.exists():
            for wav_file in sorted(_REF_AUDIO_DIR.glob("*.wav")):
                slug = wav_file.stem
                voices.append({
                    "id": slug,
                    "name": slug.replace("_", " ").title(),
                    "description": f"Reference: {wav_file.name}",
                })

        return voices
