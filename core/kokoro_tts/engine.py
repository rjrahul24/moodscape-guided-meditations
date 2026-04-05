"""Kokoro TTS engine — wraps Kokoro-82M for meditation narration.

Self-contained engine that uses Kokoro-specific preprocessing (token-aware
chunking, text expansion) and postprocessing (artifact trimming, RMS
normalization, crossfading, segment fades).
"""

import gc
import logging
import random

import numpy as np

from core.speech_engine import SAMPLE_RATE, SpeechEngine
from core.kokoro_tts.voice_manager import BRITISH_VOICES, is_british_voice, add_voice_jitter
from core.kokoro_tts.postprocessor import (
    CROSSFADE_SAMPLES,
    process_chunk,
    crossfade_chunks,
    apply_segment_fades,
    reduce_synthesis_noise,
    generate_room_tone,
)
from core.kokoro_tts.preprocessor import annotate_speed, clamp_speed, split_into_sentences

logger = logging.getLogger(__name__)

# Inter-sentence pauses for natural meditation pacing
INTER_SENTENCE_PAUSE_SEC = 0.8
ELLIPSIS_PAUSE_SEC = 1.2

VOICES = [
    "af_heart",    # Grade A — warm, calm (default for meditation)
    "af_bella",    # Grade A- — warm, friendly
    "af_nicole",   # Grade B- — calm, smooth, ASMR-like
    "af_sarah",    # Grade C+
    "af_sky",      # Grade C-
    "af_nova",     # Grade C — intimate, ASMR-like
    "am_adam",     # Grade F+
    "am_michael",  # Grade C+
    # British voices (require lang_code='b' pipeline)
    "bf_emma",     # UK Female — wise
    "bf_lily",     # UK Female — angelic
    "bm_george",   # UK Male — warm
]


class KokoroEngine(SpeechEngine):
    """Wraps Kokoro-TTS to synthesize speech from parsed script segments."""

    def __init__(self):
        self.pipeline = None
        self.pipeline_en_gb = None

    def load_model(self):
        """Load the Kokoro TTS pipeline with transformer-based G2P."""
        import warnings
        from kokoro import KPipeline

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*dropout.*")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")

            # Force CPU on Apple Silicon to prevent MPS deallocation bus errors
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass

            self.pipeline = KPipeline(
                lang_code="a",
                repo_id="hexgrad/Kokoro-82M",
                trf=True,
                device=device,
            )
            if hasattr(self.pipeline, "model") and self.pipeline.model is not None:
                self.pipeline.model.to(device)

    def _get_pipeline(self, voice):
        """Return the correct pipeline based on voice language prefix."""
        if isinstance(voice, str) and voice.split("_")[0] in ("bf", "bm"):
            if self.pipeline_en_gb is None:
                import warnings
                from kokoro import KPipeline

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message=".*dropout.*")
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
                    device = "cpu"
                    try:
                        import torch
                        if torch.cuda.is_available():
                            device = "cuda"
                    except ImportError:
                        pass
                    self.pipeline_en_gb = KPipeline(lang_code="b", device=device)
            return self.pipeline_en_gb
        return self.pipeline

    def unload_model(self):
        """Release model and free GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if self.pipeline_en_gb is not None:
            del self.pipeline_en_gb
            self.pipeline_en_gb = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    def synthesize(
        self,
        segments: list[dict],
        voice: str = "af_heart",
        speed: float = 0.90,
        progress_cb=None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize all script segments into a single audio track.

        Each speech segment is split into individual sentences via
        split_into_sentences(), then each sentence is passed to KPipeline
        with split_pattern='' to prevent any further internal splitting.
        Each yielded audio slice is artifact-trimmed, RMS-normalized, then
        crossfaded together. An inter-sentence pause is appended after each
        sentence for natural meditation pacing.

        Args:
            segments: Parsed script segments (from preprocessor.prepare_segments
                      or script_parser.parse_script).
            voice: Kokoro voice name, preset name, comma-separated blend,
                   or a pre-computed voice tensor.
            speed: Speaking speed (0.5–1.0, lower = slower).
            progress_cb: Called with (current_index, total_segments).
            seed: Optional deterministic seed for reproducible output.

        Returns:
            voice_audio: float32 numpy array, mono at 24000 Hz.
            voice_activity: bool numpy array, True where voice is speaking.
        """
        if self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")

        # Set deterministic seed
        if seed is not None:
            import torch
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Resolve voice specification — always to a tensor so per-sentence
        # jitter can be applied for natural micro-variation
        from core.kokoro_tts.voice_manager import get_voice, load_voice_tensor
        if isinstance(voice, str):
            resolved = get_voice(voice)
            # Single voice IDs return as string — convert to tensor
            if isinstance(resolved, str):
                resolved = load_voice_tensor(resolved)
            resolved_voice = resolved
        else:
            resolved_voice = voice

        # Clamp speed to safe Kokoro range
        speed = clamp_speed(speed)

        audio_chunks: list[np.ndarray] = []
        activity_chunks: list[np.ndarray] = []
        total = len(segments)

        for idx, segment in enumerate(segments):
            if segment["type"] == "speech":
                pipe = self._get_pipeline(
                    voice if isinstance(voice, str) else "af_heart"
                )
                sentences = split_into_sentences(segment["text"])
                if not sentences:
                    sentences = [segment["text"]]

                for s_idx, sentence in enumerate(sentences):
                    sentence_speed = annotate_speed(sentence, speed)
                    sentence_voice = add_voice_jitter(resolved_voice)
                    gen = pipe(
                        sentence,
                        voice=sentence_voice,
                        speed=sentence_speed,
                        split_pattern='',
                    )

                    # Collect and clean each audio slice
                    speech_parts: list[np.ndarray] = []
                    for _gs, _ps, audio in gen:
                        if audio is not None:
                            arr = audio if isinstance(audio, np.ndarray) else audio.numpy()
                            cleaned = process_chunk(arr.astype(np.float32))
                            speech_parts.append(cleaned)

                    if speech_parts:
                        speech_audio = crossfade_chunks(speech_parts)
                        speech_audio = apply_segment_fades(speech_audio)
                    else:
                        speech_audio = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)

                    audio_chunks.append(speech_audio)
                    activity_chunks.append(np.ones(len(speech_audio), dtype=bool))

                    # Pause after every sentence except:
                    # - the final sentence of the final segment, and
                    # - the final sentence of a speech segment immediately followed by an
                    #   explicit pause segment (the explicit pause provides the silence).
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
                from core.breath_sounds import load_breath

                breath_audio = load_breath(segment["subtype"], target_sr=SAMPLE_RATE)
                audio_chunks.append(breath_audio)
                activity_chunks.append(np.zeros(len(breath_audio), dtype=bool))

            if progress_cb is not None:
                progress_cb(idx + 1, total)

        if not audio_chunks:
            empty = np.zeros(0, dtype=np.float32)
            return empty, np.zeros(0, dtype=bool)

        voice_audio = np.concatenate(audio_chunks).astype(np.float32)
        voice_activity = np.concatenate(activity_chunks)

        # Stage 2b: Spectral gating — remove low-level ISTFTNet synthesis hiss.
        # Runs on the full assembled audio so the noise profile estimate is stable.
        voice_audio = reduce_synthesis_noise(voice_audio, sr=SAMPLE_RATE)

        # Stage 2c: Pitch humanization + formant warmth via pyworld.
        # Adds micro-pitch drift (~0.5 Hz / ±6 cents), subtle vibrato (5 Hz / ±3 cents),
        # random jitter (±2 cents), and 3% formant shift — all invisible to the ear
        # but transforming the robotic flatness of TTS into perceived expressiveness.
        from core.kokoro_tts.postprocessor import humanize_voice
        voice_audio = humanize_voice(voice_audio, sr=SAMPLE_RATE)

        return voice_audio, voice_activity

    def get_available_voices(self) -> list[dict]:
        """Return the list of built-in Kokoro voices."""
        return [{"id": v, "name": v, "description": ""} for v in VOICES]
