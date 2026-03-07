"""Kokoro TTS engine — wraps Kokoro-82M for meditation narration."""

import gc
import logging
import random
import re

import numpy as np

logger = logging.getLogger(__name__)

from core.speech_engine import SAMPLE_RATE, SpeechEngine
from core.voice_manager import BRITISH_VOICES, is_british_voice

# Inter-sentence pauses for natural meditation pacing
INTER_SENTENCE_PAUSE_SEC = 0.8   # Pause between regular sentences
ELLIPSIS_PAUSE_SEC = 1.2         # Longer pause after trailing "..."

# Kokoro produces poor output for very short utterances (<~20 tokens).
# Sentences shorter than this word count are merged with adjacent ones.
MIN_SENTENCE_WORDS = 4

# Token-aware chunking parameters
# Kokoro's sweet spot is 100-200 tokens; above ~510 it starts rushing.
MAX_CHUNK_TOKENS = 200
MIN_CHUNK_TOKENS = 15
HARD_CEILING_TOKENS = 400

# Crossfade between consecutive speech chunks to eliminate clicks/pops
CROSSFADE_SAMPLES = int(0.075 * SAMPLE_RATE)  # 75ms at 24kHz = 1800 samples

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


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word for English."""
    return int(len(text.split()) * 1.3)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries.

    Handles standard sentence endings (.!?) and ellipsis (...).
    Very short sentences (< MIN_SENTENCE_WORDS words) are merged with the
    next sentence so Kokoro receives enough context for natural prosody.
    """
    # Split after sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    raw = [p for p in parts if p.strip()]

    if len(raw) <= 1:
        return raw

    # Merge very short sentences with the next one to avoid Kokoro's
    # poor-quality output on tiny utterances.
    merged: list[str] = []
    carry = ""
    for s in raw:
        if carry:
            s = carry + " " + s
            carry = ""
        if len(s.split()) < MIN_SENTENCE_WORDS and s is not raw[-1]:
            carry = s
        else:
            merged.append(s)
    if carry:
        if merged:
            merged[-1] = merged[-1] + " " + carry
        else:
            merged.append(carry)

    return merged


def _merge_sentences_to_chunks(sentences: list[str]) -> list[str]:
    """Merge consecutive sentences into chunks targeting 100-200 tokens.

    Never exceeds HARD_CEILING_TOKENS per chunk.  Very short sentences
    (below MIN_CHUNK_TOKENS) are merged with neighbouring text.
    """
    if not sentences:
        return sentences

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        # If a single sentence exceeds the hard ceiling, it must be
        # force-split by word count (rare in meditation scripts).
        if sentence_tokens > HARD_CEILING_TOKENS:
            # Flush current buffer first
            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_tokens = 0

            words = sentence.split()
            max_words = int(HARD_CEILING_TOKENS / 1.3)
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue

        # Would adding this sentence exceed the target?
        if current_tokens + sentence_tokens > MAX_CHUNK_TOKENS and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = [sentence]
            current_tokens = sentence_tokens
        else:
            current_parts.append(sentence)
            current_tokens += sentence_tokens

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _crossfade_chunks(chunks: list[np.ndarray], crossfade_samples: int = CROSSFADE_SAMPLES) -> np.ndarray:
    """Stitch audio chunks with cosine crossfade at boundaries.

    Uses equal-power cosine-squared curves so that energy is preserved
    through the crossfade region.  Chunks that are too short for a full
    crossfade are simply concatenated.
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    result = chunks[0]
    for chunk in chunks[1:]:
        if len(result) < crossfade_samples or len(chunk) < crossfade_samples:
            result = np.concatenate([result, chunk])
            continue

        overlap = crossfade_samples
        # Cosine-squared fade curves (equal-power)
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)).astype(np.float32) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)).astype(np.float32) ** 2

        blended = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
        result = np.concatenate([result[:-overlap], blended, chunk[overlap:]])

    return result


# ---------------------------------------------------------------------------
# Post-generation artifact trimmer
# ---------------------------------------------------------------------------

# Energy threshold for silence detection: RMS below this (linear) is silence.
_SILENCE_THRESHOLD_LINEAR = 10 ** (-45.0 / 20)  # -45 dBFS ≈ 0.00562

# Window for energy-based silence scan (20 ms at 24 kHz = 480 samples)
_SILENCE_WINDOW_SAMPLES = int(0.020 * SAMPLE_RATE)

# Minimum tail to preserve after trimming so crossfade always has audio.
_MIN_TAIL_SAMPLES = int(0.020 * SAMPLE_RATE)

# Spectral-flatness threshold: values > this on a long tail indicate
# tonal loops / noise rather than natural speech trailing off.
_FLATNESS_THRESHOLD = 0.85

# Only run flatness check if the suspected tail is longer than this.
_FLATNESS_MIN_TAIL_SAMPLES = int(0.200 * SAMPLE_RATE)  # 200 ms

# FFT size used for spectral flatness computation.
_FLATNESS_FFT_SIZE = 2048


def _spectral_flatness(frame: np.ndarray) -> float:
    """Wiener entropy (spectral flatness) of a short audio frame.

    Returns a value in [0, 1]:
      ~0 = tonal / speech-like
      ~1 = noise-like / flat spectrum (possible repetition artifact)
    """
    win = np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(frame * win, n=_FLATNESS_FFT_SIZE))
    # Avoid log(0)
    spectrum = np.where(spectrum < 1e-10, 1e-10, spectrum)
    log_mean = np.mean(np.log(spectrum))
    arithmetic_mean = np.mean(spectrum)
    if arithmetic_mean < 1e-10:
        return 0.0
    return float(np.exp(log_mean) / arithmetic_mean)


def trim_tts_artifacts(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Strip trailing silence and repetition-loop artifacts from a TTS chunk.

    Conservative by design: only modifies audio when conditions are clearly
    anomalous.  Normal narration output is returned unchanged.

    Steps:
      1. Hard-clip guard — clamp any out-of-range samples.
      2. Trailing silence trim — strip near-zero RMS windows from the tail,
         preserving a minimum 20 ms tail for crossfades.
      3. Spectral flatness check — if the trimmed tail (≥200 ms) has very
         high spectral flatness (noise-like), trim that region too.

    Args:
        audio: 1-D float32 array from Kokoro synthesis.
        sr:    Sample rate (default 24000 Hz).

    Returns:
        Trimmed 1-D float32 array.  May be shorter than input.
    """
    if len(audio) == 0:
        return audio

    # 1. Hard-clip guard
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)

    original_len = len(audio)

    # 2. Trailing silence trim (scan RMS windows from the end)
    keep_end = len(audio)  # index of last sample to keep (exclusive)
    win = _SILENCE_WINDOW_SAMPLES

    while keep_end - win >= _MIN_TAIL_SAMPLES:
        window = audio[keep_end - win : keep_end]
        rms = float(np.sqrt(np.mean(window ** 2)))
        if rms < _SILENCE_THRESHOLD_LINEAR:
            keep_end -= win
        else:
            break

    # Always keep at least _MIN_TAIL_SAMPLES from the end
    keep_end = max(keep_end, _MIN_TAIL_SAMPLES)

    silence_trimmed = original_len - keep_end
    if silence_trimmed > 0:
        audio = audio[:keep_end]
        logger.debug(
            "[TrimArtifacts] trimmed %dms trailing silence",
            int(silence_trimmed / sr * 1000),
        )

    # 3. Spectral flatness check on trailing region
    tail_samples = min(_FLATNESS_MIN_TAIL_SAMPLES * 4, len(audio))
    if tail_samples >= _FLATNESS_MIN_TAIL_SAMPLES:
        tail = audio[-tail_samples:]
        flatness = _spectral_flatness(tail)
        if flatness > _FLATNESS_THRESHOLD:
            # Trim the flat-spectrum tail; walk backwards in segments
            # until we find a genuinely speech-like region.
            trim_end = len(audio) - tail_samples
            trim_end = max(trim_end, _MIN_TAIL_SAMPLES)
            loop_trimmed = len(audio) - trim_end
            audio = audio[:trim_end]
            logger.debug(
                "[TrimArtifacts] trimmed %dms flat-spectrum repetition tail "
                "(flatness=%.3f)",
                int(loop_trimmed / sr * 1000),
                flatness,
            )

    return audio


class KokoroEngine(SpeechEngine):
    """Wraps Kokoro-TTS to synthesize speech from parsed script segments."""

    def __init__(self):
        self.pipeline = None
        self.pipeline_en_gb = None  # Lazy-loaded British English pipeline

    def load_model(self):
        """Load the Kokoro TTS pipeline with transformer-based G2P for
        higher-quality phonemization."""
        import warnings

        from kokoro import KPipeline

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*dropout.*")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
            
            # Force CPU on Apple Silicon to prevent a known bus error during MPS deallocation
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
                trf=True,   # transformer G2P — better phonemization quality
                device=device,
            )
            # HARD OVERRIDE: KPipeline initialization sometimes ignores device=cpu
            # and auto-selects cuda/mps based on availability inside KModel.
            if hasattr(self.pipeline, "model") and self.pipeline.model is not None:
                self.pipeline.model.to(device)


    def _get_pipeline(self, voice):
        """Return the correct pipeline based on voice language prefix.

        British voices (bf_*, bm_*) require lang_code='b'.
        Blended tensors default to the American pipeline.
        """
        if isinstance(voice, str) and voice.split("_")[0] in ("bf", "bm"):
            if self.pipeline_en_gb is None:
                import warnings

                from kokoro import KPipeline

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message=".*dropout.*")
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
                    # Force CPU on Apple Silicon to prevent MPS driver deallocation fault
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
        # Carefully delete the pipeline objects to trigger GC
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
        speed: float = 0.7,
        progress_cb=None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize all script segments into a single audio track.

        Each speech segment is split into individual sentences, grouped into
        token-aware chunks (100-200 tokens), and crossfaded together for
        smooth long-form audio.  Explicit [pause:Xs] markers produce clean
        silence between speech blocks.

        Args:
            segments: Parsed script segments from script_parser.parse_script().
            voice: Kokoro voice name, preset name, comma-separated blend,
                   or a pre-computed voice tensor.
            speed: Speaking speed (0.5–1.0, lower = slower).
            progress_cb: Called with (current_index, total_segments) after each segment.
            seed: Optional deterministic seed for reproducible output.

        Returns:
            voice_audio: float32 numpy array, mono at 24000 Hz.
            voice_activity: bool numpy array, True where voice is speaking.
        """
        if self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")

        # Set deterministic seed for session consistency
        if seed is not None:
            import torch

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Resolve voice specification (preset / blend / single ID)
        from core.voice_manager import get_voice

        resolved_voice = get_voice(voice) if isinstance(voice, str) else voice

        # Clamp speed to avoid artifacts below 0.50 (Kokoro becomes very distorted below this)
        speed = max(speed, 0.50)

        audio_chunks: list[np.ndarray] = []
        activity_chunks: list[np.ndarray] = []
        total = len(segments)

        for idx, segment in enumerate(segments):
            if segment["type"] == "speech":
                # Split into sentences then merge into token-aware chunks
                sentences = _split_into_sentences(segment["text"])
                if not sentences:
                    sentences = [segment["text"]]

                chunks = _merge_sentences_to_chunks(sentences)

                # Synthesize each chunk and collect speech audio
                speech_parts_for_crossfade: list[np.ndarray] = []

                for chunk_text in chunks:
                    # Select correct pipeline for voice type
                    pipe = self._get_pipeline(
                        voice if isinstance(voice, str) else "af_heart"
                    )
                    gen = pipe(chunk_text, voice=resolved_voice, speed=speed)
                    chunk_audio_parts = []
                    for _gs, _ps, audio in gen:
                        if audio is not None:
                            chunk_audio_parts.append(audio)

                    if chunk_audio_parts:
                        raw_chunk = np.concatenate(chunk_audio_parts)
                        cleaned_chunk = trim_tts_artifacts(raw_chunk, sr=SAMPLE_RATE)
                        
                        # Apply a 10ms fade-in to the first chunk of the segment
                        # to eliminate digital clicks/transients from Kokoro's inference start.
                        if not speech_parts_for_crossfade:
                            fade_samples = int(0.010 * SAMPLE_RATE)
                            if len(cleaned_chunk) > fade_samples:
                                fade_curve = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                                cleaned_chunk[:fade_samples] *= fade_curve

                        speech_parts_for_crossfade.append(cleaned_chunk)

                # Crossfade all speech chunks within this segment
                if speech_parts_for_crossfade:
                    speech_audio = _crossfade_chunks(speech_parts_for_crossfade)
                    
                    # Phase 2 Fix: Apply a strict 30ms fade-in and 50ms fade-out to the 
                    # VERY EDGES of the final assembled segment. This guarantees a mathematically
                    # perfect zero-crossing right before and after any [pause] block, eliminating clicks.
                    fade_in_samples = int(0.030 * SAMPLE_RATE)
                    fade_out_samples = int(0.050 * SAMPLE_RATE)
                    
                    if len(speech_audio) > fade_in_samples + fade_out_samples:
                        f_in = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
                        f_out = np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
                        speech_audio[:fade_in_samples] *= f_in
                        speech_audio[-fade_out_samples:] *= f_out
                        
                else:
                    speech_audio = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)

                audio_chunks.append(speech_audio)
                activity_chunks.append(np.ones(len(speech_audio), dtype=bool))

                # Add inter-sentence trailing pause after the speech segment
                # (only if there's more content coming)
                if idx < total - 1:
                    text_end = segment["text"].rstrip()
                    pause_sec = (
                        ELLIPSIS_PAUSE_SEC
                        if text_end.endswith(("...", "\u2026"))
                        else INTER_SENTENCE_PAUSE_SEC
                    )
                    pause_samples = int(pause_sec * SAMPLE_RATE)
                    audio_chunks.append(np.zeros(pause_samples, dtype=np.float32))
                    activity_chunks.append(np.zeros(pause_samples, dtype=bool))

            elif segment["type"] == "pause":
                num_samples = int(segment["duration_sec"] * SAMPLE_RATE)
                audio_chunks.append(np.zeros(num_samples, dtype=np.float32))
                activity_chunks.append(np.zeros(num_samples, dtype=bool))

            if progress_cb is not None:
                progress_cb(idx + 1, total)

        if not audio_chunks:
            empty = np.zeros(0, dtype=np.float32)
            return empty, np.zeros(0, dtype=bool)

        voice_audio = np.concatenate(audio_chunks).astype(np.float32)
        voice_activity = np.concatenate(activity_chunks)

        return voice_audio, voice_activity

    def get_available_voices(self) -> list[dict]:
        """Return the list of built-in Kokoro voices."""
        return [{"id": v, "name": v, "description": ""} for v in VOICES]
