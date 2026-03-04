"""Kokoro TTS engine — wraps Kokoro-82M for meditation narration."""

import gc
import re

import numpy as np

from core.speech_engine import SAMPLE_RATE, SpeechEngine

# Inter-sentence pauses for natural meditation pacing
INTER_SENTENCE_PAUSE_SEC = 0.8   # Pause between regular sentences
ELLIPSIS_PAUSE_SEC = 1.2         # Longer pause after trailing "..."

# Kokoro produces poor output for very short utterances (<~20 tokens).
# Sentences shorter than this word count are merged with adjacent ones.
MIN_SENTENCE_WORDS = 4

VOICES = [
    "af_heart",    # Grade A — warm, calm (default for meditation)
    "af_bella",    # Grade A- — warm, friendly
    "af_nicole",   # Grade B- — calm, smooth, ASMR-like
    "af_sarah",    # Grade C+
    "af_sky",      # Grade C-
    "af_nova",     # Grade C — intimate, ASMR-like
    "am_adam",     # Grade F+
    "am_michael",  # Grade C+
]


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


class KokoroEngine(SpeechEngine):
    """Wraps Kokoro-TTS to synthesize speech from parsed script segments."""

    def __init__(self):
        self.pipeline = None

    def load_model(self):
        """Load the Kokoro TTS pipeline with transformer-based G2P for
        higher-quality phonemization."""
        import warnings

        from kokoro import KPipeline

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*dropout.*")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
            self.pipeline = KPipeline(
                lang_code="a",
                repo_id="hexgrad/Kokoro-82M",
                trf=True,   # transformer G2P — better phonemization quality
            )

    def unload_model(self):
        """Release model and free GPU memory."""
        del self.pipeline
        self.pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def synthesize(
        self,
        segments: list[dict],
        voice: str = "af_heart",
        speed: float = 0.7,
        progress_cb=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize all script segments into a single audio track.

        Each speech segment is split into individual sentences, with explicit
        inter-sentence pauses added for natural meditation pacing. Very short
        sentences are merged to avoid Kokoro's poor output on tiny utterances.

        Args:
            segments: Parsed script segments from script_parser.parse_script().
            voice: Kokoro voice name, or comma-separated names for blending
                   (e.g. "af_heart,af_nicole").
            speed: Speaking speed (0.5–1.0, lower = slower).
            progress_cb: Called with (current_index, total_segments) after each segment.

        Returns:
            voice_audio: float32 numpy array, mono at 24000 Hz.
            voice_activity: bool numpy array, True where voice is speaking.
        """
        if self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")

        # Clamp speed to avoid artifacts below 0.65
        speed = max(speed, 0.65)

        audio_chunks: list[np.ndarray] = []
        activity_chunks: list[np.ndarray] = []
        total = len(segments)

        for idx, segment in enumerate(segments):
            if segment["type"] == "speech":
                # Split into sentences for explicit pacing control
                sentences = _split_into_sentences(segment["text"])
                if not sentences:
                    sentences = [segment["text"]]

                for j, sentence in enumerate(sentences):
                    # Generate speech for this sentence
                    gen = self.pipeline(sentence, voice=voice, speed=speed)
                    speech_parts = []
                    for _gs, _ps, audio in gen:
                        if audio is not None:
                            speech_parts.append(audio)

                    if speech_parts:
                        speech_audio = np.concatenate(speech_parts)
                    else:
                        speech_audio = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)

                    audio_chunks.append(speech_audio)
                    activity_chunks.append(np.ones(len(speech_audio), dtype=bool))

                    # Add inter-sentence pause (not after the last sentence in a segment)
                    if j < len(sentences) - 1:
                        pause_sec = (
                            ELLIPSIS_PAUSE_SEC
                            if sentence.rstrip().endswith(("...", "\u2026"))
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
