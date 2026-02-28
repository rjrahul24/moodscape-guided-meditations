"""Kokoro TTS wrapper for generating meditation narration audio."""

import gc

import numpy as np


SAMPLE_RATE = 24000

VOICES = [
    "af_heart",    # Warm, calm (default for meditation)
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "af_nova",
    "am_adam",
    "am_michael",
]


class TTSEngine:
    """Wraps Kokoro-TTS to synthesize speech from parsed script segments."""

    def __init__(self):
        self.pipeline = None

    def load_model(self):
        """Load the Kokoro TTS pipeline."""
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code="a")  # American English

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
        speed: float = 0.85,
        progress_cb=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize all script segments into a single audio track.

        Args:
            segments: Parsed script segments from script_parser.parse_script().
            voice: Kokoro voice name.
            speed: Speaking speed (0.5–1.0, lower = slower).
            progress_cb: Called with (current_index, total_segments) after each segment.

        Returns:
            voice_audio: float32 numpy array, mono at 24000 Hz.
            voice_activity: bool numpy array, True where voice is speaking.
        """
        if self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")

        audio_chunks: list[np.ndarray] = []
        activity_chunks: list[np.ndarray] = []
        total = len(segments)

        for idx, segment in enumerate(segments):
            if segment["type"] == "speech":
                # Generate speech via Kokoro
                gen = self.pipeline(segment["text"], voice=voice, speed=speed)
                speech_parts = []
                for _gs, _ps, audio in gen:
                    if audio is not None:
                        speech_parts.append(audio)

                if speech_parts:
                    speech_audio = np.concatenate(speech_parts)
                else:
                    # Fallback: tiny silence if TTS returned nothing
                    speech_audio = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)

                audio_chunks.append(speech_audio)
                activity_chunks.append(np.ones(len(speech_audio), dtype=bool))

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
