"""Parler TTS engine for MoodScape — description-controlled speech synthesis.

Target hardware: Apple Silicon M1 Max (36 GB unified memory, 24-core GPU)
Parler TTS native sample rate: 44100 Hz -> resampled to 24000 Hz for pipeline
"""

import gc
import os
import re

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torchaudio
from transformers import set_seed

from core.speech_engine import SAMPLE_RATE, SpeechEngine

# Voice identity seed — locked globally so every chunk uses the same
# starting point in the latent speaker space, preventing voice drift.
VOICE_IDENTITY_SEED = 42

# Inter-sentence pauses (matching Kokoro behavior for consistency)
INTER_SENTENCE_PAUSE_SEC = 0.8
ELLIPSIS_PAUSE_SEC = 1.2
MIN_SENTENCE_WORDS = 6  # Parler needs more context than Kokoro for good prosody

# ── Named-Speaker Voice Presets (Identity Lock Pattern) ───────────────────
# Each preset anchors to a specific speaker name the model was trained on.
# This prevents voice drift across sentence chunks.  The description is
# deliberately kept short and identical for every chunk ("frozen").
# Template: "[Name]'s voice is [trait]. [Name] speaks [pacing]. Clear,
#            intimate recording, no background noise."

VOICE_PRESETS = [
    (
        "Laura — warm, intimate, sleep",
        "Laura's voice is very warm, compassionate, and mature. "
        "Laura speaks very slowly with a clear, intimate recording "
        "and no background noise.",
    ),
    (
        "Jenna — clear, soothing, stable",
        "Jenna's voice is clear, soothing, and soft. "
        "Jenna speaks very slowly with a clear, intimate recording "
        "and no background noise.",
    ),
    (
        "Jon — deep, grounding, body scan",
        "Jon's voice is deep, resonant, and calm. "
        "Jon speaks very slowly with a clear, intimate recording "
        "and no background noise.",
    ),
    (
        "Lea — gentle, melodic, affirmations",
        "Lea's voice is gentle and melodic with a naturally slower cadence. "
        "Lea speaks very slowly with a clear, intimate recording "
        "and no background noise.",
    ),
    (
        "Gary — wise, authoritative, safe",
        "Gary's voice is authoritative yet kind and warm. "
        "Gary speaks very slowly with a clear, intimate recording "
        "and no background noise.",
    ),
    (
        "Mike — steady, neutral, consistent",
        "Mike's voice is steady and neutral with even intonation. "
        "Mike speaks very slowly with a clear, intimate recording "
        "and no background noise.",
    ),
    (
        "Karen — breathy, whisper, ASMR",
        "Karen's voice is soft-spoken and breathy with a gentle whisper quality. "
        "Karen speaks very slowly with a clear, intimate recording "
        "and no background noise.",
    ),
    (
        "Custom Description",
        "",  # User fills in their own description via the UI textbox
    ),
]


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries.

    Handles standard endings (.!?) and ellipsis (...).
    Short sentences are merged for better Parler prosody.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    raw = [p for p in parts if p.strip()]

    if len(raw) <= 1:
        return raw

    merged: list[str] = []
    carry = ""
    for i, s in enumerate(raw):
        if carry:
            s = carry + " " + s
            carry = ""
        if len(s.split()) < MIN_SENTENCE_WORDS and i < len(raw) - 1:
            carry = s
        else:
            merged.append(s)
    if carry:
        if merged:
            merged[-1] = merged[-1] + " " + carry
        else:
            merged.append(carry)

    return merged


def _adjust_description_for_speed(description: str, speed: float) -> str:
    """Modify the voice description to reflect the requested speed.

    Parler TTS doesn't have a numeric speed parameter like Kokoro.
    Instead, we inject pacing keywords into the description.
    """
    # Remove any existing speed-related phrases to avoid conflicts
    speed_phrases = [
        "very slow",
        "slow",
        "moderate speed",
        "slightly slow",
        "fast",
        "slightly fast",
        "very fast",
        "normal speed",
    ]
    cleaned = description
    for phrase in speed_phrases:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)

    # Map numeric speed to descriptive pacing
    if speed <= 0.65:
        pace = "very slow and deliberate pacing with long pauses"
    elif speed <= 0.75:
        pace = "slow, measured pacing with gentle pauses"
    elif speed <= 0.85:
        pace = "slightly slow and calm pacing"
    elif speed <= 0.95:
        pace = "moderate, steady pacing"
    else:
        pace = "natural, conversational pacing"

    return f"{cleaned.rstrip('. ')}. Speaking with {pace}."


class ParlerTTSEngine(SpeechEngine):
    """Wraps Parler TTS Large v1 for meditation audio generation.

    Uses natural-language descriptions to control voice characteristics.
    Implements the same interface as KokoroEngine for seamless integration
    with the MoodScape pipeline.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = None
        self._native_sr = None

    def load_model(self, preferred: str = "parler-tts/parler-tts-large-v1"):
        """Load Parler TTS model with MPS acceleration.

        Candidate order:
          1. Large v1 with refs/pr/9 revision (known noise-generation fix)
          2. Large v1 default HEAD
          3. Mini v1 (memory fallback)
        """
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        # Device selection — prefer MPS on Apple Silicon
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # MPS has incomplete bfloat16 support — use float32 to avoid
        # silent tensor corruption that produces static/noise output.
        # Use bfloat16 only on CUDA where it's fully supported.
        if self.device == "mps":
            torch_dtype = torch.float32
        elif torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Each entry is (model_name, extra_kwargs_for_from_pretrained)
        candidates: list[tuple[str, dict]] = [
            # PR #9 fixes noise generation on certain GPU backends
            (preferred, {"revision": "refs/pr/9"}),
            (preferred, {}),
            ("parler-tts/parler-tts-mini-v1", {}),  # Memory fallback
        ]

        for model_name, extra_kwargs in candidates:
            rev_label = extra_kwargs.get("revision", "HEAD")
            try:
                self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    **extra_kwargs,
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, **extra_kwargs
                )
                self.model_name = model_name
                self._native_sr = self.model.config.sampling_rate
                print(
                    f"[ParlerTTS] Loaded {model_name} (rev={rev_label}) "
                    f"on {self.device} (native SR: {self._native_sr} Hz)"
                )
                return
            except Exception as e:
                print(f"[ParlerTTS] Failed to load {model_name} (rev={rev_label}): {e}")
                self.model = None
                self.tokenizer = None
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        raise RuntimeError(
            "Could not load any Parler TTS model. "
            "Ensure you have internet access for first download, "
            "and sufficient memory (Large needs ~6GB, Mini ~3GB)."
        )

    def unload_model(self):
        """Release model and free GPU/unified memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self._native_sr = None
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, AttributeError):
            pass

    def _generate_speech_chunk(self, text: str, description: str) -> np.ndarray:
        """Generate speech for a single text chunk.

        Returns float32 numpy array at 24000 Hz (resampled from native SR).
        The random seed is reset before every call to lock the speaker
        identity and prevent voice drift between sentences.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # ── Identity lock: reset seed before every chunk ──────────────
        # This ensures the model starts from the same point in the latent
        # speaker space for every sentence, preventing timbre shifts.
        set_seed(VOICE_IDENTITY_SEED)

        # Tokenize description and prompt — must include attention_mask
        # to avoid garbage output (pad token == eos token in this model).
        desc_inputs = self.tokenizer(
            description, return_tensors="pt", padding=True
        )
        prompt_inputs = self.tokenizer(
            text, return_tensors="pt", padding=True
        )

        input_ids = desc_inputs.input_ids.to(self.device)
        attention_mask = desc_inputs.attention_mask.to(self.device)
        prompt_input_ids = prompt_inputs.input_ids.to(self.device)
        prompt_attention_mask = prompt_inputs.attention_mask.to(self.device)

        try:
            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    do_sample=True,
                    temperature=0.8,
                )
        except RuntimeError as e:
            if "MPS" in str(e) or "placeholder" in str(e).lower():
                print(f"[ParlerTTS] MPS error, falling back to CPU: {e}")
                self.model.cpu()
                input_ids = input_ids.cpu()
                attention_mask = attention_mask.cpu()
                prompt_input_ids = prompt_input_ids.cpu()
                prompt_attention_mask = prompt_attention_mask.cpu()
                with torch.no_grad():
                    generation = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        prompt_input_ids=prompt_input_ids,
                        prompt_attention_mask=prompt_attention_mask,
                        do_sample=True,
                        temperature=0.8,
                    )
                # Move back to MPS for next chunk
                self.model.to(self.device)
            else:
                raise

        # Move to CPU immediately for safety
        generation = generation.cpu()

        # ── Vocal sanity check: NaN / Inf → automatic CPU fallback ────
        if torch.isnan(generation).any() or torch.isinf(generation).any():
            print(
                "[ParlerTTS] WARNING: Model produced NaN/Inf on "
                f"{self.device}. Re-generating on CPU..."
            )
            # Move everything to CPU for a guaranteed clean signal
            self.model.cpu()
            input_ids = input_ids.cpu()
            attention_mask = attention_mask.cpu()
            prompt_input_ids = prompt_input_ids.cpu()
            prompt_attention_mask = prompt_attention_mask.cpu()

            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    do_sample=True,
                    temperature=0.8,
                )

            # Restore model to original device for subsequent chunks
            self.model.to(self.device)

            if torch.isnan(generation).any() or torch.isinf(generation).any():
                raise ValueError(
                    "Model produced NaN/Inf even on CPU. "
                    "The model weights may be corrupted — try re-downloading."
                )
            print("[ParlerTTS] CPU fallback succeeded.")

        audio_native = generation.float().numpy().squeeze()

        # Secondary numpy-level NaN guard
        if np.isnan(audio_native).any():
            raise ValueError("Model produced NaN in numpy array after all fallbacks.")

        # Hard normalization to prevent 'static' clipping
        max_val = np.abs(audio_native).max()
        if max_val > 1.0:
            audio_native = audio_native / max_val

        # Handle edge case of empty generation
        if audio_native.size == 0:
            return np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)

        # Resample native SR -> 24000 Hz
        if self._native_sr != SAMPLE_RATE:
            audio_tensor = torch.tensor(audio_native).unsqueeze(0)
            audio_24k = torchaudio.functional.resample(
                audio_tensor,
                orig_freq=self._native_sr,
                new_freq=SAMPLE_RATE,
            ).squeeze(0).numpy()
        else:
            audio_24k = audio_native

        return audio_24k.astype(np.float32)

    def synthesize(
        self,
        segments: list[dict],
        voice: str = "",
        speed: float = 0.85,
        progress_cb=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize all script segments into a single audio track.

        Args:
            segments: Parsed script segments from script_parser.py.
                      Each is {"type": "speech", "text": "..."} or
                      {"type": "pause", "duration_sec": float}.
            voice: Either a preset name (matched against VOICE_PRESETS)
                   or a raw description string. If empty, uses the
                   first preset (Serene Female).
            speed: Speaking speed 0.5-1.0. Translated into description
                   keywords since Parler has no numeric speed control.
            progress_cb: Called with (current_index, total_segments).

        Returns:
            voice_audio:    float32 mono array at 24,000 Hz.
            voice_activity: bool array, True where voice is speaking.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Resolve voice description
        description = self._resolve_voice_description(voice)

        # Inject speed/pacing into description
        description = _adjust_description_for_speed(description, speed)

        audio_chunks: list[np.ndarray] = []
        activity_chunks: list[np.ndarray] = []
        total = len(segments)

        for idx, segment in enumerate(segments):
            if segment["type"] == "speech":
                sentences = _split_into_sentences(segment["text"])
                if not sentences:
                    sentences = [segment["text"]]

                for j, sentence in enumerate(sentences):
                    speech_audio = self._generate_speech_chunk(sentence, description)

                    audio_chunks.append(speech_audio)
                    activity_chunks.append(np.ones(len(speech_audio), dtype=bool))

                    # Add inter-sentence pause (not after last sentence)
                    if j < len(sentences) - 1:
                        pause_sec = (
                            ELLIPSIS_PAUSE_SEC
                            if sentence.rstrip().endswith(("...", "\u2026"))
                            else INTER_SENTENCE_PAUSE_SEC
                        )
                        pause_samples = int(pause_sec * SAMPLE_RATE)
                        audio_chunks.append(
                            np.zeros(pause_samples, dtype=np.float32)
                        )
                        activity_chunks.append(
                            np.zeros(pause_samples, dtype=bool)
                        )

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

    def _resolve_voice_description(self, voice: str) -> str:
        """Map a voice preset name to its full description string.

        If voice matches a preset label -> return that preset's description.
        If voice is a raw description string -> return it directly.
        If empty -> return the default meditation preset.
        """
        if not voice or not voice.strip():
            return VOICE_PRESETS[0][1]

        for label, desc in VOICE_PRESETS:
            if voice == label or voice == desc:
                if desc:
                    return desc
                else:
                    return voice

        # Assume it's a raw description string
        return voice

    def get_available_voices(self) -> list[dict]:
        """Return available voice presets as a list of dicts."""
        return [
            {"id": label, "name": label, "description": desc}
            for label, desc in VOICE_PRESETS
            if desc  # Exclude the empty "Custom Description" entry
        ]
