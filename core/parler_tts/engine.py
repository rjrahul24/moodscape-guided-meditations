"""Parler TTS engine for MoodScape — description-controlled speech synthesis.

Target hardware: Apple Silicon M1 Max (36 GB unified memory, 24-core GPU)
Parler TTS native sample rate: 44100 Hz -> resampled to 24000 Hz for pipeline
"""

import gc
import logging
import os
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Suppress known FutureWarnings from parler_tts / transformers internals that we
# cannot fix without modifying third-party source code:
#  • torch weight_norm: parler_tts's DAC vocoder uses the old API and also calls
#    remove_weight_norm(), which is incompatible with the new parametrizations API.
#    Replacing weight_norm directly breaks remove_weight_norm, so suppress instead.
#  • transformers AttentionMaskConverter: deprecated in v5.x, removed in v5.10.
#  • prompt_attention_mask logger: cosmetic, parler always receives the mask.
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=".*weight_norm.*",
)
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=".*AttentionMaskConverter.*",
)
logging.getLogger("parler_tts").setLevel(logging.ERROR)

# We rely on transformers >=4.46.1 but <4.47.0 to guarantee compatibility
# with Parler TTS Large v1 without internal API breaks.

import numpy as np
import torch
import torchaudio
from transformers import set_seed

from core.speech_engine import SAMPLE_RATE, SpeechEngine
from core.parler_tts.preprocessor import (
    adjust_description_for_speed,
    estimate_max_tokens,
    split_into_sentences,
)
from core.parler_tts.postprocessor import (
    crossfade_activity_chunks,
    crossfade_audio_chunks,
)

# Voice identity seed — locked globally so every chunk uses the same
# starting point in the latent speaker space, preventing voice drift.
VOICE_IDENTITY_SEED = 42

# Inter-sentence pauses (increased for guided meditation spacing)
INTER_SENTENCE_PAUSE_SEC = 2.5
ELLIPSIS_PAUSE_SEC = 5.0

# Audio prefix conditioning: length of the reference snippet (in seconds)
# extracted from the first generated sentence and fed as input_values to
# all subsequent chunks, anchoring voice identity via the audio encoder.
_VOICE_REF_SECONDS = 5.0

# Lower temperature = more deterministic = less voice drift between chunks.
_TEMPERATURE = 0.65
_GUIDANCE_SCALE = 1.0  # Lower CFG to prevent harsh static noise

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
        "wonderful speech quality, very clean audio, studio recording, and almost no background noise.",
    ),
    (
        "Jenna — clear, soothing, stable",
        "Jenna's voice is clear, soothing, and soft. "
        "Jenna speaks very slowly with a clear, intimate recording "
        "wonderful speech quality, very clean audio, studio recording, and almost no background noise.",
    ),
    (
        "Jon — deep, grounding, body scan",
        "Jon's voice is deep, resonant, and calm. "
        "Jon speaks very slowly with a clear, intimate recording "
        "wonderful speech quality, very clean audio, studio recording, and almost no background noise.",
    ),
    (
        "Lea — gentle, melodic, affirmations",
        "Lea's voice is gentle and melodic with a naturally slower cadence. "
        "Lea speaks very slowly with a clear, intimate recording "
        "wonderful speech quality, very clean audio, studio recording, and almost no background noise.",
    ),
    (
        "Gary — wise, authoritative, safe",
        "Gary's voice is authoritative yet kind and warm. "
        "Gary speaks very slowly with a clear, intimate recording "
        "wonderful speech quality, very clean audio, studio recording, and almost no background noise.",
    ),
    (
        "Mike — steady, neutral, consistent",
        "Mike's voice is steady and neutral with even intonation. "
        "Mike speaks very slowly with a clear, intimate recording "
        "wonderful speech quality, very clean audio, studio recording, and almost no background noise.",
    ),
    (
        "Karen — breathy, whisper, ASMR",
        "Karen's voice is soft-spoken and breathy with a gentle whisper quality. "
        "Karen speaks very slowly with a clear, intimate recording "
        "wonderful speech quality, very clean audio, studio recording, and almost no background noise.",
    ),
    (
        "Custom Description",
        "",  # User fills in their own description via the UI textbox
    ),
]


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
        
        # Inject monkey patches to fix compatibility with transformers >= 4.50
        try:
            from parler_tts.configuration_parler_tts import ParlerTTSConfig
            from parler_tts.modeling_parler_tts import ParlerTTSForConditionalGeneration
            from transformers import GenerationMixin
            try:
                from transformers.cache_utils import EncoderDecoderCache, DynamicCache
            except ImportError:
                EncoderDecoderCache, DynamicCache = None, None

            # 1. Fix ParlerTTSConfig __init__ being called with empty kwargs
            ParlerTTSConfig.has_no_defaults_at_init = True
            
            # 2. Add GenerationMixin to model if missing
            if GenerationMixin not in ParlerTTSForConditionalGeneration.__bases__:
                ParlerTTSForConditionalGeneration.__bases__ = (
                    GenerationMixin,
                    *ParlerTTSForConditionalGeneration.__bases__
                )
            
            # 3. Fix _get_initial_cache_position signature change
            def _patched_get_initial_cache_position(self, *args, **kwargs):
                # New signature: (cur_len, device, model_kwargs) [len(args)==2 since self is separate]
                # Old signature: (input_ids, model_kwargs) [len(args)==2 since self is separate]
                # Transformers v4.50+ calls with 3 positional args (excluding self)
                if len(args) >= 3:
                    cur_len, device, model_kwargs = args[:3]
                else:
                    # Fallback to old behavior if possible
                    return original_get_cache(self, *args, **kwargs)

                cache_position = torch.arange(cur_len, dtype=torch.int64, device=device)
                past_length = 0
                if model_kwargs.get("past_key_values") is not None:
                    cache = model_kwargs["past_key_values"]
                    if hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                        past_length = cache.get_seq_length()
                    elif isinstance(cache, (list, tuple)):
                        past_length = cache[0][0].shape[2]
                    cache_position = cache_position[past_length:]
                
                model_kwargs["cache_position"] = cache_position
                return model_kwargs

            original_get_cache = ParlerTTSForConditionalGeneration._get_initial_cache_position
            if not hasattr(original_get_cache, "_is_patched"):
                _patched_get_initial_cache_position._is_patched = True
                ParlerTTSForConditionalGeneration._get_initial_cache_position = _patched_get_initial_cache_position

            # 4. Fix prepare_inputs_for_generation negative dimension error
            # This happens because the original code only checks for EncoderDecoderCache
            # and fails on DynamicCache which is now the default.
            original_prepare = ParlerTTSForConditionalGeneration.prepare_inputs_for_generation
            def _patched_prepare(self, *args, **kwargs):
                # We need to ensure that the logic for decoder_attention_mask creation
                # handles any cache type with length 0.
                past = kwargs.get("past_key_values", None) if "past_key_values" in kwargs else (args[1] if len(args) > 1 else None)
                if past is not None:
                    past_len = 0
                    if hasattr(past, "get_seq_length"):
                        past_len = past.get_seq_length()
                    
                    if past_len == 0:
                        # Force it to act like None for the mask creation logic in the original function
                        # by overriding it in kwargs if it's there, or just letting the func handle it if we can.
                        # Actually, we can just temporarily set past_key_values to None in kwargs.
                        if "past_key_values" in kwargs:
                           old_past = kwargs["past_key_values"]
                           kwargs["past_key_values"] = None
                           res = original_prepare(self, *args, **kwargs)
                           kwargs["past_key_values"] = old_past
                           return res
                return original_prepare(self, *args, **kwargs)

            if not hasattr(original_prepare, "_is_patched"):
                _patched_prepare._is_patched = True
                ParlerTTSForConditionalGeneration.prepare_inputs_for_generation = _patched_prepare

        except ImportError:
            pass

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

    def _generate_speech_chunk(
        self,
        text: str,
        description: str,
        voice_ref: "torch.Tensor | None" = None,
    ) -> np.ndarray:
        """Generate speech for a single text chunk.

        Returns float32 numpy array at *native* sample rate (NOT resampled).
        Resampling to 24 kHz is done by the caller so that the raw native-SR
        tensor can be reused as a voice reference for subsequent chunks.

        Args:
            voice_ref: Optional tensor of shape (1, samples) at native SR.
                       When provided, fed as ``input_values`` to condition
                       the audio decoder on a reference speaker identity,
                       preventing voice drift across chunks.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # ── Identity lock: reset seed before every chunk ──────────────
        set_seed(VOICE_IDENTITY_SEED)

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

        max_tokens = estimate_max_tokens(text)

        # Build generate kwargs — optionally include audio conditioning
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            do_sample=True,
            temperature=_TEMPERATURE,
            guidance_scale=_GUIDANCE_SCALE,
            max_new_tokens=max_tokens,
        )
        if voice_ref is not None:
            gen_kwargs["input_values"] = voice_ref.to(self.device)

        max_retries = 3
        audio_native = None

        for attempt in range(max_retries):
            try:
                with torch.no_grad():
                    generation = self.model.generate(**gen_kwargs)
            except RuntimeError as e:
                if "MPS" in str(e) or "placeholder" in str(e).lower():
                    print(f"[ParlerTTS] MPS error on attempt {attempt+1}, falling back to CPU: {e}")
                    self.model.cpu()
                    gen_kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in gen_kwargs.items()}
                    with torch.no_grad():
                        generation = self.model.generate(**gen_kwargs)
                    self.model.to(self.device)
                else:
                    raise

            generation = generation.cpu()

            if torch.isnan(generation).any() or torch.isinf(generation).any():
                print(f"[ParlerTTS] WARNING: Model produced NaN/Inf on {self.device}. Re-generating on CPU...")
                self.model.cpu()
                gen_kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in gen_kwargs.items()}
                with torch.no_grad():
                    generation = self.model.generate(**gen_kwargs)
                self.model.to(self.device)

                if torch.isnan(generation).any() or torch.isinf(generation).any():
                    raise ValueError("Model produced NaN/Inf even on CPU.")

            chunk_audio = generation.float().numpy().squeeze()

            if np.isnan(chunk_audio).any():
                raise ValueError("Model produced NaN in numpy array after all fallbacks.")

            max_val = np.abs(chunk_audio).max()
            if max_val > 1.0:
                chunk_audio = chunk_audio / max_val

            if chunk_audio.size == 0:
                audio_native = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
                break

            # RMS validation for silence or static
            rms = np.sqrt(np.mean(np.square(chunk_audio)))
            is_silence = rms < 1e-4
            is_static = rms > 0.95

            if not is_silence and not is_static:
                audio_native = chunk_audio
                break
            
            print(f"[ParlerTTS] Attempt {attempt+1} generated anomaly (RMS={rms:.4f}, silence={is_silence}, static={is_static}). Retrying...")
            
            # Retry strategies
            if attempt == 0 and self.device == "mps":
                print("[ParlerTTS] Strategy 1: Moving audio_encoder (codec) to CPU while keeping transformers on MPS.")
                if hasattr(self.model, "audio_encoder"):
                    self.model.audio_encoder.to("cpu")
                    # transformers code might expect decoder outputs to be on the same device as audio_encoder
                    # This happens automatically in some HF versions, but if not we will catch it in the next loop.
            elif attempt == 1:
                print("[ParlerTTS] Strategy 2: Moving entire model to CPU.")
                self.model.cpu()
                gen_kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in gen_kwargs.items()}

        if audio_native is None:
            print("[ParlerTTS] All retries failed. Returning silence.")
            audio_native = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
            
        # Restore audio_encoder device if we moved it
        if self.device == "mps" and hasattr(self.model, "audio_encoder"):
            try:
                self.model.audio_encoder.to(self.device)
            except Exception:
                pass
            
        return audio_native.astype(np.float32)

    def _generate_speech_batch(
        self,
        texts: list[str],
        description: str,
        max_batch_size: int = 4,
    ) -> list[np.ndarray]:
        """Generate speech for multiple text chunks in batches.

        Processes all chunks under a unified latent state per batch,
        reducing inter-chunk voice drift compared to sequential calls.
        Batches are capped at max_batch_size to stay within memory limits.

        Falls back to sequential _generate_speech_chunk() on any error.

        Returns list of float32 numpy arrays at 24000 Hz, one per text.
        """
        if not texts:
            return []

        all_audio: list[np.ndarray] = []

        # Split into sub-batches to respect memory constraints
        for batch_start in range(0, len(texts), max_batch_size):
            batch_texts = texts[batch_start:batch_start + max_batch_size]

            try:
                audio_arrays = self._generate_batch_internal(batch_texts, description)
                all_audio.extend(audio_arrays)
            except Exception as e:
                print(f"[ParlerTTS] Batch generation failed, falling back to sequential: {e}")
                for text in batch_texts:
                    audio_native = self._generate_speech_chunk(text, description)
                    if self._native_sr != SAMPLE_RATE:
                        audio_native = torchaudio.functional.resample(
                            torch.tensor(audio_native).unsqueeze(0),
                            orig_freq=self._native_sr,
                            new_freq=SAMPLE_RATE,
                            lowpass_filter_width=64,
                            rolloff=0.94,
                            resampling_method="sinc_interp_kaiser",
                        ).squeeze(0).numpy()
                    all_audio.append(audio_native.astype(np.float32))

        return all_audio

    def _generate_batch_internal(
        self,
        texts: list[str],
        description: str,
    ) -> list[np.ndarray]:
        """Internal: generate a single batch of texts.

        Tokenizes all description/text pairs together with padding so the
        model processes them under a single latent state.

        Returns list of float32 numpy arrays at 24000 Hz.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        batch_size = len(texts)

        # Identity lock: set seed once per batch for unified latent state
        set_seed(VOICE_IDENTITY_SEED)

        # Tokenize descriptions (same description repeated for each chunk)
        descriptions_batch = [description] * batch_size
        desc_inputs = self.tokenizer(
            descriptions_batch, return_tensors="pt", padding=True
        )
        prompt_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True
        )

        input_ids = desc_inputs.input_ids.to(self.device)
        attention_mask = desc_inputs.attention_mask.to(self.device)
        prompt_input_ids = prompt_inputs.input_ids.to(self.device)
        prompt_attention_mask = prompt_inputs.attention_mask.to(self.device)

        # Use the longest text in the batch for the token budget
        max_tokens = max(estimate_max_tokens(t) for t in texts)

        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                do_sample=True,
                temperature=_TEMPERATURE,
                guidance_scale=_GUIDANCE_SCALE,
                max_new_tokens=max_tokens,
            )

        # generation is (batch_size, sequence_length) — extract each item
        generation = generation.cpu().float()

        results: list[np.ndarray] = []
        for i in range(batch_size):
            audio_native = generation[i].numpy().squeeze()

            # NaN / Inf guard and RMS validation
            rms = np.sqrt(np.mean(np.square(audio_native))) if audio_native.size > 0 else 0
            is_silence = rms < 1e-4
            is_static = rms > 0.95
            
            if np.isnan(audio_native).any() or not np.isfinite(audio_native).all() or is_silence or is_static:
                print(f"[ParlerTTS] WARNING: Batch item {i} anomaly (RMS={rms:.4f}, silence={is_silence}, static={is_static}, NaN={np.isnan(audio_native).any()}). Falling back to sequential.")
                audio_native = self._generate_speech_chunk(texts[i], description)
                # _generate_speech_chunk returns native SR — resample here
                if self._native_sr != SAMPLE_RATE:
                    audio_native = torchaudio.functional.resample(
                        torch.tensor(audio_native).unsqueeze(0),
                        orig_freq=self._native_sr,
                        new_freq=SAMPLE_RATE,
                    ).squeeze(0).numpy()
                results.append(audio_native.astype(np.float32))
                continue

            # Hard normalization
            max_val = np.abs(audio_native).max()
            if max_val > 1.0:
                audio_native = audio_native / max_val

            # Handle empty generation
            if audio_native.size == 0:
                results.append(np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32))
                continue

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

            results.append(audio_24k.astype(np.float32))

        return results

    def synthesize(
        self,
        segments: list[dict],
        voice: str = "",
        speed: float = 0.85,
        progress_cb=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize all script segments into a single audio track.

        Args:
            segments: Parsed script segments from parler preprocessor.
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
        description = adjust_description_for_speed(description, speed)

        audio_chunks: list[np.ndarray] = []
        activity_chunks: list[np.ndarray] = []
        total = len(segments)

        # ── Voice reference for audio-prefix conditioning ─────────────
        # After generating the first speech chunk, we extract a short
        # snippet and feed it as input_values to all subsequent chunks.
        # This anchors voice identity through the audio encoder, not just
        # the random seed, and is the most effective anti-drift technique.
        voice_ref: torch.Tensor | None = None
        ref_samples = int(_VOICE_REF_SECONDS * (self._native_sr or 44100))

        for idx, segment in enumerate(segments):
            if segment["type"] == "speech":
                sentences = split_into_sentences(segment["text"])
                if not sentences:
                    sentences = [segment["text"]]

                for si, sent in enumerate(sentences):
                    est_tokens = estimate_max_tokens(sent)
                    print(f"[ParlerTTS] Segment {idx+1}/{total}, "
                          f"sentence {si+1}/{len(sentences)} "
                          f"({len(sent.split())} words, ≤{est_tokens} tokens)")

                    # Generate at native SR
                    audio_native = self._generate_speech_chunk(
                        sent, description, voice_ref=voice_ref,
                    )

                    # Build voice reference from the first chunk
                    if voice_ref is None and audio_native.size >= ref_samples // 2:
                        snippet = audio_native[:ref_samples]
                        # Shape must be (batch, channels, time) for the DAC/EnCodec audio encoder.
                        # unsqueeze(0) = batch dim, unsqueeze(1) = channel dim (mono).
                        voice_ref = torch.tensor(
                            snippet, dtype=torch.float32
                        ).unsqueeze(0).unsqueeze(0)  # → (1, 1, T)
                        print(f"[ParlerTTS] Voice reference captured "
                              f"({len(snippet)/self._native_sr:.1f}s)")

                    # Resample native SR → 24 kHz for pipeline output
                    if self._native_sr != SAMPLE_RATE:
                        audio_24k = torchaudio.functional.resample(
                            torch.tensor(audio_native).unsqueeze(0),
                            orig_freq=self._native_sr,
                            new_freq=SAMPLE_RATE,
                            lowpass_filter_width=64,
                            rolloff=0.94,
                            resampling_method="sinc_interp_kaiser",
                        ).squeeze(0).numpy()
                    else:
                        audio_24k = audio_native

                    audio_chunks.append(audio_24k.astype(np.float32))
                    activity_chunks.append(np.ones(len(audio_24k), dtype=bool))

                    # Add inter-sentence pause (not after last sentence)
                    if si < len(sentences) - 1:
                        pause_sec = (
                            ELLIPSIS_PAUSE_SEC
                            if sent.rstrip().endswith(("...", "\u2026"))
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

        fade_ms = 30
        fade_samples = int((fade_ms / 1000.0) * SAMPLE_RATE)
        
        voice_audio = crossfade_audio_chunks(audio_chunks, fade_samples)
        voice_activity = crossfade_activity_chunks(activity_chunks, fade_samples)

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
