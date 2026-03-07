"""Parler TTS engine for MoodScape — description-controlled speech synthesis.

Target hardware: Apple Silicon M1 Max (36 GB unified memory, 24-core GPU)
Parler TTS native sample rate: 44100 Hz -> resampled to 24000 Hz for pipeline
"""

import gc
import logging
import os
import re
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

# Compatibility: transformers >=4.47 removed SlidingWindowCache from cache_utils.
# parler_tts (designed for transformers==4.46.1) imports it at module load time.
# Backfill with StaticCache before parler_tts is imported anywhere.
try:
    from transformers.cache_utils import SlidingWindowCache as _swc  # noqa: F401
except ImportError:
    import transformers.cache_utils as _tcu
    from transformers.cache_utils import StaticCache as _sc
    _tcu.SlidingWindowCache = _sc

# Compatibility: transformers >= 4.49 removed isin_mps_friendly from pytorch_utils.
# Backfill it for parler_tts to prevent import errors.
import transformers.pytorch_utils as _tpu
if not hasattr(_tpu, 'isin_mps_friendly'):
    import torch as _torch
    def _isin_mps_friendly(elements, test_elements, assume_unique=False, invert=False):
        return _torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)
    _tpu.isin_mps_friendly = _isin_mps_friendly

# Compatibility: transformers >= 4.47 attempts to instantiate ParlerTTSConfig
# without arguments to get defaults, which fails because it requires 3 configs.
try:
    from parler_tts.configuration_parler_tts import ParlerTTSConfig as _ParlerTTSConfig
    _ParlerTTSConfig.has_no_defaults_at_init = True
    # transformers >= 4.49 also removes tie_encoder_decoder from base config
    if not hasattr(_ParlerTTSConfig, "tie_encoder_decoder"):
        _ParlerTTSConfig.tie_encoder_decoder = False

    # Register ParlerTTSConfig with transformers' AutoConfig so that from_pretrained
    # does not warn "model of type parler_tts to instantiate model of type ''".
    from transformers import AutoConfig as _AutoConfig
    try:
        _AutoConfig.register("parler_tts", _ParlerTTSConfig)
    except ValueError:
        pass  # already registered

    from parler_tts.modeling_parler_tts import ParlerTTSForConditionalGeneration as _ParlerTTSGen
    _orig_tie_weights = _ParlerTTSGen.tie_weights
    def _patched_tie_weights(self, *args, **kwargs):
        _orig_tie_weights(self)
        super(_ParlerTTSGen, self).tie_weights(*args, **kwargs)
    _ParlerTTSGen.tie_weights = _patched_tie_weights

    from transformers import GenerationMixin
    if GenerationMixin not in _ParlerTTSGen.__bases__:
        _ParlerTTSGen.__bases__ = (GenerationMixin,) + _ParlerTTSGen.__bases__

    # ParlerTTSForCausalLM (the inner audio decoder) also defines
    # prepare_inputs_for_generation but doesn't inherit GenerationMixin,
    # triggering the same transformers warning.  Fix it the same way.
    from parler_tts.modeling_parler_tts import ParlerTTSForCausalLM as _ParlerTTSCausalLM
    if GenerationMixin not in _ParlerTTSCausalLM.__bases__:
        _ParlerTTSCausalLM.__bases__ = (GenerationMixin,) + _ParlerTTSCausalLM.__bases__

    # Compatibility: transformers >= 5.x changed _get_initial_cache_position signature from
    # parler's (self, input_ids_tensor, model_kwargs) to (self, seq_length_int, device, model_kwargs).
    # Replace with an implementation that accepts the new signature but runs parler's logic,
    # which computes cache positions based on input length and slices off past_length.
    import torch as _torch
    try:
        from transformers.cache_utils import Cache as _Cache
    except ImportError:
        _Cache = object

    try:
        from torch._dynamo import is_compiling as _is_torchdynamo_compiling
    except ImportError:
        def _is_torchdynamo_compiling():
            return False

    def _patched_cache_pos(self, seq_length, device, model_kwargs):
        """Cache position adapter for transformers 5.x + parler_tts compatibility.

        Transformers 5.x calls with (seq_length: int, device, model_kwargs).
        Parler's original code expected (input_ids_tensor, model_kwargs) and did
        torch.ones_like(input_ids[0, :]). We replicate the same logic using the
        integer seq_length directly so positions are correct for parler's audio decoder.
        """
        cache_position = _torch.ones(seq_length, dtype=_torch.int64, device=device).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            if not isinstance(cache, _Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()
            if not _is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    _ParlerTTSGen._get_initial_cache_position = _patched_cache_pos

    # Compatibility: transformers 5.x _update_model_kwargs_for_generation ACCUMULATES
    # cache_position as [0, 1, 2, ...], so cache_position[0] is always 0. But parler's
    # prepare_inputs_for_generation uses cache_position[0] as past_key_values_length to
    # compute decoder attention mask sizes. Fix: patch prepare_inputs_for_generation to
    # rebuild cache_position from past_key_values.get_seq_length() when cache has content,
    # so cache_position[0] correctly reflects how many tokens have been decoded so far.
    from parler_tts.modeling_parler_tts import EncoderDecoderCache as _EncoderDecoderCache

    # Compatibility: transformers 5.x DynamicCache changed internal storage from
    # .key_cache[layer_idx] / .value_cache[layer_idx] lists to .layers[layer_idx].keys/.values.
    # parler_tts cross-attention reads: past_key_value.key_cache[self.layer_idx].
    # Add .key_cache and .value_cache as list-like views backed by .layers[i].keys/values.
    from transformers.cache_utils import DynamicCache as _DynamicCache

    if not hasattr(_DynamicCache, 'key_cache'):
        class _KVView:
            __slots__ = ("_cache", "_attr")
            def __init__(self, cache, attr): self._cache, self._attr = cache, attr
            def __getitem__(self, idx): return getattr(self._cache.layers[idx], self._attr)
            def __setitem__(self, idx, val): setattr(self._cache.layers[idx], self._attr, val)

        def _key_cache_prop(self): return _KVView(self, "keys")
        def _val_cache_prop(self): return _KVView(self, "values")
        _DynamicCache.key_cache = property(_key_cache_prop)
        _DynamicCache.value_cache = property(_val_cache_prop)

    _orig_prep_inputs = _ParlerTTSGen.prepare_inputs_for_generation

    def _patched_prep_inputs(self, decoder_input_ids, past_key_values=None, cache_position=None, **kwargs):
        if (
            cache_position is not None
            and past_key_values is not None
            and isinstance(past_key_values, _EncoderDecoderCache)
            and past_key_values.get_seq_length() > 0
        ):
            # Rebuild cache_position so that cache_position[0] == get_seq_length().
            # This is what parler's attention-mask formula requires:
            #   generated_length = past_key_values_length - prompt_length + 1
            # and past_key_values_length = cache_position[0].
            seq_len = past_key_values.get_seq_length()
            cache_position = _torch.tensor(
                [seq_len], dtype=cache_position.dtype, device=cache_position.device
            )
        return _orig_prep_inputs(self, decoder_input_ids, past_key_values=past_key_values,
                                 cache_position=cache_position, **kwargs)

    _ParlerTTSGen.prepare_inputs_for_generation = _patched_prep_inputs

    _orig_generate = _ParlerTTSGen.generate
    def _patched_generate(self, *args, **kwargs):
        generation_config = kwargs.get("generation_config", None)
        if generation_config is None:
            generation_config = self.generation_config
            import copy
            generation_config = copy.deepcopy(generation_config)
            
        if getattr(generation_config, "num_return_sequences", None) is None:
            generation_config.num_return_sequences = 1
            
        kwargs["generation_config"] = generation_config
        return _orig_generate(self, *args, **kwargs)
    _ParlerTTSGen.generate = _patched_generate
except ImportError:
    pass

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

# ── Token budget estimation ──────────────────────────────────────────────
# Parler TTS Large v1 uses a DAC codec at 44100 Hz with ~86 tokens/second.
# Without max_new_tokens, generate() runs to max_length (2580 tokens ≈ 30s)
# for EVERY sentence — the #1 cause of infinite-feeling stalls.
# Heuristic: meditation speech at 0.85x speed ≈ 120 words/min ≈ 2 words/sec
# → 1 word ≈ 0.5s ≈ 43 tokens.  We use 50 tokens/word for safety margin
# and clamp to [256, 2048] to avoid too-short or model-max blowout.
_TOKENS_PER_WORD = 50
_MIN_NEW_TOKENS = 256   # ~3 seconds minimum
_MAX_NEW_TOKENS = 2048  # ~24 seconds ceiling per chunk

# Voice-drift prevention: sentences longer than this are split at clause
# boundaries to limit autoregressive drift within a single generation.
_MAX_WORDS_PER_CHUNK = 25

# Audio prefix conditioning: length of the reference snippet (in seconds)
# extracted from the first generated sentence and fed as input_values to
# all subsequent chunks, anchoring voice identity via the audio encoder.
_VOICE_REF_SECONDS = 5.0

# Lower temperature = more deterministic = less voice drift between chunks.
_TEMPERATURE = 0.65


def _estimate_max_tokens(text: str) -> int:
    """Estimate max_new_tokens for a text chunk based on word count."""
    n_words = len(text.split())
    estimate = n_words * _TOKENS_PER_WORD
    return max(_MIN_NEW_TOKENS, min(estimate, _MAX_NEW_TOKENS))

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
    Long sentences are further split at clause boundaries
    (commas, semicolons, em-dashes) to limit autoregressive drift.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    raw = [p for p in parts if p.strip()]

    if len(raw) <= 1:
        return _cap_sentence_length(raw)

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

    return _cap_sentence_length(merged)


def _cap_sentence_length(sentences: list[str]) -> list[str]:
    """Split sentences exceeding _MAX_WORDS_PER_CHUNK at clause boundaries.

    Tries commas, semicolons, and em-dashes first. Falls back to splitting
    at the word limit if no clause boundary is found.
    """
    result: list[str] = []
    for sent in sentences:
        words = sent.split()
        if len(words) <= _MAX_WORDS_PER_CHUNK:
            result.append(sent)
            continue

        # Try splitting at clause boundaries (comma, semicolon, em-dash)
        clause_parts = re.split(r"(?<=[,;—–])\s+", sent)
        if len(clause_parts) > 1:
            # Greedily merge clause parts up to the word limit
            chunk = ""
            for part in clause_parts:
                candidate = (chunk + " " + part).strip() if chunk else part
                if len(candidate.split()) > _MAX_WORDS_PER_CHUNK and chunk:
                    result.append(chunk)
                    chunk = part
                else:
                    chunk = candidate
            if chunk:
                result.append(chunk)
        else:
            # No clause boundaries — hard split at word limit
            for i in range(0, len(words), _MAX_WORDS_PER_CHUNK):
                chunk = " ".join(words[i:i + _MAX_WORDS_PER_CHUNK])
                result.append(chunk)

    return result


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

        max_tokens = _estimate_max_tokens(text)

        # Build generate kwargs — optionally include audio conditioning
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            do_sample=True,
            temperature=_TEMPERATURE,
            max_new_tokens=max_tokens,
        )
        if voice_ref is not None:
            gen_kwargs["input_values"] = voice_ref.to(self.device)

        try:
            with torch.no_grad():
                generation = self.model.generate(**gen_kwargs)
        except RuntimeError as e:
            if "MPS" in str(e) or "placeholder" in str(e).lower():
                print(f"[ParlerTTS] MPS error, falling back to CPU: {e}")
                self.model.cpu()
                gen_kwargs = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in gen_kwargs.items()
                }
                with torch.no_grad():
                    generation = self.model.generate(**gen_kwargs)
                self.model.to(self.device)
            else:
                raise

        generation = generation.cpu()

        # ── Vocal sanity check: NaN / Inf → automatic CPU fallback ────
        if torch.isnan(generation).any() or torch.isinf(generation).any():
            print(
                "[ParlerTTS] WARNING: Model produced NaN/Inf on "
                f"{self.device}. Re-generating on CPU..."
            )
            self.model.cpu()
            gen_kwargs = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in gen_kwargs.items()
            }
            with torch.no_grad():
                generation = self.model.generate(**gen_kwargs)
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
        max_tokens = max(_estimate_max_tokens(t) for t in texts)

        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                do_sample=True,
                temperature=_TEMPERATURE,
                max_new_tokens=max_tokens,
            )

        # generation is (batch_size, sequence_length) — extract each item
        generation = generation.cpu().float()

        results: list[np.ndarray] = []
        for i in range(batch_size):
            audio_native = generation[i].numpy().squeeze()

            # NaN / Inf guard
            if np.isnan(audio_native).any() or not np.isfinite(audio_native).all():
                print(f"[ParlerTTS] WARNING: Batch item {i} has NaN/Inf, falling back to sequential.")
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

        # ── Voice reference for audio-prefix conditioning ─────────────
        # After generating the first speech chunk, we extract a short
        # snippet and feed it as input_values to all subsequent chunks.
        # This anchors voice identity through the audio encoder, not just
        # the random seed, and is the most effective anti-drift technique.
        voice_ref: torch.Tensor | None = None
        ref_samples = int(_VOICE_REF_SECONDS * (self._native_sr or 44100))

        for idx, segment in enumerate(segments):
            if segment["type"] == "speech":
                sentences = _split_into_sentences(segment["text"])
                if not sentences:
                    sentences = [segment["text"]]

                for si, sent in enumerate(sentences):
                    est_tokens = _estimate_max_tokens(sent)
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
                        voice_ref = torch.tensor(
                            snippet, dtype=torch.float32
                        ).unsqueeze(0)
                        print(f"[ParlerTTS] Voice reference captured "
                              f"({len(snippet)/self._native_sr:.1f}s)")

                    # Resample native SR → 24 kHz for pipeline output
                    if self._native_sr != SAMPLE_RATE:
                        audio_24k = torchaudio.functional.resample(
                            torch.tensor(audio_native).unsqueeze(0),
                            orig_freq=self._native_sr,
                            new_freq=SAMPLE_RATE,
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
