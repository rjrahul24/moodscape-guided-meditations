# Claude Code Prompt — Replace Fish Speech with Parler TTS in MoodScape

## Overview & Objective

MoodScape is a locally-running guided meditation audio generation tool with a Gradio web UI. It currently supports two TTS engines via a Strategy Design Pattern: **Kokoro TTS** (primary) and **Fish Speech** (secondary, API-based). This prompt instructs you to **completely remove Fish Speech** and **replace it with Parler TTS** — a fully local, open-source, description-controlled text-to-speech model.

After this work is complete, the app should offer exactly two TTS engine choices:
1. **Kokoro TTS** (unchanged — the existing implementation)
2. **Parler TTS** (new — replacing Fish Speech)

**Target Hardware:** Apple Silicon M1 Max — 36 GB unified memory, 24-core GPU, MPS backend.

---

## CRITICAL RULES

1. **Do NOT modify any existing Kokoro TTS code.** The `kokoro_engine.py` (or `tts_engine.py`) file, its voice list, its speed parameter behavior, its sentence splitting logic, its MPS fallback — all must remain exactly as they are.
2. **Do NOT modify `script_parser.py`, `music_engine.py`, `audio_processor.py`, `mixer.py`, or any other downstream module** unless strictly necessary to support the new engine selection.
3. **Do NOT change the pipeline's sequential model loading pattern** (load TTS → generate → unload TTS → load MusicGen → generate → unload MusicGen). Parler TTS must follow this same lifecycle.
4. **Do NOT change the output contract.** The TTS engine must return `(voice_audio: np.ndarray float32 mono 24000Hz, voice_activity: np.ndarray bool)` — the exact same signature Kokoro uses.
5. **All Fish Speech code, files, configurations, API key references, and UI elements must be fully deleted.** No dead code, no commented-out blocks, no unused imports.

---

## Part 1: Remove Fish Speech — Complete Deletion Checklist

### Files to DELETE entirely:
- `core/fish_engine.py` (the Fish Speech engine implementation)
- Any Fish Speech-specific configuration files, JSON presets, or voice reference audio files
- Any Fish Speech test files or scripts

### Files to EDIT (remove Fish Speech references):
1. **`core/speech_engine.py`** (or wherever the abstract base class / strategy interface lives):
   - Remove Fish Speech from the engine registry/factory
   - Remove any Fish Speech-specific imports
   - Keep the abstract interface intact (Parler TTS will implement it)

2. **`core/pipeline.py`**:
   - Remove all Fish Speech fallback logic (the "if Fish Speech API fails, silently switch to Kokoro" behavior)
   - Remove Fish Speech API key checks
   - Remove Fish Speech-specific sanitization calls
   - Remove any `FISH_API_KEY` environment variable references
   - Remove the meditation breath `(sighing)` clip generation feature that was specific to Fish Speech
   - Remove Fish Speech 200-character chunking logic
   - Remove Fish Speech loudness normalization to -3 dB peak (Parler TTS will handle this differently)

3. **`app.py`** (Gradio UI):
   - Remove Fish Speech from the TTS engine dropdown/selector
   - Remove the Fish Speech API key input field (password field)
   - Remove Fish Speech voice cloning / reference audio upload UI elements
   - Remove Fish Speech emotion/tone preset selector (Deep Sleep, Anxiety Relief, etc.)
   - Remove any Fish Speech-specific settings or sliders
   - Replace with Parler TTS UI elements (detailed in Part 3)

4. **`requirements.txt`**:
   - Remove any Fish Speech SDK/client dependencies (e.g., `fish-audio-sdk`, `httpx`, or any Fish Speech-specific packages)
   - Add Parler TTS dependencies (detailed in Part 3)

5. **`README.md`**:
   - Remove all Fish Speech mentions, setup instructions, and API key documentation
   - Update to document Parler TTS as the second engine option

6. **`.env` / `.env.example` / environment configuration**:
   - Remove `FISH_API_KEY` and any Fish Speech environment variables

7. **Any Fish Speech emotion tag constants or sanitization utilities:**
   - Delete the emotion/tone tagging system: `(relaxed)`, `(whispering)`, `(sighing)`, `(soft tone)`, `(breathy)`, etc.
   - Delete the Fish Speech text sanitization layer (no ALL-CAPS conversion, special character cleaning, style header injection, `[inhale]`/`[exhale]` tag conversion)
   - Delete the meditation style preset mappings (Deep Sleep, Anxiety Relief, Morning Energy, Visualization, Calm Relaxation)

### Verification after deletion:
- `grep -r "fish" --include="*.py" .` should return zero results (case-insensitive: `grep -ri "fish"`)
- `grep -r "FISH_API" .` should return zero results
- `grep -r "fish_engine" .` should return zero results
- `grep -r "fish_speech" .` should return zero results
- The app must still start and generate meditations using Kokoro TTS with zero errors

---

## Part 2: Parler TTS — Model Research & Technical Specification

### What is Parler TTS?

Parler-TTS is an open-source, high-quality text-to-speech model from Hugging Face that generates speech controlled by **natural-language descriptions**. Instead of selecting from named voices with fixed characteristics, you write a text description of the voice you want (gender, pitch, speed, tone, recording quality, reverb) and the model generates speech matching that description.

### Model Details

| Property | Value |
|----------|-------|
| Model ID (primary) | `parler-tts/parler-tts-large-v1` |
| Model ID (fallback) | `parler-tts/parler-tts-mini-v1` |
| Parameters (Large) | 2.2 billion |
| Parameters (Mini) | 880 million |
| Training data | 45,000 hours of narrated audio |
| Architecture | Conditional diffusion model with decoder-only transformer, iSTFTNet vocoder |
| Output sample rate | **44,100 Hz** (must be resampled to 24,000 Hz for MoodScape pipeline) |
| Output format | Mono float32 numpy array |
| Named speakers | 34 speakers (e.g., Jon, Lea, Gary, Jenna, Mike, Laura) for voice consistency |
| License | Apache 2.0 (code and inference) |
| Input | Two separate text inputs: `description` (voice/style control) + `prompt` (text to speak) |
| Control method | Natural language description string |
| Library | `parler_tts` Python package |
| Dependencies | `transformers`, `torch`, `torchaudio`, `soundfile` |

### Why Large v1 for meditation:
- Richer prosody control than Mini — critical for calm, slow, expressive delivery
- 2.2B parameters fit comfortably in 36 GB unified memory (~5-6 GB with bfloat16)
- Superior voice consistency with named speakers across long meditation segments
- Better response to nuanced description keywords like "breathy," "intimate," "slow pauses"

### Apple Silicon / MPS Specifics:
- Device: Use `"mps"` when `torch.backends.mps.is_available()` is True, fallback to `"cpu"`
- Dtype: Use `torch.bfloat16` to reduce memory footprint (requires PyTorch nightly or 2.4+ on Apple Silicon)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before any torch import (same as Kokoro)
- **Important:** Some MPS operations may not be supported. If MPS fails, fall back to CPU gracefully
- `torch.compile(model)` can provide 2-4x speedup but may have MPS compatibility issues — test and use only if stable
- Batch size must be 1 on MPS to prevent OOM
- Expected inference time: ~10-30 seconds for 30 seconds of audio on M1 Max

### Sample Rate Alignment (CRITICAL):
Parler TTS outputs at **44,100 Hz**. MoodScape's pipeline standard is **24,000 Hz** (matching Kokoro). You MUST resample:
```python
import torchaudio
audio_24k = torchaudio.functional.resample(
    torch.tensor(audio_44k).unsqueeze(0),
    orig_freq=44100,
    new_freq=24000
).squeeze(0).numpy()
```

---

## Part 3: Implement Parler TTS Engine — `core/parler_engine.py`

### 3.1 Installation & Dependencies

Add to `requirements.txt`:
```
git+https://github.com/huggingface/parler-tts.git
transformers>=4.35.0
```

Note: `torch`, `torchaudio`, `soundfile`, and `numpy` are already in requirements from Kokoro/MusicGen.

### 3.2 Engine Interface Contract

The Parler TTS engine MUST implement the exact same interface as the Kokoro engine. This is the contract that `pipeline.py` depends on:

```python
class ParlerTTSEngine:
    def __init__(self): ...
    def load_model(self): ...
    def unload_model(self): ...
    def synthesize(
        self,
        segments: list[dict],        # From script_parser.py
        voice: str,                   # Voice description string OR named preset
        speed: float = 0.85,          # 0.5–1.0 (used to adjust description)
        progress_cb=None,             # Called with (current_index, total_segments)
    ) -> tuple[np.ndarray, np.ndarray]:
        # Returns:
        #   voice_audio: float32 numpy array, mono, 24000 Hz
        #   voice_activity: bool numpy array, True where speech is present
        ...
```

### 3.3 Complete Implementation

```python
"""Parler TTS engine for MoodScape — description-controlled speech synthesis.

Target hardware: Apple Silicon M1 Max (36 GB unified memory, 24-core GPU)
Parler TTS native sample rate: 44100 Hz → resampled to 24000 Hz for pipeline
"""

import gc
import os
import re

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 24000          # Pipeline standard (matches Kokoro)
PARLER_NATIVE_SR = 44100     # Parler TTS output sample rate

# Inter-sentence pauses (matching Kokoro behavior for consistency)
INTER_SENTENCE_PAUSE_SEC = 0.8
ELLIPSIS_PAUSE_SEC = 1.2
MIN_SENTENCE_WORDS = 6       # Parler needs more context than Kokoro for good prosody

# ── Voice Description Presets ─────────────────────────────────────────────
# These are carefully crafted for meditation use cases.
# The user can also type a fully custom description.

VOICE_PRESETS = [
    (
        "Serene Female — warm, calm, breathy",
        "A female speaker with a very warm, breathy, and compassionate tone. "
        "She speaks very slowly with long pauses between sentences. "
        "The recording is crystal clear, intimate, and has no background noise."
    ),
    (
        "Gentle Male — deep, grounding, steady",
        "A warm, low, male narrator voice with very slow pacing, "
        "breathy and soft consonants, intimate close-mic feel, "
        "very clear audio with slight room reverb."
    ),
    (
        "Whisper Female — soft, sleepy, ASMR",
        "A soft, warm, female whisper voice with slow cadence, "
        "gentle breathy quality, extra long pauses after commas and periods, "
        "very low pitch, clear with mild hall reverb, very clear audio."
    ),
    (
        "Calm Coach — neutral, steady, guiding",
        "A calm neutral voice, gentle and steady, slightly slow tempo, "
        "even intonation, clear audio, no background noise, "
        "slightly rounded vowels for a soothing effect."
    ),
    (
        "Jon — monotone, close-mic, consistent",
        "Jon's voice is monotone yet slightly slow in delivery, "
        "with a very close recording that almost has no background noise."
    ),
    (
        "Lea — expressive, warm, flowing",
        "Lea's voice is calming and expressive, with gentle pauses for reflection, "
        "clear and intimate sound, moderate speed."
    ),
    (
        "Custom Description",
        ""  # User fills in their own description via the UI textbox
    ),
]


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries.
    Handles standard endings (.!?) and ellipsis (...).
    Short sentences are merged for better Parler prosody.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    raw = [p for p in parts if p.strip()]

    if len(raw) <= 1:
        return raw

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


def _adjust_description_for_speed(description: str, speed: float) -> str:
    """Modify the voice description to reflect the requested speed.
    
    Parler TTS doesn't have a numeric speed parameter like Kokoro.
    Instead, we inject pacing keywords into the description.
    """
    # Remove any existing speed-related phrases to avoid conflicts
    speed_phrases = [
        "very slow", "slow", "moderate speed", "slightly slow",
        "fast", "slightly fast", "very fast", "normal speed"
    ]
    desc_lower = description.lower()
    for phrase in speed_phrases:
        desc_lower = desc_lower.replace(phrase, "")
    
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
    
    # Append pacing to description
    return f"{description.rstrip('. ')}. Speaking with {pace}."


class ParlerTTSEngine:
    """Wraps Parler TTS Large v1 for meditation audio generation.
    
    Uses natural-language descriptions to control voice characteristics.
    Implements the same interface as TTSEngine (Kokoro) for seamless
    integration with the MoodScape pipeline.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = None

    def load_model(self, preferred: str = "parler-tts/parler-tts-large-v1"):
        """Load Parler TTS model with MPS acceleration.
        
        Tries Large v1 (2.2B) first. Falls back to Mini v1 (880M) if
        memory is insufficient.
        """
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        # Device selection — prefer MPS on Apple Silicon
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        torch_dtype = torch.bfloat16

        candidates = [
            preferred,
            "parler-tts/parler-tts-mini-v1",  # Fallback
        ]

        for model_name in candidates:
            try:
                self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model_name = model_name
                print(f"[ParlerTTS] Loaded {model_name} on {self.device}")
                return
            except Exception as e:
                print(f"[ParlerTTS] Failed to load {model_name}: {e}")
                # Clean up partial load
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
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, AttributeError):
            pass

    def _generate_speech_chunk(
        self, text: str, description: str
    ) -> np.ndarray:
        """Generate speech for a single text chunk.
        
        Returns float32 numpy array at 24000 Hz (resampled from 44100).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input_ids = self.tokenizer(
            description, return_tensors="pt"
        ).input_ids.to(self.device)

        prompt_input_ids = self.tokenizer(
            text, return_tensors="pt"
        ).input_ids.to(self.device)

        try:
            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                    do_sample=True,
                    temperature=0.8,    # Lower = more consistent calm delivery
                    # top_p=0.9,        # Optional — uncomment for more variation
                )
        except RuntimeError as e:
            if "MPS" in str(e) or "placeholder" in str(e).lower():
                # MPS fallback — move to CPU for this chunk
                print(f"[ParlerTTS] MPS error, falling back to CPU: {e}")
                self.model.cpu()
                input_ids = input_ids.cpu()
                prompt_input_ids = prompt_input_ids.cpu()
                with torch.no_grad():
                    generation = self.model.generate(
                        input_ids=input_ids,
                        prompt_input_ids=prompt_input_ids,
                        do_sample=True,
                        temperature=0.8,
                    )
                # Move back to MPS for next chunk
                self.model.to(self.device)
            else:
                raise

        # Convert to numpy
        audio_44k = generation.cpu().float().numpy().squeeze()

        # Handle edge case of empty generation
        if audio_44k.size == 0:
            return np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)

        # Resample 44100 Hz → 24000 Hz
        audio_tensor = torch.tensor(audio_44k).unsqueeze(0)
        audio_24k = torchaudio.functional.resample(
            audio_tensor,
            orig_freq=PARLER_NATIVE_SR,
            new_freq=SAMPLE_RATE,
        ).squeeze(0).numpy()

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
            speed: Speaking speed 0.5–1.0. Translated into description
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
                # Split into sentences for chunk-by-chunk generation
                # This prevents quality degradation on long texts
                sentences = _split_into_sentences(segment["text"])
                if not sentences:
                    sentences = [segment["text"]]

                for j, sentence in enumerate(sentences):
                    speech_audio = self._generate_speech_chunk(
                        sentence, description
                    )

                    audio_chunks.append(speech_audio)
                    activity_chunks.append(
                        np.ones(len(speech_audio), dtype=bool)
                    )

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
                audio_chunks.append(
                    np.zeros(num_samples, dtype=np.float32)
                )
                activity_chunks.append(
                    np.zeros(num_samples, dtype=bool)
                )

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
        
        If voice matches a preset label → return that preset's description.
        If voice is a raw description string → return it directly.
        If empty → return the default meditation preset.
        """
        if not voice or not voice.strip():
            # Default: first preset (Serene Female)
            return VOICE_PRESETS[0][1]

        # Check if it matches a preset label
        for label, desc in VOICE_PRESETS:
            if voice == label or voice == desc:
                if desc:  # Not the "Custom" entry
                    return desc
                else:
                    # Custom — voice string IS the description
                    return voice

        # Assume it's a raw description string
        return voice
```

### 3.4 Voice Description Prompting Guide for Meditation

This section provides context on how to craft effective descriptions. The UI should include this as help text or a tooltip.

**Key description keywords for meditation audio:**

| Goal | Effective Keywords |
|------|-------------------|
| Warmth | "breathy," "compassionate," "intimate," "whispered," "resonant" |
| Calm energy | "expressive," "emotionally charged," "sincere," "gentle energy" |
| Slow pacing | "very slow," "long pauses," "deliberate cadence," "monotone but soothing" |
| Audio quality | "studio recording," "very clear audio," "high fidelity," "no reverb," "no background noise" |
| Consistency | Use named speakers: "Jon's voice," "Lea's voice" for uniform output across segments |

**Punctuation matters for prosody:**
- Commas (`,`) add short natural pauses
- Ellipses (`...`) create longer, more contemplative pauses
- Em-dashes (`—`) add dramatic micro-breaks
- Periods followed by line breaks create the longest natural pauses

**What NOT to put in descriptions:**
- Emotion tags like `(relaxed)` or `(whispering)` — these are Fish Speech syntax, not Parler
- SSML tags — Parler doesn't support them
- Speed numbers — use descriptive words instead ("very slow," not "0.7x speed")

---

## Part 4: Update Pipeline — Engine Selection Logic

### 4.1 Update `core/speech_engine.py` (Abstract Base / Factory)

The existing Strategy pattern should be updated to register Parler TTS instead of Fish Speech:

```python
"""TTS Engine factory — selects between Kokoro and Parler TTS."""

from enum import Enum


class TTSEngineType(Enum):
    KOKORO = "kokoro"
    PARLER = "parler"


def create_tts_engine(engine_type: TTSEngineType):
    """Factory function returning the appropriate TTS engine instance."""
    if engine_type == TTSEngineType.KOKORO:
        from core.kokoro_engine import TTSEngine
        return TTSEngine()
    elif engine_type == TTSEngineType.PARLER:
        from core.parler_engine import ParlerTTSEngine
        return ParlerTTSEngine()
    else:
        raise ValueError(f"Unknown TTS engine type: {engine_type}")
```

### 4.2 Update `core/pipeline.py`

The pipeline must handle engine selection and pass the right parameters:

```python
# In the generate() function signature, replace Fish Speech params:
# REMOVE: fish_api_key, fish_voice_ref, fish_emotion_preset, fish_style
# ADD: engine_type (str), parler_description (str, optional)

def generate(
    script: str,
    music_prompt: str,
    # Engine selection
    engine_type: str = "kokoro",          # "kokoro" or "parler"
    # Kokoro-specific
    voice: str = "af_heart",              # Kokoro voice name
    speed: float = 0.85,                  # Shared — both engines use this
    # Parler-specific
    parler_voice_preset: str = "",        # Preset name or custom description
    parler_custom_description: str = "",  # Only used if preset is "Custom Description"
    # Shared audio settings (unchanged)
    duck_db: float = -8.0,
    reverb_amount: float = 0.15,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 5.0,
    output_format: str = "wav",
    progress_callback=None,
) -> str:
    """Orchestrate the full meditation generation pipeline."""

    # 1. Parse script (unchanged)
    segments = parse_script(script)

    # 2. Create and load TTS engine
    from core.speech_engine import TTSEngineType, create_tts_engine

    if engine_type == "parler":
        tts = create_tts_engine(TTSEngineType.PARLER)
        tts.load_model()
        # Resolve voice: use custom description if preset is "Custom Description"
        voice_param = parler_custom_description if (
            parler_voice_preset == "Custom Description" and parler_custom_description
        ) else parler_voice_preset
        voice_audio, voice_activity = tts.synthesize(
            segments, voice=voice_param, speed=speed, progress_cb=progress_callback
        )
        tts.unload_model()
    else:
        # Kokoro (default — UNCHANGED from current implementation)
        tts = create_tts_engine(TTSEngineType.KOKORO)
        tts.load_model()
        voice_audio, voice_activity = tts.synthesize(
            segments, voice=voice, speed=speed, progress_cb=progress_callback
        )
        tts.unload_model()

    # 3. Everything below is UNCHANGED — music gen, FX, mixing, export
    # ... (keep all existing code exactly as-is)
```

**Key principle:** The pipeline treats both engines identically after synthesis. The output `(voice_audio, voice_activity)` is the same format regardless of which engine produced it, so all downstream processing (FX, ducking, mixing, normalization) works without any changes.

---

## Part 5: Update Gradio UI — `app.py`

### 5.1 Engine Selection

Add a radio button to choose between Kokoro and Parler TTS. When the selection changes, show/hide the relevant engine-specific controls.

```python
# Engine selector (replaces the old Kokoro/Fish Speech selector)
engine_selector = gr.Radio(
    choices=["Kokoro TTS", "Parler TTS"],
    value="Kokoro TTS",
    label="Speech Engine",
    info="Kokoro: fast, lightweight, preset voices. Parler: slower, richer, description-controlled voices."
)
```

### 5.2 Kokoro-Specific Controls (shown when Kokoro is selected)
These are UNCHANGED from the current implementation:
```python
kokoro_voice = gr.Dropdown(
    choices=[
        ("Heart — US Female (default)", "af_heart"),
        ("Sky — US Female (airy)",      "af_sky"),
        ("Nova — US Female (calm)",     "af_nova"),
        ("Nicole — US Female (ASMR)",   "af_nicole"),
        ("River — US Female (flowing)", "af_river"),
        ("Bella — US Female",           "af_bella"),
        ("Sarah — US Female",           "af_sarah"),
        ("Emma — UK Female (wise)",     "bf_emma"),
        ("Lily — UK Female (angelic)",  "bf_lily"),
        ("Adam — US Male (grounding)",  "am_adam"),
        ("Michael — US Male",           "am_michael"),
        ("George — UK Male (warm)",     "bm_george"),
    ],
    value="af_heart",
    label="Kokoro Voice",
    visible=True,
)
```

### 5.3 Parler-Specific Controls (shown when Parler is selected)
```python
parler_preset = gr.Dropdown(
    choices=[label for label, _ in VOICE_PRESETS],
    value=VOICE_PRESETS[0][0],  # "Serene Female — warm, calm, breathy"
    label="Voice Style",
    info="Select a meditation-optimized voice preset, or choose 'Custom Description' to write your own.",
    visible=False,
)

parler_custom_desc = gr.Textbox(
    label="Custom Voice Description",
    placeholder=(
        "Example: A warm, low female voice with slow, soothing delivery, "
        "breathy tone, and crystal clear studio recording with no background noise."
    ),
    lines=3,
    visible=False,
    info=(
        "Describe the voice you want. Key terms: warm/cold, breathy/clear, "
        "slow/fast, low/high pitch, male/female, close-mic/distant, "
        "reverb/no reverb. Use named speakers (Jon, Lea, etc.) for consistency."
    ),
)
```

### 5.4 Visibility Toggle Logic
```python
def on_engine_change(engine):
    """Show/hide engine-specific controls based on selection."""
    is_kokoro = engine == "Kokoro TTS"
    is_parler = engine == "Parler TTS"
    return {
        kokoro_voice: gr.update(visible=is_kokoro),
        parler_preset: gr.update(visible=is_parler),
        parler_custom_desc: gr.update(
            visible=is_parler
            # Further conditional: only show if preset == "Custom Description"
            # This can be handled with a second change handler on parler_preset
        ),
    }

engine_selector.change(
    fn=on_engine_change,
    inputs=[engine_selector],
    outputs=[kokoro_voice, parler_preset, parler_custom_desc],
)
```

### 5.5 Generate Button Handler Update
```python
def on_generate(
    script, music_prompt, engine,
    kokoro_voice_val, speed,
    parler_preset_val, parler_custom_val,
    duck_db, reverb, fade_in, fade_out, fmt,
    progress=gr.Progress()
):
    engine_type = "kokoro" if engine == "Kokoro TTS" else "parler"

    return pipeline.generate(
        script=script,
        music_prompt=music_prompt,
        engine_type=engine_type,
        voice=kokoro_voice_val,                    # Only used if engine_type=="kokoro"
        speed=speed,                                # Used by both engines
        parler_voice_preset=parler_preset_val,     # Only used if engine_type=="parler"
        parler_custom_description=parler_custom_val,
        duck_db=duck_db,
        reverb_amount=reverb,
        fade_in_sec=fade_in,
        fade_out_sec=fade_out,
        output_format=fmt,
        progress_callback=progress,
    )
```

### 5.6 Remove ALL Fish Speech UI elements:
- Delete the Fish Speech API key password field
- Delete the Fish Speech voice cloning / reference audio upload
- Delete the Fish Speech emotion preset selector
- Delete any Fish Speech-specific status messages or error handling
- Delete the Fish Speech engine option from the engine selector

---

## Part 6: Memory Management & Performance Optimization

### 6.1 Sequential Loading (CRITICAL — same pattern as existing)

The pipeline MUST follow the existing sequential pattern. Parler TTS (2.2B params) and MusicGen should NEVER be loaded simultaneously:

```python
# Step 1: Load Parler TTS → Synthesize → Unload
parler_engine.load_model()                    # ~5-6 GB with bfloat16
voice_audio, voice_activity = parler_engine.synthesize(...)
parler_engine.unload_model()                  # Free memory
gc.collect()

# Step 2: Load MusicGen → Generate → Unload (unchanged)
music_engine.load_model()
music_audio = music_engine.generate(...)
music_engine.unload_model()
gc.collect()

# Step 3: Mixing (CPU-only, no model memory needed)
```

### 6.2 Parler TTS Memory Budget on M1 Max (36 GB)

| Component | Memory |
|-----------|--------|
| Parler Large v1 (bfloat16) | ~5-6 GB |
| Tokenizer + overhead | ~0.5 GB |
| Inference buffers | ~2-3 GB |
| **Total during TTS** | **~8-10 GB** |
| After unload | ~0 GB (freed) |
| MusicGen medium (CPU) | ~4-6 GB |
| **Total during music** | **~4-6 GB** |
| System + OS overhead | ~4-6 GB |

This fits comfortably within 36 GB with significant headroom.

### 6.3 Chunking Strategy for Long Meditations

For scripts longer than ~30 seconds of speech, generate sentence-by-sentence (already implemented in the engine above). This prevents:
- Audio hallucinations (Parler losing coherence on long generations)
- Memory spikes from very long token sequences
- Quality drift where the voice changes characteristics mid-generation

After generating all chunks, they're concatenated with appropriate inter-sentence pauses. The silence insertion (from `[pause:Xs]` markers) is handled identically to Kokoro — as numpy zero arrays.

### 6.4 torch.compile Optimization (Optional, Test First)

`torch.compile()` can accelerate Parler TTS 2-4x on supported backends. However, MPS support for torch.compile is still maturing. Implementation:

```python
# In load_model(), AFTER loading the model:
try:
    self.model = torch.compile(self.model, mode="reduce-overhead")
    print("[ParlerTTS] torch.compile applied — expect faster inference")
except Exception as e:
    print(f"[ParlerTTS] torch.compile not available: {e}")
    # Continue without compilation — inference still works, just slower
```

Test thoroughly. If `torch.compile` causes MPS errors, disable it. The model works fine without it.

---

## Part 7: Updated `requirements.txt`

```
# Core
torch>=2.1.0
torchaudio>=2.1.0

# TTS Engines
kokoro>=0.9.4
misaki[en]
git+https://github.com/huggingface/parler-tts.git
transformers>=4.35.0

# Music Generation
audiocraft

# Audio Processing
pedalboard>=0.9.0
numpy
scipy
soundfile
pyloudnorm

# UI
gradio>=4.0
```

### Removed (Fish Speech dependencies):
- ~~`httpx`~~ (if it was only used for Fish Speech API calls)
- ~~`fish-audio-sdk`~~ (or whatever Fish Speech client package was used)
- ~~Any other Fish Speech-specific packages~~

Note: `httpx` may still be needed by other packages (like `gradio`). Only remove it from requirements.txt if no other package depends on it. Check with `pip show httpx` to verify reverse dependencies.

---

## Part 8: Updated `README.md` Section

Replace the Fish Speech section in README.md with:

```markdown
## Speech Engines

MoodScape supports two TTS engines:

### Kokoro TTS (Default)
- Ultra-fast, lightweight (82M parameters)
- Named voice presets (Heart, Sky, Nova, Nicole, etc.)
- Numeric speed control (0.65–1.0)
- Best for: quick generation, consistent results, familiar voices

### Parler TTS (New)
- Description-controlled (2.2B parameters)
- Control voice via natural language: describe gender, pitch, speed, tone, warmth, recording quality
- 34 named speakers for voice consistency (Jon, Lea, etc.)
- 6 meditation-optimized presets + custom description option
- Best for: rich expressive control, unique voices, cinematic quality

Both engines output at 24,000 Hz mono and integrate seamlessly with MusicGen and the Pedalboard FX chain.

### First Run Note
On first use, Parler TTS will download the model (~4.5 GB for Large, ~1.7 GB for Mini).
This only happens once — subsequent runs use the cached model.
```

---

## Part 9: Testing & Verification Checklist

After implementation, verify each of these:

### Fish Speech Removal
- [ ] `grep -ri "fish" --include="*.py" .` returns zero results
- [ ] `grep -ri "fish" --include="*.md" .` returns zero results (except this prompt file)
- [ ] `grep -ri "FISH_API" .` returns zero results
- [ ] No `.py` files with "fish" in the filename exist
- [ ] No dead imports referencing fish_engine, fish_speech, or fish-audio-sdk
- [ ] App starts without errors when Fish Speech is not installed

### Kokoro Unchanged
- [ ] Kokoro generation produces identical output to before the changes
- [ ] All Kokoro voices work (af_heart, af_sky, am_adam, etc.)
- [ ] Kokoro speed parameter works (0.65–1.0)
- [ ] Kokoro MPS fallback to CPU works
- [ ] Voice blending still works if it was supported

### Parler TTS Integration
- [ ] Parler TTS loads and generates audio without errors
- [ ] All 6 meditation presets produce distinct, appropriate voices
- [ ] Custom description input works
- [ ] Speed parameter translates correctly to description keywords
- [ ] Output is 24000 Hz mono float32 (resampled from 44100 Hz)
- [ ] voice_activity mask is correct (True during speech, False during pauses)
- [ ] MPS fallback to CPU works when MPS operations fail
- [ ] Model unloading frees memory before MusicGen loads
- [ ] Large v1 → Mini v1 fallback works if memory is insufficient

### Pipeline Integration
- [ ] Engine selector in UI correctly shows/hides engine-specific controls
- [ ] Kokoro + MusicGen + Pedalboard pipeline works end-to-end
- [ ] Parler + MusicGen + Pedalboard pipeline works end-to-end
- [ ] Ducking works correctly with Parler TTS output
- [ ] LUFS normalization produces correct levels with Parler output
- [ ] Voice FX chain (reverb, compression) sounds good on Parler output
- [ ] Output WAV plays correctly
- [ ] Output MP3 plays correctly

### Edge Cases
- [ ] Empty script produces no crash
- [ ] Very long script (10+ minutes) completes without OOM
- [ ] Script with only pauses and no speech text works
- [ ] Switching between Kokoro and Parler mid-session (different generations) works
- [ ] Progress callback updates correctly during Parler generation
- [ ] Interrupting generation doesn't leave model stuck in memory

---

## Summary of Changes

| Area | Action |
|------|--------|
| `core/fish_engine.py` | **DELETE entirely** |
| Fish Speech config/presets | **DELETE entirely** |
| `core/parler_engine.py` | **CREATE new file** (implementation above) |
| `core/speech_engine.py` | **EDIT** — replace Fish Speech with Parler in factory |
| `core/pipeline.py` | **EDIT** — remove Fish Speech params, add Parler params |
| `app.py` | **EDIT** — remove Fish Speech UI, add Parler UI |
| `requirements.txt` | **EDIT** — remove Fish Speech deps, add Parler deps |
| `README.md` | **EDIT** — remove Fish Speech docs, add Parler docs |
| `.env` / config | **EDIT** — remove FISH_API_KEY |
| `core/kokoro_engine.py` | **NO CHANGES** |
| `core/tts_engine.py` | **NO CHANGES** (if this is the Kokoro file) |
| `core/script_parser.py` | **NO CHANGES** |
| `core/music_engine.py` | **NO CHANGES** |
| `core/audio_processor.py` | **NO CHANGES** |
| `core/mixer.py` | **NO CHANGES** |