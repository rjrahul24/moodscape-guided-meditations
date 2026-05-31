"""IndexTTS-2 engine — wraps IndexTTS2 for zero-shot voice cloning with emotion control.

Voice identity is resolved at construction time via a voice slug that maps to
a speaker reference WAV file in assets/speakers/:

    IndexTTSEngine(voice_slug="calm_meditation")
    # loads: assets/speakers/calm_meditation.wav

Emotion control is applied via an optional emotion reference WAV from
assets/emotions/, or a user-uploaded emotion audio file.

Key settings:
    use_fp16=False       — float32 for MPS stability (prevents NaN errors)
    use_deepspeed=False   — DeepSpeed is CUDA-only, must be disabled on Apple Silicon
    use_cuda_kernel=False — CUDA kernel disabled for MPS compatibility
    generation mode: "free" (uncontrolled) — natural prosodic timing for meditation

Device: MPS on Apple Silicon with CPU fallback for unsupported operations.
         PYTORCH_ENABLE_MPS_FALLBACK=1 is set globally in app.py.
"""

import gc
import logging
import os
import random
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np

from core.speech_engine import SAMPLE_RATE, SpeechEngine
from core.index_tts import voice_registry

logger = logging.getLogger(__name__)

# Checkpoint location (project root / models / indextts2)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CHECKPOINT_DIR = _PROJECT_ROOT / "models" / "indextts2"
_CHECKPOINT_CFG = _CHECKPOINT_DIR / "config.yaml"

# Trailing-silence trimmer threshold and natural decay tail.
# Same approach as F5-TTS engine — IndexTTS-2 allocates output frames
# based on autoregressive decoding and may overshoot with silence.
_TRIM_THRESHOLD_DB = -45.0
_TRIM_TAIL_MS = 50.0

# Default generation speed (1.0 = natural rhythm).
# NOTE: IndexTTS-2 v2 API does NOT expose reliable time-stretching (Issue #422).
# The `speed` arg in synthesize() is accepted for ABC compliance but is a no-op
# on this engine — pacing is controlled by emotion + preprocessor pause durations.
_DEFAULT_SPEED = 1.0

# Silero VAD parameters for IndexTTS-2 output
_VAD_GAIN_FLOOR = 0.15
_VAD_CROP_TAIL_MS = 100.0

# ── Meditation-tuned inference defaults ──────────────────────────────────────
# 8D emotion vector: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
# Pure calm dimension yields a normalized internal value of ~0.5625 (post-bias),
# safely under the API's 0.8 sum-penalty threshold. Deterministic and avoids the
# Qwen3 text-emotion path's known "calm → sad" misclassification.
INDEXTTS_CALM_VECTOR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# Blend ratio: 0.65 = 65% emotion override + 35% speaker timbre preservation.
# Slightly favours the speaker's natural emotion blend for a less synthetic feel.
INDEXTTS_EMO_ALPHA = 0.65

# Sampling: pure stochastic (no beam search) at the trained-for top_p/top_k.
# Temperature lifted just above prior 0.70 to recover prosodic variance the
# previous setting flattened, while staying below the model default of 0.80.
INDEXTTS_TOP_P = 0.80
INDEXTTS_TEMPERATURE = 0.75
INDEXTTS_TOP_K = 30
INDEXTTS_NUM_BEAMS = 1
INDEXTTS_REPETITION_PENALTY = 10.0
INDEXTTS_MAX_MEL_TOKENS = 1815

# API-internal silence between micro-segments. Set to 0 — we add our own 600ms
# room-tone gap plus a 300ms cosine crossfade externally; stacking the model's
# default 200ms hard zero-pad on top produces an audibly long gap.
INDEXTTS_INTERVAL_SILENCE_MS = 0

# Larger window → fewer chunk boundaries → less emotion drift, since IndexTTS-2
# has no context carry-forward between segments. 180 sits in the safe 80-200
# range and stays well under the GPT attention ceiling (~402).
INDEXTTS_MAX_TOKENS_PER_SEG = 180

# Pitch-preserving time-stretch ratio applied per speech chunk before assembly.
# IndexTTS-2 inherits its tempo from the reference clip; common references read
# at conversational pace (~130-150 WPM), well above the slow narration target
# (~95-105 WPM) typical of meditation apps. Set to 1.0 to disable.
INDEXTTS_PACE_RATE = 0.92


def _apply_meditation_pace(audio: np.ndarray, rate: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Pitch-preserving time-stretch on a single IndexTTS-2 speech chunk.

    rate < 1.0 lengthens the audio; rate == 1.0 returns the input unchanged.
    Applied per-chunk so the assembly-time crossfade timing stays accurate.

    Prefers Rubber Band (formant-aware, transparent on voice) via pyrubberband,
    which shells out to the ``rubberband`` CLI. Falls back to librosa's
    phase-vocoder if Rubber Band is unavailable, then to the unmodified input.
    The phase-vocoder can smear/metallicise voice, so Rubber Band is preferred
    for the meditation quality target.
    """
    if abs(rate - 1.0) < 1e-3:
        return audio
    arr = audio.astype(np.float32)
    try:
        import pyrubberband as pyrb
        stretched = pyrb.time_stretch(arr, sr, rate)
        logger.debug("Pacing via Rubber Band (rate=%.2f)", rate)
        return np.asarray(stretched, dtype=np.float32)
    except Exception as e:
        logger.info("Rubber Band unavailable (%s); falling back to librosa phase-vocoder.", e)
    try:
        import librosa
        stretched = librosa.effects.time_stretch(arr, rate=rate)
        logger.debug("Pacing via librosa phase-vocoder (rate=%.2f)", rate)
        return stretched.astype(np.float32)
    except Exception as e:
        logger.warning("Pacing time-stretch failed (rate=%.2f), keeping original: %s", rate, e)
        return audio


def _patch_bigvgan_mps_safety(tts_model) -> None:
    """Wrap BigVGANv2 so mel inputs are clamped and outputs are NaN-scrubbed.

    Two MPS-specific failure modes this guards against:
      1. BigVGANv2 can emit NaN/Inf when mel inputs spike beyond ~|10| on MPS.
      2. torch.clamp(32767 * wav, ...) inside infer_v2.py passes NaN through
         on MPS — silently becoming -32767 clicks in the rendered audio.
    """
    import torch

    bigvgan = getattr(tts_model, "bigvgan", None)
    if bigvgan is None or not hasattr(bigvgan, "forward"):
        logger.warning("IndexTTS-2 BigVGAN MPS safety patch skipped: no bigvgan.forward")
        return

    original_forward = bigvgan.forward

    def safe_forward(mel, *args, **kwargs):
        mel = torch.clamp(mel, -10.0, 10.0)
        wav = original_forward(mel, *args, **kwargs)
        return torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)

    bigvgan.forward = safe_forward
    logger.info("IndexTTS-2 BigVGAN wrapped with MPS NaN-safety patch")


def _trim_trailing_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove trailing silence from an IndexTTS-2 speech chunk.

    Finds the last sample whose absolute amplitude exceeds _TRIM_THRESHOLD_DB,
    then retains a _TRIM_TAIL_MS decay tail so the audio doesn't cut off
    abruptly.
    """
    threshold = 10 ** (_TRIM_THRESHOLD_DB / 20.0)
    active = np.where(np.abs(audio) > threshold)[0]
    if len(active) == 0:
        return audio
    tail = int(_TRIM_TAIL_MS / 1000.0 * sr)
    cut = min(int(active[-1]) + tail + 1, len(audio))
    return audio[:cut]


def _apply_silero_vad(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply Silero VAD to crop trailing non-speech and attenuate interior gaps.

    Same two-pass approach as F5-TTS engine:
      1. Crop — slice after last speech endpoint + safety tail
      2. Attenuate — reduce interior non-speech to 15% for natural ambience
    """
    try:
        import torch
        import torchaudio

        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        (get_speech_timestamps, _, _, _, _) = utils

        vad_sr = 16000
        audio_torch = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        audio_16k = torchaudio.functional.resample(audio_torch, sr, vad_sr).squeeze(0)
        scale = sr / vad_sr

        speech_timestamps = get_speech_timestamps(audio_16k, model, sampling_rate=vad_sr)
        speech_timestamps = [
            {"start": int(ts["start"] * scale), "end": int(ts["end"] * scale)}
            for ts in speech_timestamps
        ]

        if not speech_timestamps:
            return audio

        # Pass 1: Crop trailing non-speech
        last_speech_end = speech_timestamps[-1]['end']
        crop_tail = int(_VAD_CROP_TAIL_MS / 1000.0 * sr)
        crop_idx = min(last_speech_end + crop_tail, len(audio))
        audio = audio[:crop_idx]

        # Pass 2: Attenuate interior non-speech
        mask = np.full(len(audio), _VAD_GAIN_FLOOR, dtype=np.float64)
        fade_samples = int(0.05 * sr)

        for ts in speech_timestamps:
            start, end = ts['start'], min(ts['end'], len(audio))
            s = max(0, start - fade_samples)
            e = min(len(mask), end + fade_samples)
            mask[s:e] = 1.0

        from scipy.ndimage import gaussian_filter1d
        mask = gaussian_filter1d(mask, sigma=fade_samples / 4.0)

        return (audio * mask).astype(np.float32)
    except Exception as e:
        logger.warning("Silero VAD failed, falling back to original audio: %s", e)
        return audio


def _select_device() -> str:
    """Detect optimal device for IndexTTS-2 inference."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS available — use it but with float32 precision to prevent NaN
        return "mps"
    else:
        return "cpu"


class IndexTTSEngine(SpeechEngine):
    """Wraps IndexTTS2 for zero-shot voice cloning meditation narration.

    Implements the SpeechEngine interface — produces mono float32 audio at
    24 000 Hz with a parallel boolean voice-activity mask, matching the
    contract expected by the pipeline's mixing and FX stages.

    The speaker reference voice is resolved once at construction time from
    the VoiceRegistry. Emotion can be changed per-generation via emotion_slug
    or emotion_audio_path parameters.
    """

    def __init__(
        self,
        voice_slug: str | None = None,
        emotion_slug: str | None = None,
        emotion_audio_path: str | None = None,
    ) -> None:
        """Initialise the engine and resolve speaker + emotion assets.

        Args:
            voice_slug: Voice identifier matching a .wav in assets/speakers/.
                If None, the first alphabetically sorted voice is used.
            emotion_slug: Emotion identifier matching a .wav in assets/emotions/.
                If None, no emotion reference is used (model defaults to neutral).
            emotion_audio_path: Direct path to an emotion reference WAV file.
                Overrides emotion_slug if both are provided (used for user-uploaded
                emotion audio from the Gradio UI).

        Raises:
            FileNotFoundError: If voice_slug is given but not registered, or
                if no voices are registered at all and voice_slug is None.
        """
        voices = voice_registry.scan_voices()

        if voice_slug is not None:
            voice_assets = voice_registry.get_voice(voice_slug)
            resolved_slug = voice_slug
        elif voices:
            resolved_slug = sorted(voices.keys())[0]
            voice_assets = voices[resolved_slug]
            logger.info(
                "IndexTTSEngine: no voice_slug given, defaulting to '%s'",
                resolved_slug,
            )
        else:
            raise FileNotFoundError(
                "No IndexTTS-2 voice assets found. "
                "Add 24 kHz mono .wav files (5-10s) to "
                f"'{voice_registry._VOCALS_DIR}'."
            )

        self._voice_slug = resolved_slug
        self._voice_audio_path = str(voice_assets["audio"])

        # Resolve emotion reference
        self._emotion_audio_path: str | None = None
        if emotion_audio_path:
            self._emotion_audio_path = emotion_audio_path
        elif emotion_slug:
            try:
                emotion_assets = voice_registry.get_emotion(emotion_slug)
                self._emotion_audio_path = str(emotion_assets["audio"])
            except FileNotFoundError:
                logger.warning(
                    "Emotion '%s' not found, proceeding without emotion reference",
                    emotion_slug,
                )

        self._model = None
        self._device = None

        logger.info(
            "IndexTTSEngine initialised with voice '%s', emotion=%s",
            self._voice_slug,
            emotion_slug or emotion_audio_path or "none",
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load IndexTTS-2 model.

        Uses float32 precision on MPS to prevent NaN errors. DeepSpeed and
        CUDA kernels are disabled for Apple Silicon compatibility.
        """
        if self._model is not None:
            return

        if not _CHECKPOINT_DIR.is_dir():
            raise FileNotFoundError(
                f"IndexTTS-2 checkpoints not found at '{_CHECKPOINT_DIR}'. "
                f"Download them with:\n"
                f"  huggingface-cli download IndexTeam/IndexTTS-2 "
                f"--local-dir=models/indextts2"
            )

        self._device = _select_device()
        logger.info("Loading IndexTTS-2 on %s (float32)", self._device)

        from indextts.infer_v2 import IndexTTS2

        # Force float32 and disable CUDA-specific features for MPS stability
        self._model = IndexTTS2(
            cfg_path=str(_CHECKPOINT_CFG),
            model_dir=str(_CHECKPOINT_DIR),
            use_fp16=False,      # float32 for MPS NaN safety
            use_deepspeed=False,  # CUDA-only
            use_cuda_kernel=False,  # CUDA-only
        )

        _patch_bigvgan_mps_safety(self._model)

        logger.info("IndexTTS-2 loaded successfully on %s", self._device)

    def unload_model(self) -> None:
        """Release model weights and free device memory."""
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
        logger.info("IndexTTS-2 model unloaded")

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        segments: list[dict],
        voice=None,
        speed: float = _DEFAULT_SPEED,
        progress_cb=None,
        seed: int | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize speech from parsed script segments using voice cloning.

        Implements the chunk-and-stitch pipeline:
          1. For each speech segment, synthesize via IndexTTS2.infer()
          2. Trim trailing silence and apply Silero VAD
          3. Stitch with room-tone gaps and cosine crossfades
          4. Insert programmatic silence for pause segments
          5. Insert breath samples for breath segments

        Args:
            segments:    Parsed segments from index_tts.preprocessor.prepare_segments().
            voice:       Unused (ABC compliance). Voice is fixed at __init__.
            speed:       Speaking speed scalar. IndexTTS-2 uses this in "free" mode.
            progress_cb: Optional callback(current_index, total_segments).
            seed:        Base seed for deterministic generation.
            **kwargs:    Absorbs additional kwargs (emotion_audio_path, etc).

        Returns:
            voice_audio:    float32 mono numpy array at 24 000 Hz.
            voice_activity: bool array of the same length; True where voice is active.
        """
        if self._model is None:
            raise RuntimeError("IndexTTS-2 model not loaded. Call load_model() first.")

        import soundfile as sf

        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        logger.info(
            "IndexTTSEngine synthesize: base seed=%d, %d segments, speed=%.2f",
            seed, len(segments), speed,
        )
        if abs(speed - 1.0) > 1e-3 and not getattr(self, "_speed_warned", False):
            logger.warning(
                "IndexTTS-2 does not support reliable time-stretching (Issue #422); "
                "speed=%.2f will be ignored. Adjust preprocessor pause durations instead.",
                speed,
            )
            self._speed_warned = True

        # Allow per-call emotion override from kwargs
        emotion_audio_path = kwargs.get("emotion_audio_path") or self._emotion_audio_path

        chunks: list[tuple[np.ndarray, np.ndarray, str, str | None]] = []
        total = len(segments)
        speech_idx = 0

        for idx, seg in enumerate(segments):
            if seg["type"] == "speech":
                # Normalise text: collapse whitespace
                gen_text = " ".join(seg["text"].split())

                # ALL_CAPS → lowercase (prevents letter-by-letter G2P)
                gen_text = re.sub(
                    r'\b[A-Z]{2,}\b',
                    lambda m: m.group().lower(),
                    gen_text,
                )

                chunk_seed = seed + speech_idx

                # Synthesize via IndexTTS-2 to a temp file
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, dir=None
                ) as tmp:
                    tmp_path = tmp.name

                try:
                    # Emotion routing: explicit audio reference takes precedence
                    # over the deterministic calm vector preset.
                    use_calm_vector = emotion_audio_path is None

                    infer_kwargs = dict(
                        spk_audio_prompt=self._voice_audio_path,
                        text=gen_text,
                        output_path=tmp_path,
                        emo_alpha=INDEXTTS_EMO_ALPHA,
                        interval_silence=INDEXTTS_INTERVAL_SILENCE_MS,
                        max_text_tokens_per_segment=INDEXTTS_MAX_TOKENS_PER_SEG,
                        top_p=INDEXTTS_TOP_P,
                        top_k=INDEXTTS_TOP_K,
                        temperature=INDEXTTS_TEMPERATURE,
                        num_beams=INDEXTTS_NUM_BEAMS,
                        repetition_penalty=INDEXTTS_REPETITION_PENALTY,
                        max_mel_tokens=INDEXTTS_MAX_MEL_TOKENS,
                        do_sample=True,
                        use_random=False,
                        verbose=False,
                    )
                    if use_calm_vector:
                        infer_kwargs["emo_vector"] = INDEXTTS_CALM_VECTOR
                    else:
                        infer_kwargs["emo_audio_prompt"] = emotion_audio_path

                    self._model.infer(**infer_kwargs)

                    # Read the generated audio
                    arr, file_sr = sf.read(tmp_path, dtype="float32")
                    if arr.ndim > 1:
                        arr = arr.mean(axis=1)  # stereo → mono

                    # Resample to SAMPLE_RATE if needed
                    if file_sr != SAMPLE_RATE:
                        import torchaudio
                        import torch
                        arr_t = torch.from_numpy(arr).unsqueeze(0)
                        arr_t = torchaudio.functional.resample(arr_t, file_sr, SAMPLE_RATE)
                        arr = arr_t.squeeze(0).numpy()

                    arr = arr.astype(np.float32)
                    speech_idx += 1

                    # Per-chunk VRAM hygiene — mitigates IndexTTS-2 issue #364
                    # (BigVGAN + CFM tensors accumulate across sequential generations,
                    # eventually OOM-ing on long meditation sessions).
                    try:
                        import torch
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                        elif torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                    # Post-processing: trim trailing silence + VAD
                    arr = _trim_trailing_silence(arr, SAMPLE_RATE)
                    arr = _apply_silero_vad(arr, SAMPLE_RATE)

                    # Slow to meditation pace (pitch-preserving) before assembly
                    arr = _apply_meditation_pace(arr, INDEXTTS_PACE_RATE, sr=SAMPLE_RATE)

                    # Build voice activity mask
                    threshold = float(np.abs(arr).mean()) * 0.15
                    activity = np.abs(arr) > threshold
                    chunks.append((arr, activity, "speech", gen_text))

                except Exception as e:
                    logger.error(
                        "IndexTTS-2 inference failed for chunk %d: %s",
                        speech_idx, e,
                    )
                    # Insert a short silence as placeholder
                    silence = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
                    chunks.append((silence, np.zeros(len(silence), dtype=bool), "speech", gen_text))
                    speech_idx += 1
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            elif seg["type"] == "pause":
                from core.kokoro_tts.postprocessor import generate_room_tone
                room_tone = generate_room_tone(seg["duration_sec"], sr=SAMPLE_RATE)
                silence_act = np.zeros(len(room_tone), dtype=bool)
                chunks.append((room_tone, silence_act, "pause", None))

            elif seg["type"] == "breath":
                from core.breath_sounds import load_breath
                breath_audio = load_breath(seg["subtype"], target_sr=SAMPLE_RATE)
                breath_act = np.zeros(len(breath_audio), dtype=bool)
                chunks.append((breath_audio, breath_act, "breath", None))

            if progress_cb is not None:
                progress_cb(idx + 1, total)

        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

        # Assemble the final audio with type-aware boundary handling:
        #   speech → speech : 0.6s room tone gap + 300ms cosine crossfade
        #   anything → pause : direct concatenation (preserves exact duration)
        #   pause → anything : direct concatenation
        FADE_N = int(0.300 * SAMPLE_RATE)   # 300ms crossfade
        GAP_N = int(0.6 * SAMPLE_RATE)      # 0.6s gap between speech chunks

        result_audio = chunks[0][0].copy().astype(np.float32)
        result_act = chunks[0][1].copy()

        for i in range(1, len(chunks)):
            prev_type = chunks[i - 1][2]
            cur_audio, cur_act, cur_type, _ = chunks[i]
            cur_audio = cur_audio.astype(np.float32)

            if prev_type == "speech" and cur_type == "speech":
                # Insert room-tone gap between consecutive speech chunks
                from core.kokoro_tts.postprocessor import generate_room_tone
                gap_tone = generate_room_tone(GAP_N / SAMPLE_RATE, sr=SAMPLE_RATE)
                gap_act = np.zeros(len(gap_tone), dtype=bool)
                result_audio = np.concatenate([result_audio, gap_tone])
                result_act = np.concatenate([result_act, gap_act])

                # Equal-power cosine crossfade
                fade = min(FADE_N, len(result_audio), len(cur_audio))
                if fade > 0:
                    _t = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
                    fade_out = np.cos(_t)
                    fade_in = np.sin(_t)
                    overlap = result_audio[-fade:] * fade_out + cur_audio[:fade] * fade_in
                    result_audio = np.concatenate([result_audio[:-fade], overlap, cur_audio[fade:]])
                    result_act = np.concatenate([result_act[:-fade], cur_act[:fade], cur_act[fade:]])
                else:
                    result_audio = np.concatenate([result_audio, cur_audio])
                    result_act = np.concatenate([result_act, cur_act])
            else:
                # Pause boundary — concatenate directly to preserve timing
                result_audio = np.concatenate([result_audio, cur_audio])
                result_act = np.concatenate([result_act, cur_act])

        min_len = min(len(result_audio), len(result_act))
        return result_audio[:min_len].astype(np.float32), result_act[:min_len]

    # ── Voice catalogue ───────────────────────────────────────────────────────

    def get_available_voices(self) -> list[dict]:
        """Return all registered voices from assets/speakers/."""
        voices = voice_registry.scan_voices()
        if not voices:
            return [{"id": "none", "name": "No voices registered",
                     "description": "Add .wav files to assets/speakers/"}]
        return [
            {
                "id": slug,
                "name": slug.replace("_", " ").title(),
                "description": f"Reference: {slug}.wav",
            }
            for slug in sorted(voices.keys())
        ]
