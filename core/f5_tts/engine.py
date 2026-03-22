"""F5-TTS engine — wraps F5TTS for zero-shot voice cloning meditation narration.

Voice identity is resolved at construction time via a voice slug that maps to a
registered asset pair in the VoiceRegistry:

    F5Engine(voice_slug="calm_brittney")
    # loads: core/f5_tts/assets/reference_audio/calm_brittney.wav
    #        core/f5_tts/assets/reference_transcript/calm_brittney.txt

If no slug is given, the engine picks the first available registered voice.
A FileNotFoundError is raised at construction time (not at inference time) if
the slug is invalid or no voices are registered at all.

Key settings:
    nfe_step=32           — production quality; use 16 for fast iteration
    cfg_strength=2.0      — F5-TTS official default; enables classifier-free guidance
    sway_sampling_coef=-1 — enables sway sampling for smoother meditative prosody
    speed=0.88            — meditation pace (~95-100 WPM); natural prosodic timing

Device: MPS on Apple Silicon, CPU fallback elsewhere.
"""

import gc
import logging
import random
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np

from core.speech_engine import SAMPLE_RATE, SpeechEngine
from core.f5_tts import voice_registry

logger = logging.getLogger(__name__)

_NFE_STEPS = 32

# Trailing-silence trimmer threshold and natural decay tail.
# F5-TTS allocates mel frames by a duration formula rather than by speech
# detection, so each chunk typically ends with 100–400 ms of near-zero
# silence. That silence accumulates between our explicit pause chunks and
# inflates pause durations well beyond what the script specifies.
_TRIM_THRESHOLD_DB = -45.0  # samples quieter than this are "silence"
_TRIM_TAIL_MS = 50.0        # keep a 50 ms natural decay tail after last active sample


def _trim_trailing_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove trailing silence from an F5-TTS speech chunk.

    Finds the last sample whose absolute amplitude exceeds _TRIM_THRESHOLD_DB,
    then retains a _TRIM_TAIL_MS decay tail so the audio doesn't cut off
    abruptly. Returns the original array unchanged if no active samples exist
    (shouldn't happen for a speech segment, but safe either way).
    """
    threshold = 10 ** (_TRIM_THRESHOLD_DB / 20.0)
    active = np.where(np.abs(audio) > threshold)[0]
    if len(active) == 0:
        return audio
    tail = int(_TRIM_TAIL_MS / 1000.0 * sr)
    cut = min(int(active[-1]) + tail + 1, len(audio))
    return audio[:cut]


_VAD_GAIN_FLOOR = 0.15  # non-speech segments attenuated to 15% (preserves natural ambience)
_VAD_CROP_TAIL_MS = 100.0  # safety tail after last detected speech before cropping


def _apply_silero_vad(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply Silero VAD to crop trailing non-speech and attenuate interior gaps.

    Two-pass approach:
      1. **Crop** — find the last speech endpoint detected by Silero VAD,
         add a short safety tail (_VAD_CROP_TAIL_MS), and slice the array.
         This removes the seconds of diffusion-generated "room tone" that
         F5-TTS appends after the last spoken word, preventing pause inflation.
      2. **Attenuate** — apply a smooth gain envelope that reduces interior
         non-speech segments to _VAD_GAIN_FLOOR (15%), preserving natural
         breath and resonance between phrases.

    Note: Silero VAD only supports 8000 Hz and multiples of 16000 Hz.  When
    the pipeline runs at 24000 Hz we resample to 16000 Hz for VAD inference,
    then scale the returned sample indices back to the original rate.
    """
    try:
        import torch
        import torchaudio
        # Only loads the model once per session via torch.hub's cache
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (get_speech_timestamps, _, _, _, _) = utils

        # Silero VAD requires 8000 or 16000 Hz (or multiples of 16000).
        # Resample to 16000 Hz for VAD, then scale timestamps back to `sr`.
        vad_sr = 16000
        audio_torch = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, T)
        audio_16k = torchaudio.functional.resample(audio_torch, sr, vad_sr).squeeze(0)  # (T')
        scale = sr / vad_sr  # factor to convert 16 kHz indices → original sr

        speech_timestamps = get_speech_timestamps(audio_16k, model, sampling_rate=vad_sr)
        # Scale indices back to the original sample rate
        speech_timestamps = [
            {"start": int(ts["start"] * scale), "end": int(ts["end"] * scale)}
            for ts in speech_timestamps
        ]

        if not speech_timestamps:
            # No speech detected — return as-is (shouldn't happen for speech chunks)
            return audio

        # ── Pass 1: Crop trailing non-speech ──────────────────────────────
        last_speech_end = speech_timestamps[-1]['end']
        crop_tail = int(_VAD_CROP_TAIL_MS / 1000.0 * sr)
        crop_idx = min(last_speech_end + crop_tail, len(audio))
        audio = audio[:crop_idx]

        # ── Pass 2: Attenuate interior non-speech ─────────────────────────
        mask = np.full(len(audio), _VAD_GAIN_FLOOR, dtype=np.float64)
        fade_samples = int(0.05 * sr)  # 50ms fade for smooth transitions

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



_DEFAULT_SPEED = 0.88  # meditation pace (~95-100 WPM); lets model assign natural prosodic timing
_DEFAULT_TARGET_WPM = None  # None = natural rhythm via speed param (recommended); set 90-150 for fixed WPM pacing
_SWAY_COEF = -1.0   # enables sway sampling for smoother prosody
_CFG_STRENGTH = 2.0  # F5-TTS official default; split-band de-esser handles HF diffusion artifacts

# Reference audio conditioning targets
_REF_TARGET_DBFS = -20.0  # RMS normalisation target for reference audio


def _condition_reference_audio(audio_path: str, sr: int) -> str:
    """Condition reference audio for optimal F5-TTS alignment.

    RMS-normalises the reference to _REF_TARGET_DBFS so the model receives
    a consistent input level regardless of the original recording gain.

    Returns a new temp WAV path with the conditioned audio (caller must clean up).
    """
    import soundfile as sf

    audio, file_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # RMS normalise to target level
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms > 1e-8:
        target_rms = 10 ** (_REF_TARGET_DBFS / 20.0)
        audio = audio * (target_rms / rms)

    # Write conditioned audio to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_conditioned.wav")
    tmp.close()
    sf.write(tmp.name, audio.astype(np.float32), file_sr, subtype="PCM_16")
    logger.debug(
        "Conditioned reference audio: RMS→%.1f dBFS → %s",
        _REF_TARGET_DBFS, tmp.name,
    )
    return tmp.name



class F5Engine(SpeechEngine):
    """Wraps F5TTS for zero-shot voice cloning meditation narration.

    Implements the SpeechEngine interface — produces mono float32 audio at
    24 000 Hz with a parallel boolean voice-activity mask, matching the
    contract expected by the pipeline's mixing and FX stages.

    The reference voice is resolved once at construction time from the
    VoiceRegistry, not at each synthesize() call. This ensures fast inference
    and a single clear error location if asset files are missing.
    """

    def __init__(self, voice_slug: str | None = None) -> None:
        """Initialise the engine and resolve the reference voice assets.

        Args:
            voice_slug: Voice identifier matching a registered asset pair,
                e.g. "calm_brittney". If None, the first alphabetically
                sorted registered voice is used.

        Raises:
            FileNotFoundError: If voice_slug is given but not registered, or
                if no voices are registered at all and voice_slug is None.
        """
        registry = voice_registry.scan()

        if voice_slug is not None:
            # Explicit slug — raises FileNotFoundError if pair is incomplete.
            paths = voice_registry.get_voice(voice_slug)
            resolved_slug = voice_slug
        elif registry:
            # No slug given — pick first available voice (alphabetical order).
            resolved_slug = sorted(registry.keys())[0]
            paths = registry[resolved_slug]
            logger.info(
                "F5Engine: no voice_slug given, defaulting to first registered voice: '%s'",
                resolved_slug,
            )
        else:
            raise FileNotFoundError(
                "No F5-TTS voice assets found. "
                "Add 24 kHz mono .wav files to "
                "core/f5_tts/assets/reference_audio/ and matching verbatim "
                ".txt transcripts to core/f5_tts/assets/reference_transcript/ "
                "to register at least one voice."
            )

        self._voice_slug = resolved_slug
        self._phases = voice_registry.get_voice(resolved_slug)
        self._phase_assets: dict[str, dict] = {}  # loaded assets (path, text) per phase
        self._model = None

        logger.info(
            "F5Engine initialised with voice '%s' (%d phases)",
            self._voice_slug,
            len(self._phases),
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load F5TTS model onto MPS (Apple Silicon) or CPU.

        Also pre-processes the reference audio with Whisper so that ref_text
        exactly matches the ≤12 s clip F5-TTS uses internally.  A mismatch
        between ref_text length and the clipped audio duration is the primary
        cause of reference words bleeding into generated output.  Passing
        ref_text="" lets F5-TTS transcribe the clipped audio with Whisper and
        cache the result — the Whisper call only runs once per session.
        """
        if self._model is not None:
            return

        import torch
        from f5_tts.api import F5TTS
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading F5TTS (F5TTS_v1_Base) on %s", device)
        # Suppress harmless torch/vocos STFT tensor-resize deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="An output with one or more elements was resized",
                category=UserWarning,
            )
            self._model = F5TTS(model="F5TTS_v1_Base", device=device)
            # Force fp16 precision to prevent distortion artifacts often seen in bf16
            # or numerical instabilities in fp32 on certain hardware.
            self._model.ema_model.to(torch.float16)

        # Pre-process ALL reference phases
        for phase_name, assets in self._phases.items():
            ref_audio_path = str(assets["audio"])
            ref_text_raw = assets["transcript"]
            
            if isinstance(ref_text_raw, Path):
                ref_text = ref_text_raw.read_text(encoding="utf-8").strip()
            else:
                ref_text = ref_text_raw

            logger.info("Pre-processing phase '%s' (audio: %s)", phase_name, ref_audio_path)
            
            # Pass ref_text="" if it's empty to let Whisper transcribe
            proc_audio_path, proc_ref_text = preprocess_ref_audio_text(
                ref_audio_path,
                ref_text if ref_text else "",
                show_info=lambda msg: logger.debug("F5 preprocess: %s", msg),
            )
            # Condition the processed reference: RMS-normalise and append
            # trailing silence to prevent phrase leakage into generated output.
            conditioned_path = _condition_reference_audio(proc_audio_path, SAMPLE_RATE)
            # Measure reference audio duration for fix_duration calculation
            import soundfile as _sf
            _ref_info = _sf.info(conditioned_path)
            ref_duration_sec = _ref_info.duration
            self._phase_assets[phase_name] = {
                "audio": conditioned_path,
                "text": proc_ref_text,
                "duration_sec": ref_duration_sec,
            }
            logger.info("Phase '%s' processed. Transcript: %r", phase_name, proc_ref_text[:80])

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

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        segments: list[dict],
        voice=None,
        speed: float = _DEFAULT_SPEED,
        progress_cb=None,
        seed: int | None = None,
        target_wpm: int | None = _DEFAULT_TARGET_WPM,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize speech from parsed script segments using voice cloning.

        The reference voice is taken from the phase assets resolved at
        construction time from the VoiceRegistry. The `voice` parameter is
        accepted for SpeechEngine ABC compliance but is not used — voice
        identity is fixed at __init__ via voice_slug.

        Args:
            segments:    Parsed segments from f5_tts.preprocessor.prepare_segments().
                         Each dict has "type" ("speech"/"pause") and either "text"
                         or "duration_sec".
            voice:       Unused (ABC compliance only). Voice is fixed at init.
            speed:       Speaking speed scalar (0.7–1.2). Used as fallback when
                         target_wpm is None.
            progress_cb: Optional callback(current_index, total_segments).
            seed:        Base seed for deterministic generation. Each chunk uses
                         seed + chunk_index so output is reproducible yet each
                         chunk starts from distinct noise.
            target_wpm:  Target words-per-minute for pacing. When set, overrides
                         the speed parameter by calculating fix_duration per chunk
                         based on word count. 110 WPM = meditation pace. Set to
                         None to fall back to speed-based duration estimation.
            **kwargs:    Absorbs additional engine-specific kwargs passed by the
                         pipeline.

        Returns:
            voice_audio:    float32 mono numpy array at 24 000 Hz.
            voice_activity: bool array of the same length; True where voice is active.
        """
        if self._model is None:
            raise RuntimeError("F5TTS model not loaded. Call load_model() first.")

        # Establish a base seed for reproducible generation across all chunks.
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        logger.info("F5Engine synthesize: base seed=%d, %d segments", seed, len(segments))

        chunks: list[tuple[np.ndarray, np.ndarray, str, str | None]] = []
        total = len(segments)
        speech_idx = 0  # counter for speech chunks only (used for per-chunk seed)

        for idx, seg in enumerate(segments):
            if seg["type"] == "speech":
                # Normalise text: collapse newlines and runs of whitespace so
                # F5-TTS's chunk_text() sees clean prose, not embedded markers.
                gen_text = " ".join(seg["text"].split())

                # ALL_CAPS words risk being spelled letter-by-letter by F5's G2P
                # (trained on natural-case transcriptions). Lower-case them only,
                # leaving sentence-case and proper nouns untouched.
                gen_text = re.sub(
                    r'\b[A-Z]{2,}\b',
                    lambda m: m.group().lower(),
                    gen_text,
                )

                # Select correct reference phase
                phase = seg.get("voice") or "default"
                if phase not in self._phase_assets:
                    if "default" in self._phase_assets:
                        phase = "default"
                    else:
                        phase = next(iter(self._phase_assets.keys()))

                assets = self._phase_assets[phase]

                # Per-chunk seed: base_seed + speech_index gives each chunk
                # distinct but reproducible diffusion noise.
                chunk_seed = seed + speech_idx

                # Calculate fix_duration for WPM-based pacing.
                # fix_duration tells F5-TTS the TOTAL mel frame count (ref + gen).
                # After generation, the reference portion is clipped off, leaving
                # exactly target_speech_sec of generated audio.
                infer_kwargs: dict = dict(
                    ref_file=assets["audio"],
                    ref_text=assets["text"],
                    gen_text=gen_text,
                    speed=speed,
                    nfe_step=_NFE_STEPS,
                    cfg_strength=_CFG_STRENGTH,
                    sway_sampling_coef=_SWAY_COEF,
                    remove_silence=False,
                    seed=chunk_seed,
                )
                if target_wpm is not None and target_wpm > 0:
                    word_count = len(gen_text.split())
                    target_speech_sec = word_count / target_wpm * 60.0
                    ref_dur = assets.get("duration_sec", 10.0)
                    infer_kwargs["fix_duration"] = ref_dur + target_speech_sec

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="An output with one or more elements was resized",
                        category=UserWarning,
                    )
                    wav, _sr, _ = self._model.infer(**infer_kwargs)
                speech_idx += 1
                # f5-tts ≥1.1 returns a numpy array; older builds return a tensor
                if hasattr(wav, "cpu"):
                    arr = wav.cpu().squeeze().numpy().astype(np.float32)
                else:
                    arr = np.asarray(wav, dtype=np.float32).squeeze()
                arr = _trim_trailing_silence(arr, SAMPLE_RATE)
                threshold = float(np.abs(arr).mean()) * 0.15
                activity = np.abs(arr) > threshold
                chunks.append((arr, activity, "speech", gen_text))

            elif seg["type"] == "pause":
                from core.kokoro_tts.postprocessor import generate_room_tone
                n = int(seg["duration_sec"] * SAMPLE_RATE)
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

        # Apply Silero VAD to each speech chunk before assembly
        processed_chunks = []
        for arr, act, ctype, text in chunks:
            if ctype == "speech":
                arr = _apply_silero_vad(arr, SAMPLE_RATE)
            processed_chunks.append((arr, act, ctype, text))
        chunks = processed_chunks

        # Assemble the final audio with type-aware boundary handling:
        #   speech → speech : 0.8s room tone gap + 300ms equal-power cosine crossfade
        #   anything → pause : direct concatenation (preserves exact duration)
        #   pause → anything : direct concatenation (preserves exact duration)
        # Consecutive speech chunks only arise from splitting the same paragraph
        # at the 400-char limit — they are sentences within ONE paragraph.
        # Explicit pause segments from the preprocessor handle paragraph breaks.
        FADE_N = int(0.300 * SAMPLE_RATE)  # 300 ms equal-power cosine crossfade between speech chunks
        GAP_N = int(0.4 * SAMPLE_RATE)     # 0.4s gap between consecutive speech chunks (same paragraph)

        result_audio = chunks[0][0].copy().astype(np.float32)
        result_act = chunks[0][1].copy()

        for i in range(1, len(chunks)):
            prev_type = chunks[i - 1][2]
            cur_audio, cur_act, cur_type, _ = chunks[i]
            cur_audio = cur_audio.astype(np.float32)

            if prev_type == "speech" and cur_type == "speech":
                gap_tone = np.zeros(GAP_N, dtype=np.float32)
                gap_act = np.zeros(GAP_N, dtype=bool)
                
                result_audio = np.concatenate([result_audio, gap_tone])
                result_act = np.concatenate([result_act, gap_act])

                # Equal-power cosine crossfade to smooth the sentence boundary.
                # cos/sin pair maintains cos²+sin²=1 at every point, eliminating
                # the amplitude dip that linear crossfades produce at the midpoint.
                fade = min(FADE_N, len(result_audio), len(cur_audio))
                if fade > 0:
                    _t = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
                    fade_out = np.cos(_t)   # 1.0 → 0.0
                    fade_in  = np.sin(_t)   # 0.0 → 1.0
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
        """Return all registered voices from the VoiceRegistry."""
        registry = voice_registry.scan()
        if not registry:
            return [{"id": "none", "name": "No voices registered",
                     "description": "Add .wav + .txt pairs to core/f5_tts/assets/"}]
        return [
            {
                "id": slug,
                "name": slug.replace("_", " ").title(),
                "description": f"Reference: {slug}.wav",
            }
            for slug in sorted(registry.keys())
        ]
