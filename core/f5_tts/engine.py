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
    sway_sampling_coef=-1 — enables sway sampling for smoother meditative prosody
    speed=0.75            — meditation-ideal pace; Kokoro default is 0.70

Device: MPS on Apple Silicon, CPU fallback elsewhere.
"""

import gc
import logging
import os
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


def _apply_silero_vad(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply Silero VAD to generate a smooth probability-based gain envelope.

    Instead of hard energy-threshold cutting, this uses Silero-VAD's probability
    scores to create a gain mask. This preserves natural breath and decay while
    aggressively suppressing background noise in non-speech segments.
    """
    try:
        import torch
        # Only loads the model once per session via torch.hub's cache
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (get_speech_timestamps, _, _, _, _) = utils

        # Silero VAD expects 16kHz for its internal processing
        audio_torch = torch.from_numpy(audio.astype(np.float32))
        
        # Get timestamps for speech segments
        speech_timestamps = get_speech_timestamps(audio_torch, model, sampling_rate=sr)
        
        # Create a smooth gain mask (initialized to 0)
        mask = np.zeros_like(audio)
        fade_samples = int(0.05 * sr)  # 50ms fade for smooth transitions

        for ts in speech_timestamps:
            start, end = ts['start'], ts['end']
            # Apply gain of 1.0 to detected speech segments with soft fades
            s = max(0, start - fade_samples)
            e = min(len(mask), end + fade_samples)
            mask[s:e] = 1.0

        # Apply gaussian smoothing to the mask to avoid clicks
        from scipy.ndimage import gaussian_filter1d
        mask = gaussian_filter1d(mask.astype(float), sigma=fade_samples/4.0)

        return audio * mask
    except Exception as e:
        logger.warning("Silero VAD failed, falling back to original audio: %s", e)
        return audio


def _last_sentence(text: str) -> str:
    """Return the last sentence of text for use as a chained ref_text.

    Uses the same sentence-boundary regex as the preprocessor so the split
    points are consistent. Falls back to the full text for single-sentence chunks.
    """
    sents = re.split(r'(?<=[.!?…])\s+', text.strip())
    return sents[-1] if sents else text


def _write_chain_ref(arr: np.ndarray, sr: int) -> str:
    """Write float32 mono audio to a temp WAV file and return its path.

    soundfile is already a project dependency (imported in mixer.py).
    The caller is responsible for deleting the file when done.
    """
    import soundfile as sf  # lazy import to avoid cost when F5 is not in use
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    sf.write(tmp.name, arr.astype(np.float32), sr, subtype="PCM_16")
    return tmp.name


_SWAY_COEF = -1.0   # enables sway sampling for smoother prosody
_CFG_STRENGTH = 2.0  # paper-validated optimal for stable generation
_ROOM_TONE_LEVEL = 1e-3  # ~-60 dBFS ambient floor for pause segments

# Reference audio conditioning targets
_REF_TARGET_DBFS = -20.0      # RMS normalisation target for reference audio
_REF_TAIL_SILENCE_SEC = 1.0   # trailing silence to prevent phrase leakage
_REF_TAIL_NOISE_DBFS = -55.0  # low-level noise (~-55 dBFS) survives F5's -42 dBFS edge trimmer


def _condition_reference_audio(audio_path: str, sr: int) -> str:
    """Condition reference audio for optimal F5-TTS alignment.

    Addresses two common sources of quality degradation:
      1. Inconsistent reference levels → normalise RMS to _REF_TARGET_DBFS
      2. Phrase leakage (reference words bleeding into output) → append
         _REF_TAIL_SILENCE_SEC of low-level noise as trailing silence

    The trailing noise is at ~-55 dBFS — above F5's internal -42 dBFS edge
    trimmer threshold so it survives preprocessing, but inaudible in practice.

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

    # Append trailing silence (low-level noise to survive edge trimming)
    tail_amplitude = 10 ** (_REF_TAIL_NOISE_DBFS / 20.0)
    tail_samples = int(_REF_TAIL_SILENCE_SEC * file_sr)
    tail = (np.random.randn(tail_samples) * tail_amplitude).astype(np.float32)
    audio = np.concatenate([audio, tail])

    # Write conditioned audio to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_conditioned.wav")
    tmp.close()
    sf.write(tmp.name, audio.astype(np.float32), file_sr, subtype="PCM_16")
    logger.debug(
        "Conditioned reference audio: RMS→%.1f dBFS, +%.1fs tail → %s",
        _REF_TARGET_DBFS, _REF_TAIL_SILENCE_SEC, tmp.name,
    )
    return tmp.name


def _normalize_chunk_wpm(
    chunks: list[tuple],
    sr: int,
    min_words: int = 5,
    max_stretch: float = 0.15,
) -> list[tuple]:
    """Normalize speaking rate across speech chunks toward the session median WPM.

    F5-TTS produces inconsistent pacing across chunks even with a fixed speed=
    scalar — some phrases are synthesised faster or slower due to the duration
    predictor's per-phrase variance. This pass measures each eligible chunk's
    actual words-per-minute and applies librosa.effects.time_stretch to bring
    outliers within ±max_stretch of the session median.

    Chunks with fewer than min_words words are excluded from measurement and
    adjustment (WPM estimates are unreliable for short phrases). Stretch ratio
    is clamped to [1−max_stretch, 1+max_stretch] to limit phase-vocoder
    artefacts — at ±15% the smearing is inaudible in meditation audio.

    This is a secondary fine-adjustment pass only. Primary speed control is
    still handled by F5's duration predictor via the speed= parameter in
    infer(), which avoids large-ratio time-stretching entirely.
    """
    import librosa

    wpm_list: list[float] = []
    for arr, _, ctype, text in chunks:
        if ctype != "speech" or not text:
            continue
        words = len(text.split())
        dur = len(arr) / sr
        if words < min_words or dur < 0.3:
            continue
        wpm_list.append((words / dur) * 60.0)

    if len(wpm_list) < 2:
        return chunks  # not enough eligible chunks — skip normalisation

    target_wpm = float(np.median(wpm_list))
    logger.debug(
        "WPM normalisation: median=%.1f WPM from %d eligible chunks",
        target_wpm, len(wpm_list),
    )

    result: list[tuple] = []
    for arr, act, ctype, text in chunks:
        if ctype != "speech" or not text:
            result.append((arr, act, ctype, text))
            continue
        words = len(text.split())
        dur = len(arr) / sr
        if words < min_words or dur < 0.3:
            result.append((arr, act, ctype, text))
            continue

        chunk_wpm = (words / dur) * 60.0
        # rate > 1.0 speeds up the audio; rate < 1.0 slows it down.
        # To match target_wpm from chunk_wpm: rate = target / chunk.
        rate = float(np.clip(
            target_wpm / chunk_wpm,
            1.0 - max_stretch,
            1.0 + max_stretch,
        ))

        if abs(rate - 1.0) < 0.02:
            result.append((arr, act, ctype, text))
            continue

        stretched = librosa.effects.time_stretch(arr, rate=rate).astype(np.float32)
        threshold = float(np.abs(stretched).mean()) * 0.15
        new_act = np.abs(stretched) > threshold
        result.append((stretched, new_act, ctype, text))
        logger.debug(
            "  chunk WPM %.1f → stretched %.2fx to match median %.1f WPM",
            chunk_wpm, rate, target_wpm,
        )

    return result


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
        self._chain_tmp_paths: list[str] = []       # temp WAV files for chained refs
        self._chain_cleanup_registered: bool = False

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
            self._phase_assets[phase_name] = {
                "audio": conditioned_path,
                "text": proc_ref_text
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

    def _cleanup_chain_tmps(self) -> None:
        """Delete all chained reference temp WAV files created during synthesis."""
        for p in self._chain_tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        self._chain_tmp_paths.clear()

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        segments: list[dict],
        voice=None,
        speed: float = 0.90,
        progress_cb=None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize speech from parsed script segments using voice cloning.

        The reference voice is taken from self._ref_audio_path and
        self._ref_text, resolved at construction time from the VoiceRegistry.
        The `voice` parameter is accepted for SpeechEngine ABC compliance but
        is not used — voice identity is fixed at __init__ via voice_slug.

        Args:
            segments:    Parsed segments from f5_tts.preprocessor.prepare_segments().
                         Each dict has "type" ("speech"/"pause") and either "text"
                         or "duration_sec".
            voice:       Unused (ABC compliance only). Voice is fixed at init.
            speed:       Speaking speed scalar (0.5–1.0). 1.0 is default/natural.
                         Recommended 0.85–1.0 when using post-processing stretch.
            progress_cb: Optional callback(current_index, total_segments).
            **kwargs:    Absorbs engine-specific kwargs (e.g. seed=) passed by the
                         pipeline that are not applicable to F5-TTS.

        Returns:
            voice_audio:    float32 mono numpy array at 24 000 Hz.
            voice_activity: bool array of the same length; True where voice is active.
        """
        if self._model is None:
            raise RuntimeError("F5TTS model not loaded. Call load_model() first.")

        # Each entry is (audio_array, activity_array, chunk_type, text_or_None).
        # text is stored for the WPM normalisation pass after synthesis.
        chunks: list[tuple[np.ndarray, np.ndarray, str, str | None]] = []
        total = len(segments)

        # ── Chained reference state ────────────────────────────────────────────
        # Instead of using the same static reference for every chunk, each speech
        # chunk (after the first) seeds the model from the tail of the previous
        # chunk's generated audio. This gives the model acoustic context from the
        # immediately preceding speech, creating natural pitch and energy continuity
        # across sentence boundaries. The static reference still serves as the
        # initial seed and is restored after pauses, phase changes, and every
        # _CHAIN_RESET_EVERY chunks to prevent voice drift over long sessions.
        _CHAIN_RESET_EVERY = 6          # reset to static ref every N consecutive speech chunks
        _CHAIN_RESET_PAUSE_SEC = 3.0    # only reset chain on pauses >= this duration
        _CHAIN_MAX_REF_SEC = 5.0        # maximum seconds to take as chained ref audio
        _CHAIN_MAX_REF_FRAC = 0.50      # also cap at 50% of the chunk length (avoids very long refs)
        _CHAIN_MIN_SAMPLES = int(0.5 * SAMPLE_RATE)  # skip chaining from chunks shorter than 0.5s

        chain_ref_audio: str | None = None  # path to current chained ref WAV (or None → use static)
        chain_ref_text: str | None = None   # last sentence of previous chunk (ref_text for next)
        chain_speech_count: int = 0         # consecutive speech chunks since last reset
        last_phase: str | None = None       # track phase changes for reset trigger
        prev_was_pause: bool = False        # True if the immediately preceding segment was a long pause
        prev_pause_duration: float = 0.0    # duration of the preceding pause (for reset threshold)

        # Register atexit cleanup once per engine instance so temp files are
        # deleted on process exit even if synthesize() is interrupted.
        if not self._chain_cleanup_registered:
            import atexit
            atexit.register(self._cleanup_chain_tmps)
            self._chain_cleanup_registered = True

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

                # Trailing ellipsis cues F5's duration predictor to generate a
                # natural decay tail on the last syllable rather than cutting
                # off abruptly. Has no effect on sentences that already trail
                # with '?', '!', or '…'.
                if not gen_text.endswith(('?', '!', '...')):
                    gen_text = gen_text.rstrip('.,') + '...'

                # Select correct reference phase
                phase = seg.get("voice") or "default"
                if phase not in self._phase_assets:
                    if "default" in self._phase_assets:
                        phase = "default"
                    else:
                        # Pick first available if default missing
                        phase = next(iter(self._phase_assets.keys()))

                assets = self._phase_assets[phase]

                # ── Chained reference selection ────────────────────────────
                # Reset to static reference when:
                #   (a) first chunk of the session (chain_ref_audio is None)
                #   (b) a LONG pause (>= _CHAIN_RESET_PAUSE_SEC) preceded this chunk
                #       Short pauses and breath segments maintain the chain for
                #       prosodic continuity through natural breathing pauses.
                #   (c) voice phase just changed
                #   (d) _CHAIN_RESET_EVERY consecutive speech chunks accumulated
                #       (prevents voice drift over long sessions)
                should_reset = (
                    chain_ref_audio is None
                    or (prev_was_pause and prev_pause_duration >= _CHAIN_RESET_PAUSE_SEC)
                    or phase != last_phase
                    or chain_speech_count >= _CHAIN_RESET_EVERY
                )
                if should_reset:
                    self._cleanup_chain_tmps()
                    chain_ref_audio = None
                    chain_ref_text = None
                    chain_speech_count = 0

                infer_ref_file = chain_ref_audio if chain_ref_audio is not None else assets["audio"]
                infer_ref_text = chain_ref_text  if chain_ref_text  is not None else assets["text"]

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="An output with one or more elements was resized",
                        category=UserWarning,
                    )
                    wav, _sr, _ = self._model.infer(
                        ref_file=infer_ref_file,
                        ref_text=infer_ref_text,
                        gen_text=gen_text,
                        # Pass speed directly to F5's duration predictor so
                        # the model generates the right number of mel frames
                        # for the target pace. This produces more natural
                        # timing than generating at 1.0 and time-stretching
                        # afterwards (which introduces large-ratio phase-vocoder
                        # smearing). WPM normalisation below handles only the
                        # small residual variance (±15%) that the predictor
                        # leaves between phrases.
                        speed=speed,
                        nfe_step=_NFE_STEPS,
                        cfg_strength=_CFG_STRENGTH,
                        sway_sampling_coef=_SWAY_COEF,
                        remove_silence=False,
                    )
                # f5-tts ≥1.1 returns a numpy array; older builds return a tensor
                if hasattr(wav, "cpu"):
                    arr = wav.cpu().squeeze().numpy().astype(np.float32)
                else:
                    arr = np.asarray(wav, dtype=np.float32).squeeze()
                # Remove trailing silence so explicit pause chunks are not
                # padded by the model's over-allocated mel frames.
                arr = _trim_trailing_silence(arr, SAMPLE_RATE)
                # Energy-threshold voice activity: active where amplitude > 15% of mean
                threshold = float(np.abs(arr).mean()) * 0.15
                activity = np.abs(arr) > threshold
                chunks.append((arr, activity, "speech", gen_text))

                # ── Update chain state for next chunk ──────────────────────
                chain_speech_count += 1
                last_phase = phase
                max_ref_samples = min(
                    int(_CHAIN_MAX_REF_SEC * SAMPLE_RATE),
                    int(len(arr) * _CHAIN_MAX_REF_FRAC),
                )
                if len(arr) >= _CHAIN_MIN_SAMPLES and max_ref_samples > 0:
                    ref_arr = arr[-max_ref_samples:]
                    # Delete the previous chain temp file before writing the new one
                    if chain_ref_audio is not None:
                        try:
                            os.unlink(chain_ref_audio)
                        except OSError:
                            pass
                        if chain_ref_audio in self._chain_tmp_paths:
                            self._chain_tmp_paths.remove(chain_ref_audio)
                    new_path = _write_chain_ref(ref_arr, SAMPLE_RATE)
                    self._chain_tmp_paths.append(new_path)
                    chain_ref_audio = new_path
                    chain_ref_text = _last_sentence(gen_text)
                    logger.debug(
                        "Chain ref updated: %d samples (%.2fs) → %s",
                        len(ref_arr), len(ref_arr) / SAMPLE_RATE, new_path,
                    )
                # (else: chunk too short to be a useful reference — keep previous intact)

            elif seg["type"] == "pause":
                n = int(seg["duration_sec"] * SAMPLE_RATE)
                # Very-low-level room tone instead of digital zeros: prevents the
                # unnatural "dead air" artefact that digital silence produces in
                # meditation audio, and keeps the reverb tail active through pauses.
                silence = (np.random.randn(n) * _ROOM_TONE_LEVEL).astype(np.float32)
                silence_act = np.zeros(n, dtype=bool)
                chunks.append((silence, silence_act, "pause", None))

            elif seg["type"] == "breath":
                from core.breath_sounds import load_breath

                breath_audio = load_breath(seg["subtype"], target_sr=SAMPLE_RATE)
                # Blend with room tone to match F5's convention of no digital silence
                room = (np.random.randn(len(breath_audio)) * _ROOM_TONE_LEVEL).astype(
                    np.float32
                )
                breath_audio = breath_audio + room
                breath_act = np.zeros(len(breath_audio), dtype=bool)
                chunks.append((breath_audio, breath_act, "breath", None))

            if seg["type"] == "pause":
                prev_was_pause = True
                prev_pause_duration = seg["duration_sec"]
            elif seg["type"] == "breath":
                prev_was_pause = True
                prev_pause_duration = 1.2  # nominal breath duration (below reset threshold)
            else:
                prev_was_pause = False
                prev_pause_duration = 0.0

            if progress_cb is not None:
                progress_cb(idx + 1, total)

        # Release all chain temp files now rather than waiting for process exit.
        self._cleanup_chain_tmps()

        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

        # WPM normalisation: smooth out F5's inter-chunk pacing variance by
        # time-stretching each eligible chunk toward the session median WPM.
        # Limited to ±15% to keep phase-vocoder artefacts inaudible.
        chunks = _normalize_chunk_wpm(chunks, SAMPLE_RATE)

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
        GAP_N = int(0.8 * SAMPLE_RATE)     # 0.8s gap between consecutive speech chunks (same paragraph)

        result_audio = chunks[0][0].copy().astype(np.float32)
        result_act = chunks[0][1].copy()

        for i in range(1, len(chunks)):
            prev_type = chunks[i - 1][2]
            cur_audio, cur_act, cur_type, _ = chunks[i]
            cur_audio = cur_audio.astype(np.float32)

            if prev_type == "speech" and cur_type == "speech":
                # Insert a 2.5s room tone gap between consecutive speech chunks
                gap_tone = (np.random.randn(GAP_N) * _ROOM_TONE_LEVEL).astype(np.float32)
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
