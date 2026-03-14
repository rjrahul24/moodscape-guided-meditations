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
import re
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
_SWAY_COEF = -1.0   # enables sway sampling for smoother prosody
_CFG_STRENGTH = 2.0  # paper-validated optimal for stable generation
_ROOM_TONE_LEVEL = 1e-3  # ~-60 dBFS ambient floor for pause segments


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
            self._phase_assets[phase_name] = {
                "audio": proc_audio_path,
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

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        segments: list[dict],
        voice=None,
        speed: float = 0.75,
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

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="An output with one or more elements was resized",
                        category=UserWarning,
                    )
                    wav, _sr, _ = self._model.infer(
                        ref_file=assets["audio"],
                        ref_text=assets["text"],
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

            elif seg["type"] == "pause":
                n = int(seg["duration_sec"] * SAMPLE_RATE)
                # Very-low-level room tone instead of digital zeros: prevents the
                # unnatural "dead air" artefact that digital silence produces in
                # meditation audio, and keeps the reverb tail active through pauses.
                silence = (np.random.randn(n) * _ROOM_TONE_LEVEL).astype(np.float32)
                silence_act = np.zeros(n, dtype=bool)
                chunks.append((silence, silence_act, "pause", None))

            if progress_cb is not None:
                progress_cb(idx + 1, total)

        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

        # WPM normalisation: smooth out F5's inter-chunk pacing variance by
        # time-stretching each eligible chunk toward the session median WPM.
        # Limited to ±15% to keep phase-vocoder artefacts inaudible.
        chunks = _normalize_chunk_wpm(chunks, SAMPLE_RATE)

        # Assemble the final audio with type-aware boundary handling:
        #   speech → speech : 20 ms crossfade (smooths chunk-cut artifacts)
        #   anything → pause : direct concatenation (preserves exact duration)
        #   pause → anything : direct concatenation (preserves exact duration)
        # This prevents the crossfade overlap from eating into pause timing.
        FADE_N = int(0.020 * SAMPLE_RATE)  # 20 ms crossfade between speech chunks

        result_audio = chunks[0][0].copy().astype(np.float32)
        result_act = chunks[0][1].copy()

        for i in range(1, len(chunks)):
            prev_type = chunks[i - 1][2]
            cur_audio, cur_act, cur_type, _ = chunks[i]
            cur_audio = cur_audio.astype(np.float32)

            if prev_type == "speech" and cur_type == "speech":
                # Crossfade to smooth the sentence boundary
                fade = min(FADE_N, len(result_audio), len(cur_audio))
                if fade > 0:
                    fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
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
