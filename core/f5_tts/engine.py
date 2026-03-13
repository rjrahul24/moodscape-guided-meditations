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

import numpy as np

from core.speech_engine import SAMPLE_RATE, SpeechEngine
from core.f5_tts.postprocessor import crossfade_chunks
from core.f5_tts import voice_registry

logger = logging.getLogger(__name__)

_NFE_STEPS = 32
_SWAY_COEF = -1.0  # enables sway sampling for smoother prosody


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
        self._ref_audio_path = str(paths["audio"])
        self._ref_text = paths["transcript"].read_text(encoding="utf-8").strip()
        self._model = None

        logger.info(
            "F5Engine initialised with voice '%s' (%s)",
            self._voice_slug,
            self._ref_audio_path,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load F5TTS model onto MPS (Apple Silicon) or CPU."""
        if self._model is not None:
            return

        import torch
        from f5_tts.api import F5TTS

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading F5TTS (F5TTS_v1_Base) on %s", device)
        self._model = F5TTS(model="F5TTS_v1_Base", device=device)

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
            speed:       Speaking speed scalar (0.5–1.0). 0.75 is meditation ideal.
            progress_cb: Optional callback(current_index, total_segments).
            **kwargs:    Absorbs engine-specific kwargs (e.g. seed=) passed by the
                         pipeline that are not applicable to F5-TTS.

        Returns:
            voice_audio:    float32 mono numpy array at 24 000 Hz.
            voice_activity: bool array of the same length; True where voice is active.
        """
        if self._model is None:
            raise RuntimeError("F5TTS model not loaded. Call load_model() first.")

        audio_chunks: list[np.ndarray] = []
        activity_chunks: list[np.ndarray] = []
        total = len(segments)

        for idx, seg in enumerate(segments):
            if seg["type"] == "speech":
                wav, _sr, _ = self._model.infer(
                    ref_file=self._ref_audio_path,
                    ref_text=self._ref_text,
                    gen_text=seg["text"],
                    speed=speed,
                    nfe_step=_NFE_STEPS,
                    sway_sampling_coef=_SWAY_COEF,
                    remove_silence=False,
                )
                arr = wav.cpu().squeeze().numpy().astype(np.float32)
                # Energy-threshold voice activity: active where amplitude > 15% of mean
                threshold = float(np.abs(arr).mean()) * 0.15
                activity = np.abs(arr) > threshold
                audio_chunks.append(arr)
                activity_chunks.append(activity)

            elif seg["type"] == "pause":
                n = int(seg["duration_sec"] * SAMPLE_RATE)
                audio_chunks.append(np.zeros(n, dtype=np.float32))
                activity_chunks.append(np.zeros(n, dtype=bool))

            if progress_cb is not None:
                progress_cb(idx + 1, total)

        if not audio_chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

        voice_audio = crossfade_chunks(audio_chunks)
        voice_activity = np.concatenate(activity_chunks)

        # Trim to the same length (crossfade can shift total length by a few samples)
        min_len = min(len(voice_audio), len(voice_activity))
        return voice_audio[:min_len].astype(np.float32), voice_activity[:min_len]

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
