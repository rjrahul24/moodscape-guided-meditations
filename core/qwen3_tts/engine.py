# core/qwen3_tts/engine.py
"""Qwen3-TTS SpeechEngine on mlx-audio (experimental, sandbox only).

Three mutually exclusive modes, because the upstream model variants are:
  clone  -> Base model, our reference clip + transcript, no instruct
  custom -> CustomVoice model, preset speaker + delivery instruct
  design -> VoiceDesign model, instruct is a narrator description
Upstream `speed` is not implemented; pacing comes from pauses (and the
optional MOODSCAPE_QWEN3_STRETCH arm).
"""

import gc
import logging
import os
import time
from pathlib import Path

import numpy as np

from core.qwen3_tts import presets
from core.qwen3_tts.postprocessor import TARGET_SR, condition_chunk, time_stretch
from core.speech_engine import SAMPLE_RATE, SpeechEngine

logger = logging.getLogger("moodscape.qwen3_tts")

INTER_CHUNK_GAP_SEC = 0.4
_RETRY_SEED_OFFSET = 1000


def _default_loader(repo: str):
    from mlx_audio.tts.utils import load_model
    return load_model(repo)


def _seed(value: int) -> None:
    import mlx.core as mx
    mx.random.seed(int(value))


def _to_numpy(audio) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class Qwen3Engine(SpeechEngine):
    def __init__(self, voice_slug: str | None = None, *, mode: str | None = None, loader=None) -> None:
        self.voice_slug = voice_slug
        self.mode = mode or presets.mode_from_env()
        self._loader = loader or _default_loader
        self._model = None
        self._warned_instruct = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def load_model(self) -> None:
        repo = presets.MODEL_REPOS[self.mode]
        try:
            self._model = self._loader(repo)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Qwen3-TTS ({self.mode}) from {repo}: {exc}. "
                f"Download it with: huggingface-cli download {repo}"
            ) from exc

    def unload_model(self) -> None:
        self._model = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    def get_available_voices(self) -> list[dict]:
        from core.f5_tts import voice_registry
        return [{"id": slug, "name": slug, "description": "Reference clone (clone mode)"}
                for slug in sorted(voice_registry.scan())]

    # ── synthesis ────────────────────────────────────────────────────────
    def _reference(self, voice: str | None) -> tuple[str, str]:
        from core.f5_tts import voice_registry
        registry = voice_registry.scan()
        slug = voice or self.voice_slug or sorted(registry)[0]
        phase = voice_registry.get_voice(slug)["default"]
        transcript = phase["transcript"]
        text = Path(transcript).read_text(encoding="utf-8").strip() if isinstance(transcript, Path) else str(transcript)
        return str(phase["audio"]), text

    def _call(self, text: str, *, content_type: str, temperature: float, ref: tuple[str, str] | None):
        instruct = presets.instruct_for(self.mode, content_type)
        if self.mode == "clone":
            if os.environ.get("MOODSCAPE_QWEN3_INSTRUCT") and not self._warned_instruct:
                logger.warning("Qwen3 clone mode ignores MOODSCAPE_QWEN3_INSTRUCT (Base model has no instruct).")
                self._warned_instruct = True
            return self._model.generate(text=text, ref_audio=ref[0], ref_text=ref[1],
                                        temperature=temperature, lang_code="english")
        if self.mode == "custom":
            return self._model.generate_custom_voice(text=text, speaker=presets.speaker_from_env(),
                                                     language="English", instruct=instruct,
                                                     temperature=temperature)
        return self._model.generate_voice_design(text=text, language="English", instruct=instruct,
                                                 temperature=temperature)

    def _render(self, text: str, seed: int, **kwargs) -> np.ndarray:
        for attempt_seed in (seed, seed + _RETRY_SEED_OFFSET):
            _seed(attempt_seed)
            results = list(self._call(text, **kwargs))
            parts = [condition_chunk(_to_numpy(r.audio), int(r.sample_rate)) for r in results]
            audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            if audio.size and np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-4:
                return audio
            logger.warning("Qwen3 produced empty audio (seed %d) for %r; retrying", attempt_seed, text[:60])
        raise RuntimeError(f"Qwen3-TTS produced no audio for chunk: {text!r}")

    def synthesize(self, segments, voice=None, speed=1.0, progress_cb=None, seed=None,
                   content_type: str = "meditation"):
        started = time.perf_counter()
        if self._model is None:
            self.load_model()
        import random
        base_seed = seed if seed is not None else random.randint(0, 2 ** 31 - 1)
        temperature = presets.temperature_from_env()
        stretch = presets.stretch_from_env()
        ref = self._reference(voice) if self.mode == "clone" else None
        gap = np.zeros(int(INTER_CHUNK_GAP_SEC * TARGET_SR), dtype=np.float32)

        audio_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        speech_idx = 0
        total = len(segments)
        for index, seg in enumerate(segments):
            if seg["type"] == "speech":
                chunk = self._render(seg["text"], base_seed + speech_idx, content_type=content_type,
                                     temperature=temperature, ref=ref)
                chunk = time_stretch(chunk, stretch)
                speech_idx += 1
                threshold = float(np.abs(chunk).mean()) * 0.15
                if audio_parts and mask_parts and mask_parts[-1].any():
                    audio_parts.append(gap)
                    mask_parts.append(np.zeros(len(gap), dtype=bool))
                audio_parts.append(chunk)
                mask_parts.append(np.abs(chunk) > threshold)
            elif seg["type"] == "pause":
                silence = np.zeros(int(float(seg["duration_sec"]) * TARGET_SR), dtype=np.float32)
                audio_parts.append(silence)
                mask_parts.append(np.zeros(len(silence), dtype=bool))
            elif seg["type"] == "breath":
                from core.breath_sounds import load_breath
                breath = load_breath(seg["subtype"], target_sr=TARGET_SR).astype(np.float32)
                audio_parts.append(breath)
                mask_parts.append(np.zeros(len(breath), dtype=bool))
            if progress_cb:
                progress_cb(index + 1, total)

        if not audio_parts:
            return np.zeros(0, np.float32), np.zeros(0, bool)
        assert TARGET_SR == SAMPLE_RATE
        audio = np.concatenate(audio_parts).astype(np.float32)
        wall = time.perf_counter() - started
        audio_sec = len(audio) / TARGET_SR
        rtf = wall / audio_sec if audio_sec > 0 else float("inf")
        logger.info("Qwen3 synth mode=%s audio=%.1fs wall=%.1fs RTF=%.2f", self.mode, audio_sec, wall, rtf)
        return audio, np.concatenate(mask_parts).astype(bool)
