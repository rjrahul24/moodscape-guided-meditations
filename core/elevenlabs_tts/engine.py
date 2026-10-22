"""ElevenLabsEngine — Eleven v4 cloud TTS behind the SpeechEngine contract.

Returns 48 kHz audio natively (sample_rate = 48000); the pipeline skips its
upsample for it. No model weights: load/unload only manage the API client.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from core.elevenlabs_tts import chunk_cache
from core.elevenlabs_tts.assembly import (
    activity_mask, chunk_lufs_target, crossfade_concat, normalize_chunk_lufs, pcm16_to_float,
)
from core.elevenlabs_tts.client import ElevenLabsAPIError, ElevenLabsClient, model_id
from core.elevenlabs_tts.pacing import apply_pacing, pacing_enabled, pacing_params
from core.elevenlabs_tts.voices import default_voice_id, list_voices
from core.speech_engine import SpeechEngine

logger = logging.getLogger("moodscape.elevenlabs")
SR = 48000
# Statuses that, when previous_request_ids were sent, mean the ids were rejected.
_STALE_IDS_STATUSES = (400, 422)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class ElevenLabsParams:
    voice_id: str
    stability: float
    similarity: float
    model_id: str
    seed: int

    def cache_extra(self, direction: str) -> dict:
        """Everything that changes ElevenLabs audio, for the whole-render TTS cache key.
        Pacing appears only when enabled, so pacing-off keys are unchanged."""
        extra = {"voice_id": self.voice_id, "stability": self.stability,
                 "similarity": self.similarity, "model_id": self.model_id,
                 "seed": self.seed, "direction": direction}
        if pacing_enabled():
            extra["pacing"] = _pacing_block()
        return extra


def _pacing_block() -> dict:
    stretch, sentence_sec, clause_sec = pacing_params()
    return {"stretch": stretch, "sentence_sec": sentence_sec, "clause_sec": clause_sec}


def resolve_params(voice_id: str | None = None, stability: float | None = None,
                   similarity: float | None = None, seed: int | None = None) -> ElevenLabsParams:
    return ElevenLabsParams(
        voice_id=voice_id or default_voice_id(),
        stability=round(float(stability if stability is not None
                              else _env_float("MOODSCAPE_ELEVENLABS_STABILITY", 0.5)), 3),
        similarity=round(float(similarity if similarity is not None
                               else _env_float("MOODSCAPE_ELEVENLABS_SIMILARITY", 0.75)), 3),
        model_id=model_id(),
        seed=int(seed if seed is not None else os.environ.get("MOODSCAPE_ELEVENLABS_SEED", "42")),
    )


class ElevenLabsEngine(SpeechEngine):
    sample_rate = SR

    def __init__(self, params: ElevenLabsParams | None = None, client=None):
        self.params = params or resolve_params()
        self._client = client
        self._injected = client is not None
        self.last_manifest: dict = {}
        self.last_qa = None

    def load_model(self) -> None:
        if self._client is None:
            self._client = ElevenLabsClient()

    def unload_model(self) -> None:
        if not self._injected:
            self._client = None

    def get_available_voices(self) -> list[dict]:
        return [{"id": v.id, "name": v.name, "description": v.notes} for v in list_voices()]

    def synthesize(self, segments, voice=None, speed=None, progress_cb=None, **kwargs):
        """Render segments; `speed` is ignored (Eleven v4 ignores voice_settings.speed).
        Opt-in pacing (MOODSCAPE_ELEVENLABS_PACING=1) stretches each chunk and
        inserts timestamp-located pauses."""
        if self._client is None:
            self.load_model()
        p = self.params
        voice_id = voice or p.voice_id
        stitch = os.environ.get("MOODSCAPE_ELEVENLABS_STITCH", "1") != "0"
        target = chunk_lufs_target()
        xfade_ms = _env_float("MOODSCAPE_ELEVENLABS_XFADE_MS", 30.0)

        pacing = pacing_enabled()
        if pacing:
            stretch, sentence_sec, clause_sec = pacing_params()

        def render(text, ids):
            kw = dict(model_id=p.model_id, stability=p.stability, similarity=p.similarity,
                      seed=p.seed, previous_request_ids=ids)
            if pacing:
                return self._client.convert_with_alignment(text, voice_id, **kw)
            pcm_, rid_ = self._client.convert(text, voice_id, **kw)
            return pcm_, rid_, None

        total_speech = sum(1 for s in segments if s["type"] == "speech")
        pieces: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        recent: list[str] = []
        manifest = {"model_id": p.model_id, "voice_id": voice_id, "stability": p.stability,
                    "similarity": p.similarity, "seed": p.seed, "chars_billed": 0,
                    "chars_cached": 0, "cache_hits": 0, "cache_misses": 0, "request_ids": [],
                    "pacing": _pacing_block() if pacing else {"enabled": False}}
        done = 0
        for idx, seg in enumerate(segments):
            if seg["type"] == "pause":
                n = int(round(float(seg["duration_sec"]) * SR))
                pieces.append(np.zeros(n, dtype=np.float32))
                masks.append(np.zeros(n, dtype=bool))
            elif seg["type"] == "speech":
                text = seg["text"]
                payload = chunk_cache.chunk_payload(text, voice_id, p.model_id,
                                                    p.stability, p.similarity, p.seed,
                                                    alignment=pacing)
                hit = chunk_cache.load(payload)
                alignment = None
                if hit is not None and pacing:
                    alignment = chunk_cache.load_alignment(payload)
                    if alignment is None:
                        hit = None  # unreadable sidecar: never pace without timestamps
                if hit is not None:
                    pcm, rid = hit
                    manifest["cache_hits"] += 1
                    manifest["chars_cached"] += len(text)
                else:
                    ids = list(recent) if (stitch and recent) else None
                    try:
                        try:
                            pcm, rid, alignment = render(text, ids)
                        except ElevenLabsAPIError as exc:
                            if not (ids and exc.status in _STALE_IDS_STATUSES):
                                raise
                            # Stale/expired request ids (e.g. from old cache hits):
                            # render this chunk unstitched and stop stitching for
                            # the rest of the run so later chunks don't fail twice.
                            logger.warning(
                                "ElevenLabs rejected previous_request_ids (status %s); "
                                "retrying chunk %d without stitching for the rest of this render",
                                exc.status, done + 1)
                            recent, stitch = [], False
                            pcm, rid, alignment = render(text, None)
                    except Exception as exc:
                        status = getattr(exc, "status", None)
                        raise ElevenLabsAPIError(
                            status,
                            f"{exc} — {done} of {total_speech} chunks rendered before failure "
                            "(cached; re-run resumes).",
                        ) from exc
                    chunk_cache.store(payload, pcm, rid, alignment=alignment)
                    manifest["cache_misses"] += 1
                    manifest["chars_billed"] += len(text)
                if rid:
                    recent = (recent + [rid])[-3:]
                    manifest["request_ids"].append(rid)
                y = pcm16_to_float(pcm)
                if pacing:
                    # Before LUFS + mask: inserted gaps read inactive.
                    y = apply_pacing(y, SR, alignment, stretch=stretch,
                                     sentence_sec=sentence_sec, clause_sec=clause_sec)
                if target is not None:
                    y = normalize_chunk_lufs(y, SR, target)
                pieces.append(y)
                masks.append(activity_mask(y, SR))
                done += 1
            if progress_cb is not None:
                progress_cb(idx + 1, len(segments))

        self.last_manifest = manifest
        if not pieces:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)
        logger.info("ElevenLabs: %d chunks (%d cached), %d chars billed",
                    done, manifest["cache_hits"], manifest["chars_billed"])
        return crossfade_concat(pieces, masks, SR, xfade_ms)
