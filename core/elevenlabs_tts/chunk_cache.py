"""Content-addressed PCM cache: never pay twice for the same chunk.

Key = SHA-256 of the payload (text, voice, model, settings, seed, format).
Request ids are stored beside the PCM but are NOT part of the key — they are
render-time continuity hints, not inputs that change the audio contract.
Pacing renders (timestamps endpoint) add ``"alignment": True`` to the payload,
so their key differs and a plain hit is never used for pacing; the character
alignment itself lives in the ``.json`` sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from core.elevenlabs_tts.client import OUTPUT_FORMAT

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "var" / "elevenlabs_cache"


def cache_dir() -> Path:
    d = Path(os.environ.get("MOODSCAPE_ELEVENLABS_CACHE_DIR", DEFAULT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def chunk_payload(text: str, voice_id: str, model_id: str, stability: float,
                  similarity: float, seed: int, *, alignment: bool = False) -> dict:
    payload = {
        "text": text, "voice_id": voice_id, "model_id": model_id,
        "stability": round(float(stability), 3), "similarity": round(float(similarity), 3),
        "seed": int(seed), "output_format": OUTPUT_FORMAT,
    }
    if alignment:
        payload["alignment"] = True
    return payload


def _key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def is_cached(payload: dict) -> bool:
    return (cache_dir() / f"{_key(payload)}.pcm").is_file()


def load(payload: dict) -> tuple[bytes, str | None] | None:
    base = cache_dir() / _key(payload)
    pcm_path = base.with_suffix(".pcm")
    if not pcm_path.is_file():
        return None
    request_id = None
    meta_path = base.with_suffix(".json")
    if meta_path.is_file():
        try:
            request_id = json.loads(meta_path.read_text(encoding="utf-8")).get("request_id")
        except (OSError, ValueError):
            request_id = None
    return pcm_path.read_bytes(), request_id


def load_alignment(payload: dict) -> dict | None:
    """The character alignment stored with this chunk, or None if absent/unreadable."""
    meta_path = (cache_dir() / _key(payload)).with_suffix(".json")
    if not meta_path.is_file():
        return None
    try:
        al = json.loads(meta_path.read_text(encoding="utf-8")).get("char_alignment")
    except (OSError, ValueError):
        return None
    if not isinstance(al, dict) or not all(k in al for k in ("characters", "starts", "ends")):
        return None
    return al


def store(payload: dict, pcm: bytes, request_id: str | None,
          alignment: dict | None = None) -> None:
    base = cache_dir() / _key(payload)
    tmp = base.with_suffix(".pcm.part")
    tmp.write_bytes(pcm)
    tmp.rename(base.with_suffix(".pcm"))
    meta = {**payload, "request_id": request_id, "created_at": time.time()}
    if alignment is not None:
        meta["char_alignment"] = alignment
    base.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
