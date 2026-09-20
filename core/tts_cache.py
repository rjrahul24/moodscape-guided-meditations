"""Persistent TTS voice caching system for MoodScape.

Caches synthesized narration audio and activity masks so that re-mixing with
different background music tracks, volumes, ducking amounts, or fades does not
require re-running long (5–10 minute) TTS inference passes.

Cache files are stored in `var/tts_cache/` (gitignored machine-local state).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("moodscape.tts_cache")

# Default directory for persistent TTS cache (under project var/)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "var" / "tts_cache"


def get_cache_dir() -> Path:
    """Return the Path to the TTS cache directory, ensuring it exists."""
    cache_dir = Path(os.environ.get("MOODSCAPE_TTS_CACHE_DIR", DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_cache_key(
    script: str,
    content_type: str = "meditation",
    tts_engine: str = "kokoro",
    voice: str | None = None,
    speed: float | None = None,
    f5_target_wpm: int | None = None,
    f5_cfg_strength: float = 2.0,
    microprosody: bool = False,
    seed: int | None = None,
) -> str:
    """Compute a deterministic SHA-256 hash key for TTS inputs.

    Normalizes text and parameters so trivial differences (e.g. trailing
    whitespace) do not cause spurious cache misses.
    """
    normalized_script = script.strip().replace("\r\n", "\n")
    key_dict = {
        "script": normalized_script,
        "content_type": str(content_type).strip().lower(),
        "tts_engine": str(tts_engine).strip().lower(),
        "voice": str(voice).strip() if voice is not None else "",
        "speed": round(float(speed), 3) if speed is not None else None,
        "f5_target_wpm": int(f5_target_wpm) if f5_target_wpm and f5_target_wpm > 0 else None,
        "f5_cfg_strength": round(float(f5_cfg_strength), 2) if f5_cfg_strength is not None else 2.0,
        "microprosody": bool(microprosody),
        "seed": int(seed) if seed is not None else None,
    }
    payload = json.dumps(key_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def has_cache(key: str) -> bool:
    """Check if a valid cache entry exists for the given key."""
    cache_dir = get_cache_dir()
    npz_path = cache_dir / f"{key}.npz"
    return npz_path.is_file()


def get_cached_tts(key: str) -> dict[str, Any] | None:
    """Retrieve cached TTS output for a given key.

    Returns:
        Dict with keys:
            - 'voice_audio': np.ndarray float32 (24 kHz)
            - 'voice_activity': np.ndarray bool
            - 'sample_rate': int
            - 'metadata': dict
        or None if not found or corrupted.
    """
    cache_dir = get_cache_dir()
    npz_path = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.json"

    if not npz_path.is_file():
        return None

    try:
        with np.load(npz_path) as data:
            voice_audio = np.asarray(data["voice_audio"], dtype=np.float32)
            voice_activity = np.asarray(data["voice_activity"], dtype=bool)
            sample_rate = int(data["sample_rate"]) if "sample_rate" in data else 24000

        metadata = {}
        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning("Failed to read TTS cache metadata for %s: %s", key, e)

        return {
            "voice_audio": voice_audio,
            "voice_activity": voice_activity,
            "sample_rate": sample_rate,
            "metadata": metadata,
        }
    except Exception as e:
        logger.error("Error loading TTS cache for %s: %s", key, e)
        return None


def save_cached_tts(
    key: str,
    voice_audio: np.ndarray,
    voice_activity: np.ndarray,
    sample_rate: int = 24000,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save synthesized voice audio and activity mask to disk cache.

    Also maintains a 'latest' pointer to easily retrieve the most recent take.

    Returns:
        The cache key.
    """
    cache_dir = get_cache_dir()
    npz_path = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.json"

    voice_audio = np.asarray(voice_audio, dtype=np.float32)
    voice_activity = np.asarray(voice_activity, dtype=bool)

    # Save compressed array
    np.savez_compressed(
        npz_path,
        voice_audio=voice_audio,
        voice_activity=voice_activity,
        sample_rate=sample_rate,
    )

    # Write metadata
    meta = metadata.copy() if metadata else {}
    meta.setdefault("timestamp", time.time())
    meta.setdefault("key", key)
    meta.setdefault("duration_sec", len(voice_audio) / float(sample_rate) if sample_rate else 0.0)

    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save TTS cache metadata for %s: %s", key, e)

    # Update 'latest' pointer
    try:
        latest_npz = cache_dir / "latest.npz"
        latest_meta = cache_dir / "latest.json"
        shutil.copyfile(npz_path, latest_npz)
        shutil.copyfile(meta_path, latest_meta)
    except Exception as e:
        logger.warning("Failed to update latest TTS cache pointer: %s", e)

    logger.info("Saved TTS cache for key %s (%.1fs audio)", key[:12], meta.get("duration_sec", 0.0))
    return key


def get_latest_tts() -> dict[str, Any] | None:
    """Retrieve the most recently cached TTS output."""
    return get_cached_tts("latest")


def clear_cache(key: str | None = None) -> int:
    """Clear a specific cache entry or all cache files.

    Returns:
        Number of cache items deleted.
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    count = 0
    if key:
        for ext in (".npz", ".json"):
            target = cache_dir / f"{key}{ext}"
            if target.is_file():
                target.unlink()
                count += 1
    else:
        for p in cache_dir.glob("*.npz"):
            p.unlink()
            count += 1
        for p in cache_dir.glob("*.json"):
            p.unlink()
            count += 1

    return count
