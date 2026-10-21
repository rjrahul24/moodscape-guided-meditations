"""Saved mix inputs so Remix can re-run only mix -> master -> export.

A session is the processed voice (+ activity mask) and processed music exactly
as passed to mixer.mix(), plus the parameters used. Sessions are large
(~350 MB for 15 min of 48 kHz float32), so only the newest few are kept.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "var" / "mix_sessions"


class MixSessionNotFound(FileNotFoundError):
    pass


@dataclass
class MixSession:
    voice_audio: np.ndarray
    voice_activity: np.ndarray
    music_audio: np.ndarray
    params: dict


def sessions_dir() -> Path:
    d = Path(os.environ.get("MOODSCAPE_MIX_SESSIONS_DIR", DEFAULT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_session(run_id: str, voice_audio, voice_activity, music_audio, params: dict) -> Path:
    d = sessions_dir() / run_id
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "voice.npy", np.asarray(voice_audio, dtype=np.float32))
    np.save(d / "voice_activity.npy", np.asarray(voice_activity, dtype=bool))
    np.save(d / "music.npy", np.asarray(music_audio, dtype=np.float32))
    (d / "mix_params.json").write_text(json.dumps(_json_safe(params), indent=2), encoding="utf-8")
    return d


def _dir_for(run_id: str) -> Path:
    d = sessions_dir() / run_id
    if not run_id or not (d / "mix_params.json").is_file():
        raise MixSessionNotFound(f"Mix session not found ({run_id!r}) — generate again to remix.")
    return d


def update_params(run_id: str, updates: dict) -> None:
    path = _dir_for(run_id) / "mix_params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(_json_safe(updates))
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def load_session(run_id: str) -> MixSession:
    d = _dir_for(run_id)
    return MixSession(
        voice_audio=np.load(d / "voice.npy"),
        voice_activity=np.load(d / "voice_activity.npy"),
        music_audio=np.load(d / "music.npy"),
        params=json.loads((d / "mix_params.json").read_text(encoding="utf-8")),
    )


def prune_sessions(keep: int | None = None) -> int:
    keep = keep if keep is not None else int(os.environ.get("MOODSCAPE_MIX_SESSIONS_KEEP", "5"))
    dirs = sorted((p for p in sessions_dir().iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    stale = dirs[max(0, keep):]
    for d in stale:
        shutil.rmtree(d, ignore_errors=True)
    return len(stale)
