"""Remix: re-run only mix -> master -> export from a saved mix session.

Voice/music trims change the voice-to-music BALANCE; export re-normalizes the
final loudness, so equal trims on both are inaudible by design.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from core.audio_processor import make_master_chain
from core.mix_session import load_session
from core.mixer import export_audio, mix
from core.qa_monitor import run_qa_checks

logger = logging.getLogger("moodscape.remix")
TRIM_RANGE_DB = (-12.0, 12.0)


@contextlib.contextmanager
def _session_env(saved: dict):
    """Apply the MOODSCAPE_* env the original mix ran with, then restore."""
    current = {k: v for k, v in os.environ.items() if k.startswith("MOODSCAPE_")}
    try:
        for k in current:
            if k not in saved:
                os.environ.pop(k, None)
        os.environ.update(saved)
        yield
    finally:
        for k in list(os.environ):
            if k.startswith("MOODSCAPE_") and k not in current:
                os.environ.pop(k, None)
        os.environ.update(current)


def remix_output_path(original: str, voice_db: float, music_db: float) -> Path:
    src = Path(original)
    stem = f"{src.stem}_remix-v{voice_db:+g}-m{music_db:+g}"
    candidate = src.with_name(f"{stem}{src.suffix}")
    n = 2
    while candidate.exists():
        candidate = src.with_name(f"{stem}-{n}{src.suffix}")
        n += 1
    return candidate


def _check(name: str, value: float) -> float:
    lo, hi = TRIM_RANGE_DB
    if not lo <= value <= hi:
        raise ValueError(f"{name} trim {value} dB is outside {lo:g}..{hi:g} dB.")
    return float(value)


def remix(run_id: str, voice_db: float = 0.0, music_db: float = 0.0, progress_cb=None) -> str:
    voice_db = _check("Voice", voice_db)
    music_db = _check("Music", music_db)
    session = load_session(run_id)
    p = session.params
    voice = session.voice_audio
    if voice_db != 0.0:
        voice = (voice * np.float32(10.0 ** (voice_db / 20.0))).astype(np.float32)
    phrases = [tuple(x) for x in p["phrases"]] if p.get("phrases") else None
    with _session_env(p.get("env", {})):
        mixed = mix(
            voice, session.voice_activity, session.music_audio,
            sample_rate=p["mix_sr"], duck_amount_db=p["duck_amount_db"],
            music_volume_db=float(np.clip(p["music_volume_db"] + music_db, -40.0, 0.0)),
            fade_in_sec=p["fade_in_sec"], fade_out_sec=p["fade_out_sec"],
            stereo_output=p["stereo_output"], phrases=phrases,
            bed_overrides=p.get("bed_overrides"),
        )
        qa = run_qa_checks(mixed, p["mix_sr"], is_vocals_only=False,
                           is_uploaded_music=p.get("use_upload", False))
        if not qa["clipping"]["passed"]:
            logger.warning("Remix %s: pre-master clipping detected", run_id)
        tmp = export_audio(
            mixed, sample_rate=p["mix_sr"], output_format=p["output_format"],
            target_sample_rate=p["export_sr"], master_chain=make_master_chain(),
            target_lufs=p["target_lufs"], mono_playback_offset_db=p["mono_offset"],
        )
    dest = remix_output_path(p["output_path"], voice_db, music_db)
    shutil.move(tmp, dest)
    dest.with_suffix(".meta.json").write_text(json.dumps(
        {"remix_of": p["output_path"], "run_id": run_id,
         "voice_db": voice_db, "music_db": music_db}, indent=2), encoding="utf-8")
    return str(dest)
