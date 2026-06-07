"""Unit tests for the background instrumental library scanner."""

import re
from pathlib import Path

import numpy as np
import soundfile as sf

from core.upload_music import background_library as bl
from core.upload_music import scan_backgrounds, BACKGROUNDS_DIR

_SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
_LABEL_RE = re.compile(r" — \d+:\d{2}$")  # "<name> — MM:SS"


def test_short_name_strips_prefix_and_id():
    assert bl._short_name("light_music-healing-forest-187590") == "Healing Forest"
    assert bl._short_name("relaxingtime-relax-music-vol12-191749") == "Relax Music Vol12"
    # Single token (no prefix/id to strip) still yields something readable.
    assert bl._short_name("meditation") == "Meditation"


def test_format_duration():
    assert bl._format_duration(0) == "0:00"
    assert bl._format_duration(72) == "1:12"
    assert bl._format_duration(1152.0) == "19:12"


def test_scan_real_backgrounds_well_formed():
    """The shipped assets/backgrounds/ folder scans into valid (label, path) pairs."""
    choices = scan_backgrounds()
    assert isinstance(choices, list)
    assert len(choices) > 0, "expected curated tracks in assets/backgrounds/"
    for label, path in choices:
        assert _LABEL_RE.search(label), f"label missing MM:SS suffix: {label!r}"
        assert Path(path).exists(), f"path does not exist: {path}"
        assert Path(path).suffix.lower() in _SUPPORTED_EXTS
    # Sorted by label, case-insensitive.
    labels = [c[0].lower() for c in choices]
    assert labels == sorted(labels)


def test_scan_missing_dir_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(bl, "BACKGROUNDS_DIR", tmp_path / "does_not_exist")
    assert scan_backgrounds() == []


def test_scan_skips_unreadable_and_nonaudio(monkeypatch, tmp_path):
    # One valid wav, one bogus .mp3, one unrelated file.
    sr = 16000
    sf.write(tmp_path / "alpha-good-clip-101.wav",
             np.zeros(sr, dtype="float32"), sr)
    (tmp_path / "beta-broken-202.mp3").write_bytes(b"not really audio")
    (tmp_path / "readme.txt").write_text("ignore me")

    monkeypatch.setattr(bl, "BACKGROUNDS_DIR", tmp_path)
    choices = scan_backgrounds()

    assert len(choices) == 1
    label, path = choices[0]
    assert Path(path).name == "alpha-good-clip-101.wav"
    assert _LABEL_RE.search(label)
