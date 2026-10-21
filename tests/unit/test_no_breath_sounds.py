"""Breath markers must render as silence — never a breath recording."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from core.speech_engine import BREATH_SILENCE_SEC, breath_silence_sec

ROOT = Path(__file__).resolve().parents[2]


def test_breath_silence_table():
    assert BREATH_SILENCE_SEC == {"breath": 1.2, "inhale": 1.5, "exhale": 1.8}
    assert breath_silence_sec("inhale") == 1.5
    assert breath_silence_sec("unknown") == 1.2


def test_duration_alias_matches():
    from core.script_gen.duration import BREATH_SEC
    assert BREATH_SEC == BREATH_SILENCE_SEC


def test_breath_sample_module_and_assets_are_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("core.breath_sounds")
    assert not (ROOT / "assets" / "breath_sounds").exists()


def test_no_engine_references_breath_samples():
    for rel in ("core/f5_tts/engine.py", "core/kokoro_tts/engine.py",
                "core/chatterbox_tts/engine.py"):
        src = (ROOT / rel).read_text()
        assert "load_breath" not in src, rel
        assert "breath_sounds" not in src, rel


def test_chatterbox_breath_segment_is_silence(monkeypatch):
    from core.chatterbox_tts import engine as cb

    eng = cb.ChatterboxEngine.__new__(cb.ChatterboxEngine)
    out = eng._breath_piece({"type": "breath", "subtype": "exhale"})
    audio, act = out
    assert len(audio) == int(round(1.8 * cb.SAMPLE_RATE))
    assert not audio.any() and not act.any()
