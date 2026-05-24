"""Verify IndexTTS-2 engine wires the full infer() API: emotion + sampling params.

These tests mock the IndexTTS2 model so they run without checkpoints. They guard
against regression of the bug where `emotion_audio_path` was loaded but never
passed to `infer()`, and where `top_p`/`temperature`/`interval_silence` /
`max_text_tokens_per_segment` defaulted to API defaults instead of our
meditation-tuned values.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.append(os.getcwd())

from core.index_tts.engine import (
    IndexTTSEngine,
    INDEXTTS_CALM_VECTOR,
    INDEXTTS_EMO_ALPHA,
    INDEXTTS_TOP_P,
    INDEXTTS_TEMPERATURE,
    INDEXTTS_INTERVAL_SILENCE_MS,
    INDEXTTS_MAX_TOKENS_PER_SEG,
)


class _MockIndexTTS2:
    """Records the kwargs of the most recent infer() call and writes a dummy WAV."""

    def __init__(self):
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        sr = 24000
        arr = np.zeros(sr, dtype=np.float32)
        sf.write(kwargs["output_path"], arr, sr, subtype="FLOAT")


def _build_engine_with_mock(emotion_audio_path: str | None = None) -> tuple[IndexTTSEngine, _MockIndexTTS2]:
    engine = IndexTTSEngine.__new__(IndexTTSEngine)
    engine._model = _MockIndexTTS2()
    engine._voice_audio_path = "/fake/voice.wav"
    engine._emotion_audio_path = emotion_audio_path
    engine._voice_slug = "test"
    engine._emotion_slug = None
    return engine, engine._model


def _segments():
    return [{"type": "speech", "text": "Breathe in, and out."}]


def test_calm_vector_passed_when_no_emotion_audio():
    """Default path: no emotion audio → calm emo_vector preset."""
    engine, mock = _build_engine_with_mock(emotion_audio_path=None)
    engine.synthesize(_segments(), speed=1.0, seed=42)

    assert len(mock.calls) == 1
    kw = mock.calls[0]
    assert kw["emo_vector"] == INDEXTTS_CALM_VECTOR
    assert "emo_audio_prompt" not in kw
    assert kw["emo_alpha"] == INDEXTTS_EMO_ALPHA
    assert kw["top_p"] == INDEXTTS_TOP_P
    assert kw["temperature"] == INDEXTTS_TEMPERATURE
    assert kw["interval_silence"] == INDEXTTS_INTERVAL_SILENCE_MS
    assert kw["max_text_tokens_per_segment"] == INDEXTTS_MAX_TOKENS_PER_SEG
    assert kw["do_sample"] is True
    assert kw["use_random"] is False


def test_emotion_audio_overrides_calm_vector():
    """When emotion_audio_path is set, the audio reference wins (vector omitted)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        emo_path = tmp.name
    try:
        engine, mock = _build_engine_with_mock(emotion_audio_path=emo_path)
        engine.synthesize(_segments(), speed=1.0, seed=42)

        assert len(mock.calls) == 1
        kw = mock.calls[0]
        assert kw["emo_audio_prompt"] == emo_path
        assert "emo_vector" not in kw
        assert kw["emo_alpha"] == INDEXTTS_EMO_ALPHA
    finally:
        Path(emo_path).unlink(missing_ok=True)


def test_per_call_emotion_kwarg_overrides_constructor_emotion():
    """kwargs['emotion_audio_path'] in synthesize() takes precedence."""
    engine, mock = _build_engine_with_mock(emotion_audio_path=None)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        override = tmp.name
    try:
        engine.synthesize(_segments(), speed=1.0, seed=42, emotion_audio_path=override)
        kw = mock.calls[0]
        assert kw["emo_audio_prompt"] == override
        assert "emo_vector" not in kw
    finally:
        Path(override).unlink(missing_ok=True)


def test_speed_param_is_noop_and_warns_once(caplog):
    """speed != 1.0 should log a one-time warning and not crash."""
    import logging
    engine, mock = _build_engine_with_mock(emotion_audio_path=None)

    with caplog.at_level(logging.WARNING, logger="core.index_tts.engine"):
        engine.synthesize(_segments(), speed=0.8, seed=1)
        engine.synthesize(_segments(), speed=0.8, seed=2)

    warnings = [r for r in caplog.records if "does not support reliable time-stretching" in r.message]
    assert len(warnings) == 1, "speed warning should fire exactly once per engine instance"


def test_meditation_lexicon_phoneticizes_sanskrit_terms():
    from core.index_tts.preprocessor import normalize_for_indextts
    out = normalize_for_indextts("Begin with Om. Practice pranayama and savasana.")
    assert "ohm" in out.lower()
    assert "prah-nah-yama" in out.lower()
    assert "shah-vah-sana" in out.lower()
