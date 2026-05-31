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
    INDEXTTS_TOP_K,
    INDEXTTS_TEMPERATURE,
    INDEXTTS_NUM_BEAMS,
    INDEXTTS_REPETITION_PENALTY,
    INDEXTTS_MAX_MEL_TOKENS,
    INDEXTTS_INTERVAL_SILENCE_MS,
    INDEXTTS_MAX_TOKENS_PER_SEG,
    _apply_meditation_pace,
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
    assert kw["top_k"] == INDEXTTS_TOP_K
    assert kw["temperature"] == INDEXTTS_TEMPERATURE
    assert kw["num_beams"] == INDEXTTS_NUM_BEAMS
    assert kw["repetition_penalty"] == INDEXTTS_REPETITION_PENALTY
    assert kw["max_mel_tokens"] == INDEXTTS_MAX_MEL_TOKENS
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


# ── Pacing (Rubber Band → librosa fallback) ──────────────────────────────────

def test_pace_rate_one_is_noop():
    """rate == 1.0 returns the input unchanged (no stretch backend invoked)."""
    sr = 24000
    sig = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    out = _apply_meditation_pace(sig, rate=1.0, sr=sr)
    assert np.array_equal(out, sig)


def test_pace_lengthens_audio():
    """rate < 1.0 lengthens the chunk by ~1/rate (Rubber Band or librosa fallback)."""
    sr = 24000
    sig = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    out = _apply_meditation_pace(sig, rate=0.92, sr=sr)
    # Expect ~1/0.92 ≈ 1.087x longer; allow generous tolerance across backends.
    assert len(out) > len(sig) * 1.03
    assert len(out) < len(sig) * 1.18
    assert out.dtype == np.float32


# ── DeepFilterNet dry/wet blend ──────────────────────────────────────────────

def test_blend_wet_dry_math_and_length_alignment():
    from core.deepfilter_enhancer import _blend_wet_dry
    dry = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    wet = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # shorter on purpose
    out = _blend_wet_dry(dry, wet, wet_amount=0.25)
    # Length aligns to the shorter (3); blend = 0.75*dry + 0.25*wet = 0.75
    assert len(out) == 3
    assert np.allclose(out, 0.75)


def test_enhance_wet_zero_bypasses_processing():
    """wet <= 0 short-circuits and returns the exact input (no model/deps needed)."""
    from core.deepfilter_enhancer import enhance_voice_deepfilter
    sr = 48000
    audio = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    out = enhance_voice_deepfilter(audio, sr=sr, wet=0.0)
    assert np.array_equal(out, audio)
