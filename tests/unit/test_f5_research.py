"""Tests for the research-driven F5 improvements:
A1 reference trailing-silence padding, A2 short-phrase pacing,
B3 latent-parameter env overrides. (B4 microprosody lives in
test_f5_microprosody.py.)"""

import os

import numpy as np
import soundfile as sf

from core.f5_tts.engine import _condition_reference_audio


def _dbfs(x: np.ndarray) -> float:
    return 20.0 * np.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12)


# ── A1: reference trailing-silence padding ──────────────────────────────────

def test_ref_pad_appends_low_level_tail(tmp_path):
    """By default _condition_reference_audio appends ~1 s of low-level noise
    so F5 leaks silence (not a stray syllable) on short generations."""
    sr = 24000
    tone = (0.5 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    src = tmp_path / "ref.wav"
    sf.write(src, tone, sr)

    out = _condition_reference_audio(str(src), sr)
    try:
        y, oy_sr = sf.read(out, dtype="float32")
        assert oy_sr == sr
        # ~1.0 s of padding appended past the original 1 s tone.
        assert len(y) >= len(tone) + int(0.9 * sr)
        # The appended tail is low-level noise: above pure digital silence but
        # below F5's -42 dBFS internal edge-trim threshold.
        tail = y[-int(0.5 * sr):]
        assert -80.0 < _dbfs(tail) < -42.0
    finally:
        if os.path.isfile(out):
            os.unlink(out)


def test_ref_pad_disabled_by_flag(tmp_path):
    """MOODSCAPE_F5_REF_PAD=0 restores the old (no-pad) behavior."""
    sr = 24000
    tone = (0.5 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    src = tmp_path / "ref.wav"
    sf.write(src, tone, sr)

    os.environ["MOODSCAPE_F5_REF_PAD"] = "0"
    try:
        out = _condition_reference_audio(str(src), sr)
    finally:
        os.environ.pop("MOODSCAPE_F5_REF_PAD", None)
    try:
        y, _ = sf.read(out, dtype="float32")
        # No padding → length within a few samples of the input.
        assert len(y) <= len(tone) + int(0.05 * sr)
    finally:
        if os.path.isfile(out):
            os.unlink(out)


# ── R3: reference-dynamics preservation ─────────────────────────────────────

def _quiet_tone(sr, rms_db=-40.0):
    amp = (10.0 ** (rms_db / 20.0)) * np.sqrt(2.0)
    return (amp * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)


def test_ref_default_normalizes_to_minus20(tmp_path):
    sr = 24000
    tone = _quiet_tone(sr, -40.0)
    src = tmp_path / "ref.wav"
    sf.write(src, tone, sr)
    out = _condition_reference_audio(str(src), sr)
    try:
        y, _ = sf.read(out, dtype="float32")
        assert abs(_dbfs(y[: len(tone)]) - (-20.0)) < 1.5  # lifted to target
    finally:
        if os.path.isfile(out):
            os.unlink(out)


def test_ref_preserve_dynamics_skips_rms_norm(tmp_path):
    sr = 24000
    tone = _quiet_tone(sr, -40.0)
    src = tmp_path / "ref.wav"
    sf.write(src, tone, sr)
    os.environ["MOODSCAPE_F5_REF_PRESERVE_DYNAMICS"] = "1"
    try:
        out = _condition_reference_audio(str(src), sr)
    finally:
        os.environ.pop("MOODSCAPE_F5_REF_PRESERVE_DYNAMICS", None)
    try:
        y, _ = sf.read(out, dtype="float32")
        # natural ~-40 dBFS level preserved, not lifted to -20
        assert _dbfs(y[: len(tone)]) < -30.0
    finally:
        if os.path.isfile(out):
            os.unlink(out)


# ── shared mock harness for synthesize() infer-kwargs ───────────────────────

def _mock_engine_capturing():
    """An F5Engine wired to a mock model that records every infer() call's
    kwargs, without loading the real network."""
    import pytest
    from core.f5_tts.engine import F5Engine
    from core.f5_tts import voice_registry

    registry = voice_registry.scan()
    if not registry:
        pytest.skip("no F5 voices registered")
    engine = F5Engine(voice_slug=sorted(registry.keys())[0])
    calls: list[dict] = []

    class MockModel:
        def infer(self, **kwargs):
            calls.append(dict(kwargs))
            return np.random.uniform(-0.1, 0.1, 24000).astype(np.float32), 24000, None

    engine._model = MockModel()
    engine._phase_assets = {
        "default": {"audio": "mock.wav", "text": "mock text", "duration_sec": 10.0}
    }
    return engine, calls


_LONG = "Allow your body to slowly sink into the chair beneath you now."


# ── A2: short-phrase pacing ─────────────────────────────────────────────────

def test_short_phrase_lowers_speed():
    engine, calls = _mock_engine_capturing()
    engine.synthesize([{"type": "speech", "text": "Breathe in."}], speed=0.88)
    assert calls[-1]["speed"] == 0.5


def test_long_phrase_keeps_speed():
    engine, calls = _mock_engine_capturing()
    engine.synthesize([{"type": "speech", "text": _LONG}], speed=0.88)
    assert calls[-1]["speed"] == 0.88


def test_short_phrase_pacing_disabled_by_flag():
    engine, calls = _mock_engine_capturing()
    os.environ["MOODSCAPE_F5_SHORT_PHRASE_PACING"] = "0"
    try:
        engine.synthesize([{"type": "speech", "text": "Breathe in."}], speed=0.88)
    finally:
        os.environ.pop("MOODSCAPE_F5_SHORT_PHRASE_PACING", None)
    assert calls[-1]["speed"] == 0.88


def test_normal_short_sentence_not_slowed():
    """Regression: a normal 6-word sentence must NOT trigger short-phrase pacing.
    The old word-count threshold (<=6 words) slowed it to 0.5, which made F5
    stretch and insert mid-word gaps ('Notice ... the ... breath ... ing')."""
    engine, calls = _mock_engine_capturing()
    engine.synthesize(
        [{"type": "speech", "text": "Notice the breathing in your body."}], speed=0.88,
    )
    assert calls[-1]["speed"] == 0.88


def test_short_phrase_char_threshold_override():
    """MOODSCAPE_F5_SHORT_PHRASE_MAX_CHARS controls the (non-space) char cutoff."""
    import os
    engine, calls = _mock_engine_capturing()
    os.environ["MOODSCAPE_F5_SHORT_PHRASE_MAX_CHARS"] = "5"
    try:
        # "Breathe in." = 10 non-space chars > 5 → not slowed under the override
        engine.synthesize([{"type": "speech", "text": "Breathe in."}], speed=0.88)
    finally:
        os.environ.pop("MOODSCAPE_F5_SHORT_PHRASE_MAX_CHARS", None)
    assert calls[-1]["speed"] == 0.88


def test_short_phrase_pacing_skipped_in_wpm_mode():
    """In fixed-WPM mode, fix_duration governs length, so the speed override
    must not fire."""
    engine, calls = _mock_engine_capturing()
    engine.synthesize([{"type": "speech", "text": "Breathe in."}], speed=0.88,
                      target_wpm=110)
    assert calls[-1]["speed"] == 0.88
    assert "fix_duration" in calls[-1]


# ── B3: latent-parameter env overrides ──────────────────────────────────────

def test_f5_param_defaults_reach_infer():
    engine, calls = _mock_engine_capturing()
    engine.synthesize([{"type": "speech", "text": _LONG}], speed=0.88)
    k = calls[-1]
    assert k["cfg_strength"] == 2.0
    assert k["sway_sampling_coef"] == -1.0
    assert k["nfe_step"] == 32


def test_f5_param_env_overrides_reach_infer():
    os.environ.update({
        "MOODSCAPE_F5_CFG": "1.8",
        "MOODSCAPE_F5_SWAY": "-0.8",
        "MOODSCAPE_F5_NFE": "64",
    })
    try:
        engine, calls = _mock_engine_capturing()
        engine.synthesize([{"type": "speech", "text": _LONG}], speed=0.88)
    finally:
        for key in ("MOODSCAPE_F5_CFG", "MOODSCAPE_F5_SWAY", "MOODSCAPE_F5_NFE"):
            os.environ.pop(key, None)
    k = calls[-1]
    assert k["cfg_strength"] == 1.8
    assert k["sway_sampling_coef"] == -0.8
    assert k["nfe_step"] == 64


# ── B4: microprosody wiring (engine gates on the flag) ──────────────────────

def test_microprosody_off_by_default(monkeypatch):
    import core.f5_tts.engine as eng
    seen = {"n": 0}
    monkeypatch.setattr(eng, "apply_microprosody",
                        lambda arr, sr, **kw: (seen.__setitem__("n", seen["n"] + 1) or arr))
    engine, _ = _mock_engine_capturing()
    engine.synthesize([{"type": "speech", "text": _LONG}], speed=0.88)
    assert seen["n"] == 0


def test_microprosody_called_per_chunk_when_flag_on(monkeypatch):
    import core.f5_tts.engine as eng
    seen = {"n": 0}
    monkeypatch.setattr(eng, "apply_microprosody",
                        lambda arr, sr, **kw: (seen.__setitem__("n", seen["n"] + 1) or arr))
    monkeypatch.setenv("MOODSCAPE_F5_MICROPROSODY", "1")
    engine, _ = _mock_engine_capturing()
    engine.synthesize([{"type": "speech", "text": _LONG}], speed=0.88)
    assert seen["n"] == 1


def test_microprosody_env_params_reach_apply(monkeypatch):
    """R1: pitch/formant/taper env overrides are forwarded to apply_microprosody."""
    import core.f5_tts.engine as eng
    captured = {}
    monkeypatch.setattr(eng, "apply_microprosody",
                        lambda arr, sr, **kw: (captured.update(kw) or arr))
    monkeypatch.setenv("MOODSCAPE_F5_MICROPROSODY", "1")
    monkeypatch.setenv("MOODSCAPE_F5_PITCH_SCALE", "1.25")
    monkeypatch.setenv("MOODSCAPE_F5_FORMANT_SHIFT", "0.97")
    monkeypatch.setenv("MOODSCAPE_F5_TAPER_MS", "250")
    monkeypatch.setenv("MOODSCAPE_F5_TAPER_FLOOR", "0.5")
    engine, _ = _mock_engine_capturing()
    engine.synthesize([{"type": "speech", "text": _LONG}], speed=0.88)
    assert captured["pitch_scale"] == 1.25
    assert captured["formant_shift"] == 0.97
    assert captured["taper_ms"] == 250.0
    assert captured["taper_floor"] == 0.5
