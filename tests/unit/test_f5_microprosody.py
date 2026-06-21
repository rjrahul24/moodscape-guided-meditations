"""B4: F5 pyworld microprosody — phrase-final pitch declination + breathiness.
R1: pitch-range widening, formant warmth, phrase-final amplitude taper."""

import numpy as np

from core.f5_tts.postprocessor import (
    _apply_amplitude_taper,
    _apply_f0_declination,
    _match_peak,
    _warp_formants,
    _widen_pitch,
    apply_microprosody,
)


def test_declination_lowers_tail_keeps_head():
    """The exponential decay drops the final frames toward the target factor
    and leaves the head untouched."""
    n = 1000
    f0 = np.full(n, 200.0)                 # constant, all voiced
    frame_period_s = 0.005                 # 5 ms (pyworld default)
    decline_cents = 120.0                  # ~1.2 semitones
    out = _apply_f0_declination(
        f0, frame_period_s, tail_ms=600.0, decline_cents=decline_cents,
    )
    self_factor = 2.0 ** (-decline_cents / 1200.0)
    # Head unchanged.
    assert abs(out[0] - 200.0) < 1e-6
    # Final frame near 200 * factor.
    assert abs(out[-1] - 200.0 * self_factor) < 1e-3
    # Monotonic non-increasing across the tail.
    tail = out[-120:]
    assert np.all(np.diff(tail) <= 1e-9)


def test_declination_only_affects_voiced_frames():
    f0 = np.full(400, 180.0)
    f0[-50:] = 0.0                         # unvoiced tail
    out = _apply_f0_declination(f0, 0.005, tail_ms=200.0, decline_cents=200.0)
    # Unvoiced frames stay exactly zero.
    assert np.all(out[-50:] == 0.0)


def test_apply_microprosody_preserves_length():
    sr = 24000
    t = np.arange(int(2.0 * sr)) / sr
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    out = apply_microprosody(audio, sr, decline_cents=150.0, ap_scale=1.05)
    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))


def test_apply_microprosody_noop_without_pyworld(monkeypatch):
    import core.f5_tts.postprocessor as pp
    monkeypatch.setattr(pp, "_check_pyworld", lambda: False)
    sr = 24000
    audio = (0.3 * np.random.randn(sr)).astype(np.float32)
    out = pp.apply_microprosody(audio, sr)
    np.testing.assert_array_equal(out, audio)


# ── R1: pitch-range widening ────────────────────────────────────────────────

def test_widen_pitch_scales_deviation_about_mean():
    f0 = np.array([180.0, 200.0, 220.0])      # mean 200
    out = _widen_pitch(f0, 2.0)
    np.testing.assert_allclose(out, [160.0, 200.0, 240.0], atol=1e-6)


def test_widen_pitch_constant_contour_unchanged():
    f0 = np.full(5, 200.0)
    np.testing.assert_allclose(_widen_pitch(f0, 1.3), f0, atol=1e-6)


def test_widen_pitch_leaves_unvoiced_zero():
    f0 = np.array([0.0, 200.0, 0.0, 220.0])
    out = _widen_pitch(f0, 1.5)
    assert out[0] == 0.0 and out[2] == 0.0
    assert out[1] > 0.0 and out[3] > 0.0


def test_widen_pitch_clamps_positive():
    f0 = np.array([50.0, 400.0])              # mean 225; large scale would go negative
    out = _widen_pitch(f0, 5.0)
    assert np.all(out[f0 > 0] > 0.0)


# ── R1: formant warmth ──────────────────────────────────────────────────────

def test_warp_formants_identity_at_one():
    rng = np.random.default_rng(0)
    sp = rng.random((10, 64)) + 0.1
    np.testing.assert_array_equal(_warp_formants(sp, 1.0), sp)


def test_warp_formants_changes_envelope_and_keeps_shape():
    rng = np.random.default_rng(1)
    sp = rng.random((10, 64)) + 0.1
    warped = _warp_formants(sp, 0.98)
    assert warped.shape == sp.shape
    assert not np.array_equal(warped, sp)


# ── R1: phrase-final amplitude taper ────────────────────────────────────────

def test_amplitude_taper_ramps_tail_keeps_head():
    sr = 24000
    audio = np.ones(sr, dtype=np.float32)
    out = _apply_amplitude_taper(audio, sr, taper_ms=300.0, floor=0.6)
    assert abs(out[0] - 1.0) < 1e-6
    assert abs(out[-1] - 0.6) < 1e-3
    tail = out[-int(0.3 * sr):]
    assert np.all(np.diff(tail) <= 1e-7)


def test_amplitude_taper_noop_when_zero_length_or_unity_floor():
    sr = 24000
    audio = np.ones(sr, dtype=np.float32)
    np.testing.assert_array_equal(_apply_amplitude_taper(audio, sr, 0.0, 0.6), audio)
    np.testing.assert_array_equal(_apply_amplitude_taper(audio, sr, 300.0, 1.0), audio)


# ── R1: peak preservation (WORLD resynthesis must not inject gain) ──────────

def test_match_peak_scales_to_target():
    audio = np.array([0.5, -0.25, 0.1], dtype=np.float32)
    out = _match_peak(audio, 0.3)
    assert abs(float(np.max(np.abs(out))) - 0.3) < 1e-6


def test_match_peak_silence_unchanged():
    audio = np.zeros(10, dtype=np.float32)
    np.testing.assert_array_equal(_match_peak(audio, 0.3), audio)


def test_apply_microprosody_does_not_inflate_peak():
    """The WORLD pass must not push the chunk hotter than it came in."""
    import pytest
    from core.f5_tts.postprocessor import _check_pyworld
    if not _check_pyworld():
        pytest.skip("pyworld unavailable")
    sr = 24000
    t = np.arange(int(2.0 * sr)) / sr
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    in_peak = float(np.max(np.abs(audio)))
    out = apply_microprosody(audio, sr, pitch_scale=1.3, formant_shift=0.97)
    assert float(np.max(np.abs(out))) <= in_peak * 1.02


def test_apply_microprosody_lowers_measured_tail_pitch():
    """End-to-end: a steady-pitch tone comes out with a lower pitch at the
    tail than at the head."""
    import pytest
    from core.f5_tts.postprocessor import _check_pyworld
    if not _check_pyworld():
        pytest.skip("pyworld unavailable")
    import pyworld as pw

    sr = 24000
    t = np.arange(int(2.0 * sr)) / sr
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    out = apply_microprosody(audio, sr, decline_cents=400.0, tail_ms=500.0)

    f0, tt = pw.harvest(out.astype(np.float64), sr)
    voiced = f0[f0 > 0]
    head = f0[: len(f0) // 4]
    tail = f0[-len(f0) // 4:]
    head_v = np.median(head[head > 0])
    tail_v = np.median(tail[tail > 0])
    assert tail_v < head_v
