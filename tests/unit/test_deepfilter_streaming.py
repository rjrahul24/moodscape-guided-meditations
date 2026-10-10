"""DeepFilterNet streaming switch, MLX cache release, peak-preserve/blend."""

import sys
import types

import numpy as np
import pytest

import core.deepfilter_enhancer as dfe

SR = 48000


class FakeModel:
    def __init__(self, scale=0.5):
        self.calls = []
        self.scale = scale

    def enhance_array(self, audio):
        self.calls.append(("offline", None))
        return audio * self.scale

    def enhance_array_streaming(self, audio, chunk_samples=None, **kw):
        self.calls.append(("streaming", chunk_samples))
        return audio * self.scale


@pytest.fixture
def fake_mx(monkeypatch):
    mx = types.ModuleType("mlx.core")
    mx.cleared = 0

    def clear_cache():
        mx.cleared += 1

    mx.clear_cache = clear_cache
    pkg = types.ModuleType("mlx")
    pkg.core = mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return mx


@pytest.fixture
def model(monkeypatch, fake_mx):
    m = FakeModel()
    monkeypatch.setattr(dfe, "_is_available", lambda: True)
    monkeypatch.setattr(dfe, "_load_model", lambda: m)
    return m


def _audio():
    rng = np.random.default_rng(0)
    return (rng.standard_normal(SR * 2) * 0.1).astype(np.float32)


def test_default_uses_offline(model, fake_mx, monkeypatch):
    monkeypatch.delenv("MOODSCAPE_DF_STREAMING", raising=False)
    dfe.enhance_voice_deepfilter(_audio(), sr=SR)
    assert model.calls == [("offline", None)]
    assert fake_mx.cleared == 1


def test_flag_one_streams_flag_zero_offline(model, fake_mx, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_DF_STREAMING", "1")
    dfe.enhance_voice_deepfilter(_audio(), sr=SR)
    monkeypatch.setenv("MOODSCAPE_DF_STREAMING", "0")
    dfe.enhance_voice_deepfilter(_audio(), sr=SR)
    assert model.calls == [("streaming", 4 * SR), ("offline", None)]
    assert fake_mx.cleared == 2


def test_streaming_requested_but_missing_falls_back_to_offline(fake_mx, monkeypatch):
    class OfflineOnly:
        calls = 0

        def enhance_array(self, audio):
            OfflineOnly.calls += 1
            return audio * 0.5

    monkeypatch.setattr(dfe, "_is_available", lambda: True)
    monkeypatch.setattr(dfe, "_load_model", lambda: OfflineOnly())
    monkeypatch.setenv("MOODSCAPE_DF_STREAMING", "1")
    dfe.enhance_voice_deepfilter(_audio(), sr=SR)
    assert OfflineOnly.calls == 1


def test_clear_cache_runs_on_failure_path(fake_mx, monkeypatch):
    class Broken:
        def enhance_array(self, audio):
            raise RuntimeError("boom")

    monkeypatch.setattr(dfe, "_is_available", lambda: True)
    monkeypatch.setattr(dfe, "_load_model", lambda: Broken())
    monkeypatch.setattr(dfe, "_fallback_spectral_denoise", lambda a, sr, wet=1.0: a)
    dfe.enhance_voice_deepfilter(_audio(), sr=SR)
    assert fake_mx.cleared == 1


def test_clear_cache_error_is_swallowed(model, fake_mx, monkeypatch):
    def boom():
        raise RuntimeError("no metal")

    fake_mx.clear_cache = boom
    monkeypatch.setattr(
        dfe, "_fallback_spectral_denoise",
        lambda *a, **k: pytest.fail("fell back to spectral gating"),
    )
    a = _audio()
    out = dfe.enhance_voice_deepfilter(a, sr=SR)
    assert model.calls
    # FakeModel scales by 0.5, then peak-preserve restores the original peak
    np.testing.assert_allclose(out, a, atol=1e-5)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_peak_preserved_and_blend_unchanged(model, monkeypatch, flag):
    monkeypatch.setenv("MOODSCAPE_DF_STREAMING", flag)
    a = _audio()
    full = dfe.enhance_voice_deepfilter(a, sr=SR, wet=1.0)
    assert np.abs(full).max() == pytest.approx(np.abs(a).max(), rel=1e-4)
    half = dfe.enhance_voice_deepfilter(a, sr=SR, wet=0.5)
    expected = dfe._blend_wet_dry(a, full, 0.5)
    np.testing.assert_allclose(half, expected, atol=1e-5)
