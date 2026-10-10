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


def test_default_uses_streaming_4s_chunks(model, fake_mx, monkeypatch):
    monkeypatch.delenv("MOODSCAPE_DF_STREAMING", raising=False)
    dfe.enhance_voice_deepfilter(_audio(), sr=SR)
    assert model.calls == [("streaming", 4 * SR)]
    assert fake_mx.cleared == 1


def test_flag_one_streams_flag_zero_offline(model, fake_mx, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_DF_STREAMING", "1")
    dfe.enhance_voice_deepfilter(_audio(), sr=SR)
    monkeypatch.setenv("MOODSCAPE_DF_STREAMING", "0")
    dfe.enhance_voice_deepfilter(_audio(), sr=SR)
    assert [c[0] for c in model.calls] == ["streaming", "offline"]
    assert fake_mx.cleared == 2


def test_clear_cache_error_is_swallowed(model, fake_mx, monkeypatch):
    def boom():
        raise RuntimeError("no metal")

    fake_mx.clear_cache = boom
    out = dfe.enhance_voice_deepfilter(_audio(), sr=SR)
    assert out.shape == _audio().shape
    assert model.calls  # DF path was used, not the spectral fallback


@pytest.mark.parametrize("flag", ["0", "1"])
def test_peak_preserved_and_blend_unchanged(model, monkeypatch, flag):
    monkeypatch.setenv("MOODSCAPE_DF_STREAMING", flag)
    a = _audio()
    full = dfe.enhance_voice_deepfilter(a, sr=SR, wet=1.0)
    assert np.abs(full).max() == pytest.approx(np.abs(a).max(), rel=1e-4)
    half = dfe.enhance_voice_deepfilter(a, sr=SR, wet=0.5)
    expected = dfe._blend_wet_dry(a, full, 0.5)
    np.testing.assert_allclose(half, expected, atol=1e-5)
