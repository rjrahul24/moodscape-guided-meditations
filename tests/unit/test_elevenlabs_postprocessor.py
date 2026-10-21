import numpy as np
from pedalboard import Convolution, Gain, HighpassFilter, Limiter, LowpassFilter, Mix, Pedalboard

from core.elevenlabs_tts.postprocessor import build_elevenlabs_voice_chain, deess_elevenlabs
from core.f5_tts.postprocessor import build_f5_voice_chain

SR = 48000


def _noise(sec=1.0):
    return (np.random.default_rng(0).standard_normal(int(SR * sec)) * 0.1).astype(np.float32)


def _old_f5_chain(reverb_amount=0.15, ir_name="warm_studio"):
    from core.audio_processor import DEFAULT_IR, IR_CATALOG
    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    ir_path = IR_CATALOG.get(ir_name, IR_CATALOG[DEFAULT_IR])["path"]
    dry_db = 20 * np.log10(max(1.0 - reverb_amount, 1e-5))
    wet_db = 20 * np.log10(max(reverb_amount, 1e-5))
    return Pedalboard([
        Mix([Gain(gain_db=dry_db), Pedalboard([
            Convolution(impulse_response_filename=ir_path, mix=1.0),
            HighpassFilter(cutoff_frequency_hz=300), LowpassFilter(cutoff_frequency_hz=6000),
            Gain(gain_db=wet_db)])]),
        Limiter(threshold_db=-1.0),
    ])


def test_f5_chain_refactor_is_bit_identical():
    x = _noise().reshape(1, -1)
    assert np.array_equal(build_f5_voice_chain()(x, SR), _old_f5_chain()(x, SR))


def test_dry_chain_is_hpf_only():
    chain = build_elevenlabs_voice_chain(0.0)
    assert len(chain) == 1 and isinstance(chain[0], HighpassFilter)


def test_reverb_adds_parallel_mix_and_no_limiter():
    chain = build_elevenlabs_voice_chain(0.2)
    assert any(isinstance(p, Mix) for p in chain)
    assert not any(isinstance(p, Limiter) for p in chain)


def test_deess_toggle(monkeypatch):
    x = _noise()
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DEESS", "0")
    assert deess_elevenlabs(x, SR) is x
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DEESS", "1")
    y = deess_elevenlabs(x, SR)
    assert y.shape == x.shape and not np.array_equal(x, y)
