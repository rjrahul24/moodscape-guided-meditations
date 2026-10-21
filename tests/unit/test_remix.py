import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core import mix_session as ms
from core.remix import remix, remix_output_path

SR = 48000


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts"))


def _pipeline_run(tmp_path, monkeypatch):
    from core.pipeline import MeditationPipeline
    from tests.unit.pipeline_fakes import FakeMusic, MockTTS24
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", FakeMusic)
    pipe = MeditationPipeline()
    out, _ = pipe.generate(script="Rest here. [pause:2s]\n\nLet go.", music_prompt="",
                           tts_engine="mock", custom_tts_engine=MockTTS24(), music_model="upload",
                           uploaded_music_path="dummy.wav")
    return pipe.last_run_id, out


def test_zero_trims_reproduce_original_exactly(tmp_path, monkeypatch):
    run_id, original = _pipeline_run(tmp_path, monkeypatch)
    new = remix(run_id, 0.0, 0.0)
    assert new != original and Path(original).exists()
    a, _ = sf.read(original); b, _ = sf.read(new)
    assert np.array_equal(a, b)


def _band_db(x, f):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    band = (freqs > f * 0.9) & (freqs < f * 1.1)
    return 20 * np.log10(spec[band].sum() + 1e-12)


def test_voice_trim_shifts_voice_to_music_ratio(tmp_path):
    t = np.arange(SR * 20) / SR
    voice = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    act = np.zeros(len(t), bool); act[: SR * 10] = True
    music = (0.1 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    out0 = str(tmp_path / "orig.wav")
    sf.write(out0, np.zeros(10), SR)
    ms.save_session("syn", voice, act, music, {
        "mix_sr": SR, "duck_amount_db": -16.0, "music_volume_db": -16.0,
        "fade_in_sec": 1.5, "fade_out_sec": 5.0, "stereo_output": False,
        "phrases": None, "bed_overrides": None, "target_lufs": -16.0, "mono_offset": -3.01,
        "output_format": "wav", "export_sr": SR, "use_upload": True,
        "content_type": "meditation", "env": {}, "output_path": out0})
    base, _ = sf.read(remix("syn", 0.0, 0.0))
    up, _ = sf.read(remix("syn", 6.0, 0.0))
    ratio0 = _band_db(base, 1000) - _band_db(base, 150)
    ratio6 = _band_db(up, 1000) - _band_db(up, 150)
    assert ratio6 - ratio0 == pytest.approx(6.0, abs=1.5)


def test_output_name_and_no_overwrite(tmp_path):
    orig = tmp_path / "a.wav"
    orig.write_bytes(b"x")
    p = remix_output_path(str(orig), 2.0, -3.0)
    assert p.name == "a_remix-v+2-m-3.wav"
    p.write_bytes(b"y")
    assert remix_output_path(str(orig), 2.0, -3.0).name == "a_remix-v+2-m-3-2.wav"


def test_missing_session_raises_clear_error():
    with pytest.raises(ms.MixSessionNotFound, match="generate again"):
        remix("gone", 1.0, 0.0)


def test_trims_out_of_range_rejected(tmp_path, monkeypatch):
    run_id, _ = _pipeline_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="-12"):
        remix(run_id, 13.0, 0.0)


def _synthetic_session(tmp_path, run_id, env):
    t = np.arange(SR * 4) / SR
    voice = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    act = np.zeros(len(t), bool); act[: SR * 2] = True
    music = np.full(len(t), 0.05, np.float32)
    out0 = str(tmp_path / f"{run_id}.wav")
    sf.write(out0, np.zeros(10), SR)
    params = {
        "mix_sr": SR, "duck_amount_db": -16.0, "music_volume_db": -16.0,
        "fade_in_sec": 0.5, "fade_out_sec": 0.5, "stereo_output": False,
        "phrases": None, "bed_overrides": None, "target_lufs": -16.0, "mono_offset": -3.01,
        "output_format": "wav", "export_sr": SR, "use_upload": True,
        "content_type": "meditation", "env": env, "output_path": out0}
    ms.save_session(run_id, voice, act, music, params)
    return params


def test_remix_env_swap_leaves_non_mix_keys_alone(tmp_path, monkeypatch):
    import core.remix as remix_mod
    _synthetic_session(tmp_path, "envs", {
        "MOODSCAPE_SCRIPT_PLANNER": "saved-planner", "MOODSCAPE_F5_CFG": "9.9"})
    monkeypatch.setenv("MOODSCAPE_SCRIPT_PLANNER", "live-planner")
    monkeypatch.setenv("MOODSCAPE_F5_CFG", "1.2")
    seen = {}
    real_mix = remix_mod.mix

    def spy(*a, **k):
        seen["planner"] = os.environ.get("MOODSCAPE_SCRIPT_PLANNER")
        seen["cfg"] = os.environ.get("MOODSCAPE_F5_CFG")
        return real_mix(*a, **k)

    monkeypatch.setattr(remix_mod, "mix", spy)
    remix("envs", 0.0, 0.0)
    assert seen == {"planner": "live-planner", "cfg": "1.2"}
    assert os.environ["MOODSCAPE_SCRIPT_PLANNER"] == "live-planner"
    assert os.environ["MOODSCAPE_F5_CFG"] == "1.2"


def test_remix_applies_saved_mix_keys_then_restores(tmp_path, monkeypatch):
    import core.remix as remix_mod
    _synthetic_session(tmp_path, "mixenv", {"MOODSCAPE_SHARED_REVERB_SEND_DB": "-30"})
    monkeypatch.delenv("MOODSCAPE_SHARED_REVERB_SEND_DB", raising=False)
    monkeypatch.setenv("MOODSCAPE_BED_RISE_LIMIT", "1")   # live, not in saved env
    seen = {}
    real_mix = remix_mod.mix

    def spy(*a, **k):
        seen["send"] = os.environ.get("MOODSCAPE_SHARED_REVERB_SEND_DB")
        seen["rise"] = os.environ.get("MOODSCAPE_BED_RISE_LIMIT")
        return real_mix(*a, **k)

    monkeypatch.setattr(remix_mod, "mix", spy)
    remix("mixenv", 0.0, 0.0)
    assert seen == {"send": "-30", "rise": None}
    assert "MOODSCAPE_SHARED_REVERB_SEND_DB" not in os.environ
    assert os.environ["MOODSCAPE_BED_RISE_LIMIT"] == "1"
