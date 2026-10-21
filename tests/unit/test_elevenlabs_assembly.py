import numpy as np

from core.elevenlabs_tts.assembly import (
    activity_mask, chunk_lufs_target, crossfade_concat, normalize_chunk_lufs, pcm16_to_float,
)

SR = 48000


def _tone(sec, amp=0.3, f=220.0):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_pcm16_to_float():
    pcm = np.array([0, 16384, -32768], dtype=np.int16).tobytes()
    assert np.allclose(pcm16_to_float(pcm), [0.0, 0.5, -1.0])


def test_crossfade_lengths_and_mask_alignment():
    a, b = _tone(1.0), np.zeros(SR, dtype=np.float32)
    ma, mb = np.ones(SR, bool), np.zeros(SR, bool)
    out, m = crossfade_concat([a, b, a], [ma, mb, ma], SR, 30.0)
    xn = int(SR * 0.03)
    assert len(out) == 3 * SR - 2 * xn
    assert len(m) == len(out) and out.dtype == np.float32


def test_mask_marks_inner_gap_inactive():
    y = np.concatenate([_tone(1.0), np.zeros(SR, np.float32), _tone(1.0)])
    m = activity_mask(y, SR)
    assert m[: int(0.9 * SR)].mean() > 0.95
    assert m[int(1.3 * SR): int(1.9 * SR)].mean() < 0.05


def test_mask_all_false_for_silent_chunk():
    assert not activity_mask(np.zeros(SR, np.float32), SR).any()


def test_chunk_lufs_normalizes_and_passes_short_or_silent():
    y = _tone(2.0, amp=0.05)
    out = normalize_chunk_lufs(y, SR, -21.0)
    assert np.max(np.abs(out)) > np.max(np.abs(y))
    short = _tone(0.2)
    assert normalize_chunk_lufs(short, SR, -21.0) is short
    quiet = _tone(2.0, amp=1e-5)
    assert normalize_chunk_lufs(quiet, SR, -21.0) is quiet


def test_chunk_lufs_target_env(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CHUNK_LUFS", "off")
    assert chunk_lufs_target() is None
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CHUNK_LUFS", "-20")
    assert chunk_lufs_target() == -20.0
