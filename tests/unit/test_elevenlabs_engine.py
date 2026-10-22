import numpy as np
import pytest

from core.elevenlabs_tts.client import ElevenLabsAPIError
from core.elevenlabs_tts.engine import ElevenLabsEngine, resolve_params

SR = 48000


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CACHE_DIR", str(tmp_path))


def _pcm(sec=0.5):
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16).tobytes()


def _alignment(text, sec=0.5):
    """Uniform per-character timing over the chunk, tag chars at zero duration."""
    tag_end = text.index("]") + 1 if text.startswith("[") else 0
    body = len(text) - tag_end
    step = sec / max(1, body)
    starts, ends = [], []
    for i in range(len(text)):
        if i < tag_end:
            starts.append(0.0); ends.append(0.0)
        else:
            j = i - tag_end
            starts.append(j * step); ends.append((j + 1) * step)
    return {"characters": list(text), "starts": starts, "ends": ends}


class FakeClient:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at
        self.methods: list[str] = []
    def convert(self, text, voice_id, **kw):
        self.calls.append(dict(text=text, voice_id=voice_id, **kw))
        self.methods.append("convert")
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise ElevenLabsAPIError(402, "ElevenLabs error 402: out of credits")
        return _pcm(), f"rid{len(self.calls)}"
    def convert_with_alignment(self, text, voice_id, **kw):
        pcm, rid = self.convert(text, voice_id, **kw)
        self.methods[-1] = "convert_with_alignment"
        return pcm, rid, _alignment(text)


def _segs(n):
    segs = []
    for i in range(n):
        segs.append({"type": "speech", "text": f"[soft, slow, calm voice] Line {i}."})
        segs.append({"type": "pause", "duration_sec": 1.0})
    return segs


def _engine(client):
    return ElevenLabsEngine(params=resolve_params(voice_id="v"), client=client)


def test_output_contract():
    audio, mask = _engine(FakeClient()).synthesize(_segs(2))
    assert audio.dtype == np.float32 and mask.dtype == bool
    assert len(audio) == len(mask)
    assert ElevenLabsEngine.sample_rate == 48000


def test_pauses_inactive_speech_active():
    audio, mask = _engine(FakeClient()).synthesize(_segs(1))
    assert mask[: int(0.4 * SR)].mean() > 0.9
    assert mask[int(0.75 * SR): int(1.3 * SR)].mean() < 0.05


def test_stitching_passes_at_most_three_ids_including_cache_hits():
    _engine(FakeClient()).synthesize(_segs(2))          # warm cache for lines 0-1
    client = FakeClient()
    _engine(client).synthesize(_segs(5))                 # lines 0-1 hit, 2-4 miss
    assert len(client.calls) == 3
    assert client.calls[0]["previous_request_ids"] == ["rid1", "rid2"]
    assert len(client.calls[-1]["previous_request_ids"]) == 3


def test_stitch_off(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_STITCH", "0")
    client = FakeClient()
    _engine(client).synthesize(_segs(2))
    assert all(c["previous_request_ids"] is None for c in client.calls)


def test_manifest_counts_billing():
    eng = _engine(FakeClient())
    eng.synthesize(_segs(2))
    m = eng.last_manifest
    assert m["cache_misses"] == 2 and m["cache_hits"] == 0
    assert m["chars_billed"] == sum(len(f"[soft, slow, calm voice] Line {i}.") for i in range(2))
    assert m["model_id"] == "eleven_v4" and m["voice_id"] == "v"
    eng2 = _engine(FakeClient())
    eng2.synthesize(_segs(2))
    assert eng2.last_manifest["chars_billed"] == 0 and eng2.last_manifest["cache_hits"] == 2


def test_midrun_failure_reports_progress_and_resume_is_cheap():
    with pytest.raises(ElevenLabsAPIError, match="4 of 9 chunks rendered") as ei:
        _engine(FakeClient(fail_at=5)).synthesize(_segs(9))
    assert ei.value.status == 402
    client = FakeClient()
    _engine(client).synthesize(_segs(9))
    assert len(client.calls) == 5


def test_speed_ignored_and_v4_settings_sent():
    client = FakeClient()
    _engine(client).synthesize(_segs(1), speed=0.7)
    call = client.calls[0]
    assert call["model_id"] == "eleven_v4" and call["seed"] == 42
    assert call["stability"] == 0.5 and call["similarity"] == 0.75
    assert "speed" not in call


def test_cache_extra_contains_all_render_params():
    p = resolve_params(voice_id="v", stability=0.4)
    extra = p.cache_extra("[warm]")
    assert extra == {"voice_id": "v", "stability": 0.4, "similarity": 0.75,
                     "model_id": "eleven_v4", "seed": 42, "direction": "[warm]"}


class StaleIdsClient(FakeClient):
    """Rejects any request that carries previous_request_ids (stale/expired ids)."""
    def __init__(self, status=422, always=False):
        super().__init__()
        self.status, self.always = status, always
    def convert(self, text, voice_id, **kw):
        if self.always or kw.get("previous_request_ids"):
            self.calls.append(dict(text=text, voice_id=voice_id, **kw))
            raise ElevenLabsAPIError(self.status, f"ElevenLabs error {self.status}: bad ids")
        return super().convert(text, voice_id, **kw)


@pytest.mark.parametrize("status", [400, 422])
def test_stale_request_ids_retry_once_without_ids(status, caplog):
    _engine(FakeClient()).synthesize(_segs(1))          # line 0 cached with an old rid
    client = StaleIdsClient(status)
    audio, _ = _engine(client).synthesize(_segs(3))     # line 0 hit, lines 1-2 miss
    assert len(audio) > 0
    assert len(client.calls) == 3
    assert client.calls[0]["previous_request_ids"] == ["rid1"]   # rejected
    assert client.calls[1]["previous_request_ids"] is None       # same chunk, retried
    assert client.calls[1]["text"] == client.calls[0]["text"]
    assert client.calls[2]["previous_request_ids"] is None       # next chunk: no ids
    assert "previous_request_ids" in caplog.text


def test_422_without_ids_still_raises():
    client = StaleIdsClient(422, always=True)
    with pytest.raises(ElevenLabsAPIError) as ei:
        _engine(client).synthesize(_segs(2))
    assert ei.value.status == 422
    assert len(client.calls) == 1


def test_other_errors_with_ids_are_not_retried():
    _engine(FakeClient()).synthesize(_segs(1))
    client = StaleIdsClient(402)
    with pytest.raises(ElevenLabsAPIError) as ei:
        _engine(client).synthesize(_segs(2))
    assert ei.value.status == 402
    assert len(client.calls) == 1


# --- pacing (timestamp-driven pauses + gentle stretch; opt-in experiment) ----

@pytest.fixture
def pacing_on(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_PACING", "1")


def test_default_is_legacy_convert_and_plain_payload(monkeypatch):
    from core.elevenlabs_tts import chunk_cache
    monkeypatch.delenv("MOODSCAPE_ELEVENLABS_PACING", raising=False)
    client = FakeClient()
    eng = _engine(client)
    eng.synthesize(_segs(1))
    assert client.methods == ["convert"]
    assert eng.last_manifest["pacing"] == {"enabled": False}
    text = "[soft, slow, calm voice] Line 0."
    plain = chunk_cache.chunk_payload(text, "v", "eleven_v4", 0.5, 0.75, 42)
    assert "alignment" not in plain and chunk_cache.is_cached(plain)

def _sentences(n):
    segs = []
    for i in range(n):
        segs.append({"type": "speech", "text": f"[soft, slow, calm voice] Rest {i}. Let go, now."})
        segs.append({"type": "pause", "duration_sec": 1.0})
    return segs


def test_pacing_on_uses_alignment_endpoint_and_caches_it(pacing_on):
    client = FakeClient()
    eng = _engine(client)
    audio, mask = eng.synthesize(_sentences(2))
    assert client.methods == ["convert_with_alignment"] * 2
    assert len(audio) == len(mask)
    assert eng.last_manifest["pacing"] == {"stretch": 0.9, "sentence_sec": 0.9, "clause_sec": 0.35}
    client2 = FakeClient()
    eng2 = _engine(client2)
    audio2, mask2 = eng2.synthesize(_sentences(2))
    assert client2.calls == []
    assert eng2.last_manifest["cache_hits"] == 2
    assert len(audio2) == len(audio)          # cache hit still paces
    np.testing.assert_allclose(audio2, audio, atol=1e-6)


def test_pacing_lengthens_speech_by_stretch_and_gaps(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_PACING", "0")
    off, _ = _engine(FakeClient()).synthesize(_sentences(1))
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_PACING", "1")
    on, mask = _engine(FakeClient()).synthesize(_sentences(1))
    speech = int(0.5 * SR)
    expected_extra = round(speech / 0.9) - speech + int(0.9 * SR) + int(0.35 * SR)
    assert abs((len(on) - len(off)) - expected_extra) <= 16
    # the inserted sentence gap reads inactive so the bed can rise there
    assert (~mask[: int(1.5 * SR)]).sum() > int(0.6 * SR)


def test_pacing_off_is_legacy_behaviour(monkeypatch):
    from core.elevenlabs_tts.assembly import (
        activity_mask, crossfade_concat, normalize_chunk_lufs, pcm16_to_float,
    )
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_PACING", "0")
    client = FakeClient()
    eng = _engine(client)
    audio, mask = eng.synthesize(_sentences(2))
    assert client.methods == ["convert"] * 2
    assert eng.last_manifest["pacing"] == {"enabled": False}
    y = normalize_chunk_lufs(pcm16_to_float(_pcm()), SR, -21.0)
    pause = np.zeros(SR, dtype=np.float32)
    pieces = [y, pause, y, pause]
    masks = [activity_mask(y, SR), np.zeros(SR, bool)] * 2
    exp_audio, exp_mask = crossfade_concat(pieces, masks, SR, 30.0)
    np.testing.assert_array_equal(audio, exp_audio)
    np.testing.assert_array_equal(mask, exp_mask)


def test_pacing_off_and_on_use_separate_cache_entries(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_PACING", "0")
    _engine(FakeClient()).synthesize(_sentences(1))
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_PACING", "1")
    client = FakeClient()
    _engine(client).synthesize(_sentences(1))
    assert client.methods == ["convert_with_alignment"]


def test_cache_extra_unchanged_when_pacing_off(monkeypatch):
    monkeypatch.delenv("MOODSCAPE_ELEVENLABS_PACING", raising=False)
    assert "pacing" not in resolve_params(voice_id="v").cache_extra("[warm]")


def test_cache_extra_includes_pacing_when_on(pacing_on):
    assert resolve_params(voice_id="v").cache_extra("[warm]")["pacing"] == {
        "stretch": 0.9, "sentence_sec": 0.9, "clause_sec": 0.35}
