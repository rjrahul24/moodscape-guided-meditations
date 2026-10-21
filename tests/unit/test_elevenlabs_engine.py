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


class FakeClient:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at
    def convert(self, text, voice_id, **kw):
        self.calls.append(dict(text=text, voice_id=voice_id, **kw))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise ElevenLabsAPIError(402, "ElevenLabs error 402: out of credits")
        return _pcm(), f"rid{len(self.calls)}"


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
