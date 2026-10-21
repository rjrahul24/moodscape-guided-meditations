import pytest

from core.elevenlabs_tts.client import (
    ElevenLabsAPIError, ElevenLabsClient, ElevenLabsConfigError, with_retries,
)


class _Err(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


class _Resp:
    def __init__(self, data, headers):
        self.data, self.headers = data, headers
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class _FakeSDK:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        sdk = self
        class _Raw:
            def convert(self, **kw):
                sdk.calls.append(kw)
                out = sdk.outcomes.pop(0)
                if isinstance(out, Exception):
                    raise out
                return _Resp([out], {"request-id": f"rid{len(sdk.calls)}"})
        class _TTS:
            with_raw_response = _Raw()
        class _Voices:
            def get(self, voice_id):
                if voice_id == "missing":
                    raise _Err(404)
                return object()
        self.text_to_speech = _TTS()
        self.voices = _Voices()


def _client(outcomes):
    return ElevenLabsClient(api_key="k", sdk=_FakeSDK(outcomes), sleep=lambda s: None)


def _convert(c, **kw):
    return c.convert("[soft] Hi.", "v", model_id="eleven_v4", stability=0.5,
                     similarity=0.75, seed=42, **kw)


def test_missing_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVEN_LABS_API_KEY", raising=False)
    with pytest.raises(ElevenLabsConfigError, match="ELEVENLABS_API_KEY"):
        ElevenLabsClient(sdk=object())


def test_returns_pcm_and_request_id_and_sends_v4_params():
    c = _client([b"\x00\x01"])
    assert _convert(c, previous_request_ids=["a", "b", "c", "d"]) == (b"\x00\x01", "rid1")
    kw = c._sdk.calls[0]
    assert kw["output_format"] == "pcm_48000" and kw["model_id"] == "eleven_v4"
    assert kw["previous_request_ids"] == ["b", "c", "d"]
    vs = kw["voice_settings"]
    assert vs.stability == 0.5 and vs.similarity_boost == 0.75
    sent = {k for k, v in vars(vs).items() if v is not None}
    assert "speed" not in sent and "style" not in sent


@pytest.mark.parametrize("status", [None, 429, 500, 503])
def test_transient_errors_retry(status):
    c = _client([_Err(status), b"\x00\x01"])
    assert _convert(c)[0] == b"\x00\x01"
    assert len(c._sdk.calls) == 2


@pytest.mark.parametrize("status,needle", [(401, "API key"), (402, "credits"), (422, "422")])
def test_client_errors_fail_fast(status, needle):
    c = _client([_Err(status), b"never"])
    with pytest.raises(ElevenLabsAPIError, match=needle) as ei:
        _convert(c)
    assert ei.value.status == status
    assert len(c._sdk.calls) == 1


def test_retry_budget_exhausts():
    calls = []
    def boom():
        calls.append(1)
        raise _Err(503)
    with pytest.raises(ElevenLabsAPIError):
        with_retries(boom, attempts=3, sleep=lambda s: None)
    assert len(calls) == 3


def test_voice_exists():
    c = _client([])
    assert c.voice_exists("ok") is True
    assert c.voice_exists("missing") is False


def test_voice_exists_retries_transient():
    c = _client([])
    state = {"n": 0}

    def flaky(voice_id):
        state["n"] += 1
        if state["n"] == 1:
            raise _Err(503)
        return object()

    c._sdk.voices.get = flaky
    assert c.voice_exists("ok") is True
    assert state["n"] == 2


def test_key_name_fallback(monkeypatch):
    from core.elevenlabs_tts.client import get_api_key, has_api_key
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("ELEVEN_LABS_API_KEY", "abc")
    assert get_api_key() == "abc" and has_api_key()
    monkeypatch.setenv("ELEVENLABS_API_KEY", "primary")
    assert get_api_key() == "primary"
