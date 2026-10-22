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


def test_408_is_transient():
    c = _client([_Err(408), b"\x00\x01"])
    assert _convert(c)[0] == b"\x00\x01"
    assert len(c._sdk.calls) == 2


@pytest.mark.parametrize("exc_type", [TypeError, AttributeError])
def test_programming_errors_are_not_retried(exc_type):
    calls = []
    def broken():
        calls.append(1)
        raise exc_type("bug")
    with pytest.raises(exc_type):
        with_retries(broken, attempts=3, sleep=lambda s: None)
    assert len(calls) == 1


class _TsData:
    def __init__(self, audio_b64, alignment=None, normalized_alignment=None):
        self.audio_base_64 = audio_b64
        self.alignment = alignment
        self.normalized_alignment = normalized_alignment


class _Align:
    def __init__(self, chars, starts, ends):
        self.characters = chars
        self.character_start_times_seconds = starts
        self.character_end_times_seconds = ends


class _RawTs:
    """Shape of elevenlabs 2.71 HttpResponse: plain object, .headers + .data (no context manager)."""
    def __init__(self, data, headers):
        self.data, self.headers = data, headers


def _ts_sdk(outcomes):
    sdk = _FakeSDK([])
    calls = sdk.calls

    def convert_with_timestamps(**kw):
        calls.append(kw)
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return _RawTs(out, {"request-id": f"rid{len(calls)}"})

    sdk.text_to_speech.with_raw_response.convert_with_timestamps = convert_with_timestamps
    return sdk


def _convert_al(c, **kw):
    return c.convert_with_alignment("[soft] Hi.", "v", model_id="eleven_v4", stability=0.5,
                                    similarity=0.75, seed=42, **kw)


def test_convert_with_alignment_returns_pcm_rid_alignment():
    import base64
    al = _Align(["H", "i"], [0.0, 0.1], [0.1, 0.2])
    sdk = _ts_sdk([_TsData(base64.b64encode(b"\x00\x01").decode(), alignment=al)])
    c = ElevenLabsClient(api_key="k", sdk=sdk, sleep=lambda s: None)
    pcm, rid, alignment = _convert_al(c, previous_request_ids=["a", "b", "c", "d"])
    assert pcm == b"\x00\x01" and rid == "rid1"
    assert alignment == {"characters": ["H", "i"], "starts": [0.0, 0.1], "ends": [0.1, 0.2]}
    kw = sdk.calls[0]
    assert kw["output_format"] == "pcm_48000" and kw["seed"] == 42
    assert kw["previous_request_ids"] == ["b", "c", "d"]
    assert kw["voice_settings"].stability == 0.5


def test_convert_with_alignment_falls_back_to_normalized():
    import base64
    al = _Align(["X"], [0.0], [0.1])
    sdk = _ts_sdk([_TsData(base64.b64encode(b"\x00\x01").decode(), normalized_alignment=al)])
    c = ElevenLabsClient(api_key="k", sdk=sdk, sleep=lambda s: None)
    assert _convert_al(c)[2]["characters"] == ["X"]


def test_convert_with_alignment_retries_like_convert():
    import base64
    data = _TsData(base64.b64encode(b"\x00\x01").decode(), alignment=_Align([], [], []))
    sdk = _ts_sdk([_Err(503), data])
    c = ElevenLabsClient(api_key="k", sdk=sdk, sleep=lambda s: None)
    assert _convert_al(c)[0] == b"\x00\x01"
    assert len(sdk.calls) == 2
    sdk2 = _ts_sdk([_Err(402), data])
    c2 = ElevenLabsClient(api_key="k", sdk=sdk2, sleep=lambda s: None)
    with pytest.raises(ElevenLabsAPIError, match="credits"):
        _convert_al(c2)
    assert len(sdk2.calls) == 1


def test_convert_with_alignment_empty_audio_is_error():
    sdk = _ts_sdk([_TsData(""), _TsData(""), _TsData("")])
    c = ElevenLabsClient(api_key="k", sdk=sdk, sleep=lambda s: None)
    with pytest.raises(ElevenLabsAPIError):
        _convert_al(c)
