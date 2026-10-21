import pytest

from core.elevenlabs_tts import chunk_cache


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CACHE_DIR", str(tmp_path))


def _p(**kw):
    base = dict(text="[soft] Hi.", voice_id="v", model_id="eleven_v4",
                stability=0.5, similarity=0.75, seed=42)
    base.update(kw)
    return chunk_cache.chunk_payload(**base)


def test_round_trip_with_request_id():
    chunk_cache.store(_p(), b"\x01\x02", "req-1")
    assert chunk_cache.load(_p()) == (b"\x01\x02", "req-1")
    assert chunk_cache.is_cached(_p())


@pytest.mark.parametrize("change", [dict(text="x"), dict(voice_id="w"),
                                    dict(stability=0.6), dict(similarity=0.7),
                                    dict(seed=1), dict(model_id="eleven_v3")])
def test_any_setting_change_misses(change):
    chunk_cache.store(_p(), b"\x01\x02", "req-1")
    assert chunk_cache.load(_p(**change)) is None


def test_payload_has_no_request_ids():
    assert "previous_request_ids" not in _p()
    assert _p()["output_format"] == "pcm_48000"
