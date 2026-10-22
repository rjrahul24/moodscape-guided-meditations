import pytest
import re

from core.elevenlabs_tts.direction import DEFAULT_DIRECTION
from core.elevenlabs_tts.preprocessor import (
    HARD_MAX_CHARS, estimate_billed_chars, prepare_segments,
)

SCRIPT = (
    "Find a comfortable position. Let your eyes close.\n\n"
    "Notice your breath. [pause:4s]\n\n"
    "[inhale] Breathe in slowly. [exhale] And let it go."
)


def _speech(segs):
    return [s["text"] for s in segs if s["type"] == "speech"]


def test_only_speech_and_pause_types():
    assert {s["type"] for s in prepare_segments(SCRIPT)} == {"speech", "pause"}


def test_direction_prefixes_every_chunk():
    for text in _speech(prepare_segments(SCRIPT)):
        assert text.startswith(DEFAULT_DIRECTION + " ")


def test_breath_markers_become_pauses_of_breath_length():
    segs = prepare_segments(SCRIPT)
    pauses = [s["duration_sec"] for s in segs if s["type"] == "pause"]
    assert 1.5 in pauses and 1.8 in pauses
    assert 4.0 in pauses
    assert 3.0 in pauses  # meditation paragraph pause


def test_sleep_story_paragraph_pause():
    segs = prepare_segments("One.\n\nTwo.", content_type="sleep_story")
    assert [s["duration_sec"] for s in segs if s["type"] == "pause"] == [2.0]


def test_no_sound_or_reaction_tags_reach_api_text():
    for text in _speech(prepare_segments(SCRIPT)):
        body = text[len(DEFAULT_DIRECTION):]
        assert "[" not in body
        assert not re.search(r"\b(breath|inhale|exhale|sigh)", text.split("]", 1)[0])


def test_oversize_paragraph_splits_under_hard_ceiling(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS", "1500")
    para = " ".join(f"Sentence number {i} drifts by softly." for i in range(400))
    texts = _speech(prepare_segments(para))
    assert len(texts) > 1
    assert all(len(t) <= 1500 for t in texts)
    assert all(len(t) <= HARD_MAX_CHARS for t in texts)


def test_never_splits_inside_brackets(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS", "200")
    para = ("Rest here. " * 8) + "[keep this. Together] " + ("Rest here. " * 8)
    for text in _speech(prepare_segments(para)):
        body = text[len(DEFAULT_DIRECTION):]
        assert body.count("[") == body.count("]")


def test_numbers_expanded():
    assert "four" in _speech(prepare_segments("Count to 4."))[0].lower()


def test_estimate_billed_chars():
    segs = prepare_segments(SCRIPT)
    assert estimate_billed_chars(segs) == sum(len(t) for t in _speech(segs))


def test_custom_direction_applied():
    texts = _speech(prepare_segments("Hello there.", direction="warm, low voice"))
    assert texts == ["[warm, low voice] Hello there."]


def test_user_typed_tags_are_stripped():
    texts = _speech(prepare_segments("Rest now. [sighs] Soft. [laughs] ok [whisper: hi] end."))
    assert texts
    for text in texts:
        body = text[len(DEFAULT_DIRECTION):]
        assert "[" not in body and "]" not in body
    joined = " ".join(texts)
    for word in ("Rest now", "Soft", "end"):
        assert word in joined


def test_unbroken_token_longer_than_limit_is_sliced(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS", "1500")
    texts = _speech(prepare_segments("a" * 12000))
    assert len(texts) > 1
    assert all(len(t) <= 1500 and len(t) <= HARD_MAX_CHARS for t in texts)


def test_malformed_chunk_env_names_variable(monkeypatch):
    import pytest
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS", "lots")
    with pytest.raises(ValueError, match="MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS"):
        prepare_segments("Hello.")


@pytest.mark.parametrize("script", [
    "Hi there [laughs [voice:brittney] ] and rest.",
    "Hello [laughs softly there.",
    "Rest [[laughs]] now.",
])
def test_unbalanced_or_reassembled_tags_never_reach_api(script):
    speech = _speech(prepare_segments(script))
    assert speech
    for text in speech:
        body = text[len(DEFAULT_DIRECTION):]
        assert "[" not in body and "]" not in body


def test_empty_direction_falls_back_to_env_default(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION", "[warm, low voice]")
    segs = prepare_segments("Hello there.", direction="")
    assert _speech(segs)[0].startswith("[warm, low voice] ")


def test_package_exports_postprocessor_and_cache():
    import core.elevenlabs_tts as pkg
    assert callable(pkg.deess_elevenlabs)
    assert callable(pkg.build_elevenlabs_voice_chain)
    assert hasattr(pkg.chunk_cache, "chunk_payload")


# ── experiments: sentence gaps + direction mode (env-only, default off) ──────

SENTENCES = ("One calm sentence. Two calm sentence. Three calm sentence. "
             "[pause:5s] Four after pause. Five after pause. [pause:1s] Six.")


def _kinds(segs):
    return [("P%g" % s["duration_sec"]) if s["type"] == "pause" else "S" for s in segs]


def test_defaults_unchanged_sentences_stay_packed(monkeypatch):
    monkeypatch.delenv("MOODSCAPE_ELEVENLABS_SENTENCE_GAP_MS", raising=False)
    monkeypatch.delenv("MOODSCAPE_ELEVENLABS_DIRECTION_MODE", raising=False)
    segs = prepare_segments(SENTENCES)
    assert _kinds(segs) == ["S", "P5", "S", "P1", "S"]
    assert all(t.startswith(DEFAULT_DIRECTION + " ") for t in _speech(segs))


def test_sentence_gap_splits_chunks_with_silence(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_SENTENCE_GAP_MS", "700")
    segs = prepare_segments(SENTENCES)
    assert _kinds(segs) == ["S", "P0.7", "S", "P0.7", "S", "P5", "S", "P0.7", "S", "P1", "S"]


def test_sentence_gap_rejects_garbage_and_clamps(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_SENTENCE_GAP_MS", "abc")
    with pytest.raises(ValueError):
        prepare_segments(SENTENCES)
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_SENTENCE_GAP_MS", "999999")
    from core.elevenlabs_tts.preprocessor import sentence_gap_ms
    assert sentence_gap_ms() == 5000


def test_direction_mode_first(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION_MODE", "first")
    texts = _speech(prepare_segments(SENTENCES))
    assert texts[0].startswith(DEFAULT_DIRECTION + " ")
    assert all(not t.startswith("[") for t in texts[1:])


def test_direction_mode_after_long_pause(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION_MODE", "after_long_pause")
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION_PAUSE_SEC", "3")
    texts = _speech(prepare_segments(SENTENCES))
    # chunk 1 (first) + chunk after the 5 s pause prefixed; chunk after the 1 s pause not
    assert [t.startswith("[") for t in texts] == [True, True, False]


def test_direction_mode_after_long_pause_with_gaps_counts_only_real_pauses(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION_MODE", "after_long_pause")
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_SENTENCE_GAP_MS", "700")
    texts = _speech(prepare_segments(SENTENCES))
    assert [t.startswith("[") for t in texts] == [True, False, False, True, False, False]


def test_direction_mode_invalid_raises(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION_MODE", "sometimes")
    with pytest.raises(ValueError):
        prepare_segments(SENTENCES)


def test_direction_mode_cuts_billed_chars(monkeypatch):
    base = estimate_billed_chars(prepare_segments(SENTENCES))
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION_MODE", "first")
    assert estimate_billed_chars(prepare_segments(SENTENCES)) < base


def test_sanitizer_still_rejects_sounds_for_stronger_direction():
    from core.elevenlabs_tts.direction import sanitize_direction
    assert sanitize_direction("very slow, soft, unhurried voice")
    for bad in ("slow, with a sigh", "[soft breathy voice]", "long exhale"):
        with pytest.raises(ValueError):
            sanitize_direction(bad)
