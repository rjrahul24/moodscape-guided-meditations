import pytest

from core.elevenlabs_tts.direction import DEFAULT_DIRECTION, default_direction, sanitize_direction


@pytest.mark.parametrize("raw,expected", [
    ("[soft, slow, calm voice]", "[soft, slow, calm voice]"),
    ("soft, slow, calm voice", "[soft, slow, calm voice]"),
    ("  [warm]  [unhurried] ", "[warm unhurried]"),
    ("", DEFAULT_DIRECTION),
    (None, DEFAULT_DIRECTION),
])
def test_normalizes(raw, expected):
    assert sanitize_direction(raw) == expected


@pytest.mark.parametrize("bad", ["[sighs]", "soft with a breath", "[exhales slowly]",
                                 "laughing softly", "[yawn]", "gentle humming"])
def test_rejects_vocal_sound_words(bad):
    with pytest.raises(ValueError, match="voice quality"):
        sanitize_direction(bad)


def test_allows_words_that_merely_contain_sound_substrings():
    assert sanitize_direction("[human, humble tone]") == "[human, humble tone]"


def test_too_long_rejected():
    with pytest.raises(ValueError, match="80"):
        sanitize_direction("calm " * 30)


def test_whitespace_only_uses_env_default(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION", "warm, low voice")
    assert sanitize_direction("   ") == "[warm, low voice]"


@pytest.mark.parametrize("env_val", ["   ", "[ ]", ""])
def test_blank_env_falls_back_to_default(monkeypatch, env_val):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION", env_val)
    assert default_direction() == DEFAULT_DIRECTION
    assert sanitize_direction("   ") == DEFAULT_DIRECTION


def test_env_default(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION", "warm, low voice")
    assert default_direction() == "[warm, low voice]"


@pytest.mark.parametrize("raw", [
    "[laughter]", "[soft laughter]", "[sighed]", "[inhaled]", "[yawned]",
    "[breathless]", "[clears throat]", "[groaning]", "[snoring]", "[sniffles]",
    "[sigh_soft]", "[mm-hmm]", "[ahh]", "[mm hmm]",
])
def test_sound_directions_rejected(raw):
    with pytest.raises(ValueError, match="voice quality"):
        sanitize_direction(raw)


@pytest.mark.parametrize("raw", [
    "[soft, slow, calm voice]", "[warm, low voice]", "[whispered, gentle]",
    "[human, humble tone]", "[sober, sighted narrator]",
])
def test_voice_quality_directions_pass(raw):
    assert sanitize_direction(raw) == raw
