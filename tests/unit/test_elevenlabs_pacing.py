import numpy as np
import pytest

from core.elevenlabs_tts.pacing import apply_pacing, pacing_enabled, pacing_params, pause_points

SR = 48000


def _align(text, char_sec=0.05, tag_zero=True):
    """Fake alignment: tag chars at ~0 duration, every other char char_sec long."""
    chars, starts, ends = [], [], []
    t, in_tag = 0.0, False
    for ch in text:
        if ch == "[":
            in_tag = True
        dur = 0.0 if (in_tag and tag_zero) else char_sec
        chars.append(ch)
        starts.append(t)
        ends.append(t + dur)
        t += dur
        if ch == "]":
            in_tag = False
    return {"characters": chars, "starts": starts, "ends": ends}


def _times(points):
    return [round(t, 3) for t, _ in points]


def test_skips_commas_inside_bracketed_tag():
    a = _align("[soft, slow, calm voice] Hello there.")
    assert pause_points(a, 0.9, 0.35) == []


def test_sentence_and_clause_points():
    text = "[soft] Breathe in, slowly. Let go; rest now."
    a = _align(text)
    pts = pause_points(a, 0.9, 0.35)
    assert [g for _, g in pts] == [0.35, 0.9, 0.35]
    comma = text.index(",")
    # point lies between the mark's end and the next word's start
    assert a["ends"][comma] <= pts[0][0] <= a["starts"][comma + 2]


def test_no_point_at_final_character():
    a = _align("Rest. Now.")
    pts = pause_points(a, 0.9, 0.35)
    assert len(pts) == 1
    assert pts[0][0] < a["starts"][-1]
    assert pause_points(_align("Rest now. "), 0.9, 0.35) == []


@pytest.mark.parametrize("text", ["Wait... then go.", "Wait… then go.",
                                  "Really? Yes.", "Now! Rest."])
def test_sentence_marks_including_ellipses(text):
    pts = pause_points(_align(text), 0.9, 0.35)
    assert [g for _, g in pts] == [0.9]


def test_em_dash_and_colon_are_clauses():
    pts = pause_points(_align("Notice — the air: cool now."), 0.9, 0.35)
    assert [g for _, g in pts] == [0.35, 0.35]


def test_no_mark_without_following_whitespace():
    assert pause_points(_align("3.5 and a,b now"), 0.9, 0.35) == []


def test_zero_gap_points_dropped():
    assert pause_points(_align("One, two. Three."), 0.0, 0.0) == []


def test_empty_alignment():
    assert pause_points(None, 0.9, 0.35) == []
    assert pause_points({"characters": [], "starts": [], "ends": []}, 0.9, 0.35) == []


def _tone(sec):
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_identity_when_no_stretch_and_no_alignment():
    y = _tone(0.5)
    out = apply_pacing(y, SR, None, stretch=1.0, sentence_sec=0.9, clause_sec=0.35)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, y)


def test_length_is_stretched_plus_gaps_and_gaps_silent():
    text = "Breathe in, slowly. Let go now."
    a = _align(text, char_sec=0.03)
    y = _tone(a["ends"][-1])
    out = apply_pacing(y, SR, a, stretch=0.9, sentence_sec=0.9, clause_sec=0.35)
    stretched = round(len(y) / 0.9)
    expected = stretched + int(round(0.9 * SR)) + int(round(0.35 * SR))
    assert abs(len(out) - expected) <= 8
    assert out.dtype == np.float32


def test_inserted_regions_silent_and_no_clicks():
    a = _align("Breathe in, slowly. Let go now.", char_sec=0.03)
    y = _tone(a["ends"][-1])
    out = apply_pacing(y, SR, a, stretch=1.0, sentence_sec=0.9, clause_sec=0.35)
    pts = pause_points(a, 0.9, 0.35)
    shift = 0
    for t, gap in pts:
        k = int(round(t * SR)) + shift
        n = int(round(gap * SR))
        assert np.all(out[k:k + n] == 0.0)
        shift += n
    # 220 Hz at 0.3 amplitude moves at most ~0.0087/sample; a hard cut would jump ~0.3
    assert np.max(np.abs(np.diff(out))) < 0.02


def test_stretch_only_without_alignment():
    y = _tone(0.5)
    out = apply_pacing(y, SR, None, stretch=0.9, sentence_sec=0.9, clause_sec=0.35)
    assert abs(len(out) - round(len(y) / 0.9)) <= 8


def test_config_defaults(monkeypatch):
    for k in ("PACING", "STRETCH", "SENTENCE_PAUSE_SEC", "CLAUSE_PAUSE_SEC"):
        monkeypatch.delenv(f"MOODSCAPE_ELEVENLABS_{k}", raising=False)
    assert pacing_enabled() is False          # opt-in: native v4 delivery is the default
    assert pacing_params() == (0.9, 0.9, 0.35)
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_PACING", "1")
    assert pacing_enabled() is True


def test_config_clamps(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_STRETCH", "0.5")
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_SENTENCE_PAUSE_SEC", "9")
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CLAUSE_PAUSE_SEC", "-1")
    assert pacing_params() == (0.75, 3.0, 0.0)
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_STRETCH", "1.4")
    assert pacing_params()[0] == 1.0


@pytest.mark.parametrize("name", ["STRETCH", "SENTENCE_PAUSE_SEC", "CLAUSE_PAUSE_SEC"])
def test_config_malformed_names_variable(monkeypatch, name):
    monkeypatch.setenv(f"MOODSCAPE_ELEVENLABS_{name}", "slow")
    with pytest.raises(ValueError, match=f"MOODSCAPE_ELEVENLABS_{name}"):
        pacing_params()
