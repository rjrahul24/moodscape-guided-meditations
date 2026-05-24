"""Tests for F5-TTS text normalization and preprocessing."""

import os
import sys

sys.path.append(os.getcwd())

from core.f5_tts.preprocessor import normalize_for_f5, prepare_segments


def test_digit_expansion():
    """Digits should be expanded to words."""
    assert "ten" in normalize_for_f5("Take 10 deep breaths.")
    assert "four" in normalize_for_f5("Hold for 4 seconds.")
    assert "one hundred" in normalize_for_f5("Count to 100.")


def test_hyphenated_numbers():
    """Breathing ratios like 4-7-8 should become 'four, seven, eight'."""
    result = normalize_for_f5("Try the 4-7-8 breathing technique.")
    assert "four, seven, eight" in result


def test_abbreviation_expansion():
    """Common abbreviations should be expanded."""
    assert "seconds" in normalize_for_f5("Hold for 5 sec.")
    assert "minutes" in normalize_for_f5("Rest for 2 min.")


def test_colon_replacement():
    """Colons should become commas (F5 ignores colons)."""
    result = normalize_for_f5("Notice: your breath is slowing.")
    assert ":" not in result
    assert "," in result


def test_ellipsis_replacement():
    """Ellipses should become single periods."""
    result = normalize_for_f5("Let go... and relax.")
    assert "..." not in result
    assert ". " in result or ".and" in result.replace(" ", "").lower()


def test_em_dash_replacement():
    """Em-dashes should become commas."""
    result = normalize_for_f5("Feel the warmth — spreading through your body.")
    assert "—" not in result
    assert "," in result


def test_compound_hyphen_removal():
    """Hyphens in compound words should be removed."""
    result = normalize_for_f5("Feel your well-being improve.")
    assert "wellbeing" in result
    assert "well-being" not in result


def test_compound_hyphen_preserves_numbers():
    """Hyphens between digits should be handled by number expansion, not removed."""
    result = normalize_for_f5("Use the 4-7 pattern.")
    # Should be expanded to words, not "47"
    assert "four" in result
    assert "seven" in result


def test_prepare_segments_applies_normalization():
    """prepare_segments should apply normalization to speech segments."""
    segments = prepare_segments("Take 10 deep breaths... and relax.")
    speech_segments = [s for s in segments if s["type"] == "speech"]
    assert len(speech_segments) > 0
    for seg in speech_segments:
        assert "10" not in seg["text"], "Digits should be expanded"
        assert "..." not in seg["text"], "Ellipses should be replaced"


def test_prepare_segments_preserves_pause_markers():
    """Normalization should not affect pause markers."""
    segments = prepare_segments("Relax now.\n\n[pause:5s]\n\nBreathe deeply.")
    pause_segments = [s for s in segments if s["type"] == "pause"]
    assert any(s["duration_sec"] == 5.0 for s in pause_segments)


if __name__ == "__main__":
    tests = [
        test_digit_expansion,
        test_hyphenated_numbers,
        test_abbreviation_expansion,
        test_colon_replacement,
        test_ellipsis_replacement,
        test_em_dash_replacement,
        test_compound_hyphen_removal,
        test_compound_hyphen_preserves_numbers,
        test_prepare_segments_applies_normalization,
        test_prepare_segments_preserves_pause_markers,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            sys.exit(1)
    print(f"\nAll {len(tests)} tests passed!")
