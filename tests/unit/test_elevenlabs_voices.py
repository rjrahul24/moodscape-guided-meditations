"""ElevenLabs voice registry (assets/elevenlabs/voices.toml)."""

import pytest

from core.elevenlabs_tts.voices import (
    default_voice_id, list_voices, load_registry, voice_for_family,
)


def _write(tmp_path, body):
    p = tmp_path / "voices.toml"
    p.write_text(body)
    return p


def test_shipped_registry_has_old_repo_voices_and_brittney_default():
    ids = {v.id for v in list_voices()}
    assert "pjcYQlDFKMbcOUp6F5GD" in ids and "pMsXgVXv3BLzUgSXRplE" in ids
    assert len(ids) == 9
    assert default_voice_id() == "pjcYQlDFKMbcOUp6F5GD"


def test_family_map_falls_back_to_default(tmp_path):
    p = _write(tmp_path, 'default_voice = "a"\n[[voice]]\nid = "a"\nname = "A"\n'
                         '[[voice]]\nid = "b"\nname = "B"\n[family_map]\nsleep = "b"\n')
    assert voice_for_family("sleep", path=p) == "b"
    assert voice_for_family("focus", path=p) == "a"


def test_unknown_family_voice_raises_with_known_list(tmp_path):
    p = _write(tmp_path, 'default_voice = "a"\n[[voice]]\nid = "a"\nname = "A"\n'
                         '[family_map]\nsleep = "zzz"\n')
    with pytest.raises(ValueError, match="zzz.*known"):
        load_registry(p)


def test_default_must_be_a_listed_voice(tmp_path):
    p = _write(tmp_path, 'default_voice = "nope"\n[[voice]]\nid = "a"\nname = "A"\n')
    with pytest.raises(ValueError, match="default_voice"):
        load_registry(p)


def test_label_includes_notes():
    v = next(v for v in list_voices() if v.name == "Serena")
    assert v.label.startswith("Serena — ")
