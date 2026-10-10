import csv
import json

import pytest

from core.eval import listen
from core.eval.ab import score


def _make_ab(root, name="abA", pairs=("00", "01")):
    ab = root / name
    (ab / "pairs").mkdir(parents=True)
    for p in pairs:
        (ab / "pairs" / f"{p}_A.wav").write_bytes(b"")
        (ab / "pairs" / f"{p}_B.wav").write_bytes(b"")
    key = {"seed": 1, "pairs": {
        "00": {"A": "baseline", "B": "challenger", "case": "c", "position": "start", "truncated": False},
        "01": {"A": "challenger", "B": "baseline", "case": "c", "position": "end", "truncated": False},
    }}
    (ab / ".key.json").write_text(json.dumps(key))
    with open(ab / "sheet.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "choice", "reason"])
        for p in pairs:
            w.writerow([p, "", ""])
    return ab


def test_build_page_data_relative_paths(tmp_path):
    ab = _make_ab(tmp_path)
    page = tmp_path / "pages" / "listen.html"
    page.parent.mkdir()
    data = listen.build_page_data([("Round A", "how", ab)], page)
    assert data[0]["round"] == "Round A" and data[0]["how"] == "how"
    assert data[0]["pairs"][0] == {"id": "00", "a": "../abA/pairs/00_A.wav", "b": "../abA/pairs/00_B.wav"}
    assert [p["id"] for p in data[0]["pairs"]] == ["00", "01"]


def test_render_page_has_lean_choices_and_storage_key():
    html = listen.render_page("R2 eval", [{"round": 'Ro"und', "how": "<b>x", "pairs": []}])
    assert 'moodscape-listen-r2-eval' in html
    assert "leanB" in html and "leanA" in html and "music balance / clarity" in html
    assert "Qwen" not in html and "qwen" not in html


def test_import_answers_writes_lean_and_tags(tmp_path):
    ab = _make_ab(tmp_path)
    counts = listen.import_answers({
        "Round A/00": {"choice": "leanA", "tags": ["better wording", "fewer glitches / gaps"], "reason": "nice"},
        "Round A/01": {"choice": "B", "tags": [], "reason": ""},
        "other/00": {"choice": "A"},
    }, {"Round A": ab})
    assert counts == {"Round A": 2}
    rows = list(csv.DictReader(open(ab / "sheet.csv", newline="")))
    assert rows[0]["choice"] == "leanA"
    assert rows[0]["reason"] == "nice [tags: better wording; fewer glitches / gaps]"
    assert rows[1]["choice"] == "B" and rows[1]["reason"] == ""


def test_import_unknown_pair_raises(tmp_path):
    ab = _make_ab(tmp_path)
    with pytest.raises(ValueError):
        listen.import_answers({"Round A/99": {"choice": "A"}}, {"Round A": ab})


def test_round_trip_score(tmp_path):
    ab = _make_ab(tmp_path)
    listen.import_answers({
        "R/00": {"choice": "leanB", "tags": ["x"], "reason": ""},  # B = challenger
        "R/01": {"choice": "A", "tags": [], "reason": "r"},        # A = challenger
    }, {"R": ab})
    res = score(ab)
    assert res["challenger_wins"] == 1 and res["challenger_leans"] == 1
    assert res["unanswered"] == 0


def test_tags_include_wording_problem_sensitive():
    assert "wording problem / sensitive" in listen.TAGS
