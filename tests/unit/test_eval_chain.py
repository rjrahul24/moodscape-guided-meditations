"""Eval chain generation, spec validation and review pages from a TOML spec."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from core.eval.chain import EvalSpecError, load_eval_spec, parse_eval_spec, render_chain
from core.eval.review import build_review_lines, inject_into_page, write_review

SPEC = """
title = "Test eval (Oct 2026)"
testset = "var/eval/rt/testset.toml"
out = "{out}"
configs = "eval=ollama:qwen3.8:27b|ollama:gemma4:31b"
positions = "start,middle"
[[arms]]
name = "base"
desc = "today's defaults"
[[arms]]
name = "spectral"
desc = "+ spectral duck"
env = {{ MOODSCAPE_SPECTRAL_DUCK = "1", MOODSCAPE_BED_RISE_LIMIT = "1" }}
cached = true
[[arms]]
name = "chunked"
desc = "+ chunking"
env = {{ MOODSCAPE_F5_CHUNKING = "1" }}
from = "base"
[[rounds]]
id = "abA"
name = "Round A: duck"
a = "base"
b = "spectral"
how = "Same voice. Listen to the music."
listen_for = "does the music sit better?"
[[rounds]]
id = "abB"
name = "Round B: voice"
a = "spectral"
b = "chunked"
how = "Different voice rendering."
"""


def _spec(tmp_path):
    p = tmp_path / "spec.toml"
    p.write_text(SPEC.format(out=str(tmp_path / "out")))
    return load_eval_spec(p)


def _data(**over):
    d = {"title": "t", "testset": "t.toml", "out": "o", "configs": "eval=ollama:m",
         "arms": [{"name": "base"}, {"name": "x", "env": {"A": "1"}}]}
    d.update(over)
    return d


class TestSpecValidation:
    def test_loads(self, tmp_path):
        s = _spec(tmp_path)
        assert [a.name for a in s.arms] == ["base", "spectral", "chunked"]
        assert s.arm("spectral").cached and s.scripts_from(s.arm("spectral")) == "base"
        assert s.ollama_models == ["qwen3.8:27b", "gemma4:31b"]
        assert s.rounds[1].listen_for == "Different voice rendering."

    @pytest.mark.parametrize("arms,msg", [
        ([{"name": "a"}, {"name": "a"}], "duplicate arm"),
        ([{"name": "a", "from": "z"}], "must not have 'from'"),
        ([{"name": "a"}, {"name": "b", "from": "nope"}], "not an earlier arm"),
        ([{"name": "a", "cached": True}], "cannot be 'cached'"),
        ([{"name": "bad name"}], "name must match"),
        ([], "at least one"),
    ])
    def test_arm_errors(self, arms, msg):
        with pytest.raises(EvalSpecError, match=msg):
            parse_eval_spec(_data(arms=arms))

    def test_round_errors(self):
        with pytest.raises(EvalSpecError, match="not a known arm"):
            parse_eval_spec(_data(rounds=[{"id": "r", "name": "R", "a": "base", "b": "ghost"}]))
        with pytest.raises(EvalSpecError, match="duplicate round"):
            parse_eval_spec(_data(rounds=[{"id": "r", "name": "R", "a": "base", "b": "x"}] * 2))

    def test_positions_and_missing_fields(self):
        with pytest.raises(EvalSpecError, match="positions"):
            parse_eval_spec(_data(positions="start,late"))
        d = _data()
        del d["out"]
        with pytest.raises(EvalSpecError, match="'out'"):
            parse_eval_spec(d)

    def test_bad_toml(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text("title = ")
        with pytest.raises(EvalSpecError, match="not valid TOML"):
            load_eval_spec(p)


class TestRenderChain:
    def test_commands_per_arm(self, tmp_path):
        text = render_chain(_spec(tmp_path))
        assert text.startswith("#!/bin/zsh")
        assert "until mkdir var/heavy-model.lock" in text and "trap " in text
        assert "ollama stop qwen3.8:27b" in text and "ollama stop gemma4:31b" in text
        assert "--testset var/eval/rt/testset.toml" in text
        assert "--configs 'eval=ollama:qwen3.8:27b|ollama:gemma4:31b'" in text
        assert "arm spectral base MOODSCAPE_SPECTRAL_DUCK=1 MOODSCAPE_BED_RISE_LIMIT=1" in text
        assert "arm chunked base MOODSCAPE_F5_CHUNKING=1" in text
        assert text.index("arm spectral") < text.index("arm chunked")
        assert "--scripts-from $OUT/$src/eval --name eval" in text
        assert "for a in spectral; do" in text  # only the cached arm is cache-checked
        assert "--positions start,middle --allow-missing" in text
        assert "ab_export.py --a $OUT/base/eval --b $OUT/spectral/eval --out $OUT/abA" in text
        assert "[[ -f $OUT/abB/sheet.csv ]]" in text
        assert "ab_listen_page.py" in text and "ab_review.py" in text

    def test_export_exit_captured_right_after_command(self, tmp_path):
        lines = render_chain(_spec(tmp_path)).splitlines()
        for i, l in enumerate(lines):
            if "ab_export.py" in l:
                assert lines[i + 1].startswith("ec=$?; log \"export ")

    def test_from_other_arm(self, tmp_path):
        d = _data(arms=[{"name": "base"}, {"name": "m"}, {"name": "n", "from": "m"}])
        assert "arm n m" in render_chain(parse_eval_spec(d))

    def test_hostile_values_are_quoted(self):
        d = _data(title="it's $(rm -rf /)", arms=[{"name": "base"}, {"name": "x", "env": {"A": "a b;c"}}],
                  rounds=[{"id": "r", "name": "R's", "a": "base", "b": "x", "how": "say \"hi\" $HOME"}])
        text = render_chain(parse_eval_spec(d))
        assert "A='a b;c'" in text
        assert "'owner: it'\"'\"'s $(rm -rf /) chain '" in text  # single-quoted, never expanded
        if shutil.which("zsh"):
            r = subprocess.run(["zsh", "-n"], input=text, text=True, capture_output=True)
            assert r.returncode == 0, r.stderr

    @pytest.mark.skipif(not shutil.which("zsh"), reason="zsh missing")
    def test_zsh_syntax_ok(self, tmp_path):
        script = tmp_path / "chain.sh"
        script.write_text(render_chain(_spec(tmp_path)))
        r = subprocess.run(["zsh", "-n", str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    def test_write_cli(self, tmp_path):
        from scripts import ab_chain
        p = tmp_path / "spec.toml"
        p.write_text(SPEC.format(out=str(tmp_path / "out")))
        with mock.patch.dict(os.environ, {}, clear=False):
            assert ab_chain.main([str(p), "--write"]) == 0
        chain = tmp_path / "out" / "chain.sh"
        assert chain.is_file() and os.access(chain, os.X_OK)


def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _row(case, **kw):
    r = {"case": case, "ok": True, "tts_cache_hit": False,
         "metrics": {"mix": {"actual_sec": 600, "integrated_lufs_dual_mono": -16.0}, "speech": {"internal_gaps": 3, "cer": 0.02}}}
    r.update(kw)
    return r


@pytest.fixture
def evaldir(tmp_path):
    spec = _spec(tmp_path)
    out = Path(spec.out)
    _write(out / "base" / "results.json", [_row("c1"), _row("c2")])
    _write(out / "spectral" / "results.json", [_row("c1", tts_cache_hit=True), _row("c2", ok=False, error="boom")])
    # chunked arm intentionally missing
    _write(out / "base" / "eval" / "c1" / "c1.meta.json",
           {"duration_control": {"length_repair": True, "pause_scale": 1.1},
            "content_check": {"ran": True, "findings_first": 2, "findings_final": 1}})
    _write(out / "spectral" / "eval" / "c1" / "c1.meta.json", {"speech_qa": {"chunks": 10, "failed_first_pass": 2, "rerendered": 2, "still_failing": 0}})
    _write(out / "abA" / ".key.json", {"pairs": {"00": {}, "01": {}}, "skipped_cases": ["c2"]})
    (out / "chain.log").write_text("[chain] base exit=0 x\n[chain] WARNING: spectral c2 re-rendered\n[chain] export abB exit=1 y\n")
    (out / "listen.html").write_text('<html><body><h1>t</h1><div id="rounds"></div></body></html>')
    return spec, out


class TestReview:
    def test_tables_and_warnings(self, evaldir):
        spec, out = evaldir
        text = "\n".join(build_review_lines(spec, out))
        assert "# Test eval (Oct 2026): start here" in text
        assert "| Round A: duck | 2 (skipped: c2) |" in text and "does the music sit better?" in text
        assert "| Round B: voice | **not built**" in text
        assert "| `base` | today's defaults | 2/2 | 10.00 | -16.00 | 3.00 | 0.02 | 0/2 |" in text
        assert "| `spectral` | + spectral duck | 1/2 | " in text and "| 1/1 |" in text
        assert "| `spectral` | 1 | 10 | 2 | 2 | 0 | 0 |" in text  # voice QA
        assert "| `base` | 1/1 | 1/1 | 1.10-1.10 | 1/1 | 2 | 1 |" in text  # script controls
        assert "all-cached for `spectral`" in text
        assert "chunked: produced no results" in text
        assert "spectral / c2: boom" in text
        assert "WARNING: spectral c2 re-rendered" in text and "export abB exit=1" in text
        assert "base exit=0" not in text.split("## Warnings")[1]

    def test_no_script_controls_section_without_meta(self, tmp_path):
        spec = _spec(tmp_path)
        Path(spec.out).mkdir()
        text = "\n".join(build_review_lines(spec, Path(spec.out)))
        assert "Script controls" not in text and "No arm recorded per-chunk voice QA" in text

    def test_write_and_idempotent_injection(self, evaldir):
        spec, out = evaldir
        write_review(spec)
        assert (out / "REVIEW.md").read_text().startswith("# Test eval")
        once = (out / "listen.html").read_text()
        assert once.count("machine-summary:start") == 1 and "<details class=\"card\">" in once
        assert once.index("machine-summary:end") < once.index('<div id="rounds">')
        assert "What the machine measured" in once and "After listening" not in once
        write_review(spec)
        assert (out / "listen.html").read_text() == once

    def test_inject_without_page_or_anchor(self, tmp_path):
        assert inject_into_page(tmp_path / "none.html", ["x"]) is False
        p = tmp_path / "p.html"
        p.write_text("<html></html>")
        assert inject_into_page(p, ["x"]) is False and p.read_text() == "<html></html>"

    def test_cli(self, evaldir, tmp_path):
        from scripts import ab_review
        spec, out = evaldir
        with mock.patch.dict(os.environ, {}, clear=False):
            assert ab_review.main([spec.path]) == 0
        assert (out / "REVIEW.md").is_file()
