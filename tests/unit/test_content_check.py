"""Tests for the LLM content check (core/script_gen/content_check.py).

Fake engines only — no model, no network.
"""

import json
import unittest

from core.script_gen.content_check import (
    CONTENT_CODE,
    CONTENT_SYSTEM_PROMPT,
    content_findings,
    findings_to_violations,
)
from core.script_gen.engine import FakeScriptEngine, ScriptEngine
from core.script_gen.linter import ADVISORY, FATAL

SCRIPT = (
    "Settle in and let your shoulders drop.\n\n"
    "[pause:5s]\n\n"
    "Picture your inner   child\nsitting beside you.\n\n"
    "[breath]\n\n"
    "Notice the weight of your hands."
)


def _reply(*findings) -> str:
    return json.dumps([{"quote": q, "reason": r} for q, r in findings])


class _RaisingEngine(ScriptEngine):
    def __init__(self):
        self.calls = 0

    def complete(self, system, user, *, max_tokens=4096, temperature=1.0, sampling=None):
        self.calls += 1
        raise RuntimeError("backend down")

    @property
    def name(self) -> str:
        return "fake:raising"


class TestContentFindings(unittest.TestCase):
    def test_clean_reply_returns_empty_list(self):
        engine = FakeScriptEngine(["[]"])
        self.assertEqual(content_findings(engine, SCRIPT, genre="body_scan"), [])
        self.assertEqual(len(engine.calls), 1)

    def test_finding_with_present_quote_is_kept(self):
        engine = FakeScriptEngine([_reply(("inner child", "younger-self imagery"))])
        found = content_findings(engine, SCRIPT, genre="body_scan")
        self.assertEqual(found, [{"quote": "inner child", "reason": "younger-self imagery"}])

    def test_quote_match_is_case_and_whitespace_insensitive(self):
        # Script has "inner   child\nsitting" across a line break.
        engine = FakeScriptEngine([_reply(("Inner Child Sitting", "younger self"))])
        found = content_findings(engine, SCRIPT, genre="body_scan")
        self.assertEqual(len(found), 1)

    def test_quote_spanning_a_stripped_tag_matches(self):
        engine = FakeScriptEngine([_reply(("beside you. notice the weight", "x"))])
        self.assertEqual(len(content_findings(engine, SCRIPT, genre="g")), 1)

    def test_hallucinated_quote_is_dropped(self):
        engine = FakeScriptEngine(
            [_reply(("your illness fades", "medical"), ("inner child", "younger self"))]
        )
        found = content_findings(engine, SCRIPT, genre="body_scan")
        self.assertEqual([f["quote"] for f in found], ["inner child"])

    def test_empty_or_missing_quote_is_dropped(self):
        engine = FakeScriptEngine(['[{"quote": "", "reason": "x"}, {"reason": "y"}, "junk"]'])
        self.assertEqual(content_findings(engine, SCRIPT, genre="g"), [])

    def test_json_inside_prose_and_fences_is_parsed(self):
        reply = (
            "<think>checking [pause] tags</think>Here you go:\n```json\n"
            + _reply(("inner child", "r"))
            + "\n```"
        )
        engine = FakeScriptEngine([reply])
        self.assertEqual(len(content_findings(engine, SCRIPT, genre="g")), 1)

    def test_malformed_once_then_valid_retries(self):
        engine = FakeScriptEngine(["not json at all", _reply(("inner child", "r"))])
        found = content_findings(engine, SCRIPT, genre="g")
        self.assertEqual(len(found), 1)
        self.assertEqual(len(engine.calls), 2)

    def test_malformed_twice_returns_none_and_logs(self):
        engine = FakeScriptEngine(["nope", '{"quote": "inner child"}'])
        with self.assertLogs("core.script_gen.content_check", level="WARNING") as logs:
            self.assertIsNone(content_findings(engine, SCRIPT, genre="g"))
        self.assertEqual(len(engine.calls), 2)
        self.assertTrue(any("skipped" in line for line in logs.output))

    def test_engine_error_returns_none(self):
        engine = _RaisingEngine()
        with self.assertLogs("core.script_gen.content_check", level="WARNING"):
            self.assertIsNone(content_findings(engine, SCRIPT, genre="g"))
        self.assertEqual(engine.calls, 1)

    def test_prompt_shape(self):
        engine = FakeScriptEngine(["[]"])
        content_findings(engine, SCRIPT, genre="body_scan", max_tokens=321)
        call = engine.calls[0]
        self.assertEqual(call["system"], CONTENT_SYSTEM_PROMPT)
        self.assertEqual(call["max_tokens"], 321)
        self.assertIn("genre: body_scan", call["user"])
        # Tags are stripped from the script the model sees.
        self.assertNotIn("[pause:5s]", call["user"])
        self.assertNotIn("[breath]", call["user"])
        self.assertIn("Notice the weight of your hands.", call["user"])
        for category in ("illness", "pregnancy", "younger self", "self-harm", "weight"):
            self.assertIn(category, CONTENT_SYSTEM_PROMPT)

    def test_grief_pack_may_mention_loss(self):
        for genre in ("grief_support", "loss_and_letting_go"):
            engine = FakeScriptEngine(["[]"])
            content_findings(engine, SCRIPT, genre=genre)
            self.assertIn("may mention loss", engine.calls[0]["user"])
        engine = FakeScriptEngine(["[]"])
        content_findings(engine, SCRIPT, genre="body_scan")
        self.assertNotIn("may mention loss", engine.calls[0]["user"])

    def test_sampling_is_passed_through(self):
        sentinel = object()
        engine = FakeScriptEngine(["[]"])
        content_findings(engine, SCRIPT, genre="g", sampling=sentinel)
        self.assertIs(engine.calls[0]["sampling"], sentinel)


class TestFindingsToViolations(unittest.TestCase):
    def test_severity_code_and_message(self):
        findings = [{"quote": "inner child", "reason": "younger-self imagery"}]
        for severity in (FATAL, ADVISORY):
            (v,) = findings_to_violations(findings, severity=severity)
            self.assertEqual(v.code, CONTENT_CODE)
            self.assertEqual(v.code, "SENSITIVE_CONTENT")
            self.assertEqual(v.severity, severity)
            self.assertIn("inner child", v.message)
            self.assertIn("younger-self imagery", v.message)

    def test_empty(self):
        self.assertEqual(findings_to_violations([], severity=FATAL), [])


if __name__ == "__main__":
    unittest.main()
