"""Opt-in live test: hits a real configured model.

Excluded from the default run. Enable with:
    MOODSCAPE_LIVE_SCRIPT_TEST=1 .venv/bin/python -m pytest \
        tests/integration/test_script_gen_live.py -v

Requires MOODSCAPE_SCRIPT_GENERATOR and MOODSCAPE_SCRIPT_JUDGE to name
reachable models (e.g. a running `ollama serve`).
"""

import os
import unittest

from core.auto_generate import AutoConfig, generate_script
from core.script_gen.engine import build_engine

LIVE = os.environ.get("MOODSCAPE_LIVE_SCRIPT_TEST") == "1"


@unittest.skipUnless(LIVE, "set MOODSCAPE_LIVE_SCRIPT_TEST=1 to run")
class TestLiveScriptGeneration(unittest.TestCase):
    def test_configured_models_produce_a_passing_script(self):
        generator = build_engine(
            os.environ.get("MOODSCAPE_SCRIPT_GENERATOR", "ollama:llama3.2:3b")
        )
        judge = build_engine(
            os.environ.get("MOODSCAPE_SCRIPT_JUDGE", "ollama:llama3.2:3b")
        )
        outcome = generate_script(
            "I'm feeling anxious and need to unwind.",
            generator_engine=generator,
            judge_engine=judge,
            config=AutoConfig(),
        )
        self.assertTrue(outcome.script.strip())
        self.assertLessEqual(outcome.repairs_used, 2)


if __name__ == "__main__":
    unittest.main()
