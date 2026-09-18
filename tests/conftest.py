"""Session-wide test fixtures.

Isolates core.originality's on-disk corpus from the real project directory
and from other tests.

Task 9 (core/auto_generate.py) turned AutoConfig.originality on by default.
Many pre-existing tests build an AutoConfig without setting corpus_dir,
which defaults to None -> core.originality.CORPUS_DIR, the project's real
var/originality/ directory. Two things follow from that, neither of which
those tests were written to expect:

  1. Cross-test pollution: every such test shares ONE persistent, real
     directory, so a script added by one test is visible to the next.
  2. Cross-call collisions within a single test: a few pre-existing tests
     (e.g. TestRecentBackgrounds, which shares one AutoConfig across several
     run() calls on purpose, to test background rotation) deliberately
     render the exact same canned script more than once. That is now
     exactly the pattern PASSAGE_LIFTED exists to catch -- just aimed at
     test fixture data instead of a real repeat generation.

Give every call that would otherwise fall through to the real CORPUS_DIR its
own empty, throw-away directory instead, so a test that never asked to
exercise the corpus always sees an empty one -- the same behaviour it had
before Task 9 introduced checking. A test that wants to exercise the corpus
(e.g. OriginalityIntegrationTest) passes its own corpus_dir explicitly, which
bypasses this fixture entirely -- _resolve() only falls back to CORPUS_DIR
when corpus_dir is None.
"""

import tempfile
from pathlib import Path

import pytest

import core.originality as originality


@pytest.fixture(autouse=True)
def _isolated_originality_corpus(monkeypatch):
    real_resolve = originality._resolve

    def _fresh_resolve(corpus_dir):
        if corpus_dir is not None:
            return real_resolve(corpus_dir)
        return Path(tempfile.mkdtemp(prefix="moodscape-test-corpus-"))

    monkeypatch.setattr(originality, "_resolve", _fresh_resolve)
    yield
