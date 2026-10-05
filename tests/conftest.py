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
before Task 9 introduced checking.

CONTRACT -- read this before writing a test that checks corpus behaviour:
this fixture gives EVERY default-resolution call (corpus_dir=None) its OWN
fresh, empty directory, per call, not per test. A test that means to
exercise corpus accumulation, persistence, or cross-call matching MUST pass
an explicit `corpus_dir` (to AutoConfig, or directly to
add_to_corpus/load_corpus/assess). This applies under tests/unit AND
tests/integration -- this file sits at the tests/ root specifically so
integration tests get the same protection from writing into the developer's
real var/originality/. Without an explicit corpus_dir, corpus state never
survives from one call to the next, so a test asserting accumulation across
several run() calls would silently pass against an always-empty corpus.
OriginalityIntegrationTest (tests/unit/test_auto_generate.py) and all of
tests/unit/test_originality.py are the examples to copy: every call in
those passes corpus_dir explicitly, which bypasses this fixture entirely --
_resolve() only falls back to CORPUS_DIR when corpus_dir is None.

Per-call (not per-test) isolation is required, not incidental: TestRun and
TestRecentBackgrounds render the exact same canned script across several
run() calls that share one AutoConfig, specifically to test unrelated
behaviour (background rotation). A per-TEST fixture that reused one
directory across those calls would make the second call collide with the
first's now-recorded script and trip PASSAGE_LIFTED -- re-breaking the very
tests this fixture exists to keep passing.

Every directory this fixture hands out is created with tempfile.mkdtemp and
is real disk state that nothing else cleans up -- monkeypatch only reverts
the attribute patch, not the filesystem. Track and remove them in teardown
so a test run doesn't leak an unbounded number of throwaway directories
into the OS temp dir.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

import core.originality as originality


@pytest.fixture(autouse=True)
def _isolated_originality_corpus(monkeypatch):
    real_resolve = originality._resolve
    created: list[str] = []

    def _fresh_resolve(corpus_dir):
        if corpus_dir is not None:
            return real_resolve(corpus_dir)
        path = tempfile.mkdtemp(prefix="moodscape-test-corpus-")
        created.append(path)
        return Path(path)

    monkeypatch.setattr(originality, "_resolve", _fresh_resolve)
    yield
    for path in created:
        shutil.rmtree(path, ignore_errors=True)
