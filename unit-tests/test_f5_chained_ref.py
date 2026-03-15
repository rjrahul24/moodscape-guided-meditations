"""Tests for F5Engine quasi-autoregressive chained reference audio.

Verifies that:
  1. First chunk always uses the static reference.
  2. Second consecutive speech chunk uses a chained temp WAV (not the static path).
  3. A long pause (>= 3.0s) resets the chain (next speech uses static again).
  4. A short pause (< 3.0s) does NOT reset the chain.
  5. A breath segment does NOT reset the chain.
  6. A voice phase change resets the chain.
  7. After _CHAIN_RESET_EVERY (6) consecutive speech chunks, reset to static.
  8. All temp files are deleted after synthesize() returns.
"""

import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from core.f5_tts.engine import F5Engine
from core.f5_tts import voice_registry


def _make_engine_with_mock():
    """Build an F5Engine with a mocked model that records infer() arguments."""
    registry = voice_registry.scan()
    if not registry:
        print("No voices registered — skipping chained-ref tests.")
        return None, None

    voice_slug = sorted(registry.keys())[0]
    engine = F5Engine(voice_slug=voice_slug)

    infer_calls = []  # records kwargs passed to each infer() call

    class MockModel:
        ema_model = type("EMA", (), {"to": lambda s, d: s})()

        def infer(self, **kwargs):
            infer_calls.append(dict(kwargs))
            # Return 1 second of audio so chain-ref length checks pass
            sr = 24000
            return np.ones(sr, dtype=np.float32) * 0.1, sr, None

    engine._model = MockModel()
    engine._phase_assets = {
        "default": {"audio": "static_ref.wav", "text": "static reference text"},
        "closing": {"audio": "closing_ref.wav", "text": "closing reference text"},
    }
    return engine, infer_calls


def test_first_chunk_uses_static_ref():
    engine, calls = _make_engine_with_mock()
    if engine is None:
        return

    segments = [{"type": "speech", "text": "Hello world.", "voice": None}]
    engine.synthesize(segments)

    assert len(calls) == 1
    assert calls[0]["ref_file"] == "static_ref.wav", (
        f"First chunk should use static ref, got: {calls[0]['ref_file']}"
    )
    print("PASS: first chunk uses static ref")


def test_second_chunk_uses_chained_ref():
    engine, calls = _make_engine_with_mock()
    if engine is None:
        return

    segments = [
        {"type": "speech", "text": "First sentence.", "voice": None},
        {"type": "speech", "text": "Second sentence.", "voice": None},
    ]
    engine.synthesize(segments)

    assert len(calls) == 2
    assert calls[0]["ref_file"] == "static_ref.wav", "First chunk should use static"
    assert calls[1]["ref_file"] != "static_ref.wav", "Second chunk should use chained ref"
    assert calls[1]["ref_file"].endswith(".wav"), "Chained ref should be a WAV file"
    print(f"PASS: second chunk uses chained ref ({calls[1]['ref_file']})")


def test_long_pause_resets_chain():
    engine, calls = _make_engine_with_mock()
    if engine is None:
        return

    segments = [
        {"type": "speech", "text": "Before pause.", "voice": None},
        {"type": "pause", "duration_sec": 4.0},  # >= 3.0s threshold → resets chain
        {"type": "speech", "text": "After pause.", "voice": None},
    ]
    engine.synthesize(segments)

    assert len(calls) == 2
    assert calls[0]["ref_file"] == "static_ref.wav", "First chunk should use static"
    assert calls[1]["ref_file"] == "static_ref.wav", (
        "Chunk after long pause (>= 3.0s) should reset to static ref"
    )
    print("PASS: long pause (4.0s) resets chain to static ref")


def test_short_pause_maintains_chain():
    engine, calls = _make_engine_with_mock()
    if engine is None:
        return

    segments = [
        {"type": "speech", "text": "Before short pause.", "voice": None},
        {"type": "pause", "duration_sec": 1.5},  # < 3.0s threshold → chain maintained
        {"type": "speech", "text": "After short pause.", "voice": None},
    ]
    engine.synthesize(segments)

    assert len(calls) == 2
    assert calls[0]["ref_file"] == "static_ref.wav", "First chunk should use static"
    assert calls[1]["ref_file"] != "static_ref.wav", (
        "Chunk after short pause (< 3.0s) should maintain chain (not reset)"
    )
    print("PASS: short pause (1.5s) maintains chain")


def test_breath_maintains_chain():
    engine, calls = _make_engine_with_mock()
    if engine is None:
        return

    segments = [
        {"type": "speech", "text": "Before breath.", "voice": None},
        {"type": "breath", "subtype": "breath"},
        {"type": "speech", "text": "After breath.", "voice": None},
    ]
    engine.synthesize(segments)

    assert len(calls) == 2
    assert calls[0]["ref_file"] == "static_ref.wav", "First chunk should use static"
    assert calls[1]["ref_file"] != "static_ref.wav", (
        "Chunk after breath should maintain chain (breath < 3.0s threshold)"
    )
    print("PASS: breath segment maintains chain")


def test_phase_change_resets_chain():
    engine, calls = _make_engine_with_mock()
    if engine is None:
        return

    segments = [
        {"type": "speech", "text": "Default voice chunk.", "voice": None},
        {"type": "speech", "text": "Still default.", "voice": None},
        {"type": "speech", "text": "Now closing phase.", "voice": "closing"},
    ]
    engine.synthesize(segments)

    assert len(calls) == 3
    assert calls[0]["ref_file"] == "static_ref.wav"
    assert calls[1]["ref_file"] != "static_ref.wav", "Second default chunk should chain"
    assert calls[2]["ref_file"] == "closing_ref.wav", (
        "Phase change should reset to static ref for the new phase"
    )
    print("PASS: phase change resets chain to phase static ref")


def test_chain_resets_every_n_chunks():
    engine, calls = _make_engine_with_mock()
    if engine is None:
        return

    # 7 consecutive speech chunks — chunk 7 (index 6) should reset (6 ≥ _CHAIN_RESET_EVERY)
    segments = [
        {"type": "speech", "text": f"Chunk {i}.", "voice": None}
        for i in range(7)
    ]
    engine.synthesize(segments)

    assert len(calls) == 7
    # Chunk 0: static
    assert calls[0]["ref_file"] == "static_ref.wav", "Chunk 0 should use static ref"
    # Chunks 1-5: chained (not static)
    for i in range(1, 6):
        assert calls[i]["ref_file"] != "static_ref.wav", (
            f"Chunk {i} should use chained ref"
        )
    # Chunk 6: reset after 6 chained chunks
    assert calls[6]["ref_file"] == "static_ref.wav", (
        "Chunk 6 should reset to static ref after _CHAIN_RESET_EVERY"
    )
    print("PASS: chain resets to static after _CHAIN_RESET_EVERY consecutive speech chunks")


def test_temp_files_deleted_after_synthesize():
    engine, calls = _make_engine_with_mock()
    if engine is None:
        return

    segments = [
        {"type": "speech", "text": "Chunk one.", "voice": None},
        {"type": "speech", "text": "Chunk two.", "voice": None},
        {"type": "speech", "text": "Chunk three.", "voice": None},
    ]
    engine.synthesize(segments)

    # Collect paths that were used as chained refs (not the static path)
    chained_paths = [
        c["ref_file"] for c in calls if c["ref_file"] != "static_ref.wav"
    ]
    assert len(chained_paths) > 0, "Expected at least one chained ref path"

    for path in chained_paths:
        assert not os.path.exists(path), (
            f"Temp file was not deleted after synthesize(): {path}"
        )

    # Engine's internal list should also be empty
    assert len(engine._chain_tmp_paths) == 0, (
        "engine._chain_tmp_paths should be empty after synthesize()"
    )
    print(f"PASS: all {len(chained_paths)} temp file(s) deleted after synthesize()")


if __name__ == "__main__":
    tests = [
        test_first_chunk_uses_static_ref,
        test_second_chunk_uses_chained_ref,
        test_long_pause_resets_chain,
        test_short_pause_maintains_chain,
        test_breath_maintains_chain,
        test_phase_change_resets_chain,
        test_chain_resets_every_n_chunks,
        test_temp_files_deleted_after_synthesize,
    ]

    failed = 0
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            import traceback
            print(f"FAIL: {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    if failed:
        print(f"\n{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests PASSED")
