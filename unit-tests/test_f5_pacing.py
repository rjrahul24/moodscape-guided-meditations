import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from core.f5_tts.engine import F5Engine
from core.f5_tts.preprocessor import prepare_segments
from core.f5_tts import voice_registry
import librosa

def test_pacing():
    print("Testing F5-TTS Pacing Improvement...")

    registry = voice_registry.scan()
    if not registry:
        print("No voices registered — skipping pacing test.")
        return
    voice_slug = sorted(registry.keys())[0]
    print(f"Using voice: {voice_slug}")

    # We don't want to actually load the heavy model in a basic unit test if possible,
    # but F5Engine.synthesize requires it.
    # Let's mock the model inference if we can to test just the stretching logic.

    engine = F5Engine(voice_slug=voice_slug)
    
    # Mock model
    recorded_speeds = []

    class MockModel:
        def infer(self, **kwargs):
            recorded_speeds.append(kwargs.get("speed"))
            sr = 24000
            # Return 1s of audio (enough to satisfy _CHAIN_MIN_SAMPLES)
            audio = np.random.uniform(-0.1, 0.1, sr).astype(np.float32)
            return audio, sr, None

    engine._model = MockModel()
    engine._phase_assets = {"default": {"audio": "mock_path", "text": "mock text"}}

    segments = [{"type": "speech", "text": "This is a test sentence."}]

    print("Running synthesis at speed=1.0...")
    audio_1, mask_1 = engine.synthesize(segments, speed=1.0)
    print(f"Speed 1.0: infer received speed={recorded_speeds[-1]}")
    assert recorded_speeds[-1] == 1.0, f"speed=1.0 not passed to infer, got {recorded_speeds[-1]}"
    assert len(audio_1) == len(mask_1), "Audio and mask length mismatch at speed=1.0"

    print("Running synthesis at speed=0.5...")
    audio_05, mask_05 = engine.synthesize(segments, speed=0.5)
    print(f"Speed 0.5: infer received speed={recorded_speeds[-1]}")
    assert recorded_speeds[-1] == 0.5, f"speed=0.5 not passed to infer, got {recorded_speeds[-1]}"
    assert len(audio_05) == len(mask_05), "Audio and mask length mismatch at speed=0.5"

    # Verify that fix_duration is NOT set when target_wpm=None (default)
    print("Running synthesis with target_wpm=None (natural rhythm)...")
    recorded_kwargs = {}
    class MockModel2:
        def infer(self, **kwargs):
            recorded_kwargs.update(kwargs)
            sr = 24000
            audio = np.random.uniform(-0.1, 0.1, sr).astype(np.float32)
            return audio, sr, None
    engine._model = MockModel2()
    engine.synthesize(segments, speed=0.88, target_wpm=None)
    assert "fix_duration" not in recorded_kwargs, \
        f"fix_duration should not be set when target_wpm=None, got {recorded_kwargs.get('fix_duration')}"
    print("Verified: fix_duration not set when target_wpm=None (natural rhythm mode)")

    # Verify fix_duration IS set when target_wpm is provided
    print("Running synthesis with target_wpm=110...")
    recorded_kwargs.clear()
    engine.synthesize(segments, speed=0.88, target_wpm=110)
    assert "fix_duration" in recorded_kwargs, \
        "fix_duration should be set when target_wpm=110"
    print(f"Verified: fix_duration={recorded_kwargs['fix_duration']} when target_wpm=110")

    print("Test passed! (Speed passthrough + natural rhythm mode verified)")

if __name__ == "__main__":
    try:
        test_pacing()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
