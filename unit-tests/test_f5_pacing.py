import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from core.f5_tts.engine import F5Engine
from core.f5_tts.preprocessor import prepare_segments
import librosa

def test_pacing():
    print("Testing F5-TTS Pacing Improvement...")
    
    # We don't want to actually load the heavy model in a basic unit test if possible,
    # but F5Engine.synthesize requires it. 
    # Let's mock the model inference if we can to test just the stretching logic.
    
    engine = F5Engine(voice_slug="female_relaxed_warm")
    
    # Mock model
    class MockModel:
        def infer(self, **kwargs):
            # Return a short audio segment to speed up JIT compilation in tests
            sr = 24000
            # Use 0.1s of noise
            audio = np.random.uniform(-0.1, 0.1, int(0.1 * sr)).astype(np.float32)
            return audio, sr, None
            
    engine._model = MockModel()
    
    segments = [{"type": "speech", "text": "This is a test sentence."}]
    
    print("Running synthesis at speed=1.0...")
    audio_1, mask_1 = engine.synthesize(segments, speed=1.0)
    print(f"Speed 1.0 length: {len(audio_1)} samples")
    
    print("Running synthesis at speed=0.5...")
    audio_05, mask_05 = engine.synthesize(segments, speed=0.5)
    print(f"Speed 0.5 length: {len(audio_05)} samples")
    
    # Speed 0.5 should be twice as long as speed 1.0
    expected_len = len(audio_1) * 2
    # Pedalboard might have minor alignment differences, check with tolerance
    diff = abs(len(audio_05) - expected_len)
    
    print(f"Difference from expected: {diff} samples")
    
    assert diff < 100, f"Pacing failed: speed=0.5 should be ~2x length. Got {len(audio_05)}, expected ~{expected_len}"
    assert len(audio_05) == len(mask_05), "Audio and mask length mismatch"
    
    print("Test passed!")

if __name__ == "__main__":
    try:
        test_pacing()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
