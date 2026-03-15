import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from core.f5_tts.engine import F5Engine
from core.f5_tts.preprocessor import prepare_segments
from core.f5_tts import voice_registry

def test_phases():
    print("Testing F5-TTS Multi-Phase Voice Support...")
    
    script = """
    This is guided text.
    [voice:closing]
    This is closing text.
    """
    
    segments = prepare_segments(script)
    
    # Verify preprocessor tags
    # print("Segments:", segments)
    speech_segs = [s for s in segments if s["type"] == "speech"]
    # In my preprocessor, the first segment has voice=None (default)
    assert speech_segs[0]["voice"] is None
    assert speech_segs[1]["voice"] == "closing"
    print("Preprocessor tags verified.")
    
    # Mock engine — use first available registered voice
    registry = voice_registry.scan()
    if not registry:
        print("No voices registered — skipping phases test.")
        return
    voice_slug = sorted(registry.keys())[0]
    engine = F5Engine(voice_slug=voice_slug)
    
    class MockModel:
        def __init__(self):
            self.history = []
        def infer(self, **kwargs):
            self.history.append(kwargs["ref_text"])
            sr = 24000
            return np.zeros(sr, dtype=np.float32), sr, None
            
    mock_model = MockModel()
    engine._model = mock_model
    
    # Mock pre-processed assets
    engine._phase_assets = {
        "default": {"audio": "def.wav", "text": "def text"},
        "closing": {"audio": "close.wav", "text": "close text"}
    }
    
    print("Running synthesis...")
    engine.synthesize(segments)
    
    # Verify switching
    print("Inference ref_text history:", mock_model.history)
    assert mock_model.history[0] == "def text"
    assert mock_model.history[1] == "close text"
    print("Reference switching verified.")
    
    print("Test passed!")

if __name__ == "__main__":
    try:
        test_phases()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
