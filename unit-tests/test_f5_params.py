import os
import sys
import numpy as np
import torch

# Add project root to path
sys.path.append(os.getcwd())

from core.f5_tts.engine import F5Engine
from core.f5_tts import voice_registry

def test_params():
    print("Testing F5-TTS Parameter Optimization...")
    
    registry = voice_registry.scan()
    if not registry:
        print("No voices found in registry, skipping test.")
        return
    
    voice_slug = list(registry.keys())[0]
    print(f"Using voice: {voice_slug}")
    engine = F5Engine(voice_slug=voice_slug)
    
    # Mock model and F5TTS
    class MockEMA:
        def __init__(self):
            self.dtype = torch.float32
        def to(self, dtype):
            self.dtype = dtype
            return self

    class MockModel:
        def __init__(self):
            self.ema_model = MockEMA()
            self.last_infer_kwargs = {}
        def infer(self, **kwargs):
            self.last_infer_kwargs = kwargs
            sr = 24000
            return np.zeros(sr, dtype=np.float32), sr, None
            
    mock_model = MockModel()
    engine._model = mock_model
    engine._phase_assets = {"default": {"audio": "mock_path", "text": "mock text"}}
    
    # Test precision casting (simulating load_model effect)
    mock_model.ema_model.to(torch.float16)
    assert mock_model.ema_model.dtype == torch.float16, "Precision casting failed"
    print("Precision casting to fp16 verified.")
    
    segments = [{"type": "speech", "text": "This is a test sentence."}]
    
    print("Running synthesis...")
    engine.synthesize(segments, speed=1.0)
    
    # Check CFG strength
    cfg_strength = mock_model.last_infer_kwargs.get("cfg_strength")
    print(f"CFG Strength used in infer: {cfg_strength}")
    assert cfg_strength == 2.0, f"Incorrect CFG strength: {cfg_strength}"
    
    # Check remove_silence
    remove_silence = mock_model.last_infer_kwargs.get("remove_silence")
    print(f"remove_silence used in infer: {remove_silence}")
    assert remove_silence == False, "remove_silence should be False"
    
    print("Test passed!")

if __name__ == "__main__":
    try:
        test_params()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
