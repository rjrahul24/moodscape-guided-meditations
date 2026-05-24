"""IndexTTS-2 — zero-shot voice cloning with emotion control for meditation narration.

Public API:
    IndexTTSEngine          — TTS engine (load_model / synthesize / unload_model)
    prepare_segments        — Script parsing + IndexTTS-2-specific text preprocessing
    build_index_voice_chain — IndexTTS-2 voice FX chain (convolution reverb + limiter)
    IndexTTSMasteringEngine — EQ / dynamics / limiting for BigVGANv2 vocoder output
    SAMPLE_RATE             — Native output sample rate (24000 Hz)
"""

from core.speech_engine import SAMPLE_RATE
