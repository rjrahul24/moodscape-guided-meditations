"""Deterministic fakes for running MeditationPipeline without models or network."""

import numpy as np

from core.speech_engine import SpeechEngine


class MockTTS24(SpeechEngine):
    """Local-style engine: 3 s of constant 24 kHz audio."""

    def __init__(self):
        self.synth = 0
        self.last_qa = None

    def load_model(self): pass
    def unload_model(self): pass
    def get_available_voices(self): return []

    def synthesize(self, segments, voice=None, speed=1.0, progress_cb=None, seed=None, **kw):
        self.synth += 1
        t = np.arange(24000 * 3) / 24000
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), np.ones(24000 * 3, bool)


class FakeEL(SpeechEngine):
    sample_rate = 48000

    def __init__(self):
        self.synth = 0
        self.last_manifest = {"chars_billed": 42, "cache_hits": 0}
        self.last_qa = None

    def load_model(self): pass
    def unload_model(self): pass
    def get_available_voices(self): return []

    def synthesize(self, segments, voice=None, speed=None, progress_cb=None, **kw):
        self.synth += 1
        assert all(s["type"] in ("speech", "pause") for s in segments)
        t = np.arange(48000 * 3) / 48000
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), np.ones(48000 * 3, bool)


class FakeMusic:
    def __init__(self, *a, **k): pass
    def load_model(self): pass
    def unload_model(self): pass
    def generate(self, prompt, duration_sec, progress_cb=None, **kw):
        return np.full(int(duration_sec * 48000), 0.05, dtype=np.float32)
