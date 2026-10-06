import unittest

from core.engine_registry import BUILTIN_ENGINES, get_engine, list_all_engines
from core.tts_cache import compute_cache_key


class TestQwen3Registry(unittest.TestCase):
    def test_builtin_entry(self):
        info = BUILTIN_ENGINES["qwen3"]
        self.assertEqual(info["base_engine"], "qwen3")
        self.assertTrue(info["experimental"])
        self.assertEqual(info["presets"]["reverb_amount"], 0.10)

    def test_qwen3_hidden_from_default_engine_list(self):
        self.assertNotIn("qwen3", [eid for _l, eid in list_all_engines()])
        self.assertIn("qwen3", [eid for _l, eid in list_all_engines(include_experimental=True)])

    def test_get_engine_returns_qwen3(self):
        from core.qwen3_tts.engine import Qwen3Engine
        self.assertIsInstance(get_engine("qwen3", voice_slug="Brittney"), Qwen3Engine)


class TestCacheKey(unittest.TestCase):
    def test_existing_cache_key_unchanged(self):
        # Golden hash captured from the pre-change implementation.
        key = compute_cache_key(script="Rest.", content_type="meditation", tts_engine="f5",
                                voice="Brittney", speed=0.8, f5_cfg_strength=2.0)
        self.assertEqual(key, GOLDEN_KEY)

    def test_extra_changes_key(self):
        base = dict(script="Rest.", tts_engine="qwen3", voice="Brittney", speed=0.8)
        a = compute_cache_key(**base, extra={"mode": "clone", "temperature": 0.7})
        b = compute_cache_key(**base, extra={"mode": "custom", "temperature": 0.7})
        self.assertNotEqual(a, b)


class TestPipelineRouting(unittest.TestCase):
    def test_pipeline_source_routes_qwen3(self):
        import inspect
        import core.pipeline as p
        src = inspect.getsource(p)
        self.assertIn("core.qwen3_tts.preprocessor", src)
        self.assertIn("Qwen3MasteringEngine", src)
        self.assertIn("MOODSCAPE_QWEN3_DF_WET", src)


GOLDEN_KEY = "f7425e0249c3e7dc3d2b03557a48f5bfb91229a77b830356fb77f4d33431eb3a"

if __name__ == "__main__":
    unittest.main()
