"""Tests for TTS Engine Registry."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.engine_registry import (
    BUILTIN_ENGINES,
    demote_model,
    get_engine_info,
    get_model_presets,
    list_all_engines,
    load_promoted_models,
    promote_model,
    save_promoted_models,
)


class TestEngineRegistry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_file = Path(self.temp_dir.name) / "promoted_models.json"
        self.patcher = patch("core.engine_registry._PROMOTED_MODELS_FILE", self.mock_file)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_list_all_engines_includes_builtins(self):
        engines = list_all_engines()
        engine_ids = [eid for _label, eid in engines]
        self.assertIn("f5", engine_ids)
        self.assertIn("kokoro", engine_ids)

    def test_get_engine_info_builtin(self):
        info_f5 = get_engine_info("f5")
        self.assertIsNotNone(info_f5)
        self.assertEqual(info_f5["name"], "F5-TTS")

        info_kokoro = get_engine_info("kokoro")
        self.assertIsNotNone(info_kokoro)
        self.assertEqual(info_kokoro["name"], "Kokoro")

    def test_promote_and_load_model(self):
        presets = {
            "speed": 0.88,
            "target_wpm": 95,
            "cfg_strength": 1.4,
            "reverb_amount": 0.20,
            "duck_amount_db": -18.0,
        }
        config = {
            "voice_slug": "calm_brittney",
        }
        record = promote_model(
            model_id="test_zen_model",
            display_name="Test Zen Model",
            base_engine="f5",
            config=config,
            presets=presets,
            description="Optimal zen voice test",
        )
        self.assertEqual(record["id"], "test_zen_model")
        self.assertEqual(record["name"], "Test Zen Model")

        # Verify it loads from disk
        promoted = load_promoted_models()
        self.assertIn("test_zen_model", promoted)
        self.assertEqual(promoted["test_zen_model"]["presets"]["cfg_strength"], 1.4)

        # Verify it appears in list_all_engines
        engines = list_all_engines()
        engine_ids = [eid for _label, eid in engines]
        self.assertIn("test_zen_model", engine_ids)

        # Verify presets lookup
        saved_presets = get_model_presets("test_zen_model")
        self.assertEqual(saved_presets["speed"], 0.88)

    def test_demote_model(self):
        promote_model(
            model_id="temp_model",
            display_name="Temporary Model",
            base_engine="kokoro",
        )
        self.assertIn("temp_model", load_promoted_models())

        success = demote_model("temp_model")
        self.assertTrue(success)
        self.assertNotIn("temp_model", load_promoted_models())

        # Demoting non-existent model returns False
        self.assertFalse(demote_model("non_existent"))

    def test_promote_model_empty_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            promote_model(model_id="   ", display_name="Empty", base_engine="f5")
