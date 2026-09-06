"""Tests for the optional BERTopic training pipeline helpers."""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

import classification_server
from lib import training
from lib import health
from lib.bertopic_backend import BERTopicBackend
from train_bertopic import f1_score, read_examples, split_examples, topic_label_map


class TrainingTests(unittest.TestCase):
    """Verify portable data capture and quality metrics."""

    def test_append_example_writes_bertopic_record(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"training": {"enabled": True, "dataset_path": "data.jsonl"}}
            training.append_example(
                config,
                directory,
                "A document",
                [{"uid": "one", "value": "One"}],
                "teacher",
            )
            examples = read_examples(Path(directory) / "data.jsonl")
            self.assertEqual(examples[0]["text"], "A document")
            self.assertEqual(examples[0]["labels"], ["one"])
            self.assertEqual(examples[0]["teacher"], "teacher")

    def test_disabled_capture_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as directory:
            training.append_example({}, directory, "document", [], "teacher")
            self.assertFalse((Path(directory) / ".cache/training.jsonl").exists())

    def test_topic_mapping_uses_support(self):
        examples = [{"labels": ["a", "b"]}, {"labels": ["a"]}, {"labels": ["c"]}]
        self.assertEqual(topic_label_map([0, 0, -1], examples, 0.6), {"0": ["a"]})

    def test_micro_f1(self):
        self.assertAlmostEqual(f1_score([{"a", "b"}], [{"a", "c"}]), 0.5)
        self.assertEqual(f1_score([set()], [set()]), 1.0)

    def test_invalid_record_reports_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(
                json.dumps({"text": 4, "labels": []}) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "line 1"):
                read_examples(path)

    def test_ineligible_model_is_rejected_before_optional_import(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            (model_path / "topic-classifier.json").write_text(
                json.dumps({"eligible": False}), encoding="utf-8"
            )
            backend = BERTopicBackend(
                {"bertopic": {"model_path": str(model_path)}}, directory
            )
            with self.assertRaisesRegex(ValueError, "quality threshold"):
                backend.classify("document")

    def test_malformed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            (model_path / "topic-classifier.json").write_text("[]", encoding="utf-8")
            backend = BERTopicBackend(
                {"bertopic": {"model_path": str(model_path)}}, directory
            )
            with self.assertRaisesRegex(ValueError, "JSON object"):
                backend.classify("document")

    def test_malformed_topic_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            (model_path / "topic-classifier.json").write_text(
                json.dumps({"eligible": True, "topic_labels": []}), encoding="utf-8"
            )
            backend = BERTopicBackend(
                {"bertopic": {"model_path": str(model_path)}}, directory
            )
            with self.assertRaisesRegex(ValueError, "topic_labels"):
                backend.classify("document")

    def test_small_training_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 6"):
            split_examples([{}] * 5, 0.2)

    def test_split_keeps_test_example(self):
        training_examples, test_examples = split_examples([{}] * 8, 0.2)
        self.assertEqual(len(training_examples), 6)
        self.assertEqual(len(test_examples), 2)

    def test_health_reports_selected_backend(self):
        state = health.health_probe_tick(
            {}, lambda _timeout: {"ai_engine": "bertopic"}, logging.getLogger(__name__)
        )
        self.assertEqual(state["status"], "ok")
        self.assertEqual(state["ai_engine"], "bertopic")

    def test_ai_health_falls_back_to_bertopic(self):
        class OfflineOllama:
            """Test double for an unavailable Ollama server."""

            @staticmethod
            def health(_timeout):
                raise requests.ConnectionError("offline")

        class AvailableBERTopic:
            """Test double for an eligible BERTopic artifact."""

            @staticmethod
            def ensure_available():
                return None

        previous_config = classification_server.CONFIG
        previous_backend = classification_server.BERTOPIC_BACKEND
        try:
            classification_server.CONFIG = {"bertopic": {"enabled": True}}
            classification_server.BERTOPIC_BACKEND = AvailableBERTopic()
            with mock.patch.object(
                classification_server, "ollama_client", return_value=OfflineOllama()
            ):
                self.assertEqual(
                    classification_server.ai_health_check(1),
                    {"ai_engine": "bertopic"},
                )
        finally:
            classification_server.CONFIG = previous_config
            classification_server.BERTOPIC_BACKEND = previous_backend


if __name__ == "__main__":
    unittest.main()
