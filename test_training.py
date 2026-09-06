"""Tests for the optional BERTopic training pipeline helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from lib import training
from lib.bertopic_backend import BERTopicBackend
from train_bertopic import f1_score, read_examples, topic_label_map


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


if __name__ == "__main__":
    unittest.main()
