"""Optional BERTopic inference backend and model manifest handling."""

from __future__ import annotations

import importlib
import json
import os
from typing import Any


def backend_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return BERTopic runtime configuration."""
    return config.get("bertopic") or {}


def configured(config: dict[str, Any]) -> bool:
    """Return whether automatic BERTopic inference is enabled."""
    value = backend_config(config).get("enabled", False)
    return value is True or str(value).lower() in {"1", "true", "yes", "on"}


class BERTopicBackend:
    """Lazily load an eligible BERTopic artifact and classify documents."""

    def __init__(self, config: dict[str, Any], base_dir: str) -> None:
        settings = backend_config(config)
        path = str(settings.get("model_path") or ".cache/bertopic-model")
        self.model_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
        self.minimum_probability = float(settings.get("minimum_probability", 0.0))
        self._model = None
        self._manifest: dict[str, Any] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        manifest_path = os.path.join(self.model_path, "topic-classifier.json")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not manifest.get("eligible"):
            raise ValueError(
                "BERTopic model has not met the configured quality threshold"
            )
        try:
            bertopic_module = importlib.import_module("bertopic")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "BERTopic inference requires: pip install -r requirements-ml.txt"
            ) from error
        self._model = bertopic_module.BERTopic.load(self.model_path)
        self._manifest = manifest

    def classify(self, document: str) -> list[str]:
        """Return taxonomy UUIDs mapped to the document's inferred topic."""
        self._load()
        topics, probabilities = self._model.transform([document])
        topic = int(topics[0])
        probability = 1.0
        if probabilities is not None:
            first = probabilities[0]
            probability = (
                float(max(first)) if hasattr(first, "__iter__") else float(first)
            )
        if topic == -1 or probability < self.minimum_probability:
            return []
        mapping = (self._manifest or {}).get("topic_labels") or {}
        return [str(uid) for uid in mapping.get(str(topic), [])]
