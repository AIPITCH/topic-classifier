"""Persist LLM classifications as reusable training examples."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_WRITE_LOCK = threading.Lock()


def training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the training-data configuration section."""
    return config.get("training") or {}


def enabled(config: dict[str, Any]) -> bool:
    """Return whether successful LLM results should be recorded."""
    value = training_config(config).get("enabled", False)
    return value is True or str(value).lower() in {"1", "true", "yes", "on"}


def dataset_path(config: dict[str, Any], base_dir: str) -> str:
    """Resolve the configured JSON Lines dataset path."""
    path = str(training_config(config).get("dataset_path") or ".cache/training.jsonl")
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def append_example(
    config: dict[str, Any],
    base_dir: str,
    document: str,
    labels: list[dict[str, Any]],
    model: str,
) -> None:
    """Append one portable BERTopic training example as a JSONL record."""
    if not enabled(config):
        return
    record = {
        "text": document,
        "labels": [str(label["uid"]) for label in labels if label.get("uid")],
        "label_values": [str(label.get("value") or "") for label in labels],
        "teacher": model,
        "created_at": int(time.time()),
    }
    path = dataset_path(config, base_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _WRITE_LOCK, open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
