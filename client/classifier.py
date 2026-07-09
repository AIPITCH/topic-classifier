#!/usr/bin/env python3
# coding=utf-8

"""
Client library for the local evaluation API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
import yaml

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(API_DIR, "config.yaml")


class ClassificationError(RuntimeError):
    """
    Raised when the evaluation API rejects or fails a request.
    """


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load project YAML config.
    """
    config_path = os.environ.get("CCE_CONFIG") or path
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


def classifier_api_base(config: dict[str, Any] | None = None) -> str:
    """
    Return API base URL from config.
    """
    flask_config = (config or {}).get("flask") or {}
    host = str(flask_config.get("host") or "127.0.0.1")
    port = int(flask_config.get("port") or 5151)
    return f"http://{host}:{port}"


@dataclass(slots=True)
class ClassificationClient:
    """
    HTTP client for classification_server.py.
    """

    api_base: str = "http://127.0.0.1:5151"
    timeout: int = 60

    @classmethod
    def from_config(cls, path: str = DEFAULT_CONFIG_PATH) -> "ClassificationClient":
        """
        Build client from project config.yaml.
        """
        config = load_config(path)
        ollama_config = config.get("ollama") or {}
        timeout = int(ollama_config.get("timeout") or 60)
        return cls(api_base=classifier_api_base(config), timeout=timeout)

    def get_models(self, capability: str = "completion") -> list[str]:
        """
        Fetch model names live from Ollama through the API.
        """
        response = requests.get(
            f"{self.api_base.rstrip('/')}/getmodels",
            params={"capability": capability},
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        payload = response.json()
        if not isinstance(payload, list):
            raise ClassificationError("invalid /getmodels response")
        return [
            str(model["name"])
            for model in payload
            if isinstance(model, dict) and model.get("name")
        ]

    def warmup_model(
        self,
        *,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        """
        Warm selected model through the API.
        """
        effective_timeout = timeout or self.timeout
        payload: dict[str, Any] = {"timeout": effective_timeout}
        if model:
            payload["model"] = model

        response = requests.post(
            f"{self.api_base.rstrip('/')}/warmup_model",
            json=payload,
            timeout=effective_timeout,
        )
        self._raise_for_error(response)
        result = response.json()
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise ClassificationError("invalid /warmup_model response")

    def evaluate_markdown(
        self,
        markdown: str,
        *,
        model: str | None = None,
        timeout: int | None = None,
        include_raw: bool = False,
        justify: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate Markdown with content-classification taxonomy.
        """
        effective_timeout = timeout or self.timeout
        payload: dict[str, Any] = {
            "data": markdown,
            "timeout": effective_timeout,
            "include_raw": include_raw,
            "justify": justify,
        }
        if model:
            payload["model"] = model

        response = requests.post(
            f"{self.api_base.rstrip('/')}/evaluate",
            json=payload,
            timeout=effective_timeout,
        )
        self._raise_for_error(response)
        result = response.json()
        if not isinstance(result, dict):
            raise ClassificationError("invalid /evaluate response")
        return result

    @staticmethod
    def _raise_for_error(response: requests.Response) -> None:
        """
        Convert HTTP failures into ClassificationError with API message.
        """
        if response.ok:
            return
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        message = payload.get("error") if isinstance(payload, dict) else None
        if not message:
            message = response.text.strip() or response.reason
        raise ClassificationError(f"{response.status_code}: {message}")
