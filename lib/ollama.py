#!/usr/bin/env python3
# coding=utf-8

"""
Ollama configuration and API interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


def api_base(ollama_config: dict[str, Any]) -> str:
    """
    Return Ollama API base from api_base or host/port config.
    """
    if ollama_config.get("api_base"):
        return str(ollama_config["api_base"]).rstrip("/")
    host = str(ollama_config.get("host") or "localhost").strip()
    port = int(ollama_config.get("port") or 11434)
    scheme = "http" if "://" not in host else ""
    prefix = f"{scheme}://" if scheme else ""
    return f"{prefix}{host}:{port}".rstrip("/")


def engine(ollama_config: dict[str, Any]) -> str:
    """
    Return default Ollama engine/model name.
    """
    return str(ollama_config.get("engine") or ollama_config.get("model") or "")


def get_blacklist(config: dict[str, Any]) -> set[str]:
    """
    Return configured model blacklist.
    """
    ollama_config = config.get("ollama") or {}
    values: list[Any] = []
    for key in ("blacklist", "bblacklist"):
        for item in (config.get(key), ollama_config.get(key)):
            if isinstance(item, list):
                values.extend(item)
            elif item:
                values.append(item)
    return {str(model).strip() for model in values if str(model).strip()}


def get_allowlist(config: dict[str, Any]) -> set[str]:
    """
    Return configured model allowlist.
    """
    ollama_config = config.get("ollama") or {}
    values: list[Any] = []
    for key in ("allowlist", "allowed_models"):
        for item in (config.get(key), ollama_config.get(key)):
            if isinstance(item, list):
                values.extend(item)
            elif item:
                values.append(item)
    return {str(model).strip() for model in values if str(model).strip()}


def ensure_model_allowed(model_name: str, config: dict[str, Any]) -> None:
    """
    Reject model use outside configured policy.
    """
    allowlist = get_allowlist(config)
    if allowlist and model_name not in allowlist:
        raise ValueError(f"model is not allowed: {model_name}")
    if model_name in get_blacklist(config):
        raise ValueError(f"model is blacklisted: {model_name}")


def request(
    session: requests.Session,
    method: str,
    url: str,
    timeout: int,
    logger,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run an Ollama API request and return decoded JSON.
    """
    logger.debug("Ollama request: %s %s", method, url)
    response = session.request(method, url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


def configured_timeout(
    ollama_config: dict[str, Any],
    max_timeout_seconds: int,
) -> int:
    """
    Return configured Ollama timeout capped by server max.
    """
    try:
        timeout = int(ollama_config.get("timeout") or 300)
    except (TypeError, ValueError) as error:
        raise ValueError("ollama timeout must be integer") from error
    if timeout <= 0:
        raise ValueError("ollama timeout must be positive")
    return min(timeout, max_timeout_seconds)


def human_size(size_bytes: int | None) -> str:
    """
    Convert byte count to compact human-readable value.
    """
    if size_bytes is None:
        return "unknown"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def extract_context(show_data: dict[str, Any], logger) -> int | None:
    """
    Extract context length from Ollama show response.
    """
    model_info = show_data.get("model_info") or {}
    for key, value in model_info.items():
        if key.endswith(".context_length") or key == "context_length":
            try:
                return int(value)
            except (TypeError, ValueError):
                logger.debug("Invalid context value for %s: %r", key, value)

    parameters = show_data.get("parameters")
    if isinstance(parameters, str):
        for line in parameters.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in {"num_ctx", "context_length"}:
                try:
                    return int(parts[1])
                except ValueError:
                    logger.debug("Invalid parameter context line: %s", line)
    return None


def list_models(  # pylint: disable=too-many-locals
    config: dict[str, Any],
    max_timeout_seconds: int,
    logger,
) -> list[dict[str, Any]]:
    """
    List Ollama models and log each model context length in debug output.
    """
    ollama_config = config.get("ollama") or {}
    base_url = api_base(ollama_config)
    timeout = configured_timeout(ollama_config, max_timeout_seconds)
    blacklist = get_blacklist(config)
    allowlist = get_allowlist(config)

    logger.info("Connecting to Ollama: %s", base_url)
    if allowlist:
        logger.info("Allowed models: %s", ", ".join(sorted(allowlist)))
    if blacklist:
        logger.info("Blacklisted models: %s", ", ".join(sorted(blacklist)))
    session = requests.Session()

    tags = request(session, "GET", f"{base_url}/api/tags", timeout, logger)
    models = sorted(tags.get("models") or [], key=lambda item: item.get("name", ""))
    logger.info("Ollama models found: %d", len(models))

    output = []
    for model in models:
        name = model.get("name")
        if not name:
            logger.debug("Skipping malformed Ollama model entry: %s", model)
            continue
        if allowlist and name not in allowlist:
            logger.debug("Skipping non-allowed Ollama model: name=%s", name)
            continue
        if name in blacklist:
            logger.debug("Skipping blacklisted Ollama model: name=%s", name)
            continue

        show_data = request(
            session,
            "POST",
            f"{base_url}/api/show",
            timeout,
            logger,
            json={"model": name},
        )
        capabilities = show_data.get("capabilities") or []
        if "completion" not in capabilities:
            logger.debug(
                "Skipping non-completion Ollama model: name=%s capabilities=%s",
                name,
                capabilities,
            )
            continue

        context_length = extract_context(show_data, logger)
        details = show_data.get("details") or model.get("details") or {}
        row = {
            "name": name,
            "id": model.get("digest", "")[:12],
            "size": human_size(model.get("size")),
            "modified_at": model.get("modified_at"),
            "architecture": details.get("family") or details.get("format"),
            "parameters": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "capabilities": capabilities,
            "context_length": context_length,
        }
        output.append(row)
        logger.debug(
            "Ollama model context: name=%s context_length=%s",
            row["name"],
            row["context_length"],
        )

    logger.info("Ollama completion models cached: %d", len(output))
    return output


def generate(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    session: requests.Session,
    config: dict[str, Any],
    max_timeout_seconds: int,
    logger,
    prompt: str,
    model: str | None = None,
    timeout_override: int | None = None,
    validate_model: bool = True,
    context: list[int] | None = None,
) -> dict[str, Any]:
    """
    Call Ollama generate with configured defaults.
    """
    ollama_config = config.get("ollama") or {}
    base_url = api_base(ollama_config)
    timeout = timeout_override or configured_timeout(ollama_config, max_timeout_seconds)
    temperature = float(ollama_config.get("temperature", 0.1))
    model_name = model or engine(ollama_config)
    if not model_name:
        raise ValueError("missing model")
    if validate_model:
        ensure_model_allowed(model_name, config)

    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    if context:
        payload["context"] = context

    return request(
        session,
        "POST",
        f"{base_url}/api/generate",
        timeout,
        logger,
        json=payload,
    )


@dataclass(slots=True)
class OllamaClient:
    """
    Small wrapper around Ollama config, timeout policy, and logging.
    """

    config: dict[str, Any]
    max_timeout_seconds: int
    logger: Any

    @property
    def ollama_config(self) -> dict[str, Any]:
        """
        Return Ollama-specific config.
        """
        return self.config.get("ollama") or {}

    def api_base(self) -> str:
        """
        Return configured API base.
        """
        return api_base(self.ollama_config)

    def engine(self) -> str:
        """
        Return configured default model.
        """
        return engine(self.ollama_config)

    def timeout(self, timeout_override: int | None = None) -> int:
        """
        Return request timeout.
        """
        return timeout_override or configured_timeout(
            self.ollama_config, self.max_timeout_seconds
        )

    def ensure_model_allowed(self, model_name: str) -> None:
        """
        Validate model policy.
        """
        ensure_model_allowed(model_name, self.config)

    def request(
        self,
        session: requests.Session,
        method: str,
        path: str,
        timeout: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run request against configured Ollama API base.
        """
        return request(
            session,
            method,
            f"{self.api_base()}{path}",
            timeout,
            self.logger,
            **kwargs,
        )

    def health(self, timeout: int) -> dict[str, Any]:
        """
        Probe Ollama tags endpoint.
        """
        session = requests.Session()
        return self.request(session, "GET", "/api/tags", timeout)

    def list_models(self) -> list[dict[str, Any]]:
        """
        List usable completion models.
        """
        return list_models(self.config, self.max_timeout_seconds, self.logger)

    def generate(
        self,
        session: requests.Session,
        prompt: str,
        model: str | None = None,
        timeout_override: int | None = None,
        validate_model: bool = True,
        context: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Call Ollama generate.
        """
        model_name = model or self.engine()
        if not model_name:
            raise ValueError("missing model")
        if validate_model:
            self.ensure_model_allowed(model_name)
        return generate(
            session,
            self.config,
            self.max_timeout_seconds,
            self.logger,
            prompt,
            model=model_name,
            timeout_override=timeout_override,
            validate_model=validate_model,
            context=context,
        )
