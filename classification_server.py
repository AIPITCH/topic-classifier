#!/usr/bin/env python3
# coding=utf-8

"""
Backend API for a channel classifier using Ollama.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any

import requests
import yaml
from flask import Flask, jsonify, request
from rich.logging import RichHandler
from werkzeug.exceptions import RequestEntityTooLarge

from queries import (
    TAXONOMY_EVALUATION_QUERY,
    TAXONOMY_JUSTIFIED_EVALUATION_QUERY,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(THIS_DIR, "config.yaml")
DEFAULT_TAXONOMY_URL = (
    "https://raw.githubusercontent.com/MISP/misp-taxonomies/main/"
    "content-classification/machinetag.json"
)
DEFAULT_TAXONOMY_CACHE_PATH = os.path.join(
    THIS_DIR, ".cache", "content-classification.json"
)
SECONDS_PER_DAY = 24 * 60 * 60
DEFAULT_TAXONOMY_CACHE_TTL_DAYS = 30
DEFAULT_TAXONOMY_FETCH_TIMEOUT = 30


def print_meta() -> None:
    """
    Print lightweight startup metadata.
    """
    logger.info("Channel Classifier API")


logger = logging.getLogger("Channel_Classifier")
logger.setLevel(logging.DEBUG)

console_handler = RichHandler()
console_handler.setLevel(logging.INFO)

log_dir = os.path.join(THIS_DIR, "log")
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "cc.log"), mode="a")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="[%X]"
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
}
MODEL_CACHE: list[dict[str, Any]] = []
CONFIG: dict[str, Any] = {}
CONTENT_CLASSIFICATION_TAGS: list[dict[str, str]] = []
USER_MARKDOWN_TOKEN_LIMIT = 10000
MAX_TIMEOUT_SECONDS = 900
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024
UUID_RE = re.compile(
    r"\b[0-9a-zA-Z]{8}-[0-9a-zA-Z]{4}-[0-9a-zA-Z]{4}-"
    r"[0-9a-zA-Z]{4}-[0-9a-zA-Z]{12}\b"
)
APPROX_TOKEN_RE = re.compile(r"\S+")


def load_config() -> dict[str, Any]:
    """
    Load YAML config from config.yaml.
    """
    config = {}
    for config_path in (
        os.environ.get("CCE_CONFIG"),
        DEFAULT_CONFIG_PATH,
    ):
        if not config_path:
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            logger.debug("Loaded config from: %s", config_path)
            break
        except FileNotFoundError:
            continue
    logger.debug("Loaded config: %s", config)
    return config


def ollama_api_base(ollama_config: dict[str, Any]) -> str:
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


def ollama_engine(ollama_config: dict[str, Any]) -> str:
    """
    Return default Ollama engine/model name.
    """
    return str(ollama_config.get("engine") or ollama_config.get("model") or "")


def configure_log_level(config: dict[str, Any]) -> None:
    """
    Configure printout and file log level from config.
    """
    log_level_name = str(config.get("log", "info")).lower()
    if log_level_name not in LOG_LEVELS:
        allowed = ", ".join(sorted(LOG_LEVELS))
        logger.error(
            "Invalid log level %r. Allowed values: %s", log_level_name, allowed
        )
        raise ValueError(f"invalid log level: {log_level_name}")

    log_level = LOG_LEVELS[log_level_name]
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)

    requests_logger = logging.getLogger("urllib3")
    requests_logger.setLevel(log_level)
    logger.debug("Log level configured: %s", log_level_name)


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


def ollama_request(
    session: requests.Session,
    method: str,
    url: str,
    timeout: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run an Ollama API request and return decoded JSON.
    """
    logger.debug("Ollama request: %s %s", method, url)
    response = session.request(method, url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


def parse_timeout(value: Any) -> int | None:
    """
    Parse optional request timeout override.
    """
    if value in (None, ""):
        return None
    try:
        timeout = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("timeout must be integer") from error
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout must be <= {MAX_TIMEOUT_SECONDS}")
    return timeout


def configured_timeout(ollama_config: dict[str, Any]) -> int:
    """
    Return configured Ollama timeout capped by server max.
    """
    try:
        timeout = int(ollama_config.get("timeout") or 300)
    except (TypeError, ValueError) as error:
        raise ValueError("ollama timeout must be integer") from error
    if timeout <= 0:
        raise ValueError("ollama timeout must be positive")
    return min(timeout, MAX_TIMEOUT_SECONDS)


def configured_max_body_bytes(config: dict[str, Any]) -> int:
    """
    Return Flask body size limit.
    """
    flask_config = config.get("flask") or {}
    value = (
        flask_config.get("max_body_bytes")
        or flask_config.get("max_content_length")
        or DEFAULT_MAX_BODY_BYTES
    )
    try:
        size = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("max_body_bytes must be integer") from error
    if size <= 0:
        raise ValueError("max_body_bytes must be positive")
    return size


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


def extract_context(show_data: dict[str, Any]) -> int | None:
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


def list_ollama_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    List Ollama models and log each model context length in debug output.
    """
    ollama_config = config.get("ollama") or {}
    api_base = ollama_api_base(ollama_config)
    timeout = configured_timeout(ollama_config)
    blacklist = get_blacklist(config)
    allowlist = get_allowlist(config)

    logger.info("Connecting to Ollama: %s", api_base)
    if allowlist:
        logger.info("Allowed models: %s", ", ".join(sorted(allowlist)))
    if blacklist:
        logger.info("Blacklisted models: %s", ", ".join(sorted(blacklist)))
    session = requests.Session()

    tags = ollama_request(session, "GET", f"{api_base}/api/tags", timeout)
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

        show_data = ollama_request(
            session,
            "POST",
            f"{api_base}/api/show",
            timeout,
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

        context_length = extract_context(show_data)
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


def taxonomy_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Return content-classification taxonomy config.
    """
    return config.get("taxonomy") or config.get("content_classification") or {}


def resolve_api_path(path: str) -> str:
    """
    Resolve API-relative paths.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(THIS_DIR, path)


def content_classification_cache_path(config: dict[str, Any]) -> str:
    """
    Return content-classification taxonomy cache path.
    """
    value = taxonomy_config(config).get("cache_path")
    if not value:
        return DEFAULT_TAXONOMY_CACHE_PATH
    return resolve_api_path(str(value))


def content_classification_url(config: dict[str, Any]) -> str:
    """
    Return content-classification taxonomy source URL.
    """
    return str(taxonomy_config(config).get("url") or DEFAULT_TAXONOMY_URL)


def content_classification_cache_ttl_seconds(config: dict[str, Any]) -> int:
    """
    Return taxonomy cache TTL in seconds from configured days.
    """
    value = taxonomy_config(config).get(
        "cache_ttl_days", DEFAULT_TAXONOMY_CACHE_TTL_DAYS
    )
    try:
        ttl_days = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("taxonomy cache_ttl_days must be integer") from error
    if ttl_days < 0:
        raise ValueError("taxonomy cache_ttl_days must be >= 0")
    return ttl_days * SECONDS_PER_DAY


def content_classification_fetch_timeout(config: dict[str, Any]) -> int:
    """
    Return taxonomy fetch timeout in seconds.
    """
    value = taxonomy_config(config).get("timeout", DEFAULT_TAXONOMY_FETCH_TIMEOUT)
    try:
        timeout = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("taxonomy timeout must be integer") from error
    if timeout <= 0:
        raise ValueError("taxonomy timeout must be positive")
    return timeout


def validate_content_classification_taxonomy(taxonomy: Any) -> dict[str, Any]:
    """
    Validate minimal content-classification taxonomy structure.
    """
    if not isinstance(taxonomy, dict):
        raise ValueError("taxonomy must be object")
    if taxonomy.get("namespace") != "content-classification":
        raise ValueError("taxonomy namespace must be content-classification")
    if not isinstance(taxonomy.get("predicates"), list):
        raise ValueError("taxonomy predicates must be array")
    if not isinstance(taxonomy.get("values"), list):
        raise ValueError("taxonomy values must be array")
    return taxonomy


def read_content_classification_taxonomy(path: str) -> dict[str, Any]:
    """
    Read and validate cached content-classification taxonomy.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return validate_content_classification_taxonomy(json.load(handle))


def write_content_classification_taxonomy(path: str, taxonomy: dict[str, Any]) -> None:
    """
    Write taxonomy cache atomically.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(taxonomy, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def taxonomy_cache_is_fresh(path: str, ttl_seconds: int) -> bool:
    """
    Return true when taxonomy cache exists and is younger than ttl.
    """
    if not os.path.exists(path):
        return False
    if ttl_seconds == 0:
        return False
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds <= ttl_seconds


def fetch_content_classification_taxonomy(config: dict[str, Any]) -> dict[str, Any]:
    """
    Fetch content-classification taxonomy from GitHub.
    """
    url = content_classification_url(config)
    timeout = content_classification_fetch_timeout(config)
    logger.info("Fetching content-classification taxonomy: %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return validate_content_classification_taxonomy(response.json())


def ensure_content_classification_taxonomy(config: dict[str, Any]) -> str:
    """
    Ensure content-classification taxonomy cache exists and is fresh enough.
    """
    path = content_classification_cache_path(config)
    ttl_seconds = content_classification_cache_ttl_seconds(config)
    if taxonomy_cache_is_fresh(path, ttl_seconds):
        read_content_classification_taxonomy(path)
        logger.info("Using cached content-classification taxonomy: %s", path)
        return path

    cache_exists = os.path.exists(path)
    try:
        taxonomy = fetch_content_classification_taxonomy(config)
        write_content_classification_taxonomy(path, taxonomy)
        logger.info("Cached content-classification taxonomy: %s", path)
    except (
        OSError,
        ValueError,
        requests.RequestException,
        json.JSONDecodeError,
    ) as error:
        if cache_exists:
            read_content_classification_taxonomy(path)
            logger.warning("Taxonomy refresh failed, using stale cache: %s", error)
            return path
        raise ValueError(f"taxonomy fetch failed: {error}") from error
    return path


def load_content_classification_tags() -> list[dict[str, str]]:
    """
    Load content-classification taxonomy tags.
    """
    taxonomy_path = ensure_content_classification_taxonomy(CONFIG)
    taxonomy = read_content_classification_taxonomy(taxonomy_path)

    namespace = str(taxonomy.get("namespace") or "content-classification")
    predicates = {
        str(item.get("value")): item
        for item in taxonomy.get("predicates") or []
        if isinstance(item, dict) and item.get("value")
    }
    tags = []
    for group in taxonomy.get("values") or []:
        if not isinstance(group, dict):
            continue
        predicate = str(group.get("predicate") or "")
        predicate_expanded = str(
            (predicates.get(predicate) or {}).get("expanded") or ""
        )
        for entry in group.get("entry") or []:
            if (
                not isinstance(entry, dict)
                or not entry.get("uuid")
                or not entry.get("value")
            ):
                continue
            value = str(entry["value"])
            label = f'{namespace}:{predicate}="{value}"'
            tags.append(
                {
                    "uid": str(entry["uuid"]).lower(),
                    "label": label,
                    "predicate": predicate,
                    "predicate_expanded": predicate_expanded,
                    "value": value,
                    "expanded": str(entry.get("expanded") or ""),
                    "description": str(entry.get("description") or ""),
                }
            )
    return sorted(tags, key=lambda item: item["label"].casefold())


def get_content_classification_tags() -> list[dict[str, str]]:
    """
    Return cached content-classification taxonomy tags.
    """
    global CONTENT_CLASSIFICATION_TAGS
    if not CONTENT_CLASSIFICATION_TAGS:
        CONTENT_CLASSIFICATION_TAGS = load_content_classification_tags()
    return CONTENT_CLASSIFICATION_TAGS


def public_taxonomy_label(
    label: dict[str, str],
    justification: str | None = None,
) -> dict[str, str]:
    """
    Return public /evaluate label shape.
    """
    public_label = {
        "uid": label["uid"],
        "value": label.get("expanded", ""),
        "predicate": label.get("predicate_expanded", ""),
        "description": label.get("description", ""),
    }
    if justification is not None:
        public_label["justification"] = justification
    return public_label


def content_classification_prompt_entry(tag: dict[str, str]) -> str:
    """
    Return one content-classification taxonomy prompt entry.
    """
    description = tag["description"] or tag["expanded"]
    if description:
        return f'{tag["uid"]}: {tag["label"]}: {description}'
    return f'{tag["uid"]}: {tag["label"]}'


def content_classification_prompt_entries() -> list[str]:
    """
    Return prompt entries for content-classification taxonomy.
    """
    entries = []
    for tag in get_content_classification_tags():
        entries.append(content_classification_prompt_entry(tag))
    return entries


def parse_uuid_output(raw_output: str) -> list[str]:
    """
    Extract unique UUIDs from model output.
    """
    seen = set()
    output = []
    for match in UUID_RE.finditer(raw_output):
        uid = match.group(0).lower()
        if uid not in seen:
            seen.add(uid)
            output.append(uid)
    return output


def parse_bool(value: Any) -> bool:
    """
    Parse truthy API values.
    """
    return str(value or "").lower() in {"1", "true", "yes"}


def strip_json_fence(text: str) -> str:
    """
    Remove common markdown JSON fences.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw


def extract_json_value(text: str) -> Any:
    """
    Parse JSON value, tolerating surrounding text.
    """
    raw = strip_json_fence(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        starts = [
            (index, char) for char in ("[", "{") if (index := raw.find(char)) >= 0
        ]
        if not starts:
            raise
        start, char = min(starts, key=lambda item: item[0])
        end_char = "]" if char == "[" else "}"
        end = raw.rfind(end_char)
        if end <= start:
            raise
        return json.loads(raw[start : end + 1])


def parse_justified_output(raw_output: str) -> list[dict[str, str]]:
    """
    Parse justified taxonomy model output.
    """
    data = extract_json_value(raw_output)
    if isinstance(data, dict):
        for key in ("labels", "items", "results", "matches"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            mapped_items = []
            for key, value in data.items():
                uid_values = parse_uuid_output(str(key))
                if not uid_values:
                    continue
                if isinstance(value, dict):
                    justification_value = (
                        value.get("justification")
                        or value.get("justifications")
                        or value.get("rationale")
                        or value.get("explanation")
                        or value.get("reason")
                        or value.get("why")
                        or value.get("evidence")
                        or ""
                    )
                else:
                    justification_value = value
                justification = str(justification_value).strip()
                if justification:
                    mapped_items.append(
                        {"uid": uid_values[0], "justification": justification}
                    )
            data = mapped_items or [data]
    if not isinstance(data, list):
        raise ValueError("justified output must be a JSON array")

    items = []
    for item in data:
        if not isinstance(item, dict):
            continue
        match_value = item.get("match")
        if match_value is False or str(match_value).lower() in {"false", "no", "0"}:
            continue
        uid_text = str(item.get("uid") or item.get("uuid") or "")
        uid_values = parse_uuid_output(uid_text)
        justification = str(
            item.get("justification")
            or item.get("justifications")
            or item.get("rationale")
            or item.get("explanation")
            or item.get("reason")
            or item.get("why")
            or item.get("evidence")
            or ""
        ).strip()
        if uid_values and justification:
            items.append({"uid": uid_values[0], "justification": justification})
    return items


def uuid_distance(left: str, right: str) -> int:
    """
    Return character distance between two UUID strings.
    """
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(
        1 for left_char, right_char in zip(left, right) if left_char != right_char
    )


def correct_uuid(uid: str, known_uids: set[str]) -> tuple[str | None, bool]:
    """
    Correct UID when exactly one known UID differs by one character.
    """
    if uid in known_uids:
        return uid, False

    candidates = [
        known_uid for known_uid in known_uids if uuid_distance(uid, known_uid) == 1
    ]
    if len(candidates) == 1:
        return candidates[0], True
    return None, False


def resolve_uids(
    selected_uids: list[str],
    known_uids: set[str],
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """
    Resolve selected UIDs with one-character correction.
    """
    resolved = []
    unknown = []
    corrections = []
    seen = set()
    for uid in selected_uids:
        corrected_uid, corrected = correct_uuid(uid, known_uids)
        if corrected_uid is None:
            unknown.append(uid)
            continue
        if corrected:
            corrections.append({"raw_uid": uid, "corrected_uid": corrected_uid})
        if corrected_uid not in seen:
            seen.add(corrected_uid)
            resolved.append(corrected_uid)
    return resolved, unknown, corrections


def truncate_user_markdown(
    markdown: str,
    token_limit: int = USER_MARKDOWN_TOKEN_LIMIT,
) -> tuple[str, int, bool]:
    """
    Truncate user Markdown to token_limit tokens.
    """
    text = markdown.strip()
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= token_limit:
            return text, len(tokens), False
        return encoding.decode(tokens[:token_limit]), len(tokens), True
    except Exception as error:
        logger.warning("Token truncation fallback used: %s", error)
        matches = list(APPROX_TOKEN_RE.finditer(text))
        estimated_tokens = max(len(matches), len(text) // 4)
        if estimated_tokens <= token_limit:
            return text, estimated_tokens, False
        if len(matches) > token_limit:
            return text[: matches[token_limit - 1].end()], estimated_tokens, True
        return text[: token_limit * 4], estimated_tokens, True


def json_to_markdown(data: Any, title: str = "sample_channel") -> str:
    """
    Convert JSON-like data to Markdown.
    """
    lines = [f"# {title}", ""]

    def render(value: Any, level: int = 2) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lines.append(f"{'#' * level} {key}")
                render(item, min(level + 1, 6))
        elif isinstance(value, list):
            for index, item in enumerate(value, 1):
                lines.append(f"{'#' * level} Item {index}")
                render(item, min(level + 1, 6))
        elif value is None:
            lines.append("")
        else:
            lines.append(str(value))
            lines.append("")

    render(data)
    return "\n".join(lines).strip()


def evaluate_markdown(
    markdown: str,
    model: str | None = None,
    timeout_override: int | None = None,
    justify: bool = False,
) -> dict[str, Any]:
    """
    Evaluate Markdown against content-classification taxonomy and return JSON data.
    """
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError("data must be non-empty markdown text")

    ollama_config = CONFIG.get("ollama") or {}
    api_base = ollama_api_base(ollama_config)
    timeout = timeout_override or configured_timeout(ollama_config)
    temperature = float(ollama_config.get("temperature", 0.1))
    model_name = model or ollama_engine(ollama_config)
    if not model_name:
        raise ValueError("missing model")
    ensure_model_allowed(model_name, CONFIG)

    user_markdown, input_tokens, input_truncated = truncate_user_markdown(markdown)
    evaluation_query = TAXONOMY_EVALUATION_QUERY.format(
        taxonomy_tags="\n".join(
            f"- {entry}" for entry in content_classification_prompt_entries()
        )
    )
    evaluation_prompt = f"{evaluation_query}\n\n# Input\n{user_markdown}"

    logger.info("Evaluating Markdown with model: %s", model_name)
    session = requests.Session()
    start_time = time.perf_counter()
    evaluation_response = ollama_request(
        session,
        "POST",
        f"{api_base}/api/generate",
        timeout,
        json={
            "model": model_name,
            "prompt": evaluation_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        },
    )
    raw_output = str(evaluation_response.get("response", "")).strip()
    by_uid = {tag["uid"]: tag for tag in get_content_classification_tags()}
    known_uids = set(by_uid)
    selected_uids = parse_uuid_output(raw_output)
    resolved_uids, _unknown_uids, _corrected_uids = resolve_uids(
        selected_uids,
        known_uids,
    )
    raw_outputs: str | dict[str, str] = raw_output

    if justify:
        labels = []
        if resolved_uids:
            selected_entries = [
                content_classification_prompt_entry(by_uid[uid])
                for uid in resolved_uids
            ]
            justification_query = TAXONOMY_JUSTIFIED_EVALUATION_QUERY.format(
                taxonomy_tags="\n".join(f"- {entry}" for entry in selected_entries)
            )
            justification_prompt = f"{justification_query}\n\n# Input\n{user_markdown}"
            logger.info("Justifying taxonomy labels with model: %s", model_name)
            justification_response = ollama_request(
                session,
                "POST",
                f"{api_base}/api/generate",
                timeout,
                json={
                    "model": model_name,
                    "prompt": justification_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    },
                },
            )
            justification_raw_output = str(
                justification_response.get("response", "")
            ).strip()
            raw_outputs = {
                "evaluation": raw_output,
                "justification": justification_raw_output,
            }
        else:
            justification_raw_output = "[]"

        selected_uid_set = set(resolved_uids)
        seen = set()
        for item in parse_justified_output(justification_raw_output):
            corrected_uid, _corrected = correct_uuid(item["uid"], selected_uid_set)
            if corrected_uid is None:
                continue
            if corrected_uid in seen:
                continue
            seen.add(corrected_uid)
            labels.append(
                public_taxonomy_label(
                    by_uid[corrected_uid],
                    justification=item["justification"],
                )
            )
    else:
        labels = [public_taxonomy_label(by_uid[uid]) for uid in resolved_uids]

    processing_time_seconds = time.perf_counter() - start_time
    return {
        "labels": labels,
        "justify": justify,
        "processing_time_seconds": round(processing_time_seconds, 3),
        "truncated": input_truncated,
        "input_tokens": input_tokens,
        "input_truncated": input_truncated,
        "input_token_limit": USER_MARKDOWN_TOKEN_LIMIT,
        "raw_output": raw_outputs,
    }


def warmup_model(
    model: str | None = None,
    timeout_override: int | None = None,
) -> None:
    """
    Send a tiny prompt to load a model before timed requests.
    """
    ollama_config = CONFIG.get("ollama") or {}
    api_base = ollama_api_base(ollama_config)
    timeout = timeout_override or configured_timeout(ollama_config)
    temperature = float(ollama_config.get("temperature", 0.1))
    model_name = model or ollama_engine(ollama_config)
    if not model_name:
        raise ValueError("missing model")
    ensure_model_allowed(model_name, CONFIG)

    logger.info("Warming up model: %s", model_name)
    session = requests.Session()
    ollama_request(
        session,
        "POST",
        f"{api_base}/api/generate",
        timeout,
        json={
            "model": model_name,
            "prompt": "say 'hello world'",
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        },
    )


def cached_model_contexts(capability: str | None = None) -> list[dict[str, Any]]:
    """
    Return cached model names and context lengths.
    """
    return [
        {
            "name": model["name"],
            "context_length": model["context_length"],
        }
        for model in MODEL_CACHE
        if capability is None or capability in model.get("capabilities", [])
    ]


def create_app() -> Flask:
    """
    Create Flask API app.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = configured_max_body_bytes(CONFIG)

    @app.errorhandler(RequestEntityTooLarge)
    def request_entity_too_large(_error):
        return jsonify({"error": "request body too large"}), 413

    @app.get("/getmodels")
    def getmodels():
        capability = request.args.get("capability")
        try:
            MODEL_CACHE[:] = list_ollama_models(CONFIG)
        except requests.RequestException as error:
            logger.error("Ollama model listing failed: %s", error)
            return jsonify({"error": "ollama model listing failed"}), 502
        return jsonify(cached_model_contexts(capability))

    @app.post("/evaluate")
    def evaluate_route():
        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            markdown = (
                payload.get("data")
                or payload.get("markdown")
                or payload.get("md")
                or payload.get("text")
            )
            model = payload.get("model") or request.args.get("model")
            include_raw = (
                payload["include_raw"]
                if "include_raw" in payload
                else request.args.get("include_raw")
            )
            justify = (
                payload["justify"]
                if "justify" in payload
                else request.args.get("justify")
            )
            timeout_value = (
                payload["timeout"]
                if "timeout" in payload
                else request.args.get("timeout")
            )
        else:
            markdown = request.get_data(as_text=True)
            model = request.args.get("model")
            include_raw = request.args.get("include_raw")
            justify = request.args.get("justify")
            timeout_value = request.args.get("timeout")

        include_raw = parse_bool(include_raw)
        justify = parse_bool(justify)
        if not isinstance(markdown, str) or not markdown.strip():
            return jsonify({"error": "data must be non-empty markdown text"}), 400
        if model is not None and not isinstance(model, str):
            return jsonify({"error": "model must be string"}), 400

        try:
            timeout_override = parse_timeout(timeout_value)
            result = evaluate_markdown(markdown, model, timeout_override, justify)
        except json.JSONDecodeError as error:
            logger.error("Ollama justified evaluation returned invalid JSON: %s", error)
            return jsonify({"error": "ollama evaluation returned invalid json"}), 502
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except requests.RequestException as error:
            logger.error("Ollama evaluation failed: %s", error)
            return jsonify({"error": "ollama evaluation failed"}), 502

        if not include_raw:
            result.pop("raw_output", None)
        result.pop("corrected_uids", None)
        return jsonify(result)

    @app.post("/warmup_model")
    def warmup_model_route():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "invalid json body"}), 400

        model = payload.get("model") or request.args.get("model")
        if model is not None and not isinstance(model, str):
            return jsonify({"error": "model must be string"}), 400

        try:
            timeout_override = parse_timeout(
                payload.get("timeout") or request.args.get("timeout")
            )
            warmup_model(model, timeout_override)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except requests.RequestException as error:
            logger.error("Ollama warmup failed: %s", error)
            return jsonify({"error": "ollama warmup failed"}), 502

        return jsonify({"status": "ok"})

    return app


def main() -> int:
    """
    Entrypoint.
    """
    parser = argparse.ArgumentParser(description="Channel Classifier")
    parser.parse_args()

    global CONFIG
    CONFIG = load_config()
    try:
        configure_log_level(CONFIG)
    except ValueError:
        return 1

    print_meta()
    try:
        ensure_content_classification_taxonomy(CONFIG)
        MODEL_CACHE[:] = list_ollama_models(CONFIG)
    except ValueError as error:
        logger.error("Invalid config: %s", error)
        return 1
    except requests.RequestException as error:
        logger.error("Ollama connection failed: %s", error)
        return 1

    flask_config = CONFIG.get("flask") or {}
    host = str(flask_config.get("host") or "127.0.0.1")
    port = int(flask_config.get("port") or 5151)
    logger.info("Starting API: http://%s:%s", host, port)
    try:
        app = create_app()
    except ValueError as error:
        logger.error("Invalid config: %s", error)
        return 1
    app.run(host=host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
