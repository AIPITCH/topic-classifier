#!/usr/bin/env python3
# coding=utf-8

"""
Backend API for a channel classifier using Ollama.
"""

import argparse
import datetime
import functools
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from functools import partial
from queue import Queue
from typing import Any

import requests
import yaml
from flask import Flask, jsonify, request, send_from_directory
from flask_httpauth import HTTPTokenAuth
from rich.logging import RichHandler
from werkzeug.exceptions import RequestEntityTooLarge

from lib import health as health_lib
from lib import ollama
from lib import scheduler as scheduler_lib
from lib import uids
from queries import (
    SUMMARY_QUERY,
    TAXONOMY_EVALUATION_QUERY,
    TAXONOMY_JUSTIFIED_EVALUATION_QUERY,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(THIS_DIR, "config.yaml")
DEFAULT_CONFIG_TEMPLATE_PATH = os.path.join(THIS_DIR, "config.yaml.default")
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
DEFAULT_JOB_CACHE_PATH = os.path.join(THIS_DIR, ".cache", "classification_jobs")
DEFAULT_JOB_CACHE_TTL_HOURS = 24
DEFAULT_JOB_WORKERS = 1
DEFAULT_LOG_RETENTION_DAYS = 30
DEFAULT_CLIENT_IP_HEADER = "X-Forwarded-For"


def print_meta() -> None:
    """
    Print lightweight startup metadata.
    """
    logger.info("Channel Classifier API")
    state = "activated" if auth_enabled(CONFIG) else "deactivated"
    logger.info("Authentication %s", state)


logger = logging.getLogger("Channel_Classifier")
logger.setLevel(logging.DEBUG)

console_handler = RichHandler()
console_handler.setLevel(logging.INFO)

log_dir = os.path.join(THIS_DIR, "log")
os.makedirs(log_dir, exist_ok=True)
LOG_INTERNAL_CLIENT_IP = "127.0.0.1"


class UtcClientIpFormatter(logging.Formatter):
    """
    Format file logs with UTC timestamp and a default internal client IP.
    """

    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "client_ip"):
            record.client_ip = LOG_INTERNAL_CLIENT_IP
        return super().format(record)


file_formatter = UtcClientIpFormatter(
    "%(asctime)s - %(levelname)s - %(client_ip)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)


class DailyLogFileHandler(logging.Handler):
    """
    Append logs to one dated file per local day and prune old files.
    """

    def __init__(
        self,
        directory: str,
        prefix: str,
        retention_days: int = DEFAULT_LOG_RETENTION_DAYS,
        rotate_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.directory = directory
        self.prefix = prefix
        self.retention_days = retention_days
        self.rotate_enabled = rotate_enabled
        self.current_date = ""
        self.current_path = ""
        self.stream = None

    def emit(self, record: logging.LogRecord) -> None:
        """
        Write one log record to today's file.
        """
        try:
            self._ensure_stream()
            if self.stream is None:
                return
            self.stream.write(f"{self.format(record)}\n")
            self.flush()
        except Exception:  # pylint: disable=broad-exception-caught
            self.handleError(record)

    def flush(self) -> None:
        """
        Flush current log file.
        """
        if self.stream is not None:
            self.stream.flush()

    def close(self) -> None:
        """
        Close current log file.
        """
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        super().close()

    def _ensure_stream(self) -> None:
        today = datetime.date.today().strftime("%Y-%m-%d")
        filename = (
            f"{self.prefix}-{today}.log"
            if self.rotate_enabled
            else f"{self.prefix}.log"
        )
        path = os.path.join(self.directory, filename)
        if (
            today == self.current_date
            and path == self.current_path
            and self.stream is not None
        ):
            return
        if self.stream is not None:
            self.stream.close()
        os.makedirs(self.directory, exist_ok=True)
        self.current_date = today
        self.current_path = path
        self.stream = open(  # pylint: disable=consider-using-with
            path,
            "a",
            encoding="utf-8",
        )
        if self.rotate_enabled:
            self._prune_old_logs(today)

    def _prune_old_logs(self, today: str) -> None:
        if self.retention_days <= 0:
            return
        cutoff = datetime.datetime.strptime(
            today, "%Y-%m-%d"
        ).date() - datetime.timedelta(days=self.retention_days)
        for filename in os.listdir(self.directory):
            if not filename.startswith(f"{self.prefix}-") or not filename.endswith(
                ".log"
            ):
                continue
            date_value = filename[len(self.prefix) + 1 : -4]
            try:
                file_date = datetime.datetime.strptime(date_value, "%Y-%m-%d").date()
            except ValueError:
                continue
            if file_date < cutoff:
                os.remove(os.path.join(self.directory, filename))


file_handler = DailyLogFileHandler(log_dir, "cc")
file_handler.setLevel(logging.INFO)
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
JOB_QUEUE: Queue[str] = Queue()
JOB_LOCK = threading.Lock()
JOB_LOG_CONTEXT = threading.local()
JOB_WORKERS_STARTED = False
QUEUED_JOB_IDS: set[str] = set()
RUNNING_JOB_IDS: set[str] = set()
TOKEN_AUTH = HTTPTokenAuth(scheme="Bearer")
USER_MARKDOWN_TOKEN_LIMIT = 10000
MAX_TIMEOUT_SECONDS = 900
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024
APPROX_TOKEN_RE = re.compile(r"\S+")


def load_config() -> dict[str, Any]:
    """
    Load YAML config.
    """
    config = {}
    for config_path in (
        os.environ.get("CCE_CONFIG"),
        DEFAULT_CONFIG_PATH,
        DEFAULT_CONFIG_TEMPLATE_PATH,
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


def configured_listen_host(flask_config: dict[str, Any]) -> str:
    """
    Return Flask listen host, supporting wildcard syntax.
    """
    raw_host = str(
        flask_config.get("listen") or flask_config.get("host") or "127.0.0.1"
    ).strip()
    if raw_host not in {"*", "all"}:
        return raw_host

    family = str(flask_config.get("listen_family") or "ipv4").strip().lower()
    if family in {"ipv4", "4"}:
        return "0.0.0.0"
    if family in {"ipv6", "6", "dual", "both"}:
        return "::"
    raise ValueError("flask listen_family must be ipv4, ipv6, or dual")


def configured_listen_url_host(host: str) -> str:
    """
    Return display-safe host for startup URL.
    """
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


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
    logrotate_config = config.get("logrotate") or {}
    file_handler.rotate_enabled = config_bool(logrotate_config.get("enabled"), True)
    try:
        file_handler.retention_days = int(
            logrotate_config.get("retention_days", DEFAULT_LOG_RETENTION_DAYS)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("logrotate retention_days must be integer") from error
    if file_handler.retention_days < 0:
        raise ValueError("logrotate retention_days must be >= 0")

    requests_logger = logging.getLogger("urllib3")
    requests_logger.setLevel(log_level)
    logger.debug("Log level configured: %s", log_level_name)


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


def configured_client_ip_header(config: dict[str, Any]) -> str:
    """
    Return request header used as client IP source.
    """
    flask_config = config.get("flask") or {}
    return str(flask_config.get("client_ip_header") or DEFAULT_CLIENT_IP_HEADER).strip()


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


def parse_bool(value: Any) -> bool:
    """
    Parse truthy API values.
    """
    return str(value or "").lower() in {"1", "true", "yes"}


def request_client_ip() -> str:
    """
    Return best-effort client IP for logs.
    """
    header_name = configured_client_ip_header(CONFIG)
    header_value = request.headers.get(header_name, "") if header_name else ""
    if header_value:
        return header_value.split(",", 1)[0].strip()
    return request.remote_addr or "unknown"


def request_bearer_token() -> str:
    """
    Return Bearer token from current request without logging it.
    """
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def token_log_id(token: str) -> str:
    """
    Return stable non-secret token identifier for logs.
    """
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"token:{digest[:12]}"


def request_user_id() -> str:
    """
    Return best-effort authenticated user identifier for logs.
    """
    if not auth_enabled(CONFIG):
        return "auth-disabled"
    token = request_bearer_token()
    if not token:
        return "unknown"
    for configured_token in configured_auth_tokens(CONFIG):
        if hmac.compare_digest(token, configured_token):
            return token_log_id(token)
    return "unknown"


def log_evaluation_request(log_data: dict[str, Any]) -> None:
    """
    Log accepted evaluation request without exposing secrets.
    """
    logger.info(
        (
            "Evaluation request received: job_id=%s user_id=%s "
            "async=%s justify=%s summary=%s model=%s"
        ),
        log_data["job_id"],
        log_data["user_id"],
        log_data["async_requested"],
        log_data["justify"],
        log_data["summary_requested"],
        log_data["model_name"],
        extra={"client_ip": log_data["client_ip"]},
    )


def evaluation_log_data(
    async_requested: bool,
    justify: bool,
    summary_requested: bool,
    model: str | None,
) -> dict[str, Any]:
    """
    Return common evaluation request log fields.
    """
    return {
        "client_ip": request_client_ip(),
        "user_id": request_user_id(),
        "async_requested": async_requested,
        "justify": justify,
        "summary_requested": summary_requested,
        "model_name": model or ollama_client().engine(),
    }


def log_queued_result_retrieval(
    job_id: str, client_ip: str, user_id: str, status: str
) -> None:
    """
    Log one queued result retrieval request.
    """
    logger.info(
        "Queued result retrieval: job_id=%s user_id=%s status=%s",
        job_id,
        user_id,
        status,
        extra={"client_ip": client_ip},
    )


def auth_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Return auth config.
    """
    return config.get("auth") or {}


def auth_enabled(config: dict[str, Any]) -> bool:
    """
    Return true when Bearer token auth is required.
    """
    return config_bool(auth_config(config).get("enabled"), False)


def configured_auth_tokens(config: dict[str, Any]) -> list[str]:
    """
    Return configured Bearer tokens.
    """
    values = auth_config(config).get("tokens") or []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("auth tokens must be a list")
    return [str(token) for token in values if str(token)]


def validate_auth_config(config: dict[str, Any]) -> None:
    """
    Validate auth config and warn on fail-closed empty token setup.
    """
    tokens = configured_auth_tokens(config)
    if auth_enabled(config) and not tokens:
        logger.warning("Auth is enabled but no tokens are configured")


@TOKEN_AUTH.verify_token
def verify_auth_token(token: str) -> str | None:
    """
    Verify Bearer token using constant-time comparison.
    """
    if not auth_enabled(CONFIG):
        return "auth-disabled"
    for configured_token in configured_auth_tokens(CONFIG):
        if hmac.compare_digest(token, configured_token):
            return "api-token"
    return None


@TOKEN_AUTH.error_handler
def auth_error(status: int):
    """
    Return JSON auth errors.
    """
    return jsonify({"error": "authentication required"}), status


def require_auth(view_func):
    """
    Apply token auth when configured.
    """

    @functools.wraps(view_func)
    def wrapper(*args, **kwargs):
        if not auth_enabled(CONFIG):
            return view_func(*args, **kwargs)
        return TOKEN_AUTH.login_required(view_func)(*args, **kwargs)

    return wrapper


def config_bool(value: Any, default: bool) -> bool:
    """
    Parse optional config boolean.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def queue_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Return async queue config.
    """
    return config.get("queue") or {}


def queue_enabled(config: dict[str, Any]) -> bool:
    """
    Return true when async queue requests are allowed.
    """
    return config_bool(queue_config(config).get("enabled"), True)


def queue_allow_sync(config: dict[str, Any]) -> bool:
    """
    Return true when synchronous /evaluate requests are allowed.
    """
    return config_bool(queue_config(config).get("allow_sync"), True)


def queue_cache_path(config: dict[str, Any]) -> str:
    """
    Return async job cache directory.
    """
    value = queue_config(config).get("cache_path")
    if not value:
        return DEFAULT_JOB_CACHE_PATH
    return resolve_api_path(str(value))


def queue_cache_ttl_seconds(config: dict[str, Any]) -> int:
    """
    Return async job cache TTL in seconds.
    """
    value = queue_config(config).get("cache_ttl_hours", DEFAULT_JOB_CACHE_TTL_HOURS)
    try:
        ttl_hours = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("queue cache_ttl_hours must be integer") from error
    if ttl_hours <= 0:
        raise ValueError("queue cache_ttl_hours must be positive")
    return ttl_hours * 60 * 60


def queue_worker_count(config: dict[str, Any]) -> int:
    """
    Return async queue worker count.
    """
    value = queue_config(config).get("workers", DEFAULT_JOB_WORKERS)
    try:
        workers = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("queue workers must be integer") from error
    if workers <= 0:
        raise ValueError("queue workers must be positive")
    return workers


def job_path(job_id: str) -> str:
    """
    Return sharded path for one cached job file.
    """
    try:
        parsed_job_id = str(uuid.UUID(job_id))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid job id") from error
    return os.path.join(
        queue_cache_path(CONFIG),
        parsed_job_id[0],
        f"{parsed_job_id}.json",
    )


def now_seconds() -> float:
    """
    Return current timestamp seconds.
    """
    return time.time()


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    """
    Return public job status data.
    """
    output = {
        "id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    if job.get("error"):
        output["error"] = job["error"]
    return output


def write_job(job: dict[str, Any]) -> None:
    """
    Write async job atomically.
    """
    path = job_path(str(job["id"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with JOB_LOCK:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(job, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)


def read_job(job_id: str) -> dict[str, Any]:
    """
    Read async job from disk.
    """
    with open(job_path(job_id), "r", encoding="utf-8") as handle:
        job = json.load(handle)
    if not isinstance(job, dict) or job.get("id") != str(uuid.UUID(job_id)):
        raise ValueError("invalid job data")
    return job


def job_is_expired(job: dict[str, Any]) -> bool:
    """
    Return true when job is older than configured TTL.
    """
    created_at = float(job.get("created_at", 0))
    return now_seconds() - created_at > queue_cache_ttl_seconds(CONFIG)


def read_current_job(job_id: str) -> dict[str, Any]:
    """
    Read job and mark expired jobs.
    """
    job = read_job(job_id)
    if job.get("status") in {"done", "failed"} and job_is_expired(job):
        job["status"] = "expired"
        job["updated_at"] = now_seconds()
        write_job(job)
    return job


def enqueue_job_id(job_id: str) -> bool:
    """
    Put job id in memory queue once.
    """
    job_id = str(uuid.UUID(job_id))
    with JOB_LOCK:
        if job_id in QUEUED_JOB_IDS or job_id in RUNNING_JOB_IDS:
            return False
        QUEUED_JOB_IDS.add(job_id)
    JOB_QUEUE.put(job_id)
    return True


def enqueue_evaluation_job(
    markdown: str,
    model: str | None,
    timeout_override: int | None,
    justify: bool,
    summary: bool,
) -> dict[str, Any]:
    """
    Create async evaluation job and queue work.
    """
    job_id = str(uuid.uuid4())
    timestamp = now_seconds()
    job = {
        "id": job_id,
        "status": "queued",
        "created_at": timestamp,
        "updated_at": timestamp,
        "request": {
            "markdown": markdown,
            "model": model,
            "timeout": timeout_override,
            "justify": justify,
            "summary": summary,
        },
    }
    write_job(job)
    enqueue_job_id(job_id)
    return job


def process_evaluation_job(job_id: str) -> None:
    """
    Run one queued evaluation job.
    """
    job_id = str(uuid.UUID(job_id))
    with JOB_LOCK:
        QUEUED_JOB_IDS.discard(job_id)
        RUNNING_JOB_IDS.add(job_id)
    try:
        job = read_job(job_id)
        if job_is_expired(job):
            job["status"] = "expired"
            job["updated_at"] = now_seconds()
            write_job(job)
            return

        job["status"] = "running"
        job["updated_at"] = now_seconds()
        write_job(job)

        request_data = job.get("request") or {}
        JOB_LOG_CONTEXT.job_id = job_id
        result = evaluate_markdown(
            request_data.get("markdown", ""),
            request_data.get("model"),
            request_data.get("timeout"),
            parse_bool(request_data.get("justify")),
            parse_bool(request_data.get("summary")),
        )
        job["status"] = "done"
        job["result"] = result
        job["updated_at"] = now_seconds()
        write_job(job)
    except json.JSONDecodeError as error:
        store_failed_job(job_id, "ollama evaluation returned invalid json", error)
    except ValueError as error:
        store_failed_job(job_id, str(error), error)
    except requests.RequestException as error:
        store_failed_job(job_id, "ollama evaluation failed", error)
    finally:
        if hasattr(JOB_LOG_CONTEXT, "job_id"):
            del JOB_LOG_CONTEXT.job_id
        with JOB_LOCK:
            RUNNING_JOB_IDS.discard(job_id)


def store_failed_job(job_id: str, public_error: str, error: Exception) -> None:
    """
    Store failed async job state.
    """
    logger.error("Async evaluation job failed: id=%s error=%s", job_id, error)
    try:
        job = read_job(job_id)
    except (OSError, ValueError, json.JSONDecodeError):
        timestamp = now_seconds()
        job = {
            "id": job_id,
            "status": "failed",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
    job["status"] = "failed"
    job["error"] = public_error
    job["updated_at"] = now_seconds()
    write_job(job)


def evaluation_worker() -> None:
    """
    Background worker for async evaluation jobs.
    """
    while True:
        job_id = JOB_QUEUE.get()
        try:
            process_evaluation_job(job_id)
        finally:
            JOB_QUEUE.task_done()


def queue_scheduler_context() -> scheduler_lib.QueueSchedulerContext:
    """
    Return queue scheduler dependencies.
    """
    return scheduler_lib.QueueSchedulerContext(
        config=CONFIG,
        logger=logger,
        queue_cache_path=queue_cache_path,
        queue_cache_ttl_seconds=queue_cache_ttl_seconds,
        now_seconds=now_seconds,
        write_job=write_job,
        enqueue_job_id=enqueue_job_id,
        running_job_ids=RUNNING_JOB_IDS,
        job_lock=JOB_LOCK,
    )


def start_queue_scheduler() -> None:
    """
    Start queue maintenance scheduler.
    """
    scheduler_lib.start_queue_scheduler(
        enabled=queue_enabled(CONFIG),
        interval_seconds=scheduler_lib.queue_scheduler_interval_seconds(CONFIG),
        tick=partial(scheduler_lib.queue_scheduler_tick, queue_scheduler_context()),
        logger=logger,
    )


def start_health_scheduler() -> None:
    """
    Start cached health probe scheduler.
    """
    scheduler_lib.start_health_scheduler(
        interval_seconds=health_lib.health_scheduler_interval_seconds(CONFIG),
        tick=partial(
            health_lib.health_probe_tick,
            CONFIG,
            ollama_client().health,
            logger,
        ),
        logger=logger,
    )


def start_job_workers() -> None:
    """
    Start async queue worker threads once.
    """
    global JOB_WORKERS_STARTED
    if JOB_WORKERS_STARTED or not queue_enabled(CONFIG):
        return
    os.makedirs(queue_cache_path(CONFIG), exist_ok=True)
    start_queue_scheduler()
    for index in range(queue_worker_count(CONFIG)):
        worker = threading.Thread(
            target=evaluation_worker,
            name=f"classification-worker-{index + 1}",
            daemon=True,
        )
        worker.start()
    JOB_WORKERS_STARTED = True


def ollama_client() -> ollama.OllamaClient:
    """
    Return configured Ollama client.
    """
    return ollama.OllamaClient(CONFIG, MAX_TIMEOUT_SECONDS, logger)


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
                uid_values = uids.parse_uuid_output(str(key))
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
        uid_values = uids.parse_uuid_output(uid_text)
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
    summary: bool = False,
) -> dict[str, Any]:
    """
    Evaluate Markdown against content-classification taxonomy and return JSON data.
    """
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError("data must be non-empty markdown text")

    job_id = getattr(JOB_LOG_CONTEXT, "job_id", "sync")
    client = ollama_client()
    model_name = model or client.engine()
    if not model_name:
        raise ValueError("missing model")
    client.ensure_model_allowed(model_name)

    user_markdown, input_tokens, input_truncated = truncate_user_markdown(markdown)
    evaluation_query = TAXONOMY_EVALUATION_QUERY.format(
        taxonomy_tags="\n".join(
            f"- {entry}" for entry in content_classification_prompt_entries()
        )
    )
    evaluation_prompt = f"{evaluation_query}\n\n# Input\n{user_markdown}"

    logger.info("Evaluating Markdown: job_id=%s model=%s", job_id, model_name)
    session = requests.Session()
    start_time = time.perf_counter()
    evaluation_response = client.generate(
        session,
        evaluation_prompt,
        model=model_name,
        timeout_override=timeout_override,
        validate_model=False,
    )
    raw_output = str(evaluation_response.get("response", "")).strip()
    summary_context = evaluation_response.get("context")
    if not isinstance(summary_context, list) or not all(
        isinstance(item, int) for item in summary_context
    ):
        summary_context = None
    by_uid = {tag["uid"]: tag for tag in get_content_classification_tags()}
    known_uids = set(by_uid)
    selected_uids = uids.parse_uuid_output(raw_output)
    resolved_uids, _unknown_uids, _corrected_uids = uids.resolve_uids(
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
            logger.info(
                "Justifying taxonomy labels: job_id=%s model=%s", job_id, model_name
            )
            justification_response = client.generate(
                session,
                justification_prompt,
                model=model_name,
                timeout_override=timeout_override,
                validate_model=False,
            )
            justification_raw_output = str(
                justification_response.get("response", "")
            ).strip()
            justification_context = justification_response.get("context")
            if isinstance(justification_context, list) and all(
                isinstance(item, int) for item in justification_context
            ):
                summary_context = justification_context
            raw_outputs = {
                "evaluation": raw_output,
                "justification": justification_raw_output,
            }
        else:
            justification_raw_output = "[]"

        selected_uid_set = set(resolved_uids)
        seen = set()
        for item in parse_justified_output(justification_raw_output):
            corrected_uid, _corrected = uids.correct_uuid(item["uid"], selected_uid_set)
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

    summary_text = None
    if summary:
        summary_prompt = f"{SUMMARY_QUERY}\n\n# Input\n{user_markdown}"
        logger.info("Summarizing Markdown: job_id=%s model=%s", job_id, model_name)
        summary_response = client.generate(
            session,
            summary_prompt,
            model=model_name,
            timeout_override=timeout_override,
            validate_model=False,
            context=summary_context,
        )
        summary_text = str(summary_response.get("response", "")).strip()
        if isinstance(raw_outputs, dict):
            raw_outputs["summary"] = summary_text
        else:
            raw_outputs = {
                "evaluation": raw_output,
                "summary": summary_text,
            }

    processing_time_seconds = time.perf_counter() - start_time
    result = {
        "labels": labels,
        "justify": justify,
        "processing_time_seconds": round(processing_time_seconds, 3),
        "truncated": input_truncated,
        "input_tokens": input_tokens,
        "input_truncated": input_truncated,
        "input_token_limit": USER_MARKDOWN_TOKEN_LIMIT,
        "raw_output": raw_outputs,
    }
    if summary:
        result["summary"] = summary_text or ""
    return result


def warmup_model(
    model: str | None = None,
    timeout_override: int | None = None,
) -> None:
    """
    Send a tiny prompt to load a model before timed requests.
    """
    client = ollama_client()
    model_name = model or client.engine()
    if not model_name:
        raise ValueError("missing model")
    client.ensure_model_allowed(model_name)

    logger.info("Warming up model: %s", model_name)
    session = requests.Session()
    client.generate(
        session,
        "say 'hello world'",
        model=model_name,
        timeout_override=timeout_override,
        validate_model=False,
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
    validate_auth_config(CONFIG)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = configured_max_body_bytes(CONFIG)
    start_health_scheduler()
    start_job_workers()

    @app.errorhandler(RequestEntityTooLarge)
    def request_entity_too_large(_error):
        return jsonify({"error": "request body too large"}), 413

    @app.get("/")
    def index_route():
        return send_from_directory(os.path.join(THIS_DIR, "static"), "index.html")

    @app.get("/health")
    def health_route():
        state = health_lib.health_state()
        status_code = 200 if state.get("status") == "ok" else 500
        return jsonify(state), status_code

    @app.get("/getmodels")
    @require_auth
    def getmodels():
        capability = request.args.get("capability")
        try:
            MODEL_CACHE[:] = ollama_client().list_models()
        except requests.RequestException as error:
            logger.error("Ollama model listing failed: %s", error)
            return jsonify({"error": "ollama model listing failed"}), 502
        return jsonify(cached_model_contexts(capability))

    @app.post("/evaluate")
    @require_auth
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
            summary_requested = (
                payload["summary"]
                if "summary" in payload
                else request.args.get("summary")
            )
            async_requested = (
                payload["async"] if "async" in payload else request.args.get("async")
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
            summary_requested = request.args.get("summary")
            async_requested = request.args.get("async")
            timeout_value = request.args.get("timeout")

        include_raw = parse_bool(include_raw)
        justify = parse_bool(justify)
        summary_requested = parse_bool(summary_requested)
        async_requested = parse_bool(async_requested)
        if not isinstance(markdown, str) or not markdown.strip():
            return jsonify({"error": "data must be non-empty markdown text"}), 400
        if model is not None and not isinstance(model, str):
            return jsonify({"error": "model must be string"}), 400
        log_data = evaluation_log_data(
            async_requested, justify, summary_requested, model
        )

        try:
            timeout_override = parse_timeout(timeout_value)
            if async_requested:
                if not queue_enabled(CONFIG):
                    return jsonify({"error": "async queue is disabled"}), 400
                job = enqueue_evaluation_job(
                    markdown,
                    model,
                    timeout_override,
                    justify,
                    summary_requested,
                )
                log_evaluation_request({**log_data, "job_id": job["id"]})
                return jsonify(public_job(job)), 202
            if not queue_allow_sync(CONFIG):
                return jsonify({"error": "synchronous evaluation is disabled"}), 400
            log_evaluation_request({**log_data, "job_id": "sync"})
            result = evaluate_markdown(
                markdown,
                model,
                timeout_override,
                justify,
                summary_requested,
            )
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

    @app.get("/evaluate/<job_id>/status")
    @require_auth
    def evaluate_status_route(job_id: str):
        try:
            job = read_current_job(job_id)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            return jsonify({"error": "job not found"}), 404
        return jsonify(public_job(job))

    @app.get("/evaluate/<job_id>/result")
    @require_auth
    def evaluate_result_route(job_id: str):
        client_ip = request_client_ip()
        user_id = request_user_id()
        include_raw = parse_bool(request.args.get("include_raw"))
        try:
            job = read_current_job(job_id)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            log_queued_result_retrieval(job_id, client_ip, user_id, "not_found")
            return jsonify({"error": "job not found"}), 404

        status = job.get("status")
        log_queued_result_retrieval(job_id, client_ip, user_id, str(status))
        if status in {"queued", "running"}:
            return jsonify(public_job(job)), 202
        if status == "expired":
            return jsonify({"error": "job expired"}), 404
        if status == "failed":
            return jsonify({"error": job.get("error") or "job failed"}), 500
        if status != "done":
            return jsonify({"error": "invalid job status"}), 500

        result = job.get("result")
        if not isinstance(result, dict):
            return jsonify({"error": "invalid job result"}), 500
        result = dict(result)
        if not include_raw:
            result.pop("raw_output", None)
        result.pop("corrected_uids", None)
        return jsonify(result)

    @app.post("/warmup_model")
    @require_auth
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
        MODEL_CACHE[:] = ollama_client().list_models()
    except ValueError as error:
        logger.error("Invalid config: %s", error)
        return 1
    except requests.RequestException as error:
        logger.error("Ollama connection failed: %s", error)
        return 1

    flask_config = CONFIG.get("flask") or {}
    host = configured_listen_host(flask_config)
    port = int(flask_config.get("port") or 5151)
    logger.info("Starting API: http://%s:%s", configured_listen_url_host(host), port)
    try:
        app = create_app()
    except ValueError as error:
        logger.error("Invalid config: %s", error)
        return 1
    app.run(host=host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
