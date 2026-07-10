#!/usr/bin/env python3
# coding=utf-8

"""
Cached health probe state and Ollama health checks.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

import requests

DEFAULT_HEALTH_SCHEDULER_INTERVAL_SECONDS = 30
DEFAULT_HEALTH_TIMEOUT_SECONDS = 5

HEALTH_LOCK = threading.Lock()
HEALTH_STATE: dict[str, Any] = {
    "status": "error",
    "ai_engine": "unknown",
    "error": "health probe has not run yet",
    "last_check": 0,
}
LAST_HEALTH_LOG_STATE = ""


def health_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Return health probe config.
    """
    return config.get("health") or {}


def health_scheduler_interval_seconds(config: dict[str, Any]) -> int:
    """
    Return health probe scheduler interval.
    """
    value = health_config(config).get(
        "scheduler_interval_seconds", DEFAULT_HEALTH_SCHEDULER_INTERVAL_SECONDS
    )
    try:
        interval = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("health scheduler_interval_seconds must be integer") from error
    if interval <= 0:
        raise ValueError("health scheduler_interval_seconds must be positive")
    return interval


def health_timeout_seconds(config: dict[str, Any]) -> int:
    """
    Return health probe timeout.
    """
    value = health_config(config).get("timeout_seconds", DEFAULT_HEALTH_TIMEOUT_SECONDS)
    try:
        timeout = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("health timeout_seconds must be integer") from error
    if timeout <= 0:
        raise ValueError("health timeout_seconds must be positive")
    return timeout


def health_state() -> dict[str, Any]:
    """
    Return copy of current health state.
    """
    with HEALTH_LOCK:
        return dict(HEALTH_STATE)


def set_health_state(
    status: str, ai_engine: str, logger, error: str | None = None
) -> None:
    """
    Store and log health state changes.
    """
    global LAST_HEALTH_LOG_STATE  # pylint: disable=global-statement
    state = {
        "status": status,
        "ai_engine": ai_engine,
        "last_check": int(time.time()),
    }
    if error:
        state["error"] = error

    log_key = json.dumps(
        {"status": status, "ai_engine": ai_engine, "error": error or ""},
        sort_keys=True,
    )
    with HEALTH_LOCK:
        HEALTH_STATE.clear()
        HEALTH_STATE.update(state)
        should_log = log_key != LAST_HEALTH_LOG_STATE
        if should_log:
            LAST_HEALTH_LOG_STATE = log_key

    if should_log:
        if status == "ok":
            logger.info("Health probe recovered: ai_engine=ok")
        else:
            logger.warning(
                "Health probe failed: ai_engine=%s error=%s", ai_engine, error
            )


def health_probe_tick(
    config: dict[str, Any],
    health_check: Callable[[int], dict[str, Any]],
    logger,
) -> dict[str, Any]:
    """
    Probe Ollama and cache health state.
    """
    timeout = health_timeout_seconds(config)
    try:
        data = health_check(timeout)
        if not isinstance(data, dict):
            raise ValueError("invalid ollama health response")
    except (ValueError, requests.RequestException) as error:
        set_health_state("error", "error", logger, str(error))
        return health_state()

    set_health_state("ok", "ok", logger)
    return health_state()
