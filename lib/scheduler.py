#!/usr/bin/env python3
# coding=utf-8
# pylint: disable=too-many-instance-attributes

"""
Schedulers and queued job cache maintenance.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from apscheduler.schedulers.background import BackgroundScheduler

DEFAULT_JOB_STALE_TTL_HOURS = 24
DEFAULT_QUEUE_SCHEDULER_INTERVAL_SECONDS = 30

APP_SCHEDULER: BackgroundScheduler | None = None
APP_SCHEDULER_LOCK = threading.Lock()


@dataclass(slots=True)  # pylint: disable=too-many-instance-attributes
class QueueSchedulerContext:
    """
    Dependencies needed for queue maintenance.
    """

    config: dict[str, Any]
    logger: Any
    queue_cache_path: Callable[[dict[str, Any]], str]
    queue_cache_ttl_seconds: Callable[[dict[str, Any]], int]
    now_seconds: Callable[[], float]
    write_job: Callable[[dict[str, Any]], None]
    enqueue_job_id: Callable[[str], bool]
    running_job_ids: set[str]
    job_lock: Any


def queue_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Return queue config.
    """
    return config.get("queue") or {}


def queue_stale_job_ttl_seconds(config: dict[str, Any]) -> int:
    """
    Return stale queued/running job TTL in seconds.
    """
    value = queue_config(config).get("stale_job_ttl_hours", DEFAULT_JOB_STALE_TTL_HOURS)
    try:
        ttl_hours = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("queue stale_job_ttl_hours must be integer") from error
    if ttl_hours <= 0:
        raise ValueError("queue stale_job_ttl_hours must be positive")
    return ttl_hours * 60 * 60


def queue_scheduler_interval_seconds(config: dict[str, Any]) -> int:
    """
    Return queue scheduler interval in seconds.
    """
    value = queue_config(config).get(
        "scheduler_interval_seconds", DEFAULT_QUEUE_SCHEDULER_INTERVAL_SECONDS
    )
    try:
        interval = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("queue scheduler_interval_seconds must be integer") from error
    if interval <= 0:
        raise ValueError("queue scheduler_interval_seconds must be positive")
    return interval


def iter_job_paths(context: QueueSchedulerContext) -> list[str]:
    """
    Return cached job JSON file paths.
    """
    directory = context.queue_cache_path(context.config)
    try:
        filenames = os.listdir(directory)
    except FileNotFoundError:
        return []

    paths = []
    for filename in filenames:
        if not filename.endswith(".json"):
            continue
        paths.append(os.path.join(directory, filename))
    return paths


def read_job_path(context: QueueSchedulerContext, path: str) -> dict[str, Any] | None:
    """
    Read and validate job JSON by path.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            job = json.load(handle)
        if not isinstance(job, dict) or not job.get("id"):
            return None
        uuid.UUID(str(job["id"]))
        return job
    except (OSError, ValueError, json.JSONDecodeError):
        context.logger.warning("Queue scheduler ignored invalid job file: %s", path)
        return None


def job_age_seconds(context: QueueSchedulerContext, job: dict[str, Any]) -> float:
    """
    Return job age in seconds.
    """
    return context.now_seconds() - float(job.get("created_at", 0))


def queue_scheduler_tick(context: QueueSchedulerContext) -> dict[str, int]:
    """
    Cleanup stale jobs and requeue persisted queued jobs.
    """
    summary = {
        "deleted": 0,
        "requeued": 0,
        "stale_failed": 0,
        "invalid": 0,
    }
    cache_ttl = context.queue_cache_ttl_seconds(context.config)
    stale_ttl = queue_stale_job_ttl_seconds(context.config)

    for path in iter_job_paths(context):
        job = read_job_path(context, path)
        if job is None:
            summary["invalid"] += 1
            continue

        status = str(job.get("status") or "")
        age = job_age_seconds(context, job)
        job_id = str(job["id"])

        if status in {"done", "failed", "expired"} and age > cache_ttl:
            try:
                os.remove(path)
                summary["deleted"] += 1
                context.logger.info(
                    "Queue scheduler deleted expired job: id=%s", job_id
                )
            except FileNotFoundError:
                continue
            except OSError as error:
                context.logger.error(
                    "Queue scheduler failed to delete job %s: %s", job_id, error
                )
            continue

        if status == "queued":
            if age > stale_ttl:
                job["status"] = "failed"
                job["error"] = "queued job expired before dispatch"
                job["updated_at"] = context.now_seconds()
                context.write_job(job)
                summary["stale_failed"] += 1
                context.logger.info(
                    "Queue scheduler marked stale queued job failed: id=%s", job_id
                )
                continue
            if context.enqueue_job_id(job_id):
                summary["requeued"] += 1
                context.logger.info(
                    "Queue scheduler requeued persisted job: id=%s", job_id
                )
            continue

        if status == "running" and age > stale_ttl:
            with context.job_lock:
                is_running_here = job_id in context.running_job_ids
            if is_running_here:
                continue
            job["status"] = "failed"
            job["error"] = "running job stale after server restart"
            job["updated_at"] = context.now_seconds()
            context.write_job(job)
            summary["stale_failed"] += 1
            context.logger.info(
                "Queue scheduler marked stale running job failed: id=%s", job_id
            )

    if any(summary.values()):
        context.logger.info(
            "Queue scheduler tick: deleted=%s requeued=%s stale_failed=%s invalid=%s",
            summary["deleted"],
            summary["requeued"],
            summary["stale_failed"],
            summary["invalid"],
        )
    return summary


def ensure_app_scheduler() -> BackgroundScheduler:
    """
    Return single process-wide APScheduler instance.
    """
    global APP_SCHEDULER  # pylint: disable=global-statement
    with APP_SCHEDULER_LOCK:
        if APP_SCHEDULER is not None and APP_SCHEDULER.running:
            return APP_SCHEDULER
        scheduler = BackgroundScheduler()
        scheduler.start()
        APP_SCHEDULER = scheduler
        return scheduler


def start_queue_scheduler(
    enabled: bool,
    interval_seconds: int,
    tick: Callable[[], dict[str, int]],
    logger,
) -> None:
    """
    Start APScheduler job for queue maintenance.
    """
    if not enabled:
        return
    scheduler = ensure_app_scheduler()
    if scheduler.get_job("queue_scheduler") is not None:
        return
    scheduler.add_job(
        func=tick,
        trigger="interval",
        seconds=interval_seconds,
        max_instances=1,
        id="queue_scheduler",
        replace_existing=True,
    )
    logger.info("Queue scheduler started: interval_seconds=%s", interval_seconds)


def start_health_scheduler(
    interval_seconds: int, tick: Callable[[], dict[str, Any]], logger
) -> None:
    """
    Start APScheduler job for cached health probing.
    """
    scheduler = ensure_app_scheduler()
    if scheduler.get_job("health_probe") is not None:
        return
    scheduler.add_job(
        func=tick,
        trigger="interval",
        seconds=interval_seconds,
        max_instances=1,
        id="health_probe",
        next_run_time=datetime.datetime.now(),
        replace_existing=True,
    )
    logger.info("Health scheduler started: interval_seconds=%s", interval_seconds)
