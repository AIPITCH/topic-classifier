#!/usr/bin/env python3
# coding=utf-8
# pylint: disable=duplicate-code

"""
Independent queued evaluation API example.

Start the API first:
    python3 classification_server.py

Run:
    python3 demo/test_classify_queue.py
    python3 demo/test_classify_queue.py --justify
    python3 demo/test_classify_queue.py --model gemma4:31b
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_CHANNEL_PATH = ROOT_DIR / "test_data" / "test_sample_channel.json"
DEFAULT_REQUEST_TIMEOUT = 120


def auth_headers(args: argparse.Namespace) -> dict[str, str]:
    """
    Return optional Bearer token auth headers.
    """
    if not args.token:
        return {}
    return {"Authorization": f"Bearer {args.token}"}


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


def load_sample_markdown(path: Path = SAMPLE_CHANNEL_PATH) -> str:
    """
    Load sample channel JSON and convert it to Markdown.
    """
    with path.open("r", encoding="utf-8") as handle:
        sample_channel = json.load(handle)
    return json_to_markdown(sample_channel, title="sample_channel")


def raise_for_api_error(response: requests.Response) -> None:
    """
    Raise RuntimeError with API error payload when available.
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
    raise RuntimeError(f"{response.status_code}: {message}")


def submit_job(args: argparse.Namespace, markdown: str) -> str:
    """
    Submit queued evaluation and return job id.
    """
    payload: dict[str, Any] = {
        "data": markdown,
        "async": True,
        "justify": args.justify,
        "resume": args.resume,
    }
    if args.model:
        payload["model"] = args.model

    response = requests.post(
        f"{args.api_base.rstrip('/')}/evaluate",
        params={"async": "true"},
        json=payload,
        headers=auth_headers(args),
        timeout=args.request_timeout,
    )
    raise_for_api_error(response)
    data = response.json()
    job_id = data.get("id")
    if not isinstance(job_id, str):
        raise RuntimeError("queued response did not include job id")
    return job_id


def submit_job_with_progress(args: argparse.Namespace, markdown: str) -> str:
    """
    Submit queued evaluation while printing wait output until job id arrives.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(submit_job, args, markdown)
        while True:
            try:
                return future.result(timeout=args.poll_interval)
            except concurrent.futures.TimeoutError:
                print("Waiting for queued job UID...", flush=True)


def wait_for_job(args: argparse.Namespace, job_id: str) -> dict[str, Any]:
    """
    Poll status route until job reaches terminal state.
    """
    deadline = time.monotonic() + args.wait_timeout
    while time.monotonic() < deadline:
        response = requests.get(
            f"{args.api_base.rstrip('/')}/evaluate/{job_id}/status",
            headers=auth_headers(args),
            timeout=args.request_timeout,
        )
        raise_for_api_error(response)
        status = response.json()
        state = status.get("status")
        print(f"Poll job UID {job_id}: {state}", flush=True)
        if state in {"done", "failed", "expired"}:
            return status
        time.sleep(args.poll_interval)
    raise TimeoutError(f"job did not finish within {args.wait_timeout} seconds")


def fetch_result(args: argparse.Namespace, job_id: str) -> dict[str, Any]:
    """
    Fetch completed queued evaluation result.
    """
    response = requests.get(
        f"{args.api_base.rstrip('/')}/evaluate/{job_id}/result",
        params={"include_raw": "true" if args.include_raw else "false"},
        headers=auth_headers(args),
        timeout=args.request_timeout,
    )
    raise_for_api_error(response)
    return response.json()


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Submit a queued evaluation request and poll for result."
    )
    parser.add_argument(
        "--url",
        help="Classifier API base URL.",
    )
    parser.add_argument(
        "--api-base",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model",
        help="Optional Ollama model override.",
    )
    parser.add_argument(
        "--justify",
        action="store_true",
        help="Ask for justified taxonomy matches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ask for a summary in the final result.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw model output in final result.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=15.0,
        help="Seconds between status checks.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=900,
        help="Maximum seconds to wait for the queued job.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP timeout for submit/status/result requests.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CCE_API_TOKEN"),
        help="Bearer token when API auth is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Run queued API demo.
    """
    args = parse_args()
    args.api_base = args.url or args.api_base or "http://127.0.0.1:5151"
    markdown = load_sample_markdown()

    try:
        print("Submitting queued evaluation request...", flush=True)
        job_id = submit_job_with_progress(args, markdown)
        print(f"Queued job UID: {job_id}", flush=True)
        status = wait_for_job(args, job_id)
        if status.get("status") != "done":
            print(json.dumps(status, indent=2, ensure_ascii=False), file=sys.stderr)
            return 1

        result = fetch_result(args, job_id)
    except requests.Timeout as error:
        print(
            "API error: timed out before queued job UID was received. "
            "Restart classification_server.py so the async queue route is active, "
            "or increase --request-timeout.",
            file=sys.stderr,
        )
        print(f"Details: {error}", file=sys.stderr)
        return 1
    except (RuntimeError, TimeoutError, requests.RequestException) as error:
        print(f"API error: {error}", file=sys.stderr)
        return 1

    print("\nEvaluation result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
