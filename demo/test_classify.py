#!/usr/bin/env python3
# coding=utf-8

"""
Minimal example for the local evaluation API.

Start the API first:
    python3 classification_server.py

Run:
    python3 demo/test_classify.py
    python3 demo/test_classify.py --model gemma4:31b
    python3 demo/test_classify.py --model gemma4:31b --warmup
    python3 demo/test_classify.py --justify
    python3 demo/test_classify.py --list-model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

# pylint: disable=wrong-import-position
from client.classifier import ClassificationClient, ClassificationError
from classification_server import json_to_markdown

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_CHANNEL_PATH = ROOT_DIR / "test_data" / "test_sample_channel.json"


def load_sample_channel(path: Path = SAMPLE_CHANNEL_PATH) -> dict[str, Any]:
    """
    Load test sample_channel JSON.
    """
    with path.open("r", encoding="utf-8") as handle:
        sample_channel = json.load(handle)
    if not isinstance(sample_channel, dict):
        raise ValueError("sample_channel JSON must be an object")
    return sample_channel


def load_sample_markdown(path: Path = SAMPLE_CHANNEL_PATH) -> str:
    """
    Load test sample_channel JSON and convert it to Markdown for /evaluate.
    """
    return json_to_markdown(load_sample_channel(path), title="sample_channel")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Minimal example for the local evaluation API."
    )
    parser.add_argument(
        "--model",
        help="Optional Ollama model override.",
    )
    parser.add_argument(
        "--justify",
        action="store_true",
        help="Ask /evaluate for justified taxonomy matches.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm selected model before /evaluate.",
    )
    parser.add_argument(
        "--list-model",
        action="store_true",
        help="List available Ollama models through the API and exit.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Run demo command.
    """
    args = parse_args()
    model = args.model
    client = ClassificationClient.from_config()

    try:
        if args.list_model:
            models = client.get_models()
            print("Available models:")
            for name in models:
                print(f"- {name}")
            return 0

        if args.warmup:
            client.warmup_model(model=model)
            print("Model warmed.")

        sample_channel = load_sample_channel()
        sample_markdown = json_to_markdown(sample_channel, title="sample_channel")

        result = client.evaluate_markdown(
            sample_markdown,
            model=model,
            justify=args.justify,
        )
    except ClassificationError as error:
        print(f"API error: {error}", file=sys.stderr)
        return 1

    print("\nEvaluation result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
