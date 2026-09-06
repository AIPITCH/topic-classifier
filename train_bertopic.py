#!/usr/bin/env python3
"""Train and quality-gate a BERTopic model from topic-classifier JSONL data."""

from __future__ import annotations

import argparse
import importlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_examples(path: Path) -> list[dict]:
    """Read valid labelled examples from a JSON Lines dataset."""
    examples = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("text"), str) or not isinstance(
                item.get("labels"), list
            ):
                raise ValueError(f"invalid training record on line {number}")
            examples.append(item)
    return examples


def f1_score(expected: list[set[str]], predicted: list[set[str]]) -> float:
    """Calculate micro F1 agreement for multi-label results."""
    true_positive = sum(len(left & right) for left, right in zip(expected, predicted))
    false_positive = sum(len(right - left) for left, right in zip(expected, predicted))
    false_negative = sum(len(left - right) for left, right in zip(expected, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return (2 * true_positive / denominator) if denominator else 1.0


def topic_label_map(
    topics: list[int], examples: list[dict], minimum_support: float
) -> dict[str, list[str]]:
    """Map each discovered topic to teacher labels meeting vote support."""
    totals = Counter(topics)
    votes: dict[int, Counter] = defaultdict(Counter)
    for topic, example in zip(topics, examples):
        if topic != -1:
            votes[topic].update(set(map(str, example["labels"])))
    return {
        str(topic): sorted(
            label
            for label, count in counts.items()
            if count / totals[topic] >= minimum_support
        )
        for topic, counts in votes.items()
    }


def main() -> int:
    """Train on a split, benchmark against held-out LLM labels, and save."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSONL produced by topic-classifier")
    parser.add_argument("output", type=Path, help="BERTopic artifact directory")
    parser.add_argument(
        "--threshold", type=float, default=0.85, help="minimum held-out micro-F1"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--minimum-support", type=float, default=0.5)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers/BERT embedding model name or local path",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        not 0 < args.test_size < 1
        or not 0 <= args.threshold <= 1
        or not 0 < args.minimum_support <= 1
    ):
        parser.error("test-size, threshold, and minimum-support must be proportions")
    try:
        bertopic_module = importlib.import_module("bertopic")
    except ModuleNotFoundError as error:
        parser.error(f"BERTopic is required; install requirements-ml.txt ({error})")

    examples = read_examples(args.dataset)
    if len(examples) < 5:
        parser.error("at least five training examples are required")
    random.Random(args.seed).shuffle(examples)
    split = max(1, min(len(examples) - 1, round(len(examples) * (1 - args.test_size))))
    training, test = examples[:split], examples[split:]
    model = bertopic_module.BERTopic(
        embedding_model=args.embedding_model,
        calculate_probabilities=True,
    )
    topics, _ = model.fit_transform([item["text"] for item in training])
    mapping = topic_label_map(topics, training, args.minimum_support)
    test_topics, _ = model.transform([item["text"] for item in test])
    predicted = [set(mapping.get(str(topic), [])) for topic in test_topics]
    score = f1_score([set(map(str, item["labels"])) for item in test], predicted)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(
        str(args.output),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=args.embedding_model,
    )
    manifest = {
        "format_version": 1,
        "metric": "micro_f1",
        "score": score,
        "threshold": args.threshold,
        "eligible": score >= args.threshold,
        "training_examples": len(training),
        "test_examples": len(test),
        "topic_labels": mapping,
        "embedding_model": args.embedding_model,
    }
    with (args.output / "topic-classifier.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
