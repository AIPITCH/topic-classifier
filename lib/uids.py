#!/usr/bin/env python3
# coding=utf-8

"""
UUID extraction and correction helpers.
"""

from __future__ import annotations

import re

UUID_RE = re.compile(
    r"\b[0-9a-zA-Z]{8}-[0-9a-zA-Z]{4}-[0-9a-zA-Z]{4}-"
    r"[0-9a-zA-Z]{4}-[0-9a-zA-Z]{12}\b"
)


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
