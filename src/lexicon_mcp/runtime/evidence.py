"""Canonical JSON serialization for release-acceptance evidence."""

from __future__ import annotations

import json


def compact_evidence_json(value: object) -> str:
    """Serialize evidence deterministically, compactly, and without non-finite floats."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
