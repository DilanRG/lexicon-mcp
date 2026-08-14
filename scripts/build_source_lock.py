"""Compute, update, or verify logical-row metadata in sources.lock.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lexicon_mcp.pipeline.common import file_sha256, write_json_atomic
from lexicon_mcp.pipeline.source_rows import measure_source


def _source_argument(value: str) -> tuple[str, Path]:
    identifier, separator, raw_path = value.partition("=")
    if not separator or not identifier or not raw_path:
        raise argparse.ArgumentTypeError("source must use ID=PATH")
    return identifier, Path(raw_path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("mode", choices=("compute", "update", "verify"))
    value.add_argument("--lock", type=Path, help="schema-v1 sources.lock.json")
    value.add_argument("--output", type=Path, help="also write measured metadata as JSON")
    value.add_argument(
        "--source",
        action="append",
        type=_source_argument,
        required=True,
        metavar="ID=PATH",
        help="repeat once for every source record being measured",
    )
    return value


def _load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("source lock must be a schema-version 1 object")
    if not isinstance(value.get("sources"), list):
        raise ValueError("source lock must contain a sources list")
    return {str(key): item for key, item in value.items()}


def main() -> None:
    args = parser().parse_args()
    sources = dict(args.source)
    if len(sources) != len(args.source):
        raise ValueError("source IDs must be unique")
    measured = {
        identifier: measure_source(identifier, path)
        for identifier, path in sorted(sources.items())
    }
    payload = {
        identifier: result.lock_fields() for identifier, result in measured.items()
    }
    if args.mode == "compute":
        if args.output is not None:
            write_json_atomic(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.lock is None:
        raise ValueError("--lock is required for update and verify")
    lock = _load_lock(args.lock)
    records = lock["sources"]
    by_id: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        if not isinstance(identifier, str):
            raise ValueError("every source lock record requires a string ID")
        by_id[identifier] = item
    if set(measured) != set(by_id):
        raise ValueError(
            f"source IDs mismatch; measured={sorted(measured)}, lock={sorted(by_id)}"
        )
    failures: list[str] = []
    for identifier, result in measured.items():
        record = by_id[identifier]
        fields = result.lock_fields()
        if args.mode == "verify":
            path = sources[identifier]
            if record.get("size") != path.stat().st_size:
                failures.append(
                    f"{identifier}.size: expected {record.get('size')!r}, "
                    f"observed {path.stat().st_size!r}"
                )
            observed_sha256 = file_sha256(path)
            if record.get("sha256") != observed_sha256:
                failures.append(
                    f"{identifier}.sha256: expected {record.get('sha256')!r}, "
                    f"observed {observed_sha256!r}"
                )
            for key, observed in fields.items():
                if record.get(key) != observed:
                    failures.append(
                        f"{identifier}.{key}: expected {record.get(key)!r}, observed {observed!r}"
                    )
        else:
            record.update(fields)
    if failures:
        raise ValueError("source lock row metadata mismatch: " + "; ".join(failures))
    if args.mode == "update":
        write_json_atomic(args.lock, lock)
    if args.output is not None:
        write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
