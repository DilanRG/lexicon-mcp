"""Build compact-v2 prefix samples and report actual SQLite bytes per source row."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

from lexicon_mcp.pipeline.common import (
    configure_build_db,
    finalize_readonly_db,
    open_binary,
)
from lexicon_mcp.pipeline.conceptnet import build_conceptnet
from lexicon_mcp.pipeline.schema import create_lexical_query_indexes, create_lexical_schema
from lexicon_mcp.pipeline.wiktextract import build_wiktextract


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--wiktextract", type=Path, required=True)
    value.add_argument("--conceptnet", type=Path, required=True)
    value.add_argument("--work-dir", type=Path, required=True)
    value.add_argument("--wiktextract-rows", type=int, default=20_000)
    value.add_argument("--conceptnet-rows", type=int, default=200_000)
    return value


def _prefix(source: Path, destination: Path, rows: int) -> None:
    if rows < 1:
        raise ValueError("sample row counts must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open_binary(source) as input_stream, destination.open("wb") as output_stream:
        written = 0
        for raw in input_stream:
            if not raw.strip():
                continue
            output_stream.write(raw.rstrip(b"\r\n") + b"\n")
            written += 1
            if written >= rows:
                break
    if written != rows:
        raise ValueError(f"{source} contains only {written} sample rows")


def _build(
    path: Path, builder: Callable[[sqlite3.Connection], dict[str, int]]
) -> tuple[dict[str, int], int, dict[str, int]]:
    connection = sqlite3.connect(path)
    configure_build_db(connection)
    create_lexical_schema(connection, "size-calibration-v2")
    counts = builder(connection)
    create_lexical_query_indexes(connection)
    table_counts = {
        str(name): int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        for name in (
            "lexical_terms",
            "lexical_entries",
            "senses",
            "examples",
            "pronunciations",
            "translations",
            "synonyms",
            "relations",
        )
    }
    finalize_readonly_db(connection)
    connection.close()
    return counts, path.stat().st_size, table_counts


def main() -> None:
    args = parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    wiki_sample = args.work_dir / "wiktextract-prefix.jsonl"
    concept_sample = args.work_dir / "conceptnet-prefix.tsv"
    _prefix(args.wiktextract, wiki_sample, args.wiktextract_rows)
    _prefix(args.conceptnet, concept_sample, args.conceptnet_rows)
    wiki_db = args.work_dir / "wiktextract-v2.sqlite3"
    concept_db = args.work_dir / "conceptnet-v2.sqlite3"
    for path in (wiki_db, concept_db):
        if path.exists():
            raise FileExistsError(f"refusing to replace calibration database: {path}")
    wiki_counts, wiki_size, wiki_tables = _build(
        wiki_db,
        lambda connection: build_wiktextract(connection, [wiki_sample]),
    )
    concept_counts, concept_size, concept_tables = _build(
        concept_db,
        lambda connection: build_conceptnet(connection, concept_sample),
    )
    print(
        json.dumps(
            {
                "schema_version": 2,
                "wiktextract": {
                    "source_rows": args.wiktextract_rows,
                    "artifact_bytes": wiki_size,
                    "bytes_per_source_row": wiki_size / args.wiktextract_rows,
                    "builder_counts": wiki_counts,
                    "table_counts": wiki_tables,
                },
                "conceptnet": {
                    "source_rows": args.conceptnet_rows,
                    "artifact_bytes": concept_size,
                    "bytes_per_source_row": concept_size / args.conceptnet_rows,
                    "builder_counts": concept_counts,
                    "table_counts": concept_tables,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
