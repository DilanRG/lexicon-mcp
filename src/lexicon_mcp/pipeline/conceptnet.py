"""Streaming ConceptNet 5.7 assertions importer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import unquote

from .common import checked_language, iter_text_lines
from .constants import DIRECTION_CODES, RELATION_CODES
from .interner import CorpusInterner

SOURCE = "ConceptNet 5.7"
SOURCE_LICENSE = "CC-BY-SA-4.0"
SOURCE_URL = "https://conceptnet.io/"

_MAP = {
    "Synonym": ("synonym", "symmetric", "synonym", "symmetric"),
    "Antonym": ("antonym", "symmetric", "antonym", "symmetric"),
    "IsA": ("hypernym", "outbound", "hyponym", "inbound"),
    "PartOf": ("holonym", "outbound", "meronym", "inbound"),
    "HasA": ("meronym", "outbound", "holonym", "inbound"),
    "DerivedFrom": ("derived_from", "outbound", "derived_from", "inbound"),
    "EtymologicallyRelatedTo": (
        "etymologically_related",
        "symmetric",
        "etymologically_related",
        "symmetric",
    ),
    "UsedFor": ("used_for", "outbound", "used_for", "inbound"),
    "CapableOf": ("capable_of", "outbound", "capable_of", "inbound"),
    "AtLocation": ("at_location", "outbound", "at_location", "inbound"),
    "RelatedTo": ("related", "symmetric", "related", "symmetric"),
    "SimilarTo": ("related", "symmetric", "related", "symmetric"),
}


def _concept(uri: str) -> tuple[str, str] | None:
    pieces = uri.split("/")
    if len(pieces) < 4 or pieces[1] != "c":
        return None
    language = checked_language(pieces[2])
    if language == "und":
        return None
    term = unquote(pieces[3]).replace("_", " ").strip()
    return (language, term) if term else None


def build_conceptnet(
    connection: sqlite3.Connection,
    path: Path,
    commit_interval: int = 100_000,
    allowed_languages: frozenset[str] | None = None,
) -> dict[str, int]:
    counts = {
        "source_assertions": 0,
        "assertions": 0,
        "relations": 0,
        "skipped": 0,
        "malformed": 0,
    }
    interner = CorpusInterner(connection)
    provenance_id = interner.provenance(SOURCE, SOURCE_LICENSE, SOURCE_URL)
    connection.execute("DELETE FROM relations WHERE provenance_id=?", (provenance_id,))
    connection.commit()
    for _line_number, line in iter_text_lines(path):
        if not line:
            continue
        fields = line.split("\t", 4)
        if len(fields) < 4:
            counts["malformed"] += 1
            continue
        counts["source_assertions"] += 1
        relation_name = fields[1].rsplit("/", 1)[-1]
        mapping = _MAP.get(relation_name)
        source = _concept(fields[2])
        target = _concept(fields[3])
        if not mapping or not source or not target:
            counts["skipped"] += 1
            continue
        if allowed_languages is not None and (
            source[0] not in allowed_languages or target[0] not in allowed_languages
        ):
            counts["skipped"] += 1
            continue
        if len(fields) == 5:
            try:
                metadata = json.loads(fields[4])
                if isinstance(metadata, dict) and float(metadata.get("weight", 1.0)) <= 0:
                    counts["skipped"] += 1
                    continue
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        forward, forward_direction, _reverse, _reverse_direction = mapping
        inserted = _insert(
            connection,
            interner,
            provenance_id,
            source,
            forward,
            target,
            forward_direction,
        )
        counts["assertions"] += 1
        counts["relations"] += int(inserted)
        if counts["assertions"] % commit_interval == 0:
            connection.commit()
    connection.commit()
    return counts


def _insert(
    connection: sqlite3.Connection,
    interner: CorpusInterner,
    provenance_id: int,
    source: tuple[str, str],
    relation: str,
    target: tuple[str, str],
    direction: str,
) -> bool:
    source_language, source_term = source
    target_language, target_term = target
    source_term_id = interner.term(source_term, source_language)
    target_term_id = interner.term(target_term, target_language)
    cursor = connection.execute(
        "INSERT OR IGNORE INTO relations VALUES (?,?,?,?,?,?,?)",
        (
            source_term_id,
            None,
            RELATION_CODES[relation],
            target_term_id,
            None,
            DIRECTION_CODES[direction],
            provenance_id,
        ),
    )
    return cursor.rowcount > 0
