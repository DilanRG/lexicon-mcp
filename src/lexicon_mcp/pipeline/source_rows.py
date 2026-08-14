"""Deterministic logical-row counts and digests for pinned corpus sources.

Each digest is SHA-256 over canonical UTF-8 records in source order.  The
canonicalization intentionally describes source rows, not rows accepted by a
particular builder; accepted/skipped/malformed builder counts belong in the
build manifest.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import iter_text_lines, open_binary

_OEWN_LOGICAL_TAGS = frozenset(
    {
        "LexicalEntry",
        "Lemma",
        "Sense",
        "SenseRelation",
        "Synset",
        "SynsetRelation",
        "Definition",
        "Example",
        "Pronunciation",
    }
)


@dataclass(frozen=True)
class SourceRows:
    """Logical source-row metadata plus non-row diagnostic counts."""

    row_count: int
    row_digest: str
    malformed: int = 0

    def lock_fields(self) -> dict[str, int | str]:
        return {"row_count": self.row_count, "row_digest": self.row_digest}


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _finish(digest: Any, count: int, malformed: int = 0) -> SourceRows:
    return SourceRows(row_count=count, row_digest=str(digest.hexdigest()), malformed=malformed)


def measure_oewn(path: Path) -> SourceRows:
    """Hash selected WN-LMF semantic elements in XML document order.

    A row is ``local-tag<TAB>sorted-attribute-json<TAB>NFKC-collapsed-itertext``.
    Namespace prefixes and XML formatting therefore do not affect the digest.
    """

    digest = hashlib.sha256()
    count = 0
    with open_binary(path) as stream:
        for _event, element in ET.iterparse(stream, events=("end",)):
            tag = _local(element.tag)
            if tag not in _OEWN_LOGICAL_TAGS:
                continue
            attributes = {
                _local(str(key)): unicodedata.normalize("NFKC", str(value))
                for key, value in element.attrib.items()
            }
            attribute_json = json.dumps(
                attributes,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            text = _normalized_text("".join(element.itertext()))
            digest.update(f"{tag}\t{attribute_json}\t{text}\n".encode())
            count += 1
            if tag in {"LexicalEntry", "Synset"}:
                element.clear()
    return _finish(digest, count)


def _json_dumps(value: object) -> bytes:
    try:
        import orjson
    except ImportError:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    return bytes(orjson.dumps(value, option=orjson.OPT_SORT_KEYS))


def _json_loads(line: str) -> object:
    try:
        import orjson
    except ImportError:
        return json.loads(line)
    return orjson.loads(line)


def measure_wiktextract(path: Path) -> SourceRows:
    """Hash each syntactically valid JSON object as sorted compact JSON."""

    digest = hashlib.sha256()
    count = 0
    malformed = 0
    for _line_number, line in iter_text_lines(path):
        if not line.strip():
            continue
        try:
            value = _json_loads(line)
        except (ValueError, json.JSONDecodeError):
            malformed += 1
            continue
        if not isinstance(value, dict):
            malformed += 1
            continue
        digest.update(_json_dumps(value))
        digest.update(b"\n")
        count += 1
    return _finish(digest, count, malformed)


def measure_conceptnet(path: Path) -> SourceRows:
    """Hash decoded TSV rows containing the four required assertion fields."""

    digest = hashlib.sha256()
    count = 0
    malformed = 0
    for _line_number, line in iter_text_lines(path):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            malformed += 1
            continue
        digest.update(("\t".join(fields) + "\n").encode())
        count += 1
    return _finish(digest, count, malformed)


def measure_numberbatch(path: Path) -> SourceRows:
    """Hash dimension-valid vector rows using original whitespace-delimited tokens."""

    digest = hashlib.sha256()
    count = 0
    malformed = 0
    with open_binary(path) as stream:
        try:
            raw_header = next(iter(stream))
        except StopIteration as exc:
            raise ValueError("Numberbatch source is empty") from exc
        header_fields = raw_header.split()
        if len(header_fields) != 2:
            raise ValueError("Numberbatch header must contain row and dimension counts")
        expected, dimensions = (int(value) for value in header_fields)
        if expected < 0 or dimensions < 1:
            raise ValueError("Numberbatch header contains invalid counts")
        invalid_values = {b"nan", b"+nan", b"-nan", b"inf", b"+inf", b"-inf"}
        for raw in stream:
            stripped = raw.strip()
            if not stripped:
                continue
            # The pinned Numberbatch snapshot uses one ASCII space between
            # every token. This path avoids allocating 300 Python strings per
            # row while producing exactly the same whitespace-canonical row.
            if b"\t" not in stripped and b"  " not in stripped:
                if stripped.count(b" ") != dimensions:
                    malformed += 1
                    continue
                lowered = stripped.lower()
                if b"nan" in lowered or b"inf" in lowered:
                    fields = stripped.split()
                    if any(value.lower() in invalid_values for value in fields[1:]):
                        malformed += 1
                        continue
                canonical = stripped
            else:
                fields = stripped.split()
                if len(fields) != dimensions + 1:
                    malformed += 1
                    continue
                if any(value.lower() in invalid_values for value in fields[1:]):
                    malformed += 1
                    continue
                canonical = b" ".join(fields)
            digest.update(canonical)
            digest.update(b"\n")
            count += 1
    return _finish(digest, count, malformed)


def measure_cmudict(path: Path) -> SourceRows:
    """Hash non-comment rows containing a word and at least one phoneme."""

    digest = hashlib.sha256()
    count = 0
    malformed = 0
    for _line_number, line in iter_text_lines(path):
        stripped = line.strip()
        if not stripped or stripped.startswith((";;;", "#")):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            malformed += 1
            continue
        digest.update((" ".join(fields) + "\n").encode())
        count += 1
    return _finish(digest, count, malformed)


_MEASURERS: dict[str, Callable[[Path], SourceRows]] = {
    "oewn": measure_oewn,
    "wiktextract": measure_wiktextract,
    "conceptnet": measure_conceptnet,
    "numberbatch": measure_numberbatch,
    "cmudict": measure_cmudict,
}


def measure_source(source_id: str, path: Path) -> SourceRows:
    """Measure a canonical or sharded source ID."""

    canonical = "wiktextract" if source_id.startswith("wiktextract-") else source_id
    try:
        measurer = _MEASURERS[canonical]
    except KeyError as exc:
        raise ValueError(f"unsupported corpus source ID: {source_id}") from exc
    return measurer(path)


def verify_source_row_fields(source_id: str, record: dict[str, Any], path: Path) -> SourceRows:
    """Measure one source and require exact lock row metadata."""

    expected_count = record.get("row_count")
    expected_digest = record.get("row_digest")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 0
    ):
        raise ValueError(f"source {source_id} has no valid row_count")
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ValueError(f"source {source_id} has no valid row_digest")
    observed = measure_source(source_id, path)
    if observed.row_count != expected_count:
        raise ValueError(
            f"source {source_id} row_count mismatch: expected {expected_count}, "
            f"got {observed.row_count}"
        )
    if observed.row_digest != expected_digest:
        raise ValueError(f"source {source_id} row_digest mismatch")
    return observed
