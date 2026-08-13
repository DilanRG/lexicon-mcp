"""Checkpointed full-corpus build orchestration."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cmudict import build_cmudict
from .common import (
    Checkpoints,
    configure_build_db,
    file_sha256,
    finalize_readonly_db,
    source_fingerprint,
    write_json_atomic,
)
from .conceptnet import build_conceptnet
from .manifest import source_record, write_sources_lock
from .numberbatch import build_numberbatch
from .oewn import build_oewn
from .schema import create_lexical_schema
from .wiktextract import build_wiktextract


@dataclass(frozen=True)
class BuildInputs:
    oewn: Path
    wiktextract: tuple[Path, ...]
    conceptnet: Path
    numberbatch: Path
    cmudict: Path
    source_lock: Path | None = None
    notices_dir: Path | None = None

    def validate(self) -> None:
        paths = (self.oewn, *self.wiktextract, self.conceptnet, self.numberbatch, self.cmudict)
        if not self.wiktextract:
            raise ValueError("at least one Wiktextract JSONL source is required")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing corpus inputs: {', '.join(missing)}")
        if self.source_lock is not None and not self.source_lock.is_file():
            raise FileNotFoundError(f"source lock does not exist: {self.source_lock}")


def _copy_notices(source: Path | None, destination: Path) -> None:
    if source is None:
        return
    names = (
        "OEWN-LICENSE.md",
        "PRINCETON-WORDNET.txt",
        "CC-BY-4.0.txt",
        "CC-BY-SA-4.0.txt",
        "GFDL-1.3.txt",
        "CMUDICT.txt",
    )
    data_licenses = source / "DATA_LICENSES.md"
    missing = [name for name in names if not (source / "licenses" / name).is_file()]
    if not data_licenses.is_file():
        missing.insert(0, "DATA_LICENSES.md")
    if missing:
        raise FileNotFoundError(f"notice directory is incomplete: {', '.join(missing)}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_licenses, destination / "DATA_LICENSES.md")
    licenses = destination / "licenses"
    licenses.mkdir()
    for name in names:
        shutil.copy2(source / "licenses" / name, licenses / name)


def _composite_fingerprint(paths: tuple[Path, ...], known: dict[Path, str]) -> str:
    return "|".join(known.get(path.resolve(), source_fingerprint(path)) for path in paths)


def _source_records(
    inputs: BuildInputs, retrieved_at: str | None
) -> tuple[list[dict[str, Any]], dict[Path, str]]:
    path_by_id: dict[str, Path] = {
        "oewn": inputs.oewn,
        "conceptnet": inputs.conceptnet,
        "numberbatch": inputs.numberbatch,
        "cmudict": inputs.cmudict,
    }
    for position, path in enumerate(inputs.wiktextract):
        identifier = (
            "wiktextract"
            if len(inputs.wiktextract) == 1
            else f"wiktextract-{position:04d}"
        )
        path_by_id[identifier] = path
    if inputs.source_lock is not None:
        value = json.loads(inputs.source_lock.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or not isinstance(value.get("sources"), list):
            raise ValueError("source lock must use schema_version 1 and contain a sources list")
        raw_records = value["sources"]
        records: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        for item in raw_records:
            if not isinstance(item, dict):
                raise ValueError("source lock records must be objects")
            candidate_id = item.get("id")
            if not isinstance(candidate_id, str):
                raise ValueError("source lock record IDs must be strings")
            record = {str(key): value for key, value in item.items()}
            records.append(record)
            by_id[candidate_id] = record
        if set(path_by_id) != set(by_id):
            missing = sorted(set(path_by_id) - set(by_id))
            extra = sorted(set(by_id) - set(path_by_id))
            raise ValueError(f"source lock IDs mismatch; missing={missing}, extra={extra}")
        fingerprints: dict[Path, str] = {}
        for identifier, path in path_by_id.items():
            record = by_id[identifier]
            expected_size = record.get("size")
            expected_sha = record.get("sha256")
            if not isinstance(expected_size, int) or expected_size < 0:
                raise ValueError(f"source {identifier} has no valid size")
            if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                raise ValueError(f"source {identifier} has no valid SHA-256")
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"source {identifier} size mismatch: expected {expected_size}, "
                    f"got {actual_size}"
                )
            actual_sha = file_sha256(path)
            if actual_sha != expected_sha:
                raise ValueError(f"source {identifier} SHA-256 mismatch")
            fingerprints[path.resolve()] = f"sha256:{actual_sha};size:{actual_size}"
        return sorted(records, key=lambda item: item["id"]), fingerprints

    records = [
        source_record("oewn", inputs.oewn, retrieved_at=retrieved_at),
        source_record("conceptnet", inputs.conceptnet, retrieved_at=retrieved_at),
        source_record("numberbatch", inputs.numberbatch, retrieved_at=retrieved_at),
        source_record("cmudict", inputs.cmudict, retrieved_at=retrieved_at),
    ]
    for position, path in enumerate(inputs.wiktextract):
        record = source_record("wiktextract", path, retrieved_at=retrieved_at)
        record["id"] = (
            "wiktextract"
            if len(inputs.wiktextract) == 1
            else f"wiktextract-{position:04d}"
        )
        record["name"] = record["id"]
        records.append(record)
    record_by_id = {record["id"]: record for record in records}
    fingerprints = {}
    for identifier, path in path_by_id.items():
        record = record_by_id[identifier]
        fingerprints[path.resolve()] = f"sha256:{record['sha256']};size:{record['size']}"
    return records, fingerprints


def _db_counts(connection: sqlite3.Connection) -> dict[str, int]:
    tables = (
        "senses",
        "examples",
        "pronunciations",
        "translations",
        "synonyms",
        "relations",
        "pronunciations_words",
    )
    return {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in tables
    }


def build_full_corpus(
    inputs: BuildInputs,
    output_dir: Path,
    build_state: Path,
    *,
    dataset_version: str = "data-v1.0.0",
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    inputs.validate()
    if output_dir.exists():
        manifest_path = output_dir / "build-manifest.json"
        if manifest_path.exists():
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            if value.get("dataset_version") == dataset_version:
                if not isinstance(value, dict):
                    raise ValueError("build manifest must be an object")
                return {str(key): item for key, item in value.items()}
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    partial = output_dir.with_name(output_dir.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    _copy_notices(inputs.notices_dir, partial / "notices")
    checkpoints = Checkpoints(build_state / dataset_version / "checkpoints")
    lexical_path = partial / "lexicon.sqlite3"
    connection = sqlite3.connect(lexical_path)
    configure_build_db(connection)
    create_lexical_schema(connection, dataset_version)
    stage_counts: dict[str, Any] = {}
    records, fingerprints = _source_records(inputs, retrieved_at)

    stages: tuple[
        tuple[str, tuple[Path, ...], Any],
        ...,
    ] = (
        ("oewn", (inputs.oewn,), lambda: build_oewn(connection, inputs.oewn)),
        (
            "wiktextract",
            inputs.wiktextract,
            lambda: build_wiktextract(connection, list(inputs.wiktextract)),
        ),
        (
            "conceptnet",
            (inputs.conceptnet,),
            lambda: build_conceptnet(connection, inputs.conceptnet),
        ),
        ("cmudict", (inputs.cmudict,), lambda: build_cmudict(connection, inputs.cmudict)),
    )
    for stage, paths, builder in stages:
        fingerprint = _composite_fingerprint(paths, fingerprints)
        if checkpoints.complete(stage, fingerprint) and lexical_path.exists():
            stage_counts[stage] = {"resumed": True}
            continue
        counts = builder()
        checkpoints.mark(stage, fingerprint, counts=counts)
        stage_counts[stage] = counts
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if not integrity or integrity[0] != "ok":
        connection.close()
        raise RuntimeError(f"lexical database integrity check failed: {integrity!r}")
    lexical_counts = _db_counts(connection)
    finalize_readonly_db(connection)
    connection.close()

    semantic_fingerprint = fingerprints[inputs.numberbatch.resolve()]
    semantic_dir = partial / "semantic"
    if checkpoints.complete("numberbatch", semantic_fingerprint) and semantic_dir.exists():
        stage_counts["numberbatch"] = {"resumed": True}
    else:
        if semantic_dir.exists():
            shutil.rmtree(semantic_dir)
        semantic_counts = build_numberbatch(inputs.numberbatch, semantic_dir, dataset_version)
        checkpoints.mark("numberbatch", semantic_fingerprint, counts=semantic_counts)
        stage_counts["numberbatch"] = semantic_counts

    write_sources_lock(partial / "sources.lock.json", records)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "profile": "full",
        "lexical_counts": lexical_counts,
        "stage_counts": stage_counts,
        "semantic": stage_counts["numberbatch"],
        "ngrams_included": False,
    }
    write_json_atomic(partial / "build-manifest.json", manifest)
    os.replace(partial, output_dir)
    return manifest
