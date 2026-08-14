"""Checkpointed full-corpus build orchestration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lexicon_mcp.data.locking import InstallationLock

from .cmudict import build_cmudict
from .common import (
    Checkpoints,
    configure_build_db,
    file_sha256,
    finalize_readonly_db,
    source_fingerprint,
    write_json_atomic,
)
from .conceptnet import SOURCE as CONCEPTNET_SOURCE
from .conceptnet import build_conceptnet
from .manifest import source_record, write_sources_lock
from .numberbatch import (
    build_numberbatch,
    resume_numberbatch_partial,
    verify_saved_index,
)
from .oewn import SOURCE as OEWN_SOURCE
from .oewn import build_oewn
from .schema import create_lexical_query_indexes, create_lexical_schema
from .size_estimator import INSTALLED_LIMIT, assert_size_targets
from .source_rows import measure_source, verify_source_row_fields
from .wiktextract import SOURCE as WIKTEXTRACT_SOURCE
from .wiktextract import build_wiktextract

FULL_CORPUS_FLOORS: dict[str, dict[str, int]] = {
    "oewn": {"synsets": 100_000, "senses": 180_000},
    "wiktextract": {
        "entries": 10_000_000,
        "senses": 12_000_000,
        "translations": 3_000_000,
        "synonyms": 3_500_000,
        "pronunciations": 6_000_000,
        "language_codes": 4_000,
    },
    "conceptnet": {
        "source_assertions": 20_000_000,
        "assertions": 18_000_000,
        "relations": 18_000_000,
    },
    "numberbatch": {"expected_rows": 9_000_000, "terms": 9_000_000},
    "cmudict": {"entries": 130_000},
}

_LEXICAL_STAGE_SOURCES = {
    "oewn": OEWN_SOURCE,
    "wiktextract": WIKTEXTRACT_SOURCE,
    "conceptnet": CONCEPTNET_SOURCE,
}


def _pipeline_identity() -> str:
    """Hash pipeline implementation bytes so changed builders cannot resume stale data."""

    digest = hashlib.sha256()
    directory = Path(__file__).parent
    for path in sorted(directory.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    compatibility = directory.parent / "usearch_compat.py"
    if compatibility.is_file():
        digest.update(b"../usearch_compat.py\0")
        digest.update(compatibility.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
    licenses.mkdir(exist_ok=True)
    for name in names:
        shutil.copy2(source / "licenses" / name, licenses / name)


def _composite_fingerprint(
    paths: tuple[Path, ...], known: dict[Path, str], pipeline_identity: str
) -> str:
    source = "|".join(known.get(path.resolve(), source_fingerprint(path)) for path in paths)
    return f"pipeline:{pipeline_identity}|{source}"


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
            logical = verify_source_row_fields(identifier, record, path)
            fingerprints[path.resolve()] = (
                f"sha256:{actual_sha};size:{actual_size};rows:{logical.row_count};"
                f"row-digest:{logical.row_digest}"
            )
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
        logical = measure_source(identifier, path)
        record.update(logical.lock_fields())
        fingerprints[path.resolve()] = (
            f"sha256:{record['sha256']};size:{record['size']};rows:{logical.row_count};"
            f"row-digest:{logical.row_digest}"
        )
    return records, fingerprints


def _db_counts(connection: sqlite3.Connection) -> dict[str, int]:
    tables = (
        "provenance",
        "lexical_terms",
        "lexical_entries",
        "senses",
        "examples",
        "pronunciations",
        "translations",
        "synonyms",
        "relations",
        "pronunciations_words",
    )
    counts = {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in tables
    }
    # A contentless FTS5 table intentionally cannot be scanned. Its one row
    # per pronunciation term is proved by the source-table count and MATCH
    # acceptance tests rather than by relying on FTS5 shadow-table internals.
    counts["wordplay_fts"] = int(
        connection.execute(
            "SELECT COUNT(DISTINCT term_id) FROM pronunciations_words"
        ).fetchone()[0]
    )
    return counts


def _installed_size(path: Path) -> int:
    """Measure promoted artifact bytes while excluding transient SQLite files."""

    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.name.endswith(("-wal", "-shm", ".partial"))
    )


def evaluate_corpus_floors(
    stage_counts: dict[str, Any],
) -> tuple[dict[str, dict[str, dict[str, int | bool]]], list[str]]:
    """Compare exact observed builder counts with public full-corpus floors."""

    results: dict[str, dict[str, dict[str, int | bool]]] = {}
    failures: list[str] = []
    for stage, metrics in FULL_CORPUS_FLOORS.items():
        stage_value = stage_counts.get(stage)
        stage_results: dict[str, dict[str, int | bool]] = {}
        for metric, minimum in metrics.items():
            observed_value = stage_value.get(metric) if isinstance(stage_value, dict) else None
            observed = (
                observed_value
                if isinstance(observed_value, int) and not isinstance(observed_value, bool)
                else -1
            )
            passed = observed >= minimum
            stage_results[metric] = {
                "observed": observed,
                "minimum": minimum,
                "passed": passed,
            }
            if not passed:
                failures.append(
                    f"{stage}.{metric}: observed {observed}, required at least {minimum}"
                )
        results[stage] = stage_results
    return results, failures


def _metadata_value(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _lexical_stage_physical_counts(
    connection: sqlite3.Connection, stage: str
) -> dict[str, int]:
    """Count rows physically owned by one resumable lexical source stage."""

    if stage == "cmudict":
        row = connection.execute("SELECT COUNT(*) FROM pronunciations_words").fetchone()
        return {"pronunciations_words": int(row[0])}
    source = _LEXICAL_STAGE_SOURCES[stage]
    provenance = connection.execute(
        "SELECT provenance_id FROM provenance WHERE source=?", (source,)
    ).fetchone()
    if provenance is None:
        raise RuntimeError(f"lexical stage {stage} has no on-disk provenance row")
    provenance_id = int(provenance[0])
    statements = {
        "entries": "SELECT COUNT(*) FROM lexical_entries WHERE provenance_id=?",
        "senses": """SELECT COUNT(*) FROM senses s JOIN lexical_entries e
            ON e.entry_id=s.entry_id WHERE e.provenance_id=?""",
        "examples": """SELECT COUNT(*) FROM examples x JOIN senses s
            ON s.sense_id=x.sense_id JOIN lexical_entries e ON e.entry_id=s.entry_id
            WHERE e.provenance_id=?""",
        "pronunciations": """SELECT COUNT(*) FROM pronunciations p JOIN lexical_entries e
            ON e.entry_id=p.entry_id WHERE e.provenance_id=?""",
        "translations": "SELECT COUNT(*) FROM translations WHERE provenance_id=?",
        "synonyms": "SELECT COUNT(*) FROM synonyms WHERE provenance_id=?",
        "relations": "SELECT COUNT(*) FROM relations WHERE provenance_id=?",
    }
    return {
        name: int(connection.execute(statement, (provenance_id,)).fetchone()[0])
        for name, statement in statements.items()
    }


def _record_lexical_checkpoint(
    connection: sqlite3.Connection,
    checkpoints: Checkpoints,
    stage: str,
    fingerprint: str,
    logical_counts: dict[str, Any],
) -> None:
    physical_counts = _lexical_stage_physical_counts(connection, stage)
    values = {
        f"checkpoint.{stage}.fingerprint": fingerprint,
        f"checkpoint.{stage}.logical_counts": json.dumps(
            logical_counts, sort_keys=True, separators=(",", ":")
        ),
        f"checkpoint.{stage}.physical_counts": json.dumps(
            physical_counts, sort_keys=True, separators=(",", ":")
        ),
    }
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", values.items()
    )
    connection.commit()
    checkpoints.mark(
        stage,
        fingerprint,
        counts=logical_counts,
        physical_counts=physical_counts,
    )


def _resume_lexical_checkpoint(
    connection: sqlite3.Connection,
    checkpoints: Checkpoints,
    stage: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    """Trust a marker only when its internal ledger and physical rows agree."""

    marker = checkpoints.record(stage, fingerprint)
    if marker is None:
        return None
    logical = marker.get("counts")
    physical = marker.get("physical_counts")
    if not isinstance(logical, dict) or not isinstance(physical, dict):
        return None
    try:
        stored_logical = json.loads(
            _metadata_value(connection, f"checkpoint.{stage}.logical_counts") or "null"
        )
        stored_physical = json.loads(
            _metadata_value(connection, f"checkpoint.{stage}.physical_counts") or "null"
        )
        if (
            _metadata_value(connection, f"checkpoint.{stage}.fingerprint") != fingerprint
            or stored_logical != logical
            or stored_physical != physical
            or _lexical_stage_physical_counts(connection, stage) != physical
        ):
            return None
    except (json.JSONDecodeError, KeyError, RuntimeError, sqlite3.Error):
        return None
    counts = {str(key): item for key, item in logical.items()}
    counts["resumed"] = True
    return counts


def _record_semantic_checkpoint(
    semantic_dir: Path, fingerprint: str, counts: dict[str, object]
) -> None:
    mapping = semantic_dir / "mapping.sqlite3"
    connection = sqlite3.connect(mapping)
    try:
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
            (
                ("checkpoint.numberbatch.fingerprint", fingerprint),
                (
                    "checkpoint.numberbatch.logical_counts",
                    json.dumps(counts, sort_keys=True, separators=(",", ":")),
                ),
            ),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()


def _verified_semantic_counts(
    semantic_dir: Path, dataset_version: str
) -> dict[str, object]:
    """Reopen mapping, vectors, global index, and every language shard."""

    root = semantic_dir.resolve()
    mapping = root / "mapping.sqlite3"
    connection = sqlite3.connect(
        f"file:{mapping.as_posix()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        quick = connection.execute("PRAGMA quick_check").fetchall()
        if len(quick) != 1 or str(quick[0][0]) != "ok":
            raise RuntimeError(f"semantic mapping quick_check failed: {quick[:3]!r}")
        foreign_key_failure = connection.execute("PRAGMA foreign_key_check").fetchone()
        if foreign_key_failure is not None:
            raise RuntimeError(
                f"semantic mapping foreign-key check failed: {foreign_key_failure!r}"
            )
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM metadata")
        }
        if metadata.get("dataset_version") != dataset_version:
            raise RuntimeError("semantic dataset version does not match the build")
        if metadata.get("schema_version") != "2":
            raise RuntimeError("semantic mapping is not schema version 2")
        dimensions = int(metadata["dimensions"])
        terms = int(metadata["term_count"])
        mapped_terms = int(
            connection.execute("SELECT COUNT(*) FROM semantic_terms").fetchone()[0]
        )
        if terms < 1 or dimensions < 1 or mapped_terms != terms:
            raise RuntimeError("semantic mapping term/dimension counts are invalid")
        language_rows = connection.execute(
            """SELECT language,index_file,term_count
            FROM semantic_languages ORDER BY language"""
        ).fetchall()
        languages = {
            str(row["language"]): int(row["term_count"]) for row in language_rows
        }
        derived_languages = {
            str(row["language"]): int(row["term_count"])
            for row in connection.execute(
                """SELECT t.language,COUNT(*) AS term_count
                FROM semantic_terms s JOIN lexical_terms t ON t.term_id=s.term_id
                GROUP BY t.language ORDER BY t.language"""
            )
        }
        if languages != derived_languages or sum(languages.values()) != terms:
            raise RuntimeError(
                "semantic language rows do not exactly partition the global terms"
            )
        try:
            metadata_language_count = int(metadata.get("language_count", "-1"))
        except ValueError as exc:
            raise RuntimeError(
                "semantic metadata language_count is not an integer"
            ) from exc
        if metadata_language_count != len(languages):
            raise RuntimeError("semantic metadata language_count is not exact")
        if metadata.get("language_index_dir") != "indexes/languages":
            raise RuntimeError("semantic metadata language index directory is invalid")
        index_files = [str(row["index_file"]) for row in language_rows]
        if len(index_files) != len(set(index_files)):
            raise RuntimeError("semantic language index paths are not unique")
        vector_path = (root / metadata["vector_file"]).resolve()
        global_path = (root / metadata["global_index"]).resolve()
        if not vector_path.is_relative_to(root) or not global_path.is_relative_to(root):
            raise RuntimeError("semantic metadata contains an escaping artifact path")
        expected_vector_bytes = terms * dimensions * 2
        if vector_path.stat().st_size != expected_vector_bytes:
            raise RuntimeError("semantic vector byte count does not match mapping metadata")
        verify_saved_index(global_path, terms, dimensions)
        for row in language_rows:
            language = str(row["language"])
            expected_relative = f"indexes/languages/{language.replace('-', '_')}.usearch"
            if str(row["index_file"]) != expected_relative:
                raise RuntimeError("semantic language index path is not canonical")
            index_path = (root / str(row["index_file"])).resolve()
            if not index_path.is_relative_to(root):
                raise RuntimeError("semantic language index path escapes its artifact root")
            verify_saved_index(index_path, int(row["term_count"]), dimensions)
    finally:
        connection.close()
    return {"terms": terms, "dimensions": dimensions, "languages": languages}


def _resume_semantic_checkpoint(
    checkpoints: Checkpoints,
    semantic_dir: Path,
    dataset_version: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    marker = checkpoints.record("numberbatch", fingerprint)
    if marker is None or not semantic_dir.is_dir():
        return None
    logical = marker.get("counts")
    if not isinstance(logical, dict):
        return None
    try:
        connection = sqlite3.connect(
            f"file:{(semantic_dir / 'mapping.sqlite3').as_posix()}?mode=ro&immutable=1",
            uri=True,
        )
        try:
            stored_fingerprint = _metadata_value(
                connection, "checkpoint.numberbatch.fingerprint"
            )
            stored_counts = json.loads(
                _metadata_value(connection, "checkpoint.numberbatch.logical_counts") or "null"
            )
        finally:
            connection.close()
        verified = _verified_semantic_counts(semantic_dir, dataset_version)
        expected_languages = logical.get("languages")
        if (
            stored_fingerprint != fingerprint
            or stored_counts != logical
            or logical.get("terms") != verified["terms"]
            or logical.get("dimensions") != verified["dimensions"]
            or expected_languages != verified["languages"]
        ):
            return None
    except (json.JSONDecodeError, KeyError, OSError, RuntimeError, sqlite3.Error, ValueError):
        return None
    counts = {str(key): item for key, item in logical.items()}
    counts["resumed"] = True
    return counts


def build_full_corpus(
    inputs: BuildInputs,
    output_dir: Path,
    build_state: Path,
    *,
    dataset_version: str = "data-v1.0.0",
    retrieved_at: str | None = None,
    enforce_corpus_floors: bool = False,
) -> dict[str, Any]:
    inputs.validate()
    # Freeze one implementation identity for every stage in this invocation;
    # a source edit during a long build cannot silently produce mixed markers.
    pipeline_identity = _pipeline_identity()
    # Run the calibrated resource gate before creating output/checkpoint state.
    size_projection = assert_size_targets()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    partial = output_dir.with_name(output_dir.name + ".partial")
    preserved_semantic = [
        path
        for path in (partial / "semantic.partial", partial / "semantic")
        if path.exists()
    ]
    if preserved_semantic:
        raise FileExistsError(
            "refusing normal build over preserved semantic state; use the validated "
            f"recovery command: {[str(path) for path in preserved_semantic]!r}"
        )
    records, fingerprints = _source_records(inputs, retrieved_at)
    if enforce_corpus_floors:
        disk_root = output_dir.parent
        while not disk_root.exists() and disk_root != disk_root.parent:
            disk_root = disk_root.parent
        available = shutil.disk_usage(disk_root).free
        required = size_projection["peak_build"]
        if available < required:
            raise RuntimeError(
                f"peak-build free-space gate failure: {available} bytes available, "
                f"{required} required"
            )
    partial.mkdir(parents=True, exist_ok=True)
    _copy_notices(inputs.notices_dir, partial / "notices")
    checkpoints = Checkpoints(build_state / dataset_version / "checkpoints")
    lexical_path = partial / "lexicon.sqlite3"
    connection = sqlite3.connect(lexical_path)
    stage_counts: dict[str, Any] = {}
    try:
        configure_build_db(connection)
        create_lexical_schema(connection, dataset_version)
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
            fingerprint = _composite_fingerprint(paths, fingerprints, pipeline_identity)
            resumed_counts = _resume_lexical_checkpoint(
                connection, checkpoints, stage, fingerprint
            )
            if resumed_counts is not None:
                stage_counts[stage] = resumed_counts
                continue
            counts = builder()
            _record_lexical_checkpoint(
                connection, checkpoints, stage, fingerprint, counts
            )
            stage_counts[stage] = counts
        create_lexical_query_indexes(connection)
        lexical_counts = _db_counts(connection)
        for stage, counts in stage_counts.items():
            if not isinstance(counts, dict):
                continue
            for metric, value in counts.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    connection.execute(
                        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                        (f"build.{stage}.{metric}", str(value)),
                    )
        foreign_key_failure = connection.execute("PRAGMA foreign_key_check").fetchone()
        if foreign_key_failure is not None:
            raise RuntimeError(
                f"lexical database foreign-key check failed: {foreign_key_failure!r}"
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise RuntimeError(f"lexical database integrity check failed: {integrity!r}")
        finalize_readonly_db(connection)
    finally:
        connection.close()

    semantic_fingerprint = _composite_fingerprint(
        (inputs.numberbatch,), fingerprints, pipeline_identity
    )
    semantic_dir = partial / "semantic"
    resumed_counts = _resume_semantic_checkpoint(
        checkpoints,
        semantic_dir,
        dataset_version,
        semantic_fingerprint,
    )
    if resumed_counts is not None:
        stage_counts["numberbatch"] = resumed_counts
    else:
        if semantic_dir.exists():
            shutil.rmtree(semantic_dir)
        semantic_counts = build_numberbatch(inputs.numberbatch, semantic_dir, dataset_version)
        _record_semantic_checkpoint(
            semantic_dir, semantic_fingerprint, semantic_counts
        )
        verified_semantic = _verified_semantic_counts(semantic_dir, dataset_version)
        if (
            semantic_counts.get("terms") != verified_semantic["terms"]
            or semantic_counts.get("dimensions") != verified_semantic["dimensions"]
            or semantic_counts.get("languages") != verified_semantic["languages"]
        ):
            raise RuntimeError("fresh Numberbatch artifacts disagree with their build counts")
        checkpoints.mark("numberbatch", semantic_fingerprint, counts=semantic_counts)
        stage_counts["numberbatch"] = semantic_counts

    corpus_floors, floor_failures = evaluate_corpus_floors(stage_counts)
    if enforce_corpus_floors and floor_failures:
        raise RuntimeError("full-corpus floor failure: " + "; ".join(floor_failures))

    write_sources_lock(partial / "sources.lock.json", records)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "profile": "full",
        "pipeline_identity": pipeline_identity,
        "lexical_counts": lexical_counts,
        "stage_counts": stage_counts,
        "semantic": stage_counts["numberbatch"],
        "corpus_floors": corpus_floors,
        "corpus_floors_enforced": enforce_corpus_floors,
        "resource_projection": size_projection,
        "installed_size": 0,
        "installed_size_limit": INSTALLED_LIMIT,
        "ngrams_included": False,
    }
    manifest_path = partial / "build-manifest.json"
    for _attempt in range(3):
        write_json_atomic(manifest_path, manifest)
        installed_size = _installed_size(partial)
        if installed_size > INSTALLED_LIMIT:
            raise RuntimeError(
                f"installed-size gate failure: {installed_size} bytes exceeds {INSTALLED_LIMIT}"
            )
        if manifest["installed_size"] == installed_size:
            break
        manifest["installed_size"] = installed_size
    else:  # pragma: no cover - digit length converges in at most two writes
        raise RuntimeError("installed-size manifest did not converge")
    os.replace(partial, output_dir)
    return manifest


def _recorded_lexical_pipeline_identity(checkpoint_dir: Path) -> str:
    identities: set[str] = set()
    for stage in ("oewn", "wiktextract", "conceptnet", "cmudict"):
        path = checkpoint_dir / f"{stage}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"lexical recovery checkpoint is unreadable: {path}") from exc
        if not isinstance(value, dict) or value.get("state") != "complete":
            raise RuntimeError(f"lexical recovery checkpoint is incomplete: {path}")
        fingerprint = value.get("fingerprint")
        if not isinstance(fingerprint, str):
            raise RuntimeError(f"lexical recovery checkpoint has no fingerprint: {path}")
        match = re.match(r"^pipeline:([0-9a-f]{64})\|", fingerprint)
        if match is None:
            raise RuntimeError(f"lexical recovery checkpoint fingerprint is malformed: {path}")
        identities.add(match.group(1))
    if len(identities) != 1:
        raise RuntimeError(
            f"lexical recovery checkpoints mix pipeline identities: {sorted(identities)!r}"
        )
    return identities.pop()


def _validated_recovery_lexical_state(
    inputs: BuildInputs,
    partial: Path,
    checkpoints: Checkpoints,
    fingerprints: dict[Path, str],
    *,
    dataset_version: str,
    original_pipeline_identity: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    lexical_path = partial / "lexicon.sqlite3"
    if not lexical_path.is_file():
        raise FileNotFoundError(f"recovery lexical database does not exist: {lexical_path}")
    for suffix in ("-wal", "-shm"):
        if lexical_path.with_name(lexical_path.name + suffix).exists():
            raise RuntimeError("recovery lexical database has unfinalized SQLite sidecars")

    connection = sqlite3.connect(
        f"file:{lexical_path.as_posix()}?mode=ro&immutable=1", uri=True
    )
    try:
        quick = connection.execute("PRAGMA quick_check").fetchall()
        if len(quick) != 1 or str(quick[0][0]) != "ok":
            raise RuntimeError(f"recovery lexical quick_check failed: {quick[:3]!r}")
        foreign_key_failure = connection.execute("PRAGMA foreign_key_check").fetchone()
        if foreign_key_failure is not None:
            raise RuntimeError(
                f"recovery lexical foreign-key check failed: {foreign_key_failure!r}"
            )
        metadata = {
            str(key): str(value)
            for key, value in connection.execute("SELECT key,value FROM metadata")
        }
        if metadata.get("schema_version") != "2":
            raise RuntimeError("recovery lexical database is not schema version 2")
        if metadata.get("dataset_version") != dataset_version:
            raise RuntimeError("recovery lexical dataset version does not match")

        stage_sources: dict[str, tuple[Path, ...]] = {
            "oewn": (inputs.oewn,),
            "wiktextract": inputs.wiktextract,
            "conceptnet": (inputs.conceptnet,),
            "cmudict": (inputs.cmudict,),
        }
        stage_counts: dict[str, Any] = {}
        for stage, paths in stage_sources.items():
            fingerprint = _composite_fingerprint(
                paths, fingerprints, original_pipeline_identity
            )
            resumed = _resume_lexical_checkpoint(
                connection, checkpoints, stage, fingerprint
            )
            if resumed is None:
                raise RuntimeError(
                    f"recovery lexical checkpoint/database verification failed for {stage}"
                )
            resumed.pop("resumed", None)
            stage_counts[stage] = resumed
        lexical_counts = _db_counts(connection)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise RuntimeError(f"recovery lexical integrity_check failed: {integrity!r}")
    finally:
        connection.close()
    return stage_counts, lexical_counts


def _completed_semantic_recovery_counts(
    semantic_dir: Path, dataset_version: str
) -> dict[str, object]:
    verified = _verified_semantic_counts(semantic_dir, dataset_version)
    mapping = semantic_dir / "mapping.sqlite3"
    connection = sqlite3.connect(
        f"file:{mapping.as_posix()}?mode=ro&immutable=1", uri=True
    )
    try:
        metadata = {
            str(key): str(value)
            for key, value in connection.execute("SELECT key,value FROM metadata")
        }
        terms_value = verified.get("terms")
        dimensions_value = verified.get("dimensions")
        languages = verified.get("languages")
        if (
            not isinstance(terms_value, int)
            or isinstance(terms_value, bool)
            or not isinstance(dimensions_value, int)
            or isinstance(dimensions_value, bool)
            or not isinstance(languages, dict)
        ):
            raise RuntimeError("completed recovery semantic verification is malformed")
        terms = terms_value
        row = connection.execute(
            """SELECT MIN(semantic_id),MAX(semantic_id),MIN(vector_offset),
            MAX(vector_offset),
            SUM(CASE WHEN semantic_id!=vector_offset THEN 1 ELSE 0 END)
            FROM semantic_terms"""
        ).fetchone()
        expected = (0, terms - 1, 0, terms - 1, 0)
        observed = tuple(None if value is None else int(value) for value in row or ())
        if observed != expected:
            raise RuntimeError(
                "completed recovery semantic IDs/vector offsets are not exact and contiguous"
            )
        expected_rows = int(metadata["source_expected_rows"])
        malformed = int(metadata["source_malformed"])
        duplicates = int(metadata["source_duplicates"])
        if terms + malformed + duplicates != expected_rows:
            raise RuntimeError("completed recovery semantic import counters are inconsistent")
    except (KeyError, ValueError) as exc:
        raise RuntimeError("completed recovery semantic metadata is incomplete") from exc
    finally:
        connection.close()
    return {
        "expected_rows": expected_rows,
        "terms": terms,
        "dimensions": dimensions_value,
        "languages": languages,
        "malformed": malformed,
        "duplicates": duplicates,
    }


def _require_sha256(value: str, *, field: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256")
    return value


def _recovery_semantic_root(partial: Path) -> Path:
    candidates = [
        path
        for path in (partial / "semantic.partial", partial / "semantic")
        if path.is_dir()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "recovery requires exactly one semantic.partial or semantic directory"
        )
    return candidates[0]


def _verify_recovery_artifact_anchors(
    partial: Path,
    *,
    expected_lexicon_sha256: str,
    expected_global_index_sha256: str,
    expected_global_vectors_sha256: str,
) -> dict[str, str]:
    semantic_root = _recovery_semantic_root(partial)
    paths = {
        "lexicon.sqlite3": partial / "lexicon.sqlite3",
        "global.usearch": semantic_root / "indexes" / "global.usearch",
        "global.f16": semantic_root / "vectors" / "global.f16",
    }
    expected = {
        "lexicon.sqlite3": _require_sha256(
            expected_lexicon_sha256, field="expected_lexicon_sha256"
        ),
        "global.usearch": _require_sha256(
            expected_global_index_sha256, field="expected_global_index_sha256"
        ),
        "global.f16": _require_sha256(
            expected_global_vectors_sha256, field="expected_global_vectors_sha256"
        ),
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"recovery anchor artifact does not exist: {path}")
        observed[name] = file_sha256(path)
        if observed[name] != expected[name]:
            raise RuntimeError(
                f"recovery anchor mismatch for {name}: "
                f"expected {expected[name]}, got {observed[name]}"
            )
    return observed


def _verify_recovery_notices(source: Path | None, staged: Path) -> None:
    if source is None:
        raise FileNotFoundError("recovery requires the supplied repository notices directory")
    pairs = [(source / "DATA_LICENSES.md", staged / "DATA_LICENSES.md")]
    pairs.extend(
        (source / "licenses" / name, staged / "licenses" / name)
        for name in (
            "OEWN-LICENSE.md",
            "PRINCETON-WORDNET.txt",
            "CC-BY-4.0.txt",
            "CC-BY-SA-4.0.txt",
            "GFDL-1.3.txt",
            "CMUDICT.txt",
        )
    )
    for supplied, recovered in pairs:
        if not supplied.is_file() or not recovered.is_file():
            raise FileNotFoundError(
                f"recovery notice pair is incomplete: {supplied}, {recovered}"
            )
        if supplied.read_bytes() != recovered.read_bytes():
            raise RuntimeError(f"staged recovery notice differs from supplied bytes: {recovered}")


def _recovery_implementation_provenance() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    lock_path = repository / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError("recovery provenance requires the repository uv.lock")
    return {
        "usearch_version": importlib.metadata.version("usearch"),
        "uv_lock_sha256": file_sha256(lock_path),
    }


def _verify_final_recovery_tree(
    partial: Path,
    semantic_dir: Path,
    *,
    dataset_version: str,
    notices_dir: Path | None,
) -> dict[str, object]:
    counts = _completed_semantic_recovery_counts(semantic_dir, dataset_version)
    mapping = semantic_dir / "mapping.sqlite3"
    connection = sqlite3.connect(
        f"file:{mapping.as_posix()}?mode=ro&immutable=1", uri=True
    )
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise RuntimeError(f"final semantic integrity_check failed: {integrity!r}")
    finally:
        connection.close()
    _verify_recovery_notices(notices_dir, partial / "notices")
    transient = [
        path
        for path in partial.rglob("*")
        if path.is_file()
        and path.name.endswith((".partial", "-wal", "-shm"))
    ]
    if transient:
        raise RuntimeError(
            f"recovery tree contains transient files: {[str(path) for path in transient]!r}"
        )
    return counts


def _recover_full_corpus_from_semantic_partial_locked(
    inputs: BuildInputs,
    output_dir: Path,
    build_state: Path,
    *,
    dataset_version: str = "data-v1.0.0",
    retrieved_at: str | None = None,
    enforce_corpus_floors: bool = True,
    original_build_commit: str,
    recovery_commit: str,
    expected_lexicon_sha256: str,
    expected_global_index_sha256: str,
    expected_global_vectors_sha256: str,
    expected_original_pipeline_identity: str | None = None,
) -> dict[str, Any]:
    """Finish one verified post-global build without rerunning lexical stages.

    This is deliberately separate from :func:`build_full_corpus`: it has no
    destructive fallback, never opens the completed lexical database writable,
    and remains idempotent when semantic completion succeeded but a later gate
    prevented final dataset promotion.
    """

    for label, commit in (
        ("original_build_commit", original_build_commit),
        ("recovery_commit", recovery_commit),
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"{label} must be a lowercase 40-hex Git commit")
    inputs.validate()
    recovery_pipeline_identity = _pipeline_identity()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    partial = output_dir.with_name(output_dir.name + ".partial")
    if not partial.is_dir():
        raise FileNotFoundError(f"recovery dataset partial does not exist: {partial}")
    _verify_recovery_notices(inputs.notices_dir, partial / "notices")
    artifact_anchors = _verify_recovery_artifact_anchors(
        partial,
        expected_lexicon_sha256=expected_lexicon_sha256,
        expected_global_index_sha256=expected_global_index_sha256,
        expected_global_vectors_sha256=expected_global_vectors_sha256,
    )
    implementation_provenance = _recovery_implementation_provenance()

    records, fingerprints = _source_records(inputs, retrieved_at)
    checkpoint_dir = build_state / dataset_version / "checkpoints"
    original_pipeline_identity = _recorded_lexical_pipeline_identity(checkpoint_dir)
    if expected_original_pipeline_identity is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_original_pipeline_identity):
            raise ValueError(
                "expected_original_pipeline_identity must be a lowercase 64-hex digest"
            )
        if original_pipeline_identity != expected_original_pipeline_identity:
            raise RuntimeError(
                "recorded lexical pipeline identity does not match original build commit"
            )
    checkpoints = Checkpoints(checkpoint_dir)
    stage_counts, lexical_counts = _validated_recovery_lexical_state(
        inputs,
        partial,
        checkpoints,
        fingerprints,
        dataset_version=dataset_version,
        original_pipeline_identity=original_pipeline_identity,
    )

    semantic_dir = partial / "semantic"
    semantic_partial = partial / "semantic.partial"
    if semantic_dir.is_dir():
        if semantic_partial.exists():
            raise RuntimeError("recovery dataset contains both semantic and semantic.partial")
        semantic_counts = _completed_semantic_recovery_counts(
            semantic_dir, dataset_version
        )
    else:
        if not semantic_partial.is_dir():
            raise FileNotFoundError("recovery semantic.partial does not exist")
        semantic_counts = resume_numberbatch_partial(
            inputs.numberbatch, semantic_dir, dataset_version
        )
        verified_semantic = _completed_semantic_recovery_counts(
            semantic_dir, dataset_version
        )
        if semantic_counts != verified_semantic:
            raise RuntimeError("recovered Numberbatch artifacts disagree with verification")

    semantic_fingerprint = _composite_fingerprint(
        (inputs.numberbatch,), fingerprints, recovery_pipeline_identity
    )
    _record_semantic_checkpoint(semantic_dir, semantic_fingerprint, semantic_counts)
    checkpoints.mark("numberbatch", semantic_fingerprint, counts=semantic_counts)
    stage_counts["numberbatch"] = semantic_counts

    corpus_floors, floor_failures = evaluate_corpus_floors(stage_counts)
    if enforce_corpus_floors and floor_failures:
        raise RuntimeError("full-corpus floor failure: " + "; ".join(floor_failures))

    write_sources_lock(partial / "sources.lock.json", records)
    size_projection = assert_size_targets()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "profile": "full",
        "pipeline_identity": recovery_pipeline_identity,
        "recovery_provenance": {
            "mode": "validated-post-global-semantic-resume",
            "original_build_commit": original_build_commit,
            "original_pipeline_identity": original_pipeline_identity,
            "recovery_commit": recovery_commit,
            "recovery_pipeline_identity": recovery_pipeline_identity,
            "artifact_sha256": artifact_anchors,
            **implementation_provenance,
        },
        "lexical_counts": lexical_counts,
        "stage_counts": stage_counts,
        "semantic": semantic_counts,
        "corpus_floors": corpus_floors,
        "corpus_floors_enforced": enforce_corpus_floors,
        "resource_projection": size_projection,
        "installed_size": 0,
        "installed_size_limit": INSTALLED_LIMIT,
        "ngrams_included": False,
    }
    manifest_path = partial / "build-manifest.json"
    for _attempt in range(3):
        write_json_atomic(manifest_path, manifest)
        installed_size = _installed_size(partial)
        if installed_size > INSTALLED_LIMIT:
            raise RuntimeError(
                f"installed-size gate failure: {installed_size} bytes exceeds {INSTALLED_LIMIT}"
            )
        if manifest["installed_size"] == installed_size:
            break
        manifest["installed_size"] = installed_size
    else:  # pragma: no cover - digit length converges in at most two writes
        raise RuntimeError("installed-size manifest did not converge")
    final_semantic_counts = _verify_final_recovery_tree(
        partial,
        semantic_dir,
        dataset_version=dataset_version,
        notices_dir=inputs.notices_dir,
    )
    if final_semantic_counts != semantic_counts:
        raise RuntimeError("final semantic verification disagrees with recovery counts")
    final_artifact_anchors = _verify_recovery_artifact_anchors(
        partial,
        expected_lexicon_sha256=expected_lexicon_sha256,
        expected_global_index_sha256=expected_global_index_sha256,
        expected_global_vectors_sha256=expected_global_vectors_sha256,
    )
    if final_artifact_anchors != artifact_anchors:
        raise RuntimeError("immutable recovery artifact anchors changed during recovery")
    os.replace(partial, output_dir)
    return manifest


def recover_full_corpus_from_semantic_partial(
    inputs: BuildInputs,
    output_dir: Path,
    build_state: Path,
    *,
    dataset_version: str = "data-v1.0.0",
    retrieved_at: str | None = None,
    enforce_corpus_floors: bool = True,
    original_build_commit: str,
    recovery_commit: str,
    expected_lexicon_sha256: str,
    expected_global_index_sha256: str,
    expected_global_vectors_sha256: str,
    expected_original_pipeline_identity: str | None = None,
) -> dict[str, Any]:
    """Exclusively finish one verified post-global corpus build."""

    lock_path = build_state / dataset_version / ".recovery.lock"
    with InstallationLock(lock_path):
        return _recover_full_corpus_from_semantic_partial_locked(
            inputs,
            output_dir,
            build_state,
            dataset_version=dataset_version,
            retrieved_at=retrieved_at,
            enforce_corpus_floors=enforce_corpus_floors,
            original_build_commit=original_build_commit,
            recovery_commit=recovery_commit,
            expected_lexicon_sha256=expected_lexicon_sha256,
            expected_global_index_sha256=expected_global_index_sha256,
            expected_global_vectors_sha256=expected_global_vectors_sha256,
            expected_original_pipeline_identity=expected_original_pipeline_identity,
        )
