"""Memory-bounded ConceptNet Numberbatch importer and USearch index builder."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from usearch.index import Index

from lexicon_mcp.usearch_compat import index_count, open_index_view

from .common import configure_build_db, finalize_readonly_db, iter_text_lines
from .interner import CorpusInterner
from .schema import create_semantic_query_indexes, create_semantic_schema

SOURCE = "ConceptNet Numberbatch 19.08"
SOURCE_LICENSE = "CC-BY-SA-4.0"
SOURCE_URL = "https://github.com/commonsense/conceptnet-numberbatch"


def _parse_concept(value: str) -> tuple[str, str] | None:
    pieces = value.split("/")
    if len(pieces) < 4 or pieces[1] != "c":
        return None
    language = pieces[2].casefold()
    if not re.fullmatch(r"[a-z0-9]{2,8}(?:-[a-z0-9]{1,8})*", language):
        return None
    term = unquote(pieces[3]).replace("_", " ").strip()
    return (language, term) if term else None


def _index(dimensions: int, _expected: int) -> Index:
    return Index(
        ndim=dimensions,
        metric="cos",
        dtype="i8",
        connectivity=16,
        expansion_add=256,
        expansion_search=512,
        enable_key_lookups=False,
    )


def _flush_index(
    index: Index,
    pending_keys: list[int],
    pending_vectors: list[np.ndarray],
) -> None:
    if not pending_keys:
        return
    keys: np.ndarray = np.asarray(pending_keys, dtype=np.uint64)
    vectors: np.ndarray = np.stack(pending_vectors).astype(np.float32, copy=False)
    index.add(keys, vectors, copy=False)
    pending_keys.clear()
    pending_vectors.clear()


def verify_saved_index(path: Path, expected_count: int, dimensions: int) -> None:
    """Reopen a persisted mmap index and require its exact key count."""

    observed = index_count(path, dimensions=dimensions)
    if observed != expected_count:
        raise RuntimeError(
            f"saved USearch index count mismatch for {path}: "
            f"expected {expected_count}, got {observed}"
        )


def _source_shape(source_path: Path) -> tuple[int, int]:
    lines = iter_text_lines(source_path)
    try:
        try:
            _header_line, header = next(lines)
        except StopIteration as exc:
            raise ValueError("Numberbatch source is empty") from exc
    finally:
        close = getattr(lines, "close", None)
        if close is not None:
            close()
    header_parts = header.split()
    if len(header_parts) != 2:
        raise ValueError("Numberbatch header must contain row and dimension counts")
    expected_rows, dimensions = (int(item) for item in header_parts)
    if expected_rows < 1 or dimensions < 1:
        raise ValueError("Numberbatch header counts must be positive")
    return expected_rows, dimensions


def _validate_post_global_partial(
    partial: Path,
    dataset_version: str,
    *,
    expected_rows: int,
    dimensions: int,
) -> tuple[int, int, int]:
    """Validate an interrupted build without mutating or trusting file names alone."""

    if not partial.is_dir():
        raise FileNotFoundError(f"semantic partial directory does not exist: {partial}")
    mapping_path = partial / "mapping.sqlite3"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"semantic mapping does not exist: {mapping_path}")
    connection = sqlite3.connect(
        f"file:{mapping_path.as_posix()}?mode=ro", uri=True
    )
    try:
        quick = connection.execute("PRAGMA quick_check").fetchall()
        if len(quick) != 1 or str(quick[0][0]) != "ok":
            raise RuntimeError(f"semantic partial quick_check failed: {quick[:3]!r}")
        foreign_key_failure = connection.execute("PRAGMA foreign_key_check").fetchone()
        if foreign_key_failure is not None:
            raise RuntimeError(
                f"semantic partial foreign-key check failed: {foreign_key_failure!r}"
            )
        metadata = {
            str(key): str(value)
            for key, value in connection.execute("SELECT key,value FROM metadata")
        }
        required = {
            "schema_version": "2",
            "dataset_version": dataset_version,
            "dimensions": str(dimensions),
            "vector_dtype": "float16",
            "vector_file": "vectors/global.f16",
            "global_index": "indexes/global.usearch",
            "index_metric": "cos",
            "index_dtype": "i8",
        }
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in required.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"semantic partial metadata mismatch: {mismatches!r}")
        row = connection.execute(
            """SELECT COUNT(*),MIN(semantic_id),MAX(semantic_id),
            MIN(vector_offset),MAX(vector_offset),
            SUM(CASE WHEN semantic_id!=vector_offset THEN 1 ELSE 0 END)
            FROM semantic_terms"""
        ).fetchone()
        if row is None:
            raise RuntimeError("semantic partial has no term statistics")
        accepted = int(row[0])
        expected_bounds = (0, accepted - 1, 0, accepted - 1, 0)
        observed_bounds = tuple(None if value is None else int(value) for value in row[1:])
        if accepted < 1 or accepted > expected_rows or observed_bounds != expected_bounds:
            raise RuntimeError(
                "semantic partial IDs/vector offsets are not exact and contiguous: "
                f"terms={accepted}, bounds={observed_bounds!r}"
            )
        lexical_term_failure = connection.execute(
            """SELECT 1 FROM semantic_terms s
            LEFT JOIN lexical_terms t ON t.term_id=s.term_id
            WHERE t.term_id IS NULL LIMIT 1"""
        ).fetchone()
        if lexical_term_failure is not None:
            raise RuntimeError("semantic partial contains an unresolved lexical term")
        source_rows = metadata.get("source_expected_rows")
        if source_rows is not None and int(source_rows) != expected_rows:
            raise RuntimeError("semantic partial source row count does not match source header")
        malformed_value = metadata.get("source_malformed")
        duplicates_value = metadata.get("source_duplicates")
        if malformed_value is None or duplicates_value is None:
            # The first production build predates persisted import counters. It is
            # recoverable only when every declared source row became one term.
            if accepted != expected_rows:
                raise RuntimeError(
                    "semantic partial lacks import counters and not every source row was accepted"
                )
            malformed = duplicates = 0
        else:
            malformed = int(malformed_value)
            duplicates = int(duplicates_value)
            if min(malformed, duplicates) < 0 or accepted + malformed + duplicates != expected_rows:
                raise RuntimeError("semantic partial import counters are inconsistent")
    finally:
        connection.close()

    vector_path = partial / "vectors" / "global.f16"
    expected_vector_bytes = accepted * dimensions * np.dtype("<f2").itemsize
    if not vector_path.is_file() or vector_path.stat().st_size != expected_vector_bytes:
        raise RuntimeError(
            "semantic partial vector byte count does not match its exact matrix shape"
        )
    verify_saved_index(
        partial / "indexes" / "global.usearch", accepted, dimensions
    )
    return accepted, malformed, duplicates


def _language_index_artifact_is_valid(
    connection: sqlite3.Connection,
    path: Path,
    *,
    language: str,
    expected_count: int,
    dimensions: int,
) -> bool:
    if not path.is_file():
        return False
    index = None
    try:
        # Validate native metadata and the large-file-safe read-only view before
        # opening a second, temporary view with reverse key lookups enabled.
        validated = open_index_view(
            path,
            dimensions=dimensions,
            expected_count=expected_count,
        )
        validated.reset()
        # USearch 2.26's key iterator needs the reverse lookup map populated.
        # Build it only while validating one language shard, never for the
        # 9.16-million-key global index or normal runtime search.
        index = Index(
            ndim=dimensions,
            metric="cos",
            dtype="i8",
            connectivity=16,
            expansion_add=256,
            expansion_search=512,
            enable_key_lookups=True,
        )
        index.view(str(path))
        if len(index) != expected_count:
            return False
        observed_keys = np.sort(np.asarray(index.keys, dtype=np.uint64))
        expected_keys: np.ndarray = np.fromiter(
            (
                int(row[0])
                for row in connection.execute(
                    """SELECT s.semantic_id FROM semantic_terms s
                    JOIN lexical_terms t ON t.term_id=s.term_id
                    WHERE t.language=? ORDER BY s.semantic_id""",
                    (language,),
                )
            ),
            dtype=np.uint64,
            count=expected_count,
        )
        return bool(np.array_equal(observed_keys, expected_keys))
    except (OSError, RuntimeError, ValueError):
        return False
    finally:
        if index is not None:
            index.reset()


def _record_language_shard(
    connection: sqlite3.Connection,
    *,
    language: str,
    relative: str,
    term_count: int,
) -> None:
    connection.execute(
        """INSERT OR REPLACE INTO semantic_languages
        (language,index_file,term_count) VALUES (?,?,?)""",
        (language, relative, term_count),
    )
    connection.commit()


def _build_language_index(
    connection: sqlite3.Connection,
    vectors: np.memmap,
    destination: Path,
    *,
    language: str,
    term_count: int,
    dimensions: int,
    batch_size: int,
) -> None:
    """Build and atomically replace one verified language shard."""

    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    language_index = _index(dimensions, term_count)
    try:
        cursor = connection.execute(
            """SELECT s.semantic_id,s.vector_offset
            FROM semantic_terms s JOIN lexical_terms t ON t.term_id=s.term_id
            WHERE t.language=? ORDER BY s.semantic_id""",
            (language,),
        )
        while rows := cursor.fetchmany(batch_size):
            keys: np.ndarray = np.fromiter(
                (row[0] for row in rows), dtype=np.uint64, count=len(rows)
            )
            offsets: np.ndarray = np.fromiter(
                (row[1] for row in rows), dtype=np.int64, count=len(rows)
            )
            batch = np.asarray(vectors[offsets], dtype=np.float32)
            language_index.add(keys, batch, copy=False)
        language_index.save(temporary)
    finally:
        language_index.reset()
    if not _language_index_artifact_is_valid(
        connection,
        temporary,
        language=language,
        expected_count=term_count,
        dimensions=dimensions,
    ):
        raise RuntimeError(f"newly built language index failed validation: {language}")
    os.replace(temporary, destination)


def _finish_numberbatch_partial(
    partial: Path,
    semantic_dir: Path,
    dataset_version: str,
    *,
    expected_rows: int,
    dimensions: int,
    accepted: int,
    malformed: int,
    duplicates: int,
    batch_size: int,
) -> dict[str, object]:
    mapping_path = partial / "mapping.sqlite3"
    connection = sqlite3.connect(mapping_path)
    vectors: np.memmap | None = None
    try:
        configure_build_db(connection)
        vectors = np.memmap(
            partial / "vectors" / "global.f16",
            mode="r",
            dtype="<f2",
            shape=(accepted, dimensions),
        )
        language_counts = [
            (str(language), int(term_count))
            for language, term_count in connection.execute(
                """SELECT t.language,COUNT(*)
                FROM semantic_terms s JOIN lexical_terms t ON t.term_id=s.term_id
                GROUP BY t.language ORDER BY t.language"""
            )
        ]
        if sum(count for _, count in language_counts) != accepted:
            raise RuntimeError("semantic language counts do not cover every semantic term")
        expected_languages = {language: count for language, count in language_counts}
        recorded_rows = {
            str(language): (str(index_file), int(term_count))
            for language, index_file, term_count in connection.execute(
                "SELECT language,index_file,term_count FROM semantic_languages"
            )
        }
        unknown = sorted(set(recorded_rows) - set(expected_languages))
        if unknown:
            raise RuntimeError(f"semantic partial contains unknown language rows: {unknown!r}")

        language_dir = partial / "indexes" / "languages"
        language_dir.mkdir(parents=True, exist_ok=True)
        for language, term_count in language_counts:
            filename = f"{language.replace('-', '_')}.usearch"
            relative = f"indexes/languages/{filename}"
            destination = language_dir / filename
            stale_temporary = destination.with_name(destination.name + ".partial")
            destination_valid = _language_index_artifact_is_valid(
                connection,
                destination,
                language=language,
                expected_count=term_count,
                dimensions=dimensions,
            )
            temporary_valid = _language_index_artifact_is_valid(
                connection,
                stale_temporary,
                language=language,
                expected_count=term_count,
                dimensions=dimensions,
            )
            if destination_valid:
                stale_temporary.unlink(missing_ok=True)
                if recorded_rows.get(language) != (relative, term_count):
                    _record_language_shard(
                        connection,
                        language=language,
                        relative=relative,
                        term_count=term_count,
                    )
                continue
            if temporary_valid:
                os.replace(stale_temporary, destination)
                _record_language_shard(
                    connection,
                    language=language,
                    relative=relative,
                    term_count=term_count,
                )
                continue
            _build_language_index(
                connection,
                vectors,
                destination,
                language=language,
                term_count=term_count,
                dimensions=dimensions,
                batch_size=batch_size,
            )
            _record_language_shard(
                connection,
                language=language,
                relative=relative,
                term_count=term_count,
            )

        expected_files = {
            f"{language.replace('-', '_')}.usearch" for language in expected_languages
        }
        actual_files = {
            path.name for path in language_dir.iterdir() if path.is_file()
        }
        if actual_files != expected_files:
            raise RuntimeError(
                "semantic language artifact set mismatch: "
                f"missing={sorted(expected_files - actual_files)!r}, "
                f"extra={sorted(actual_files - expected_files)!r}"
            )
        del vectors
        vectors = None

        provenance_row = connection.execute(
            "SELECT provenance_id FROM provenance WHERE source=?",
            (SOURCE,),
        ).fetchone()
        if provenance_row is None:
            raise RuntimeError("semantic source provenance row is missing")
        provenance_id = int(provenance_row[0])
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
            (
                ("term_count", str(accepted)),
                ("source_expected_rows", str(expected_rows)),
                ("source_malformed", str(malformed)),
                ("source_duplicates", str(duplicates)),
                ("language_count", str(len(language_counts))),
                ("language_index_dir", "indexes/languages"),
                ("connectivity", "16"),
                ("expansion_add", "256"),
                ("expansion_search", "512"),
                ("source", SOURCE),
                ("source_license", SOURCE_LICENSE),
                ("source_url", SOURCE_URL),
                ("source_provenance_id", str(provenance_id)),
            ),
        )
        create_semantic_query_indexes(connection)
        foreign_key_failure = connection.execute("PRAGMA foreign_key_check").fetchone()
        if foreign_key_failure is not None:
            raise RuntimeError(
                f"semantic mapping foreign-key check failed: {foreign_key_failure!r}"
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise RuntimeError(f"semantic mapping integrity check failed: {integrity!r}")
        finalize_readonly_db(connection)
    finally:
        if vectors is not None:
            del vectors
        connection.close()

    if semantic_dir.exists():
        raise FileExistsError(f"refusing to replace existing semantic directory: {semantic_dir}")
    os.replace(partial, semantic_dir)
    return {
        "expected_rows": expected_rows,
        "terms": accepted,
        "dimensions": dimensions,
        "languages": {language: count for language, count in language_counts},
        "malformed": malformed,
        "duplicates": duplicates,
    }


def resume_numberbatch_partial(
    source_path: Path,
    semantic_dir: Path,
    dataset_version: str,
    *,
    batch_size: int = 8192,
) -> dict[str, object]:
    """Strictly validate and finish an interrupted post-global semantic build."""

    if semantic_dir.exists():
        raise FileExistsError(f"refusing to replace existing semantic directory: {semantic_dir}")
    expected_rows, dimensions = _source_shape(source_path)
    partial = semantic_dir.with_name(semantic_dir.name + ".partial")
    accepted, malformed, duplicates = _validate_post_global_partial(
        partial,
        dataset_version,
        expected_rows=expected_rows,
        dimensions=dimensions,
    )
    return _finish_numberbatch_partial(
        partial,
        semantic_dir,
        dataset_version,
        expected_rows=expected_rows,
        dimensions=dimensions,
        accepted=accepted,
        malformed=malformed,
        duplicates=duplicates,
        batch_size=batch_size,
    )


def build_numberbatch(
    source_path: Path,
    semantic_dir: Path,
    dataset_version: str,
    *,
    batch_size: int = 8192,
) -> dict[str, object]:
    """Build float16 seed storage and global/per-language i8 cosine indexes.

    All USearch indexes use the global ``semantic_id`` as their key. The vector
    offset is a row offset into ``vectors/global.f16``, not a byte offset.
    Artifacts are assembled under a sibling partial directory and promoted only
    after validation succeeds.
    """

    if semantic_dir.exists():
        raise FileExistsError(f"refusing to replace existing semantic directory: {semantic_dir}")
    partial = semantic_dir.with_name(semantic_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(
            f"refusing to replace existing semantic partial: {partial}; "
            "use the validated recovery command or remove it manually"
        )
    vector_dir = partial / "vectors"
    index_dir = partial / "indexes"
    language_dir = index_dir / "languages"
    language_dir.mkdir(parents=True)
    vector_dir.mkdir(parents=True)

    lines = iter_text_lines(source_path)
    try:
        _header_line, header = next(lines)
    except StopIteration as exc:
        raise ValueError("Numberbatch source is empty") from exc
    header_parts = header.split()
    if len(header_parts) != 2:
        raise ValueError("Numberbatch header must contain row and dimension counts")
    expected_rows, dimensions = (int(item) for item in header_parts)
    if expected_rows < 1 or dimensions < 1:
        raise ValueError("Numberbatch header counts must be positive")

    mapping_path = partial / "mapping.sqlite3"
    connection = sqlite3.connect(mapping_path)
    global_index: Index | None = None
    accepted = malformed = duplicates = 0
    try:
        configure_build_db(connection)
        create_semantic_schema(connection, dataset_version, dimensions)
        interner = CorpusInterner(connection)
        interner.provenance(SOURCE, SOURCE_LICENSE, SOURCE_URL)
        global_index = _index(dimensions, expected_rows)
        vector_path = vector_dir / "global.f16"
        pending_keys: list[int] = []
        pending_vectors: list[np.ndarray] = []

        with vector_path.open("wb") as vector_stream:
            for _line_number, line in lines:
                concept, separator, raw_vector = line.partition(" ")
                if not separator:
                    malformed += 1
                    continue
                parsed = _parse_concept(concept)
                if not parsed:
                    malformed += 1
                    continue
                language, term = parsed
                try:
                    vector: np.ndarray = np.fromstring(
                        raw_vector, dtype=np.float32, sep=" "
                    )
                except ValueError:
                    malformed += 1
                    continue
                if vector.size != dimensions or not np.isfinite(vector).all():
                    malformed += 1
                    continue
                norm = float(np.linalg.norm(vector))
                if norm == 0.0:
                    malformed += 1
                    continue
                vector /= norm
                semantic_id = accepted
                try:
                    connection.execute(
                        """INSERT INTO semantic_terms
                        (semantic_id,concept,term_id,vector_offset) VALUES (?,?,?,?)""",
                        (
                            semantic_id,
                            concept,
                            interner.term(term, language),
                            semantic_id,
                        ),
                    )
                except sqlite3.IntegrityError:
                    duplicates += 1
                    continue
                vector.astype("<f2", copy=False).tofile(vector_stream)
                pending_keys.append(semantic_id)
                pending_vectors.append(vector)
                accepted += 1
                if len(pending_keys) >= batch_size:
                    _flush_index(global_index, pending_keys, pending_vectors)
                    connection.commit()
            _flush_index(global_index, pending_keys, pending_vectors)
            vector_stream.flush()
            os.fsync(vector_stream.fileno())

        if accepted == 0:
            raise ValueError("Numberbatch import produced no usable vectors")
        if accepted + malformed + duplicates != expected_rows:
            raise RuntimeError(
                "Numberbatch header row count does not match imported, malformed, "
                "and duplicate rows"
            )
        expected_bytes = accepted * dimensions * np.dtype("<f2").itemsize
        if vector_path.stat().st_size != expected_bytes:
            raise RuntimeError("float16 vector artifact has an invalid size")
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
            (
                ("source_expected_rows", str(expected_rows)),
                ("source_malformed", str(malformed)),
                ("source_duplicates", str(duplicates)),
            ),
        )
        connection.commit()

        global_path = index_dir / "global.usearch"
        global_index.save(global_path)
        global_index.reset()
        global_index = None
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        if global_index is not None:
            global_index.reset()
        connection.close()
        close = getattr(lines, "close", None)
        if close is not None:
            close()

    verify_saved_index(partial / "indexes" / "global.usearch", accepted, dimensions)
    return _finish_numberbatch_partial(
        partial,
        semantic_dir,
        dataset_version,
        expected_rows=expected_rows,
        dimensions=dimensions,
        accepted=accepted,
        malformed=malformed,
        duplicates=duplicates,
        batch_size=batch_size,
    )
