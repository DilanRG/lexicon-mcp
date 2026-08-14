"""Memory-bounded ConceptNet Numberbatch importer and USearch index builder."""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from usearch.index import Index

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


def verify_saved_index(path: Path, expected_count: int) -> None:
    """Reopen a persisted mmap index and require its exact key count."""

    index = Index.restore(path, view=True)
    if index is None:  # pragma: no cover - USearch ordinarily raises
        raise RuntimeError(f"USearch could not reopen saved index: {path}")
    try:
        observed = len(index)
        if observed != expected_count:
            raise RuntimeError(
                f"saved USearch index count mismatch for {path}: "
                f"expected {expected_count}, got {observed}"
            )
    finally:
        index.reset()


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
        shutil.rmtree(partial)
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
    configure_build_db(connection)
    create_semantic_schema(connection, dataset_version, dimensions)
    interner = CorpusInterner(connection)
    provenance_id = interner.provenance(SOURCE, SOURCE_LICENSE, SOURCE_URL)
    global_index = _index(dimensions, expected_rows)
    vector_path = vector_dir / "global.f16"
    accepted = 0
    malformed = 0
    duplicates = 0
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
    connection.commit()
    if accepted == 0:
        connection.close()
        raise ValueError("Numberbatch import produced no usable vectors")
    expected_bytes = accepted * dimensions * np.dtype("<f2").itemsize
    if vector_path.stat().st_size != expected_bytes:
        connection.close()
        raise RuntimeError("float16 vector artifact has an invalid size")

    global_path = index_dir / "global.usearch"
    global_index.save(global_path)
    del global_index
    verify_saved_index(global_path, accepted)

    vectors: np.memmap = np.memmap(
        vector_path, mode="r", dtype="<f2", shape=(accepted, dimensions)
    )
    language_counts = connection.execute(
        """SELECT t.language,COUNT(*)
        FROM semantic_terms s JOIN lexical_terms t ON t.term_id=s.term_id
        GROUP BY t.language ORDER BY t.language"""
    ).fetchall()
    for language, term_count in language_counts:
        filename = f"{language.replace('-', '_')}.usearch"
        relative = f"indexes/languages/{filename}"
        language_index = _index(dimensions, term_count)
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
        destination = language_dir / filename
        language_index.save(destination)
        del language_index
        verify_saved_index(destination, int(term_count))
        connection.execute(
            "INSERT INTO semantic_languages(language,index_file,term_count) VALUES (?,?,?)",
            (language, relative, term_count),
        )
        connection.commit()
    del vectors

    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
        (
            ("term_count", str(accepted)),
            ("source_expected_rows", str(expected_rows)),
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
        connection.close()
        raise RuntimeError(
            f"semantic mapping foreign-key check failed: {foreign_key_failure!r}"
        )
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if not integrity or integrity[0] != "ok":
        connection.close()
        raise RuntimeError(f"semantic mapping integrity check failed: {integrity!r}")
    finalize_readonly_db(connection)
    connection.close()
    os.replace(partial, semantic_dir)
    return {
        "expected_rows": expected_rows,
        "terms": accepted,
        "dimensions": dimensions,
        "languages": {language: count for language, count in language_counts},
        "malformed": malformed,
        "duplicates": duplicates,
    }
