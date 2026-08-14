"""Read-only exact-recall validation for the Numberbatch ANN artifacts."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from usearch.index import Index

from .locator import ActiveDataset

DEFAULT_LANGUAGES = ("en", "de", "es", "fr", "it", "pt", "ru", "ja", "ar", "hi")
SUPPORTED_SCHEMA_VERSION = "2"


@dataclass(frozen=True, slots=True)
class AnnSeedResult:
    """ANN and exact rankings for one deterministic seed and index scope."""

    index_scope: str
    language: str
    seed_semantic_id: int
    seed_term: str
    ann_semantic_ids: tuple[int, ...]
    ann_terms: tuple[str, ...]
    ann_languages: tuple[str, ...]
    exact_semantic_ids: tuple[int, ...]
    exact_terms: tuple[str, ...]
    exact_languages: tuple[str, ...]
    recall_at_k: float
    deterministic: bool
    strict_language_filtering: bool


@dataclass(frozen=True, slots=True)
class AnnAcceptanceReport:
    """Separate global and language-shard ANN quality for the same seed set."""

    languages: tuple[str, ...]
    seeds_per_language: int
    k: int
    results: tuple[AnnSeedResult, ...]

    def _scope_recall(self, scope: str) -> float:
        values = [item.recall_at_k for item in self.results if item.index_scope == scope]
        return sum(values) / len(values) if values else 0.0

    @property
    def recall_at_k(self) -> float:
        if not self.results:
            return 0.0
        return sum(item.recall_at_k for item in self.results) / len(self.results)

    @property
    def global_recall_at_k(self) -> float:
        return self._scope_recall("global")

    @property
    def shard_recall_at_k(self) -> float:
        return self._scope_recall("language_shard")

    @property
    def per_language_recall(self) -> dict[str, float]:
        report: dict[str, float] = {}
        for language in self.languages:
            values = [
                item.recall_at_k
                for item in self.results
                if item.index_scope == "language_shard" and item.language == language
            ]
            if values:
                report[language] = sum(values) / len(values)
        return report


@dataclass(frozen=True, slots=True)
class _SemanticRow:
    semantic_id: int
    vector_offset: int
    normalized_term: str
    language: str
    term: str

    @property
    def identity(self) -> tuple[str, str]:
        return (self.language, self.normalized_term)


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    semantic_id: int
    normalized_term: str
    language: str
    similarity: float

    @property
    def identity(self) -> tuple[str, str]:
        return (self.language, self.normalized_term)


def _sqlite_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row["key"]): str(row["value"])
        for row in connection.execute("SELECT key, value FROM metadata")
    }


def _safe_artifact(directory: Path, relative: str, description: str) -> Path:
    root = directory.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f"{description} path escapes the semantic artifact directory")
    if not path.is_file():
        raise RuntimeError(f"{description} is missing: {path}")
    return path


def _semantic_row(row: sqlite3.Row) -> _SemanticRow:
    return _SemanticRow(
        semantic_id=int(row["semantic_id"]),
        vector_offset=int(row["vector_offset"]),
        normalized_term=str(row["normalized_term"]),
        language=str(row["language"]),
        term=str(row["term"]),
    )


_ROW_SELECT = """
SELECT semantic.semantic_id, semantic.vector_offset, term.normalized_term,
       term.language, term.term
FROM semantic_terms AS semantic
JOIN lexical_terms AS term ON term.term_id = semantic.term_id
"""


def _seed_rows(
    connection: sqlite3.Connection, language: str, count: int
) -> tuple[_SemanticRow, ...]:
    """Choose stable row-count quantiles without materializing a language."""

    count_row = connection.execute(
        """
        SELECT COUNT(*) AS term_count
        FROM semantic_terms AS semantic
        JOIN lexical_terms AS term ON term.term_id = semantic.term_id
        WHERE term.language = ? AND term.normalized_term <> ''
          AND NOT EXISTS (
              SELECT 1
              FROM semantic_terms AS duplicate
              JOIN lexical_terms AS duplicate_term
                ON duplicate_term.term_id = duplicate.term_id
              WHERE duplicate_term.language = term.language
                AND duplicate_term.normalized_term = term.normalized_term
                AND duplicate.semantic_id <> semantic.semantic_id
          )
        """,
        (language,),
    ).fetchone()
    available = 0 if count_row is None else int(count_row["term_count"] or 0)
    if available < count:
        raise RuntimeError(
            f"language {language!r} has {available} semantic terms; {count} seeds required"
        )
    seeds: list[_SemanticRow] = []
    for stratum in range(count):
        offset = ((2 * stratum + 1) * available) // (2 * count)
        row = connection.execute(
            _ROW_SELECT
            + """
            WHERE term.language = ? AND term.normalized_term <> ''
              AND NOT EXISTS (
                  SELECT 1
                  FROM semantic_terms AS duplicate
                  JOIN lexical_terms AS duplicate_term
                    ON duplicate_term.term_id = duplicate.term_id
                  WHERE duplicate_term.language = term.language
                    AND duplicate_term.normalized_term = term.normalized_term
                    AND duplicate.semantic_id <> semantic.semantic_id
              )
            ORDER BY semantic.semantic_id LIMIT 1 OFFSET ?
            """,
            (language, offset),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"language {language!r} quantile seed {stratum} is missing")
        seeds.append(_semantic_row(row))
    return tuple(seeds)


def _rows_by_id(
    connection: sqlite3.Connection, semantic_ids: Iterable[int]
) -> dict[int, _SemanticRow]:
    ids = tuple(semantic_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = connection.execute(
        _ROW_SELECT + f" WHERE semantic.semantic_id IN ({placeholders})", ids
    ).fetchall()
    return {int(row["semantic_id"]): _semantic_row(row) for row in rows}


def _unit_vector(matrix: np.memmap[Any, Any], offset: int) -> np.ndarray[Any, Any]:
    vector = np.asarray(matrix[offset], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not math.isfinite(norm):
        raise RuntimeError(f"semantic vector offset {offset} is not finite and non-zero")
    result: np.ndarray[Any, Any] = vector / norm
    return result


def _merge_best(
    retained: list[_RankedCandidate],
    candidates: Iterable[_RankedCandidate],
    k: int,
) -> list[_RankedCandidate]:
    by_identity = {item.identity: item for item in retained}
    for candidate in candidates:
        current = by_identity.get(candidate.identity)
        if current is None or (-candidate.similarity, candidate.semantic_id) < (
            -current.similarity,
            current.semantic_id,
        ):
            by_identity[candidate.identity] = candidate
    return sorted(
        by_identity.values(), key=lambda item: (-item.similarity, item.semantic_id)
    )[:k]


def _maximum_identity_multiplicity(
    connection: sqlite3.Connection, language: str | None
) -> int:
    clause = "WHERE term.language = ?" if language is not None else ""
    grouping = "term.language, term.normalized_term"
    parameters: tuple[str, ...] = (language,) if language is not None else ()
    row = connection.execute(
        f"""
        SELECT MAX(term_count)
        FROM (
            SELECT COUNT(*) AS term_count
            FROM semantic_terms AS semantic
            JOIN lexical_terms AS term ON term.term_id = semantic.term_id
            {clause}
            GROUP BY {grouping}
        )
        """,
        parameters,
    ).fetchone()
    return max(1, int(row[0] or 1)) if row is not None else 1


def _exact_neighbors_batch(
    connection: sqlite3.Connection,
    matrix: np.memmap[Any, Any],
    seeds: tuple[_SemanticRow, ...],
    *,
    candidate_language: str | None,
    k: int,
    chunk_size: int,
) -> dict[int, tuple[_RankedCandidate, ...]]:
    """Stream candidates once for a batch and retain exact distinct-term top-k."""

    if not seeds:
        return {}
    queries: np.ndarray[Any, Any] = np.stack(
        [_unit_vector(matrix, seed.vector_offset) for seed in seeds]
    )
    sql = _ROW_SELECT
    parameters: tuple[str, ...] = ()
    if candidate_language is not None:
        sql += " WHERE term.language = ?"
        parameters = (candidate_language,)
    sql += " ORDER BY semantic.semantic_id"
    cursor = connection.execute(sql, parameters)
    retained: list[list[_RankedCandidate]] = [[] for _ in seeds]
    max_duplicates = _maximum_identity_multiplicity(connection, candidate_language)
    raw_keep = max(k + 1, k * max_duplicates + 1)

    while rows := cursor.fetchmany(chunk_size):
        offsets: np.ndarray[Any, Any] = np.fromiter(
            (int(row["vector_offset"]) for row in rows), dtype=np.int64, count=len(rows)
        )
        if bool(((offsets < 0) | (offsets >= matrix.shape[0])).any()):
            raise RuntimeError("semantic mapping contains an out-of-range vector offset")
        vectors = np.asarray(matrix[offsets], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(norms) & (norms > 0.0)
        vectors[valid] /= norms[valid, np.newaxis]
        scores: np.ndarray[Any, Any] = vectors @ queries.T
        scores[~valid, :] = -np.inf
        take = min(len(rows), raw_keep)
        for query_index, seed in enumerate(seeds):
            column = scores[:, query_index]
            candidate_indices: np.ndarray[Any, Any]
            if take == len(rows):
                candidate_indices = np.arange(len(rows), dtype=np.int64)
            else:
                partition = np.argpartition(column, -take)[-take:]
                cutoff = float(column[partition].min())
                greater = np.flatnonzero(column > cutoff)
                # Rows arrive in semantic-ID order. Include the lowest IDs at
                # an exact-score tie so the oracle is independent of chunking
                # and NumPy's intentionally unstable argpartition ordering.
                needed = take - len(greater)
                equal = np.flatnonzero(column == cutoff)[:needed]
                candidate_indices = np.concatenate((greater, equal))
            local: list[_RankedCandidate] = []
            for row_index in candidate_indices:
                row = rows[int(row_index)]
                semantic_id = int(row["semantic_id"])
                normalized_term = str(row["normalized_term"])
                language = str(row["language"])
                score = float(scores[int(row_index), query_index])
                identity = (language, normalized_term)
                if (
                    semantic_id == seed.semantic_id
                    or identity == seed.identity
                    or not normalized_term
                    or not math.isfinite(score)
                ):
                    continue
                local.append(
                    _RankedCandidate(semantic_id, normalized_term, language, score)
                )
            retained[query_index] = _merge_best(retained[query_index], local, k)

    return {
        seed.semantic_id: tuple(candidates)
        for seed, candidates in zip(seeds, retained, strict=True)
    }


def _ann_neighbors(
    index: Any,
    connection: sqlite3.Connection,
    matrix: np.memmap[Any, Any],
    query: np.ndarray[Any, Any],
    seed: _SemanticRow,
    *,
    k: int,
    index_size: int,
    expected_language: str | None,
) -> tuple[tuple[_SemanticRow, ...], bool, bool]:
    requested = min(index_size, max(200, k * 10, k + 1))
    first = index.search(query, requested)
    second = index.search(query, requested)
    first_ids = tuple(int(key) for key in first.keys)
    second_ids = tuple(int(key) for key in second.keys)
    candidate_ids = tuple(dict.fromkeys((*first_ids, *second_ids)))
    rows = _rows_by_id(connection, candidate_ids)
    if len(rows) != len(candidate_ids):
        missing = sorted(set(candidate_ids) - rows.keys())
        raise RuntimeError(f"ANN index returned unknown semantic IDs: {missing[:5]}")

    def rerank(candidate_ids: tuple[int, ...]) -> tuple[_SemanticRow, ...]:
        ranked: list[tuple[float, int, _SemanticRow]] = []
        for semantic_id in candidate_ids:
            row = rows[semantic_id]
            if not 0 <= row.vector_offset < matrix.shape[0]:
                raise RuntimeError("ANN candidate has an out-of-range vector offset")
            candidate = np.asarray(matrix[row.vector_offset], dtype=np.float32)
            norm = float(np.linalg.norm(candidate))
            if norm == 0.0 or not math.isfinite(norm):
                continue
            similarity = float((candidate / norm) @ query)
            if math.isfinite(similarity):
                ranked.append((similarity, semantic_id, row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected: list[_SemanticRow] = []
        seen_identities: set[tuple[str, str]] = set()
        for _similarity, _semantic_id, row in ranked:
            if (
                row.semantic_id == seed.semantic_id
                or row.identity == seed.identity
                or row.identity in seen_identities
            ):
                continue
            seen_identities.add(row.identity)
            selected.append(row)
            if len(selected) == k:
                break
        return tuple(selected)

    selected = rerank(first_ids)
    repeated = rerank(second_ids)
    deterministic = tuple(row.semantic_id for row in selected) == tuple(
        row.semantic_id for row in repeated
    )
    strict_language = expected_language is None or all(
        row.language == expected_language for row in selected
    )
    return selected, deterministic, strict_language


def _seed_result(
    scope: str,
    seed: _SemanticRow,
    ann: tuple[_SemanticRow, ...],
    exact: tuple[_RankedCandidate, ...],
    deterministic: bool,
    strict_language: bool,
    k: int,
) -> AnnSeedResult:
    exact_identities = {item.identity for item in exact}
    overlap = sum(row.identity in exact_identities for row in ann)
    denominator = min(k, len(exact))
    recall = overlap / denominator if denominator else 0.0
    return AnnSeedResult(
        index_scope=scope,
        language=seed.language,
        seed_semantic_id=seed.semantic_id,
        seed_term=seed.term,
        ann_semantic_ids=tuple(row.semantic_id for row in ann),
        ann_terms=tuple(row.normalized_term for row in ann),
        ann_languages=tuple(row.language for row in ann),
        exact_semantic_ids=tuple(item.semantic_id for item in exact),
        exact_terms=tuple(item.normalized_term for item in exact),
        exact_languages=tuple(item.language for item in exact),
        recall_at_k=recall,
        deterministic=deterministic,
        strict_language_filtering=strict_language,
    )


def validate_ann_acceptance(
    dataset: ActiveDataset,
    *,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    seeds_per_language: int = 10,
    k: int = 20,
    chunk_size: int = 8192,
) -> AnnAcceptanceReport:
    """Compare global and per-language mmap ANN indexes with exact cosine."""

    if not languages or len(set(languages)) != len(languages):
        raise ValueError("languages must be a non-empty sequence of unique language tags")
    if seeds_per_language < 1 or k < 1 or chunk_size < k:
        raise ValueError("seeds_per_language and k must be positive; chunk_size must be >= k")

    semantic = dataset.semantic_directory.resolve()
    mapping_path = _safe_artifact(semantic, "mapping.sqlite3", "semantic mapping")
    connection = _sqlite_ro(mapping_path)
    try:
        metadata = _metadata(connection)
        if metadata.get("dataset_version") != dataset.version:
            raise RuntimeError("semantic dataset version does not match the active dataset")
        if metadata.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
            raise RuntimeError("ANN acceptance requires compact semantic schema version 2")
        if metadata.get("vector_dtype") != "float16":
            raise RuntimeError("ANN acceptance requires float16 Numberbatch vectors")
        if metadata.get("index_metric") != "cos" or metadata.get("index_dtype") != "i8":
            raise RuntimeError("ANN acceptance requires cosine/i8 indexes")
        try:
            dimensions = int(metadata["dimensions"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("semantic metadata has no valid dimensions") from exc
        vector_path = _safe_artifact(
            semantic, metadata.get("vector_file", "vectors/global.f16"), "semantic vectors"
        )
        if dimensions < 1:
            raise RuntimeError("semantic metadata dimensions must be positive")
        try:
            expansion_search = int(metadata.get("expansion_search", "512"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("semantic metadata has no valid expansion_search") from exc
        if expansion_search < 512:
            raise RuntimeError("semantic expansion_search must be at least 512")
        row_bytes = dimensions * np.dtype("<f2").itemsize
        vector_bytes = vector_path.stat().st_size
        if vector_bytes % row_bytes:
            raise RuntimeError("semantic vector file size does not match its dimensions")
        row_count = vector_bytes // row_bytes
        matrix: np.memmap[Any, Any] = np.memmap(
            vector_path, mode="r", dtype="<f2", shape=(row_count, dimensions)
        )

        seeds_by_language = {
            language: _seed_rows(connection, language, seeds_per_language)
            for language in languages
        }
        all_seeds = tuple(
            seed for language in languages for seed in seeds_by_language[language]
        )
        for seed in all_seeds:
            if not 0 <= seed.vector_offset < row_count:
                raise RuntimeError(
                    f"seed {seed.semantic_id} has an out-of-range vector offset"
                )

        results: list[AnnSeedResult] = []
        global_path = _safe_artifact(
            semantic,
            metadata.get("global_index", "indexes/global.usearch"),
            "global ANN index",
        )
        global_size = int(metadata.get("term_count", row_count))
        global_exact = _exact_neighbors_batch(
            connection,
            matrix,
            all_seeds,
            candidate_language=None,
            k=k,
            chunk_size=chunk_size,
        )
        global_index = Index.restore(global_path, view=True)
        if global_index is None:
            raise RuntimeError("USearch could not restore the global ANN index")
        global_index.expansion_search = expansion_search
        for seed in all_seeds:
            query = _unit_vector(matrix, seed.vector_offset)
            ann, deterministic, strict_language = _ann_neighbors(
                global_index,
                connection,
                matrix,
                query,
                seed,
                k=k,
                index_size=global_size,
                expected_language=None,
            )
            results.append(
                _seed_result(
                    "global",
                    seed,
                    ann,
                    global_exact[seed.semantic_id],
                    deterministic,
                    strict_language,
                    k,
                )
            )
        del global_index

        for language in languages:
            shard = connection.execute(
                "SELECT index_file, term_count FROM semantic_languages WHERE language = ?",
                (language,),
            ).fetchone()
            if shard is None:
                raise RuntimeError(f"full corpus has no ANN shard for language {language!r}")
            index_size = int(shard["term_count"])
            if index_size <= k:
                raise RuntimeError(
                    f"ANN shard {language!r} has {index_size} terms; more than {k} required"
                )
            exact = _exact_neighbors_batch(
                connection,
                matrix,
                seeds_by_language[language],
                candidate_language=language,
                k=k,
                chunk_size=chunk_size,
            )
            index_path = _safe_artifact(semantic, str(shard["index_file"]), "ANN shard")
            index = Index.restore(index_path, view=True)
            if index is None:
                raise RuntimeError(f"USearch could not restore ANN shard {language!r}")
            index.expansion_search = expansion_search
            for seed in seeds_by_language[language]:
                query = _unit_vector(matrix, seed.vector_offset)
                ann, deterministic, strict_language = _ann_neighbors(
                    index,
                    connection,
                    matrix,
                    query,
                    seed,
                    k=k,
                    index_size=index_size,
                    expected_language=language,
                )
                results.append(
                    _seed_result(
                        "language_shard",
                        seed,
                        ann,
                        exact[seed.semantic_id],
                        deterministic,
                        strict_language,
                        k,
                    )
                )
            del index
        del matrix
    finally:
        connection.close()

    return AnnAcceptanceReport(languages, seeds_per_language, k, tuple(results))
