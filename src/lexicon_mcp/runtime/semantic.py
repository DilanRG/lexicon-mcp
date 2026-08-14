"""Lazy, process-isolated semantic-neighbour search."""

from __future__ import annotations

import atexit
import math
import multiprocessing
import sqlite3
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .offline import deny_network


class SemanticSearch(Protocol):
    @property
    def available(self) -> bool: ...

    def search(
        self,
        word: str,
        source_language: str,
        target_language: str | None,
        limit: int,
        min_similarity: float | None,
    ) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _SemanticRequest:
    directory: str
    dataset_version: str
    word: str
    source_language: str
    target_language: str | None
    limit: int
    min_similarity: float | None


_INDEX_CACHE: OrderedDict[str, Any] = OrderedDict()
_INDEX_CACHE_SIZE = 4
_SUPPORTED_SCHEMA_VERSION = "2"


def _sqlite_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True, check_same_thread=False
    )
    connection.row_factory = sqlite3.Row
    return connection


def _load_index(path: Path, expansion_search: int) -> Any:
    key = str(path.resolve())
    index = _INDEX_CACHE.pop(key, None)
    if index is not None:
        index.expansion_search = expansion_search
        _INDEX_CACHE[key] = index
        return index
    try:
        from usearch.index import Index
    except ImportError as exc:  # pragma: no cover - packaging guarantees the dependency
        raise RuntimeError("USearch is unavailable; semantic search cannot run") from exc
    index = Index.restore(key, view=True)
    if index is None:  # pragma: no cover - USearch raises for ordinary restore failures
        raise RuntimeError(f"USearch could not restore semantic index: {path}")
    index.expansion_search = expansion_search
    _INDEX_CACHE[key] = index
    while len(_INDEX_CACHE) > _INDEX_CACHE_SIZE:
        _INDEX_CACHE.popitem(last=False)
    return index


def _semantic_search_task(request: _SemanticRequest) -> list[dict[str, Any]]:
    """Execute one semantic query with process-local networking denied."""

    with deny_network():
        return _semantic_search_task_offline(request)


def _semantic_search_task_offline(request: _SemanticRequest) -> list[dict[str, Any]]:
    """Worker entrypoint; imports NumPy and USearch only in the semantic process."""

    import numpy as np

    directory = Path(request.directory)
    mapping_path = directory / "mapping.sqlite3"
    connection = _sqlite_ro(mapping_path)
    try:
        metadata = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key, value FROM metadata")
        }
        if metadata.get("dataset_version") != request.dataset_version:
            raise RuntimeError(
                "Semantic artifact version does not match the active lexical dataset"
            )
        if metadata.get("schema_version") != _SUPPORTED_SCHEMA_VERSION:
            raise RuntimeError("Semantic artifact has an unsupported schema version")
        try:
            dimensions = int(metadata["dimensions"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Semantic metadata has no valid dimensions") from exc
        if metadata.get("vector_dtype") != "float16":
            raise RuntimeError("Semantic vectors must use float16 storage")
        if metadata.get("index_metric") != "cos" or metadata.get("index_dtype") != "i8":
            raise RuntimeError("Semantic indexes must use cosine/i8 storage")
        try:
            expansion_search = int(metadata.get("expansion_search", "512"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Semantic metadata has no valid expansion_search") from exc
        if expansion_search < 512:
            raise RuntimeError("Semantic expansion_search must be at least 512")
        vector_relative = metadata.get("vector_file", "vectors/global.f16")
        vector_path = (directory / vector_relative).resolve()
        if not vector_path.is_relative_to(directory.resolve()) or not vector_path.is_file():
            raise RuntimeError("Semantic vector path is missing or unsafe")

        seed = connection.execute(
            """
            SELECT semantic.semantic_id, semantic.vector_offset
            FROM semantic_terms AS semantic
            JOIN lexical_terms AS term ON term.term_id = semantic.term_id
            WHERE term.normalized_term = ? AND term.language = ?
            ORDER BY semantic.semantic_id
            LIMIT 1
            """,
            (request.word, request.source_language),
        ).fetchone()
        if seed is None:
            return []

        item_size = np.dtype("<f2").itemsize
        vector_bytes = vector_path.stat().st_size
        row_bytes = dimensions * item_size
        if row_bytes <= 0 or vector_bytes % row_bytes:
            raise RuntimeError("Semantic vector file size does not match its dimensions")
        row_count = vector_bytes // row_bytes
        vector_offset = int(seed["vector_offset"])
        if not 0 <= vector_offset < row_count:
            raise RuntimeError("Semantic seed vector offset is outside the vector matrix")
        matrix: Any = np.memmap(
            vector_path, dtype="<f2", mode="r", shape=(row_count, dimensions)
        )
        query_vector = np.asarray(matrix[vector_offset], dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0.0 or not math.isfinite(query_norm):
            raise RuntimeError("Semantic seed vector is not finite and non-zero")
        query_vector /= query_norm

        if request.target_language:
            shard = connection.execute(
                "SELECT index_file FROM semantic_languages WHERE language = ?",
                (request.target_language,),
            ).fetchone()
            if shard is None:
                return []
            index_path = (directory / str(shard["index_file"])).resolve()
        else:
            index_path = (
                directory / metadata.get("global_index", "indexes/global.usearch")
            ).resolve()
        if not index_path.is_relative_to(directory.resolve()):
            raise RuntimeError("Semantic index path escapes its artifact directory")
        if not index_path.is_file():
            return []
        index = _load_index(index_path, expansion_search)
        overfetch = min(max(200, request.limit * 10, request.limit + 1), 500)
        matches = index.search(query_vector, overfetch)
        keys = [int(key) for key in matches.keys]
        if not keys:
            return []

        placeholders = ",".join("?" for _ in keys)
        rows = connection.execute(
            f"""
            SELECT semantic.semantic_id, semantic.concept,
                   semantic.vector_offset, term.term, term.normalized_term,
                   term.language
            FROM semantic_terms AS semantic
            JOIN lexical_terms AS term ON term.term_id = semantic.term_id
            WHERE semantic.semantic_id IN ({placeholders})
            """,
            keys,
        ).fetchall()
        by_id = {int(row["semantic_id"]): row for row in rows}
        ranked: list[tuple[float, int]] = []
        for semantic_id in keys:
            row = by_id.get(semantic_id)
            if row is None:
                continue
            candidate_offset = int(row["vector_offset"])
            if not 0 <= candidate_offset < row_count:
                raise RuntimeError(
                    "Semantic candidate vector offset is outside the vector matrix"
                )
            candidate = np.asarray(matrix[candidate_offset], dtype=np.float32)
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm == 0.0 or not math.isfinite(candidate_norm):
                continue
            similarity = float((candidate / candidate_norm) @ query_vector)
            if math.isfinite(similarity):
                ranked.append((max(-1.0, min(1.0, similarity)), semantic_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        provenance = {
            "source": metadata.get("source", "ConceptNet Numberbatch"),
            "license": metadata.get("source_license", "CC-BY-SA-4.0"),
            "url": metadata.get("source_url", "https://conceptnet.io/"),
        }
    finally:
        connection.close()

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    seed_id = int(seed["semantic_id"])
    seed_identity = (request.source_language, request.word)
    for similarity, semantic_id in ranked:
        row = by_id.get(semantic_id)
        if row is None or semantic_id == seed_id:
            continue
        language = str(row["language"])
        if request.target_language and language != request.target_language:
            continue
        if request.min_similarity is not None and similarity < request.min_similarity:
            continue
        identity = (language, str(row["normalized_term"]))
        if identity == seed_identity or identity in seen:
            continue
        seen.add(identity)
        results.append(
            {
                "semantic_id": semantic_id,
                "concept": row["concept"],
                "term": row["term"],
                "language": language,
                "similarity": round(similarity, 6),
                "sense_scope": "unsensed",
                "provenance": dict(provenance),
            }
        )
        if len(results) == request.limit:
            break
    return results


class SemanticWorker:
    """Own exactly one lazily-created worker process per MCP server process."""

    def __init__(self, directory: Path, dataset_version: str) -> None:
        self.directory = directory.resolve()
        self.dataset_version = dataset_version
        self._executor: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    @property
    def available(self) -> bool:
        return (
            (self.directory / "mapping.sqlite3").is_file()
            and (self.directory / "indexes" / "global.usearch").is_file()
            and (self.directory / "vectors" / "global.f16").is_file()
        )

    def _pool(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._executor is None:
                context = multiprocessing.get_context("spawn")
                self._executor = ProcessPoolExecutor(max_workers=1, mp_context=context)
            return self._executor

    def search(
        self,
        word: str,
        source_language: str,
        target_language: str | None,
        limit: int,
        min_similarity: float | None,
    ) -> list[dict[str, Any]]:
        if not self.available:
            return []
        request = _SemanticRequest(
            str(self.directory),
            self.dataset_version,
            word,
            source_language,
            target_language,
            limit,
            min_similarity,
        )
        return self._pool().submit(_semantic_search_task, request).result(timeout=30)

    def close(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


class UnavailableSemanticSearch:
    @property
    def available(self) -> bool:
        return False

    def search(
        self,
        word: str,
        source_language: str,
        target_language: str | None,
        limit: int,
        min_similarity: float | None,
    ) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        return None
