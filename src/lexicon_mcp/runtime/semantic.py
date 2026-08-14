"""Lazy, process-isolated semantic-neighbour search."""

from __future__ import annotations

import atexit
import math
import multiprocessing
import os
import sqlite3
import threading
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..usearch_compat import open_index_view
from .ann_search import ann_candidate_count
from .offline import deny_network, install_network_guard


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
_WORKER_IDLE_SECONDS = 180.0


def _initialize_semantic_worker() -> None:
    """Bound OpenBLAS and permanently deny networking in the worker process."""

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    install_network_guard()


def _sqlite_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True, check_same_thread=False
    )
    connection.row_factory = sqlite3.Row
    return connection


def _load_index(
    path: Path,
    dimensions: int,
    connectivity: int,
    expansion_add: int,
    expansion_search: int,
    expected_count: int,
) -> Any:
    key = str(path.resolve())
    index = _INDEX_CACHE.pop(key, None)
    if index is not None:
        if len(index) != expected_count:
            index.reset()
            raise RuntimeError(f"Cached semantic index count mismatch: {path}")
        index.expansion_search = expansion_search
        _INDEX_CACHE[key] = index
        return index
    index = open_index_view(
        path,
        dimensions=dimensions,
        metric="cos",
        dtype="i8",
        connectivity=connectivity,
        expansion_add=expansion_add,
        expansion_search=expansion_search,
        expected_count=expected_count,
    )
    _INDEX_CACHE[key] = index
    while len(_INDEX_CACHE) > _INDEX_CACHE_SIZE:
        _evicted_path, evicted = _INDEX_CACHE.popitem(last=False)
        evicted.reset()
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
            connectivity = int(metadata.get("connectivity", "16"))
            expansion_add = int(metadata.get("expansion_add", "256"))
            expansion_search = int(metadata.get("expansion_search", "512"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Semantic metadata has no valid USearch schema") from exc
        if connectivity != 16 or expansion_add != 256 or expansion_search < 512:
            raise RuntimeError("Semantic metadata has an unsupported USearch schema")
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
                """
                SELECT index_file, term_count
                FROM semantic_languages
                WHERE language = ?
                """,
                (request.target_language,),
            ).fetchone()
            if shard is None:
                return []
            index_path = (directory / str(shard["index_file"])).resolve()
            try:
                expected_count = int(shard["term_count"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Semantic index term count must be an integer"
                ) from exc
        else:
            index_path = (
                directory / metadata.get("global_index", "indexes/global.usearch")
            ).resolve()
            expected_count = row_count
        if expected_count < 0:
            raise RuntimeError("Semantic index term count must be non-negative")
        if not index_path.is_relative_to(directory.resolve()):
            raise RuntimeError("Semantic index path escapes its artifact directory")
        if not index_path.is_file():
            return []
        index = _load_index(
            index_path,
            dimensions,
            connectivity,
            expansion_add,
            expansion_search,
            expected_count,
        )
        overfetch = ann_candidate_count(request.limit, expected_count)
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

    def __init__(
        self,
        directory: Path,
        dataset_version: str,
        *,
        _idle_timeout_seconds: float = _WORKER_IDLE_SECONDS,
    ) -> None:
        if not math.isfinite(_idle_timeout_seconds) or _idle_timeout_seconds <= 0:
            raise ValueError("semantic worker idle timeout must be finite and positive")
        self.directory = directory.resolve()
        self.dataset_version = dataset_version
        self._idle_timeout_seconds = _idle_timeout_seconds
        self._executor: ProcessPoolExecutor | None = None
        self._in_flight: set[Future[list[dict[str, Any]]]] = set()
        self._idle_timer: threading.Timer | None = None
        self._closed = False
        self._lock = threading.Lock()
        atexit.register(self.close)

    @property
    def available(self) -> bool:
        return (
            (self.directory / "mapping.sqlite3").is_file()
            and (self.directory / "indexes" / "global.usearch").is_file()
            and (self.directory / "vectors" / "global.f16").is_file()
        )

    def _pool_locked(self) -> ProcessPoolExecutor:
        if self._closed:
            raise RuntimeError("semantic worker is closed")
        if self._executor is None:
            context = multiprocessing.get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=_initialize_semantic_worker,
            )
        return self._executor

    def _cancel_idle_timer_locked(self) -> None:
        timer, self._idle_timer = self._idle_timer, None
        if timer is not None:
            timer.cancel()

    def _schedule_idle_teardown_locked(self, executor: ProcessPoolExecutor) -> None:
        self._cancel_idle_timer_locked()
        timer = threading.Timer(
            self._idle_timeout_seconds,
            self._retire_idle_executor,
            args=(executor,),
        )
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _future_done(self, future: Future[list[dict[str, Any]]]) -> None:
        with self._lock:
            self._in_flight.discard(future)
            executor = self._executor
            if self._closed or executor is None or self._in_flight:
                return
            self._schedule_idle_teardown_locked(executor)

    def _retire_idle_executor(self, executor: ProcessPoolExecutor) -> None:
        current = threading.current_thread()
        with self._lock:
            if (
                self._idle_timer is not current
                or self._closed
                or self._executor is not executor
                or self._in_flight
            ):
                return
            self._idle_timer = None
            # Hold the worker lock until process teardown completes. A concurrent
            # close or query then cannot lose the retiring process handle or
            # briefly overlap it with a replacement worker.
            executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

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
        with self._lock:
            self._cancel_idle_timer_locked()
            executor = self._pool_locked()
            future = executor.submit(_semantic_search_task, request)
            self._in_flight.add(future)
        future.add_done_callback(self._future_done)
        return future.result(timeout=30)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cancel_idle_timer_locked()
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._in_flight.clear()


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
