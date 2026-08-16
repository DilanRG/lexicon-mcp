"""Semantic neighbour search across per-language semantic packs.

A schema-1 dataset holds one mapping, one vector matrix and a global index, so a
search seeded anywhere can find neighbours everywhere in a single lookup.  A
schema-2 install holds one pack per language, and the pack holding the seed word
does not contain the index that would find neighbours in another language.

Numberbatch places every language in one vector space, so the seed vector taken
from one pack can legitimately be searched against another pack's index.  This
searches each installed semantic pack, reranks exactly against that pack's own
vectors, and merges under one deterministic order -- reproducing what the global
index would have returned, restricted to what is actually installed.

Restricted, and honest about it: a language whose semantic pack is absent is
reported rather than quietly missing from the neighbours.
"""

from __future__ import annotations

import math
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from ..data.activation import ActivationComponent
from .ann_search import ann_candidate_count
from .router import PackRouter
from .semantic import _load_index

SUPPORTED_PACK_SCHEMA = "4"
DEFAULT_OPEN_INDEXES = 4


class PackSemanticError(RuntimeError):
    """A semantic pack could not be read."""


class PackSemanticSearch:
    """Search installed semantic packs as one logical index."""

    def __init__(
        self,
        router: PackRouter,
        dataset_version: str,
        *,
        max_open_indexes: int = DEFAULT_OPEN_INDEXES,
    ) -> None:
        self._router = router
        self._dataset_version = dataset_version
        self._max_open_indexes = max(1, max_open_indexes)
        self._indexes: OrderedDict[str, Any] = OrderedDict()
        self._closed = False

    @property
    def available(self) -> bool:
        return bool(self._router.installed_languages("semantic"))

    def installed_languages(self) -> tuple[str, ...]:
        return self._router.installed_languages("semantic")

    # ------------------------------------------------------------- artifacts

    def _components(self, language: str) -> dict[str, ActivationComponent] | None:
        """The mapping, vectors and index components serving *language*."""

        activation = self._router.activation
        for pack in activation.packs:
            if pack.capability != "semantic" or language not in pack.languages:
                continue
            found: dict[str, ActivationComponent] = {}
            for component_id in pack.components:
                component = activation.component(component_id)
                if component.path.endswith(".usearch"):
                    found["index"] = component
                elif component.path.endswith(".f16"):
                    found["vectors"] = component
                else:
                    found["mapping"] = component
            if {"mapping", "vectors", "index"} <= set(found):
                return found
            raise PackSemanticError(
                f"semantic pack for {language!r} is missing an artifact"
            )
        return None

    def _mapping(self, path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _metadata(self, connection: sqlite3.Connection) -> dict[str, str]:
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM metadata")
        }
        if metadata.get("dataset_version") != self._dataset_version:
            raise PackSemanticError(
                "semantic pack version does not match the active dataset"
            )
        if metadata.get("schema_version") != SUPPORTED_PACK_SCHEMA:
            raise PackSemanticError("semantic pack has an unsupported schema version")
        if metadata.get("vector_dtype") != "float16":
            raise PackSemanticError("semantic vectors must use float16 storage")
        if metadata.get("index_metric") != "cos" or metadata.get("index_dtype") != "i8":
            raise PackSemanticError("semantic indexes must use cosine/i8 storage")
        return metadata

    def _vectors(self, path: Path, dimensions: int) -> Any:
        size = path.stat().st_size
        row_bytes = dimensions * 2
        if row_bytes <= 0 or size % row_bytes:
            raise PackSemanticError("semantic vector file does not match its dimensions")
        return np.memmap(path, dtype="<f2", mode="r", shape=(size // row_bytes, dimensions))

    def _index(self, language: str, path: Path, metadata: dict[str, str], count: int) -> Any:
        cached = self._indexes.get(language)
        if cached is not None:
            self._indexes.move_to_end(language)
            return cached
        index = _load_index(
            path,
            int(metadata["dimensions"]),
            int(metadata.get("connectivity", "16")),
            int(metadata.get("expansion_add", "256")),
            int(metadata.get("expansion_search", "512")),
            count,
        )
        self._indexes[language] = index
        while len(self._indexes) > self._max_open_indexes:
            self._indexes.popitem(last=False)
        return index

    # ---------------------------------------------------------------- search

    def _seed(self, source_language: str, word: str) -> tuple[Any, int] | None:
        """The normalized seed vector and its semantic id, from its own pack."""

        components = self._components(source_language)
        if components is None:
            return None
        connection = self._mapping(self._router.store.open_path(components["mapping"].sha256))
        try:
            metadata = self._metadata(connection)
            dimensions = int(metadata["dimensions"])
            row = connection.execute(
                """
                SELECT s.semantic_id, s.vector_offset
                FROM semantic_terms AS s
                JOIN lexical_terms AS t ON t.term_id = s.term_id
                WHERE t.normalized_term = ? AND t.language = ?
                ORDER BY s.semantic_id
                LIMIT 1
                """,
                (word, source_language),
            ).fetchone()
            if row is None:
                return None
            matrix = self._vectors(
                self._router.store.open_path(components["vectors"].sha256), dimensions
            )
            offset = int(row["vector_offset"])
            if not 0 <= offset < matrix.shape[0]:
                raise PackSemanticError("seed vector offset is outside its matrix")
            vector = np.asarray(matrix[offset], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm == 0.0 or not math.isfinite(norm):
                raise PackSemanticError("seed vector is not finite and non-zero")
            return vector / norm, int(row["semantic_id"])
        finally:
            connection.close()

    def _search_pack(
        self, language: str, query: Any, limit: int
    ) -> list[tuple[float, int, dict[str, Any]]]:
        """Candidates from one pack, reranked exactly against its own vectors."""

        components = self._components(language)
        if components is None:
            return []
        connection = self._mapping(self._router.store.open_path(components["mapping"].sha256))
        try:
            metadata = self._metadata(connection)
            dimensions = int(metadata["dimensions"])
            if dimensions != int(query.shape[0]):
                raise PackSemanticError("semantic packs disagree on vector dimensions")
            count = int(
                connection.execute("SELECT COUNT(*) FROM semantic_terms").fetchone()[0]
            )
            if not count:
                return []
            index = self._index(
                language,
                self._router.store.open_path(components["index"].sha256),
                metadata,
                count,
            )
            matches = index.search(query, ann_candidate_count(limit, count))
            keys = [int(key) for key in matches.keys]
            if not keys:
                return []
            matrix = self._vectors(
                self._router.store.open_path(components["vectors"].sha256), dimensions
            )
            placeholders = ",".join("?" for _ in keys)
            rows = connection.execute(
                f"""
                SELECT s.semantic_id, s.concept, s.vector_offset,
                       t.term, t.normalized_term, t.language
                FROM semantic_terms AS s
                JOIN lexical_terms AS t ON t.term_id = s.term_id
                WHERE s.semantic_id IN ({placeholders})
                """,
                keys,
            ).fetchall()
            candidates: list[tuple[float, int, dict[str, Any]]] = []
            for row in rows:
                offset = int(row["vector_offset"])
                if not 0 <= offset < matrix.shape[0]:
                    raise PackSemanticError("candidate vector offset is outside its matrix")
                candidate = np.asarray(matrix[offset], dtype=np.float32)
                norm = float(np.linalg.norm(candidate))
                if norm == 0.0 or not math.isfinite(norm):
                    continue
                similarity = float((candidate / norm) @ query)
                if not math.isfinite(similarity):
                    continue
                candidates.append(
                    (
                        max(-1.0, min(1.0, similarity)),
                        int(row["semantic_id"]),
                        {
                            "semantic_id": int(row["semantic_id"]),
                            "concept": row["concept"],
                            "term": row["term"],
                            "normalized_term": row["normalized_term"],
                            "language": row["language"],
                            "source": metadata.get("source", "ConceptNet Numberbatch"),
                            "source_license": metadata.get("source_license", "CC-BY-SA-4.0"),
                            "source_url": metadata.get("source_url", "https://conceptnet.io/"),
                        },
                    )
                )
            return candidates
        finally:
            connection.close()

    def search(
        self,
        word: str,
        source_language: str,
        target_language: str | None,
        limit: int,
        min_similarity: float | None,
    ) -> list[dict[str, Any]]:
        if self._closed or not self.available:
            return []
        seed = self._seed(source_language, word)
        if seed is None:
            return []
        query, seed_id = seed

        if target_language is not None:
            targets: tuple[str, ...] = (target_language,)
        else:
            targets = self.installed_languages()

        pooled: list[tuple[float, int, dict[str, Any]]] = []
        for language in targets:
            pooled.extend(self._search_pack(language, query, limit))

        # Highest similarity first, then semantic id, exactly as the global
        # index path orders its exact rerank.
        pooled.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        seed_identity = (source_language, word)
        for similarity, semantic_id, payload in pooled:
            if semantic_id == seed_id:
                continue
            if min_similarity is not None and similarity < min_similarity:
                continue
            identity = (str(payload["language"]), str(payload["normalized_term"]))
            if identity == seed_identity or identity in seen:
                continue
            seen.add(identity)
            results.append(
                {
                    "semantic_id": payload["semantic_id"],
                    "concept": payload["concept"],
                    "term": payload["term"],
                    "language": payload["language"],
                    "similarity": round(similarity, 6),
                    "sense_scope": "unsensed",
                    "provenance": {
                        "source": payload["source"],
                        "license": payload["source_license"],
                        "url": payload["source_url"],
                    },
                }
            )
            if len(results) == limit:
                break
        return results

    def close(self) -> None:
        self._closed = True
        self._indexes.clear()
