from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from usearch.index import Index

import lexicon_mcp.runtime.semantic as semantic_module
from lexicon_mcp.pipeline.schema import create_semantic_schema
from lexicon_mcp.runtime.semantic import SemanticWorker, _semantic_search_task, _SemanticRequest


@pytest.fixture()
def semantic_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "semantic"
    (directory / "vectors").mkdir(parents=True)
    (directory / "indexes" / "languages").mkdir(parents=True)
    mapping = directory / "mapping.sqlite3"
    with sqlite3.connect(mapping) as connection:
        create_semantic_schema(connection, "data-test-v1", dimensions=3)
        connection.execute(
            "INSERT INTO provenance VALUES (?, ?, ?, ?)",
            (
                1,
                "ConceptNet Numberbatch",
                "CC-BY-SA-4.0",
                "https://conceptnet.io/",
            ),
        )
        connection.executemany(
            "INSERT INTO lexical_terms VALUES (?, ?, ?, ?)",
            [
                (1, "cat", "cat", "en"),
                (2, "dog", "dog", "en"),
                (3, "Katze", "katze", "de"),
                (4, "chat", "chat", "en"),
                (5, "chat", "chat", "fr"),
            ],
        )
        connection.executemany(
            "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
            [
                ("source", "ConceptNet Numberbatch"),
                ("source_license", "CC-BY-SA-4.0"),
                ("source_url", "https://conceptnet.io/"),
                ("semantic_provenance_id", "1"),
                ("expansion_search", "512"),
            ],
        )
        rows = [
            (1, "/c/en/cat", 1, 0),
            (2, "/c/en/dog", 2, 1),
            (3, "/c/de/Katze", 3, 2),
            (4, "/c/en/chat", 4, 3),
            (5, "/c/fr/chat", 5, 4),
        ]
        connection.executemany(
            "INSERT INTO semantic_terms VALUES (?, ?, ?, ?)", rows
        )
        connection.execute(
            "INSERT INTO semantic_languages VALUES (?, ?, ?)",
            ("de", "indexes/languages/de.usearch", 1),
        )
        connection.commit()

    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.79, 0.21, 0.0],
        ],
        dtype=np.float32,
    )
    vectors.astype(np.float16).tofile(directory / "vectors" / "global.f16")
    global_index = Index(ndim=3, metric="cos", dtype="i8")
    global_index.add(np.asarray([1, 2, 3, 4, 5], dtype=np.uint64), vectors)
    global_index.save(directory / "indexes" / "global.usearch")
    german_index = Index(ndim=3, metric="cos", dtype="i8")
    german_index.add(np.asarray([3], dtype=np.uint64), vectors[[2]])
    german_index.save(directory / "indexes" / "languages" / "de.usearch")
    return directory


def test_semantic_task_uses_dense_ids_mmap_and_language_shard(
    semantic_directory: Path,
) -> None:
    results = _semantic_search_task(
        _SemanticRequest(
            str(semantic_directory), "data-test-v1", "cat", "en", "de", 5, 0.5
        )
    )
    assert len(results) == 1
    assert results[0]["semantic_id"] == 3
    assert results[0]["term"] == "Katze"
    assert results[0]["language"] == "de"
    assert 0.5 <= results[0]["similarity"] <= 1.0
    assert results[0]["provenance"]["license"] == "CC-BY-SA-4.0"
    assert semantic_module._INDEX_CACHE
    assert all(
        index.expansion_search == 512
        for index in semantic_module._INDEX_CACHE.values()
    )


def test_semantic_task_exact_reranks_overfetched_i8_candidates(
    semantic_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ReversedApproximateIndex:
        expansion_search: int | None = None

        def search(self, _query: object, _count: int) -> object:
            return type(
                "Matches",
                (),
                {
                    "keys": np.asarray([2, 3, 1], dtype=np.uint64),
                    "distances": np.asarray([0.1, 0.2, 0.0], dtype=np.float32),
                },
            )()

    approximate = ReversedApproximateIndex()

    def fake_load(_path: Path, expansion_search: int) -> ReversedApproximateIndex:
        approximate.expansion_search = expansion_search
        return approximate

    monkeypatch.setattr(semantic_module, "_load_index", fake_load)
    results = _semantic_search_task(
        _SemanticRequest(
            str(semantic_directory), "data-test-v1", "cat", "en", None, 2, None
        )
    )

    assert approximate.expansion_search == 512
    assert [item["semantic_id"] for item in results] == [3, 2]
    assert results[0]["similarity"] > results[1]["similarity"]


def test_semantic_global_search_preserves_same_spelling_in_different_languages(
    semantic_directory: Path,
) -> None:
    results = _semantic_search_task(
        _SemanticRequest(
            str(semantic_directory), "data-test-v1", "cat", "en", None, 10, None
        )
    )

    chat_results = [item for item in results if item["term"] == "chat"]
    assert {item["language"] for item in chat_results} == {"en", "fr"}


def test_semantic_worker_is_lazy_single_process_and_cleanly_closes(
    semantic_directory: Path,
) -> None:
    worker = SemanticWorker(semantic_directory, "data-test-v1")
    assert worker.available is True
    assert worker._executor is None
    results = worker.search("cat", "en", "de", 5, None)
    assert results[0]["term"] == "Katze"
    executor = worker._executor
    assert executor is not None
    worker.close()
    assert worker._executor is None


def test_semantic_task_rejects_mixed_dataset_versions(semantic_directory: Path) -> None:
    with sqlite3.connect(semantic_directory / "mapping.sqlite3") as connection:
        connection.execute(
            "UPDATE metadata SET value = 'other-version' WHERE key = 'dataset_version'"
        )
        connection.commit()
    with pytest.raises(RuntimeError, match="does not match"):
        _semantic_search_task(
            _SemanticRequest(
                str(semantic_directory), "data-test-v1", "cat", "en", None, 5, None
            )
        )


def test_semantic_task_rejects_incompatible_schema(semantic_directory: Path) -> None:
    with sqlite3.connect(semantic_directory / "mapping.sqlite3") as connection:
        connection.execute(
            "UPDATE metadata SET value = '1' WHERE key = 'schema_version'"
        )
        connection.commit()
    with pytest.raises(RuntimeError, match="unsupported schema version"):
        _semantic_search_task(
            _SemanticRequest(
                str(semantic_directory), "data-test-v1", "cat", "en", None, 5, None
            )
        )
