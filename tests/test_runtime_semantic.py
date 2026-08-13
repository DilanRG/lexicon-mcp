from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from usearch.index import Index

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
        rows = [
            (
                1,
                "/c/en/cat",
                "cat",
                "cat",
                "en",
                0,
                "ConceptNet Numberbatch",
                "CC-BY-SA-4.0",
                "https://conceptnet.io/",
            ),
            (
                2,
                "/c/en/dog",
                "dog",
                "dog",
                "en",
                1,
                "ConceptNet Numberbatch",
                "CC-BY-SA-4.0",
                "https://conceptnet.io/",
            ),
            (
                3,
                "/c/de/Katze",
                "Katze",
                "katze",
                "de",
                2,
                "ConceptNet Numberbatch",
                "CC-BY-SA-4.0",
                "https://conceptnet.io/",
            ),
        ]
        connection.executemany(
            "INSERT INTO semantic_terms VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        connection.execute(
            "INSERT INTO semantic_languages VALUES (?, ?, ?)",
            ("de", "indexes/languages/de.usearch", 1),
        )
        connection.commit()

    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32
    )
    vectors.astype(np.float16).tofile(directory / "vectors" / "global.f16")
    global_index = Index(ndim=3, metric="cos", dtype="i8")
    global_index.add(np.asarray([1, 2, 3], dtype=np.uint64), vectors)
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
