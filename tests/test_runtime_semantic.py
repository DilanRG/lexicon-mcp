from __future__ import annotations

import io
import json
import os
import queue
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from usearch.index import Index

import lexicon_mcp.runtime.semantic as semantic_module
from lexicon_mcp.pipeline.schema import (
    create_lexical_query_indexes,
    create_lexical_schema,
    create_semantic_schema,
)
from lexicon_mcp.runtime.offline import NetworkDisabledError
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

    vectors: np.ndarray = np.asarray(
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

    def fake_load(
        _path: Path,
        _dimensions: int,
        _connectivity: int,
        _expansion_add: int,
        expansion_search: int,
        _expected_count: int,
    ) -> ReversedApproximateIndex:
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


def test_semantic_query_views_index_without_metadata_or_restore(
    semantic_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for cached in semantic_module._INDEX_CACHE.values():
        cached.reset()
    semantic_module._INDEX_CACHE.clear()

    original_metadata = Index.metadata

    def mmap_metadata_only(path_or_buffer: Any) -> Any:
        if isinstance(path_or_buffer, (str, os.PathLike)):
            raise AssertionError("large indexes must not use path metadata")
        return original_metadata(path_or_buffer)

    def forbidden_restore(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("large indexes must not use USearch restore")

    monkeypatch.setattr(Index, "metadata", staticmethod(mmap_metadata_only))
    monkeypatch.setattr(Index, "restore", staticmethod(forbidden_restore))
    try:
        results = _semantic_search_task(
            _SemanticRequest(
                str(semantic_directory),
                "data-test-v1",
                "cat",
                "en",
                "de",
                5,
                None,
            )
        )
    finally:
        for cached in semantic_module._INDEX_CACHE.values():
            cached.reset()
        semantic_module._INDEX_CACHE.clear()

    assert [item["term"] for item in results] == ["Katze"]


def test_semantic_query_rejects_shard_count_mismatch(
    semantic_directory: Path,
) -> None:
    for cached in semantic_module._INDEX_CACHE.values():
        cached.reset()
    semantic_module._INDEX_CACHE.clear()
    with sqlite3.connect(semantic_directory / "mapping.sqlite3") as connection:
        connection.execute(
            "UPDATE semantic_languages SET term_count = 2 WHERE language = 'de'"
        )
        connection.commit()

    try:
        with pytest.raises(RuntimeError, match="count_present mismatch"):
            _semantic_search_task(
                _SemanticRequest(
                    str(semantic_directory),
                    "data-test-v1",
                    "cat",
                    "en",
                    "de",
                    5,
                    None,
                )
            )
    finally:
        for cached in semantic_module._INDEX_CACHE.values():
            cached.reset()
        semantic_module._INDEX_CACHE.clear()


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
    assert worker._process is None
    assert worker._idle_timeout_seconds == 180.0
    results = worker.search("cat", "en", "de", 5, None)
    assert results[0]["term"] == "Katze"
    process = worker._process
    assert process is not None
    worker.close()
    assert worker._process is None


def test_semantic_worker_initializer_bounds_openblas_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "32")
    guard_observations: list[str] = []

    def observe_guard_installation() -> None:
        guard_observations.append(os.environ["OPENBLAS_NUM_THREADS"])

    monkeypatch.setattr(
        semantic_module,
        "install_network_guard",
        observe_guard_installation,
    )

    semantic_module._initialize_semantic_worker()

    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert guard_observations == ["1"]


def test_semantic_worker_initializer_blocks_dns_and_socket_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Register the real functions with monkeypatch so the initializer's permanent
    # process-local assignments are restored when this test ends.
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", socket.getaddrinfo)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "32")

    semantic_module._initialize_semantic_worker()

    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    with pytest.raises(NetworkDisabledError):
        socket.getaddrinfo("example.invalid", 443)
    with pytest.raises(NetworkDisabledError):
        socket.create_connection(("example.invalid", 443))
    with socket.socket() as client, pytest.raises(NetworkDisabledError):
        client.connect(("127.0.0.1", 9))
    with socket.socket() as client, pytest.raises(NetworkDisabledError):
        client.connect_ex(("127.0.0.1", 9))


def test_semantic_worker_idle_teardown_tracks_in_flight_and_respawns(
    semantic_directory: Path,
) -> None:
    worker = SemanticWorker(
        semantic_directory,
        "data-test-v1",
        _idle_timeout_seconds=0.2,
    )
    assert worker.search("cat", "en", "de", 5, None)[0]["term"] == "Katze"
    first = worker._process
    assert first is not None
    assert worker._idle_timer is not None

    time.sleep(0.05)
    assert worker.search("cat", "en", "de", 5, None)[0]["term"] == "Katze"
    assert worker._process is first
    time.sleep(0.1)
    assert worker._process is first

    deadline = time.monotonic() + 5
    while worker._process is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker._process is None
    assert first.poll() is not None

    assert worker.search("cat", "en", "de", 5, None)[0]["term"] == "Katze"
    second = worker._process
    assert second is not None and second is not first
    worker.close()
    assert worker._process is None
    assert worker._idle_timer is None
    assert second.poll() is not None
    with pytest.raises(RuntimeError, match="closed"):
        worker.search("cat", "en", None, 5, None)


def test_semantic_worker_timeout_reaps_worker(
    semantic_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingProcess:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    fake = HangingProcess()
    process = cast(subprocess.Popen[bytes], fake)
    worker = SemanticWorker(
        semantic_directory,
        "data-test-v1",
        _query_timeout_seconds=0.01,
    )

    def worker_locked() -> subprocess.Popen[bytes]:
        worker._process = process
        return process

    monkeypatch.setattr(worker, "_worker_locked", worker_locked)

    with pytest.raises(TimeoutError, match="timed out"):
        worker.search("cat", "en", "de", 5, None)

    assert fake.terminated is True
    assert worker._process is None
    worker.close()


@pytest.mark.skipif(sys.platform != "win32", reason="reproduces nested Windows stdio IPC")
def test_windows_stdio_semantic_worker_handles_nested_process_and_unicode(
    tmp_path: Path,
    semantic_directory: Path,
) -> None:
    database = tmp_path / "lexicon.sqlite3"
    with sqlite3.connect(database) as connection:
        create_lexical_schema(connection, "data-test-v1")
        create_lexical_query_indexes(connection)
        connection.commit()

    child_code = f"""
import anyio
from pathlib import Path
from lexicon_mcp.runtime.offline import install_network_guard
from lexicon_mcp.runtime.service import LexiconService
from lexicon_mcp.server import create_mcp

service = LexiconService(
    Path({str(database)!r}),
    "data-test-v1",
    semantic_directory=Path({str(semantic_directory)!r}),
)
mcp = create_mcp(service)

async def run() -> None:
    install_network_guard()
    try:
        await mcp.run_stdio_async()
    finally:
        service.close()

anyio.run(run)
"""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=Path.cwd(),
        env=environment,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    responses: queue.Queue[str] = queue.Queue()

    def read_responses() -> None:
        for line in process.stdout:
            responses.put(line)

    threading.Thread(target=read_responses, daemon=True).start()
    try:
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "semantic-stdio-test", "version": "1"},
            },
        }
        process.stdin.write(json.dumps(initialize) + "\n")
        process.stdin.flush()
        assert json.loads(responses.get(timeout=10))["id"] == 1
        process.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
            )
            + "\n"
        )

        for request_id, word, expected_count in ((2, "cat", 1), (3, "café", 0)):
            call = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": "dictionary_semantic_neighbors",
                    "arguments": {
                        "word": word,
                        "source_language": "en",
                        "target_language": "de",
                        "limit": 5,
                    },
                },
            }
            process.stdin.write(json.dumps(call, ensure_ascii=False) + "\n")
            process.stdin.flush()
            response = json.loads(responses.get(timeout=15))
            assert response["id"] == request_id
            assert response["result"].get("isError", False) is False
            payload = json.loads(response["result"]["content"][0]["text"])
            assert payload["count"] == expected_count
    finally:
        with suppress(OSError):
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    assert process.returncode == 0


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
