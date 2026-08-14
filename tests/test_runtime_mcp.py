from __future__ import annotations

import json
import os
import queue
import socket
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest
from mcp.server.fastmcp.exceptions import ToolError

import lexicon_mcp.server as server_module
from lexicon_mcp.pipeline.schema import (
    create_lexical_query_indexes,
    create_lexical_schema,
)
from lexicon_mcp.runtime.offline import NetworkDisabledError
from lexicon_mcp.runtime.service import LexiconService
from lexicon_mcp.server import create_mcp


@pytest.fixture()
def service(tmp_path: Path) -> LexiconService:
    database = tmp_path / "lexicon.sqlite3"
    with sqlite3.connect(database) as connection:
        create_lexical_schema(connection, "data-test-v1")
        connection.execute(
            "INSERT INTO provenance VALUES (?, ?, ?, ?)",
            (
                1,
                "Open English WordNet",
                "CC-BY-4.0",
                "https://en-word.net/",
            ),
        )
        connection.execute("INSERT INTO lexical_terms VALUES (1, 'cat', 'cat', 'en')")
        connection.execute(
            "INSERT INTO lexical_entries VALUES (?, ?, ?, ?, ?)",
            ("oewn:entry:cat", 1, "noun", None, 1),
        )
        connection.execute(
            "INSERT INTO senses VALUES (?, ?, ?)",
            ("oewn:cat-n", "oewn:entry:cat", "A small domesticated feline."),
        )
        create_lexical_query_indexes(connection)
        connection.commit()
    instance = LexiconService(database, "data-test-v1")
    yield instance
    instance.close()


@pytest.mark.asyncio
async def test_mcp_exposes_exactly_six_structured_tools_and_no_admin(
    service: LexiconService,
) -> None:
    mcp = create_mcp(service)
    tools = await mcp.list_tools()
    names = {tool.name for tool in tools}
    assert names == {
        "dictionary_lookup",
        "dictionary_synonyms",
        "dictionary_translate",
        "dictionary_relations",
        "dictionary_semantic_neighbors",
        "rhymes",
    }
    assert all(tool.outputSchema is not None for tool in tools)
    assert not any("install" in name or "repair" in name or "rollback" in name for name in names)

    by_name = {tool.name: tool for tool in tools}
    for tool in tools:
        limit_schema = tool.inputSchema["properties"]["limit"]
        assert limit_schema["minimum"] == 1
        assert limit_schema["maximum"] == 100

    lookup_properties = by_name["dictionary_lookup"].inputSchema["properties"]
    assert lookup_properties["word"]["minLength"] == 1
    assert lookup_properties["word"]["maxLength"] == 256
    assert lookup_properties["limit"]["default"] == 8
    for field, default in (
        ("examples_limit", 8),
        ("pronunciations_limit", 8),
        ("translations_limit", 20),
    ):
        assert lookup_properties[field]["default"] == default
        assert lookup_properties[field]["minimum"] == 0
        assert lookup_properties[field]["maximum"] == 100

    synonym_properties = by_name["dictionary_synonyms"].inputSchema["properties"]
    assert synonym_properties["max_senses"] == {
        "default": 20,
        "description": "Maximum source-native lexical senses to inspect.",
        "maximum": 100,
        "minimum": 1,
        "title": "Max Senses",
        "type": "integer",
    }
    assert synonym_properties["unsensed_limit"]["default"] == 5
    assert synonym_properties["unsensed_limit"]["minimum"] == 0
    assert synonym_properties["unsensed_limit"]["maximum"] == 100

    relation_properties = by_name["dictionary_relations"].inputSchema["properties"]
    assert relation_properties["max_depth"]["default"] == 2
    assert relation_properties["max_depth"]["enum"] == [1, 2]
    assert relation_properties["transitive_limit"]["default"] == 5
    assert relation_properties["transitive_limit"]["minimum"] == 0
    assert relation_properties["transitive_limit"]["maximum"] == 100

    translate_properties = by_name["dictionary_translate"].inputSchema["properties"]
    assert translate_properties["max_senses"]["default"] == 100
    assert translate_properties["max_senses"]["minimum"] == 1
    assert translate_properties["max_senses"]["maximum"] == 100
    assert translate_properties["source_language"]["minLength"] == 2
    assert translate_properties["source_language"]["maxLength"] == 256

    semantic_properties = by_name["dictionary_semantic_neighbors"].inputSchema["properties"]
    similarity_number = semantic_properties["min_similarity"]["anyOf"][0]
    assert similarity_number["minimum"] == -1.0
    assert similarity_number["maximum"] == 1.0

    rhyme_properties = by_name["rhymes"].inputSchema["properties"]
    assert rhyme_properties["text"]["minLength"] == 1
    assert rhyme_properties["text"]["maxLength"] == 256
    assert rhyme_properties["mode"]["enum"] == ["exact", "near"]


@pytest.mark.asyncio
async def test_mcp_call_returns_native_structured_content(service: LexiconService) -> None:
    mcp = create_mcp(service)
    result = await mcp.call_tool("dictionary_lookup", {"word": "cat", "language": "en"})
    assert isinstance(result, tuple)
    content, structured = result
    assert content
    assert structured is not None
    assert structured["type"] == "dictionary_lookup"
    assert structured["results"][0]["sense_id"] == "oewn:cat-n"


@pytest.mark.asyncio
async def test_mcp_similarity_accepts_integer_json_numbers_but_rejects_bool(
    service: LexiconService,
) -> None:
    mcp = create_mcp(service)
    result = await mcp.call_tool(
        "dictionary_semantic_neighbors",
        {"word": "cat", "min_similarity": 0},
    )
    assert isinstance(result, tuple)
    _content, structured = result
    assert structured is not None
    assert structured["query"]["min_similarity"] == 0.0

    with pytest.raises(ToolError, match="valid number"):
        await mcp.call_tool(
            "dictionary_semantic_neighbors",
            {"word": "cat", "min_similarity": True},
        )


def test_server_main_creates_event_loop_before_guard_and_denies_external_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_run = anyio.run

    def install_guard() -> None:
        events.append("network_guard")

        def blocked(*_args: object, **_kwargs: object) -> object:
            raise NetworkDisabledError("blocked by test guard")

        monkeypatch.setattr(socket, "create_connection", blocked)

    async def run_stdio_async() -> None:
        events.append("run_stdio_async")
        with pytest.raises(NetworkDisabledError, match="blocked by test guard"):
            socket.create_connection(("example.com", 443))

    def run_with_event_loop(function: Callable[..., object], *args: object) -> None:
        events.append("event_loop")
        original_run(function, *args)

    monkeypatch.setattr(server_module, "install_network_guard", install_guard)
    monkeypatch.setattr(server_module.mcp, "run_stdio_async", run_stdio_async)
    monkeypatch.setattr(anyio, "run", run_with_event_loop)
    server_module.main()

    assert events == ["event_loop", "network_guard", "run_stdio_async"]


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the Windows Proactor loop")
def test_windows_stdio_server_initializes_under_permanent_network_guard() -> None:
    """The guarded server must start after Proactor has made its local socketpair."""

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "lexicon-stdio-test", "version": "1"},
        },
    }
    environment = os.environ.copy()
    environment.pop("LEXICON_DATA_DIR", None)
    process = subprocess.Popen(
        [sys.executable, "-m", "lexicon_mcp.server"],
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
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()

    lines: queue.Queue[str] = queue.Queue(maxsize=1)
    threading.Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True).start()
    try:
        line = lines.get(timeout=10)
    except queue.Empty:
        process.terminate()
        _stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f"guarded stdio server did not initialize:\n{stderr}")
    else:
        process.terminate()
        _stdout, stderr = process.communicate(timeout=5)

    assert line, stderr
    response = json.loads(line)
    assert response["id"] == 1
    assert "result" in response
    assert "NetworkDisabledError" not in stderr
