from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from mcp.server.fastmcp.exceptions import ToolError

import lexicon_mcp.server as server_module
from lexicon_mcp.pipeline.schema import (
    create_lexical_query_indexes,
    create_lexical_schema,
)
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
        "dictionary_wordplay",
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

    wordplay_properties = by_name["dictionary_wordplay"].inputSchema["properties"]
    assert wordplay_properties["text"]["minLength"] == 1
    assert wordplay_properties["text"]["maxLength"] == 256


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


def test_server_main_installs_network_guard_before_stdio_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        server_module,
        "install_network_guard",
        lambda: events.append("network_guard"),
    )

    def run(*, transport: str) -> None:
        events.append(f"run:{transport}")

    monkeypatch.setattr(server_module.mcp, "run", run)

    server_module.main()

    assert events == ["network_guard", "run:stdio"]
