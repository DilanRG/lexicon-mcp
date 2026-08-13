from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lexicon_mcp.pipeline.schema import create_lexical_schema
from lexicon_mcp.runtime.service import LexiconService
from lexicon_mcp.server import create_mcp


@pytest.fixture()
def service(tmp_path: Path) -> LexiconService:
    database = tmp_path / "lexicon.sqlite3"
    with sqlite3.connect(database) as connection:
        create_lexical_schema(connection, "data-test-v1")
        connection.execute(
            """
            INSERT INTO senses(
                sense_id, word, normalized_word, language, part_of_speech, gloss,
                etymology, source, source_license, source_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "oewn:cat-n",
                "cat",
                "cat",
                "en",
                "noun",
                "A small domesticated feline.",
                None,
                "Open English WordNet",
                "CC-BY-4.0",
                "https://en-word.net/",
            ),
        )
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
