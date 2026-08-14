from __future__ import annotations

import sqlite3

from lexicon_mcp.pipeline.interner import CorpusInterner
from lexicon_mcp.pipeline.schema import create_lexical_schema


def test_term_interner_returns_inserted_id_and_reuses_duplicates() -> None:
    assert sqlite3.sqlite_version_info >= (3, 35, 0), "SQLite RETURNING is required"
    connection = sqlite3.connect(":memory:")
    create_lexical_schema(connection, "fixture-v1")
    interner = CorpusInterner(connection, cache_size=1)

    first = interner.term("Café", "fr")
    assert first > 0
    assert interner.term("Café", "fr") == first  # hot LRU path

    interner.term("chat", "fr")  # evicts Café
    assert interner.term("Café", "fr") == first  # uncached duplicate SELECT fallback
    assert connection.execute("SELECT COUNT(*) FROM lexical_terms").fetchone()[0] == 2
    connection.close()
