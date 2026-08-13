from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from lexicon_mcp.integration import update_connections


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE config ("key" TEXT NOT NULL PRIMARY KEY, '
        "value JSON NOT NULL, updated_at BIGINT)"
    )
    existing = [
        {"info": {"id": "filesystem", "name": "Filesystem"}},
        {"info": {"id": "calculator", "name": "Calculator"}},
    ]
    connection.execute(
        "INSERT INTO config(key, value, updated_at) VALUES (?, ?, 1)",
        ("tool_server.connections", json.dumps(existing)),
    )
    connection.commit()
    connection.close()


def _ids(path: Path) -> list[str]:
    connection = sqlite3.connect(path)
    value = connection.execute(
        "SELECT value FROM config WHERE key='tool_server.connections'"
    ).fetchone()[0]
    connection.close()
    return [item["info"]["id"] for item in json.loads(value)]


def test_add_is_idempotent_and_preserves_existing(tmp_path: Path) -> None:
    database = tmp_path / "webui.db"
    _database(database)
    assert update_connections(database, present=True)[0]
    assert not update_connections(database, present=True)[0]
    assert _ids(database) == ["filesystem", "calculator", "lexicon"]


def test_remove_only_lexicon(tmp_path: Path) -> None:
    database = tmp_path / "webui.db"
    _database(database)
    update_connections(database, present=True)
    assert update_connections(database, present=False)[0]
    assert _ids(database) == ["filesystem", "calculator"]
