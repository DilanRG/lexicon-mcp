"""Safe helpers for integrating Lexicon with persisted frontend configuration."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

LEXICON_ID = "lexicon"
LEXICON_CONNECTION: dict[str, Any] = {
    "url": "http://host.docker.internal:18010/lexicon",
    "path": "openapi.json",
    "type": "openapi",
    "auth_type": "none",
    "headers": None,
    "key": "",
    "config": {"enable": True},
    "info": {"id": LEXICON_ID, "name": "Lexicon"},
}


def _connection_id(connection: object) -> str | None:
    if not isinstance(connection, dict):
        return None
    info = connection.get("info")
    if not isinstance(info, dict):
        return None
    value = info.get("id")
    return value if isinstance(value, str) else None


def update_connections(database: Path, *, present: bool) -> tuple[bool, list[str]]:
    """Add or remove Lexicon and return ``(changed, resulting_ids)``."""

    if not database.is_file():
        raise FileNotFoundError(f"Open WebUI database not found: {database}")

    connection = sqlite3.connect(database, timeout=30)
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT value FROM config WHERE key = ?", ("tool_server.connections",)
        ).fetchone()
        if row is None:
            raise RuntimeError("Open WebUI config key tool_server.connections is missing")
        value = json.loads(row[0])
        if not isinstance(value, list):
            raise RuntimeError("tool_server.connections is not a JSON array")

        retained = [item for item in value if _connection_id(item) != LEXICON_ID]
        updated = [*retained, LEXICON_CONNECTION] if present else retained
        changed = updated != value
        if changed:
            encoded = json.dumps(updated, ensure_ascii=False, separators=(",", ":"))
            connection.execute(
                "UPDATE config SET value = ?, updated_at = ? WHERE key = ?",
                (encoded, int(time.time()), "tool_server.connections"),
            )
        connection.commit()
        return changed, [item for item in (_connection_id(entry) for entry in updated) if item]
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()

