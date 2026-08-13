"""Idempotently add or remove Lexicon from Open WebUI's persisted tool connections.

The Open WebUI container must be stopped before this script is used. Text configuration
files are intentionally managed separately; this utility touches only the SQLite row.
"""

import argparse
import json
from pathlib import Path

from lexicon_mcp.integration import update_connections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("add", "remove"))
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    changed, ids = update_connections(args.db, present=args.action == "add")
    print(json.dumps({"changed": changed, "connection_ids": ids}, ensure_ascii=False))


if __name__ == "__main__":
    main()
