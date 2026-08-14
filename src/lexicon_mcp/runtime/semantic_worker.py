"""Internal entrypoint for the isolated semantic JSONL worker."""

from __future__ import annotations

from lexicon_mcp.runtime.semantic import _worker_main

if __name__ == "__main__":
    _worker_main()
