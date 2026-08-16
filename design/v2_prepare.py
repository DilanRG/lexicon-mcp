"""Precompute the two corpus-wide inputs every pack build needs.

Both are single expensive scans over the v1.1.0 corpus, and both are reused by
every pack afterwards, so they run once here rather than per pack.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexicon_mcp.pipeline.transform import build_term_counts, language_sizes  # noqa: E402

SOURCE = Path("E:/AI/data/lexicon-mcp/versions/data-v1.1.0/lexicon.sqlite3")
WORK = Path("E:/AI/state/lexicon-mcp-v2")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    print("counting languages ...", flush=True)
    sizes = language_sizes(SOURCE)
    (WORK / "language-sizes.json").write_text(
        json.dumps([{"language": item.language, "terms": item.terms} for item in sizes], indent=2),
        encoding="utf-8",
    )
    print(
        f"  {len(sizes):,} languages, {sum(item.terms for item in sizes):,} terms"
        f"  [{time.monotonic() - started:.1f}s]",
        flush=True,
    )

    started = time.monotonic()
    print("materializing full-corpus term counts ...", flush=True)
    cache = build_term_counts(SOURCE, WORK / "term-counts.sqlite3")
    print(
        f"  {cache.stat().st_size / 1024**2:,.1f} MiB  [{time.monotonic() - started:.1f}s]",
        flush=True,
    )
    print("done")


if __name__ == "__main__":
    main()
