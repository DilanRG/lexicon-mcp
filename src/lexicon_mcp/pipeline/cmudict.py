"""CMU Pronouncing Dictionary importer for English wordplay tools."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .common import iter_text_lines, normalize_term

_VARIANT = re.compile(r"\(\d+\)$")


def build_cmudict(connection: sqlite3.Connection, path: Path) -> dict[str, int]:
    counts = {"pronunciations": 0, "malformed": 0}
    for _line_number, line in iter_text_lines(path):
        stripped = line.strip()
        if not stripped or stripped.startswith(";;;") or stripped.startswith("#"):
            continue
        pieces = stripped.split()
        if len(pieces) < 2:
            counts["malformed"] += 1
            continue
        raw_word = pieces[0]
        word = _VARIANT.sub("", raw_word).replace("_", " ")
        phonemes = " ".join(pieces[1:]).upper()
        connection.execute(
            "INSERT OR IGNORE INTO pronunciations_words VALUES (?,?,?)",
            (word, normalize_term(word), phonemes),
        )
        counts["pronunciations"] += 1
        if counts["pronunciations"] % 50_000 == 0:
            connection.commit()
    connection.commit()
    return counts

