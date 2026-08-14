"""CMU Pronouncing Dictionary importer for English wordplay tools."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .common import iter_text_lines
from .interner import CorpusInterner

_VARIANT = re.compile(r"\(\d+\)$")


def rhyme_key(phonemes: str) -> str:
    """Return CMU phonemes from the final stressed vowel onward."""

    tokens = phonemes.upper().split()
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index][-1:] in {"1", "2"}:
            return " ".join(tokens[index:])
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index][-1:].isdigit():
            return " ".join(tokens[index:])
    return " ".join(tokens[-2:])


def build_cmudict(connection: sqlite3.Connection, path: Path) -> dict[str, int]:
    counts = {"entries": 0, "pronunciations": 0, "malformed": 0}
    interner = CorpusInterner(connection)
    connection.execute("DELETE FROM pronunciations_words")
    connection.commit()
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
        term_id = interner.term(word, "en")
        connection.execute(
            "INSERT OR IGNORE INTO pronunciations_words VALUES (?,?,?)",
            (term_id, phonemes, rhyme_key(phonemes)),
        )
        counts["entries"] += 1
        counts["pronunciations"] += 1
        if counts["pronunciations"] % 50_000 == 0:
            connection.commit()
    connection.commit()
    return counts
