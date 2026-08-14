"""Indexed wordplay queries over the compact lexical SQLite schema."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from .normalization import normalize_key, validate_limit

WORDPLAY_MODES = frozenset(
    {"rhyme", "near_rhyme", "sounds_like", "spelled_like", "prefix"}
)

_CMU_PROVENANCE = {
    "source": "CMU Pronouncing Dictionary",
    "license": "CMUdict license",
    "url": "https://github.com/cmusphinx/cmudict",
}

# CMUdict's ARPAbet inventory.  Near-rhyme lookup expands one-token edits from
# the (usually short) source rhyme, then probes the indexed rhyme_key column.
# This keeps the full pronunciation table out of Python.
_CONSONANTS = (
    "B",
    "CH",
    "D",
    "DH",
    "DX",
    "EL",
    "EM",
    "EN",
    "F",
    "G",
    "HH",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "NX",
    "P",
    "Q",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "WH",
    "Y",
    "Z",
    "ZH",
)
_VOWELS = (
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AX",
    "AXR",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IX",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
    "UX",
)
_ARPABET = _CONSONANTS + tuple(
    f"{vowel}{stress}" for vowel in _VOWELS for stress in ("0", "1", "2")
)

# SQLite builds supported by Python guarantee at least 999 host parameters.
_VALUE_CHUNK_SIZE = 800


def _like_pattern(text: str, *, wildcards: bool) -> str:
    """Translate documented wildcards to LIKE while escaping LIKE syntax."""

    translated: list[str] = []
    for character in text:
        if wildcards and character == "?":
            translated.append("_")
        elif wildcards and character == "*":
            translated.append("%")
        elif character in {"%", "_", "\\"}:
            translated.append(f"\\{character}")
        else:
            translated.append(character)
    return "".join(translated)


def _one_edit_rhyme_keys(rhyme_key: str) -> set[str]:
    """Return every non-empty ARPAbet sequence one token edit from a key."""

    tokens = tuple(rhyme_key.split())
    if not tokens:
        return set()

    candidates: set[str] = set()
    for index in range(len(tokens)):
        deleted = tokens[:index] + tokens[index + 1 :]
        if deleted:
            candidates.add(" ".join(deleted))
        for phoneme in _ARPABET:
            if phoneme != tokens[index]:
                candidates.add(" ".join((*tokens[:index], phoneme, *tokens[index + 1 :])))
    for index in range(len(tokens) + 1):
        for phoneme in _ARPABET:
            candidates.add(" ".join((*tokens[:index], phoneme, *tokens[index:])))
    return candidates


class SQLiteWordplaySearch:
    """Read-only, thread-safe wordplay search for lexical schema v2."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).resolve()
        if not self.database_path.is_file():
            raise RuntimeError(f"Lexical database does not exist: {self.database_path}")
        self._connection = sqlite3.connect(
            f"file:{self.database_path.as_posix()}?mode=ro&immutable=1",
            uri=True,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA query_only = ON")
        self._connection.execute("PRAGMA case_sensitive_like = ON")
        self._lock = threading.RLock()
        try:
            self._validate_schema()
        except BaseException:
            self._connection.close()
            raise

    def _validate_schema(self) -> None:
        tables = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        required = {"metadata", "lexical_terms", "pronunciations_words"}
        required.add("wordplay_fts")
        missing = sorted(required - tables)
        if missing:
            raise RuntimeError(
                "Lexical database is missing compact wordplay tables: " + ", ".join(missing)
            )
        schema_row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if schema_row is None or str(schema_row[0]) != "2":
            value = None if schema_row is None else str(schema_row[0])
            raise RuntimeError(f"Compact wordplay requires lexical schema version 2, got {value!r}")

        expected_columns = {
            "lexical_terms": {"term_id", "term", "normalized_term", "language"},
            "pronunciations_words": {"term_id", "phonemes", "rhyme_key"},
        }
        for table, expected in expected_columns.items():
            actual = {
                str(row[1])
                for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if not expected.issubset(actual):
                absent = ", ".join(sorted(expected - actual))
                raise RuntimeError(f"Compact wordplay table {table} is missing columns: {absent}")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteWordplaySearch:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(self, mode: str, text: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return deterministic English wordplay matches, capped at 100."""

        if not isinstance(mode, str) or mode not in WORDPLAY_MODES:
            raise ValueError(f"mode must be one of: {', '.join(sorted(WORDPLAY_MODES))}")
        key = normalize_key(text, field="text", allow_wildcards=mode == "spelled_like")
        limit = validate_limit(limit)

        if mode == "prefix":
            rows = self._prefix_rows(key, limit)
        elif mode == "spelled_like":
            rows = self._pattern_rows(_like_pattern(key, wildcards=True), key, limit)
        else:
            pronunciations = self._source_pronunciations(key)
            if mode == "sounds_like":
                rows = self._value_rows(
                    "phonemes", {phonemes for phonemes, _rhyme in pronunciations}, key, limit
                )
            else:
                source_rhymes = {rhyme for _phonemes, rhyme in pronunciations if rhyme}
                if mode == "rhyme":
                    rows = self._value_rows("rhyme_key", source_rhymes, key, limit)
                else:
                    near_keys: set[str] = set()
                    for rhyme_key in source_rhymes:
                        near_keys.update(_one_edit_rhyme_keys(rhyme_key))
                    near_keys.difference_update(source_rhymes)
                    rows = self._value_rows("rhyme_key", near_keys, key, limit)
        return [self._result(row, mode) for row in rows]

    def _source_pronunciations(self, key: str) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT p.phonemes, p.rhyme_key
                FROM lexical_terms AS t
                JOIN pronunciations_words AS p ON p.term_id = t.term_id
                WHERE t.language = 'en' AND t.normalized_term = ?
                ORDER BY p.phonemes COLLATE BINARY, p.rhyme_key COLLATE BINARY
                """,
                (key,),
            ).fetchall()
        return [(str(row["phonemes"]), str(row["rhyme_key"])) for row in rows]

    def _prefix_rows(self, key: str, limit: int) -> list[tuple[str, str, str]]:
        """Use the contentless FTS5 prefix index, retaining exact string semantics."""

        pattern = _like_pattern(key, wildcards=False) + "%"
        # contentless detail=none FTS5 supports token prefixes, not phrase
        # queries. Multi-token and punctuated headwords retain exact prefix
        # semantics through the indexed language+normalized-term LIKE path.
        if not key.isalnum():
            return self._pattern_rows(pattern, key, limit)
        quoted = key.replace('"', '""')
        query = f'"{quoted}"*'
        try:
            with self._lock:
                rows = self._connection.execute(
                    """
                    SELECT MIN(term.term) AS term, term.normalized_term,
                           MIN(pronunciation.phonemes) AS phonemes
                    FROM wordplay_fts
                    JOIN lexical_terms AS term ON term.term_id = wordplay_fts.rowid
                    JOIN pronunciations_words AS pronunciation
                      ON pronunciation.term_id = term.term_id
                    WHERE wordplay_fts MATCH ?
                      AND term.language = 'en'
                      AND term.normalized_term LIKE ? ESCAPE '\\'
                      AND term.normalized_term <> ?
                    GROUP BY term.normalized_term
                    ORDER BY term.normalized_term COLLATE BINARY,
                             term COLLATE BINARY, phonemes COLLATE BINARY
                    LIMIT ?
                    """,
                    (query, pattern, key, limit),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            raise RuntimeError("Lexical FTS5 wordplay index is invalid") from exc
        return [
            (str(row["term"]), str(row["normalized_term"]), str(row["phonemes"]))
            for row in rows
        ]

    def _pattern_rows(
        self, pattern: str, excluded_key: str, limit: int
    ) -> list[tuple[str, str, str]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT MIN(t.term) AS term, t.normalized_term,
                       MIN(p.phonemes) AS phonemes
                FROM lexical_terms AS t
                JOIN pronunciations_words AS p ON p.term_id = t.term_id
                WHERE t.language = 'en'
                  AND t.normalized_term LIKE ? ESCAPE '\\'
                  AND t.normalized_term <> ?
                GROUP BY t.normalized_term
                ORDER BY t.normalized_term COLLATE BINARY, term COLLATE BINARY,
                         phonemes COLLATE BINARY
                LIMIT ?
                """,
                (pattern, excluded_key, limit),
            ).fetchall()
        return [
            (str(row["term"]), str(row["normalized_term"]), str(row["phonemes"]))
            for row in rows
        ]

    def _value_rows(
        self,
        column: str,
        values: set[str],
        excluded_key: str,
        limit: int,
    ) -> list[tuple[str, str, str]]:
        if not values:
            return []
        if column not in {"phonemes", "rhyme_key"}:
            raise AssertionError(f"Unsupported indexed wordplay column: {column}")

        best_by_normalized: dict[str, tuple[str, str, str]] = {}
        ordered_values = sorted(values)
        for offset in range(0, len(ordered_values), _VALUE_CHUNK_SIZE):
            chunk = ordered_values[offset : offset + _VALUE_CHUNK_SIZE]
            placeholders = ",".join("?" for _value in chunk)
            parameters: list[str | int] = [*chunk, excluded_key, limit]
            with self._lock:
                rows = self._connection.execute(
                    f"""
                    SELECT MIN(t.term) AS term, t.normalized_term,
                           MIN(p.phonemes) AS phonemes
                    FROM pronunciations_words AS p
                    JOIN lexical_terms AS t ON t.term_id = p.term_id
                    WHERE p.{column} IN ({placeholders})
                      AND t.language = 'en'
                      AND t.normalized_term <> ?
                    GROUP BY t.normalized_term
                    ORDER BY t.normalized_term COLLATE BINARY, term COLLATE BINARY,
                             phonemes COLLATE BINARY
                    LIMIT ?
                    """,
                    parameters,
                ).fetchall()
            for row in rows:
                candidate = (
                    str(row["term"]),
                    str(row["normalized_term"]),
                    str(row["phonemes"]),
                )
                current = best_by_normalized.get(candidate[1])
                if current is None or (candidate[0], candidate[2]) < (current[0], current[2]):
                    best_by_normalized[candidate[1]] = candidate

        return sorted(
            best_by_normalized.values(), key=lambda row: (row[1], row[0], row[2])
        )[:limit]

    @staticmethod
    def _result(row: tuple[str, str, str], mode: str) -> dict[str, Any]:
        term, _normalized_term, phonemes = row
        return {
            "term": term,
            "language": "en",
            "mode": mode,
            "phonemes": phonemes,
            "sense_scope": "unsensed",
            "provenance": dict(_CMU_PROVENANCE),
        }
