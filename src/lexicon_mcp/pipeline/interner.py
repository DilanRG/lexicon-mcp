"""Bounded-cache SQLite interning for compact corpus dimensions."""

from __future__ import annotations

import sqlite3
from collections import OrderedDict

from .common import normalize_term


class CorpusInterner:
    """Intern terms and provenance in SQLite without an unbounded Python map."""

    def __init__(self, connection: sqlite3.Connection, *, cache_size: int = 100_000) -> None:
        self.connection = connection
        self.cache_size = cache_size
        self._terms: OrderedDict[tuple[str, str, str], int] = OrderedDict()
        self._provenance: dict[tuple[str, str, str], int] = {}

    def provenance(self, source: str, license_name: str, url: str) -> int:
        key = (source, license_name, url)
        cached = self._provenance.get(key)
        if cached is not None:
            return cached
        self.connection.execute(
            "INSERT OR IGNORE INTO provenance(source,source_license,source_url) VALUES (?,?,?)",
            key,
        )
        row = self.connection.execute(
            """SELECT provenance_id FROM provenance
            WHERE source=? AND source_license=? AND source_url=?""",
            key,
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to intern corpus provenance")
        value = int(row[0])
        self._provenance[key] = value
        return value

    def term(self, term: str, language: str) -> int:
        normalized = normalize_term(term)
        key = (language, normalized, term)
        cached = self._terms.pop(key, None)
        if cached is not None:
            self._terms[key] = cached
            return cached
        inserted = self.connection.execute(
            """INSERT OR IGNORE INTO lexical_terms(term,normalized_term,language)
            VALUES (?,?,?) RETURNING term_id""",
            (term, normalized, language),
        ).fetchone()
        row = inserted
        if row is None:
            row = self.connection.execute(
                """SELECT term_id FROM lexical_terms
                WHERE language=? AND normalized_term=? AND term=?""",
                key,
            ).fetchone()
        if row is None:
            raise RuntimeError("failed to intern lexical term")
        value = int(row[0])
        self._terms[key] = value
        if len(self._terms) > self.cache_size:
            self._terms.popitem(last=False)
        return value
