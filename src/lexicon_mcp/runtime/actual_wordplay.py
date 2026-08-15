"""Indexed actual-wordplay queries over the compact lexical schema v3.

Four deterministic, corpus-backed kinds share this module:

* ``anagram``    -- eligible English headwords with the same normalized
  letter multiset but a different letter sequence.
* ``palindrome`` -- corpus terms whose normalized letters read identically
  in reverse.  The lookup enumerates stored palindromes in
  ``(normalized_letters, term_id)`` index order starting at the query's
  letters (wrapping once at the end of the index), excluding the query
  itself; the query's own palindrome status is reported separately by the
  service as ``input_is_palindrome``.
* ``spoonerism`` -- onset swaps between the CMU pronunciations of exactly
  two English headwords, with a fixed source-alternative cap.
* ``pun``        -- exact CMUdict homophones whose source-native senses
  differ from the query term's; labelled candidates, never jokes.

Every query path is bounded: SQL ``LIMIT`` everywhere, chunked ``IN``
probes, and fixed Python-side caps documented below.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from ..pipeline.wordplay import (
    letter_signature,
    normalized_letters,
    split_arpabet_onset,
)
from .normalization import validate_limit

WORDPLAY_KINDS = frozenset({"anagram", "palindrome", "spoonerism", "pun"})

_CMU_PROVENANCE = {
    "source": "CMU Pronouncing Dictionary",
    "license": "CMUdict license",
    "url": "https://github.com/cmusphinx/cmudict",
}

# Spoonerism resolves at most this many CMU pronunciation alternatives per
# source word before any Cartesian pairing work, bounding every loop.
_SPOONERISM_SOURCE_ALTERNATIVES = 8

# Pun inspects at most this many source-native sense IDs per term.
_PUN_SENSE_FETCH = 8

# SQLite builds supported by Python guarantee at least 999 host parameters.
_VALUE_CHUNK_SIZE = 800

_ANAGRAM_EXPLANATION = "same normalized letters"
_PALINDROME_EXPLANATION = "normalized letters read identically in reverse"
_SPOONERISM_EXPLANATION = "initial consonant clusters exchanged"
_PUN_EXPLANATION = "exact CMUdict homophone with distinct source-native senses"


class SQLiteActualWordplaySearch:
    """Read-only, thread-safe actual-wordplay search for lexical schema v3."""

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
        with self._lock:
            objects = {
                (str(row[0]), str(row[1]))
                for row in self._connection.execute(
                    "SELECT name, type FROM sqlite_master"
                    " WHERE type IN ('table', 'index')"
                ).fetchall()
            }
        names = {name for name, _kind in objects}
        required_tables = {
            "metadata",
            "provenance",
            "lexical_terms",
            "lexical_entries",
            "senses",
            "pronunciations_words",
            "wordplay_terms",
            "pronunciation_onsets",
        }
        missing = sorted(required_tables - names)
        if missing:
            raise RuntimeError(
                "Lexical database is missing actual-wordplay tables: " + ", ".join(missing)
            )
        required_indexes = {
            "wordplay_terms_anagram",
            "wordplay_terms_palindrome",
            "pronunciation_onsets_lookup",
            "pronunciation_onsets_reverse",
        }
        missing_indexes = sorted(required_indexes - names)
        if missing_indexes:
            raise RuntimeError(
                "Lexical database is missing actual-wordplay indexes: "
                + ", ".join(missing_indexes)
            )
        metadata = {
            str(row["key"]): str(row["value"])
            for row in self._connection.execute(
                "SELECT key, value FROM metadata"
                " WHERE key IN ('schema_version', 'wordplay_index_version')"
            ).fetchall()
        }
        if metadata.get("schema_version") != "3":
            raise RuntimeError(
                "Actual wordplay requires lexical schema version 3, got "
                f"{metadata.get('schema_version')!r}"
            )
        if metadata.get("wordplay_index_version") != "1":
            raise RuntimeError(
                "Actual wordplay requires wordplay index version 1, got "
                f"{metadata.get('wordplay_index_version')!r}"
            )
        expected_columns = {
            "wordplay_terms": {
                "term_id",
                "normalized_letters",
                "letter_signature",
                "reverse_letters",
                "is_palindrome",
                "wordplay_eligible",
            },
            "pronunciation_onsets": {"term_id", "phonemes", "onset", "remainder"},
        }
        for table, expected in expected_columns.items():
            actual = {
                str(row[1])
                for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if not expected.issubset(actual):
                absent = ", ".join(sorted(expected - actual))
                raise RuntimeError(
                    f"Actual-wordplay table {table} is missing columns: {absent}"
                )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteActualWordplaySearch:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------ kinds

    def anagram(self, text: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return eligible headwords sharing the letter multiset of ``text``."""

        limit = validate_limit(limit)
        letters = normalized_letters(text)
        if not letters:
            return []
        signature = letter_signature(letters)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT MIN(t.term) AS term, t.normalized_term, w.letter_signature
                FROM wordplay_terms AS w
                JOIN lexical_terms AS t ON t.term_id = w.term_id
                WHERE w.letter_signature = ?
                  AND w.wordplay_eligible = 1
                  AND w.normalized_letters <> ?
                GROUP BY w.normalized_letters
                ORDER BY w.normalized_letters COLLATE BINARY, term COLLATE BINARY
                LIMIT ?
                """,
                (signature, letters, limit),
            ).fetchall()
            return [
                {
                    "term": str(row["term"]),
                    "normalized_term": str(row["normalized_term"]),
                    "signature": str(row["letter_signature"]),
                    "language": "en",
                    "explanation": _ANAGRAM_EXPLANATION,
                    "provenance": self._term_provenance(str(row["normalized_term"])),
                }
                for row in rows
            ]

    def palindrome(self, text: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return stored corpus palindromes, excluding the query's letters.

        Enumeration starts at the query's normalized letters in
        ``(normalized_letters, term_id)`` index order and wraps once at the
        end so a bounded, deterministic page is always available.
        """

        limit = validate_limit(limit)
        letters = normalized_letters(text)
        if len(letters) < 2:
            return []
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT MIN(t.term) AS term, t.normalized_term, w.normalized_letters
                FROM wordplay_terms AS w
                JOIN lexical_terms AS t ON t.term_id = w.term_id
                WHERE w.is_palindrome = 1
                  AND w.normalized_letters <> ?
                  AND w.normalized_letters >= ?
                GROUP BY w.normalized_letters
                ORDER BY w.normalized_letters COLLATE BINARY, term COLLATE BINARY
                LIMIT ?
                """,
                (letters, letters, limit),
            ).fetchall()
            candidates = [dict(row) for row in rows]
            if len(candidates) < limit:
                wrapped = self._connection.execute(
                    """
                    SELECT MIN(t.term) AS term, t.normalized_term, w.normalized_letters
                    FROM wordplay_terms AS w
                    JOIN lexical_terms AS t ON t.term_id = w.term_id
                    WHERE w.is_palindrome = 1
                      AND w.normalized_letters <> ?
                      AND w.normalized_letters < ?
                    GROUP BY w.normalized_letters
                    ORDER BY w.normalized_letters COLLATE BINARY, term COLLATE BINARY
                    LIMIT ?
                    """,
                    (letters, letters, limit - len(candidates)),
                ).fetchall()
                candidates.extend(dict(row) for row in wrapped)
            return [
                {
                    "term": str(row["term"]),
                    "normalized_term": str(row["normalized_term"]),
                    "palindrome_key": str(row["normalized_letters"]),
                    "language": "en",
                    "explanation": _PALINDROME_EXPLANATION,
                    "provenance": self._term_provenance(str(row["normalized_term"])),
                }
                for row in candidates
            ]

    def spoonerism(self, left: str, right: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Swap initial consonant clusters across two English headwords."""

        limit = validate_limit(limit)
        left_rows = self._pronunciation_alternatives(left)
        right_rows = self._pronunciation_alternatives(right)
        results: list[dict[str, Any]] = []
        seen_swaps: set[tuple[str, str]] = set()
        for left_term, left_phonemes in left_rows:
            left_onset, left_remainder = split_arpabet_onset(left_phonemes)
            for right_term, right_phonemes in right_rows:
                if len(results) >= limit:
                    break
                right_onset, right_remainder = split_arpabet_onset(right_phonemes)
                # Empty-to-empty swaps change nothing, and equal onsets would
                # reproduce the source phrase verbatim.
                if (not left_onset and not right_onset) or left_onset == right_onset:
                    continue
                swapped_left = " ".join(
                    part for part in (right_onset, left_remainder) if part
                )
                swapped_right = " ".join(
                    part for part in (left_onset, right_remainder) if part
                )
                if (swapped_left, swapped_right) in seen_swaps:
                    continue
                seen_swaps.add((swapped_left, swapped_right))
                left_resolved = self._resolve_phoneme_term(swapped_left)
                right_resolved = self._resolve_phoneme_term(swapped_right)
                both_lexical = left_resolved is not None and right_resolved is not None
                results.append(
                    {
                        "left": {"term": left_term, "phonemes": left_phonemes},
                        "right": {"term": right_term, "phonemes": right_phonemes},
                        "swapped_left": swapped_left,
                        "swapped_right": swapped_right,
                        "swapped_left_term": left_resolved,
                        "swapped_right_term": right_resolved,
                        "onset_left": left_onset,
                        "onset_right": right_onset,
                        "language": "en",
                        "explanation": _SPOONERISM_EXPLANATION,
                        "lexicality_scope": (
                            "lexical_term" if both_lexical else "generated_candidate"
                        ),
                        "provenance": [dict(_CMU_PROVENANCE)],
                    }
                )
            if len(results) >= limit:
                break
        return results

    def pun(
        self, text: str, *, context_scope: str = "uncontextualized", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return exact-homophone candidates with distinct native senses."""

        limit = validate_limit(limit)
        query_term_ids = self._term_ids(text)
        if not query_term_ids:
            return []
        query_phonemes = self._distinct_phonemes(query_term_ids)
        if not query_phonemes:
            return []
        query_senses = self._term_senses(query_term_ids, _PUN_SENSE_FETCH)
        with self._lock:
            placeholders = ",".join("?" for _value in query_phonemes)
            rows = self._connection.execute(
                f"""
                SELECT MIN(t.term) AS term, t.normalized_term, p.phonemes
                FROM pronunciations_words AS p
                JOIN lexical_terms AS t ON t.term_id = p.term_id
                WHERE p.phonemes IN ({placeholders})
                  AND t.language = 'en'
                  AND t.normalized_term <> ?
                GROUP BY t.normalized_term, p.phonemes
                ORDER BY t.normalized_term COLLATE BINARY, p.phonemes COLLATE BINARY
                LIMIT ?
                """,
                (*query_phonemes, text, limit * 4),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            if len(results) >= limit:
                break
            candidate_term_ids = self._term_ids(str(row["normalized_term"]))
            if not candidate_term_ids:
                continue
            candidate_senses = self._term_senses(candidate_term_ids, _PUN_SENSE_FETCH)
            # A pun candidate needs at least one source-native sense that is
            # not already a sense of the query term.
            if not candidate_senses or set(candidate_senses) & set(query_senses):
                continue
            provenance: list[dict[str, str | None]] = [dict(_CMU_PROVENANCE)]
            for entry in self._entry_provenance(query_term_ids + candidate_term_ids):
                if entry not in provenance:
                    provenance.append(entry)
            results.append(
                {
                    "term": str(row["term"]),
                    "phonemes": str(row["phonemes"]),
                    "query_sense_ids": query_senses,
                    "candidate_sense_ids": candidate_senses,
                    "sound_relation": "homophone",
                    "context_scope": context_scope,
                    "result_class": "candidate",
                    "explanation": _PUN_EXPLANATION,
                    "provenance": provenance,
                }
            )
        return results

    # ------------------------------------------------------------- internals

    def _term_ids(self, normalized_term: str) -> list[int]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT term_id FROM lexical_terms
                WHERE language = 'en' AND normalized_term = ?
                ORDER BY term_id
                """,
                (normalized_term,),
            ).fetchall()
        return [int(row["term_id"]) for row in rows]

    def _distinct_phonemes(self, term_ids: list[int]) -> list[str]:
        phonemes: set[str] = set()
        for offset in range(0, len(term_ids), _VALUE_CHUNK_SIZE):
            chunk = term_ids[offset : offset + _VALUE_CHUNK_SIZE]
            placeholders = ",".join("?" for _value in chunk)
            with self._lock:
                rows = self._connection.execute(
                    f"""
                    SELECT DISTINCT phonemes FROM pronunciations_words
                    WHERE term_id IN ({placeholders})
                    """,
                    chunk,
                ).fetchall()
            phonemes.update(str(row["phonemes"]) for row in rows)
        return sorted(phonemes)

    def _term_senses(self, term_ids: list[int], fetch: int) -> list[str]:
        senses: list[str] = []
        for offset in range(0, len(term_ids), _VALUE_CHUNK_SIZE):
            chunk = term_ids[offset : offset + _VALUE_CHUNK_SIZE]
            placeholders = ",".join("?" for _value in chunk)
            with self._lock:
                rows = self._connection.execute(
                    f"""
                    SELECT s.sense_id
                    FROM senses AS s
                    JOIN lexical_entries AS e ON e.entry_id = s.entry_id
                    WHERE e.term_id IN ({placeholders})
                    ORDER BY s.sense_id COLLATE BINARY
                    LIMIT ?
                    """,
                    (*chunk, fetch),
                ).fetchall()
            senses.extend(str(row["sense_id"]) for row in rows)
        return sorted(set(senses))

    def _entry_provenance(self, term_ids: list[int]) -> list[dict[str, str | None]]:
        entries: list[dict[str, str | None]] = []
        for offset in range(0, len(term_ids), _VALUE_CHUNK_SIZE):
            chunk = term_ids[offset : offset + _VALUE_CHUNK_SIZE]
            placeholders = ",".join("?" for _value in chunk)
            with self._lock:
                rows = self._connection.execute(
                    f"""
                    SELECT DISTINCT p.source, p.source_license, p.source_url
                    FROM lexical_entries AS e
                    JOIN provenance AS p ON p.provenance_id = e.provenance_id
                    WHERE e.term_id IN ({placeholders})
                    ORDER BY p.source COLLATE BINARY, p.source_license COLLATE BINARY
                    """,
                    chunk,
                ).fetchall()
            for row in rows:
                entries.append(
                    {
                        "source": str(row["source"] or "unknown"),
                        "license": str(row["source_license"] or "unknown"),
                        "url": str(row["source_url"]) if row["source_url"] else None,
                    }
                )
        return entries

    def _term_provenance(self, normalized: str) -> list[dict[str, str | None]]:
        """Provenance of a term's stored entries, else its CMU attestation."""

        term_ids = self._term_ids(normalized)
        entries = self._entry_provenance(term_ids)
        if entries:
            return entries
        return [dict(_CMU_PROVENANCE)]

    def _pronunciation_alternatives(self, normalized_term: str) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT MIN(t.term) AS term, p.phonemes
                FROM lexical_terms AS t
                JOIN pronunciations_words AS p ON p.term_id = t.term_id
                WHERE t.language = 'en' AND t.normalized_term = ?
                GROUP BY p.phonemes
                ORDER BY p.phonemes COLLATE BINARY, term COLLATE BINARY
                LIMIT ?
                """,
                (normalized_term, _SPOONERISM_SOURCE_ALTERNATIVES),
            ).fetchall()
        return [(str(row["term"]), str(row["phonemes"])) for row in rows]

    def _resolve_phoneme_term(self, phonemes: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT MIN(t.term) AS term
                FROM pronunciations_words AS p
                JOIN lexical_terms AS t ON t.term_id = p.term_id
                WHERE p.phonemes = ? AND t.language = 'en'
                """,
                (phonemes,),
            ).fetchone()
        if row is None or row["term"] is None:
            return None
        return str(row["term"])
