"""Protocol-independent lexical query service over immutable SQLite artifacts."""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .locator import ActiveDataset, DatasetLocator
from .normalization import (
    normalize_key,
    normalize_language,
    normalize_optional_text,
    sense_scope,
    validate_limit,
)
from .semantic import SemanticSearch, SemanticWorker, UnavailableSemanticSearch
from .wordplay import SQLiteWordplaySearch

RELATIONS = frozenset(
    {
        "antonym",
        "hypernym",
        "hyponym",
        "meronym",
        "holonym",
        "derived_from",
        "etymologically_related",
        "used_for",
        "capable_of",
        "at_location",
        "related",
    }
)
WORDPLAY_MODES = frozenset(
    {"rhyme", "near_rhyme", "sounds_like", "spelled_like", "prefix"}
)
SUPPORTED_SCHEMA_VERSION = "2"

_RELATION_CODES = {
    "synonym": 1,
    "antonym": 2,
    "hypernym": 3,
    "hyponym": 4,
    "meronym": 5,
    "holonym": 6,
    "derived_from": 7,
    "etymologically_related": 8,
    "used_for": 9,
    "capable_of": 10,
    "at_location": 11,
    "related": 12,
}
_RELATION_NAMES = {code: name for name, code in _RELATION_CODES.items()}
_DIRECTION_NAMES = {1: "outbound", 2: "inbound", 3: "symmetric"}
_INVERSE_RELATION_CODES = {
    1: 1,
    2: 2,
    3: 4,
    4: 3,
    5: 6,
    6: 5,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
}
_INVERSE_DIRECTION_CODES = {1: 2, 2: 1, 3: 3}


def _provenance(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "source": str(row["source"] or "unknown"),
        "license": str(row["source_license"] or "unknown"),
        "url": str(row["source_url"]) if row["source_url"] else None,
    }


class LexiconService:
    """Thread-safe queries over one immutable, already-activated dataset."""

    def __init__(
        self,
        database_path: str | Path,
        dataset_version: str,
        *,
        semantic_directory: str | Path | None = None,
        semantic_search: SemanticSearch | None = None,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        if not self.database_path.is_file():
            raise RuntimeError(f"Lexical database does not exist: {self.database_path}")
        if not dataset_version:
            raise ValueError("dataset_version cannot be empty")
        self.dataset_version = dataset_version
        self._connection = sqlite3.connect(
            f"file:{self.database_path.as_posix()}?mode=ro&immutable=1",
            uri=True,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA query_only = ON")
        self._lock = threading.RLock()
        self._tables = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        required_tables = {
            "metadata",
            "provenance",
            "lexical_terms",
            "lexical_entries",
            "senses",
            "examples",
            "pronunciations",
            "translations",
            "synonyms",
            "relations",
            "pronunciations_words",
        }
        missing = required_tables - self._tables
        if missing:
            self._connection.close()
            raise RuntimeError(
                "Lexical database is missing required compact-schema tables: "
                + ", ".join(sorted(missing))
            )
        self._check_dataset_version()

        if semantic_search is not None:
            self._semantic = semantic_search
        elif semantic_directory is not None:
            candidate = Path(semantic_directory)
            self._semantic = (
                SemanticWorker(candidate, dataset_version)
                if candidate.is_dir()
                else UnavailableSemanticSearch()
            )
        else:
            self._semantic = UnavailableSemanticSearch()
        self._wordplay = SQLiteWordplaySearch(self.database_path)

    @classmethod
    def from_active_dataset(cls, dataset: ActiveDataset) -> LexiconService:
        return cls(
            dataset.lexical_database,
            dataset.version,
            semantic_directory=dataset.semantic_directory,
        )

    @classmethod
    def from_locator(cls, locator: DatasetLocator | None = None) -> LexiconService:
        return cls.from_active_dataset((locator or DatasetLocator()).active())

    def _check_dataset_version(self) -> None:
        metadata = {
            str(row["key"]): str(row["value"])
            for row in self._connection.execute(
                "SELECT key, value FROM metadata WHERE key IN ('dataset_version', 'schema_version')"
            ).fetchall()
        }
        if metadata.get("dataset_version") != self.dataset_version:
            self._connection.close()
            raise RuntimeError(
                "Lexical database version does not match the activated dataset version"
            )
        if metadata.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
            self._connection.close()
            raise RuntimeError(
                f"Unsupported lexical schema version {metadata.get('schema_version')!r}; "
                f"this server supports {SUPPORTED_SCHEMA_VERSION!r}"
            )

    def close(self) -> None:
        self._semantic.close()
        self._wordplay.close()
        with self._lock:
            self._connection.close()

    def __enter__(self) -> LexiconService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _sense_rows(
        self,
        word: str,
        language: str,
        part_of_speech: str | None,
        sense_id: str | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        clauses = ["term.normalized_term = ?", "term.language = ?"]
        parameters: list[Any] = [word, language]
        if part_of_speech is not None:
            clauses.append("LOWER(entry.part_of_speech) = ?")
            parameters.append(part_of_speech)
        if sense_id is not None:
            clauses.append("sense.sense_id = ?")
            parameters.append(sense_id)
        parameters.append(limit)
        with self._lock:
            return self._connection.execute(
                f"""
                SELECT sense.sense_id, entry.entry_id, term.term AS word,
                       term.normalized_term AS normalized_word,
                       term.language, entry.part_of_speech, sense.gloss,
                       entry.etymology, provenance.source,
                       provenance.source_license, provenance.source_url
                FROM senses AS sense
                JOIN lexical_entries AS entry ON entry.entry_id = sense.entry_id
                JOIN lexical_terms AS term ON term.term_id = entry.term_id
                JOIN provenance ON provenance.provenance_id = entry.provenance_id
                WHERE {' AND '.join(clauses)}
                ORDER BY CASE WHEN sense.gloss IS NULL THEN 1 ELSE 0 END,
                         sense.sense_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()

    def _dependent_rows(
        self,
        table: str,
        columns: str,
        identifier: str,
        *,
        id_column: str = "sense_id",
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        if table not in self._tables:
            return []
        if id_column not in {"sense_id", "entry_id"}:
            raise RuntimeError(f"unsafe dependent-row key {id_column!r}")
        with self._lock:
            return self._connection.execute(
                f"SELECT {columns} FROM {table} WHERE {id_column} = ? "
                "ORDER BY position LIMIT ?",
                (identifier, limit),
            ).fetchall()

    def _translation_rows(
        self,
        sense_id: str,
        *,
        target_language: str | None = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        clauses = ["translation.sense_id = ?"]
        parameters: list[Any] = [sense_id]
        if target_language is not None:
            clauses.append("term.language = ?")
            parameters.append(target_language)
        parameters.append(limit)
        with self._lock:
            return self._connection.execute(
                f"""
                SELECT term.term, term.normalized_term,
                       term.language AS target_language,
                       translation.part_of_speech, translation.position,
                       provenance.source, provenance.source_license,
                       provenance.source_url
                FROM translations AS translation
                JOIN lexical_terms AS term
                  ON term.term_id = translation.target_term_id
                JOIN provenance
                  ON provenance.provenance_id = translation.provenance_id
                WHERE {' AND '.join(clauses)}
                ORDER BY translation.position, term.language,
                         term.normalized_term, term.term
                LIMIT ?
                """,
                parameters,
            ).fetchall()

    def _synonym_rows(self, sense_id: str, *, limit: int) -> list[sqlite3.Row]:
        with self._lock:
            return self._connection.execute(
                """
                SELECT term.term, term.normalized_term, term.language,
                       synonym.part_of_speech, synonym.position,
                       provenance.source, provenance.source_license,
                       provenance.source_url
                FROM synonyms AS synonym
                JOIN lexical_terms AS term
                  ON term.term_id = synonym.target_term_id
                JOIN provenance
                  ON provenance.provenance_id = synonym.provenance_id
                WHERE synonym.sense_id = ?
                ORDER BY synonym.position, term.language,
                         term.normalized_term, term.term
                LIMIT ?
                """,
                (sense_id, limit),
            ).fetchall()

    def _sense_result(self, row: sqlite3.Row) -> dict[str, Any]:
        sense_id = str(row["sense_id"])
        entry_id = str(row["entry_id"])
        examples = [
            str(item["example"])
            for item in self._dependent_rows("examples", "example, position", sense_id)
        ]
        pronunciations = [
            {"ipa": item["ipa"], "region": item["region"]}
            for item in self._dependent_rows(
                "pronunciations",
                "ipa, region, position",
                entry_id,
                id_column="entry_id",
            )
        ]
        translations = [
            {
                "term": item["term"],
                "language": item["target_language"],
                "part_of_speech": item["part_of_speech"],
                "sense_id": sense_id,
                "sense_scope": sense_scope(sense_id),
                "provenance": _provenance(item),
            }
            for item in self._translation_rows(sense_id)
        ]
        return {
            "sense_id": sense_id,
            "sense_scope": sense_scope(sense_id),
            "word": row["word"],
            "language": row["language"],
            "part_of_speech": row["part_of_speech"],
            "gloss": row["gloss"],
            "examples": examples,
            "pronunciations": pronunciations,
            "etymology": row["etymology"],
            "translations": translations,
            "provenance": _provenance(row),
        }

    def dictionary_lookup(
        self,
        word: str,
        language: str = "en",
        part_of_speech: str | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        original = word.strip() if isinstance(word, str) else word
        key = normalize_key(word)
        language = normalize_language(language)
        part_of_speech = normalize_optional_text(part_of_speech, field="part_of_speech")
        limit = validate_limit(limit)
        results = [
            self._sense_result(row)
            for row in self._sense_rows(key, language, part_of_speech, None, limit)
        ]
        return self._response(
            "dictionary_lookup",
            {
                "word": original,
                "normalized_word": key,
                "language": language,
                "part_of_speech": part_of_speech,
            },
            results,
        )

    def dictionary_synonyms(
        self,
        word: str,
        language: str = "en",
        sense_id: str | None = None,
        part_of_speech: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        original = word.strip() if isinstance(word, str) else word
        key = normalize_key(word)
        language = normalize_language(language)
        part_of_speech = normalize_optional_text(part_of_speech, field="part_of_speech")
        sense_id = self._validate_sense_id(sense_id)
        limit = validate_limit(limit)
        rows = self._sense_rows(key, language, part_of_speech, sense_id, limit)
        groups: list[dict[str, Any]] = []
        remaining = limit
        for row in rows:
            if remaining <= 0:
                break
            row_sense_id = str(row["sense_id"])
            candidates: list[dict[str, Any]] = []
            seen_in_sense: set[tuple[str, str]] = set()
            for item in self._synonym_rows(row_sense_id, limit=remaining):
                identity = (str(item["normalized_term"]), str(item["language"]))
                if identity == (key, language) or identity in seen_in_sense:
                    continue
                seen_in_sense.add(identity)
                candidates.append(
                    {
                        "term": item["term"],
                        "language": item["language"],
                        "part_of_speech": item["part_of_speech"],
                        "sense_id": row_sense_id,
                        "sense_scope": sense_scope(row_sense_id),
                        "provenance": _provenance(item),
                    }
                )
                remaining -= 1
            if candidates:
                groups.append(
                    {
                        "sense_id": row_sense_id,
                        "sense_scope": sense_scope(row_sense_id),
                        "word": row["word"],
                        "language": row["language"],
                        "part_of_speech": row["part_of_speech"],
                        "gloss": row["gloss"],
                        "synonyms": candidates,
                        "provenance": _provenance(row),
                    }
                )
        if sense_id is None and remaining > 0:
            # ConceptNet fallback is an independently unsensed association, so
            # do not suppress it merely because the same display term occurs
            # in a source-scoped sense group above.
            fallback = self._relation_synonyms(
                key, language, part_of_speech, remaining, set()
            )
            if fallback:
                groups.append(fallback)
        return self._response(
            "dictionary_synonyms",
            {
                "word": original,
                "normalized_word": key,
                "language": language,
                "sense_id": sense_id,
                "part_of_speech": part_of_speech,
            },
            groups,
        )

    def _relation_rows(
        self,
        word: str,
        language: str,
        relation_code: int,
        *,
        target_language: str | None,
        source_sense_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Orient compact physical assertions around the requested source term."""

        forward_clauses = [
            "source.normalized_term = ?",
            "source.language = ?",
            "relation.relation_code = ?",
        ]
        forward_parameters: list[Any] = [word, language, relation_code]
        if target_language is not None:
            forward_clauses.append("target.language = ?")
            forward_parameters.append(target_language)
        if source_sense_id is not None:
            forward_clauses.append("relation.source_sense_id = ?")
            forward_parameters.append(source_sense_id)
        forward_parameters.append(limit)

        inverse_code = _INVERSE_RELATION_CODES[relation_code]
        reverse_clauses = [
            "target.normalized_term = ?",
            "target.language = ?",
            "relation.relation_code = ?",
        ]
        reverse_parameters: list[Any] = [word, language, inverse_code]
        if target_language is not None:
            reverse_clauses.append("source.language = ?")
            reverse_parameters.append(target_language)
        if source_sense_id is not None:
            reverse_clauses.append("relation.target_sense_id = ?")
            reverse_parameters.append(source_sense_id)
        reverse_parameters.append(limit)

        select_forward = f"""
            SELECT source.term AS source_term,
                   source.normalized_term AS source_normalized,
                   source.language AS source_language,
                   relation.source_sense_id,
                   target.term AS target_term,
                   target.normalized_term AS target_normalized,
                   target.language AS target_language,
                   relation.target_sense_id,
                   relation.direction_code,
                   provenance.source, provenance.source_license,
                   provenance.source_url
            FROM relations AS relation
            JOIN lexical_terms AS source
              ON source.term_id = relation.source_term_id
            JOIN lexical_terms AS target
              ON target.term_id = relation.target_term_id
            JOIN provenance
              ON provenance.provenance_id = relation.provenance_id
            WHERE {' AND '.join(forward_clauses)}
            ORDER BY target.language, target.normalized_term, target.term,
                     relation.target_sense_id
            LIMIT ?
        """
        select_reverse = f"""
            SELECT target.term AS source_term,
                   target.normalized_term AS source_normalized,
                   target.language AS source_language,
                   relation.target_sense_id AS source_sense_id,
                   source.term AS target_term,
                   source.normalized_term AS target_normalized,
                   source.language AS target_language,
                   relation.source_sense_id AS target_sense_id,
                   relation.direction_code,
                   provenance.source, provenance.source_license,
                   provenance.source_url
            FROM relations AS relation
            JOIN lexical_terms AS source
              ON source.term_id = relation.source_term_id
            JOIN lexical_terms AS target
              ON target.term_id = relation.target_term_id
            JOIN provenance
              ON provenance.provenance_id = relation.provenance_id
            WHERE {' AND '.join(reverse_clauses)}
            ORDER BY source.language, source.normalized_term, source.term,
                     relation.source_sense_id
            LIMIT ?
        """
        with self._lock:
            forward = self._connection.execute(
                select_forward, forward_parameters
            ).fetchall()
            reverse = self._connection.execute(
                select_reverse, reverse_parameters
            ).fetchall()

        oriented: list[dict[str, Any]] = []
        for row in forward:
            item = dict(row)
            item["relation_code"] = relation_code
            oriented.append(item)
        for row in reverse:
            item = dict(row)
            item["relation_code"] = relation_code
            item["direction_code"] = _INVERSE_DIRECTION_CODES[
                int(item["direction_code"])
            ]
            oriented.append(item)
        oriented.sort(
            key=lambda item: (
                str(item["target_language"]),
                str(item["target_normalized"]),
                str(item["target_term"]),
                str(item["target_sense_id"] or ""),
                int(item["direction_code"]),
                str(item["source"]),
            )
        )
        return oriented[:limit]

    def _relation_synonyms(
        self,
        word: str,
        language: str,
        part_of_speech: str | None,
        limit: int,
        seen: set[tuple[str, str]],
    ) -> dict[str, Any] | None:
        if part_of_speech is not None:
            # Relations carry no POS. A requested POS must never receive unscoped fallback data.
            return None
        rows = self._relation_rows(
            word,
            language,
            _RELATION_CODES["synonym"],
            target_language=None,
            source_sense_id=None,
            limit=limit * 2,
        )
        candidates: list[dict[str, Any]] = []
        for row in rows:
            if row["source_sense_id"] is not None:
                continue
            identity = (str(row["target_normalized"]), str(row["target_language"]))
            if identity == (word, language) or identity in seen:
                continue
            seen.add(identity)
            candidates.append(
                {
                    "term": row["target_term"],
                    "language": row["target_language"],
                    "part_of_speech": None,
                    "sense_id": None,
                    "sense_scope": "unsensed",
                    "provenance": _provenance(row),
                }
            )
            if len(candidates) == limit:
                break
        if not candidates:
            return None
        return {
            "sense_id": None,
            "sense_scope": "unsensed",
            "word": word,
            "language": language,
            "part_of_speech": None,
            "gloss": None,
            "synonyms": candidates,
            "provenance": candidates[0]["provenance"],
        }

    def dictionary_translate(
        self,
        word: str,
        source_language: str,
        target_language: str,
        sense_id: str | None = None,
        part_of_speech: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        original = word.strip() if isinstance(word, str) else word
        key = normalize_key(word)
        source_language = normalize_language(source_language, field="source_language")
        target_language = normalize_language(target_language, field="target_language")
        sense_id = self._validate_sense_id(sense_id)
        part_of_speech = normalize_optional_text(part_of_speech, field="part_of_speech")
        limit = validate_limit(limit)
        rows = self._sense_rows(key, source_language, part_of_speech, sense_id, limit)
        groups: list[dict[str, Any]] = []
        remaining = limit
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            if remaining <= 0:
                break
            row_sense_id = str(row["sense_id"])
            translations: list[dict[str, Any]] = []
            for item in self._translation_rows(
                row_sense_id, target_language=target_language, limit=limit
            ):
                identity = (
                    row_sense_id,
                    str(item["normalized_term"]),
                    str(item["target_language"]),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                translations.append(
                    {
                        "term": item["term"],
                        "language": item["target_language"],
                        "part_of_speech": item["part_of_speech"],
                        "sense_id": row_sense_id,
                        "sense_scope": sense_scope(row_sense_id),
                        "provenance": _provenance(item),
                    }
                )
                remaining -= 1
                if remaining == 0:
                    break
            if translations:
                groups.append(
                    {
                        "sense_id": row_sense_id,
                        "sense_scope": sense_scope(row_sense_id),
                        "word": row["word"],
                        "source_language": row["language"],
                        "part_of_speech": row["part_of_speech"],
                        "gloss": row["gloss"],
                        "translations": translations,
                        "provenance": _provenance(row),
                    }
                )
        return self._response(
            "dictionary_translate",
            {
                "word": original,
                "normalized_word": key,
                "source_language": source_language,
                "target_language": target_language,
                "sense_id": sense_id,
                "part_of_speech": part_of_speech,
            },
            groups,
        )

    def dictionary_relations(
        self,
        word: str,
        relation: str,
        language: str = "en",
        target_language: str | None = None,
        sense_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        original = word.strip() if isinstance(word, str) else word
        key = normalize_key(word)
        language = normalize_language(language)
        target_language = (
            normalize_language(target_language, field="target_language")
            if target_language is not None
            else None
        )
        if not isinstance(relation, str) or relation not in RELATIONS:
            raise ValueError(f"relation must be one of: {', '.join(sorted(RELATIONS))}")
        sense_id = self._validate_sense_id(sense_id)
        limit = validate_limit(limit)
        results: list[dict[str, Any]] = []
        rows = self._relation_rows(
            key,
            language,
            _RELATION_CODES[relation],
            target_language=target_language,
            source_sense_id=sense_id,
            limit=limit * 2,
        )
        seen: set[tuple[str, str, str | None, str | None]] = set()
        for row in rows:
            identity = (
                str(row["target_normalized"]),
                str(row["target_language"]),
                row["source_sense_id"],
                row["target_sense_id"],
            )
            if identity in seen:
                continue
            seen.add(identity)
            source_id = row["source_sense_id"]
            relation_code = int(row["relation_code"])
            direction_code = int(row["direction_code"])
            try:
                relation_name = _RELATION_NAMES[relation_code]
                direction_name = _DIRECTION_NAMES[direction_code]
            except KeyError as exc:  # artifact corruption, not model input
                raise RuntimeError("Relation artifact contains an unknown code") from exc
            results.append(
                {
                    "source_term": row["source_term"],
                    "source_language": row["source_language"],
                    "source_sense_id": source_id,
                    "sense_scope": sense_scope(source_id),
                    "relation": relation_name,
                    "target_term": row["target_term"],
                    "target_language": row["target_language"],
                    "target_sense_id": row["target_sense_id"],
                    "direction": direction_name,
                    "provenance": _provenance(row),
                }
            )
            if len(results) == limit:
                break
        return self._response(
            "dictionary_relations",
            {
                "word": original,
                "normalized_word": key,
                "language": language,
                "relation": relation,
                "target_language": target_language,
                "sense_id": sense_id,
            },
            results,
        )

    def dictionary_semantic_neighbors(
        self,
        word: str,
        source_language: str = "en",
        target_language: str | None = None,
        limit: int = 20,
        min_similarity: float | None = None,
    ) -> dict[str, Any]:
        original = word.strip() if isinstance(word, str) else word
        key = normalize_key(word)
        source_language = normalize_language(source_language, field="source_language")
        target_language = (
            normalize_language(target_language, field="target_language")
            if target_language is not None
            else None
        )
        limit = validate_limit(limit)
        if min_similarity is not None:
            if isinstance(min_similarity, bool) or not isinstance(min_similarity, (int, float)):
                raise ValueError("min_similarity must be a finite number")
            min_similarity = float(min_similarity)
            if not math.isfinite(min_similarity) or not -1.0 <= min_similarity <= 1.0:
                raise ValueError("min_similarity must be between -1 and 1")
        results = self._semantic.search(
            key, source_language, target_language, limit, min_similarity
        )
        response = self._response(
            "dictionary_semantic_neighbors",
            {
                "word": original,
                "normalized_word": key,
                "source_language": source_language,
                "target_language": target_language,
                "min_similarity": min_similarity,
            },
            results,
        )
        response["available"] = self._semantic.available
        return response

    def dictionary_wordplay(
        self, mode: str, text: str, limit: int = 20
    ) -> dict[str, Any]:
        if not isinstance(mode, str) or mode not in WORDPLAY_MODES:
            raise ValueError(f"mode must be one of: {', '.join(sorted(WORDPLAY_MODES))}")
        original = text.strip() if isinstance(text, str) else text
        key = normalize_key(text, field="text", allow_wildcards=mode == "spelled_like")
        limit = validate_limit(limit)
        results = self._wordplay.search(mode, key, limit=limit)
        return self._response(
            "dictionary_wordplay",
            {"text": original, "normalized_text": key, "language": "en", "mode": mode},
            results,
        )

    @staticmethod
    def _validate_sense_id(value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("sense_id must be text")
        value = value.strip()
        if not value or len(value) > 256:
            raise ValueError("sense_id must contain between 1 and 256 characters")
        return value

    def _response(
        self, response_type: str, query: dict[str, Any], results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "type": response_type,
            "dataset_version": self.dataset_version,
            "query": query,
            "count": len(results),
            "results": results,
        }
