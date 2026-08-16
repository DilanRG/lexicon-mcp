"""Protocol-independent lexical query service over immutable SQLite artifacts."""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from ..pipeline.wordplay import is_palindrome, normalized_letters
from .actual_wordplay import WORDPLAY_KINDS, SQLiteActualWordplaySearch
from .locator import ActiveComponents, ActiveDataset, DatasetLocator
from .normalization import (
    normalize_key,
    normalize_language,
    normalize_optional_text,
    sense_scope,
    validate_limit,
)
from .pack_queries import relation_rows as pack_relation_rows
from .pack_queries import translation_coverage as pack_translation_coverage
from .pack_queries import translation_rows as pack_translation_rows
from .router import PackRouter
from .semantic import SemanticSearch, SemanticWorker, UnavailableSemanticSearch
from .wordplay import SQLiteWordplaySearch, UnavailableWordplaySearch

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
WORDPLAY_MODES = frozenset({"rhyme", "near_rhyme", "sounds_like", "spelled_like", "prefix"})
SUPPORTED_SCHEMA_VERSION = "3"
SUPPORTED_DATASET_PROFILES = frozenset({"full", "english"})

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

_MAX_QUERY_BUDGET = 100
_MAX_CONTEXT_LENGTH = 512
_RELATION_SCAN_FLOOR = 256
_RELATION_SCAN_CEILING = 512
_RELATION_BATCH_SOURCE_LIMIT = 200
_T = TypeVar("_T")


def _validate_bounded_integer(
    value: int,
    *,
    field: str,
    minimum: int,
    maximum: int = _MAX_QUERY_BUDGET,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


def _validate_allocation(value: int, *, field: str, limit: int) -> int:
    if value == -1:
        raise ValueError(f"{field}=-1 automatic allocation is reserved but unsupported in v1")
    value = _validate_bounded_integer(value, field=field, minimum=0)
    if value > limit:
        raise ValueError(f"{field} must not exceed limit")
    return value


def _validate_fixed_budget(value: int, *, field: str) -> int:
    """Validate a fixed caller budget that is independent of another limit."""

    if value == -1:
        raise ValueError(f"{field}=-1 automatic allocation is reserved but unsupported in v1")
    return _validate_bounded_integer(value, field=field, minimum=0)


def _round_robin_allocate(candidate_lists: list[list[_T]], budget: int) -> list[list[_T]]:
    """Allocate a total budget fairly across ordered candidate groups."""

    selected: list[list[_T]] = [[] for _candidates in candidate_lists]
    position = 0
    remaining = budget
    while remaining > 0:
        added = False
        for index, candidates in enumerate(candidate_lists):
            if position >= len(candidates):
                continue
            selected[index].append(candidates[position])
            remaining -= 1
            added = True
            if remaining == 0:
                break
        if not added:
            break
        position += 1
    return selected


def _relation_scan_limit(limit: int) -> int:
    """Overfetch enough rows for diversity ranking without unbounded scans."""

    return min(_RELATION_SCAN_CEILING, max(_RELATION_SCAN_FLOOR, limit * 8))


def _letters_form_palindrome(key: str) -> bool:
    return is_palindrome(normalized_letters(key))


def _provenance(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "source": str(row["source"] or "unknown"),
        "license": str(row["source_license"] or "unknown"),
        "url": str(row["source_url"]) if row["source_url"] else None,
    }


class LanguageNotInstalled(RuntimeError):
    """A query named a language this install does not serve.

    Raised rather than returning nothing, so a caller can never mistake an
    uninstalled language for a word that does not exist. Tool entry points
    translate it into a typed unavailable response.
    """

    def __init__(self, language: str) -> None:
        super().__init__(f"language is not installed: {language}")
        self.language = language


class LexiconService:
    """Thread-safe queries over one immutable, already-activated dataset."""

    def __init__(
        self,
        database_path: str | Path,
        dataset_version: str,
        *,
        semantic_directory: str | Path | None = None,
        semantic_search: SemanticSearch | None = None,
        dataset_profile: str = "full",
        router: PackRouter | None = None,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        if not self.database_path.is_file():
            raise RuntimeError(f"Lexical database does not exist: {self.database_path}")
        if not dataset_version:
            raise ValueError("dataset_version cannot be empty")
        if router is None and dataset_profile not in SUPPORTED_DATASET_PROFILES:
            raise ValueError("dataset_profile must be full or english")
        self.dataset_version = dataset_version
        self.dataset_profile = dataset_profile
        self._router = router
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
        if router is not None:
            # A lexical pack carries the catalogue instead of the wordplay
            # indexes, which live in their own pack.
            required_tables = (required_tables - {"pronunciations_words"}) | {
                "target_catalogue"
            }
        missing = required_tables - self._tables
        if missing:
            self._connection.close()
            raise RuntimeError(
                "Lexical database is missing required compact-schema tables: "
                + ", ".join(sorted(missing))
            )
        if router is None:
            self._check_dataset_version()
            self._check_dataset_profile()

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
        wordplay_path = self.database_path
        if router is not None:
            component = router.activation.component_for("wordplay", "en")
            wordplay_path = (
                router.store.open_path(component.sha256) if component is not None else None
            )
        try:
            if wordplay_path is None:
                self._wordplay = UnavailableWordplaySearch()
                self._actual_wordplay = UnavailableWordplaySearch()
            else:
                self._wordplay = SQLiteWordplaySearch(wordplay_path)
                self._actual_wordplay = SQLiteActualWordplaySearch(wordplay_path)
        except BaseException:
            self._connection.close()
            raise

    @classmethod
    def from_active_dataset(cls, dataset: ActiveDataset) -> LexiconService:
        return cls(
            dataset.lexical_database,
            dataset.version,
            semantic_directory=dataset.semantic_directory,
            dataset_profile=str(dataset.manifest.get("profile", "full")),
        )

    @classmethod
    def from_locator(cls, locator: DatasetLocator | None = None) -> LexiconService:
        return cls.from_active_dataset((locator or DatasetLocator()).active())

    @classmethod
    def from_components(cls, active: ActiveComponents) -> LexiconService:
        """Serve a schema-2 install, routing each language to its own pack.

        The primary connection is any installed lexical pack: it is only read
        for schema introspection, because every language-scoped query resolves
        its own connection through the router.
        """

        router = active.router()
        languages = router.installed_languages("lexical")
        if not languages:
            router.close()
            raise RuntimeError(
                "The active installation has no lexical languages. Install at least "
                "one with: lexicon-data add-language --languages en"
            )
        component = active.activation.component_for("lexical", languages[0])
        assert component is not None  # a listed language always has a component
        try:
            return cls(
                active.store.open_path(component.sha256),
                active.version,
                dataset_profile="components",
                router=router,
            )
        except BaseException:
            router.close()
            raise

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

    def _check_dataset_profile(self) -> None:
        if self.dataset_profile != "english":
            return
        bounds = [
            self._connection.execute(
                f"SELECT language FROM lexical_terms ORDER BY language {direction} LIMIT 1"
            ).fetchone()
            for direction in ("ASC", "DESC")
        ]
        if any(row is not None and row["language"] != "en" for row in bounds):
            self._connection.close()
            raise RuntimeError(
                "English dataset profile contains a non-English lexical term"
            )

    def _supports_languages(self, *languages: str | None) -> bool:
        return self.dataset_profile != "english" or all(
            language is None or language == "en" for language in languages
        )

    def _unsupported_language_response(
        self,
        response_type: str,
        query: dict[str, Any],
        *,
        candidate_count: bool = False,
    ) -> dict[str, Any]:
        response = self._response(response_type, query, [])
        response["available"] = False
        response["unavailable_reason"] = "english_profile_supports_only_en"
        if candidate_count:
            response["candidate_count"] = 0
        return response

    def close(self) -> None:
        self._semantic.close()
        self._wordplay.close()
        with self._lock:
            self._connection.close()

    def __enter__(self) -> LexiconService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _db(self, language: str | None = None) -> sqlite3.Connection:
        """The connection serving *language*.

        A monolith answers for every language from one file. A schema-2 install
        answers from the pack that owns the language, so every language-scoped
        query resolves its connection here rather than assuming one exists.
        """

        if self._router is None:
            return self._connection
        if language is None:
            raise RuntimeError(
                "a schema-2 query must name its language; routing cannot guess "
                "which pack holds a sense"
            )
        connection = self._router.connection_for("lexical", language)
        if connection is None:
            raise LanguageNotInstalled(language)
        return connection

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
            return self._db(language).execute(
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
                WHERE {" AND ".join(clauses)}
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
        language: str | None = None,
        id_column: str = "sense_id",
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        if table not in self._tables:
            return []
        if id_column not in {"sense_id", "entry_id"}:
            raise RuntimeError(f"unsafe dependent-row key {id_column!r}")
        with self._lock:
            return self._db(language).execute(
                f"SELECT {columns} FROM {table} WHERE {id_column} = ? ORDER BY position LIMIT ?",
                (identifier, limit),
            ).fetchall()

    def _translation_rows(
        self,
        sense_id: str,
        *,
        language: str | None = None,
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
            if self._router is not None:
                return pack_translation_rows(
                    self._db(language),
                    sense_id=sense_id,
                    limit=limit,
                    target_language=target_language,
                )
            return self._db(language).execute(
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
                WHERE {" AND ".join(clauses)}
                ORDER BY translation.position, term.language,
                         term.normalized_term, term.term
                LIMIT ?
                """,
                parameters,
            ).fetchall()

    def _translation_coverage(
        self, sense_ids: list[str], *, language: str | None = None
    ) -> dict[str, tuple[int, int]]:
        """Return distinct-language and row counts for a bounded sense set."""

        if not sense_ids or "translations" not in self._tables:
            return {}
        placeholders = ", ".join("?" for _sense_id in sense_ids)
        with self._lock:
            if self._router is not None:
                return pack_translation_coverage(self._db(language), sense_ids)
            rows = self._db(language).execute(
                f"""
                SELECT translation.sense_id,
                       COUNT(DISTINCT term.language) AS language_count,
                       COUNT(*) AS translation_count
                FROM translations AS translation
                JOIN lexical_terms AS term
                  ON term.term_id = translation.target_term_id
                WHERE translation.sense_id IN ({placeholders})
                GROUP BY translation.sense_id
                """,
                sense_ids,
            ).fetchall()
        return {
            str(row["sense_id"]): (int(row["language_count"]), int(row["translation_count"]))
            for row in rows
        }

    def _lookup_sense_rows(
        self,
        normalized_word: str,
        language: str,
        part_of_speech: str | None,
        *,
        limit: int,
        prefer_translated: bool,
    ) -> list[sqlite3.Row]:
        """Select bounded senses while retaining translation-bearing source scopes."""

        scan_limit = _MAX_QUERY_BUDGET if prefer_translated else limit
        candidates = self._sense_rows(
            normalized_word,
            language,
            part_of_speech,
            None,
            scan_limit,
        )
        if len(candidates) <= limit or not prefer_translated:
            return candidates[:limit]

        sense_ids = [str(row["sense_id"]) for row in candidates]
        coverage = self._translation_coverage(sense_ids, language=language)
        if not coverage:
            return candidates[:limit]

        candidate_positions = {
            str(row["sense_id"]): position for position, row in enumerate(candidates)
        }
        translated = sorted(
            (row for row in candidates if str(row["sense_id"]) in coverage),
            key=lambda row: (
                -coverage[str(row["sense_id"])][0],
                -coverage[str(row["sense_id"])][1],
                candidate_positions[str(row["sense_id"])],
            ),
        )
        translation_reserve = 0 if limit < 2 else max(1, limit // 4)
        selected_ids = {str(row["sense_id"]) for row in translated[:translation_reserve]}
        for row in candidates:
            if len(selected_ids) >= limit:
                break
            selected_ids.add(str(row["sense_id"]))
        return [row for row in candidates if str(row["sense_id"]) in selected_ids][:limit]

    def _synonym_rows(
        self, sense_id: str, *, limit: int, language: str | None = None
    ) -> list[sqlite3.Row]:
        with self._lock:
            return self._db(language).execute(
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

    @staticmethod
    def _sense_result(
        row: sqlite3.Row,
        *,
        examples: list[str],
        pronunciations: list[dict[str, Any]],
        translations: list[dict[str, Any]],
        truncated_fields: list[str],
    ) -> dict[str, Any]:
        sense_id = str(row["sense_id"])
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
            "truncated_fields": truncated_fields,
            "provenance": _provenance(row),
        }

    def dictionary_lookup(
        self,
        word: str,
        language: str = "en",
        part_of_speech: str | None = None,
        limit: int = 8,
        examples_limit: int = 8,
        pronunciations_limit: int = 8,
        translations_limit: int = 20,
    ) -> dict[str, Any]:
        original = word.strip() if isinstance(word, str) else word
        key = normalize_key(word)
        language = normalize_language(language)
        part_of_speech = normalize_optional_text(part_of_speech, field="part_of_speech")
        limit = validate_limit(limit)
        examples_limit = _validate_fixed_budget(examples_limit, field="examples_limit")
        pronunciations_limit = _validate_fixed_budget(
            pronunciations_limit, field="pronunciations_limit"
        )
        translations_limit = _validate_fixed_budget(translations_limit, field="translations_limit")
        query = {
            "word": original,
            "normalized_word": key,
            "language": language,
            "part_of_speech": part_of_speech,
            "limit": limit,
            "examples_limit": examples_limit,
            "pronunciations_limit": pronunciations_limit,
            "translations_limit": translations_limit,
        }
        if not self._supports_languages(language):
            return self._unsupported_language_response("dictionary_lookup", query)
        rows = self._lookup_sense_rows(
            key,
            language,
            part_of_speech,
            limit=limit,
            prefer_translated=translations_limit > 0,
        )

        example_candidates: list[list[str]] = []
        pronunciation_candidates: list[list[dict[str, Any]]] = []
        translation_candidates: list[list[dict[str, Any]]] = []
        for row in rows:
            sense_id = str(row["sense_id"])
            entry_id = str(row["entry_id"])
            example_candidates.append(
                [
                    str(item["example"])
                    for item in self._dependent_rows(
                        "examples",
                        "example, position",
                        sense_id,
                        language=language,
                        limit=examples_limit + 1,
                    )
                ]
            )
            pronunciation_candidates.append(
                [
                    {"ipa": item["ipa"], "region": item["region"]}
                    for item in self._dependent_rows(
                        "pronunciations",
                        "ipa, region, position",
                        entry_id,
                        language=language,
                        id_column="entry_id",
                        limit=pronunciations_limit + 1,
                    )
                ]
            )
            translation_candidates.append(
                [
                    {
                        "term": item["term"],
                        "language": item["target_language"],
                        "part_of_speech": item["part_of_speech"],
                        "sense_id": sense_id,
                        "sense_scope": sense_scope(sense_id),
                        "provenance": _provenance(item),
                    }
                    for item in self._translation_rows(
                        sense_id, limit=translations_limit + 1, language=language
                    )
                ]
            )

        selected_examples = _round_robin_allocate(example_candidates, examples_limit)
        selected_pronunciations = _round_robin_allocate(
            pronunciation_candidates, pronunciations_limit
        )
        selected_translations = _round_robin_allocate(translation_candidates, translations_limit)
        results: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            truncated_fields = [
                field
                for field, candidates, selected in (
                    (
                        "examples",
                        example_candidates[index],
                        selected_examples[index],
                    ),
                    (
                        "pronunciations",
                        pronunciation_candidates[index],
                        selected_pronunciations[index],
                    ),
                    (
                        "translations",
                        translation_candidates[index],
                        selected_translations[index],
                    ),
                )
                if len(candidates) > len(selected)
            ]
            results.append(
                self._sense_result(
                    row,
                    examples=selected_examples[index],
                    pronunciations=selected_pronunciations[index],
                    translations=selected_translations[index],
                    truncated_fields=truncated_fields,
                )
            )
        return self._response(
            "dictionary_lookup",
            query,
            results,
        )

    def dictionary_synonyms(
        self,
        word: str,
        language: str = "en",
        sense_id: str | None = None,
        part_of_speech: str | None = None,
        limit: int = 20,
        max_senses: int = 20,
        unsensed_limit: int = 5,
    ) -> dict[str, Any]:
        original = word.strip() if isinstance(word, str) else word
        key = normalize_key(word)
        language = normalize_language(language)
        part_of_speech = normalize_optional_text(part_of_speech, field="part_of_speech")
        sense_id = self._validate_sense_id(sense_id)
        limit = validate_limit(limit)
        max_senses = _validate_bounded_integer(max_senses, field="max_senses", minimum=1)
        unsensed_limit = _validate_allocation(unsensed_limit, field="unsensed_limit", limit=limit)
        query = {
            "word": original,
            "normalized_word": key,
            "language": language,
            "sense_id": sense_id,
            "part_of_speech": part_of_speech,
            "limit": limit,
            "max_senses": max_senses,
            "unsensed_limit": unsensed_limit,
        }
        if not self._supports_languages(language):
            return self._unsupported_language_response(
                "dictionary_synonyms", query, candidate_count=True
            )
        rows = self._sense_rows(key, language, part_of_speech, sense_id, max_senses)

        group_specs: list[dict[str, Any]] = []
        strict_support: dict[tuple[str, str], int] = {}
        for row in rows:
            row_sense_id = str(row["sense_id"])
            candidates: list[tuple[tuple[str, str], dict[str, Any]]] = []
            seen_in_sense: set[tuple[str, str]] = set()
            for item in self._synonym_rows(
                row_sense_id, limit=_MAX_QUERY_BUDGET, language=language
            ):
                identity = (str(item["normalized_term"]), str(item["language"]))
                if identity == (key, language) or identity in seen_in_sense:
                    continue
                seen_in_sense.add(identity)
                candidates.append(
                    (
                        identity,
                        {
                            "term": item["term"],
                            "language": item["language"],
                            "part_of_speech": item["part_of_speech"],
                            "sense_id": row_sense_id,
                            "sense_scope": sense_scope(row_sense_id),
                            "provenance": _provenance(item),
                        },
                    )
                )
            for identity in seen_in_sense:
                strict_support[identity] = strict_support.get(identity, 0) + 1
            if candidates:
                group_specs.append(
                    {
                        "sense_scope": sense_scope(row_sense_id),
                        "row": row,
                        "candidates": candidates,
                    }
                )

        candidate_lists = [spec["candidates"] for spec in group_specs]
        scoped_indexes = [
            index for index, spec in enumerate(group_specs) if spec["sense_scope"] == "sense"
        ]
        native_unsensed_indexes = [
            index for index, spec in enumerate(group_specs) if spec["sense_scope"] == "unsensed"
        ]

        unsensed_budget = unsensed_limit
        native_unsensed = self._allocate_grouped_synonyms(
            candidate_lists, native_unsensed_indexes, unsensed_budget
        )
        native_unsensed_count = sum(len(items) for items in native_unsensed.values())

        relation_candidates: list[dict[str, Any]] = []
        if sense_id is None and part_of_speech is None and native_unsensed_count < unsensed_budget:
            relation_candidates = self._unsensed_relation_synonym_candidates(
                key, language, strict_support
            )
        relation_slots = min(unsensed_budget - native_unsensed_count, len(relation_candidates))
        scoped_budget = limit - native_unsensed_count - relation_slots
        scoped = self._allocate_grouped_synonyms(candidate_lists, scoped_indexes, scoped_budget)

        selected_identities = {
            identity
            for allocation in (scoped, native_unsensed)
            for items in allocation.values()
            for identity, _candidate in items
        }
        ranked_relation_candidates = self._rank_unsensed_relation_synonyms(
            relation_candidates, selected_identities
        )[:relation_slots]

        groups: list[dict[str, Any]] = []
        for index, spec in enumerate(group_specs):
            selected = [
                *scoped.get(index, ()),
                *native_unsensed.get(index, ()),
            ]
            if not selected:
                continue
            row = spec["row"]
            row_sense_id = str(row["sense_id"])
            groups.append(
                {
                    "sense_id": row_sense_id,
                    "sense_scope": spec["sense_scope"],
                    "word": row["word"],
                    "language": row["language"],
                    "part_of_speech": row["part_of_speech"],
                    "gloss": row["gloss"],
                    "synonyms": [candidate for _identity, candidate in selected],
                    "provenance": _provenance(row),
                }
            )
        if ranked_relation_candidates:
            groups.append(
                {
                    "sense_id": None,
                    "sense_scope": "unsensed",
                    "word": key,
                    "language": language,
                    "part_of_speech": None,
                    "gloss": None,
                    "synonyms": ranked_relation_candidates,
                    "provenance": ranked_relation_candidates[0]["provenance"],
                }
            )
        response = self._response(
            "dictionary_synonyms",
            query,
            groups,
        )
        response["candidate_count"] = sum(len(group["synonyms"]) for group in groups)
        return response

    @staticmethod
    def _allocate_grouped_synonyms(
        candidate_lists: list[list[tuple[tuple[str, str], dict[str, Any]]]],
        group_indexes: list[int],
        budget: int,
    ) -> dict[int, list[tuple[tuple[str, str], dict[str, Any]]]]:
        """Allocate one candidate per sense group before taking variants."""

        selected: dict[int, list[tuple[tuple[str, str], dict[str, Any]]]] = {}
        position = 0
        remaining = budget
        while remaining > 0:
            added = False
            for index in group_indexes:
                candidates = candidate_lists[index]
                if position >= len(candidates):
                    continue
                selected.setdefault(index, []).append(candidates[position])
                remaining -= 1
                added = True
                if remaining == 0:
                    break
            if not added:
                break
            position += 1
        return selected

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
        """Oriented assertions for one source term, ranked."""

        return self._rank_oriented_relation_rows(
            self._oriented_relation_rows(
                word,
                language,
                relation_code,
                target_language=target_language,
                source_sense_id=source_sense_id,
                limit=limit,
            ),
            limit=limit,
        )

    def _oriented_relation_rows(
        self,
        word: str,
        language: str,
        relation_code: int,
        *,
        target_language: str | None,
        source_sense_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Orient compact physical assertions around the requested source term.

        Deliberately unranked: callers that rank a merged set of their own must
        not receive rows that were already ranked and truncated per source.
        """

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
            WITH candidates AS (
                SELECT source.term_id AS source_term_id,
                       source.term AS source_term,
                       source.normalized_term AS source_normalized,
                       source.language AS source_language,
                       relation.source_sense_id,
                       target.term_id AS target_term_id,
                       target.term AS target_term,
                       target.normalized_term AS target_normalized,
                       target.language AS target_language,
                       relation.target_sense_id,
                       relation.direction_code,
                       provenance.provenance_id,
                       provenance.source, provenance.source_license,
                       provenance.source_url,
                       (SELECT COUNT(*)
                          FROM lexical_entries AS target_entry
                         WHERE target_entry.term_id = target.term_id
                       ) AS target_entry_count,
                       (SELECT COUNT(*)
                          FROM senses AS target_sense
                          JOIN lexical_entries AS target_sense_entry
                            ON target_sense_entry.entry_id = target_sense.entry_id
                         WHERE target_sense_entry.term_id = target.term_id
                       ) AS target_sense_count,
                       ROW_NUMBER() OVER (
                           PARTITION BY target.term_id
                           ORDER BY CASE
                                        WHEN relation.source_sense_id IS NULL
                                         AND relation.target_sense_id IS NULL THEN 1
                                        ELSE 0
                                    END,
                                    relation.source_sense_id,
                                    relation.target_sense_id,
                                    provenance.provenance_id,
                                    relation.direction_code
                       ) AS target_variant_rank,
                       1 AS query_orientation
                FROM relations AS relation
                JOIN lexical_terms AS source
                  ON source.term_id = relation.source_term_id
                JOIN lexical_terms AS target
                  ON target.term_id = relation.target_term_id
                JOIN provenance
                  ON provenance.provenance_id = relation.provenance_id
                WHERE {" AND ".join(forward_clauses)}
            )
            SELECT * FROM candidates
            ORDER BY target_variant_rank,
                     target_sense_count DESC, target_entry_count DESC,
                     (LENGTH(target_normalized) -
                      LENGTH(REPLACE(target_normalized, ' ', ''))),
                     LENGTH(target_normalized), target_language,
                     target_normalized, target_term, target_sense_id,
                     provenance_id, direction_code
            LIMIT ?
        """
        select_reverse = f"""
            WITH candidates AS (
                SELECT target.term_id AS source_term_id,
                       target.term AS source_term,
                       target.normalized_term AS source_normalized,
                       target.language AS source_language,
                       relation.target_sense_id AS source_sense_id,
                       source.term_id AS target_term_id,
                       source.term AS target_term,
                       source.normalized_term AS target_normalized,
                       source.language AS target_language,
                       relation.source_sense_id AS target_sense_id,
                       relation.direction_code,
                       provenance.provenance_id,
                       provenance.source, provenance.source_license,
                       provenance.source_url,
                       (SELECT COUNT(*)
                          FROM lexical_entries AS target_entry
                         WHERE target_entry.term_id = source.term_id
                       ) AS target_entry_count,
                       (SELECT COUNT(*)
                          FROM senses AS target_sense
                          JOIN lexical_entries AS target_sense_entry
                            ON target_sense_entry.entry_id = target_sense.entry_id
                         WHERE target_sense_entry.term_id = source.term_id
                       ) AS target_sense_count,
                       ROW_NUMBER() OVER (
                           PARTITION BY source.term_id
                           ORDER BY CASE
                                        WHEN relation.source_sense_id IS NULL
                                         AND relation.target_sense_id IS NULL THEN 1
                                        ELSE 0
                                    END,
                                    relation.target_sense_id,
                                    relation.source_sense_id,
                                    provenance.provenance_id,
                                    relation.direction_code
                       ) AS target_variant_rank,
                       2 AS query_orientation
                FROM relations AS relation
                JOIN lexical_terms AS source
                  ON source.term_id = relation.source_term_id
                JOIN lexical_terms AS target
                  ON target.term_id = relation.target_term_id
                JOIN provenance
                  ON provenance.provenance_id = relation.provenance_id
                WHERE {" AND ".join(reverse_clauses)}
            )
            SELECT * FROM candidates
            ORDER BY target_variant_rank,
                     target_sense_count DESC, target_entry_count DESC,
                     (LENGTH(target_normalized) -
                      LENGTH(REPLACE(target_normalized, ' ', ''))),
                     LENGTH(target_normalized), target_language,
                     target_normalized, target_term, target_sense_id,
                     provenance_id, direction_code
            LIMIT ?
        """
        with self._lock:
            if self._router is None:
                forward = self._connection.execute(select_forward, forward_parameters).fetchall()
                reverse = self._connection.execute(select_reverse, reverse_parameters).fetchall()
            else:
                # A pack's lexical_terms holds only its own headwords, so the
                # monolith's join would silently drop every foreign target. The
                # pack form resolves those through target_catalogue instead.
                pack = self._db(language)
                forward = pack_relation_rows(
                    pack,
                    word=word,
                    language=language,
                    relation_code=relation_code,
                    limit=limit,
                    target_language=target_language,
                    sense_id=source_sense_id,
                )
                reverse = pack_relation_rows(
                    pack,
                    word=word,
                    language=language,
                    relation_code=inverse_code,
                    limit=limit,
                    reverse=True,
                    target_language=target_language,
                    sense_id=source_sense_id,
                )

        oriented: list[dict[str, Any]] = []
        for row in forward:
            item = dict(row)
            item["relation_code"] = relation_code
            oriented.append(item)
        for row in reverse:
            item = dict(row)
            item["relation_code"] = relation_code
            item["direction_code"] = _INVERSE_DIRECTION_CODES[int(item["direction_code"])]
            oriented.append(item)
        return oriented

    @staticmethod
    def _source_prefetch_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        """Mirror the batched statement's source_prefetch_rank ordering."""

        normalized = str(item["target_normalized"])
        return (
            1
            if item["source_sense_id"] is None and item["target_sense_id"] is None
            else 0,
            1 if item["target_sense_id"] is None else 0,
            normalized.count(" "),
            len(normalized),
            str(item["target_language"]),
            normalized,
            str(item["target_term"]),
            str(item["source_sense_id"] or ""),
            str(item["target_sense_id"] or ""),
            int(item["provenance_id"]),
            int(item["direction_code"]),
        )

    @staticmethod
    def _relation_row_rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        normalized = str(item["target_normalized"])
        return (
            int(item.get("target_variant_rank", 1)),
            -int(item.get("target_sense_count", 0)),
            -int(item.get("target_entry_count", 0)),
            normalized.count(" "),
            len(normalized),
            str(item["target_language"]),
            normalized,
            str(item["target_term"]),
            str(item["source_sense_id"] or ""),
            str(item["target_sense_id"] or ""),
            int(item["direction_code"]),
            str(item["source"]),
        )

    def _rank_oriented_relation_rows(
        self, oriented: list[dict[str, Any]], *, limit: int
    ) -> list[dict[str, Any]]:
        buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for item in oriented:
            bucket_key = (
                int(item["provenance_id"]),
                int(item["query_orientation"]),
            )
            buckets.setdefault(bucket_key, []).append(item)
        for bucket_rows in buckets.values():
            bucket_rows.sort(key=self._relation_row_rank_key)

        # Relation sources and physical orientations have complementary
        # coverage. Round-robin prevents one dense source/orientation from
        # consuming the bounded scan before another can contribute a target.
        ranked: list[dict[str, Any]] = []
        position = 0
        bucket_keys = sorted(buckets)
        while len(ranked) < limit:
            added = False
            for bucket_key in bucket_keys:
                bucket_rows = buckets[bucket_key]
                if position < len(bucket_rows):
                    ranked.append(bucket_rows[position])
                    added = True
                    if len(ranked) == limit:
                        break
            if not added:
                break
            position += 1
        return ranked

    def _relation_rows_many(
        self,
        sources: list[tuple[str, str]],
        relation_code: int,
        *,
        target_language: str | None,
        branch_limit: int,
    ) -> list[dict[str, Any]]:
        """Return bounded per-source second hops with a constant number of queries."""

        unique_sources = sorted(set(sources))
        if not unique_sources:
            return []

        by_source: dict[tuple[str, str], list[dict[str, Any]]] = {}
        if self._router is not None:
            # A frontier spans languages, and each language lives in a different
            # pack, so the batch is routed per source rather than issued as one
            # statement. Pack queries are far cheaper than the monolith's, which
            # more than pays for losing the batching.
            # Mirror the batched statement exactly: each orientation is bounded
            # independently by source_prefetch_rank at a prefetch limit, and the
            # two are appended forward-first. Bounding the merged set instead
            # changes which second-hop candidates survive.
            prefetch_limit = min(128, max(32, branch_limit * 4))
            batched: list[dict[str, Any]] = []
            for normalized, source_language in unique_sources:
                pack_connection = self._router.connection_for("lexical", source_language)
                if pack_connection is None:
                    continue
                for reverse in (False, True):
                    code = (
                        _INVERSE_RELATION_CODES[relation_code] if reverse else relation_code
                    )
                    rows = [
                        dict(row)
                        for row in pack_relation_rows(
                            pack_connection,
                            word=normalized,
                            language=source_language,
                            relation_code=code,
                            limit=prefetch_limit,
                            reverse=reverse,
                            target_language=target_language,
                        )
                    ]
                    for item in rows:
                        item["relation_code"] = relation_code
                        if reverse:
                            item["direction_code"] = _INVERSE_DIRECTION_CODES[
                                int(item["direction_code"])
                            ]
                    rows.sort(key=self._source_prefetch_key)
                    batched.extend(rows[:prefetch_limit])
            for item in batched:
                identity = (
                    str(item["source_normalized"]),
                    str(item["source_language"]),
                )
                by_source.setdefault(identity, []).append(item)
        else:
            for offset in range(0, len(unique_sources), _RELATION_BATCH_SOURCE_LIMIT):
                batch = unique_sources[offset : offset + _RELATION_BATCH_SOURCE_LIMIT]
                for item in self._batched_relation_rows(
                    batch,
                    relation_code,
                    target_language=target_language,
                    per_orientation_limit=branch_limit,
                ):
                    identity = (
                        str(item["source_normalized"]),
                        str(item["source_language"]),
                    )
                    by_source.setdefault(identity, []).append(item)

        ranked: list[dict[str, Any]] = []
        for source in unique_sources:
            ranked.extend(
                self._rank_oriented_relation_rows(
                    by_source.get(source, []),
                    limit=branch_limit,
                )
            )
        return ranked

    def _batched_relation_rows(
        self,
        sources: list[tuple[str, str]],
        relation_code: int,
        *,
        target_language: str | None,
        per_orientation_limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch both physical orientations for many normalized source terms."""

        requested_values = ", ".join("(?, ?)" for _source in sources)
        requested_parameters: list[Any] = [
            value for source in sources for value in source
        ]

        def select_orientation(*, reverse: bool) -> tuple[str, list[Any]]:
            if reverse:
                physical_source = "target"
                physical_target = "source"
                source_sense = "relation.target_sense_id"
                target_sense = "relation.source_sense_id"
                query_relation_code = _INVERSE_RELATION_CODES[relation_code]
                query_orientation = 2
            else:
                physical_source = "source"
                physical_target = "target"
                source_sense = "relation.source_sense_id"
                target_sense = "relation.target_sense_id"
                query_relation_code = relation_code
                query_orientation = 1

            target_clause = ""
            parameters = [*requested_parameters, query_relation_code]
            if target_language is not None:
                target_clause = f"AND {physical_target}.language = ?"
                parameters.append(target_language)
            prefetch_limit = min(128, max(32, per_orientation_limit * 4))
            parameters.append(prefetch_limit)
            sql = f"""
                WITH requested(normalized_term, language) AS (
                    VALUES {requested_values}
                ),
                candidates AS (
                    SELECT {physical_source}.term_id AS source_term_id,
                           {physical_source}.term AS source_term,
                           {physical_source}.normalized_term AS source_normalized,
                           {physical_source}.language AS source_language,
                           {source_sense} AS source_sense_id,
                           {physical_target}.term_id AS target_term_id,
                           {physical_target}.term AS target_term,
                           {physical_target}.normalized_term AS target_normalized,
                           {physical_target}.language AS target_language,
                           {target_sense} AS target_sense_id,
                           relation.direction_code,
                           provenance.provenance_id,
                           provenance.source, provenance.source_license,
                           provenance.source_url,
                           ROW_NUMBER() OVER (
                               PARTITION BY {physical_source}.normalized_term,
                                            {physical_source}.language
                               ORDER BY CASE
                                            WHEN relation.source_sense_id IS NULL
                                             AND relation.target_sense_id IS NULL THEN 1
                                            ELSE 0
                                        END,
                                        CASE WHEN {target_sense} IS NULL THEN 1 ELSE 0 END,
                                        (LENGTH({physical_target}.normalized_term) -
                                         LENGTH(REPLACE(
                                             {physical_target}.normalized_term, ' ', ''
                                         ))),
                                        LENGTH({physical_target}.normalized_term),
                                        {physical_target}.language,
                                        {physical_target}.normalized_term,
                                        {physical_target}.term,
                                        {source_sense}, {target_sense},
                                        provenance.provenance_id,
                                        relation.direction_code
                           ) AS source_prefetch_rank,
                           {query_orientation} AS query_orientation
                    FROM requested
                    JOIN lexical_terms AS {physical_source}
                      ON {physical_source}.normalized_term = requested.normalized_term
                     AND {physical_source}.language = requested.language
                    JOIN relations AS relation
                      ON relation.{physical_source}_term_id = {physical_source}.term_id
                     AND relation.relation_code = ?
                    JOIN lexical_terms AS {physical_target}
                      ON {physical_target}.term_id = relation.{physical_target}_term_id
                    JOIN provenance
                      ON provenance.provenance_id = relation.provenance_id
                    WHERE 1 = 1 {target_clause}
                ),
                bounded AS MATERIALIZED (
                    SELECT * FROM candidates WHERE source_prefetch_rank <= ?
                )
                SELECT bounded.*,
                       (SELECT COUNT(*)
                          FROM lexical_entries AS target_entry
                         WHERE target_entry.term_id = bounded.target_term_id
                       ) AS target_entry_count,
                       (SELECT COUNT(*)
                          FROM senses AS target_sense_row
                          JOIN lexical_entries AS target_sense_entry
                            ON target_sense_entry.entry_id = target_sense_row.entry_id
                         WHERE target_sense_entry.term_id = bounded.target_term_id
                       ) AS target_sense_count
                FROM bounded
                ORDER BY source_language, source_normalized, source_prefetch_rank
            """
            return sql, parameters

        oriented: list[dict[str, Any]] = []
        with self._lock:
            for reverse in (False, True):
                sql, parameters = select_orientation(reverse=reverse)
                rows = self._connection.execute(sql, parameters).fetchall()
                orientation_rows: list[dict[str, Any]] = []
                for row in rows:
                    item = dict(row)
                    item["relation_code"] = relation_code
                    if reverse:
                        item["direction_code"] = _INVERSE_DIRECTION_CODES[
                            int(item["direction_code"])
                        ]
                    orientation_rows.append(item)

                variants: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
                for item in orientation_rows:
                    identity = (
                        str(item["source_normalized"]),
                        str(item["source_language"]),
                        int(item["target_term_id"]),
                    )
                    variants.setdefault(identity, []).append(item)
                for variant_rows in variants.values():
                    variant_rows.sort(
                        key=lambda item: (
                            1
                            if item["source_sense_id"] is None
                            and item["target_sense_id"] is None
                            else 0,
                            str(item["source_sense_id"] or ""),
                            str(item["target_sense_id"] or ""),
                            int(item["provenance_id"]),
                            int(item["direction_code"]),
                        )
                    )
                    for rank, item in enumerate(variant_rows, start=1):
                        item["target_variant_rank"] = rank
                oriented.extend(orientation_rows)
        return oriented

    @staticmethod
    def _relation_path_is_scoped(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
        """Require exact sense continuity or a wholly unsensed two-edge path."""

        sense_ids = (
            first["source_sense_id"],
            first["target_sense_id"],
            second["source_sense_id"],
            second["target_sense_id"],
        )
        if all(sense_id is None for sense_id in sense_ids):
            return bool(first["target_term_id"] == second["source_term_id"])
        return all(sense_id is not None for sense_id in sense_ids) and (
            first["target_sense_id"] == second["source_sense_id"]
        )

    def _transitive_frontier_rows(
        self,
        rows: list[dict[str, Any]],
        relation_code: int,
        *,
        budget: int,
    ) -> list[dict[str, Any]]:
        """Select a bounded, sense/source-diverse set of first-hop edges."""

        preferred_direction = 1 if relation_code == _RELATION_CODES["hypernym"] else 2
        diverse_rows = self._target_diverse_relation_rows(
            sorted(
                rows,
                key=lambda row: (
                    0 if int(row["direction_code"]) == preferred_direction else 1,
                    self._relation_row_rank_key(row),
                ),
            )
        )
        scopes_by_target: dict[tuple[str, str], set[bool]] = {}
        for row in diverse_rows:
            identity = (
                str(row["target_normalized"]),
                str(row["target_language"]),
            )
            scopes_by_target.setdefault(identity, set()).add(
                row["source_sense_id"] is not None
            )

        buckets: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        for row in diverse_rows:
            bucket_key = (
                int(row["provenance_id"]),
                int(row["query_orientation"]),
                str(row["source_sense_id"] or ""),
            )
            buckets.setdefault(bucket_key, []).append(row)

        def frontier_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
            target = (
                str(row["target_normalized"]),
                str(row["target_language"]),
            )
            normalized = target[0]
            return (
                0 if len(scopes_by_target[target]) > 1 else 1,
                0 if row["target_sense_id"] is not None else 1,
                -int(row.get("target_sense_count", 0)),
                -int(row.get("target_entry_count", 0)),
                int(row.get("target_variant_rank", 1)),
                normalized.count(" "),
                len(normalized),
                str(row["target_language"]),
                normalized,
                str(row["target_term"]),
                str(row["target_sense_id"] or ""),
            )

        for bucket_rows in buckets.values():
            bucket_rows.sort(key=frontier_key)

        selected: list[dict[str, Any]] = []

        def take_round_robin(
            candidate_buckets: dict[tuple[int, int, str], list[dict[str, Any]]],
        ) -> None:
            position = 0
            bucket_keys = sorted(candidate_buckets)
            while len(selected) < budget:
                added = False
                for bucket_key in bucket_keys:
                    bucket_rows = candidate_buckets[bucket_key]
                    if position < len(bucket_rows):
                        selected.append(bucket_rows[position])
                        added = True
                        if len(selected) == budget:
                            break
                if not added:
                    break
                position += 1

        # Cross-scope corroboration is high-value evidence and must not be
        # crowded out by the many source senses of a polysemous headword.
        corroborated = {
            bucket_key: [
                row
                for row in bucket_rows
                if len(
                    scopes_by_target[
                        (
                            str(row["target_normalized"]),
                            str(row["target_language"]),
                        )
                    ]
                )
                > 1
            ]
            for bucket_key, bucket_rows in buckets.items()
        }
        take_round_robin(corroborated)
        selected_row_ids = {id(row) for row in selected}
        remaining = {
            bucket_key: [row for row in bucket_rows if id(row) not in selected_row_ids]
            for bucket_key, bucket_rows in buckets.items()
        }
        take_round_robin(remaining)
        return selected

    def _transitive_relation_rows(
        self,
        word: str,
        language: str,
        relation_code: int,
        *,
        target_language: str | None,
        source_sense_id: str | None,
        limit: int,
        first_edges: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Expand one exact, homogeneous hierarchy hop to distance two."""

        if relation_code not in {
            _RELATION_CODES["hypernym"],
            _RELATION_CODES["hyponym"],
        }:
            return []
        # Scan a fixed overfetch, then expand a smaller frontier tied to the
        # caller's explicit transitive allocation. Sense/provenance round-robin
        # prevents one polysemous source or dense graph source from dominating.
        frontier_scan_limit = 256
        if first_edges is None:
            first_edges = self._relation_rows(
                word,
                language,
                relation_code,
                target_language=None,
                source_sense_id=source_sense_id,
                limit=frontier_scan_limit,
            )
        else:
            first_edges = first_edges[:frontier_scan_limit]
        first_edges = self._transitive_frontier_rows(
            first_edges,
            relation_code,
            budget=16 if limit <= 5 else min(64, limit * 4),
        )
        second_edges = self._relation_rows_many(
            [
                (str(edge["target_normalized"]), str(edge["target_language"]))
                for edge in first_edges
            ],
            relation_code,
            target_language=target_language,
            branch_limit=max(16, min(32, limit * 2)),
        )
        by_source: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for edge in second_edges:
            identity = (
                str(edge["source_normalized"]),
                str(edge["source_language"]),
            )
            by_source.setdefault(identity, []).append(edge)

        candidates: list[dict[str, Any]] = []
        for first in first_edges:
            intermediate = (
                str(first["target_normalized"]),
                str(first["target_language"]),
            )
            for second in by_source.get(intermediate, ()):
                if first["provenance_id"] != second["provenance_id"]:
                    continue
                if first["direction_code"] != second["direction_code"]:
                    continue
                if not self._relation_path_is_scoped(first, second):
                    continue
                if (
                    str(second["target_normalized"]),
                    str(second["target_language"]),
                ) == (word, language):
                    continue
                item = dict(second)
                item["source_term_id"] = first["source_term_id"]
                item["source_term"] = first["source_term"]
                item["source_normalized"] = first["source_normalized"]
                item["source_language"] = first["source_language"]
                item["source_sense_id"] = first["source_sense_id"]
                item["relation_code"] = relation_code
                item["path_rows"] = (first, second)
                candidates.append(item)

        preferred_direction = 1 if relation_code == _RELATION_CODES["hypernym"] else 2
        sensed_signatures = {
            tuple(
                (
                    str(edge["source_normalized"]),
                    str(edge["source_language"]),
                    str(edge["target_normalized"]),
                    str(edge["target_language"]),
                )
                for edge in item["path_rows"]
            )
            for item in candidates
            if item["source_sense_id"] is not None
        }
        for item in candidates:
            signature = tuple(
                (
                    str(edge["source_normalized"]),
                    str(edge["source_language"]),
                    str(edge["target_normalized"]),
                    str(edge["target_language"]),
                )
                for edge in item["path_rows"]
            )
            item["scope_priority"] = (
                0
                if item["source_sense_id"] is None and signature in sensed_signatures
                else 1
                if item["source_sense_id"] is not None
                else 2
            )
        candidates.sort(
            key=lambda item: (
                int(item["scope_priority"]),
                -int(item.get("target_sense_count", 0)),
                -int(item.get("target_entry_count", 0)),
                str(item["target_language"]),
                str(item["target_normalized"]),
                str(item["target_term"]),
                str(item["target_sense_id"] or ""),
                0 if int(item["direction_code"]) == preferred_direction else 1,
                int(item["direction_code"]),
                str(item["source"]),
                tuple(
                    (
                        str(edge["source_normalized"]),
                        str(edge["source_sense_id"] or ""),
                        str(edge["target_normalized"]),
                        str(edge["target_sense_id"] or ""),
                    )
                    for edge in item["path_rows"]
                ),
            )
        )
        return candidates

    def _unsensed_relation_synonym_candidates(
        self,
        word: str,
        language: str,
        strict_support: Mapping[tuple[str, str], int],
    ) -> list[dict[str, Any]]:
        """Collect same-language, wholly-unsensed relation candidates."""

        rows = self._relation_rows(
            word,
            language,
            _RELATION_CODES["synonym"],
            target_language=language,
            source_sense_id=None,
            limit=_RELATION_SCAN_CEILING,
        )
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            if row["source_sense_id"] is not None or row["target_sense_id"] is not None:
                continue
            identity = (str(row["target_normalized"]), str(row["target_language"]))
            if identity == (word, language):
                continue
            candidate = grouped.get(identity)
            if candidate is None:
                grouped[identity] = {
                    "identity": identity,
                    "row": row,
                    "orientations": {int(row["query_orientation"])},
                    "strict_support": strict_support.get(identity, 0),
                    "lexical_affinity": (
                        identity[0].startswith(word) or word.startswith(identity[0])
                    ),
                }
                continue
            candidate["orientations"].add(int(row["query_orientation"]))
            if self._relation_row_rank_key(row) < self._relation_row_rank_key(candidate["row"]):
                candidate["row"] = row
        return list(grouped.values())

    def _rank_unsensed_relation_synonyms(
        self,
        candidates: list[dict[str, Any]],
        selected_strict: set[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        """Rank corroborated additions, then balance one-way orientations."""

        def candidate_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
            row = candidate["row"]
            target_sense_count = int(row.get("target_sense_count", 0))
            support = int(candidate["strict_support"])
            return (
                0 if target_sense_count > 0 else 1,
                0 if candidate["lexical_affinity"] else 1,
                0 if support > 0 else 1,
                -target_sense_count,
                -support,
                *self._relation_row_rank_key(row),
            )

        ranked: list[dict[str, Any]] = []
        for duplicates_strict in (False, True):
            partition = [
                candidate
                for candidate in candidates
                if (candidate["identity"] in selected_strict) is duplicates_strict
            ]
            corroborated = [
                candidate for candidate in partition if len(candidate["orientations"]) > 1
            ]
            corroborated.sort(key=candidate_key)
            ranked.extend(corroborated)

            one_way_buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for candidate in partition:
                if len(candidate["orientations"]) > 1:
                    continue
                row = candidate["row"]
                bucket_key = (
                    next(iter(candidate["orientations"])),
                    int(row["provenance_id"]),
                )
                one_way_buckets.setdefault(bucket_key, []).append(candidate)
            for bucket_rows in one_way_buckets.values():
                bucket_rows.sort(key=candidate_key)
            position = 0
            bucket_keys = sorted(one_way_buckets)
            while True:
                added = False
                for bucket_key in bucket_keys:
                    bucket_rows = one_way_buckets[bucket_key]
                    if position < len(bucket_rows):
                        ranked.append(bucket_rows[position])
                        added = True
                if not added:
                    break
                position += 1

        results: list[dict[str, Any]] = []
        for candidate in ranked:
            row = candidate["row"]
            results.append(
                {
                    "term": row["target_term"],
                    "language": row["target_language"],
                    "part_of_speech": None,
                    "sense_id": None,
                    "sense_scope": "unsensed",
                    "provenance": _provenance(row),
                }
            )
        return results

    def dictionary_translate(
        self,
        word: str,
        source_language: str,
        target_language: str,
        sense_id: str | None = None,
        part_of_speech: str | None = None,
        limit: int = 20,
        max_senses: int = 100,
    ) -> dict[str, Any]:
        original = word.strip() if isinstance(word, str) else word
        key = normalize_key(word)
        source_language = normalize_language(source_language, field="source_language")
        target_language = normalize_language(target_language, field="target_language")
        sense_id = self._validate_sense_id(sense_id)
        part_of_speech = normalize_optional_text(part_of_speech, field="part_of_speech")
        limit = validate_limit(limit)
        max_senses = _validate_bounded_integer(max_senses, field="max_senses", minimum=1)
        query = {
            "word": original,
            "normalized_word": key,
            "source_language": source_language,
            "target_language": target_language,
            "sense_id": sense_id,
            "part_of_speech": part_of_speech,
            "limit": limit,
            "max_senses": max_senses,
        }
        if not self._supports_languages(source_language, target_language):
            return self._unsupported_language_response(
                "dictionary_translate", query, candidate_count=True
            )
        rows = self._sense_rows(key, source_language, part_of_speech, sense_id, max_senses)
        group_specs: list[tuple[sqlite3.Row, list[dict[str, Any]]]] = []
        for row in rows:
            row_sense_id = str(row["sense_id"])
            translations: list[dict[str, Any]] = []
            seen_in_sense: set[tuple[str, str]] = set()
            for item in self._translation_rows(
                row_sense_id,
                target_language=target_language,
                limit=limit,
                language=source_language,
            ):
                identity = (
                    str(item["normalized_term"]),
                    str(item["target_language"]),
                )
                if identity in seen_in_sense:
                    continue
                seen_in_sense.add(identity)
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
            if translations:
                group_specs.append((row, translations))

        allocated = _round_robin_allocate(
            [translations for _row, translations in group_specs], limit
        )
        groups: list[dict[str, Any]] = []
        for (row, _candidates), translations in zip(group_specs, allocated, strict=True):
            if not translations:
                continue
            row_sense_id = str(row["sense_id"])
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
        response = self._response(
            "dictionary_translate",
            query,
            groups,
        )
        response["candidate_count"] = sum(len(group["translations"]) for group in groups)
        return response

    def dictionary_relations(
        self,
        word: str,
        relation: str,
        language: str = "en",
        target_language: str | None = None,
        sense_id: str | None = None,
        limit: int = 20,
        max_depth: int = 2,
        transitive_limit: int = 5,
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
        max_depth = _validate_bounded_integer(max_depth, field="max_depth", minimum=1, maximum=2)
        transitive_limit = _validate_allocation(
            transitive_limit, field="transitive_limit", limit=limit
        )
        query = {
            "word": original,
            "normalized_word": key,
            "language": language,
            "relation": relation,
            "target_language": target_language,
            "sense_id": sense_id,
            "limit": limit,
            "max_depth": max_depth,
            "transitive_limit": transitive_limit,
        }
        if not self._supports_languages(language, target_language):
            return self._unsupported_language_response("dictionary_relations", query)

        direct_frontier = self._relation_rows(
            key,
            language,
            _RELATION_CODES[relation],
            target_language=target_language,
            source_sense_id=sense_id,
            limit=_relation_scan_limit(limit),
        )
        direct_rows = [dict(row) for row in direct_frontier]
        for row in direct_rows:
            edge = dict(row)
            row["relation_scope"] = "direct"
            row["distance"] = 1
            row["path_rows"] = (edge,)
            row["scope_priority"] = 1 if row["source_sense_id"] is not None else 2
        direct_rows = self._target_diverse_relation_rows(direct_rows)

        transitive_rows: list[dict[str, Any]] = []
        supports_transitive = relation in {"hypernym", "hyponym"}
        transitive_budget = transitive_limit if max_depth == 2 and supports_transitive else 0
        if max_depth == 2 and supports_transitive and transitive_budget > 0:
            # With no final-language filter, the direct overfetch is exactly the
            # first-hop hierarchy frontier. Reuse it rather than issuing the
            # same high-degree query twice. A cross-lingual final filter still
            # needs an unfiltered first hop and is fetched inside the helper.
            transitive_frontier = direct_frontier if target_language is None else None
            transitive_rows = self._transitive_relation_rows(
                key,
                language,
                _RELATION_CODES[relation],
                target_language=target_language,
                source_sense_id=sense_id,
                limit=transitive_budget,
                first_edges=transitive_frontier,
            )
            for row in transitive_rows:
                row["relation_scope"] = "transitive"
                row["distance"] = 2

            direct_identities = {self._relation_identity(row) for row in direct_rows}
            transitive_rows = self._target_diverse_relation_rows(
                transitive_rows, seen=direct_identities
            )

        selected_transitive = transitive_rows[:transitive_budget]
        # The caller's transitive allocation is reserved only when truthful
        # paths exist. Any unused allocation returns to direct results.
        selected_direct = direct_rows[: limit - len(selected_transitive)]
        ordered_rows = [*selected_direct, *selected_transitive]

        results: list[dict[str, Any]] = []
        for row in ordered_rows:
            source_id = row["source_sense_id"]
            relation_code = int(row["relation_code"])
            direction_code = int(row["direction_code"])
            try:
                relation_name = _RELATION_NAMES[relation_code]
                direction_name = _DIRECTION_NAMES[direction_code]
            except KeyError as exc:  # artifact corruption, not model input
                raise RuntimeError("Relation artifact contains an unknown code") from exc
            path: list[dict[str, Any]] = []
            for edge in row["path_rows"]:
                try:
                    edge_relation = _RELATION_NAMES[int(edge["relation_code"])]
                    edge_direction = _DIRECTION_NAMES[int(edge["direction_code"])]
                except KeyError as exc:  # artifact corruption, not model input
                    raise RuntimeError("Relation artifact contains an unknown code") from exc
                path.append(
                    {
                        "source_term": edge["source_term"],
                        "source_language": edge["source_language"],
                        "source_sense_id": edge["source_sense_id"],
                        "relation": edge_relation,
                        "target_term": edge["target_term"],
                        "target_language": edge["target_language"],
                        "target_sense_id": edge["target_sense_id"],
                        "direction": edge_direction,
                        "provenance": _provenance(edge),
                    }
                )
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
                    "relation_scope": row["relation_scope"],
                    "distance": row["distance"],
                    "path": path,
                    "provenance": _provenance(row),
                }
            )
        return self._response(
            "dictionary_relations",
            query,
            results,
        )

    @staticmethod
    def _relation_identity(
        row: Mapping[str, Any],
    ) -> tuple[str, str, str | None, str | None]:
        return (
            str(row["target_normalized"]),
            str(row["target_language"]),
            row["source_sense_id"],
            row["target_sense_id"],
        )

    def _target_diverse_relation_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        seen: set[tuple[str, str, str | None, str | None]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return one row per target before additional sense variants."""

        exact_seen = set() if seen is None else set(seen)
        unique_rows: list[dict[str, Any]] = []
        for row in rows:
            identity = self._relation_identity(row)
            if identity in exact_seen:
                continue
            exact_seen.add(identity)
            unique_rows.append(row)

        target_seen: set[tuple[str, str]] = set()
        primary_rows: list[dict[str, Any]] = []
        sense_variants: list[dict[str, Any]] = []
        for row in unique_rows:
            target_identity = (
                str(row["target_language"]),
                str(row["target_normalized"]),
            )
            if target_identity in target_seen:
                sense_variants.append(row)
            else:
                target_seen.add(target_identity)
                primary_rows.append(row)
        return [*primary_rows, *sense_variants]

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
        query = {
            "word": original,
            "normalized_word": key,
            "source_language": source_language,
            "target_language": target_language,
            "limit": limit,
            "min_similarity": min_similarity,
        }
        if not self._supports_languages(source_language, target_language):
            return self._unsupported_language_response(
                "dictionary_semantic_neighbors", query
            )
        effective_target_language = (
            "en"
            if self.dataset_profile == "english" and target_language is None
            else target_language
        )
        results = self._semantic.search(
            key, source_language, effective_target_language, limit, min_similarity
        )
        if self.dataset_profile == "english":
            results = [item for item in results if item.get("language") == "en"]
        response = self._response(
            "dictionary_semantic_neighbors",
            query,
            results,
        )
        response["available"] = self._semantic.available
        return response

    def dictionary_wordplay(self, mode: str, text: str, limit: int = 20) -> dict[str, Any]:
        if not isinstance(mode, str) or mode not in WORDPLAY_MODES:
            raise ValueError(f"mode must be one of: {', '.join(sorted(WORDPLAY_MODES))}")
        original = text.strip() if isinstance(text, str) else text
        key = normalize_key(text, field="text", allow_wildcards=mode == "spelled_like")
        limit = validate_limit(limit)
        results = self._wordplay.search(mode, key, limit=limit)
        return self._response(
            "dictionary_wordplay",
            {
                "text": original,
                "normalized_text": key,
                "language": "en",
                "mode": mode,
                "limit": limit,
            },
            results,
        )

    def rhymes(self, text: str, mode: str = "exact", limit: int = 20) -> dict[str, Any]:
        if not isinstance(mode, str) or mode not in {"exact", "near"}:
            raise ValueError("mode must be one of: exact, near")
        original = text.strip() if isinstance(text, str) else text
        key = normalize_key(text, field="text")
        limit = validate_limit(limit)
        internal_mode = "rhyme" if mode == "exact" else "near_rhyme"
        results = self._wordplay.search(internal_mode, key, limit=limit)
        for result in results:
            result["mode"] = mode
        return self._response(
            "rhymes",
            {
                "text": original,
                "normalized_text": key,
                "language": "en",
                "mode": mode,
                "limit": limit,
            },
            results,
        )

    def wordplay(
        self,
        text: str,
        kind: str,
        context: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return deterministic, corpus-backed wordplay candidates."""

        if not isinstance(kind, str) or kind not in WORDPLAY_KINDS:
            raise ValueError(
                "kind must be one of: " + ", ".join(sorted(WORDPLAY_KINDS))
            )
        if context is not None and kind != "pun":
            raise ValueError("context is only accepted when kind is 'pun'")
        original = text.strip() if isinstance(text, str) else text
        key = normalize_key(text, field="text")
        limit = validate_limit(limit)
        normalized_context: str | None = None
        context_scope = "uncontextualized"
        if context is not None:
            if not isinstance(context, str):
                raise ValueError("context must be text")
            normalized_context = context.strip()
            if not normalized_context or len(normalized_context) > _MAX_CONTEXT_LENGTH:
                raise ValueError(
                    "context must contain between 1 and "
                    f"{_MAX_CONTEXT_LENGTH} characters"
                )
            context_scope = "contextualized"

        query: dict[str, Any] = {
            "text": original,
            "normalized_text": key,
            "kind": kind,
            "context": normalized_context,
            "limit": limit,
        }
        if kind == "anagram":
            results = self._actual_wordplay.anagram(key, limit=limit)
        elif kind == "palindrome":
            query["input_is_palindrome"] = _letters_form_palindrome(key)
            results = self._actual_wordplay.palindrome(key, limit=limit)
        elif kind == "spoonerism":
            parts = key.split()
            if len(parts) != 2:
                raise ValueError(
                    "spoonerism requires exactly two whitespace-separated words"
                )
            results = self._actual_wordplay.spoonerism(parts[0], parts[1], limit=limit)
        else:
            results = self._actual_wordplay.pun(
                key, context_scope=context_scope, limit=limit
            )
        return self._response("wordplay", query, results)

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
