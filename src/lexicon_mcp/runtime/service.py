"""Protocol-independent lexical query service over immutable SQLite artifacts."""

from __future__ import annotations

import math
import re
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
SUPPORTED_SCHEMA_VERSION = "1"

_CMU_PROVENANCE = {
    "source": "CMU Pronouncing Dictionary",
    "license": "CMUdict license",
    "url": "https://github.com/cmusphinx/cmudict",
}


def _provenance(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "source": str(row["source"] or "unknown"),
        "license": str(row["source_license"] or "unknown"),
        "url": str(row["source_url"]) if row["source_url"] else None,
    }


def _phoneme_rhyme(phonemes: str) -> tuple[str, ...]:
    tokens = tuple(phonemes.upper().split())
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if token[-1:].isdigit() and token[-1] in {"1", "2"}:
            return tokens[index:]
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index][-1:].isdigit():
            return tokens[index:]
    return tokens[-2:]


def _token_edit_distance(left: tuple[str, ...], right: tuple[str, ...], ceiling: int = 1) -> int:
    if abs(len(left) - len(right)) > ceiling:
        return ceiling + 1
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        if min(current) > ceiling:
            return ceiling + 1
        previous = current
    return previous[-1]


def _spelling_pattern_matches(pattern: str, candidate: str) -> bool:
    """Match only the documented ? and * wildcards; all other text is literal."""

    expression = "".join(
        "." if char == "?" else ".*" if char == "*" else re.escape(char)
        for char in pattern
    )
    return re.fullmatch(expression, candidate, flags=re.DOTALL) is not None


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
        if "senses" not in self._tables:
            self._connection.close()
            raise RuntimeError("Lexical database has no senses table")
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
        self._wordplay_rows: tuple[tuple[str, str, str], ...] | None = None

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
        if "metadata" not in self._tables:
            return
        metadata = {
            str(row["key"]): str(row["value"])
            for row in self._connection.execute(
                "SELECT key, value FROM metadata WHERE key IN ('dataset_version', 'schema_version')"
            ).fetchall()
        }
        if metadata.get("dataset_version") not in {None, self.dataset_version}:
            self._connection.close()
            raise RuntimeError(
                "Lexical database version does not match the activated dataset version"
            )
        if metadata.get("schema_version") not in {None, SUPPORTED_SCHEMA_VERSION}:
            self._connection.close()
            raise RuntimeError(
                f"Unsupported lexical schema version {metadata['schema_version']!r}; "
                f"this server supports {SUPPORTED_SCHEMA_VERSION!r}"
            )

    def close(self) -> None:
        self._semantic.close()
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
        clauses = ["normalized_word = ?", "language = ?"]
        parameters: list[Any] = [word, language]
        if part_of_speech is not None:
            clauses.append("LOWER(part_of_speech) = ?")
            parameters.append(part_of_speech)
        if sense_id is not None:
            clauses.append("sense_id = ?")
            parameters.append(sense_id)
        parameters.append(limit)
        with self._lock:
            return self._connection.execute(
                f"""
                SELECT sense_id, word, normalized_word, language, part_of_speech,
                       gloss, etymology, source, source_license, source_url
                FROM senses
                WHERE {' AND '.join(clauses)}
                ORDER BY CASE WHEN gloss IS NULL THEN 1 ELSE 0 END, sense_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()

    def _dependent_rows(
        self, table: str, columns: str, sense_id: str, *, limit: int = 100
    ) -> list[sqlite3.Row]:
        if table not in self._tables:
            return []
        with self._lock:
            return self._connection.execute(
                f"SELECT {columns} FROM {table} WHERE sense_id = ? ORDER BY position LIMIT ?",
                (sense_id, limit),
            ).fetchall()

    def _sense_result(self, row: sqlite3.Row) -> dict[str, Any]:
        sense_id = str(row["sense_id"])
        examples = [
            str(item["example"])
            for item in self._dependent_rows("examples", "example, position", sense_id)
        ]
        pronunciations = [
            {"ipa": item["ipa"], "region": item["region"]}
            for item in self._dependent_rows(
                "pronunciations", "ipa, region, position", sense_id
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
            for item in self._dependent_rows(
                "translations",
                "target_language, term, part_of_speech, source, source_license, "
                "source_url, position",
                sense_id,
            )
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
        seen: set[tuple[str, str]] = set()
        remaining = limit
        for row in rows:
            if remaining <= 0:
                break
            row_sense_id = str(row["sense_id"])
            candidates: list[dict[str, Any]] = []
            for item in self._dependent_rows(
                "synonyms",
                "term, normalized_term, language, part_of_speech, source, "
                "source_license, source_url, position",
                row_sense_id,
                limit=remaining,
            ):
                identity = (str(item["normalized_term"]), str(item["language"]))
                if identity == (key, language) or identity in seen:
                    continue
                seen.add(identity)
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
            fallback = self._relation_synonyms(key, language, part_of_speech, remaining, seen)
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

    def _relation_synonyms(
        self,
        word: str,
        language: str,
        part_of_speech: str | None,
        limit: int,
        seen: set[tuple[str, str]],
    ) -> dict[str, Any] | None:
        if "relations" not in self._tables:
            return None
        clauses = ["source_normalized = ?", "source_language = ?", "relation = 'synonym'"]
        parameters: list[Any] = [word, language]
        if part_of_speech is not None:
            # Relations carry no POS. A requested POS must never receive unscoped fallback data.
            return None
        parameters.append(limit * 2)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT target_term, target_normalized, target_language, source,
                       source_license, source_url
                FROM relations
                WHERE {' AND '.join(clauses)}
                ORDER BY target_language, target_normalized
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        candidates: list[dict[str, Any]] = []
        for row in rows:
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
            for item in self._dependent_rows(
                "translations",
                "target_language, term, normalized_term, part_of_speech, source, "
                "source_license, source_url, position",
                row_sense_id,
                limit=limit,
            ):
                if item["target_language"] != target_language:
                    continue
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
        if "relations" in self._tables:
            clauses = ["source_normalized = ?", "source_language = ?", "relation = ?"]
            parameters: list[Any] = [key, language, relation]
            if target_language is not None:
                clauses.append("target_language = ?")
                parameters.append(target_language)
            if sense_id is not None:
                clauses.append("source_sense_id = ?")
                parameters.append(sense_id)
            parameters.append(limit * 2)
            with self._lock:
                rows = self._connection.execute(
                    f"""
                    SELECT source_term, source_sense_id, relation, target_term,
                           target_language, target_sense_id, direction, source,
                           source_license, source_url, target_normalized
                    FROM relations
                    WHERE {' AND '.join(clauses)}
                    ORDER BY target_language, target_normalized, target_sense_id
                    LIMIT ?
                    """,
                    parameters,
                ).fetchall()
            seen: set[tuple[str, str, str | None]] = set()
            for row in rows:
                identity = (
                    str(row["target_normalized"]),
                    str(row["target_language"]),
                    row["target_sense_id"],
                )
                if identity in seen:
                    continue
                seen.add(identity)
                source_id = row["source_sense_id"]
                results.append(
                    {
                        "source_term": row["source_term"],
                        "source_language": language,
                        "source_sense_id": source_id,
                        "sense_scope": sense_scope(source_id),
                        "relation": row["relation"],
                        "target_term": row["target_term"],
                        "target_language": row["target_language"],
                        "target_sense_id": row["target_sense_id"],
                        "direction": row["direction"],
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
        rows = self._load_wordplay_rows()
        source_pronunciations = [
            phonemes for _word, normalized, phonemes in rows if normalized == key
        ]
        source_phoneme_set = set(source_pronunciations)
        source_rhymes = {_phoneme_rhyme(phonemes) for phonemes in source_pronunciations}
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for word, normalized, phonemes in rows:
            if normalized == key or normalized in seen:
                continue
            match_kind: str | None = None
            if mode == "prefix" and normalized.startswith(key):
                match_kind = "prefix"
            elif mode == "spelled_like" and _spelling_pattern_matches(key, normalized):
                match_kind = "spelled_like"
            elif mode == "sounds_like" and phonemes in source_phoneme_set:
                match_kind = "sounds_like"
            elif mode == "rhyme" and source_rhymes and _phoneme_rhyme(phonemes) in source_rhymes:
                match_kind = "rhyme"
            elif mode == "near_rhyme" and source_rhymes:
                candidate_rhyme = _phoneme_rhyme(phonemes)
                if candidate_rhyme not in source_rhymes and any(
                    _token_edit_distance(candidate_rhyme, source_rhyme) == 1
                    for source_rhyme in source_rhymes
                ):
                    match_kind = "near_rhyme"
            if match_kind is None:
                continue
            seen.add(normalized)
            results.append(
                {
                    "term": word,
                    "language": "en",
                    "mode": match_kind,
                    "phonemes": phonemes,
                    "sense_scope": "unsensed",
                    "provenance": dict(_CMU_PROVENANCE),
                }
            )
            if len(results) == limit:
                break
        return self._response(
            "dictionary_wordplay",
            {"text": original, "normalized_text": key, "language": "en", "mode": mode},
            results,
        )

    def _load_wordplay_rows(self) -> tuple[tuple[str, str, str], ...]:
        with self._lock:
            if self._wordplay_rows is not None:
                return self._wordplay_rows
            if "pronunciations_words" not in self._tables:
                self._wordplay_rows = ()
            else:
                self._wordplay_rows = tuple(
                    (str(row["word"]), str(row["normalized_word"]), str(row["phonemes"]))
                    for row in self._connection.execute(
                        """
                        SELECT word, normalized_word, phonemes
                        FROM pronunciations_words
                        ORDER BY normalized_word, phonemes
                        """
                    ).fetchall()
                )
            return self._wordplay_rows

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
