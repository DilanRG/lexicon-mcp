"""Streaming importer for Kaikki/Wiktextract JSONL snapshots."""

from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .common import checked_language, iter_text_lines, normalize_term, stable_id
from .constants import DIRECTION_CODES, RELATION_CODES
from .interner import CorpusInterner

try:
    _orjson: Any = importlib.import_module("orjson")
except ImportError:  # pragma: no cover - standard-library fallback
    _orjson = None

SOURCE = "Wiktionary via Wiktextract"
SOURCE_LICENSE = "CC-BY-SA-4.0 and GFDL-1.3-or-later"
SOURCE_URL = "https://kaikki.org/"


def _objects(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _load_json(line: str) -> object:
    return _orjson.loads(line) if _orjson is not None else json.loads(line)


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _item_term(item: dict[str, Any]) -> str | None:
    for key in ("word", "term", "alt"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _sense_id(
    entry_id: str,
    sense: dict[str, Any],
    position: int,
) -> str:
    native_value = sense.get("id") or sense.get("senseid") or sense.get("sense_id")
    native: object
    if isinstance(native_value, list):
        native = tuple(sorted(str(item) for item in native_value))
    elif native_value is None:
        native = None
    else:
        native = str(native_value)
    return stable_id("wikt:sense", entry_id, native, _strings(sense.get("glosses")), position)


def _unsensed_id(entry_id: str) -> str:
    return stable_id("wikt:unsensed", entry_id)


def _labeled_id(
    entry_id: str,
    label: str,
) -> str:
    """Return a stable ID for an explicit source label, without sense matching."""

    return stable_id("wikt:labeled", entry_id, label)


def _entry_id(
    word: str,
    language: str,
    part_of_speech: str | None,
    entry: dict[str, Any],
    source_position: int,
) -> str:
    return stable_id(
        "wikt:entry",
        language,
        word,
        part_of_speech,
        entry.get("etymology_number"),
        _entry_etymology(entry),
        source_position,
    )


def _item_sense_label(item: dict[str, Any]) -> str | None:
    """Preserve a non-empty Wiktextract relation label exactly as supplied."""

    value = item.get("sense")
    return value if isinstance(value, str) and value.strip() else None


def _entry_relation_groups(
    entry: dict[str, Any],
) -> dict[str | None, dict[str, list[dict[str, Any]]]]:
    """Group entry-level lexical facts by their explicit source sense label.

    The labels are not compared with, or attached to, Wiktextract sense
    glosses. That would be heuristic linking. Equal exact labels share one
    standalone source-sense row; missing labels share the unsensed row.
    """

    groups: dict[str | None, dict[str, list[dict[str, Any]]]] = {}
    for relation in ("translations", "synonyms", "antonyms"):
        for item in _objects(entry.get(relation)):
            label = _item_sense_label(item)
            group = groups.setdefault(
                label,
                {"translations": [], "synonyms": [], "antonyms": []},
            )
            group[relation].append(item)
    return groups


def _insert_source_sense(
    connection: sqlite3.Connection,
    sense_id: str,
    entry_id: str,
    gloss: str | None,
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO senses VALUES (?,?,?)",
        (sense_id, entry_id, gloss),
    )


def build_wiktextract(
    connection: sqlite3.Connection,
    paths: list[Path],
    commit_interval: int = 10_000,
    allowed_languages: frozenset[str] | None = None,
) -> dict[str, int]:
    counts = {
        "entries": 0,
        "senses": 0,
        "examples": 0,
        "pronunciations": 0,
        "translations": 0,
        "synonyms": 0,
        "antonyms": 0,
        "labeled_senses": 0,
        "unsensed_senses": 0,
        "language_codes": 0,
        "malformed": 0,
    }
    language_codes: set[str] = set()
    interner = CorpusInterner(connection)
    provenance_id = interner.provenance(SOURCE, SOURCE_LICENSE, SOURCE_URL)
    # A stage can be resumed after committed batches. Remove only this source's
    # rows so restart is idempotent while preserving completed earlier stages.
    connection.execute("DELETE FROM relations WHERE provenance_id=?", (provenance_id,))
    connection.execute("DELETE FROM lexical_entries WHERE provenance_id=?", (provenance_id,))
    connection.commit()
    source_position = 0
    for path in sorted(paths, key=lambda item: item.as_posix()):
        for _line_number, line in iter_text_lines(path):
            if not line.strip():
                continue
            try:
                entry = _load_json(line)
            except ValueError:
                counts["malformed"] += 1
                continue
            if not isinstance(entry, dict):
                counts["malformed"] += 1
                continue
            source_position += 1
            word = str(entry.get("word") or "").strip()
            if not word:
                counts["malformed"] += 1
                continue
            language = checked_language(entry.get("lang_code") or entry.get("language_code"))
            if allowed_languages is not None and language not in allowed_languages:
                continue
            if language != "und":
                language_codes.add(language)
            part_of_speech = str(entry.get("pos") or "").strip() or None
            etymology = _entry_etymology(entry)
            entry_id = _entry_id(word, language, part_of_speech, entry, source_position)
            term_id = interner.term(word, language)
            connection.execute(
                "INSERT INTO lexical_entries VALUES (?,?,?,?,?)",
                (entry_id, term_id, part_of_speech, etymology, provenance_id),
            )
            sense_ids: list[str] = []
            for position, sense in enumerate(_objects(entry.get("senses"))):
                sense_id = _sense_id(entry_id, sense, position)
                glosses = _strings(sense.get("glosses")) or _strings(sense.get("raw_glosses"))
                _insert_source_sense(
                    connection,
                    sense_id,
                    entry_id,
                    "; ".join(glosses) or None,
                )
                _insert_examples(connection, sense_id, sense, counts)
                _insert_synonyms(
                    connection,
                    interner,
                    provenance_id,
                    sense_id,
                    word,
                    language,
                    part_of_speech,
                    sense,
                    counts,
                    allowed_languages,
                )
                _insert_antonyms(
                    connection,
                    interner,
                    provenance_id,
                    sense_id,
                    word,
                    language,
                    sense,
                    counts,
                    allowed_languages,
                )
                _insert_translations(
                    connection,
                    interner,
                    provenance_id,
                    sense_id,
                    language,
                    part_of_speech,
                    sense,
                    counts,
                    allowed_languages,
                )
                sense_ids.append(sense_id)
                counts["senses"] += 1

            relation_groups = _entry_relation_groups(entry)
            needs_unsensed = bool(
                None in relation_groups
                or (not sense_ids and not any(label is not None for label in relation_groups))
            )
            if needs_unsensed:
                unsensed_id = _unsensed_id(entry_id)
                _insert_source_sense(
                    connection,
                    unsensed_id,
                    entry_id,
                    None,
                )
                unlabeled = relation_groups.get(None)
                if unlabeled:
                    _insert_relation_group(
                        connection,
                        interner,
                        provenance_id,
                        unsensed_id,
                        word,
                        language,
                        part_of_speech,
                        unlabeled,
                        counts,
                        allowed_languages,
                    )
                counts["senses"] += 1
                counts["unsensed_senses"] += 1
            _insert_pronunciations(connection, entry_id, entry, counts)
            for label, group in relation_groups.items():
                if label is None:
                    continue
                sense_id = _labeled_id(entry_id, label)
                _insert_source_sense(
                    connection,
                    sense_id,
                    entry_id,
                    label,
                )
                _insert_relation_group(
                    connection,
                    interner,
                    provenance_id,
                    sense_id,
                    word,
                    language,
                    part_of_speech,
                    group,
                    counts,
                    allowed_languages,
                )
                counts["senses"] += 1
                counts["labeled_senses"] += 1
            counts["entries"] += 1
            if counts["entries"] % commit_interval == 0:
                connection.commit()
    connection.commit()
    counts["language_codes"] = len(language_codes)
    return counts


def _insert_relation_group(
    connection: sqlite3.Connection,
    interner: CorpusInterner,
    provenance_id: int,
    sense_id: str,
    source_word: str,
    language: str,
    part_of_speech: str | None,
    group: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    allowed_languages: frozenset[str] | None,
) -> None:
    """Insert one exact-label or unlabeled group through shared relation paths."""

    _insert_synonyms(
        connection,
        interner,
        provenance_id,
        sense_id,
        source_word,
        language,
        part_of_speech,
        {"synonyms": group["synonyms"]},
        counts,
        allowed_languages,
    )
    _insert_antonyms(
        connection,
        interner,
        provenance_id,
        sense_id,
        source_word,
        language,
        {"antonyms": group["antonyms"]},
        counts,
        allowed_languages,
    )
    _insert_translations(
        connection,
        interner,
        provenance_id,
        sense_id,
        language,
        part_of_speech,
        {"translations": group["translations"]},
        counts,
        allowed_languages,
    )


def _entry_etymology(entry: dict[str, Any]) -> str | None:
    direct = entry.get("etymology_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    values = _strings(entry.get("etymology_texts"))
    return "\n\n".join(values) or None


def _insert_examples(
    connection: sqlite3.Connection,
    sense_id: str,
    container: dict[str, Any],
    counts: dict[str, int],
) -> None:
    for position, example in enumerate(_objects(container.get("examples"))):
        text = example.get("text") or example.get("example")
        if isinstance(text, str) and text.strip():
            connection.execute(
                "INSERT OR REPLACE INTO examples VALUES (?,?,?)",
                (sense_id, text.strip(), position),
            )
            counts["examples"] += 1


def _insert_pronunciations(
    connection: sqlite3.Connection,
    entry_id: str,
    container: dict[str, Any],
    counts: dict[str, int],
) -> None:
    position = 0
    seen: set[tuple[str, str]] = set()
    for sound in _objects(container.get("sounds")):
        ipa = sound.get("ipa")
        if not isinstance(ipa, str) or not ipa.strip():
            continue
        region_values = _strings(sound.get("tags"))
        region = ", ".join(region_values)
        value = (ipa.strip(), region)
        if value in seen:
            continue
        seen.add(value)
        connection.execute(
            "INSERT OR REPLACE INTO pronunciations VALUES (?,?,?,?)",
            (entry_id, value[0], value[1], position),
        )
        position += 1
        counts["pronunciations"] += 1


def _insert_synonyms(
    connection: sqlite3.Connection,
    interner: CorpusInterner,
    provenance_id: int,
    sense_id: str,
    source_word: str,
    language: str,
    part_of_speech: str | None,
    container: dict[str, Any],
    counts: dict[str, int],
    allowed_languages: frozenset[str] | None,
) -> None:
    position = 0
    seen: set[tuple[str, str]] = set()
    for item in _objects(container.get("synonyms")):
        term = _item_term(item)
        target_language = checked_language(item.get("lang_code"), language)
        if allowed_languages is not None and target_language not in allowed_languages:
            continue
        if not term or normalize_term(term) == normalize_term(source_word):
            continue
        marker = (target_language, normalize_term(term))
        if marker in seen:
            continue
        seen.add(marker)
        connection.execute(
            "INSERT OR IGNORE INTO synonyms VALUES (?,?,?,?,?)",
            (
                sense_id,
                interner.term(term, target_language),
                part_of_speech,
                provenance_id,
                position,
            ),
        )
        position += 1
        counts["synonyms"] += 1


def _insert_antonyms(
    connection: sqlite3.Connection,
    interner: CorpusInterner,
    provenance_id: int,
    sense_id: str,
    source_word: str,
    language: str,
    container: dict[str, Any],
    counts: dict[str, int],
    allowed_languages: frozenset[str] | None,
) -> None:
    seen: set[tuple[str, str]] = set()
    for item in _objects(container.get("antonyms")):
        term = _item_term(item)
        if not term:
            continue
        target_language = checked_language(item.get("lang_code"), language)
        if allowed_languages is not None and target_language not in allowed_languages:
            continue
        marker = (target_language, normalize_term(term))
        if marker in seen:
            continue
        seen.add(marker)
        inserted = _relation(
            connection,
            interner,
            provenance_id,
            source_word,
            language,
            sense_id,
            "antonym",
            term,
            target_language,
            None,
            "symmetric",
        )
        counts["antonyms"] += int(inserted)


def _insert_translations(
    connection: sqlite3.Connection,
    interner: CorpusInterner,
    provenance_id: int,
    sense_id: str,
    language: str,
    part_of_speech: str | None,
    container: dict[str, Any],
    counts: dict[str, int],
    allowed_languages: frozenset[str] | None,
) -> None:
    position = 0
    seen: set[tuple[str, str]] = set()
    for item in _objects(container.get("translations")):
        term = _item_term(item)
        target_language = checked_language(item.get("code") or item.get("lang_code"))
        if not term or target_language == "und":
            continue
        if allowed_languages is not None and target_language not in allowed_languages:
            continue
        marker = (target_language, normalize_term(term))
        if marker in seen:
            continue
        seen.add(marker)
        connection.execute(
            "INSERT OR IGNORE INTO translations VALUES (?,?,?,?,?)",
            (
                sense_id,
                interner.term(term, target_language),
                str(item.get("pos") or part_of_speech or "").strip() or None,
                provenance_id,
                position,
            ),
        )
        position += 1
        counts["translations"] += 1


def _relation(
    connection: sqlite3.Connection,
    interner: CorpusInterner,
    provenance_id: int,
    source_word: str,
    source_language: str,
    source_sense_id: str | None,
    relation: str,
    target_word: str,
    target_language: str,
    target_sense_id: str | None,
    direction: str,
) -> bool:
    cursor = connection.execute(
        "INSERT OR IGNORE INTO relations VALUES (?,?,?,?,?,?,?)",
        (
            interner.term(source_word, source_language),
            source_sense_id,
            RELATION_CODES[relation],
            interner.term(target_word, target_language),
            target_sense_id,
            DIRECTION_CODES[direction],
            provenance_id,
        ),
    )
    return cursor.rowcount > 0
