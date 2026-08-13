"""Streaming importer for Kaikki/Wiktextract JSONL snapshots."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .common import checked_language, iter_text_lines, normalize_term, stable_id

SOURCE = "Wiktionary via Wiktextract"
SOURCE_LICENSE = "CC-BY-SA-4.0 and GFDL-1.3-or-later"
SOURCE_URL = "https://kaikki.org/"


def _objects(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


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
    word: str,
    language: str,
    part_of_speech: str | None,
    sense: dict[str, Any],
    position: int,
) -> str:
    native = sense.get("id") or sense.get("senseid") or sense.get("sense_id")
    return stable_id(
        f"wikt:{language}",
        word,
        part_of_speech,
        native,
        _strings(sense.get("glosses")),
        position,
    )


def _unsensed_id(word: str, language: str, part_of_speech: str | None) -> str:
    return stable_id("wikt:unsensed", language, word, part_of_speech)


def build_wiktextract(
    connection: sqlite3.Connection,
    paths: list[Path],
    commit_interval: int = 10_000,
) -> dict[str, int]:
    counts = {
        "entries": 0,
        "senses": 0,
        "examples": 0,
        "pronunciations": 0,
        "translations": 0,
        "synonyms": 0,
        "antonyms": 0,
        "malformed": 0,
    }
    for path in sorted(paths, key=lambda item: item.as_posix()):
        for _line_number, line in iter_text_lines(path):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed"] += 1
                continue
            if not isinstance(entry, dict):
                counts["malformed"] += 1
                continue
            word = str(entry.get("word") or "").strip()
            if not word:
                counts["malformed"] += 1
                continue
            language = checked_language(entry.get("lang_code") or entry.get("language_code"))
            part_of_speech = str(entry.get("pos") or "").strip() or None
            etymology = _entry_etymology(entry)
            sense_ids: list[str] = []
            for position, sense in enumerate(_objects(entry.get("senses"))):
                sense_id = _sense_id(word, language, part_of_speech, sense, position)
                glosses = _strings(sense.get("glosses")) or _strings(sense.get("raw_glosses"))
                connection.execute(
                    """INSERT OR REPLACE INTO senses VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        sense_id,
                        word,
                        normalize_term(word),
                        language,
                        part_of_speech,
                        "; ".join(glosses) or None,
                        etymology,
                        SOURCE,
                        SOURCE_LICENSE,
                        SOURCE_URL,
                    ),
                )
                _insert_examples(connection, sense_id, sense, counts)
                _insert_synonyms(
                    connection, sense_id, word, language, part_of_speech, sense, counts
                )
                _insert_antonyms(connection, sense_id, word, language, sense, counts)
                _insert_translations(
                    connection, sense_id, language, part_of_speech, sense, counts
                )
                sense_ids.append(sense_id)
                counts["senses"] += 1

            entry_level = any(
                entry.get(key)
                for key in ("sounds", "translations", "synonyms", "antonyms", "etymology_text")
            )
            if entry_level or not sense_ids:
                unsensed_id = _unsensed_id(word, language, part_of_speech)
                connection.execute(
                    "INSERT OR REPLACE INTO senses VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        unsensed_id,
                        word,
                        normalize_term(word),
                        language,
                        part_of_speech,
                        None,
                        etymology,
                        SOURCE,
                        SOURCE_LICENSE,
                        SOURCE_URL,
                    ),
                )
                _insert_pronunciations(connection, unsensed_id, entry, counts)
                _insert_synonyms(
                    connection, unsensed_id, word, language, part_of_speech, entry, counts
                )
                _insert_antonyms(connection, unsensed_id, word, language, entry, counts)
                _insert_translations(
                    connection, unsensed_id, language, part_of_speech, entry, counts
                )
                counts["senses"] += 1
            counts["entries"] += 1
            if counts["entries"] % commit_interval == 0:
                connection.commit()
    connection.commit()
    return counts


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
    sense_id: str,
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
            (sense_id, value[0], value[1], position),
        )
        position += 1
        counts["pronunciations"] += 1


def _insert_synonyms(
    connection: sqlite3.Connection,
    sense_id: str,
    source_word: str,
    language: str,
    part_of_speech: str | None,
    container: dict[str, Any],
    counts: dict[str, int],
) -> None:
    position = 0
    seen: set[tuple[str, str]] = set()
    for item in _objects(container.get("synonyms")):
        term = _item_term(item)
        target_language = checked_language(item.get("lang_code"), language)
        if not term or normalize_term(term) == normalize_term(source_word):
            continue
        marker = (target_language, normalize_term(term))
        if marker in seen:
            continue
        seen.add(marker)
        connection.execute(
            "INSERT OR IGNORE INTO synonyms VALUES (?,?,?,?,?,?,?,?,?)",
            (
                sense_id,
                term,
                marker[1],
                target_language,
                part_of_speech,
                SOURCE,
                SOURCE_LICENSE,
                SOURCE_URL,
                position,
            ),
        )
        position += 1
        counts["synonyms"] += 1


def _insert_antonyms(
    connection: sqlite3.Connection,
    sense_id: str,
    source_word: str,
    language: str,
    container: dict[str, Any],
    counts: dict[str, int],
) -> None:
    for item in _objects(container.get("antonyms")):
        term = _item_term(item)
        if not term:
            continue
        target_language = checked_language(item.get("lang_code"), language)
        _relation(
            connection,
            source_word,
            language,
            sense_id,
            "antonym",
            term,
            target_language,
            None,
            "symmetric",
        )
        counts["antonyms"] += 1


def _insert_translations(
    connection: sqlite3.Connection,
    sense_id: str,
    language: str,
    part_of_speech: str | None,
    container: dict[str, Any],
    counts: dict[str, int],
) -> None:
    position = 0
    seen: set[tuple[str, str]] = set()
    for item in _objects(container.get("translations")):
        term = _item_term(item)
        target_language = checked_language(item.get("code") or item.get("lang_code"))
        if not term or target_language == "und":
            continue
        marker = (target_language, normalize_term(term))
        if marker in seen:
            continue
        seen.add(marker)
        connection.execute(
            "INSERT OR IGNORE INTO translations VALUES (?,?,?,?,?,?,?,?,?)",
            (
                sense_id,
                target_language,
                term,
                marker[1],
                str(item.get("pos") or part_of_speech or "").strip() or None,
                SOURCE,
                SOURCE_LICENSE,
                SOURCE_URL,
                position,
            ),
        )
        position += 1
        counts["translations"] += 1


def _relation(
    connection: sqlite3.Connection,
    source_word: str,
    source_language: str,
    source_sense_id: str | None,
    relation: str,
    target_word: str,
    target_language: str,
    target_sense_id: str | None,
    direction: str,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO relations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            source_word,
            normalize_term(source_word),
            source_language,
            source_sense_id,
            relation,
            target_word,
            normalize_term(target_word),
            target_language,
            target_sense_id,
            direction,
            SOURCE,
            SOURCE_LICENSE,
            SOURCE_URL,
        ),
    )
