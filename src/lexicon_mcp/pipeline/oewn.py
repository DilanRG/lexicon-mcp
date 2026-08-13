"""Streaming Open English WordNet (WN-LMF XML) importer."""

from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import normalize_term, open_binary

SOURCE = "Open English WordNet"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_URL = "https://en-word.net/"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [item for item in element.iter() if _local(item.tag) == name]


def _text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = " ".join("".join(element.itertext()).split())
    return value or None


def _sense_id(raw: str) -> str:
    return raw if raw.startswith("oewn:") else f"oewn:{raw}"


def _prepare_work_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS build_oewn_synsets;
        DROP TABLE IF EXISTS build_oewn_members;
        DROP TABLE IF EXISTS build_oewn_synset_relations;
        CREATE TABLE build_oewn_synsets (
            synset_id TEXT PRIMARY KEY,
            part_of_speech TEXT,
            gloss TEXT
        );
        CREATE TABLE build_oewn_members (
            synset_id TEXT NOT NULL,
            sense_id TEXT NOT NULL,
            word TEXT NOT NULL,
            normalized_word TEXT NOT NULL,
            part_of_speech TEXT,
            PRIMARY KEY (synset_id, sense_id)
        );
        CREATE TABLE build_oewn_synset_relations (
            source_synset TEXT NOT NULL,
            relation TEXT NOT NULL,
            target_synset TEXT NOT NULL,
            PRIMARY KEY (source_synset, relation, target_synset)
        );
        """
    )


_RELATIONS = {
    "antonym": "antonym",
    "hypernym": "hypernym",
    "instance_hypernym": "hypernym",
    "hyponym": "hyponym",
    "instance_hyponym": "hyponym",
    "mero_part": "meronym",
    "mero_member": "meronym",
    "mero_substance": "meronym",
    "holo_part": "holonym",
    "holo_member": "holonym",
    "holo_substance": "holonym",
    "derivation": "derived_from",
    "also": "related",
    "similar": "related",
}


def build_oewn(connection: sqlite3.Connection, path: Path) -> dict[str, int]:
    """Import an OEWN XML or XML.gz file into an initialized lexical database.

    Parsing is two-pass and bounded by compact SQLite work tables rather than an
    in-memory representation of the WordNet graph.
    """

    _prepare_work_tables(connection)
    counts = {"synsets": 0, "senses": 0, "examples": 0, "synonyms": 0, "relations": 0}

    with open_binary(path) as stream:
        for _event, element in ET.iterparse(stream, events=("end",)):
            if _local(element.tag) != "Synset":
                continue
            synset_id = element.get("id")
            if not synset_id:
                element.clear()
                continue
            definitions = [_text(item) for item in _children(element, "Definition")]
            gloss = "; ".join(value for value in definitions if value) or None
            connection.execute(
                "INSERT OR REPLACE INTO build_oewn_synsets VALUES (?, ?, ?)",
                (synset_id, element.get("partOfSpeech"), gloss),
            )
            for relation in _children(element, "SynsetRelation"):
                canonical = _RELATIONS.get((relation.get("relType") or "").casefold())
                target = relation.get("target")
                if canonical and target:
                    connection.execute(
                        "INSERT OR IGNORE INTO build_oewn_synset_relations VALUES (?, ?, ?)",
                        (synset_id, canonical, target),
                    )
            counts["synsets"] += 1
            if counts["synsets"] % 10_000 == 0:
                connection.commit()
            element.clear()
    connection.commit()

    with open_binary(path) as stream:
        for _event, entry in ET.iterparse(stream, events=("end",)):
            if _local(entry.tag) != "LexicalEntry":
                continue
            lemmas = _children(entry, "Lemma")
            if not lemmas:
                entry.clear()
                continue
            lemma = lemmas[0]
            word = lemma.get("writtenForm") or _text(lemma)
            part_of_speech = lemma.get("partOfSpeech")
            if not word:
                entry.clear()
                continue
            for sense in _children(entry, "Sense"):
                raw_id = sense.get("id")
                synset = sense.get("synset")
                if not raw_id:
                    continue
                sense_id = _sense_id(raw_id)
                synset_record = connection.execute(
                    "SELECT part_of_speech, gloss FROM build_oewn_synsets WHERE synset_id=?",
                    (synset,),
                ).fetchone()
                sense_pos = part_of_speech or (synset_record[0] if synset_record else None)
                gloss = synset_record[1] if synset_record else None
                sense_definitions = [_text(item) for item in _children(sense, "Definition")]
                if any(sense_definitions):
                    gloss = "; ".join(value for value in sense_definitions if value)
                connection.execute(
                    """INSERT OR REPLACE INTO senses
                    (sense_id,word,normalized_word,language,part_of_speech,gloss,etymology,
                     source,source_license,source_url) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        sense_id,
                        word,
                        normalize_term(word),
                        "en",
                        sense_pos,
                        gloss,
                        None,
                        SOURCE,
                        SOURCE_LICENSE,
                        SOURCE_URL,
                    ),
                )
                if synset:
                    connection.execute(
                        "INSERT OR IGNORE INTO build_oewn_members VALUES (?,?,?,?,?)",
                        (synset, sense_id, word, normalize_term(word), sense_pos),
                    )
                for position, example in enumerate(_children(sense, "Example")):
                    value = _text(example)
                    if value:
                        connection.execute(
                            "INSERT OR REPLACE INTO examples VALUES (?, ?, ?)",
                            (sense_id, value, position),
                        )
                        counts["examples"] += 1
                for relation in _children(sense, "SenseRelation"):
                    canonical = _RELATIONS.get((relation.get("relType") or "").casefold())
                    target = relation.get("target")
                    if not canonical or not target:
                        continue
                    target_id = _sense_id(target)
                    target_row = connection.execute(
                        "SELECT word, normalized_word FROM senses WHERE sense_id=?", (target_id,)
                    ).fetchone()
                    if target_row:
                        _insert_relation(
                            connection,
                            word,
                            sense_id,
                            canonical,
                            target_row[0],
                            target_id,
                        )
                        counts["relations"] += 1
                counts["senses"] += 1
            if counts["senses"] % 10_000 == 0:
                connection.commit()
            entry.clear()
    connection.commit()

    members = connection.execute(
        "SELECT synset_id,sense_id,word,normalized_word,part_of_speech "
        "FROM build_oewn_members ORDER BY synset_id,sense_id"
    )
    groups: dict[str, list[tuple[str, str, str, str | None]]] = {}
    for synset, sense_id, word, normalized, pos in members:
        groups.setdefault(synset, []).append((sense_id, word, normalized, pos))
    for group in groups.values():
        for sense_id, word, _normalized, pos in group:
            position = 0
            for other_id, other_word, other_normalized, _other_pos in group:
                if other_id == sense_id or other_normalized == normalize_term(word):
                    continue
                connection.execute(
                    "INSERT OR IGNORE INTO synonyms VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        sense_id,
                        other_word,
                        other_normalized,
                        "en",
                        pos,
                        SOURCE,
                        SOURCE_LICENSE,
                        SOURCE_URL,
                        position,
                    ),
                )
                position += 1
                counts["synonyms"] += 1

    relation_rows = connection.execute(
        """SELECT s.word,s.sense_id,r.relation,t.word,t.sense_id
        FROM build_oewn_synset_relations r
        JOIN build_oewn_members s ON s.synset_id=r.source_synset
        JOIN build_oewn_members t ON t.synset_id=r.target_synset
        ORDER BY r.source_synset,r.relation,r.target_synset,s.sense_id,t.sense_id"""
    )
    for source_word, source_id, relation, target_word, target_id in relation_rows:
        _insert_relation(connection, source_word, source_id, relation, target_word, target_id)
        counts["relations"] += 1
    connection.executescript(
        "DROP TABLE build_oewn_synsets; DROP TABLE build_oewn_members; "
        "DROP TABLE build_oewn_synset_relations;"
    )
    connection.commit()
    return counts


def _insert_relation(
    connection: sqlite3.Connection,
    source_word: str,
    source_id: str,
    relation: str,
    target_word: str,
    target_id: str,
) -> None:
    connection.execute(
        """INSERT OR IGNORE INTO relations
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            source_word,
            normalize_term(source_word),
            "en",
            source_id,
            relation,
            target_word,
            normalize_term(target_word),
            "en",
            target_id,
            "outbound",
            SOURCE,
            SOURCE_LICENSE,
            SOURCE_URL,
        ),
    )
