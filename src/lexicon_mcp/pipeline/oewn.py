"""Streaming Open English WordNet (WN-LMF XML) importer for schema v2."""

from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import open_binary
from .constants import DIRECTION_CODES, RELATION_CODES
from .interner import CorpusInterner

SOURCE = "Open English WordNet"
SOURCE_LICENSE = "CC-BY-4.0 and Princeton WordNet License"
SOURCE_URL = "https://en-word.net/"

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


def _entry_id(raw: str) -> str:
    return raw if raw.startswith("oewn:entry:") else f"oewn:entry:{raw}"


def _prepare_work_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS temp.build_oewn_synsets;
        DROP TABLE IF EXISTS temp.build_oewn_examples;
        DROP TABLE IF EXISTS temp.build_oewn_members;
        DROP TABLE IF EXISTS temp.build_oewn_synset_relations;
        DROP TABLE IF EXISTS temp.build_oewn_sense_relations;
        CREATE TEMP TABLE build_oewn_synsets (
            synset_id TEXT PRIMARY KEY,
            part_of_speech TEXT,
            gloss TEXT
        );
        CREATE TEMP TABLE build_oewn_examples (
            synset_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            example TEXT NOT NULL,
            PRIMARY KEY (synset_id, position)
        );
        CREATE TEMP TABLE build_oewn_members (
            synset_id TEXT NOT NULL,
            sense_id TEXT NOT NULL,
            term_id INTEGER NOT NULL,
            part_of_speech TEXT,
            PRIMARY KEY (synset_id, sense_id)
        );
        CREATE TEMP TABLE build_oewn_synset_relations (
            source_synset TEXT NOT NULL,
            relation_code INTEGER NOT NULL,
            target_synset TEXT NOT NULL,
            PRIMARY KEY (source_synset, relation_code, target_synset)
        );
        CREATE TEMP TABLE build_oewn_sense_relations (
            source_sense TEXT NOT NULL,
            relation_code INTEGER NOT NULL,
            target_sense TEXT NOT NULL,
            PRIMARY KEY (source_sense, relation_code, target_sense)
        );
        """
    )


def build_oewn(connection: sqlite3.Connection, path: Path) -> dict[str, int]:
    """Import OEWN with source-native IDs using compact SQLite work tables."""

    _prepare_work_tables(connection)
    interner = CorpusInterner(connection)
    provenance_id = interner.provenance(SOURCE, SOURCE_LICENSE, SOURCE_URL)
    connection.execute("DELETE FROM relations WHERE provenance_id=?", (provenance_id,))
    connection.execute("DELETE FROM lexical_entries WHERE provenance_id=?", (provenance_id,))
    connection.commit()
    counts = {
        "entries": 0,
        "synsets": 0,
        "senses": 0,
        "examples": 0,
        "pronunciations": 0,
        "synonyms": 0,
        "relations": 0,
    }

    with open_binary(path) as stream:
        for _event, element in ET.iterparse(stream, events=("end",)):
            tag = _local(element.tag)
            if tag != "Synset":
                if tag == "LexicalEntry":
                    element.clear()
                continue
            synset_id = element.get("id")
            if not synset_id:
                element.clear()
                continue
            definitions = [_text(item) for item in _children(element, "Definition")]
            gloss = "; ".join(value for value in definitions if value) or None
            connection.execute(
                "INSERT OR REPLACE INTO build_oewn_synsets VALUES (?,?,?)",
                (synset_id, element.get("partOfSpeech"), gloss),
            )
            for position, example in enumerate(_children(element, "Example")):
                value = _text(example)
                if value:
                    connection.execute(
                        "INSERT OR REPLACE INTO build_oewn_examples VALUES (?,?,?)",
                        (synset_id, position, value),
                    )
            for relation in _children(element, "SynsetRelation"):
                canonical = _RELATIONS.get((relation.get("relType") or "").casefold())
                target = relation.get("target")
                if canonical and target:
                    connection.execute(
                        "INSERT OR IGNORE INTO build_oewn_synset_relations VALUES (?,?,?)",
                        (synset_id, RELATION_CODES[canonical], target),
                    )
            counts["synsets"] += 1
            if counts["synsets"] % 10_000 == 0:
                connection.commit()
            element.clear()
    connection.commit()

    with open_binary(path) as stream:
        for _event, entry in ET.iterparse(stream, events=("end",)):
            tag = _local(entry.tag)
            if tag != "LexicalEntry":
                if tag == "Synset":
                    entry.clear()
                continue
            raw_entry_id = entry.get("id")
            lemmas = _children(entry, "Lemma")
            if not raw_entry_id or not lemmas:
                entry.clear()
                continue
            lemma = lemmas[0]
            word = lemma.get("writtenForm") or _text(lemma)
            part_of_speech = lemma.get("partOfSpeech")
            if not word:
                entry.clear()
                continue
            entry_id = _entry_id(raw_entry_id)
            term_id = interner.term(word, "en")
            connection.execute(
                "INSERT OR REPLACE INTO lexical_entries VALUES (?,?,?,?,?)",
                (entry_id, term_id, part_of_speech, None, provenance_id),
            )
            for position, pronunciation in enumerate(_children(entry, "Pronunciation")):
                value = _text(pronunciation)
                if not value:
                    continue
                connection.execute(
                    "INSERT OR REPLACE INTO pronunciations VALUES (?,?,?,?)",
                    (entry_id, value, pronunciation.get("variety") or "", position),
                )
                counts["pronunciations"] += 1
            for sense in _children(entry, "Sense"):
                raw_id = sense.get("id")
                synset = sense.get("synset")
                if not raw_id:
                    continue
                sense_id = _sense_id(raw_id)
                synset_record = connection.execute(
                    "SELECT part_of_speech,gloss FROM build_oewn_synsets WHERE synset_id=?",
                    (synset,),
                ).fetchone()
                sense_pos = part_of_speech or (synset_record[0] if synset_record else None)
                gloss = synset_record[1] if synset_record else None
                sense_definitions = [_text(item) for item in _children(sense, "Definition")]
                if any(sense_definitions):
                    gloss = "; ".join(value for value in sense_definitions if value)
                connection.execute(
                    "INSERT OR REPLACE INTO senses VALUES (?,?,?)",
                    (sense_id, entry_id, gloss),
                )
                if synset:
                    connection.execute(
                        "INSERT OR IGNORE INTO build_oewn_members VALUES (?,?,?,?)",
                        (synset, sense_id, term_id, sense_pos),
                    )
                    for position, example in connection.execute(
                        """SELECT position,example FROM build_oewn_examples
                        WHERE synset_id=? ORDER BY position""",
                        (synset,),
                    ):
                        connection.execute(
                            "INSERT OR REPLACE INTO examples VALUES (?,?,?)",
                            (sense_id, example, position),
                        )
                        counts["examples"] += 1
                for relation in _children(sense, "SenseRelation"):
                    canonical = _RELATIONS.get((relation.get("relType") or "").casefold())
                    target = relation.get("target")
                    if canonical and target:
                        connection.execute(
                            "INSERT OR IGNORE INTO build_oewn_sense_relations VALUES (?,?,?)",
                            (sense_id, RELATION_CODES[canonical], _sense_id(target)),
                        )
                counts["senses"] += 1
            counts["entries"] += 1
            if counts["entries"] % 10_000 == 0:
                connection.commit()
            entry.clear()
    connection.commit()

    changes = connection.total_changes
    connection.execute(
        """WITH candidates AS (
            SELECT DISTINCT m1.sense_id AS source_sense,
                   m2.term_id AS target_term_id,
                   m1.part_of_speech AS part_of_speech
            FROM build_oewn_members m1
            JOIN build_oewn_members m2 USING(synset_id)
            WHERE m1.sense_id<>m2.sense_id AND m1.term_id<>m2.term_id
        )
        INSERT OR IGNORE INTO synonyms
        (sense_id,target_term_id,part_of_speech,provenance_id,position)
        SELECT source_sense,target_term_id,part_of_speech,?,
               ROW_NUMBER() OVER (
                   PARTITION BY source_sense ORDER BY target_term_id
               ) - 1
        FROM candidates""",
        (provenance_id,),
    )
    counts["synonyms"] = connection.total_changes - changes

    relation_rows = connection.execute(
        """SELECT s.term_id,s.sense_id,r.relation_code,t.term_id,t.sense_id
        FROM build_oewn_synset_relations r
        JOIN build_oewn_members s ON s.synset_id=r.source_synset
        JOIN build_oewn_members t ON t.synset_id=r.target_synset
        ORDER BY r.source_synset,r.relation_code,r.target_synset,s.sense_id,t.sense_id"""
    )
    for source_term_id, source_id, code, target_term_id, target_id in relation_rows:
        counts["relations"] += int(
            _insert_relation(
                connection,
                int(source_term_id),
                str(source_id),
                int(code),
                int(target_term_id),
                str(target_id),
                provenance_id,
            )
        )
    sense_relation_rows = connection.execute(
        """SELECT se.term_id,r.source_sense,r.relation_code,te.term_id,r.target_sense
        FROM build_oewn_sense_relations r
        JOIN senses s ON s.sense_id=r.source_sense
        JOIN lexical_entries se ON se.entry_id=s.entry_id
        JOIN senses t ON t.sense_id=r.target_sense
        JOIN lexical_entries te ON te.entry_id=t.entry_id
        ORDER BY r.source_sense,r.relation_code,r.target_sense"""
    )
    for source_term_id, source_id, code, target_term_id, target_id in sense_relation_rows:
        counts["relations"] += int(
            _insert_relation(
                connection,
                int(source_term_id),
                str(source_id),
                int(code),
                int(target_term_id),
                str(target_id),
                provenance_id,
            )
        )
    connection.executescript(
        "DROP TABLE build_oewn_synsets; DROP TABLE build_oewn_examples; "
        "DROP TABLE build_oewn_members; DROP TABLE build_oewn_synset_relations; "
        "DROP TABLE build_oewn_sense_relations;"
    )
    connection.commit()
    return counts


def _insert_relation(
    connection: sqlite3.Connection,
    source_term_id: int,
    source_sense_id: str,
    relation_code: int,
    target_term_id: int,
    target_sense_id: str,
    provenance_id: int,
) -> bool:
    cursor = connection.execute(
        "INSERT OR IGNORE INTO relations VALUES (?,?,?,?,?,?,?)",
        (
            source_term_id,
            source_sense_id,
            relation_code,
            target_term_id,
            target_sense_id,
            DIRECTION_CODES["outbound"],
            provenance_id,
        ),
    )
    return cursor.rowcount > 0
