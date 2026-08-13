"""Stable on-disk schema shared by builders and the query runtime."""

from __future__ import annotations

import sqlite3

from .constants import SCHEMA_VERSION

LEXICAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS senses (
    sense_id TEXT PRIMARY KEY,
    word TEXT NOT NULL,
    normalized_word TEXT NOT NULL,
    language TEXT NOT NULL,
    part_of_speech TEXT,
    gloss TEXT,
    etymology TEXT,
    source TEXT NOT NULL,
    source_license TEXT NOT NULL,
    source_url TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS examples (
    sense_id TEXT NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    example TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (sense_id, position)
);
CREATE TABLE IF NOT EXISTS pronunciations (
    sense_id TEXT NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    ipa TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL,
    PRIMARY KEY (sense_id, position)
);
CREATE TABLE IF NOT EXISTS translations (
    sense_id TEXT NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    target_language TEXT NOT NULL,
    term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    part_of_speech TEXT,
    source TEXT NOT NULL,
    source_license TEXT NOT NULL,
    source_url TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (sense_id, target_language, normalized_term, position)
);
CREATE TABLE IF NOT EXISTS synonyms (
    sense_id TEXT NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    language TEXT NOT NULL,
    part_of_speech TEXT,
    source TEXT NOT NULL,
    source_license TEXT NOT NULL,
    source_url TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (sense_id, language, normalized_term, position)
);
CREATE TABLE IF NOT EXISTS relations (
    source_term TEXT NOT NULL,
    source_normalized TEXT NOT NULL,
    source_language TEXT NOT NULL,
    source_sense_id TEXT,
    relation TEXT NOT NULL,
    target_term TEXT NOT NULL,
    target_normalized TEXT NOT NULL,
    target_language TEXT NOT NULL,
    target_sense_id TEXT,
    direction TEXT NOT NULL,
    source TEXT NOT NULL,
    source_license TEXT NOT NULL,
    source_url TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS relations_unique ON relations (
    source_normalized, source_language, IFNULL(source_sense_id, ''), relation,
    target_normalized, target_language, IFNULL(target_sense_id, ''), direction, source
);
CREATE TABLE IF NOT EXISTS pronunciations_words (
    word TEXT NOT NULL,
    normalized_word TEXT NOT NULL,
    phonemes TEXT NOT NULL,
    PRIMARY KEY (normalized_word, phonemes)
);
CREATE INDEX IF NOT EXISTS senses_lookup
    ON senses(language, normalized_word, part_of_speech);
CREATE INDEX IF NOT EXISTS translations_lookup
    ON translations(target_language, normalized_term);
CREATE INDEX IF NOT EXISTS synonyms_lookup
    ON synonyms(language, normalized_term);
CREATE INDEX IF NOT EXISTS relations_source_lookup
    ON relations(source_language, source_normalized, relation);
CREATE INDEX IF NOT EXISTS relations_target_lookup
    ON relations(target_language, target_normalized, relation);
CREATE INDEX IF NOT EXISTS pronunciations_words_lookup
    ON pronunciations_words(normalized_word);
"""


SEMANTIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS semantic_terms (
    semantic_id INTEGER PRIMARY KEY,
    concept TEXT NOT NULL UNIQUE,
    term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    language TEXT NOT NULL,
    vector_offset INTEGER NOT NULL UNIQUE,
    source TEXT NOT NULL,
    source_license TEXT NOT NULL,
    source_url TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS semantic_languages (
    language TEXT PRIMARY KEY,
    index_file TEXT NOT NULL,
    term_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS semantic_terms_lookup
    ON semantic_terms(language, normalized_term);
"""


def create_lexical_schema(connection: sqlite3.Connection, dataset_version: str) -> None:
    connection.executescript(LEXICAL_SCHEMA)
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (("schema_version", SCHEMA_VERSION), ("dataset_version", dataset_version)),
    )
    connection.commit()


def create_semantic_schema(
    connection: sqlite3.Connection,
    dataset_version: str,
    dimensions: int,
) -> None:
    connection.executescript(SEMANTIC_SCHEMA)
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (
            ("schema_version", SCHEMA_VERSION),
            ("dataset_version", dataset_version),
            ("dimensions", str(dimensions)),
            ("vector_dtype", "float16"),
            ("vector_file", "vectors/global.f16"),
            ("global_index", "indexes/global.usearch"),
            ("index_metric", "cos"),
            ("index_dtype", "i8"),
        ),
    )
    connection.commit()
