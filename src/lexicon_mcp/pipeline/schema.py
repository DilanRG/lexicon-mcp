"""Compact v3 on-disk schema shared by corpus builders and query runtime.

No v1 dataset was publicly released, so no migration is required.  v2 artifacts
remain immutable; builders create v3 artifacts from pinned upstream sources,
adding the bounded wordplay reverse indexes on top of the v2 tables.
"""

from __future__ import annotations

import sqlite3

from .constants import SCHEMA_VERSION, SEMANTIC_SCHEMA_VERSION
from .wordplay import (
    is_palindrome,
    is_wordplay_eligible,
    letter_signature,
    normalized_letters,
    reverse_letters,
    split_arpabet_onset,
)

# Streaming batch size for index population; keeps Python memory bounded
# regardless of corpus size.
_WORDPLAY_BATCH_SIZE = 10_000

DIMENSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS provenance (
    provenance_id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    source_license TEXT NOT NULL,
    source_url TEXT NOT NULL,
    UNIQUE (source, source_license, source_url)
);
CREATE TABLE IF NOT EXISTS lexical_terms (
    term_id INTEGER PRIMARY KEY,
    term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    language TEXT NOT NULL,
    UNIQUE (language, normalized_term, term)
);
"""

LEXICAL_SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
    + DIMENSION_SCHEMA
    + """
CREATE TABLE IF NOT EXISTS lexical_entries (
    entry_id TEXT PRIMARY KEY,
    term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
    part_of_speech TEXT,
    etymology TEXT,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
);
CREATE TABLE IF NOT EXISTS senses (
    sense_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES lexical_entries(entry_id) ON DELETE CASCADE,
    gloss TEXT
);
CREATE TABLE IF NOT EXISTS examples (
    sense_id TEXT NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    example TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (sense_id, position)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS pronunciations (
    entry_id TEXT NOT NULL REFERENCES lexical_entries(entry_id) ON DELETE CASCADE,
    ipa TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL,
    PRIMARY KEY (entry_id, position)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS translations (
    sense_id TEXT NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    target_term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
    part_of_speech TEXT,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id),
    position INTEGER NOT NULL,
    PRIMARY KEY (sense_id, target_term_id, position)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS synonyms (
    sense_id TEXT NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    target_term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
    part_of_speech TEXT,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id),
    position INTEGER NOT NULL,
    PRIMARY KEY (sense_id, target_term_id, position)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS relations (
    source_term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
    source_sense_id TEXT,
    relation_code INTEGER NOT NULL CHECK (relation_code BETWEEN 1 AND 12),
    target_term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
    target_sense_id TEXT,
    direction_code INTEGER NOT NULL CHECK (direction_code BETWEEN 1 AND 3),
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS relations_unique ON relations (
    source_term_id, IFNULL(source_sense_id, ''), relation_code,
    target_term_id, IFNULL(target_sense_id, ''), direction_code, provenance_id
);
CREATE TABLE IF NOT EXISTS pronunciations_words (
    term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
    phonemes TEXT NOT NULL,
    rhyme_key TEXT NOT NULL,
    PRIMARY KEY (term_id, phonemes)
) WITHOUT ROWID;
"""
)

LEXICAL_QUERY_INDEXES = """
CREATE INDEX IF NOT EXISTS lexical_entries_lookup
    ON lexical_entries(term_id, part_of_speech);
CREATE INDEX IF NOT EXISTS senses_entry_lookup
    ON senses(entry_id);
CREATE INDEX IF NOT EXISTS relations_source_lookup
    ON relations(source_term_id, relation_code);
CREATE INDEX IF NOT EXISTS relations_target_lookup
    ON relations(target_term_id, relation_code);
CREATE INDEX IF NOT EXISTS pronunciations_words_rhyme
    ON pronunciations_words(rhyme_key);
CREATE INDEX IF NOT EXISTS pronunciations_words_phonemes
    ON pronunciations_words(phonemes);
DROP TABLE IF EXISTS wordplay_fts;
CREATE VIRTUAL TABLE wordplay_fts USING fts5(
    normalized_term,
    content='',
    detail=none,
    columnsize=0,
    tokenize='unicode61 remove_diacritics 0',
    prefix='2 3 4 5 6 7 8'
);
"""

WORDPLAY_INDEX_SCHEMA = """
DROP TABLE IF EXISTS wordplay_terms;
DROP TABLE IF EXISTS pronunciation_onsets;
CREATE TABLE wordplay_terms (
  term_id INTEGER PRIMARY KEY REFERENCES lexical_terms(term_id),
  normalized_letters TEXT NOT NULL,
  letter_signature TEXT NOT NULL,
  reverse_letters TEXT NOT NULL,
  is_palindrome INTEGER NOT NULL CHECK (is_palindrome IN (0,1)),
  wordplay_eligible INTEGER NOT NULL CHECK (wordplay_eligible IN (0,1))
) WITHOUT ROWID;
CREATE INDEX wordplay_terms_anagram
  ON wordplay_terms(letter_signature, normalized_letters, term_id)
  WHERE wordplay_eligible = 1;
CREATE INDEX wordplay_terms_palindrome
  ON wordplay_terms(normalized_letters, term_id)
  WHERE is_palindrome = 1;
CREATE TABLE pronunciation_onsets (
  term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
  phonemes TEXT NOT NULL,
  onset TEXT NOT NULL,
  remainder TEXT NOT NULL,
  PRIMARY KEY (term_id, phonemes)
) WITHOUT ROWID;
CREATE INDEX pronunciation_onsets_lookup
  ON pronunciation_onsets(onset, remainder, term_id);
CREATE INDEX pronunciation_onsets_reverse
  ON pronunciation_onsets(remainder, onset, term_id);
"""


SEMANTIC_SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
    + DIMENSION_SCHEMA
    + """
CREATE TABLE IF NOT EXISTS semantic_terms (
    semantic_id INTEGER PRIMARY KEY,
    concept TEXT NOT NULL UNIQUE,
    term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
    vector_offset INTEGER NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS semantic_languages (
    language TEXT PRIMARY KEY,
    index_file TEXT NOT NULL,
    term_count INTEGER NOT NULL
);
"""
)

SEMANTIC_QUERY_INDEXES = """
CREATE INDEX IF NOT EXISTS semantic_terms_term_lookup
    ON semantic_terms(term_id);
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
            ("schema_version", SEMANTIC_SCHEMA_VERSION),
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


def create_lexical_query_indexes(connection: sqlite3.Connection) -> None:
    """Create read-path B-tree and FTS5 indexes after bulk imports complete."""

    connection.executescript(LEXICAL_QUERY_INDEXES)
    connection.execute(
        """INSERT INTO wordplay_fts(rowid, normalized_term)
        SELECT DISTINCT term.term_id, term.normalized_term
        FROM lexical_terms AS term
        JOIN pronunciations_words AS pronunciation
          ON pronunciation.term_id = term.term_id
        WHERE term.language = 'en' AND term.normalized_term <> ''"""
    )
    connection.execute(
        "INSERT INTO wordplay_fts(wordplay_fts) VALUES('integrity-check')"
    )
    connection.commit()


def create_wordplay_indexes(connection: sqlite3.Connection) -> dict[str, int]:
    """Build the v3 bounded wordplay reverse indexes after bulk imports.

    Both source tables are streamed in fixed-size batches so the full term
    corpus is never held in Python.  The returned counts feed the build
    report and dataset metadata.
    """

    connection.executescript(WORDPLAY_INDEX_SCHEMA)
    eligible_terms = 0
    palindromes = 0
    term_rows: list[tuple[int, str, str, str, int, int]] = []
    cursor = connection.execute(
        "SELECT term_id, term FROM lexical_terms WHERE language = 'en' ORDER BY term_id"
    )
    while True:
        batch = cursor.fetchmany(_WORDPLAY_BATCH_SIZE)
        if not batch:
            break
        for term_id, term in batch:
            letters = normalized_letters(str(term))
            eligible = is_wordplay_eligible(str(term))
            palindrome = int(eligible and is_palindrome(letters))
            term_rows.append(
                (
                    int(term_id),
                    letters,
                    letter_signature(letters),
                    reverse_letters(letters),
                    palindrome,
                    int(eligible),
                )
            )
            eligible_terms += int(eligible)
            palindromes += palindrome
        connection.executemany(
            "INSERT INTO wordplay_terms"
            " (term_id, normalized_letters, letter_signature, reverse_letters,"
            " is_palindrome, wordplay_eligible) VALUES (?, ?, ?, ?, ?, ?)",
            term_rows,
        )
        term_rows.clear()
    cursor.close()

    onset_rows: list[tuple[int, str, str, str]] = []
    onset_count = 0
    cursor = connection.execute(
        "SELECT term_id, phonemes FROM pronunciations_words ORDER BY term_id, phonemes"
    )
    while True:
        batch = cursor.fetchmany(_WORDPLAY_BATCH_SIZE)
        if not batch:
            break
        for term_id, phonemes in batch:
            onset, remainder = split_arpabet_onset(str(phonemes))
            onset_rows.append((int(term_id), str(phonemes), onset, remainder))
            onset_count += 1
        connection.executemany(
            "INSERT INTO pronunciation_onsets (term_id, phonemes, onset, remainder)"
            " VALUES (?, ?, ?, ?)",
            onset_rows,
        )
        onset_rows.clear()
    cursor.close()
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
        (
            ("wordplay_index_version", "1"),
            ("wordplay.eligible_terms", str(eligible_terms)),
            ("wordplay.palindromes", str(palindromes)),
            ("wordplay.pronunciation_onsets", str(onset_count)),
        ),
    )
    connection.commit()
    return {
        "eligible_terms": eligible_terms,
        "palindromes": palindromes,
        "pronunciation_onsets": onset_count,
    }


def create_semantic_query_indexes(connection: sqlite3.Connection) -> None:
    """Create semantic mapping read-path indexes after interning completes."""

    connection.executescript(SEMANTIC_QUERY_INDEXES)
    connection.commit()
