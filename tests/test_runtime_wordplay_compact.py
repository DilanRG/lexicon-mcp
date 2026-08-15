from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from lexicon_mcp.pipeline.schema import create_wordplay_indexes
from lexicon_mcp.runtime.wordplay import SQLiteWordplaySearch


def _create_database(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES ('schema_version', '3');
            CREATE TABLE lexical_terms (
                term_id INTEGER PRIMARY KEY,
                term TEXT NOT NULL,
                normalized_term TEXT NOT NULL,
                language TEXT NOT NULL,
                UNIQUE(language, normalized_term, term)
            );
            CREATE TABLE provenance (
                provenance_id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                source_license TEXT NOT NULL,
                source_url TEXT NOT NULL,
                UNIQUE (source, source_license, source_url)
            );
            CREATE TABLE lexical_entries (
                entry_id TEXT PRIMARY KEY,
                term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
                part_of_speech TEXT,
                etymology TEXT,
                provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
            );
            CREATE TABLE senses (
                sense_id TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL REFERENCES lexical_entries(entry_id) ON DELETE CASCADE,
                gloss TEXT
            );
            CREATE TABLE pronunciations_words (
                term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
                phonemes TEXT NOT NULL,
                rhyme_key TEXT NOT NULL,
                PRIMARY KEY(term_id, phonemes)
            );
            CREATE INDEX lexical_terms_lookup
                ON lexical_terms(language, normalized_term);
            CREATE INDEX pronunciations_words_rhyme
                ON pronunciations_words(rhyme_key);
            CREATE INDEX pronunciations_words_phonemes
                ON pronunciations_words(phonemes);
            """
        )
        for term_id, (term, normalized, phonemes, rhyme_key) in enumerate(rows, start=1):
            connection.execute(
                "INSERT INTO lexical_terms VALUES (?, ?, ?, 'en')",
                (term_id, term, normalized),
            )
            connection.execute(
                "INSERT INTO pronunciations_words VALUES (?, ?, ?)",
                (term_id, phonemes, rhyme_key),
            )
        connection.executescript(
            """
            CREATE VIRTUAL TABLE wordplay_fts USING fts5(
                normalized_term,
                content='',
                detail=none,
                columnsize=0,
                tokenize='unicode61 remove_diacritics 0',
                prefix='2 3 4 5 6 7 8'
            );
            INSERT INTO wordplay_fts(rowid, normalized_term)
            SELECT DISTINCT term.term_id, term.normalized_term
            FROM lexical_terms AS term
            JOIN pronunciations_words AS pronunciation
              ON pronunciation.term_id = term.term_id;
            """
        )
        create_wordplay_indexes(connection)
        connection.commit()


@pytest.fixture()
def compact_database(tmp_path: Path) -> Path:
    path = tmp_path / "compact.sqlite3"
    _create_database(
        path,
        [
            ("cat", "cat", "K AE1 T", "AE1 T"),
            ("at", "at", "AE1 T", "AE1 T"),
            ("bat", "bat", "B AE1 T", "AE1 T"),
            ("scat", "scat", "S K AE1 T", "AE1 T"),
            ("cats", "cats", "K AE1 T S", "AE1 T S"),
            ("cut", "cut", "K AH1 T", "AH1 T"),
            ("kit", "kit", "K IH1 T", "IH1 T"),
            ("knight", "knight", "N AY1 T", "AY1 T"),
            ("night", "night", "N AY1 T", "AY1 T"),
            ("hallo", "hallo", "HH AE1 L OW0", "AE1 L OW0"),
            ("hello", "hello", "HH AH0 L OW1", "OW1"),
            ("help", "help", "HH EH1 L P", "EH1 L P"),
            ("serendipity", "serendipity", "S EH2 R AH0 N D IH1 P AH0 T IY0", "IH1 P AH0 T IY0"),
            ("hullo", "hullo", "HH AH0 L OW1", "OW1"),
            ("100%", "100%", "W AH1 N", "AH1 N"),
            ("100%proof", "100%proof", "P R UW1 F", "UW1 F"),
            ("100xproof", "100xproof", "P R UW1 F", "UW1 F"),
            ("new yorker", "new yorker", "N UW1 Y AO1 R K ER0", "AO1 R K ER0"),
            ("new yorkshire", "new yorkshire", "N UW1 Y AO1 R K SH ER0", "AO1 R K SH ER0"),
            ("under_score", "under_score", "AH1 N D ER0", "AH1 N D ER0"),
            (
                "under_score_more",
                "under_score_more",
                "AH1 N D ER0 M AO1 R",
                "AO1 R",
            ),
            ("underxscore_more", "underxscore_more", "S K AO1 R", "AO1 R"),
        ],
    )
    return path


def _terms(search: SQLiteWordplaySearch, mode: str, text: str, limit: int = 20) -> list[str]:
    return [item["term"] for item in search.search(mode, text, limit=limit)]


def test_indexed_pronunciation_modes_are_deterministic_and_exclude_self(
    compact_database: Path,
) -> None:
    with SQLiteWordplaySearch(compact_database) as search:
        assert _terms(search, "rhyme", "cat") == ["at", "bat", "scat"]
        assert _terms(search, "rhyme", "cat", limit=2) == ["at", "bat"]
        assert _terms(search, "near_rhyme", "cat") == [
            "cats",
            "cut",
            "kit",
            "knight",
            "night",
        ]
        assert _terms(search, "sounds_like", "knight") == ["night"]
        assert _terms(search, "rhyme", "not-in-cmudict") == []


def test_pattern_modes_translate_only_question_and_star_and_escape_sql_like(
    compact_database: Path,
) -> None:
    with SQLiteWordplaySearch(compact_database) as search:
        assert _terms(search, "spelled_like", "h?llo") == ["hallo", "hello", "hullo"]
        assert _terms(search, "spelled_like", "h[ae]llo") == []
        assert _terms(search, "prefix", "hel") == ["hello", "help"]
        assert _terms(search, "prefix", "seren") == ["serendipity"]
        assert _terms(search, "prefix", "100%") == ["100%proof"]
        assert _terms(search, "prefix", "new y") == ["new yorker", "new yorkshire"]
        assert _terms(search, "spelled_like", "under_score*") == [
            "under_score",
            "under_score_more",
        ]


def test_result_contract_and_input_limits(compact_database: Path) -> None:
    with SQLiteWordplaySearch(compact_database) as search:
        result = search.search("sounds_like", "night", limit=1)[0]
        assert result == {
            "term": "knight",
            "language": "en",
            "mode": "sounds_like",
            "phonemes": "N AY1 T",
            "sense_scope": "unsensed",
            "provenance": {
                "source": "CMU Pronouncing Dictionary",
                "license": "CMUdict license",
                "url": "https://github.com/cmusphinx/cmudict",
            },
        }
        with pytest.raises(ValueError, match="between 1 and 100"):
            search.search("rhyme", "cat", limit=101)
        with pytest.raises(ValueError, match="mode must be one of"):
            search.search("anagram", "cat")


def test_rejects_non_v3_or_non_compact_schema(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES ('schema_version', '2');
            CREATE TABLE lexical_terms (
                term_id INTEGER PRIMARY KEY, term TEXT, normalized_term TEXT, language TEXT
            );
            CREATE TABLE pronunciations_words (term_id INTEGER, phonemes TEXT, rhyme_key TEXT);
            CREATE VIRTUAL TABLE wordplay_fts USING fts5(normalized_term, content='');
            """
        )
    with pytest.raises(RuntimeError, match="schema version 3"):
        SQLiteWordplaySearch(path)


@pytest.mark.performance
def test_indexed_sounds_like_microbenchmark(tmp_path: Path) -> None:
    path = tmp_path / "large-compact.sqlite3"
    rows = [
        (
            f"term{index:05d}",
            f"term{index:05d}",
            f"T ER1 M {index:05d}",
            f"ER1 M {index:05d}",
        )
        for index in range(20_000)
    ]
    rows.append(("homophone", "homophone", "T ER1 M 19999", "ER1 M 19999"))
    _create_database(path, rows)

    with SQLiteWordplaySearch(path) as search:
        assert _terms(search, "sounds_like", "term19999") == ["homophone"]
        started = time.perf_counter()
        for _iteration in range(100):
            assert _terms(search, "sounds_like", "term19999") == ["homophone"]
        elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"100 indexed sounds-like queries took {elapsed:.3f}s"
