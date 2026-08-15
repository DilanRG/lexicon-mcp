"""Transform a schema-1 dataset into schema-2 capability packs.

This is a repartition, not a rebuild.  Lexical content is copied verbatim out of
an existing verified corpus, so a pack's rows are bit-identical to the monolith's
and the two can be compared directly by the differential release gate.

The source is always attached read-only and immutable.  The corpus this reads is
the only surviving copy of its sources, so the transform must be incapable of
writing to it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .packs import (
    CORE_PACK_SCHEMA,
    LEXICAL_PACK_INDEXES,
    LEXICAL_PACK_SCHEMA,
    LanguageSize,
    PlannedPack,
)

DATASET_SCHEMA_VERSION = 4


class TransformError(RuntimeError):
    """The dataset could not be repartitioned."""


@dataclass(frozen=True, slots=True)
class PackResult:
    """What a built pack actually contains, for the manifest and for logging."""

    pack: PlannedPack
    path: Path
    raw_bytes: int
    terms: int
    stubs: int
    entries: int
    senses: int
    relations: int
    translations: int


def _read_only_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"


@contextmanager
def _writable(path: Path) -> Iterator[sqlite3.Connection]:
    """Open a new database that may ATTACH read-only URI sources.

    ``uri=True`` is required: without it SQLite refuses URI filenames in ATTACH,
    and the tempting workaround -- attaching the source as a plain path -- would
    open the corpus read-write.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}", uri=True)
    try:
        connection.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
        yield connection
    finally:
        connection.close()


def language_sizes(source: Path) -> tuple[LanguageSize, ...]:
    """Count lexical terms per language in the source corpus."""

    connection = sqlite3.connect(_read_only_uri(source), uri=True)
    try:
        rows = connection.execute(
            "SELECT language, COUNT(*) FROM lexical_terms GROUP BY language"
        ).fetchall()
    finally:
        connection.close()
    return tuple(
        sorted(
            (LanguageSize(str(language), int(count)) for language, count in rows),
            key=lambda item: (-item.terms, item.language),
        )
    )


def build_term_counts(source: Path, cache: Path) -> Path:
    """Materialize full-corpus entry and sense counts for every term.

    These are the ranking inputs relation queries order by.  Computing them from
    the whole corpus once, here, is what makes a pack's result ordering identical
    regardless of which other languages the user installed -- the invariant the
    differential gate checks.

    Kept in its own database so each pack build joins against an index instead of
    re-running correlated subqueries over the corpus.
    """

    with _writable(cache) as connection:
        connection.execute(
            "CREATE TABLE term_counts ("
            " term_id INTEGER PRIMARY KEY,"
            " entry_count INTEGER NOT NULL,"
            " sense_count INTEGER NOT NULL)"
        )
        connection.execute("ATTACH DATABASE ? AS src", (_read_only_uri(source),))
        connection.execute(
            """
            INSERT INTO term_counts (term_id, entry_count, sense_count)
            SELECT t.term_id,
                   COALESCE(e.entries, 0),
                   COALESCE(e.senses, 0)
            FROM src.lexical_terms AS t
            LEFT JOIN (
                SELECT entry.term_id AS term_id,
                       COUNT(DISTINCT entry.entry_id) AS entries,
                       COUNT(sense.sense_id) AS senses
                FROM src.lexical_entries AS entry
                LEFT JOIN src.senses AS sense ON sense.entry_id = entry.entry_id
                GROUP BY entry.term_id
            ) AS e ON e.term_id = t.term_id
            """
        )
        connection.commit()
        connection.execute("DETACH DATABASE src")
    return cache


def build_lexical_pack(
    source: Path,
    counts: Path,
    destination: Path,
    pack: PlannedPack,
    *,
    dataset_version: str,
) -> PackResult:
    """Emit one lexical pack containing the payload for *pack*'s languages."""

    if pack.capability != "lexical":
        raise TransformError(f"expected a lexical pack, got {pack.capability!r}")
    if not pack.languages:
        raise TransformError(f"pack {pack.id!r} has no languages")

    with _writable(destination) as db:
        db.executescript(LEXICAL_PACK_SCHEMA)
        db.execute("ATTACH DATABASE ? AS src", (_read_only_uri(source),))
        db.execute("ATTACH DATABASE ? AS counts", (_read_only_uri(counts),))

        db.execute("CREATE TEMP TABLE pack_language (language TEXT PRIMARY KEY)")
        db.executemany(
            "INSERT INTO pack_language VALUES (?)",
            [(language,) for language in pack.languages],
        )
        db.execute("CREATE TEMP TABLE pack_term (term_id INTEGER PRIMARY KEY)")
        db.execute(
            "INSERT INTO pack_term SELECT term_id FROM src.lexical_terms"
            " WHERE language IN (SELECT language FROM pack_language)"
        )

        db.execute("INSERT INTO provenance SELECT * FROM src.provenance")
        db.execute(
            "INSERT INTO lexical_terms SELECT * FROM src.lexical_terms"
            " WHERE term_id IN (SELECT term_id FROM pack_term)"
        )
        db.execute(
            "INSERT INTO lexical_entries SELECT * FROM src.lexical_entries"
            " WHERE term_id IN (SELECT term_id FROM pack_term)"
        )
        db.execute(
            "INSERT INTO senses SELECT s.* FROM src.senses AS s"
            " WHERE s.entry_id IN (SELECT entry_id FROM lexical_entries)"
        )
        db.execute(
            "INSERT INTO examples SELECT e.* FROM src.examples AS e"
            " WHERE e.sense_id IN (SELECT sense_id FROM senses)"
        )
        db.execute(
            "INSERT INTO pronunciations SELECT p.* FROM src.pronunciations AS p"
            " WHERE p.entry_id IN (SELECT entry_id FROM lexical_entries)"
        )
        db.execute(
            "INSERT INTO translations SELECT t.* FROM src.translations AS t"
            " WHERE t.sense_id IN (SELECT sense_id FROM senses)"
        )
        db.execute(
            "INSERT INTO synonyms SELECT y.* FROM src.synonyms AS y"
            " WHERE y.sense_id IN (SELECT sense_id FROM senses)"
        )
        # Both orientations. The runtime queries relations forward and reverse,
        # matching on target.language for the reverse direction, so a pack that
        # only held edges it sourced would silently lose half its answers.
        db.execute(
            "INSERT INTO relations SELECT * FROM src.relations"
            " WHERE source_term_id IN (SELECT term_id FROM pack_term)"
            "    OR target_term_id IN (SELECT term_id FROM pack_term)"
        )

        db.execute(
            """
            CREATE TEMP TABLE foreign_term AS
            SELECT DISTINCT term_id FROM (
                SELECT source_term_id AS term_id FROM relations
                UNION SELECT target_term_id FROM relations
                UNION SELECT target_term_id FROM translations
                UNION SELECT target_term_id FROM synonyms
            )
            WHERE term_id NOT IN (SELECT term_id FROM pack_term)
            """
        )
        db.execute(
            """
            INSERT INTO target_catalogue
            SELECT t.term_id, t.term, t.normalized_term, t.language,
                   c.entry_count, c.sense_count
            FROM src.lexical_terms AS t
            JOIN counts.term_counts AS c ON c.term_id = t.term_id
            WHERE t.term_id IN (SELECT term_id FROM foreign_term)
            """
        )

        for key, value in (
            ("schema_version", str(DATASET_SCHEMA_VERSION)),
            ("dataset_version", dataset_version),
            ("pack_id", pack.id),
            ("capability", pack.capability),
            ("languages", ",".join(pack.languages)),
        ):
            db.execute("INSERT INTO metadata VALUES (?, ?)", (key, value))

        result = {
            table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "lexical_terms",
                "target_catalogue",
                "lexical_entries",
                "senses",
                "relations",
                "translations",
            )
        }
        _assert_pack_is_closed(db)
        db.executescript(LEXICAL_PACK_INDEXES)
        db.commit()
        db.execute("DETACH DATABASE src")
        db.execute("DETACH DATABASE counts")
        db.execute("VACUUM")

    return PackResult(
        pack=pack,
        path=destination,
        raw_bytes=destination.stat().st_size,
        terms=result["lexical_terms"],
        stubs=result["target_catalogue"],
        entries=result["lexical_entries"],
        senses=result["senses"],
        relations=result["relations"],
        translations=result["translations"],
    )


def _assert_pack_is_closed(db: sqlite3.Connection) -> None:
    """Fail if any edge references a term the pack cannot name.

    A pack must be self-contained: every term id an edge points at resolves
    either to a local headword or to a catalogue stub. Without this the runtime
    would have to open another shard just to render a result, which is the
    property the whole design depends on not being true.
    """

    dangling = db.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT source_term_id AS term_id FROM relations
            UNION SELECT target_term_id FROM relations
            UNION SELECT target_term_id FROM translations
            UNION SELECT target_term_id FROM synonyms
        )
        WHERE term_id NOT IN (SELECT term_id FROM lexical_terms)
          AND term_id NOT IN (SELECT term_id FROM target_catalogue)
        """
    ).fetchone()[0]
    if dangling:
        raise TransformError(f"pack has {dangling} edge targets it cannot resolve")

    # Stubs and headwords must not overlap, or a term would be reachable through
    # two tables with two different truths about whether its payload is present.
    overlap = db.execute(
        "SELECT COUNT(*) FROM target_catalogue"
        " WHERE term_id IN (SELECT term_id FROM lexical_terms)"
    ).fetchone()[0]
    if overlap:
        raise TransformError(f"pack has {overlap} terms in both lexical_terms and the catalogue")


def build_core_pack(
    source: Path,
    counts: Path,
    destination: Path,
    *,
    dataset_version: str,
    semantic_languages: Sequence[str],
    pronunciation_languages: Sequence[str],
    wordplay_languages: Sequence[str],
) -> Path:
    """Emit the always-installed catalogue of languages and capabilities.

    Payload never belongs here. This is what `lexicon-data languages` reads to
    answer what exists, what is installed, and what is merely unavailable
    upstream -- distinctions a caller cannot make from an empty result.
    """

    semantic = set(semantic_languages)
    pronunciation = set(pronunciation_languages)
    wordplay = set(wordplay_languages)
    with _writable(destination) as db:
        db.executescript(CORE_PACK_SCHEMA)
        db.execute("ATTACH DATABASE ? AS src", (_read_only_uri(source),))
        db.execute("ATTACH DATABASE ? AS counts", (_read_only_uri(counts),))
        # Summed from the per-term counts rather than recomputed. Grouping
        # COUNT(DISTINCT) over lexical_terms joined to entries joined to senses
        # builds an enormous temporary b-tree and does not finish in usable time
        # on the real corpus; this is one indexed pass over already-derived data.
        rows = db.execute(
            """
            SELECT t.language, COUNT(*), SUM(c.entry_count), SUM(c.sense_count)
            FROM src.lexical_terms AS t
            JOIN counts.term_counts AS c ON c.term_id = t.term_id
            GROUP BY t.language
            """
        ).fetchall()
        translation_rows = dict(
            db.execute(
                """
                SELECT t.language, COUNT(*)
                FROM src.translations AS tr
                JOIN src.senses AS s ON s.sense_id = tr.sense_id
                JOIN src.lexical_entries AS e ON e.entry_id = s.entry_id
                JOIN src.lexical_terms AS t ON t.term_id = e.term_id
                GROUP BY t.language
                """
            ).fetchall()
        )
        # Rows a language participates in, counted once each. A pack retains an
        # edge when it owns either endpoint, so an edge with both endpoints in
        # one language must not count twice or this would not predict pack size.
        # Inclusion-exclusion rather than a UNION so the corpus is scanned in
        # grouped passes instead of deduplicating ~38M endpoint pairs.
        relation_rows = dict(
            db.execute(
                """
                SELECT language, SUM(tally) FROM (
                    SELECT t.language AS language, COUNT(*) AS tally
                    FROM src.relations AS r
                    JOIN src.lexical_terms AS t ON t.term_id = r.source_term_id
                    GROUP BY t.language
                    UNION ALL
                    SELECT t.language, COUNT(*)
                    FROM src.relations AS r
                    JOIN src.lexical_terms AS t ON t.term_id = r.target_term_id
                    GROUP BY t.language
                    UNION ALL
                    SELECT source.language, -COUNT(*)
                    FROM src.relations AS r
                    JOIN src.lexical_terms AS source ON source.term_id = r.source_term_id
                    JOIN src.lexical_terms AS target ON target.term_id = r.target_term_id
                    WHERE source.language = target.language
                    GROUP BY source.language
                )
                GROUP BY language
                """
            ).fetchall()
        )
        db.executemany(
            "INSERT INTO language_catalogue VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    str(language),
                    int(terms),
                    int(entries),
                    int(senses),
                    int(translation_rows.get(language, 0)),
                    int(relation_rows.get(language, 0)),
                    1 if language in semantic else 0,
                    1 if language in pronunciation else 0,
                    1 if language in wordplay else 0,
                )
                for language, terms, entries, senses in rows
            ],
        )
        for key, value in (
            ("schema_version", str(DATASET_SCHEMA_VERSION)),
            ("dataset_version", dataset_version),
            ("pack_id", "core"),
            ("capability", "core"),
        ):
            db.execute("INSERT INTO metadata VALUES (?, ?)", (key, value))
        db.commit()
        db.execute("DETACH DATABASE src")
        db.execute("DETACH DATABASE counts")
        db.execute("VACUUM")
    return destination


SEMANTIC_PACK_SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE provenance (
    provenance_id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    source_license TEXT NOT NULL,
    source_url TEXT NOT NULL
);
CREATE TABLE lexical_terms (
    term_id INTEGER PRIMARY KEY,
    term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    language TEXT NOT NULL,
    UNIQUE (language, normalized_term, term)
);
CREATE TABLE semantic_terms (
    semantic_id INTEGER PRIMARY KEY,
    concept TEXT NOT NULL UNIQUE,
    term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
    vector_offset INTEGER NOT NULL UNIQUE
);
CREATE TABLE semantic_languages (
    language TEXT PRIMARY KEY,
    index_file TEXT NOT NULL,
    term_count INTEGER NOT NULL
);
CREATE INDEX semantic_terms_term_lookup ON semantic_terms(term_id);
"""


@dataclass(frozen=True, slots=True)
class SemanticPackResult:
    language: str
    mapping: Path
    vectors: Path
    index: Path
    terms: int
    dimensions: int


def build_semantic_pack(
    semantic_root: Path,
    destination: Path,
    language: str,
    *,
    dataset_version: str,
) -> SemanticPackResult:
    """Emit the three artifacts one semantic language needs.

    ``semantic_id`` is preserved, because the per-language USearch index keys on
    it -- so that index ships verbatim and is never rebuilt.  Only
    ``vector_offset`` is rewritten, since the pack carries its own gathered
    vector file rather than the 5.5 GB global one.

    Vectors are gathered in source-offset order so the global file is read
    forwards rather than seeking randomly across it.
    """

    import shutil as _shutil

    import numpy as np

    mapping_source = semantic_root / "mapping.sqlite3"
    reader = sqlite3.connect(_read_only_uri(mapping_source), uri=True)
    try:
        metadata = dict(reader.execute("SELECT key, value FROM metadata"))
        dimensions = int(metadata["dimensions"])
        vector_source = semantic_root / metadata["vector_file"]
        shard = reader.execute(
            "SELECT index_file, term_count FROM semantic_languages WHERE language = ?",
            (language,),
        ).fetchone()
        if shard is None:
            raise TransformError(f"no semantic vectors for {language!r}")
        rows = reader.execute(
            """
            SELECT s.semantic_id, s.concept, s.term_id, s.vector_offset,
                   t.term, t.normalized_term, t.language
            FROM semantic_terms AS s
            JOIN lexical_terms AS t ON t.term_id = s.term_id
            WHERE t.language = ?
            ORDER BY s.vector_offset
            """,
            (language,),
        ).fetchall()
        provenance = reader.execute("SELECT * FROM provenance").fetchall()
    finally:
        reader.close()

    if not rows:
        raise TransformError(f"semantic language {language!r} has no terms")
    if len(rows) != int(shard[1]):
        raise TransformError(
            f"semantic language {language!r} has {len(rows)} terms, "
            f"but its index declares {shard[1]}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    vectors_path = destination / "vectors.f16"
    row_bytes = dimensions * 2
    total = vector_source.stat().st_size // row_bytes
    matrix: Any = np.memmap(vector_source, dtype="<f2", mode="r", shape=(total, dimensions))
    with vectors_path.open("wb") as sink:
        for _semantic_id, _concept, _term_id, offset, *_ in rows:
            if not 0 <= int(offset) < total:
                raise TransformError("semantic vector offset is outside the matrix")
            sink.write(matrix[int(offset)].tobytes())
    del matrix

    index_source = semantic_root / str(shard[0])
    index_path = destination / f"{language.replace('-', '_')}.usearch"
    _shutil.copyfile(index_source, index_path)

    mapping_path = destination / "mapping.sqlite3"
    with _writable(mapping_path) as db:
        db.executescript(SEMANTIC_PACK_SCHEMA)
        db.executemany("INSERT INTO provenance VALUES (?,?,?,?)", provenance)
        db.executemany(
            "INSERT INTO lexical_terms VALUES (?,?,?,?)",
            [(row[2], row[4], row[5], row[6]) for row in rows],
        )
        db.executemany(
            "INSERT INTO semantic_terms VALUES (?,?,?,?)",
            [(row[0], row[1], row[2], position) for position, row in enumerate(rows)],
        )
        db.execute(
            "INSERT INTO semantic_languages VALUES (?,?,?)",
            (language, index_path.name, len(rows)),
        )
        for key, value in (
            ("schema_version", str(DATASET_SCHEMA_VERSION)),
            ("dataset_version", dataset_version),
            ("capability", "semantic"),
            ("language", language),
            ("dimensions", str(dimensions)),
            ("vector_dtype", metadata["vector_dtype"]),
            ("vector_file", vectors_path.name),
            ("index_dtype", metadata["index_dtype"]),
            ("index_metric", metadata["index_metric"]),
            ("connectivity", metadata["connectivity"]),
            ("expansion_add", metadata["expansion_add"]),
            ("expansion_search", metadata["expansion_search"]),
            ("source", metadata["source"]),
            ("source_license", metadata["source_license"]),
            ("source_url", metadata["source_url"]),
            ("term_count", str(len(rows))),
        ):
            db.execute("INSERT INTO metadata VALUES (?, ?)", (key, value))
        db.commit()
        db.execute("VACUUM")

    if vectors_path.stat().st_size != len(rows) * row_bytes:
        raise TransformError("gathered vector file does not match its term count")
    return SemanticPackResult(
        language=language,
        mapping=mapping_path,
        vectors=vectors_path,
        index=index_path,
        terms=len(rows),
        dimensions=dimensions,
    )


WORDPLAY_PACK_SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE lexical_terms (
    term_id INTEGER PRIMARY KEY,
    term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    language TEXT NOT NULL,
    UNIQUE (language, normalized_term, term)
);
CREATE TABLE pronunciations_words (
    term_id INTEGER NOT NULL,
    phonemes TEXT NOT NULL,
    rhyme_key TEXT NOT NULL,
    PRIMARY KEY (term_id, phonemes)
) WITHOUT ROWID;
CREATE TABLE wordplay_terms (
    term_id INTEGER PRIMARY KEY,
    normalized_letters TEXT NOT NULL,
    letter_signature TEXT NOT NULL,
    reverse_letters TEXT NOT NULL,
    is_palindrome INTEGER NOT NULL CHECK (is_palindrome IN (0,1)),
    wordplay_eligible INTEGER NOT NULL CHECK (wordplay_eligible IN (0,1))
) WITHOUT ROWID;
CREATE TABLE pronunciation_onsets (
    term_id INTEGER NOT NULL,
    phonemes TEXT NOT NULL,
    onset TEXT NOT NULL,
    remainder TEXT NOT NULL,
    PRIMARY KEY (term_id, phonemes)
) WITHOUT ROWID;
CREATE VIRTUAL TABLE wordplay_fts USING fts5(
    normalized_term,
    content='',
    detail=none,
    columnsize=0,
    tokenize='unicode61 remove_diacritics 0',
    prefix='2 3 4 5 6 7 8'
);
"""

WORDPLAY_PACK_INDEXES = """
CREATE INDEX pronunciations_words_rhyme ON pronunciations_words(rhyme_key, term_id);
CREATE INDEX pronunciations_words_phonemes ON pronunciations_words(phonemes, term_id);
CREATE INDEX wordplay_terms_anagram
    ON wordplay_terms(letter_signature, normalized_letters, term_id)
    WHERE wordplay_eligible = 1;
CREATE INDEX wordplay_terms_palindrome
    ON wordplay_terms(normalized_letters, term_id)
    WHERE is_palindrome = 1;
CREATE INDEX pronunciation_onsets_lookup
    ON pronunciation_onsets(onset, remainder, term_id);
CREATE INDEX pronunciation_onsets_reverse
    ON pronunciation_onsets(remainder, onset, term_id);
"""


def build_wordplay_pack(
    source: Path,
    destination: Path,
    *,
    dataset_version: str,
    language: str = "en",
) -> PackResult:
    """Emit the English rhyme, anagram, spoonerism and homophone indexes.

    Self-contained on purpose. Every wordplay query in the runtime joins these
    indexes to `lexical_terms`, so the pack carries its own copy of that
    language's terms; somebody who wants rhymes and anagrams then needs this
    pack alone rather than the full 2 GB English dictionary.

    The reverse indexes are copied rather than recomputed so they are identical
    to the corpus they came from. The FTS index cannot be: it is contentless, so
    its rows cannot be read back out, and it is rebuilt from the same statement
    the corpus build used.
    """

    with _writable(destination) as db:
        db.executescript(WORDPLAY_PACK_SCHEMA)
        db.execute("ATTACH DATABASE ? AS src", (_read_only_uri(source),))
        db.execute(
            "INSERT INTO lexical_terms SELECT * FROM src.lexical_terms WHERE language = ?",
            (language,),
        )
        db.execute(
            "INSERT INTO pronunciations_words SELECT p.* FROM src.pronunciations_words AS p"
            " WHERE p.term_id IN (SELECT term_id FROM lexical_terms)"
        )
        db.execute(
            "INSERT INTO wordplay_terms SELECT w.* FROM src.wordplay_terms AS w"
            " WHERE w.term_id IN (SELECT term_id FROM lexical_terms)"
        )
        db.execute(
            "INSERT INTO pronunciation_onsets SELECT o.* FROM src.pronunciation_onsets AS o"
            " WHERE o.term_id IN (SELECT term_id FROM lexical_terms)"
        )
        db.execute(
            """INSERT INTO wordplay_fts(rowid, normalized_term)
            SELECT DISTINCT term.term_id, term.normalized_term
            FROM lexical_terms AS term
            JOIN pronunciations_words AS pronunciation
              ON pronunciation.term_id = term.term_id
            WHERE term.language = ? AND term.normalized_term <> ''""",
            (language,),
        )
        db.execute("INSERT INTO wordplay_fts(wordplay_fts) VALUES('integrity-check')")

        counts = {
            table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("lexical_terms", "pronunciations_words", "wordplay_terms")
        }
        if not counts["wordplay_terms"]:
            raise TransformError(f"no wordplay indexes for {language!r} in the corpus")
        for key, value in (
            ("schema_version", str(DATASET_SCHEMA_VERSION)),
            ("dataset_version", dataset_version),
            ("pack_id", f"wordplay-{language}"),
            ("capability", "wordplay"),
            ("languages", language),
        ):
            db.execute("INSERT INTO metadata VALUES (?, ?)", (key, value))
        db.executescript(WORDPLAY_PACK_INDEXES)
        db.commit()
        db.execute("DETACH DATABASE src")
        db.execute("VACUUM")

    return PackResult(
        pack=PlannedPack(f"wordplay-{language}", "wordplay", (language,), 0),
        path=destination,
        raw_bytes=destination.stat().st_size,
        terms=counts["lexical_terms"],
        stubs=0,
        entries=0,
        senses=0,
        relations=0,
        translations=counts["pronunciations_words"],
    )
