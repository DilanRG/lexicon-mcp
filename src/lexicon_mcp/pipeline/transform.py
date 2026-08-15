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
