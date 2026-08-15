"""Repartitioning a schema-1 corpus into schema-2 packs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lexicon_mcp.pipeline.packs import LanguageSize, PlannedPack
from lexicon_mcp.pipeline.schema import create_lexical_schema
from lexicon_mcp.pipeline.transform import (
    TransformError,
    build_core_pack,
    build_lexical_pack,
    build_term_counts,
    language_sizes,
)

VERSION = "data-v2.0.0"

# term_id -> (term, language). English has two headwords, French one, Welsh one.
TERMS = {
    1: ("dog", "en"),
    2: ("hound", "en"),
    3: ("chien", "fr"),
    4: ("ci", "cy"),
}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A miniature schema-1 corpus exercising every cross-language edge shape."""

    path = tmp_path / "lexicon.sqlite3"
    connection = sqlite3.connect(path)
    create_lexical_schema(connection, "data-v1.1.0")
    connection.execute(
        "INSERT INTO provenance VALUES (1,'fixture','CC0-1.0','https://fixtures.invalid')"
    )
    for term_id, (term, language) in TERMS.items():
        connection.execute(
            "INSERT INTO lexical_terms VALUES (?,?,?,?)", (term_id, term, term, language)
        )
    # 'dog' carries two entries and three senses; the others carry one each. The
    # asymmetry is what makes the catalogue's full-corpus counts checkable.
    entries = [("e1", 1), ("e1b", 1), ("e2", 2), ("e3", 3), ("e4", 4)]
    for entry_id, term_id in entries:
        connection.execute(
            "INSERT INTO lexical_entries VALUES (?,?,?,?,?)",
            (entry_id, term_id, "noun", None, 1),
        )
    senses = [("s1", "e1"), ("s1b", "e1"), ("s1c", "e1b"), ("s2", "e2"), ("s3", "e3"), ("s4", "e4")]
    for sense_id, entry_id in senses:
        connection.execute(
            "INSERT INTO senses VALUES (?,?,?)", (sense_id, entry_id, f"gloss {sense_id}")
        )
    connection.execute("INSERT INTO examples VALUES ('s1','the dog barks',0)")
    connection.execute("INSERT INTO pronunciations VALUES ('e1','/dɒg/','GB',0)")
    # en -> fr translation: the target lives in another pack.
    connection.execute("INSERT INTO translations VALUES ('s1',3,'noun',1,0)")
    # en -> en synonym: same-language, as every synonym in the real corpus is.
    connection.execute("INSERT INTO synonyms VALUES ('s1',2,'noun',1,0)")
    relations = [
        (1, "s1", 1, 2, "s2", 1, 1),  # en -> en
        (1, "s1", 1, 3, "s3", 1, 1),  # en -> fr, English is the source
        (3, "s3", 1, 4, "s4", 1, 1),  # fr -> cy, English is not involved
        (4, "s4", 1, 1, "s1", 1, 1),  # cy -> en, English is only the target
    ]
    for row in relations:
        connection.execute("INSERT INTO relations VALUES (?,?,?,?,?,?,?)", row)
    connection.commit()
    connection.close()
    return path


def read(path: Path, query: str) -> list[tuple]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def build_en(corpus: Path, tmp_path: Path) -> Path:
    counts = build_term_counts(corpus, tmp_path / "counts.sqlite3")
    destination = tmp_path / "packs" / "lexical-en.sqlite3"
    build_lexical_pack(
        corpus,
        counts,
        destination,
        PlannedPack("lexical-en", "lexical", ("en",), 0),
        dataset_version=VERSION,
    )
    return destination


def test_language_sizes_are_counted_largest_first(corpus: Path) -> None:
    assert language_sizes(corpus) == (
        LanguageSize("en", 2),
        LanguageSize("cy", 1),
        LanguageSize("fr", 1),
    )


def test_pack_owns_only_its_own_headwords(corpus: Path, tmp_path: Path) -> None:
    pack = build_en(corpus, tmp_path)

    assert read(pack, "SELECT term_id, term, language FROM lexical_terms ORDER BY term_id") == [
        (1, "dog", "en"),
        (2, "hound", "en"),
    ]


def test_pack_carries_edges_in_both_orientations(corpus: Path, tmp_path: Path) -> None:
    """Reverse relation queries match on target language, so both sides ship."""

    pack = build_en(corpus, tmp_path)

    edges = read(pack, "SELECT source_term_id, target_term_id FROM relations ORDER BY 1, 2")

    assert edges == [
        (1, 2),  # en -> en
        (1, 3),  # en -> fr, English sources it
        (4, 1),  # cy -> en, English is only the target
    ]
    # The fr -> cy edge touches no English term and must not be here.
    assert (3, 4) not in edges


def test_catalogue_names_foreign_targets_with_full_corpus_counts(
    corpus: Path, tmp_path: Path
) -> None:
    pack = build_en(corpus, tmp_path)

    catalogue = read(
        pack,
        "SELECT term_id, term, language, entry_count, sense_count"
        " FROM target_catalogue ORDER BY term_id",
    )

    # 'chien' arrives through both the translation and a relation; 'ci' only as
    # the source of the cy -> en edge. Both are named, neither is a headword.
    assert catalogue == [(3, "chien", "fr", 1, 1), (4, "ci", "cy", 1, 1)]


def test_catalogue_counts_come_from_the_whole_corpus_not_the_pack(
    corpus: Path, tmp_path: Path
) -> None:
    """Ranking must not shift with the installed set, so counts are global."""

    counts = build_term_counts(corpus, tmp_path / "counts.sqlite3")
    destination = tmp_path / "packs" / "lexical-cy.sqlite3"
    build_lexical_pack(
        corpus,
        counts,
        destination,
        PlannedPack("lexical-cy", "lexical", ("cy",), 0),
        dataset_version=VERSION,
    )

    # 'dog' has 2 entries and 3 senses in the corpus. Welsh sees none of that
    # payload, but must still rank it as though it had.
    assert read(
        destination,
        "SELECT term, entry_count, sense_count FROM target_catalogue WHERE term_id = 1",
    ) == [("dog", 2, 3)]


def test_dependent_rows_follow_their_senses(corpus: Path, tmp_path: Path) -> None:
    pack = build_en(corpus, tmp_path)

    assert read(pack, "SELECT sense_id, example FROM examples") == [("s1", "the dog barks")]
    assert read(pack, "SELECT entry_id, ipa FROM pronunciations") == [("e1", "/dɒg/")]
    assert read(pack, "SELECT sense_id, target_term_id FROM translations") == [("s1", 3)]
    assert read(pack, "SELECT sense_id, target_term_id FROM synonyms") == [("s1", 2)]
    # A pack holds only the senses of the entries it owns.
    assert {row[0] for row in read(pack, "SELECT sense_id FROM senses")} == {
        "s1",
        "s1b",
        "s1c",
        "s2",
    }


def test_pack_records_its_own_identity(corpus: Path, tmp_path: Path) -> None:
    pack = build_en(corpus, tmp_path)

    metadata = dict(read(pack, "SELECT key, value FROM metadata"))

    assert metadata["pack_id"] == "lexical-en"
    assert metadata["capability"] == "lexical"
    assert metadata["languages"] == "en"
    assert metadata["dataset_version"] == VERSION


def test_bundled_pack_holds_every_requested_language(corpus: Path, tmp_path: Path) -> None:
    counts = build_term_counts(corpus, tmp_path / "counts.sqlite3")
    destination = tmp_path / "packs" / "bundle.sqlite3"

    result = build_lexical_pack(
        corpus,
        counts,
        destination,
        PlannedPack("lexical-bundle-001", "lexical", ("cy", "fr"), 0),
        dataset_version=VERSION,
    )

    assert result.terms == 2
    assert {row[0] for row in read(destination, "SELECT language FROM lexical_terms")} == {
        "cy",
        "fr",
    }
    # The fr -> cy edge is internal to this bundle, so neither endpoint is a stub.
    assert read(destination, "SELECT term_id FROM target_catalogue ORDER BY term_id") == [(1,)]


def test_a_pack_never_holds_a_term_as_both_headword_and_stub(
    corpus: Path, tmp_path: Path
) -> None:
    pack = build_en(corpus, tmp_path)

    assert read(
        pack,
        "SELECT COUNT(*) FROM target_catalogue"
        " WHERE term_id IN (SELECT term_id FROM lexical_terms)",
    ) == [(0,)]


def test_transform_refuses_a_pack_it_cannot_describe(corpus: Path, tmp_path: Path) -> None:
    counts = build_term_counts(corpus, tmp_path / "counts.sqlite3")

    with pytest.raises(TransformError, match="expected a lexical pack"):
        build_lexical_pack(
            corpus,
            counts,
            tmp_path / "bad.sqlite3",
            PlannedPack("semantic-en", "semantic", ("en",), 0),
            dataset_version=VERSION,
        )

    with pytest.raises(TransformError, match="no languages"):
        build_lexical_pack(
            corpus,
            counts,
            tmp_path / "bad.sqlite3",
            PlannedPack("lexical-empty", "lexical", (), 0),
            dataset_version=VERSION,
        )


def test_transform_never_writes_to_the_source_corpus(corpus: Path, tmp_path: Path) -> None:
    before = corpus.stat().st_mtime_ns, corpus.stat().st_size

    build_en(corpus, tmp_path)

    assert (corpus.stat().st_mtime_ns, corpus.stat().st_size) == before


def test_core_pack_reports_capability_coverage(corpus: Path, tmp_path: Path) -> None:
    destination = build_core_pack(
        corpus,
        build_term_counts(corpus, tmp_path / "counts.sqlite3"),
        tmp_path / "core.sqlite3",
        dataset_version=VERSION,
        semantic_languages=["en", "fr"],
        pronunciation_languages=["en"],
        wordplay_languages=["en"],
    )

    rows = read(
        destination,
        "SELECT language, term_count, entry_count, sense_count, translation_count,"
        " relation_count, has_semantic, has_pronunciation, has_wordplay"
        " FROM language_catalogue ORDER BY language",
    )

    # relation_count is rows the language participates in, counted once each:
    # English sources two edges and is the target of two, but the en -> en edge
    # is the same row on both sides, so it is 3 rather than 4.
    assert rows == [
        ("cy", 1, 1, 1, 0, 2, 0, 0, 0),
        ("en", 2, 3, 4, 1, 3, 1, 1, 1),
        ("fr", 1, 1, 1, 0, 2, 1, 0, 0),
    ]


MONOLITH_FORWARD = """
WITH candidates AS (
    SELECT target.term AS target_term,
           target.language AS target_language,
           (SELECT COUNT(*) FROM lexical_entries AS e
             WHERE e.term_id = target.term_id) AS target_entry_count,
           (SELECT COUNT(*) FROM senses AS s
              JOIN lexical_entries AS e2 ON e2.entry_id = s.entry_id
             WHERE e2.term_id = target.term_id) AS target_sense_count,
           target.normalized_term AS target_normalized,
           relation.target_sense_id, provenance.provenance_id,
           relation.direction_code,
           ROW_NUMBER() OVER (
               PARTITION BY target.term_id
               ORDER BY CASE WHEN relation.source_sense_id IS NULL
                              AND relation.target_sense_id IS NULL THEN 1 ELSE 0 END,
                        relation.source_sense_id, relation.target_sense_id,
                        provenance.provenance_id, relation.direction_code
           ) AS target_variant_rank
    FROM relations AS relation
    JOIN lexical_terms AS source ON source.term_id = relation.source_term_id
    JOIN lexical_terms AS target ON target.term_id = relation.target_term_id
    JOIN provenance ON provenance.provenance_id = relation.provenance_id
    WHERE source.normalized_term = ? AND source.language = ?
      AND relation.relation_code = ?
)
SELECT * FROM candidates
ORDER BY target_variant_rank, target_sense_count DESC, target_entry_count DESC,
         (LENGTH(target_normalized) - LENGTH(REPLACE(target_normalized, ' ', ''))),
         LENGTH(target_normalized), target_language, target_normalized,
         target_term, target_sense_id, provenance_id, direction_code
LIMIT ?
"""


def compare_key(rows) -> list[tuple]:
    return [
        (
            row["target_term"],
            row["target_language"],
            row["target_entry_count"],
            row["target_sense_count"],
            row["target_variant_rank"],
        )
        for row in rows
    ]


def test_pack_relations_match_the_monolith_exactly(corpus: Path, tmp_path: Path) -> None:
    """The differential gate, at unit scale."""

    from lexicon_mcp.runtime.pack_queries import relation_rows

    pack = build_en(corpus, tmp_path)

    monolith = sqlite3.connect(f"file:{corpus.as_posix()}?mode=ro", uri=True)
    monolith.row_factory = sqlite3.Row
    expected = compare_key(monolith.execute(MONOLITH_FORWARD, ("dog", "en", 1, 10)).fetchall())
    monolith.close()

    sharded = sqlite3.connect(f"file:{pack.as_posix()}?mode=ro", uri=True)
    try:
        actual = compare_key(
            relation_rows(sharded, word="dog", language="en", relation_code=1, limit=10)
        )
    finally:
        sharded.close()

    # Includes the en -> fr edge, whose target is a stub in this pack.
    assert expected == [("hound", "en", 1, 1, 1), ("chien", "fr", 1, 1, 1)]
    assert actual == expected


def test_reverse_relations_resolve_through_the_catalogue(
    corpus: Path, tmp_path: Path
) -> None:
    """cy -> en: English is only the target, and the source is a foreign stub."""

    from lexicon_mcp.runtime.pack_queries import relation_rows

    pack = build_en(corpus, tmp_path)
    connection = sqlite3.connect(f"file:{pack.as_posix()}?mode=ro", uri=True)
    try:
        rows = relation_rows(
            connection, word="dog", language="en", relation_code=1, limit=10, reverse=True
        )
    finally:
        connection.close()

    assert [(row["target_term"], row["target_language"]) for row in rows] == [("ci", "cy")]
    assert rows[0]["target_payload_local"] == 0


def test_translations_survive_without_the_target_language(
    corpus: Path, tmp_path: Path
) -> None:
    from lexicon_mcp.runtime.pack_queries import as_result, translation_rows

    pack = build_en(corpus, tmp_path)
    connection = sqlite3.connect(f"file:{pack.as_posix()}?mode=ro", uri=True)
    try:
        rows = translation_rows(connection, sense_id="s1")
    finally:
        connection.close()

    result = as_result(rows[0])
    assert result["term"] == "chien"
    assert result["target_language"] == "fr"
    # Usable as an answer, and honest that the French entry is not installed.
    assert result["target_language_installed"] is False
    assert result["target_details_available"] is False


def build_semantic_source(root: Path) -> Path:
    """A miniature semantic artifact set: mapping, vectors, per-language index."""

    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    (root / "vectors").mkdir(exist_ok=True)
    (root / "indexes" / "languages").mkdir(parents=True, exist_ok=True)

    dimensions = 4
    # Four vectors; rows 1 and 3 belong to French, so English gathers a
    # non-contiguous slice and its offsets must be rewritten.
    matrix = np.arange(4 * dimensions, dtype="<f2").reshape(4, dimensions)
    (root / "vectors" / "global.f16").write_bytes(matrix.tobytes())
    (root / "indexes" / "languages" / "en.usearch").write_bytes(b"usearch-en-index")

    connection = sqlite3.connect(root / "mapping.sqlite3")
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE provenance (provenance_id INTEGER PRIMARY KEY, source TEXT NOT NULL,
            source_license TEXT NOT NULL, source_url TEXT NOT NULL);
        CREATE TABLE lexical_terms (term_id INTEGER PRIMARY KEY, term TEXT NOT NULL,
            normalized_term TEXT NOT NULL, language TEXT NOT NULL);
        CREATE TABLE semantic_terms (semantic_id INTEGER PRIMARY KEY, concept TEXT NOT NULL,
            term_id INTEGER NOT NULL, vector_offset INTEGER NOT NULL UNIQUE);
        CREATE TABLE semantic_languages (language TEXT PRIMARY KEY, index_file TEXT NOT NULL,
            term_count INTEGER NOT NULL);
        """
    )
    connection.execute(
        "INSERT INTO provenance VALUES (1,'Numberbatch','CC-BY-SA-4.0','https://conceptnet.io/')"
    )
    for term_id, term, language in ((1, "dog", "en"), (2, "chien", "fr"), (3, "cat", "en")):
        connection.execute(
            "INSERT INTO lexical_terms VALUES (?,?,?,?)", (term_id, term, term, language)
        )
    # offsets deliberately interleaved across languages
    connection.execute("INSERT INTO semantic_terms VALUES (10,'/c/en/dog',1,0)")
    connection.execute("INSERT INTO semantic_terms VALUES (11,'/c/fr/chien',2,1)")
    connection.execute("INSERT INTO semantic_terms VALUES (12,'/c/en/cat',3,2)")
    connection.execute(
        "INSERT INTO semantic_languages VALUES ('en','indexes/languages/en.usearch',2)"
    )
    for key, value in (
        ("dimensions", str(dimensions)),
        ("vector_file", "vectors/global.f16"),
        ("vector_dtype", "float16"),
        ("index_dtype", "i8"),
        ("index_metric", "cos"),
        ("connectivity", "16"),
        ("expansion_add", "256"),
        ("expansion_search", "512"),
        ("source", "ConceptNet Numberbatch 19.08"),
        ("source_license", "CC-BY-SA-4.0"),
        ("source_url", "https://github.com/commonsense/conceptnet-numberbatch"),
    ):
        connection.execute("INSERT INTO metadata VALUES (?,?)", (key, value))
    connection.commit()
    connection.close()
    return root


def test_semantic_pack_gathers_its_own_vectors_and_rewrites_offsets(
    tmp_path: Path,
) -> None:
    import numpy as np

    from lexicon_mcp.pipeline.transform import build_semantic_pack

    source = build_semantic_source(tmp_path / "semantic")

    result = build_semantic_pack(
        source, tmp_path / "pack", "en", dataset_version=VERSION
    )

    assert result.terms == 2
    assert result.dimensions == 4
    # semantic_id is preserved, because the USearch index keys on it.
    rows = read(result.mapping, "SELECT semantic_id, concept, vector_offset FROM semantic_terms")
    assert rows == [(10, "/c/en/dog", 0), (12, "/c/en/cat", 1)]

    # The gathered file holds only English rows, in their new offset order.
    gathered = np.frombuffer(result.vectors.read_bytes(), dtype="<f2").reshape(2, 4)
    original = np.arange(16, dtype="<f2").reshape(4, 4)
    assert np.array_equal(gathered[0], original[0])
    assert np.array_equal(gathered[1], original[2])


def test_semantic_pack_ships_the_existing_index_verbatim(tmp_path: Path) -> None:
    """The index keys on semantic_id, so it never needs rebuilding."""

    from lexicon_mcp.pipeline.transform import build_semantic_pack

    source = build_semantic_source(tmp_path / "semantic")

    result = build_semantic_pack(
        source, tmp_path / "pack", "en", dataset_version=VERSION
    )

    assert result.index.read_bytes() == b"usearch-en-index"
    assert read(
        result.mapping, "SELECT language, index_file, term_count FROM semantic_languages"
    ) == [("en", "en.usearch", 2)]


def test_semantic_pack_refuses_a_language_without_vectors(tmp_path: Path) -> None:
    from lexicon_mcp.pipeline.transform import TransformError, build_semantic_pack

    source = build_semantic_source(tmp_path / "semantic")

    with pytest.raises(TransformError, match="no semantic vectors"):
        build_semantic_pack(source, tmp_path / "pack", "cy", dataset_version=VERSION)
