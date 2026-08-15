from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lexicon_mcp.pipeline.schema import (
    create_lexical_query_indexes,
    create_lexical_schema,
    create_wordplay_indexes,
)
from lexicon_mcp.pipeline.wordplay import (
    is_palindrome,
    is_wordplay_eligible,
    letter_signature,
    normalized_letters,
    split_arpabet_onset,
)
from lexicon_mcp.runtime.actual_wordplay import SQLiteActualWordplaySearch
from lexicon_mcp.runtime.service import LexiconService

_CMU_PROVENANCE = {
    "source": "CMU Pronouncing Dictionary",
    "license": "CMUdict license",
    "url": "https://github.com/cmusphinx/cmudict",
}


def test_pure_letter_derivations() -> None:
    assert normalized_letters("Listen") == "listen"
    assert normalized_letters("Co-Operate") == "cooperate"
    # NFKC+casefold does not fold every diacritic to ASCII; non-ASCII
    # code points are dropped rather than transliterated.
    assert normalized_letters("Café") == "caf"
    assert normalized_letters("100%") == ""
    assert is_wordplay_eligible("listen")
    assert not is_wordplay_eligible("hot dog")
    assert not is_wordplay_eligible("co-operate")
    assert not is_wordplay_eligible("100%")
    assert not is_wordplay_eligible("naïve")
    assert letter_signature("silent") == letter_signature("listen") == "eilnst"
    assert letter_signature("enlist") == "eilnst"
    assert is_palindrome("level")
    assert is_palindrome("noon")
    assert not is_palindrome("ab")
    assert not is_palindrome("a")
    assert not is_palindrome("")


def test_split_arpabet_onset() -> None:
    assert split_arpabet_onset("K AE1 T") == ("K", "AE1 T")
    assert split_arpabet_onset("S T R IY1 T") == ("S T R", "IY1 T")
    assert split_arpabet_onset("AH1 N D ER0") == ("", "AH1 N D ER0")
    assert split_arpabet_onset("K AE1 T S") == ("K", "AE1 T S")


def _populate(connection: sqlite3.Connection) -> None:
    connection.executemany(
        "INSERT INTO provenance VALUES (?, ?, ?, ?)",
        [
            (1, "Open English WordNet", "CC-BY-4.0", "https://en-word.net/"),
            (
                2,
                "Wiktionary via Wiktextract",
                "CC-BY-SA-4.0 and GFDL-1.3-or-later",
                "https://kaikki.org/",
            ),
        ],
    )
    terms = [
        # term_id, term, normalized_term, language
        (1, "listen", "listen", "en"),
        (2, "silent", "silent", "en"),
        (3, "enlist", "enlist", "en"),
        (4, "tinsel", "tinsel", "en"),
        (5, "inlets", "inlets", "en"),
        (6, "level", "level", "en"),
        (7, "noon", "noon", "en"),
        (8, "deed", "deed", "en"),
        (9, "civic", "civic", "en"),
        (10, "light", "light", "en"),
        (11, "rain", "rain", "en"),
        (12, "right", "right", "en"),
        (13, "lane", "lane", "en"),
        (14, "see", "see", "en"),
        (15, "sea", "sea", "en"),
        (16, "cee", "cee", "en"),
        (17, "hot dog", "hot dog", "en"),
    ]
    connection.executemany(
        "INSERT INTO lexical_terms VALUES (?, ?, ?, ?)", terms
    )
    entries = [
        ("oewn:entry:listen", 1, "verb", None, 1),
        ("oewn:entry:silent", 2, "adjective", None, 1),
        ("oewn:entry:see", 14, "verb", None, 1),
        ("wikt:entry:sea", 15, "noun", None, 2),
        ("oewn:entry:light", 10, "noun", None, 1),
        ("oewn:entry:rain", 11, "noun", None, 1),
        ("oewn:entry:right", 12, "adjective", None, 1),
        ("oewn:entry:lane", 13, "noun", None, 1),
        ("wikt:entry:level", 6, "noun", None, 2),
        ("wikt:entry:noon", 7, "noun", None, 2),
        ("wikt:entry:deed", 8, "noun", None, 2),
        ("wikt:entry:civic", 9, "adjective", None, 2),
    ]
    connection.executemany(
        "INSERT INTO lexical_entries VALUES (?, ?, ?, ?, ?)", entries
    )
    senses = [
        ("oewn:listen-v-1", "oewn:entry:listen", "to pay attention"),
        ("oewn:silent-a-1", "oewn:entry:silent", "making no sound"),
        ("oewn:see-v-1", "oewn:entry:see", "to perceive with the eyes"),
        ("wikt:sea-n-1", "wikt:entry:sea", "a large body of salt water"),
        ("oewn:light-n-1", "oewn:entry:light", "electromagnetic radiation"),
        ("oewn:rain-n-1", "oewn:entry:rain", "condensed water falling"),
        ("oewn:right-a-1", "oewn:entry:right", "correct or true"),
        ("oewn:lane-n-1", "oewn:entry:lane", "a narrow road"),
        ("wikt:level-n-1", "wikt:entry:level", "a flat surface"),
        ("wikt:noon-n-1", "wikt:entry:noon", "midday"),
        ("wikt:deed-n-1", "wikt:entry:deed", "an act"),
        ("wikt:civic-a-1", "wikt:entry:civic", "relating to a city"),
    ]
    connection.executemany("INSERT INTO senses VALUES (?, ?, ?)", senses)
    pronunciations = [
        (1, "L IH1 S AH0 N", "IH1 S AH0 N"),
        (2, "S AY1 L AH0 N T", "AH0 N T"),
        (3, "IH1 N L IH1 S T", "IH1 S T"),
        (4, "T IH1 N S AH0 L", "AH0 L"),
        (5, "IH1 N L AH0 T S", "AH0 T S"),
        (6, "L EH1 V AH0 L", "EH1 V AH0 L"),
        (7, "N UW1 N", "UW1 N"),
        (8, "D IY1 D", "IY1 D"),
        (9, "S IH1 V IH0 K", "IH1 V IH0 K"),
        (10, "L AY1 T", "AY1 T"),
        (11, "R EY1 N", "EY1 N"),
        (12, "R AY1 T", "AY1 T"),
        (13, "L EY1 N", "EY1 N"),
        (14, "S IY1", "IY1"),
        (15, "S IY1", "IY1"),
        (16, "S IY1", "IY1"),
        (17, "HH AA1 T D AO1 G", "AO1 G"),
    ]
    connection.executemany(
        "INSERT INTO pronunciations_words VALUES (?, ?, ?)", pronunciations
    )


@pytest.fixture()
def database(tmp_path: Path) -> Path:
    path = tmp_path / "lexicon.sqlite3"
    with sqlite3.connect(path) as connection:
        create_lexical_schema(connection, "data-test-v1")
        _populate(connection)
        create_lexical_query_indexes(connection)
        create_wordplay_indexes(connection)
    return path


@pytest.fixture()
def service(database: Path):
    instance = LexiconService(database, "data-test-v1")
    yield instance
    instance.close()


def _assert_provenance_array(entry: object) -> None:
    assert isinstance(entry, list) and entry
    for item in entry:
        assert {"source", "license", "url"} <= set(item)


def test_anagram_returns_signature_group_and_excludes_self(
    service: LexiconService,
) -> None:
    response = service.wordplay("Listen", "anagram")
    assert response["type"] == "wordplay"
    assert response["dataset_version"] == "data-test-v1"
    assert response["query"] == {
        "text": "Listen",
        "normalized_text": "listen",
        "kind": "anagram",
        "context": None,
        "limit": 20,
    }
    terms = [item["term"] for item in response["results"]]
    assert terms == ["enlist", "inlets", "silent", "tinsel"]
    for item in response["results"]:
        assert item["signature"] == "eilnst"
        assert item["language"] == "en"
        assert item["explanation"] == "same normalized letters"
        assert item["normalized_term"] == item["term"]
        _assert_provenance_array(item["provenance"])
    limited = service.wordplay("listen", "anagram", limit=2)
    assert [item["term"] for item in limited["results"]] == ["enlist", "inlets"]


def test_palindrome_returns_corpus_alternatives_and_input_flag(
    service: LexiconService,
) -> None:
    response = service.wordplay("level", "palindrome")
    assert response["query"]["input_is_palindrome"] is True
    terms = [item["term"] for item in response["results"]]
    assert "level" not in terms
    assert set(terms) == {"civic", "deed", "noon"}
    for item in response["results"]:
        assert item["palindrome_key"] == item["term"]
        assert item["explanation"] == "normalized letters read identically in reverse"
        _assert_provenance_array(item["provenance"])
    flagged = service.wordplay("listen", "palindrome")
    assert flagged["query"]["input_is_palindrome"] is False
    # One code point and punctuation-only inputs never yield candidates.
    assert service.wordplay("a", "palindrome")["results"] == []
    assert service.wordplay("!!!", "palindrome")["results"] == []


def test_spoonerism_swaps_onsets_with_lexicality_labelling(
    service: LexiconService,
) -> None:
    response = service.wordplay("light rain", "spoonerism")
    assert len(response["results"]) == 1
    item = response["results"][0]
    assert item["left"] == {"term": "light", "phonemes": "L AY1 T"}
    assert item["right"] == {"term": "rain", "phonemes": "R EY1 N"}
    assert item["onset_left"] == "L"
    assert item["onset_right"] == "R"
    assert item["swapped_left"] == "R AY1 T"
    assert item["swapped_right"] == "L EY1 N"
    assert item["swapped_left_terms"] == ["right"]
    assert item["swapped_right_terms"] == ["lane"]
    assert item["lexicality_scope"] == "lexical_term"
    assert item["explanation"] == "initial consonant clusters exchanged"
    assert item["provenance"] == [_CMU_PROVENANCE]

    generated = service.wordplay("civic noon", "spoonerism")["results"]
    assert len(generated) == 1
    assert generated[0]["lexicality_scope"] == "generated_candidate"
    assert generated[0]["swapped_left_terms"] == []
    assert generated[0]["swapped_right_terms"] == []

    with pytest.raises(ValueError, match="exactly two"):
        service.wordplay("light", "spoonerism")
    with pytest.raises(ValueError, match="exactly two"):
        service.wordplay("one two three", "spoonerism")
    with pytest.raises(ValueError, match="context"):
        service.wordplay("light rain", "spoonerism", context="weather talk")


def test_pun_requires_distinct_native_senses_and_labels_context(
    service: LexiconService,
) -> None:
    contextualized = service.wordplay("sea", "pun", context="the sea was calm")
    assert contextualized["query"]["context"] == "the sea was calm"
    terms = [item["term"] for item in contextualized["results"]]
    # cee is an exact homophone but has no source-native senses: suppressed.
    assert terms == ["see"]
    item = contextualized["results"][0]
    assert item["phonemes"] == "S IY1"
    assert item["sound_relation"] == "homophone"
    assert item["context_scope"] == "contextualized"
    assert item["result_class"] == "candidate"
    assert item["query_sense_ids"] == ["wikt:sea-n-1"]
    assert item["candidate_sense_ids"] == ["oewn:see-v-1"]
    assert set(item["query_sense_ids"]).isdisjoint(item["candidate_sense_ids"])
    sources = {entry["source"] for entry in item["provenance"]}
    assert "CMU Pronouncing Dictionary" in sources
    assert "Open English WordNet" in sources

    uncontextualized = service.wordplay("sea", "pun")
    assert uncontextualized["query"]["context"] is None
    assert uncontextualized["results"][0]["context_scope"] == "uncontextualized"
    # No joke is ever claimed for a pun candidate.
    assert "joke" not in str(uncontextualized).casefold()


def test_wordplay_input_limit_and_kind_validation(service: LexiconService) -> None:
    for limit in (-1, 0, 101):
        with pytest.raises(ValueError, match="limit"):
            service.wordplay("listen", "anagram", limit=limit)
    with pytest.raises(ValueError, match="kind must be one of"):
        service.wordplay("listen", "spooner")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="context is only accepted"):
        service.wordplay("listen", "anagram", context="letters everywhere")
    with pytest.raises(ValueError, match="between 1 and 512"):
        service.wordplay("sea", "pun", context="x" * 513)
    with pytest.raises(ValueError, match="between 1 and 512"):
        service.wordplay("sea", "pun", context="   ")


def test_actual_wordplay_fails_closed_on_missing_or_mismatched_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lexicon.sqlite3"
    with sqlite3.connect(path) as connection:
        create_lexical_schema(connection, "data-test-v1")
        _populate(connection)
        create_lexical_query_indexes(connection)
    with pytest.raises(RuntimeError, match="missing actual-wordplay tables"):
        SQLiteActualWordplaySearch(path)

    mismatch = tmp_path / "mismatch.sqlite3"
    with sqlite3.connect(mismatch) as connection:
        create_lexical_schema(connection, "data-test-v1")
        _populate(connection)
        create_lexical_query_indexes(connection)
        create_wordplay_indexes(connection)
        connection.execute(
            "UPDATE metadata SET value = '2' WHERE key = 'wordplay_index_version'"
        )
        connection.commit()
    with pytest.raises(RuntimeError, match="wordplay index version"):
        SQLiteActualWordplaySearch(mismatch)


def test_anagram_query_plan_uses_partial_index(database: Path) -> None:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        plan = connection.execute(
            """
            SELECT w.normalized_letters FROM wordplay_terms AS w
            WHERE w.letter_signature = 'eilnst' AND w.wordplay_eligible = 1
            """
        ).fetchall()
        assert plan
        detail = " ".join(
            str(row)
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT term_id FROM wordplay_terms"
                " WHERE letter_signature = 'eilnst' AND wordplay_eligible = 1"
            ).fetchall()
        )
    finally:
        connection.close()
    assert "wordplay_terms_anagram" in detail


def test_readonly_lookup_creates_no_database_files(service: LexiconService) -> None:
    directory = service.database_path.parent
    before = {path.name for path in directory.iterdir()}
    service.wordplay("listen", "anagram")
    service.wordplay("light rain", "spoonerism")
    service.wordplay("sea", "pun")
    after = {path.name for path in directory.iterdir()}
    assert before == after
