from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from lexicon_mcp.pipeline.schema import (
    create_lexical_query_indexes,
    create_lexical_schema,
    create_wordplay_indexes,
)
from lexicon_mcp.runtime.locator import DatasetLocator
from lexicon_mcp.runtime.normalization import normalize_key, normalize_language
from lexicon_mcp.runtime.service import LexiconService


class FakeSemanticSearch:
    available = True

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def search(
        self,
        word: str,
        source_language: str,
        target_language: str | None,
        limit: int,
        min_similarity: float | None,
    ) -> list[dict[str, Any]]:
        self.calls.append((word, source_language, target_language, limit, min_similarity))
        return [
            {
                "semantic_id": 8,
                "concept": "/c/de/Katze",
                "term": "Katze",
                "language": "de",
                "similarity": 0.8125,
                "sense_scope": "unsensed",
                "provenance": {
                    "source": "ConceptNet Numberbatch",
                    "license": "CC-BY-SA-4.0",
                    "url": "https://conceptnet.io/",
                },
            }
        ]

    def close(self) -> None:
        return None


def _insert_legacy_fixture_data(connection: sqlite3.Connection) -> None:
    source = ("Wiktextract", "CC-BY-SA-4.0", "https://kaikki.org/")
    senses = [
        (
            "oewn:bank-finance-n",
            "bank",
            "bank",
            "en",
            "noun",
            "A financial institution.",
            "From Italian banca.",
            "Open English WordNet",
            "CC-BY-4.0",
            "https://en-word.net/",
        ),
        (
            "oewn:bank-river-n",
            "bank",
            "bank",
            "en",
            "noun",
            "Sloping land beside a river.",
            None,
            "Open English WordNet",
            "CC-BY-4.0",
            "https://en-word.net/",
        ),
        (
            "wikt:lead-metal",
            "lead",
            "lead",
            "en",
            "noun",
            "A heavy metallic element.",
            None,
            *source,
        ),
        (
            "wikt:lead-verb",
            "lead",
            "lead",
            "en",
            "verb",
            "To guide.",
            None,
            *source,
        ),
        (
            "wikt:unsensed:bright-adj",
            "bright",
            "bright",
            "en",
            "adjective",
            None,
            None,
            *source,
        ),
        (
            "wikt:cafe-fr",
            "café",
            "café",
            "fr",
            "noun",
            "Établissement où l'on sert des boissons.",
            None,
            *source,
        ),
    ]
    connection.executemany("INSERT INTO senses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", senses)
    connection.executemany(
        "INSERT INTO examples VALUES (?, ?, ?)",
        [
            ("oewn:bank-finance-n", "She deposited money at the bank.", 0),
            ("oewn:bank-river-n", "They sat on the river bank.", 0),
        ],
    )
    connection.executemany(
        "INSERT INTO pronunciations VALUES (?, ?, ?, ?)",
        [
            ("wikt:lead-metal", "/lɛd/", "", 0),
            ("wikt:lead-verb", "/li\u02d0d/", "", 0),
        ],
    )
    connection.executemany(
        "INSERT INTO translations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "oewn:bank-finance-n",
                "de",
                "Bank",
                "bank",
                "noun",
                *source,
                0,
            ),
            (
                "oewn:bank-river-n",
                "de",
                "Ufer",
                "ufer",
                "noun",
                *source,
                0,
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO synonyms VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "oewn:bank-finance-n",
                "financial institution",
                "financial institution",
                "en",
                "noun",
                "Open English WordNet",
                "CC-BY-4.0",
                "https://en-word.net/",
                0,
            ),
            (
                "oewn:bank-river-n",
                "riverbank",
                "riverbank",
                "en",
                "noun",
                "Open English WordNet",
                "CC-BY-4.0",
                "https://en-word.net/",
                0,
            ),
            (
                "wikt:unsensed:bright-adj",
                "luminous",
                "luminous",
                "en",
                "adjective",
                *source,
                0,
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "dog",
                "dog",
                "en",
                None,
                "hypernym",
                "animal",
                "animal",
                "en",
                None,
                "outgoing",
                "ConceptNet",
                "CC-BY-SA-4.0",
                "https://conceptnet.io/",
            ),
            (
                "bright",
                "bright",
                "en",
                None,
                "synonym",
                "radiant",
                "radiant",
                "en",
                None,
                "outgoing",
                "ConceptNet",
                "CC-BY-SA-4.0",
                "https://conceptnet.io/",
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO pronunciations_words VALUES (?, ?, ?)",
        [
            ("cat", "cat", "K AE1 T"),
            ("bat", "bat", "B AE1 T"),
            ("kit", "kit", "K IH1 T"),
            ("night", "night", "N AY1 T"),
            ("knight", "knight", "N AY1 T"),
            ("hello", "hello", "HH AH0 L OW1"),
            ("hallo", "hallo", "HH AE1 L OW0"),
            ("help", "help", "HH EH1 L P"),
        ],
    )
    connection.commit()


def _insert_fixture_data(connection: sqlite3.Connection) -> None:
    connection.executemany(
        "INSERT INTO provenance VALUES (?, ?, ?, ?)",
        [
            (1, "Open English WordNet", "CC-BY-4.0", "https://en-word.net/"),
            (2, "Wiktextract", "CC-BY-SA-4.0", "https://kaikki.org/"),
            (3, "ConceptNet", "CC-BY-SA-4.0", "https://conceptnet.io/"),
        ],
    )
    terms = [
        (1, "bank", "bank", "en"),
        (2, "lead", "lead", "en"),
        (3, "bright", "bright", "en"),
        (4, "café", "café", "fr"),
        (5, "Bank", "bank", "de"),
        (6, "Ufer", "ufer", "de"),
        (7, "financial institution", "financial institution", "en"),
        (8, "riverbank", "riverbank", "en"),
        (9, "luminous", "luminous", "en"),
        (10, "dog", "dog", "en"),
        (11, "animal", "animal", "en"),
        (12, "radiant", "radiant", "en"),
        (13, "cat", "cat", "en"),
        (14, "bat", "bat", "en"),
        (15, "kit", "kit", "en"),
        (16, "night", "night", "en"),
        (17, "knight", "knight", "en"),
        (18, "hello", "hello", "en"),
        (19, "hallo", "hallo", "en"),
        (20, "help", "help", "en"),
        (21, "wheel", "wheel", "en"),
        (22, "car", "car", "en"),
        (23, "shared candidate", "shared candidate", "en"),
    ]
    connection.executemany("INSERT INTO lexical_terms VALUES (?, ?, ?, ?)", terms)
    connection.executemany(
        "INSERT INTO lexical_entries VALUES (?, ?, ?, ?, ?)",
        [
            ("entry:bank-finance", 1, "noun", "From Italian banca.", 1),
            ("entry:bank-river", 1, "noun", None, 1),
            ("entry:lead-metal", 2, "noun", None, 2),
            ("entry:lead-verb", 2, "verb", None, 2),
            ("entry:bright", 3, "adjective", None, 2),
            ("entry:cafe", 4, "noun", None, 2),
        ],
    )
    connection.executemany(
        "INSERT INTO senses VALUES (?, ?, ?)",
        [
            ("oewn:bank-finance-n", "entry:bank-finance", "A financial institution."),
            ("oewn:bank-river-n", "entry:bank-river", "Sloping land beside a river."),
            ("wikt:lead-metal", "entry:lead-metal", "A heavy metallic element."),
            ("wikt:lead-verb", "entry:lead-verb", "To guide."),
            ("wikt:unsensed:bright-adj", "entry:bright", None),
            (
                "wikt:cafe-fr",
                "entry:cafe",
                "Établissement où l'on sert des boissons.",
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO examples VALUES (?, ?, ?)",
        [
            ("oewn:bank-finance-n", "She deposited money at the bank.", 0),
            ("oewn:bank-river-n", "They sat on the river bank.", 0),
        ],
    )
    connection.executemany(
        "INSERT INTO pronunciations VALUES (?, ?, ?, ?)",
        [
            ("entry:lead-metal", "/lɛd/", "", 0),
            ("entry:lead-verb", "/li\u02d0d/", "", 0),
        ],
    )
    connection.executemany(
        "INSERT INTO translations VALUES (?, ?, ?, ?, ?)",
        [
            ("oewn:bank-finance-n", 5, "noun", 2, 0),
            ("oewn:bank-river-n", 6, "noun", 2, 0),
        ],
    )
    connection.executemany(
        "INSERT INTO synonyms VALUES (?, ?, ?, ?, ?)",
        [
            ("oewn:bank-finance-n", 7, "noun", 1, 0),
            ("oewn:bank-finance-n", 23, "noun", 1, 1),
            ("oewn:bank-river-n", 8, "noun", 1, 0),
            ("oewn:bank-river-n", 23, "noun", 1, 1),
            ("wikt:unsensed:bright-adj", 9, "adjective", 2, 0),
        ],
    )
    connection.executemany(
        "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (10, None, 3, 11, None, 1, 3),
            (3, None, 1, 12, None, 3, 3),
            (21, None, 6, 22, None, 1, 3),
        ],
    )
    connection.executemany(
        "INSERT INTO pronunciations_words VALUES (?, ?, ?)",
        [
            (13, "K AE1 T", "AE1 T"),
            (14, "B AE1 T", "AE1 T"),
            (15, "K IH1 T", "IH1 T"),
            (16, "N AY1 T", "AY1 T"),
            (17, "N AY1 T", "AY1 T"),
            (18, "HH AH0 L OW1", "OW1"),
            (19, "HH AE1 L OW0", "AE1 L OW0"),
            (20, "HH EH1 L P", "EH1 L P"),
        ],
    )
    connection.commit()


@pytest.fixture()
def lexical_database(tmp_path: Path) -> Path:
    path = tmp_path / "lexicon.sqlite3"
    with sqlite3.connect(path) as connection:
        create_lexical_schema(connection, "data-test-v1")
        _insert_fixture_data(connection)
        create_lexical_query_indexes(connection)
        create_wordplay_indexes(connection)
    return path


@pytest.fixture()
def service(lexical_database: Path) -> LexiconService:
    instance = LexiconService(lexical_database, "data-test-v1")
    yield instance
    instance.close()


def test_lookup_preserves_distinct_senses_translations_and_pronunciations(
    service: LexiconService,
) -> None:
    bank = service.dictionary_lookup("bank")
    assert bank["type"] == "dictionary_lookup"
    assert bank["dataset_version"] == "data-test-v1"
    assert bank["count"] == 2
    by_id = {item["sense_id"]: item for item in bank["results"]}
    assert by_id["oewn:bank-finance-n"]["translations"][0]["term"] == "Bank"
    assert by_id["oewn:bank-river-n"]["translations"][0]["term"] == "Ufer"
    assert by_id["oewn:bank-finance-n"]["provenance"]["license"] == "CC-BY-4.0"
    assert all(not item["truncated_fields"] for item in bank["results"])

    lead = service.dictionary_lookup("lead")
    assert {item["pronunciations"][0]["ipa"] for item in lead["results"]} == {
        "/lɛd/",
        "/li\u02d0d/",
    }


def test_lookup_detail_budgets_are_total_fair_and_truthfully_truncated(
    lexical_database: Path,
) -> None:
    with sqlite3.connect(lexical_database) as connection:
        connection.executemany(
            "INSERT INTO lexical_terms VALUES (?, ?, ?, ?)",
            [
                (24, "Finanzbank", "finanzbank", "de"),
                (25, "Kreditinstitut", "kreditinstitut", "de"),
                (26, "Flussufer", "flussufer", "de"),
                (27, "Böschung", "böschung", "de"),
            ],
        )
        connection.executemany(
            "INSERT INTO examples VALUES (?, ?, ?)",
            [
                ("oewn:bank-finance-n", "Finance example 2.", 1),
                ("oewn:bank-finance-n", "Finance example 3.", 2),
                ("oewn:bank-river-n", "River example 2.", 1),
                ("oewn:bank-river-n", "River example 3.", 2),
            ],
        )
        connection.executemany(
            "INSERT INTO pronunciations VALUES (?, ?, ?, ?)",
            [
                ("entry:bank-finance", "/bank-finance-1/", "", 0),
                ("entry:bank-finance", "/bank-finance-2/", "", 1),
                ("entry:bank-finance", "/bank-finance-3/", "", 2),
                ("entry:bank-river", "/bank-river-1/", "", 0),
                ("entry:bank-river", "/bank-river-2/", "", 1),
                ("entry:bank-river", "/bank-river-3/", "", 2),
            ],
        )
        connection.executemany(
            "INSERT INTO translations VALUES (?, ?, ?, ?, ?)",
            [
                ("oewn:bank-finance-n", 24, "noun", 2, 1),
                ("oewn:bank-finance-n", 25, "noun", 2, 2),
                ("oewn:bank-river-n", 26, "noun", 2, 1),
                ("oewn:bank-river-n", 27, "noun", 2, 2),
            ],
        )

    with LexiconService(lexical_database, "data-test-v1") as instance:
        bounded = instance.dictionary_lookup(
            "bank",
            limit=2,
            examples_limit=3,
            pronunciations_limit=3,
            translations_limit=3,
        )
        assert bounded["query"] == {
            "word": "bank",
            "normalized_word": "bank",
            "language": "en",
            "part_of_speech": None,
            "limit": 2,
            "examples_limit": 3,
            "pronunciations_limit": 3,
            "translations_limit": 3,
        }
        assert [len(item["examples"]) for item in bounded["results"]] == [2, 1]
        assert [len(item["pronunciations"]) for item in bounded["results"]] == [2, 1]
        assert [len(item["translations"]) for item in bounded["results"]] == [2, 1]
        assert all(
            set(item["truncated_fields"]) == {"examples", "pronunciations", "translations"}
            for item in bounded["results"]
        )

        disabled = instance.dictionary_lookup(
            "bank",
            examples_limit=0,
            pronunciations_limit=0,
            translations_limit=0,
        )
        assert all(
            not item[field]
            for item in disabled["results"]
            for field in ("examples", "pronunciations", "translations")
        )
        assert all(
            set(item["truncated_fields"]) == {"examples", "pronunciations", "translations"}
            for item in disabled["results"]
        )

        complete = instance.dictionary_lookup(
            "bank",
            examples_limit=100,
            pronunciations_limit=100,
            translations_limit=100,
        )
        assert all(not item["truncated_fields"] for item in complete["results"])
        assert sum(len(item["examples"]) for item in complete["results"]) == 6
        assert sum(len(item["pronunciations"]) for item in complete["results"]) == 6
        assert sum(len(item["translations"]) for item in complete["results"]) == 6


def test_lookup_retains_high_coverage_translation_senses_within_result_limit(
    lexical_database: Path,
) -> None:
    with sqlite3.connect(lexical_database) as connection:
        connection.execute(
            "DELETE FROM translations WHERE sense_id IN (?, ?)",
            ("oewn:bank-finance-n", "oewn:bank-river-n"),
        )
        distractor_entries = [
            (f"entry:bank-distractor-{index:02d}", 1, "noun", None, 1) for index in range(8)
        ]
        connection.executemany(
            "INSERT INTO lexical_entries VALUES (?, ?, ?, ?, ?)",
            [
                *distractor_entries,
                ("entry:bank-wikt-finance", 1, "noun", None, 2),
                ("entry:bank-wikt-river", 1, "noun", None, 2),
            ],
        )
        connection.executemany(
            "INSERT INTO senses VALUES (?, ?, ?)",
            [
                *[
                    (
                        f"oewn:bank-distractor-{index:02d}",
                        f"entry:bank-distractor-{index:02d}",
                        f"Distractor sense {index}.",
                    )
                    for index in range(8)
                ],
                (
                    "wikt:labeled:bank-finance",
                    "entry:bank-wikt-finance",
                    "institution",
                ),
                (
                    "wikt:labeled:bank-river",
                    "entry:bank-wikt-river",
                    "edge of river or lake",
                ),
            ],
        )
        connection.executemany(
            "INSERT INTO translations VALUES (?, ?, ?, ?, ?)",
            [
                ("wikt:labeled:bank-finance", 5, "noun", 2, 0),
                ("wikt:labeled:bank-river", 6, "noun", 2, 0),
            ],
        )

    with LexiconService(lexical_database, "data-test-v1") as instance:
        result = instance.dictionary_lookup("bank", limit=10, translations_limit=20)
        by_id = {item["sense_id"]: item for item in result["results"]}
        assert result["count"] == 10
        assert by_id["wikt:labeled:bank-finance"]["translations"][0]["term"] == "Bank"
        assert by_id["wikt:labeled:bank-river"]["translations"][0]["term"] == "Ufer"

        metadata_only = instance.dictionary_lookup("bank", limit=10, translations_limit=0)
        assert all(
            not item["sense_id"].startswith("wikt:labeled:") for item in metadata_only["results"]
        )


def test_lookup_uses_nfkc_casefold_and_returns_typed_oov(service: LexiconService) -> None:
    result = service.dictionary_lookup("  ＣＡＦÉ  ", "fr")
    assert result["query"]["word"] == "ＣＡＦÉ"
    assert result["query"]["normalized_word"] == "café"
    assert result["results"][0]["word"] == "café"
    assert service.dictionary_lookup("cafe\u0301", "fr")["results"][0]["word"] == "café"
    assert normalize_key("  two\u2003  words  ") == "two words"
    assert normalize_language("zh-Hant") == "zh-hant"
    assert normalize_language("PT_br") == "pt-br"

    oov = service.dictionary_lookup("definitely-not-present", "en")
    assert oov == {
        "type": "dictionary_lookup",
        "dataset_version": "data-test-v1",
        "query": {
            "word": "definitely-not-present",
            "normalized_word": "definitely-not-present",
            "language": "en",
            "part_of_speech": None,
            "limit": 8,
            "examples_limit": 8,
            "pronunciations_limit": 8,
            "translations_limit": 20,
        },
        "count": 0,
        "results": [],
    }


def test_synonyms_are_sense_grouped_and_unsensed_is_explicit(service: LexiconService) -> None:
    finance = service.dictionary_synonyms(
        "bank", sense_id="oewn:bank-finance-n", part_of_speech="noun"
    )
    assert finance["count"] == 1
    assert [item["term"] for item in finance["results"][0]["synonyms"]] == [
        "financial institution",
        "shared candidate",
    ]
    assert finance["results"][0]["sense_scope"] == "sense"

    all_bank = service.dictionary_synonyms("bank")
    assert all_bank["candidate_count"] == 4
    shared_scopes = {
        group["sense_id"]
        for group in all_bank["results"]
        if any(item["term"] == "shared candidate" for item in group["synonyms"])
    }
    assert shared_scopes == {"oewn:bank-finance-n", "oewn:bank-river-n"}

    bright = service.dictionary_synonyms("bright")
    groups = {group["sense_id"]: group for group in bright["results"]}
    assert groups["wikt:unsensed:bright-adj"]["sense_scope"] == "unsensed"
    assert groups[None]["sense_scope"] == "unsensed"
    terms = {candidate["term"] for group in bright["results"] for candidate in group["synonyms"]}
    assert terms == {"luminous", "radiant"}

    native_only = service.dictionary_synonyms("bright", unsensed_limit=1)
    assert [item["term"] for group in native_only["results"] for item in group["synonyms"]] == [
        "luminous"
    ]
    assert service.dictionary_synonyms("bright", unsensed_limit=0)["results"] == []

    # No unsensed candidates exist for bank, so its unused allocation returns
    # to the scoped sense groups instead of reducing their candidate budget.
    bank = service.dictionary_synonyms("bank", unsensed_limit=19)
    assert sum(len(group["synonyms"]) for group in bank["results"]) == 4
    for small_limit in (1, 2, 3, 5):
        small_bank = service.dictionary_synonyms("bank", limit=small_limit, unsensed_limit=0)
        assert all(group["sense_scope"] == "sense" for group in small_bank["results"])
        assert sum(len(group["synonyms"]) for group in small_bank["results"]) == min(small_limit, 4)


def test_synonym_depth_and_unsensed_allocation_are_caller_controlled(
    lexical_database: Path,
) -> None:
    strict_terms = [
        (24 + index, f"strict synonym {index:02d}", f"strict synonym {index:02d}", "en")
        for index in range(15)
    ]
    fallback_terms = [
        (39, "canine", "canine", "en"),
        (40, "cur", "cur", "en"),
        (41, "doggy", "doggy", "en"),
        (42, "hound", "hound", "en"),
        (43, "pooch", "pooch", "en"),
        (44, "Hund", "hund", "de"),
    ]
    with sqlite3.connect(lexical_database) as connection:
        connection.executemany(
            "INSERT INTO lexical_terms VALUES (?, ?, ?, ?)",
            [*strict_terms, *fallback_terms],
        )
        connection.execute(
            "INSERT INTO lexical_entries VALUES (?, ?, ?, ?, ?)",
            ("entry:dog-fixture", 10, "noun", None, 1),
        )
        senses = [
            (
                f"fixture:dog:{index:02d}",
                "entry:dog-fixture",
                "animal" if index == 12 else f"fixture sense {index:02d}",
            )
            for index in range(13)
        ]
        connection.executemany("INSERT INTO senses VALUES (?, ?, ?)", senses)
        connection.executemany(
            "INSERT INTO synonyms VALUES (?, ?, ?, ?, ?)",
            [
                (sense_id, 24 + index, "noun", 1, 0)
                for index, (sense_id, _entry_id, _gloss) in enumerate(senses)
            ]
            + [
                (senses[0][0], 37, "noun", 1, 1),
                (senses[1][0], 38, "noun", 1, 1),
            ],
        )
        connection.executemany(
            "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(10, None, 1, term_id, None, 3, 3) for term_id in range(39, 45)],
        )

    with LexiconService(lexical_database, "data-test-v1") as instance:
        result = instance.dictionary_synonyms("dog")
        candidates = [item for group in result["results"] for item in group["synonyms"]]
        assert len(candidates) == 20
        assert result["query"]["max_senses"] == 20
        assert result["query"]["unsensed_limit"] == 5
        animal_groups = [group for group in result["results"] if group["gloss"] == "animal"]
        assert len(animal_groups) == 1
        unsensed = next(group for group in result["results"] if group["sense_id"] is None)
        assert {item["term"] for item in unsensed["synonyms"]} == {
            "canine",
            "cur",
            "doggy",
            "hound",
            "pooch",
        }
        assert {item["language"] for item in unsensed["synonyms"]} == {"en"}

        strict_only = instance.dictionary_synonyms("dog", unsensed_limit=0)
        assert strict_only["results"]
        assert all(group["sense_scope"] == "sense" for group in strict_only["results"])
        assert sum(len(group["synonyms"]) for group in strict_only["results"]) == 15

        shallow = instance.dictionary_synonyms("dog", max_senses=5)
        assert {group["sense_id"] for group in shallow["results"] if group["sense_id"]} <= {
            sense[0] for sense in senses[:5]
        }

        small = instance.dictionary_synonyms("dog", limit=6, unsensed_limit=5)
        assert sum(len(group["synonyms"]) for group in small["results"]) == 6
        assert (
            sum(
                len(group["synonyms"])
                for group in small["results"]
                if group["sense_scope"] == "unsensed"
            )
            == 5
        )


def test_translation_never_crosses_requested_sense(service: LexiconService) -> None:
    finance = service.dictionary_translate("bank", "en", "de", sense_id="oewn:bank-finance-n")
    assert finance["count"] == 1
    assert finance["candidate_count"] == 1
    assert finance["results"][0]["translations"][0]["term"] == "Bank"
    assert all(
        item["term"] != "Ufer" for group in finance["results"] for item in group["translations"]
    )

    all_senses = service.dictionary_translate("bank", "en", "de")
    assert all_senses["candidate_count"] == 2
    assert all_senses["query"]["limit"] == 20
    assert all_senses["query"]["max_senses"] == 100
    mapping = {
        group["sense_id"]: group["translations"][0]["term"] for group in all_senses["results"]
    }
    assert mapping == {"oewn:bank-finance-n": "Bank", "oewn:bank-river-n": "Ufer"}


def test_translation_max_senses_and_candidate_limit_are_independent_and_fair(
    lexical_database: Path,
) -> None:
    with sqlite3.connect(lexical_database) as connection:
        connection.execute(
            "INSERT INTO lexical_terms VALUES (?, ?, ?, ?)",
            (24, "Finanzhaus", "finanzhaus", "de"),
        )
        connection.execute(
            "INSERT INTO translations VALUES (?, ?, ?, ?, ?)",
            ("oewn:bank-finance-n", 24, "noun", 2, 1),
        )

    with LexiconService(lexical_database, "data-test-v1") as instance:
        result = instance.dictionary_translate("bank", "en", "de", limit=2, max_senses=2)
        assert result["query"]["limit"] == 2
        assert result["query"]["max_senses"] == 2
        assert result["count"] == 2
        assert result["candidate_count"] == 2
        assert {
            group["sense_id"]: [item["term"] for item in group["translations"]]
            for group in result["results"]
        } == {
            "oewn:bank-finance-n": ["Bank"],
            "oewn:bank-river-n": ["Ufer"],
        }

        shallow = instance.dictionary_translate("bank", "en", "de", limit=2, max_senses=1)
        assert shallow["count"] == 1
        assert shallow["candidate_count"] == 2
        assert [item["term"] for item in shallow["results"][0]["translations"]] == [
            "Bank",
            "Finanzhaus",
        ]


def test_relations_include_direction_language_and_unsensed_scope(service: LexiconService) -> None:
    result = service.dictionary_relations("dog", "hypernym")
    assert result["count"] == 1
    relation = result["results"][0]
    assert relation["target_term"] == "animal"
    assert relation["direction"] == "outbound"
    assert relation["source_language"] == relation["target_language"] == "en"
    assert relation["sense_scope"] == "unsensed"


def test_relations_orient_one_physical_assertion_both_ways(
    service: LexiconService,
) -> None:
    animal = service.dictionary_relations("animal", "hyponym")
    assert [item["target_term"] for item in animal["results"]] == ["dog"]
    assert animal["results"][0]["direction"] == "inbound"

    wheel = service.dictionary_relations("wheel", "holonym")
    assert [item["target_term"] for item in wheel["results"]] == ["car"]
    assert wheel["results"][0]["direction"] == "outbound"

    car = service.dictionary_relations("car", "meronym")
    assert [item["target_term"] for item in car["results"]] == ["wheel"]
    assert car["results"][0]["direction"] == "inbound"


def test_hierarchy_relations_add_truthful_depth_two_paths(
    lexical_database: Path,
) -> None:
    dog_sense = "oewn:oewn-dog__1.05.00.."
    domestic_sense = "oewn:oewn-domestic_animal__1.05.00.."
    animal_sense = "oewn:oewn-animal__1.03.00.."
    with sqlite3.connect(lexical_database) as connection:
        connection.execute(
            "DELETE FROM relations "
            "WHERE source_term_id = 10 AND relation_code = 3 "
            "AND target_term_id = 11"
        )
        connection.executemany(
            "INSERT INTO lexical_terms VALUES (?, ?, ?, ?)",
            [
                (24, "domestic animal", "domestic animal", "en"),
                (25, "partially scoped", "partially scoped", "en"),
                (26, "wrong sense", "wrong sense", "en"),
            ],
        )
        connection.executemany(
            "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (10, None, 3, 24, None, 1, 3),
                (24, None, 3, 11, None, 1, 3),
                (10, dog_sense, 3, 24, domestic_sense, 1, 1),
                (24, domestic_sense, 3, 11, animal_sense, 1, 1),
                # OEWN can publish the reciprocal hierarchy labels too. The
                # query path must still retain the requested relation's
                # canonical direction instead of whichever row is seen first.
                (11, animal_sense, 4, 24, domestic_sense, 1, 1),
                (24, domestic_sense, 4, 10, dog_sense, 1, 1),
                # Neither a partially scoped path nor a mismatched
                # intermediate sense may be presented as transitive.
                (24, domestic_sense, 3, 25, None, 1, 1),
                (24, "oewn:other-domestic-sense", 3, 26, "oewn:wrong", 1, 1),
            ],
        )

    with LexiconService(lexical_database, "data-test-v1") as instance:
        response = instance.dictionary_relations("dog", "hypernym")
        assert response["query"]["max_depth"] == 2
        assert response["query"]["transitive_limit"] == 5
        distances = [item["distance"] for item in response["results"]]
        assert distances == sorted(distances)
        animal_rows = [item for item in response["results"] if item["target_term"] == "animal"]
        assert len(animal_rows) == 2
        by_scope = {item["sense_scope"]: item for item in animal_rows}

        unsensed = by_scope["unsensed"]
        assert unsensed["relation_scope"] == "transitive"
        assert unsensed["distance"] == 2
        assert unsensed["direction"] == "outbound"
        assert [edge["target_term"] for edge in unsensed["path"]] == [
            "domestic animal",
            "animal",
        ]
        assert all(edge["source_sense_id"] is None for edge in unsensed["path"])
        assert all(edge["target_sense_id"] is None for edge in unsensed["path"])
        assert {edge["provenance"]["source"] for edge in unsensed["path"]} == {"ConceptNet"}

        sensed = by_scope["sense"]
        assert sensed["source_sense_id"] == dog_sense
        assert sensed["target_sense_id"] == animal_sense
        assert sensed["relation_scope"] == "transitive"
        assert sensed["distance"] == 2
        assert sensed["direction"] == "outbound"
        assert sensed["path"][0]["target_sense_id"] == domestic_sense
        assert sensed["path"][1]["source_sense_id"] == domestic_sense
        assert {edge["provenance"]["source"] for edge in sensed["path"]} == {"Open English WordNet"}
        assert not {
            "partially scoped",
            "wrong sense",
        }.intersection(item["target_term"] for item in response["results"])

        direct = next(
            item
            for item in response["results"]
            if item["target_term"] == "domestic animal" and item["sense_scope"] == "unsensed"
        )
        assert direct["relation_scope"] == "direct"
        assert direct["distance"] == 1
        assert len(direct["path"]) == 1

        reverse = instance.dictionary_relations("animal", "hyponym")
        reverse_dog = [item for item in reverse["results"] if item["target_term"] == "dog"]
        assert len(reverse_dog) == 2
        assert all(item["direction"] == "inbound" for item in reverse_dog)
        assert all(item["relation_scope"] == "transitive" for item in reverse_dog)
        assert all(item["distance"] == 2 for item in reverse_dog)
        assert all(
            [edge["target_term"] for edge in item["path"]] == ["domestic animal", "dog"]
            for item in reverse_dog
        )

        direct_only = instance.dictionary_relations("dog", "hypernym", max_depth=1)
        assert direct_only["results"]
        assert all(item["distance"] == 1 for item in direct_only["results"])
        assert (
            instance.dictionary_relations("dog", "hypernym", transitive_limit=0)["results"]
            == direct_only["results"]
        )

        allocated = instance.dictionary_relations(
            "dog", "hypernym", limit=2, max_depth=2, transitive_limit=1
        )
        assert [item["distance"] for item in allocated["results"]] == [1, 2]

        for small_limit in (1, 2, 3, 5):
            balanced = instance.dictionary_relations(
                "dog", "hypernym", limit=small_limit, transitive_limit=0
            )
            distances = [item["distance"] for item in balanced["results"]]
            assert distances == sorted(distances)
            assert distances.count(2) == 0


def test_transitive_frontier_is_bounded_and_preserves_scope_source_diversity(
    lexical_database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite3.connect(lexical_database) as connection:
        terms = [
            (24 + index, f"intermediate {index:02d}", f"intermediate {index:02d}", "en")
            for index in range(40)
        ]
        connection.executemany("INSERT INTO lexical_terms VALUES (?, ?, ?, ?)", terms)
        relations: list[tuple[Any, ...]] = []
        for index, (term_id, _term, _normalized, _language) in enumerate(terms):
            relations.append(
                (
                    10,
                    f"oewn:dog-sense-{index % 4}",
                    3,
                    term_id,
                    f"oewn:intermediate-{index:02d}",
                    1,
                    1,
                )
            )
            if index < 12:
                relations.append((10, None, 3, term_id, None, 1, 3))
        connection.executemany(
            "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?)", relations
        )

    with LexiconService(lexical_database, "data-test-v1") as instance:
        first_edges = instance._relation_rows(
            "dog",
            "en",
            3,
            target_language=None,
            source_sense_id=None,
            limit=256,
        )
        frontier = instance._transitive_frontier_rows(first_edges, 3, budget=16)
        assert len(frontier) == 16
        assert {int(row["provenance_id"]) for row in frontier} >= {1, 3}
        assert any(row["source_sense_id"] is None for row in frontier)
        assert len({row["source_sense_id"] for row in frontier if row["source_sense_id"]}) >= 4

        observed_source_counts: list[int] = []
        original = instance._relation_rows_many

        def observed(
            sources: list[tuple[str, str]],
            relation_code: int,
            *,
            target_language: str | None,
            branch_limit: int,
        ) -> list[dict[str, Any]]:
            observed_source_counts.append(len(set(sources)))
            return original(
                sources,
                relation_code,
                target_language=target_language,
                branch_limit=branch_limit,
            )

        monkeypatch.setattr(instance, "_relation_rows_many", observed)
        instance.dictionary_relations("dog", "hypernym")
        assert len(observed_source_counts) == 1
        assert 1 <= observed_source_counts[0] <= 16


def test_relations_prioritize_target_diversity_before_sense_variants(
    lexical_database: Path,
) -> None:
    with sqlite3.connect(lexical_database) as connection:
        distractors = [
            (25 + index, f"component {index:03d}", f"component {index:03d}", "en")
            for index in range(40)
        ]
        connection.executemany(
            "INSERT INTO lexical_terms VALUES (?, ?, ?, ?)",
            [(24, "axle", "axle", "en"), *distractors],
        )
        connection.executemany(
            "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(24, f"fixture:axle:{index:03d}", 6, 22, None, 1, 3) for index in range(110)]
            + [
                (term_id, None, 6, 22, None, 1, 3)
                for term_id, _term, _normalized, _language in distractors
            ],
        )

    with LexiconService(lexical_database, "data-test-v1") as instance:
        car = instance.dictionary_relations("car", "meronym")
        wheels = [item for item in car["results"] if item["target_term"] == "wheel"]
        assert len(wheels) == 1
        assert {
            "source_term": wheels[0]["source_term"],
            "relation": wheels[0]["relation"],
            "target_term": wheels[0]["target_term"],
            "direction": wheels[0]["direction"],
            "provenance": wheels[0]["provenance"]["source"],
        } == {
            "source_term": "car",
            "relation": "meronym",
            "target_term": "wheel",
            "direction": "inbound",
            "provenance": "ConceptNet",
        }

        wheel = instance.dictionary_relations("wheel", "holonym")
        assert [
            (
                item["source_term"],
                item["relation"],
                item["target_term"],
                item["direction"],
                item["provenance"]["source"],
            )
            for item in wheel["results"]
            if item["target_term"] == "car"
        ] == [("wheel", "holonym", "car", "outbound", "ConceptNet")]


def test_default_relation_limit_prefers_lexical_targets_over_phrase_noise(
    lexical_database: Path,
) -> None:
    anchors = [
        (24, "knife", "knife", "en"),
        (25, "cutting", "cutting", "en"),
        (26, "book", "book", "en"),
        (27, "library", "library", "en"),
    ]
    use_noise = [
        (28 + index, f"use phrase {index:02d}", f"use phrase {index:02d}", "en")
        for index in range(30)
    ]
    location_noise = [
        (58 + index, f"place phrase {index:02d}", f"place phrase {index:02d}", "en")
        for index in range(30)
    ]
    with sqlite3.connect(lexical_database) as connection:
        connection.executemany(
            "INSERT INTO lexical_terms VALUES (?, ?, ?, ?)",
            [*anchors, *use_noise, *location_noise],
        )
        connection.executemany(
            "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(24, None, 9, 25, None, 1, 3)]
            + [
                (24, None, 9, term_id, None, 1, 3)
                for term_id, _term, _normalized, _language in use_noise
            ]
            + [(26, None, 11, 27, None, 1, 3)]
            + [
                (26, None, 11, term_id, None, 1, 3)
                for term_id, _term, _normalized, _language in location_noise
            ],
        )

    with LexiconService(lexical_database, "data-test-v1") as instance:
        cases = (
            ("knife", "used_for", "cutting"),
            ("book", "at_location", "library"),
        )
        for word, relation, target in cases:
            response = instance.dictionary_relations(word, relation)
            assert len(response["results"]) == 20
            matches = [item for item in response["results"] if item["target_term"] == target]
            assert len(matches) == 1
            assert matches[0]["relation_scope"] == "direct"
            assert matches[0]["distance"] == 1
            assert all(item["distance"] == 1 for item in response["results"])


def test_wordplay_modes_are_deterministic_and_exclude_query(service: LexiconService) -> None:
    assert [item["term"] for item in service.dictionary_wordplay("rhyme", "cat")["results"]] == [
        "bat"
    ]
    assert [
        item["term"] for item in service.dictionary_wordplay("sounds_like", "knight")["results"]
    ] == ["night"]
    assert {
        item["term"] for item in service.dictionary_wordplay("spelled_like", "h?llo")["results"]
    } == {"hallo", "hello"}
    assert service.dictionary_wordplay("spelled_like", "h[ae]llo")["results"] == []
    assert [item["term"] for item in service.dictionary_wordplay("prefix", "hel")["results"]] == [
        "hello",
        "help",
    ]
    exact = service.rhymes("cat")
    assert exact["type"] == "rhymes"
    assert exact["query"]["mode"] == "exact"
    assert [item["term"] for item in exact["results"]] == ["bat"]
    assert all(item["mode"] == "exact" for item in exact["results"])


def test_all_public_tools_echo_the_effective_result_limit(
    service: LexiconService,
) -> None:
    responses = (
        service.dictionary_lookup("bank", limit=2),
        service.dictionary_synonyms("bank", limit=6, unsensed_limit=5),
        service.dictionary_translate("bank", "en", "de", limit=2),
        service.dictionary_relations("dog", "hypernym", limit=5),
        service.dictionary_semantic_neighbors("cat", limit=4),
        service.rhymes("cat", limit=3),
    )
    assert [response["query"]["limit"] for response in responses] == [
        2,
        6,
        2,
        5,
        4,
        3,
    ]


def test_semantic_artifacts_may_be_absent_and_backend_is_lazy(
    lexical_database: Path, tmp_path: Path
) -> None:
    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()
    with LexiconService(
        lexical_database, "data-test-v1", semantic_directory=semantic_dir
    ) as unavailable:
        result = unavailable.dictionary_semantic_neighbors("cat")
        assert result["available"] is False
        assert result["count"] == 0
        assert result["results"] == []

    fake = FakeSemanticSearch()
    with LexiconService(lexical_database, "data-test-v1", semantic_search=fake) as available:
        result = available.dictionary_semantic_neighbors(
            "\uff23\uff21\uff34", "en", "de", limit=4, min_similarity=0.5
        )
        assert result["available"] is True
        assert result["results"][0]["term"] == "Katze"
        assert fake.calls == [("cat", "en", "de", 4, 0.5)]


@pytest.mark.parametrize(
    ("method", "arguments", "message"),
    [
        ("dictionary_lookup", ("",), "word cannot be empty"),
        ("dictionary_lookup", ("cat", "not a tag!"), "BCP-47"),
        ("dictionary_lookup", ("cat", "en", None, 0), "between 1 and 100"),
        (
            "dictionary_lookup",
            ("cat", "en", None, 8, -1),
            "automatic allocation is reserved but unsupported in v1",
        ),
        (
            "dictionary_lookup",
            ("cat", "en", None, 8, 8, 101),
            "pronunciations_limit must be between 0 and 100",
        ),
        (
            "dictionary_translate",
            ("cat", "en", "de", None, None, 20, 0),
            "max_senses must be between 1 and 100",
        ),
        ("dictionary_relations", ("cat", "synonym"), "relation must be one of"),
        (
            "dictionary_synonyms",
            ("cat", "en", None, None, 20, 0),
            "max_senses must be between 1 and 100",
        ),
        (
            "dictionary_synonyms",
            ("cat", "en", None, None, 20, 20, -1),
            "automatic allocation is reserved but unsupported in v1",
        ),
        (
            "dictionary_synonyms",
            ("cat", "en", None, None, 4, 20, 5),
            "unsensed_limit must not exceed limit",
        ),
        (
            "dictionary_relations",
            ("cat", "related", "en", None, None, 20, 3),
            "max_depth must be between 1 and 2",
        ),
        (
            "dictionary_relations",
            ("cat", "related", "en", None, None, 20, 2, -1),
            "automatic allocation is reserved but unsupported in v1",
        ),
        (
            "dictionary_relations",
            ("cat", "related", "en", None, None, 4, 2, 5),
            "transitive_limit must not exceed limit",
        ),
        ("dictionary_wordplay", ("anagram", "cat"), "mode must be one of"),
        (
            "dictionary_semantic_neighbors",
            ("cat", "en", None, 20, float("nan")),
            "between -1 and 1",
        ),
    ],
)
def test_invalid_inputs_raise_clear_validation_errors(
    service: LexiconService, method: str, arguments: tuple[Any, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        getattr(service, method)(*arguments)


def test_locator_accepts_atomic_activation_and_rejects_escape(
    tmp_path: Path, lexical_database: Path
) -> None:
    root = tmp_path / "data-root"
    version = root / "versions" / "data-test-v1"
    version.mkdir(parents=True)
    target = version / "lexicon.sqlite3"
    target.write_bytes(lexical_database.read_bytes())
    (version / "manifest.json").write_text(
        json.dumps({"version": "data-test-v1"}), encoding="utf-8"
    )
    (root / "current.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version": "data-test-v1",
                "path": "versions/data-test-v1",
            }
        ),
        encoding="utf-8",
    )
    active = DatasetLocator(root).active()
    assert active.version == "data-test-v1"
    assert active.lexical_database == target

    (root / "current.json").write_text(
        json.dumps({"version": "data-test-v1", "path": "../outside"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="escapes"):
        DatasetLocator(root).active()


def test_service_rejects_incompatible_dataset_metadata(lexical_database: Path) -> None:
    with sqlite3.connect(lexical_database) as connection:
        connection.execute("UPDATE metadata SET value = '99' WHERE key = 'schema_version'")
        connection.commit()
    with pytest.raises(RuntimeError, match="Unsupported lexical schema version"):
        LexiconService(lexical_database, "data-test-v1")


def test_english_profile_rejects_contaminated_lexical_artifact(
    lexical_database: Path,
) -> None:
    with pytest.raises(RuntimeError, match="non-English lexical term"):
        LexiconService(
            lexical_database,
            "data-test-v1",
            dataset_profile="english",
        )


def test_english_profile_fails_closed_for_cross_language_requests(
    tmp_path: Path,
    lexical_database: Path,
) -> None:
    english_database = tmp_path / "english.sqlite3"
    english_database.write_bytes(lexical_database.read_bytes())
    with sqlite3.connect(english_database) as connection:
        connection.execute("DELETE FROM lexical_terms WHERE language <> 'en'")
        connection.commit()
    semantic = FakeSemanticSearch()

    with LexiconService(
        english_database,
        "data-test-v1",
        semantic_search=semantic,
        dataset_profile="english",
    ) as instance:
        responses = [
            instance.dictionary_lookup("chat", "fr"),
            instance.dictionary_synonyms("chat", "fr"),
            instance.dictionary_translate("bank", "en", "de"),
            instance.dictionary_relations("dog", "hypernym", "en", "de"),
            instance.dictionary_semantic_neighbors("cat", "de"),
        ]
        for response in responses:
            assert response["available"] is False
            assert response["unavailable_reason"] == "english_profile_supports_only_en"
            assert response["results"] == []
        assert responses[1]["candidate_count"] == 0
        assert responses[2]["candidate_count"] == 0
        assert semantic.calls == []

        english = instance.dictionary_semantic_neighbors("cat", "en")
        assert semantic.calls == [("cat", "en", "en", 20, None)]
        assert english["available"] is True
        assert english["results"] == []
