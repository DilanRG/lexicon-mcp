from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from lexicon_mcp.pipeline.schema import (
    create_lexical_query_indexes,
    create_lexical_schema,
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
    connection.executemany(
        "INSERT INTO senses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", senses
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

    lead = service.dictionary_lookup("lead")
    assert {item["pronunciations"][0]["ipa"] for item in lead["results"]} == {
        "/lɛd/",
        "/li\u02d0d/",
    }


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
    terms = {
        candidate["term"]
        for group in bright["results"]
        for candidate in group["synonyms"]
    }
    assert terms == {"luminous", "radiant"}


def test_translation_never_crosses_requested_sense(service: LexiconService) -> None:
    finance = service.dictionary_translate(
        "bank", "en", "de", sense_id="oewn:bank-finance-n"
    )
    assert finance["count"] == 1
    assert finance["results"][0]["translations"][0]["term"] == "Bank"
    assert all(
        item["term"] != "Ufer"
        for group in finance["results"]
        for item in group["translations"]
    )

    all_senses = service.dictionary_translate("bank", "en", "de")
    mapping = {
        group["sense_id"]: group["translations"][0]["term"]
        for group in all_senses["results"]
    }
    assert mapping == {"oewn:bank-finance-n": "Bank", "oewn:bank-river-n": "Ufer"}


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
    assert [
        item["term"] for item in service.dictionary_wordplay("prefix", "hel")["results"]
    ] == ["hello", "help"]


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
    with LexiconService(
        lexical_database, "data-test-v1", semantic_search=fake
    ) as available:
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
        ("dictionary_relations", ("cat", "synonym"), "relation must be one of"),
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
        connection.execute(
            "UPDATE metadata SET value = '99' WHERE key = 'schema_version'"
        )
        connection.commit()
    with pytest.raises(RuntimeError, match="Unsupported lexical schema version"):
        LexiconService(lexical_database, "data-test-v1")
