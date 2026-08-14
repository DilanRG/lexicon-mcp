from __future__ import annotations

from typing import Any

import pytest

from lexicon_mcp.runtime.acceptance import (
    AcceptanceDatasetUnavailable,
    load_acceptance_dataset,
)
from lexicon_mcp.runtime.offline import deny_network
from lexicon_mcp.runtime.service import LexiconService
from lexicon_mcp.server import create_mcp


def _dataset_or_skip() -> Any:
    try:
        return load_acceptance_dataset()
    except AcceptanceDatasetUnavailable as exc:
        pytest.skip(str(exc))


def _assert_direct_relation(
    response: dict[str, Any],
    *,
    source: str,
    relation: str,
    target: str,
    direction: str,
    provenance_source: str,
    sense_scope: str,
) -> None:
    matches = [
        item
        for item in response["results"]
        if str(item["target_term"]).casefold() == target.casefold()
        and item["provenance"]["source"] == provenance_source
        and item["sense_scope"] == sense_scope
    ]
    assert len(matches) == 1, (source, relation, target, response)
    item = matches[0]
    assert str(item["source_term"]).casefold() == source.casefold()
    assert item["source_language"] == item["target_language"] == "en"
    if sense_scope == "unsensed":
        assert item["source_sense_id"] is None
        assert item["target_sense_id"] is None
    else:
        assert item["source_sense_id"]
        assert item["target_sense_id"]
    assert item["relation"] == relation
    assert item["direction"] == direction
    assert item["relation_scope"] == "direct"
    assert item["distance"] == 1
    assert len(item["path"]) == 1
    edge = item["path"][0]
    assert (
        str(edge["source_term"]).casefold(),
        edge["relation"],
        str(edge["target_term"]).casefold(),
        edge["direction"],
        edge["provenance"],
    ) == (
        source.casefold(),
        relation,
        target.casefold(),
        direction,
        item["provenance"],
    )


def _assert_transitive_hierarchy_relation(
    response: dict[str, Any],
    *,
    source: str,
    relation: str,
    target: str,
    direction: str,
    intermediate: str,
    provenance_source: str,
    source_sense_id: str | None,
    intermediate_sense_id: str | None,
    target_sense_id: str | None,
) -> None:
    matches = [
        item
        for item in response["results"]
        if str(item["target_term"]).casefold() == target.casefold()
        and item["relation_scope"] == "transitive"
        and item["distance"] == 2
        and item["provenance"]["source"] == provenance_source
        and item["source_sense_id"] == source_sense_id
        and item["target_sense_id"] == target_sense_id
    ]
    assert matches, (source, relation, target, response)
    item = matches[0]
    assert str(item["source_term"]).casefold() == source.casefold()
    assert item["source_language"] == item["target_language"] == "en"
    assert item["relation"] == relation
    assert item["direction"] == direction
    assert len(item["path"]) == 2
    first, second = item["path"]
    assert str(first["source_term"]).casefold() == source.casefold()
    assert str(first["target_term"]).casefold() == intermediate.casefold()
    assert str(second["source_term"]).casefold() == intermediate.casefold()
    assert str(second["target_term"]).casefold() == target.casefold()
    assert first["target_language"] == second["source_language"] == "en"
    assert all(edge["relation"] == relation for edge in item["path"])
    assert all(edge["direction"] == direction for edge in item["path"])
    assert {edge["provenance"]["source"] for edge in item["path"]} == {provenance_source}
    assert all(edge["provenance"] == item["provenance"] for edge in item["path"])
    assert (
        first["source_sense_id"],
        first["target_sense_id"],
        second["source_sense_id"],
        second["target_sense_id"],
    ) == (
        source_sense_id,
        intermediate_sense_id,
        intermediate_sense_id,
        target_sense_id,
    )


@pytest.mark.full_corpus
def test_required_full_corpus_anchors_offline() -> None:
    dataset = _dataset_or_skip()
    with LexiconService.from_active_dataset(dataset) as service, deny_network():
        bank = service.dictionary_lookup(
            "bank",
            "en",
            limit=100,
            examples_limit=8,
            pronunciations_limit=8,
            translations_limit=20,
        )
        glosses = [str(item["gloss"] or "").casefold() for item in bank["results"]]
        assert any("financial" in gloss for gloss in glosses)
        assert any("river" in gloss or "water" in gloss for gloss in glosses)
        assert sum(len(sense["examples"]) for sense in bank["results"]) <= 8
        assert sum(len(sense["pronunciations"]) for sense in bank["results"]) <= 8
        assert sum(len(sense["translations"]) for sense in bank["results"]) <= 20
        assert any(sense["translations"] for sense in bank["results"])
        assert any("translations" in sense["truncated_fields"] for sense in bank["results"])

        bounded_bank = service.dictionary_lookup("bank", "en", limit=10)
        bounded_by_gloss = {
            str(item["gloss"] or "").casefold(): item for item in bounded_bank["results"]
        }
        bounded_finance = bounded_by_gloss["institution"]
        bounded_river = bounded_by_gloss["edge of river or lake"]
        assert bounded_finance["sense_id"].startswith("wikt:labeled:")
        assert bounded_river["sense_id"].startswith("wikt:labeled:")
        scoped_river_translation = service.dictionary_translate(
            "bank",
            "en",
            "de",
            sense_id=bounded_river["sense_id"],
        )
        assert scoped_river_translation["candidate_count"] == 1
        assert scoped_river_translation["results"][0]["sense_id"] == bounded_river["sense_id"]
        assert scoped_river_translation["results"][0]["translations"][0]["term"] == "Ufer"

        translations = service.dictionary_translate("bank", "en", "de")
        assert translations["query"]["limit"] == 20
        assert translations["query"]["max_senses"] == 100
        assert translations["candidate_count"] <= 20
        bank_groups = [
            group
            for group in translations["results"]
            if any(item["term"] == "Bank" for item in group["translations"])
            and any(
                marker in str(group["gloss"] or "").casefold()
                for marker in ("financial", "institution")
            )
        ]
        ufer_groups = [
            group
            for group in translations["results"]
            if any(item["term"] == "Ufer" for item in group["translations"])
            and any(
                marker in str(group["gloss"] or "").casefold()
                for marker in ("river", "water", "edge", "lake")
            )
        ]
        assert bank_groups and len(ufer_groups) == 1
        assert bank_groups[0]["sense_id"] != ufer_groups[0]["sense_id"]
        assert bank_groups[0]["sense_scope"] == ufer_groups[0]["sense_scope"] == "sense"

        lead = service.dictionary_lookup("lead", "en", limit=100, pronunciations_limit=100)
        pronunciations = {
            str(item["ipa"]) for sense in lead["results"] for item in sense["pronunciations"]
        }
        assert any("lɛd" in ipa for ipa in pronunciations)
        assert any("li\u02d0d" in ipa or "li:d" in ipa for ipa in pronunciations)

        # Keep non-ASCII source literals escaped so Windows PowerShell-based
        # audit helpers cannot silently rewrite the acceptance anchors.
        anchors = (
            ("Schloss", "de", 2),
            ("Gift", "de", 1),
            ("bright", "en", 1),
            ("feliz", "es", 1),
            ("caf\u00e9", "fr", 1),
            ("\u732b", "ja", 1),
            ("\u0633\u0644\u0627\u0645", "ar", 1),
            ("\u0918\u0930", "hi", 1),
        )
        for word, language, minimum_senses in anchors:
            lookup = service.dictionary_lookup(word, language, limit=20)
            assert lookup["count"] >= minimum_senses, (word, language, lookup)
            assert all(item["language"] == language for item in lookup["results"])
            assert all(item["part_of_speech"] for item in lookup["results"])
            assert any(item["gloss"] for item in lookup["results"])

        direct_relations = (
            ("hot", "antonym", "cold", "outbound", "Open English WordNet", "sense"),
            ("dog", "capable_of", "bark", "outbound", "ConceptNet 5.7", "unsensed"),
            ("poodle", "hypernym", "dog", "outbound", "ConceptNet 5.7", "unsensed"),
            ("car", "meronym", "wheel", "inbound", "ConceptNet 5.7", "unsensed"),
            ("knife", "used_for", "cutting", "outbound", "ConceptNet 5.7", "unsensed"),
            ("book", "at_location", "library", "outbound", "ConceptNet 5.7", "unsensed"),
        )
        for word, relation, target, direction, provenance_source, scope in direct_relations:
            response = service.dictionary_relations(word, relation, "en")
            _assert_direct_relation(
                response,
                source=word,
                relation=relation,
                target=target,
                direction=direction,
                provenance_source=provenance_source,
                sense_scope=scope,
            )

        dog_hypernyms = service.dictionary_relations(
            "dog", "hypernym", "en", limit=100, max_depth=2, transitive_limit=20
        )
        _assert_transitive_hierarchy_relation(
            dog_hypernyms,
            source="dog",
            relation="hypernym",
            target="animal",
            direction="outbound",
            intermediate="domestic animal",
            provenance_source="ConceptNet 5.7",
            source_sense_id=None,
            intermediate_sense_id=None,
            target_sense_id=None,
        )
        dog_sense = "oewn:oewn-dog__1.05.00.."
        domestic_animal_sense = "oewn:oewn-domestic_animal__1.05.00.."
        animal_sense = "oewn:oewn-animal__1.03.00.."
        sensed_dog_hypernyms = service.dictionary_relations(
            "dog",
            "hypernym",
            "en",
            sense_id=dog_sense,
            limit=100,
            max_depth=2,
            transitive_limit=20,
        )
        _assert_transitive_hierarchy_relation(
            sensed_dog_hypernyms,
            source="dog",
            relation="hypernym",
            target="animal",
            direction="outbound",
            intermediate="domestic animal",
            provenance_source="Open English WordNet",
            source_sense_id=dog_sense,
            intermediate_sense_id=domestic_animal_sense,
            target_sense_id=animal_sense,
        )
        sensed_animal_hyponyms = service.dictionary_relations(
            "animal",
            "hyponym",
            "en",
            sense_id=animal_sense,
            limit=100,
            max_depth=2,
            transitive_limit=20,
        )
        _assert_transitive_hierarchy_relation(
            sensed_animal_hyponyms,
            source="animal",
            relation="hyponym",
            target="dog",
            direction="inbound",
            intermediate="domestic animal",
            provenance_source="Open English WordNet",
            source_sense_id=animal_sense,
            intermediate_sense_id=domestic_animal_sense,
            target_sense_id=dog_sense,
        )

        # These reverse queries prove that one stored assertion is exposed
        # from its target with the same logical relation and inverted direction.
        inverse_relations = (
            ("cold", "antonym", "hot", "outbound", "Open English WordNet", "sense"),
            ("bark", "capable_of", "dog", "inbound", "ConceptNet 5.7", "unsensed"),
            ("wheel", "holonym", "car", "outbound", "ConceptNet 5.7", "unsensed"),
            ("cutting", "used_for", "knife", "inbound", "ConceptNet 5.7", "unsensed"),
            ("library", "at_location", "book", "inbound", "ConceptNet 5.7", "unsensed"),
        )
        for word, relation, target, direction, provenance_source, scope in inverse_relations:
            response = service.dictionary_relations(word, relation, "en")
            _assert_direct_relation(
                response,
                source=word,
                relation=relation,
                target=target,
                direction=direction,
                provenance_source=provenance_source,
                sense_scope=scope,
            )

        dog_synonyms = service.dictionary_synonyms("dog", "en")
        dog_candidates = [
            candidate for group in dog_synonyms["results"] for candidate in group["synonyms"]
        ]
        assert len(dog_candidates) <= 20
        assert {candidate["language"] for candidate in dog_candidates} == {"en"}
        assert any(
            str(group["gloss"] or "").casefold() == "animal" and group["sense_scope"] == "sense"
            for group in dog_synonyms["results"]
        )
        dog_unsensed = [
            candidate
            for group in dog_synonyms["results"]
            if group["sense_scope"] == "unsensed"
            for candidate in group["synonyms"]
        ]
        assert len(dog_unsensed) <= 5
        assert {
            "canine",
            "cur",
            "doggy",
            "hound",
            "pooch",
        } <= {str(candidate["term"]).casefold() for candidate in dog_unsensed}

        important = service.dictionary_synonyms("important", "en")
        assert any(
            group["sense_scope"] == "unsensed"
            and {"significant", "key"}
            <= {str(candidate["term"]).casefold() for candidate in group["synonyms"]}
            for group in important["results"]
        )
        for word in ("important", "bright", "bank"):
            for small_limit in (1, 2, 3, 5):
                small = service.dictionary_synonyms(word, "en", limit=small_limit, unsensed_limit=0)
                scoped_count = sum(
                    len(group["synonyms"])
                    for group in small["results"]
                    if group["sense_scope"] == "sense"
                )
                unsensed_count = sum(
                    len(group["synonyms"])
                    for group in small["results"]
                    if group["sense_scope"] == "unsensed"
                )
                assert scoped_count + unsensed_count <= small_limit
                assert scoped_count >= min(small_limit, 4)
                assert unsensed_count == 0

        assert any(
            item["term"].casefold() == "bat"
            for item in service.dictionary_wordplay("rhyme", "cat", 100)["results"]
        )
        assert any(
            item["term"].casefold() == "night"
            for item in service.dictionary_wordplay("sounds_like", "knight", 100)["results"]
        )
        assert any(
            item["term"].casefold() == "hello"
            for item in service.dictionary_wordplay("spelled_like", "h?llo", 100)["results"]
        )
        assert any(
            item["term"].casefold() == "serendipity"
            for item in service.dictionary_wordplay("prefix", "seren", 100)["results"]
        )

        english = service.dictionary_semantic_neighbors("cat", "en", "en", 20)
        german = service.dictionary_semantic_neighbors("cat", "en", "de", 20)
        assert english["available"] and english["results"]
        assert german["available"] and german["results"]
        assert all(item["language"] == "en" for item in english["results"])
        assert all(item["language"] == "de" for item in german["results"])
        assert all(item["provenance"]["license"] for item in english["results"])


@pytest.mark.full_corpus
@pytest.mark.asyncio
async def test_full_corpus_mcp_exposes_exactly_six_offline_tools() -> None:
    dataset = _dataset_or_skip()
    with LexiconService.from_active_dataset(dataset) as service, deny_network():
        mcp = create_mcp(service)
        tools = await mcp.list_tools()
        assert {tool.name for tool in tools} == {
            "dictionary_lookup",
            "dictionary_synonyms",
            "dictionary_translate",
            "dictionary_relations",
            "dictionary_semantic_neighbors",
            "dictionary_wordplay",
        }
        assert all(tool.outputSchema is not None for tool in tools)
        _content, structured = await mcp.call_tool(
            "dictionary_lookup", {"word": "Schloss", "language": "de"}
        )
        assert structured is not None
        assert structured["results"]
