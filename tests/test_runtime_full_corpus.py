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


@pytest.mark.full_corpus
def test_required_full_corpus_anchors_offline() -> None:
    dataset = _dataset_or_skip()
    with LexiconService.from_active_dataset(dataset) as service, deny_network():
        bank = service.dictionary_lookup("bank", "en", limit=100)
        glosses = [str(item["gloss"] or "").casefold() for item in bank["results"]]
        assert any("financial" in gloss for gloss in glosses)
        assert any("river" in gloss or "water" in gloss for gloss in glosses)
        lookup_translations = {
            item["term"]
            for sense in bank["results"]
            for item in sense["translations"]
            if item["language"] == "de"
        }
        assert {"Bank", "Ufer"} <= lookup_translations

        translations = service.dictionary_translate("bank", "en", "de", limit=100)
        bank_groups = [
            group
            for group in translations["results"]
            if any(item["term"] == "Bank" for item in group["translations"])
        ]
        ufer_groups = [
            group
            for group in translations["results"]
            if any(item["term"] == "Ufer" for item in group["translations"])
        ]
        assert len(bank_groups) == len(ufer_groups) == 1
        assert bank_groups[0]["sense_id"] != ufer_groups[0]["sense_id"]
        assert bank_groups[0]["sense_scope"] == ufer_groups[0]["sense_scope"] == "sense"
        assert any(
            marker in str(bank_groups[0]["gloss"] or "").casefold()
            for marker in ("financial", "institution")
        )
        assert any(
            marker in str(ufer_groups[0]["gloss"] or "").casefold()
            for marker in ("river", "water", "edge", "lake")
        )

        lead = service.dictionary_lookup("lead", "en", limit=100)
        pronunciations = {
            str(item["ipa"])
            for sense in lead["results"]
            for item in sense["pronunciations"]
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

        relations = (
            ("hot", "antonym", "cold"),
            ("dog", "hypernym", "animal"),
            ("dog", "capable_of", "bark"),
            ("poodle", "hypernym", "dog"),
            ("car", "meronym", "wheel"),
            ("knife", "used_for", "cutting"),
            ("book", "at_location", "library"),
        )
        for word, relation, target in relations:
            response = service.dictionary_relations(word, relation, "en", limit=100)
            assert any(
                str(item["target_term"]).casefold() == target for item in response["results"]
            ), (word, relation, response)
            assert all(item["direction"] for item in response["results"])

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
