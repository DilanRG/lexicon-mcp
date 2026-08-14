from __future__ import annotations

from pathlib import Path

from lexicon_mcp.pipeline import BuildInputs, build_full_corpus
from lexicon_mcp.runtime.service import LexiconService


def test_pipeline_artifacts_are_directly_queryable_by_runtime(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "build_inputs"
    output = tmp_path / "dataset"
    build_full_corpus(
        BuildInputs(
            oewn=fixtures / "oewn.xml",
            wiktextract=(fixtures / "kaikki.jsonl",),
            conceptnet=fixtures / "conceptnet.tsv",
            numberbatch=fixtures / "numberbatch.txt",
            cmudict=fixtures / "cmudict.dict",
            notices_dir=fixtures / "notices",
        ),
        output,
        tmp_path / "build-state",
        dataset_version="fixture-v1",
        retrieved_at="2026-01-01T00:00:00Z",
    )

    assert (output / "lexicon.sqlite3").is_file()
    with LexiconService(
        output / "lexicon.sqlite3",
        "fixture-v1",
        semantic_directory=output / "semantic",
    ) as service:
        bank = service.dictionary_lookup("bank", "en")
        assert {item["gloss"] for item in bank["results"]} >= {
            "a financial institution",
            "the edge of a river",
        }

        translations = service.dictionary_translate("bank", "en", "de", limit=100)
        by_gloss = {
            item["gloss"]: item["translations"][0]["term"]
            for item in translations["results"]
        }
        assert by_gloss["institution"] == "Bank"
        assert by_gloss["edge of a river or lake"] == "Ufer"
        assert len({item["sense_id"] for item in translations["results"]}) == 2

        relation_pairs = (
            ("dog", "hypernym", "animal", "outbound"),
            ("animal", "hyponym", "dog", "inbound"),
            ("poodle", "hypernym", "dog", "outbound"),
            ("dog", "hyponym", "poodle", "inbound"),
            ("wheel", "holonym", "car", "outbound"),
            ("car", "meronym", "wheel", "inbound"),
            ("knife", "used_for", "cutting", "outbound"),
            ("cutting", "used_for", "knife", "inbound"),
            ("book", "at_location", "library", "outbound"),
            ("library", "at_location", "book", "inbound"),
        )
        for word, relation, target, direction in relation_pairs:
            response = service.dictionary_relations(word, relation)
            matches = [
                item
                for item in response["results"]
                if item["target_term"] == target
                and item["relation_scope"] == "direct"
            ]
            assert len(matches) == 1
            item = matches[0]
            assert (
                item["source_term"],
                item["relation"],
                item["target_term"],
                item["direction"],
                item["distance"],
            ) == (word, relation, target, direction, 1)
            assert len(item["path"]) == 1
            edge = item["path"][0]
            assert (
                edge["source_term"],
                edge["relation"],
                edge["target_term"],
                edge["direction"],
                edge["provenance"],
            ) == (word, relation, target, direction, item["provenance"])

        rhyme = service.dictionary_wordplay("rhyme", "cat")
        assert [item["term"] for item in rhyme["results"]] == ["bat"]

        neighbors = service.dictionary_semantic_neighbors("cat", "en", "de")
        assert neighbors["available"] is True
        assert neighbors["count"] >= 1
        assert all(item["language"] == "de" for item in neighbors["results"])
