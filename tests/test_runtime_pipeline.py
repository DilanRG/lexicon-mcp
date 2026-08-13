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

        translations = service.dictionary_translate("bank", "en", "de")
        by_gloss = {
            item["gloss"]: item["translations"][0]["term"]
            for item in translations["results"]
        }
        assert by_gloss["a financial institution"] == "Bank"
        assert by_gloss["the edge of a river"] == "Ufer"

        relation = service.dictionary_relations("dog", "hypernym")
        assert relation["results"][0]["target_term"] == "animal"
        assert relation["results"][0]["direction"] == "outbound"

        rhyme = service.dictionary_wordplay("rhyme", "cat")
        assert [item["term"] for item in rhyme["results"]] == ["bat"]

        neighbors = service.dictionary_semantic_neighbors("cat", "en", "de")
        assert neighbors["available"] is True
        assert neighbors["count"] >= 1
        assert all(item["language"] == "de" for item in neighbors["results"])
