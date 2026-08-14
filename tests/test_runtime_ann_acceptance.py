from __future__ import annotations

from pathlib import Path

import pytest

from lexicon_mcp.pipeline import BuildInputs, build_full_corpus
from lexicon_mcp.runtime.acceptance import (
    AcceptanceDatasetUnavailable,
    load_acceptance_dataset,
)
from lexicon_mcp.runtime.ann_validation import DEFAULT_LANGUAGES, validate_ann_acceptance
from lexicon_mcp.runtime.locator import ActiveDataset
from lexicon_mcp.runtime.normalization import normalize_key
from lexicon_mcp.runtime.offline import deny_network


def test_ann_validator_uses_pipeline_fixture_read_only(tmp_path: Path) -> None:
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
    dataset = ActiveDataset(tmp_path, "fixture-v1", output, {"profile": "full"})

    report = validate_ann_acceptance(
        dataset, languages=("en",), seeds_per_language=1, k=2, chunk_size=2
    )

    assert len(report.results) == 2
    assert {result.index_scope for result in report.results} == {
        "global",
        "language_shard",
    }
    for result in report.results:
        assert result.deterministic is True
        assert result.strict_language_filtering is True
        assert len(result.ann_semantic_ids) == 2
        identities = set(zip(result.ann_languages, result.ann_terms, strict=True))
        assert len(identities) == len(result.ann_terms)
        assert result.seed_semantic_id not in result.ann_semantic_ids
        assert (result.language, normalize_key(result.seed_term)) not in identities
        assert result.ann_semantic_ids == result.exact_semantic_ids
        assert result.recall_at_k == 1.0


@pytest.mark.full_corpus
@pytest.mark.ann
def test_numberbatch_ann_recall_and_language_shards_offline() -> None:
    try:
        dataset = load_acceptance_dataset()
    except AcceptanceDatasetUnavailable as exc:
        pytest.skip(str(exc))

    with deny_network():
        report = validate_ann_acceptance(dataset)

    assert report.languages == DEFAULT_LANGUAGES
    assert len(report.results) == 200
    assert all(
        sum(result.language == language for result in report.results) == 20
        for language in DEFAULT_LANGUAGES
    )
    assert sum(result.index_scope == "global" for result in report.results) == 100
    assert sum(result.index_scope == "language_shard" for result in report.results) == 100
    assert all(result.deterministic for result in report.results)
    assert all(result.strict_language_filtering for result in report.results)
    assert all(len(result.ann_semantic_ids) == 20 for result in report.results)
    assert all(len(result.exact_semantic_ids) == 20 for result in report.results)
    assert all(
        len(result.ann_terms)
        == len(set(zip(result.ann_languages, result.ann_terms, strict=True)))
        for result in report.results
    )
    assert all(
        result.seed_semantic_id not in result.ann_semantic_ids for result in report.results
    )
    assert all(
        (result.language, normalize_key(result.seed_term))
        not in set(zip(result.ann_languages, result.ann_terms, strict=True))
        for result in report.results
    )
    assert all(
        len(result.exact_terms)
        == len(set(zip(result.exact_languages, result.exact_terms, strict=True)))
        for result in report.results
    )
    assert report.global_recall_at_k >= 0.90, {
        "global_recall_at_20": report.global_recall_at_k,
        "worst_global_seeds": sorted(
            (
                (result.language, result.seed_term, result.recall_at_k)
                for result in report.results
                if result.index_scope == "global"
            ),
            key=lambda item: item[2],
        )[:10],
    }
    assert report.shard_recall_at_k >= 0.90, {
        "shard_recall_at_20": report.shard_recall_at_k,
        "per_language": report.per_language_recall,
        "worst_seeds": sorted(
            (
                (result.language, result.seed_term, result.recall_at_k)
                for result in report.results
                if result.index_scope == "language_shard"
            ),
            key=lambda item: item[2],
        )[:10],
    }
