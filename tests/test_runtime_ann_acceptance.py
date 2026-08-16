from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from usearch.index import Index

from lexicon_mcp.pipeline import BuildInputs, build_full_corpus
from lexicon_mcp.runtime.acceptance import (
    AcceptanceDatasetUnavailable,
    load_acceptance_dataset,
)
from lexicon_mcp.runtime.ann_validation import (
    DEFAULT_LANGUAGES,
    validate_ann_acceptance,
    validate_ann_acceptance_packs,
)
from lexicon_mcp.runtime.locator import ActiveDataset
from lexicon_mcp.runtime.normalization import normalize_key
from lexicon_mcp.runtime.offline import deny_network


def test_ann_validator_uses_pipeline_fixture_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    original_metadata = Index.metadata

    def mmap_metadata_only(path_or_buffer: Any) -> Any:
        if isinstance(path_or_buffer, (str, os.PathLike)):
            raise AssertionError("large indexes must not use path metadata")
        return original_metadata(path_or_buffer)

    def forbidden_restore(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("large indexes must not use USearch restore")

    monkeypatch.setattr(Index, "metadata", staticmethod(mmap_metadata_only))
    monkeypatch.setattr(Index, "restore", staticmethod(forbidden_restore))
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

    evidence = report.to_evidence()
    assert evidence == json.loads(report.to_json())
    assert report.to_json() == report.to_json()
    assert "\n" not in report.to_json()
    assert evidence["comparison_count"] == 2
    assert evidence["unique_seed_count"] == 1
    assert evidence["global"] == {
        "comparison_count": 1,
        "deterministic": True,
        "recall_at_k": 1.0,
    }
    assert evidence["language_shards"] == {
        "comparison_count": 1,
        "deterministic": True,
        "recall_at_k": 1.0,
        "strict_language_filtering": True,
    }
    assert evidence["per_language"] == {
        "en": {
            "comparison_count": 2,
            "deterministic": True,
            "global_recall_at_k": 1.0,
            "language_shard_recall_at_k": 1.0,
            "seed_count": 1,
            "strict_language_filtering": True,
        }
    }


@pytest.mark.full_corpus
@pytest.mark.ann
def test_numberbatch_ann_recall_and_language_shards_offline() -> None:
    try:
        dataset = load_acceptance_dataset()
    except AcceptanceDatasetUnavailable as exc:
        pytest.skip(str(exc))

    # A schema-2 release ships one index per semantic language and no combined
    # one, so the language shards are the whole of what queries run against.
    components = getattr(dataset, "is_components", False)
    with deny_network():
        report = (
            validate_ann_acceptance_packs(dataset)
            if components
            else validate_ann_acceptance(dataset)
        )
    print(report.to_json(), flush=True)

    expected_scopes = {"language_shard"} if components else {"global", "language_shard"}
    per_language = 10 if components else 20

    assert report.languages == DEFAULT_LANGUAGES
    assert len(report.results) == 100 * len(expected_scopes)
    seed_scopes: dict[int, set[str]] = {}
    for result in report.results:
        seed_scopes.setdefault(result.seed_semantic_id, set()).add(result.index_scope)
    assert len(seed_scopes) == 100
    assert all(scopes == expected_scopes for scopes in seed_scopes.values())
    assert all(
        sum(result.language == language for result in report.results) == per_language
        for language in DEFAULT_LANGUAGES
    )
    assert sum(result.index_scope == "language_shard" for result in report.results) == 100
    if not components:
        assert sum(result.index_scope == "global" for result in report.results) == 100
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
    evidence = report.to_evidence()
    assert evidence["comparison_count"] == 100 * len(expected_scopes)
    assert evidence["unique_seed_count"] == 100
    if not components:
        assert evidence["global"]["comparison_count"] == 100
    assert evidence["language_shards"]["comparison_count"] == 100
    assert all(
        language_report["seed_count"] == 10
        and language_report["comparison_count"] == per_language
        and language_report["deterministic"] is True
        and language_report["strict_language_filtering"] is True
        for language_report in evidence["per_language"].values()
    )
    if not components:
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
