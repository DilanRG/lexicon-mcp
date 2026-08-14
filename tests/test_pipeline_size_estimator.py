from __future__ import annotations

import pytest

from lexicon_mcp.pipeline.size_estimator import (
    GIB,
    INSTALLED_LIMIT,
    PEAK_BUILD_LIMIT,
    FullCorpusShape,
    SizeRates,
    assert_size_targets,
    estimate_installed_bytes,
    estimate_peak_build_bytes,
)


def test_pinned_full_corpus_projection_stays_under_resource_gates() -> None:
    projection = assert_size_targets()
    assert 20 * GIB < projection["total"] <= INSTALLED_LIMIT
    assert projection["peak_build"] <= PEAK_BUILD_LIMIT
    assert projection["semantic/vectors/global.f16"] == 9_161_912 * 300 * 2


def test_projection_counts_each_vector_once_per_global_and_language_index() -> None:
    shape = FullCorpusShape(numberbatch_terms=10, numberbatch_dimensions=4)
    rates = SizeRates(
        wiktextract_entry=0,
        wiktextract_sense=0,
        wiktextract_example=0,
        wiktextract_link=0,
        wiktextract_pronunciation=0,
        conceptnet_assertion=0,
        oewn_row=0,
        cmudict_entry=0,
        semantic_mapping=2,
        hnsw_i8=3,
    )
    result = estimate_installed_bytes(shape, rates)
    assert result["semantic/vectors/global.f16"] == 80
    assert result["semantic/mapping.sqlite3"] == 20
    assert result["semantic/indexes/global.usearch"] == 30
    assert result["semantic/indexes/languages"] == 30


def test_peak_estimator_accounts_for_retained_sources_and_staging() -> None:
    assert estimate_peak_build_bytes(
        10,
        compressed_sources=20,
        packaging_workspace=30,
        sqlite_wal_and_temp=40,
    ) == 100


def test_projection_regression_catches_a_reintroduced_wide_relation_schema() -> None:
    wide = SizeRates(conceptnet_assertion=750.0)
    projection = estimate_installed_bytes(rates=wide)
    assert projection["total"] > INSTALLED_LIMIT
    with pytest.raises(RuntimeError, match="projected installed size"):
        assert_size_targets(rates=wide)
