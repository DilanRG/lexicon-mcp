from __future__ import annotations

import pytest

from lexicon_mcp.runtime.acceptance import (
    MIB,
    AcceptanceDatasetUnavailable,
    load_acceptance_dataset,
    run_isolated_performance,
)


@pytest.mark.full_corpus
@pytest.mark.performance
def test_full_corpus_latency_and_private_memory_gates() -> None:
    try:
        dataset = load_acceptance_dataset()
    except AcceptanceDatasetUnavailable as exc:
        pytest.skip(str(exc))
    report = run_isolated_performance(dataset)
    assert report.lexical_p95_ms <= 150, report
    assert report.semantic_cold_ms <= 2_000, report
    assert report.semantic_warm_p95_ms <= 500, report
    assert report.idle_private_bytes <= 512 * MIB, report
    assert report.semantic_worker_peak_private_bytes > 0, report
    assert report.semantic_worker_peak_private_bytes <= 1_024 * MIB, report
    # Mapped index/vector/SQLite residency is reported separately and is not
    # charged against the semantic private-memory gate above.
    assert report.semantic_worker_peak_mapped_artifact_rss_bytes > 0, report
