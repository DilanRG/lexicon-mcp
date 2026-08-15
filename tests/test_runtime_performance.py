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
    print(report.to_json(), flush=True)
    assert report.lexical_p95_ms <= 150, report
    # Wordplay gates: warm p95 <= 100 ms and first-call <= 500 ms per kind
    # at limits 1, 20, and 100 on the packaged artifact.
    assert set(report.wordplay_warm_p95_ms) == {
        "anagram",
        "palindrome",
        "spoonerism",
        "pun",
    }
    for kind, value in report.wordplay_warm_p95_ms.items():
        assert value <= 100, (kind, report)
    for kind, value in report.wordplay_cold_ms.items():
        assert value <= 500, (kind, report)
    assert report.semantic_cold_ms <= 2_000, report
    assert report.semantic_warm_p95_ms <= 500, report
    assert report.idle_private_bytes <= 512 * MIB, report
    assert report.semantic_worker_peak_private_bytes > 0, report
    assert report.semantic_worker_peak_private_bytes <= 1_024 * MIB, report
    # Total worker working set and mapped index/vector/SQLite residency are
    # diagnostics only and do not change the private-memory release gate.
    assert report.semantic_worker_peak_working_set_bytes > 0, report
    assert report.semantic_worker_peak_mapped_artifact_rss_bytes > 0, report
