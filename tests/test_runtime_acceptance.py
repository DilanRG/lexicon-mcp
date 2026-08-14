from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import psutil  # type: ignore[import-untyped]
import pytest

from lexicon_mcp.pipeline import BuildInputs, build_full_corpus
from lexicon_mcp.runtime import acceptance
from lexicon_mcp.runtime.acceptance import (
    PerformanceReport,
    percentile,
    process_mapped_artifact_rss_bytes,
    run_isolated_performance,
)
from lexicon_mcp.runtime.locator import ActiveDataset


def _fixture_dataset(tmp_path: Path) -> ActiveDataset:
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
    return ActiveDataset(output, "fixture-v1", output, {"profile": "full"})


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([5.0, 1.0, 4.0, 2.0, 3.0], 0.95) == 5.0
    assert percentile([5.0, 1.0, 4.0, 2.0, 3.0], 0.50) == 3.0


def test_mapped_artifact_rss_counts_only_resident_semantic_files(tmp_path: Path) -> None:
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    vector = semantic / "global.f16"
    vector.write_bytes(b"x" * 8192)
    unrelated = tmp_path / "unrelated.usearch"
    unrelated.write_bytes(b"y" * 8192)

    with vector.open("rb") as vector_stream, unrelated.open("rb") as unrelated_stream:
        import mmap

        with (
            mmap.mmap(vector_stream.fileno(), 0, access=mmap.ACCESS_READ) as vector_map,
            mmap.mmap(unrelated_stream.fileno(), 0, access=mmap.ACCESS_READ) as unrelated_map,
        ):
            assert vector_map[0] == ord("x")
            assert unrelated_map[0] == ord("y")
            assert process_mapped_artifact_rss_bytes(
                psutil.Process().pid, semantic
            ) > 0


def test_semantic_child_sampling_tracks_independent_peaks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = SimpleNamespace(
        children=lambda recursive: (
            SimpleNamespace(pid=101),
            SimpleNamespace(pid=202),
        )
    )
    monkeypatch.setattr(
        acceptance, "process_private_bytes", lambda pid: {101: 110, 202: 120}[pid]
    )
    monkeypatch.setattr(
        acceptance,
        "process_working_set_bytes",
        lambda pid: {101: 1_450, 202: 1_400}[pid],
    )
    monkeypatch.setattr(
        acceptance,
        "process_mapped_artifact_rss_bytes",
        lambda pid, directory: {101: 3_830, 202: 3_900}[pid],
    )

    private_peak = [115]
    working_set_peak = [1_000]
    mapped_peak = [4_000]
    acceptance._sample_semantic_children(
        parent,
        private_peak,
        working_set_peak,
        mapped_peak,
        tmp_path,
    )

    assert private_peak == [120]
    assert working_set_peak == [1_450]
    assert mapped_peak == [4_000]


def test_performance_report_shape_names_working_set_separately() -> None:
    report = PerformanceReport(
        lexical_p95_ms=1.0,
        semantic_cold_ms=2.0,
        semantic_warm_p95_ms=3.0,
        idle_private_bytes=4,
        semantic_worker_peak_private_bytes=5,
        semantic_worker_peak_working_set_bytes=6,
        semantic_worker_peak_mapped_artifact_rss_bytes=7,
        lexical_samples=8,
        semantic_warm_samples=9,
        semantic_seed="cat",
        semantic_language="en",
    )

    assert asdict(report) == {
        "lexical_p95_ms": 1.0,
        "semantic_cold_ms": 2.0,
        "semantic_warm_p95_ms": 3.0,
        "idle_private_bytes": 4,
        "semantic_worker_peak_private_bytes": 5,
        "semantic_worker_peak_working_set_bytes": 6,
        "semantic_worker_peak_mapped_artifact_rss_bytes": 7,
        "lexical_samples": 8,
        "semantic_warm_samples": 9,
        "semantic_seed": "cat",
        "semantic_language": "en",
    }


def test_isolated_performance_helper_runs_on_pipeline_fixture(tmp_path: Path) -> None:
    report = run_isolated_performance(
        _fixture_dataset(tmp_path),
        lexical_iterations=5,
        semantic_iterations=2,
        timeout_seconds=60,
    )
    assert report.lexical_samples == 5
    assert report.semantic_warm_samples == 2
    assert report.semantic_seed == "cat"
    assert report.semantic_language == "en"
    assert report.idle_private_bytes > 0
    assert report.semantic_worker_peak_private_bytes > 0
    assert report.semantic_worker_peak_working_set_bytes > 0
    assert report.semantic_worker_peak_mapped_artifact_rss_bytes > 0
