from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from lexicon_mcp.live_acceptance import (
    EXPECTED_LEXICON_TOOLS,
    AcceptanceConfig,
    AcceptanceFailure,
    CommandObservation,
    EvidenceWriter,
    FileRecord,
    FixtureHost,
    InventorySnapshot,
    LiveAcceptanceRunner,
    ProcessRecord,
    WindowsHost,
    _active_models,
    _decode_notify_buffer,
    _WindowsChangeMonitor,
    capture_mcpo_tree,
    inventory_diff,
    main,
    validate_lexicon_openapi,
)

FIXTURE = Path(__file__).parent / "fixtures" / "live_acceptance" / "happy.json"


def _fixture_value() -> dict[str, Any]:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _run_fixture(
    tmp_path: Path, value: dict[str, Any] | None = None
) -> tuple[dict[str, Any], FixtureHost]:
    host = FixtureHost(value or _fixture_value())
    evidence = EvidenceWriter(tmp_path / "events.jsonl", run_id="fixture")
    try:
        report = LiveAcceptanceRunner(host, AcceptanceConfig(), evidence).run()
    finally:
        evidence.close()
    return report, host


def test_fixture_replay_drives_exactly_ten_stop_start_pairs(tmp_path: Path) -> None:
    report, host = _run_fixture(tmp_path)

    assert len(report["cycles"]) == 10
    assert host.script_calls == tuple(["stop.ps1", "start.ps1"] * 10)
    assert report["live_stack_ok"] is False
    assert report["restart_cycles"] == 0
    assert report["simulated_restart_cycles"] == 10
    assert report["completed_restart_cycles"] == 10
    assert report["active_models"] == []
    assert all(cycle["old_tree_exit_verified"] for cycle in report["cycles"])
    assert all(cycle["exclusive_artifact_opens"] == 1 for cycle in report["cycles"])
    assert [cycle["six_tool_acceptance_executed"] for cycle in report["cycles"]] == [
        True,
        *([False] * 9),
    ]

    suite = report["six_tool_acceptance"]
    assert suite["transport"] == "fixture-replay"
    assert suite["live_evidence"] is False
    assert set(suite["tool_names"]) == EXPECTED_LEXICON_TOOLS
    assert set(suite["tools"]) == EXPECTED_LEXICON_TOOLS
    assert suite["cross_tool_sense_flow"] == {
        "word": "bank",
        "lookup_language": "en",
        "target_language": "de",
        "river": {
            "selected_gloss": "edge of river or lake",
            "selected_sense_id": "wikt:labeled:bank-river",
            "translate_request_sense_id": "wikt:labeled:bank-river",
            "required_translation": "Ufer",
            "source_sense_preserved": True,
        },
        "financial": {
            "selected_gloss": "institution",
            "selected_sense_id": "wikt:labeled:bank-finance",
            "translate_request_sense_id": "wikt:labeled:bank-finance",
            "required_translation": "Bank",
            "source_sense_preserved": True,
        },
    }
    lookup_evidence = suite["tools"]["dictionary_lookup"]
    assert lookup_evidence["translations_limit"] == 3
    assert lookup_evidence["embedded_translation_count"] == 3
    assert lookup_evidence["translation_truncation_sense_ids"] == [
        "wikt:labeled:bank-finance",
        "wikt:labeled:bank-river",
    ]
    assert suite["tools"]["dictionary_translate"]["call_count"] == 2
    assert report["checks"]["all_six_lexicon_tools_validated"] is True
    assert report["checks"]["all_six_lexicon_tools_via_live_mcpo"] is False
    assert report["live_stack_scope"]["open_webui_ordinary_prompts"] is False

    successful_posts = Counter(
        str(call["url"]).rsplit("/", 1)[-1]
        for call in host.request_calls
        if call["method"] == "POST"
    )
    assert successful_posts == Counter(
        {
            "dictionary_lookup": 11,
            "dictionary_synonyms": 1,
            "dictionary_translate": 2,
            "dictionary_relations": 1,
            "dictionary_semantic_neighbors": 1,
            "dictionary_wordplay": 1,
            "calculate": 10,
        }
    )
    translate_calls = [
        call for call in host.request_calls if str(call["url"]).endswith("/dictionary_translate")
    ]
    assert {call["payload"]["sense_id"] for call in translate_calls} == {
        "wikt:labeled:bank-river",
        "wikt:labeled:bank-finance",
    }
    assert all(call["payload"]["max_senses"] == 100 for call in translate_calls)
    bank_lookup_call = next(
        call
        for call in host.request_calls
        if str(call["url"]).endswith("/dictionary_lookup") and call["payload"]["word"] == "bank"
    )
    assert bank_lookup_call["payload"]["translations_limit"] == 3
    synonym_call = next(
        call for call in host.request_calls if str(call["url"]).endswith("/dictionary_synonyms")
    )
    assert synonym_call["payload"] == {
        "word": "important",
        "language": "en",
        "limit": 20,
        "max_senses": 20,
        "unsensed_limit": 5,
    }
    relation_call = next(
        call for call in host.request_calls if str(call["url"]).endswith("/dictionary_relations")
    )
    assert relation_call["payload"]["max_depth"] == 1
    assert relation_call["payload"]["transitive_limit"] == 0


def test_dry_run_cli_writes_non_publishable_jsonl_and_final_report(tmp_path: Path) -> None:
    result = main(
        [
            "--dry-run-fixture",
            str(FIXTURE),
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "dry-test",
        ]
    )

    assert result == 0
    report_path = tmp_path / "live-acceptance-dry-test.json"
    events_path = tmp_path / "live-acceptance-dry-test.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert report["dry_run"] is True
    assert report["live_stack_ok"] is False
    assert report["restart_cycles"] == 0
    assert report["simulated_restart_cycles"] == 10
    assert report["events_sha256"]
    assert report["six_tool_acceptance"]["live_evidence"] is False
    assert any(event["event"] == "six_tool_acceptance_completed" for event in events)
    assert events[-1]["event"] == "acceptance_completed"


def test_fixture_replay_rejects_translation_that_crosses_selected_sense(
    tmp_path: Path,
) -> None:
    value = _fixture_value()
    value["translation_crosses_sense"] = True

    with pytest.raises(AcceptanceFailure, match="crossed or discarded"):
        _run_fixture(tmp_path, value)


def test_fixture_replay_rejects_cross_sense_embedded_lookup_translation(
    tmp_path: Path,
) -> None:
    value = _fixture_value()
    value["lookup_translation_crosses_sense"] = True

    with pytest.raises(AcceptanceFailure, match="attached a translation to the wrong sense"):
        _run_fixture(tmp_path, value)


def test_fixture_replay_requires_lookup_translation_truncation_metadata(
    tmp_path: Path,
) -> None:
    value = _fixture_value()
    value["lookup_missing_translation_truncation"] = True

    with pytest.raises(AcceptanceFailure, match="expected translation truncation"):
        _run_fixture(tmp_path, value)


@pytest.mark.parametrize(
    ("flag", "term"),
    (
        ("missing_river_translation", "Ufer"),
        ("missing_financial_translation", "Bank"),
    ),
)
def test_fixture_replay_requires_both_sense_specific_translation_anchors(
    tmp_path: Path, flag: str, term: str
) -> None:
    value = _fixture_value()
    value[flag] = True

    with pytest.raises(AcceptanceFailure, match=term):
        _run_fixture(tmp_path, value)


def test_fixture_replay_rejects_wrong_language_semantic_results(tmp_path: Path) -> None:
    value = _fixture_value()
    value["semantic_wrong_language"] = True

    with pytest.raises(AcceptanceFailure, match="language or sense scope"):
        _run_fixture(tmp_path, value)


def test_fixture_replay_rejects_wordplay_query_echo(tmp_path: Path) -> None:
    value = _fixture_value()
    value["wordplay_echo_query"] = True

    with pytest.raises(AcceptanceFailure, match="echoed its query"):
        _run_fixture(tmp_path, value)


def test_exact_lexicon_openapi_rejects_any_seventh_operation() -> None:
    paths: dict[str, object] = {f"/{name}": {"post": {}} for name in EXPECTED_LEXICON_TOOLS}
    assert set(validate_lexicon_openapi({"paths": paths})) == EXPECTED_LEXICON_TOOLS

    paths["/repair"] = {"post": {}}
    with pytest.raises(AcceptanceFailure, match="exact six-tool contract"):
        validate_lexicon_openapi({"paths": paths})


def test_active_models_supports_router_status_objects_without_weakening_empty_gate() -> None:
    body = {
        "object": "list",
        "data": [
            {"id": "idle", "status": {"value": "unloaded"}},
            {"id": "resident", "status": {"value": "loaded"}},
        ],
    }

    assert _active_models(body) == ["resident"]
    assert _active_models({"active_models": []}) == []
    with pytest.raises(AcceptanceFailure, match=r"status\.value"):
        _active_models({"data": [{"id": "ambiguous"}]})


def test_capture_mcpo_tree_tracks_transitive_children_and_markers() -> None:
    class ProcessHost(FixtureHost):
        def process_table(self) -> tuple[ProcessRecord, ...]:
            return (
                ProcessRecord(10, 1, "root", "uvx.exe", None, "uvx mcpo"),
                ProcessRecord(11, 10, "shim", "cmd.exe", None, "cmd"),
                ProcessRecord(12, 11, "lex", "python.exe", None, "lexicon-mcp"),
                ProcessRecord(13, 10, "calc", "python.exe", None, "calculator.py"),
            )

        def read_pid(self, path: Path) -> int:
            del path
            return 10

    tree = capture_mcpo_tree(ProcessHost(_fixture_value()), AcceptanceConfig())

    assert tree.pids == {10, 11, 12, 13}
    assert tree.marker_pids == {"lexicon-mcp": (12,), "calculator.py": (13,)}


def test_inventory_diff_reports_added_removed_and_rewritten() -> None:
    before_records = (
        FileRecord("same", 1, 1, 1, 1, None),
        FileRecord("removed", 1, 1, 1, 2, None),
        FileRecord("changed", 1, 1, 1, 3, None),
    )
    after_records = (
        FileRecord("same", 1, 1, 1, 1, None),
        FileRecord("added", 1, 1, 1, 4, None),
        FileRecord("changed", 2, 2, 1, 3, None),
    )
    before = InventorySnapshot(Path("before"), 3, 3, "a", None, before_records)
    after = InventorySnapshot(Path("before"), 3, 4, "b", None, after_records)

    assert inventory_diff(before, after) == {
        "added": ["added"],
        "removed": ["removed"],
        "changed": ["changed"],
    }


def test_fixture_replay_fails_on_inventory_drift(tmp_path: Path) -> None:
    value = _fixture_value()
    value["inventory_drift_cycle"] = 1

    with pytest.raises(AcceptanceFailure, match="inventory changed"):
        _run_fixture(tmp_path, value)


def test_failure_after_successful_stop_restores_stack_and_preserves_interrupt(
    tmp_path: Path,
) -> None:
    class InterruptedHost(FixtureHost):
        def __init__(self, value: dict[str, Any]) -> None:
            super().__init__(value)
            self.interrupted = False

        def exclusive_open(self, paths: Sequence[Path]) -> int:
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt("simulated operator interrupt while stopped")
            return super().exclusive_open(paths)

    host = InterruptedHost(_fixture_value())
    evidence_path = tmp_path / "events.jsonl"
    evidence = EvidenceWriter(evidence_path, run_id="restore-success")
    runner = LiveAcceptanceRunner(host, AcceptanceConfig(), evidence)
    try:
        with pytest.raises(KeyboardInterrupt, match="simulated operator interrupt"):
            runner.run()
    finally:
        evidence.close()

    assert host.script_calls == ("stop.ps1", "stop.ps1", "start.ps1")
    assert runner.failure_recovery is not None
    assert runner.failure_recovery["status"] == "restored"
    assert runner.failure_recovery["method"] == "stop_start"
    capture_mcpo_tree(host, AcceptanceConfig())
    events = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    recovery = [item for item in events if item["event"] == "failure_stack_recovery"]
    assert len(recovery) == 1
    assert recovery[0]["status"] == "ok"


def test_failed_stack_restoration_does_not_replace_original_failure(
    tmp_path: Path,
) -> None:
    class OriginalFailure(RuntimeError):
        pass

    class FailedRecoveryHost(FixtureHost):
        def __init__(self, value: dict[str, Any]) -> None:
            super().__init__(value)
            self.failed = False

        def exclusive_open(self, paths: Sequence[Path]) -> int:
            del paths
            if not self.failed:
                self.failed = True
                raise OriginalFailure("primary acceptance failure")
            raise AssertionError("exclusive_open unexpectedly retried")

        def run_script(
            self, path: Path, *, timeout_seconds: float
        ) -> CommandObservation:
            if "start" not in path.name.casefold():
                return super().run_script(path, timeout_seconds=timeout_seconds)
            self._script_calls.append(path.name.casefold())
            empty = "e3b0c44298fc1c149afbf4c8996fb924"
            return CommandObservation(
                str(path),
                1,
                0.0,
                empty + "27ae41e4649b934ca495991b7852b855",
                empty + "27ae41e4649b934ca495991b7852b855",
                "",
                "simulated recovery start failure",
            )

    host = FailedRecoveryHost(_fixture_value())
    evidence_path = tmp_path / "events.jsonl"
    evidence = EvidenceWriter(evidence_path, run_id="restore-failure")
    config = AcceptanceConfig(startup_timeout_seconds=0.01, poll_interval_seconds=0.01)
    runner = LiveAcceptanceRunner(host, config, evidence)
    try:
        with pytest.raises(OriginalFailure, match="primary acceptance failure") as caught:
            runner.run()
    finally:
        evidence.close()

    assert host.script_calls == ("stop.ps1", "stop.ps1", "start.ps1")
    assert runner.failure_recovery is not None
    assert runner.failure_recovery["status"] == "failed"
    assert any("restoration also failed" in note for note in caught.value.__notes__)
    events = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    recovery = [item for item in events if item["event"] == "failure_stack_recovery"]
    assert len(recovery) == 1
    assert recovery[0]["status"] == "failed"


def test_fixture_replay_fails_when_old_child_survives_stop(tmp_path: Path) -> None:
    value = _fixture_value()
    value["orphan_cycle"] = 0

    with pytest.raises(AcceptanceFailure, match="old MCPO process tree still alive"):
        _run_fixture(tmp_path, value)


def test_fixture_replay_rejects_preexisting_orphan_child(tmp_path: Path) -> None:
    value = _fixture_value()
    value["extra_orphan"] = True

    with pytest.raises(AcceptanceFailure, match="orphan MCP child"):
        _run_fixture(tmp_path, value)


def test_final_model_gate_runs_after_full_content_verification(tmp_path: Path) -> None:
    value = _fixture_value()
    value["late_active_model"] = True

    with pytest.raises(AcceptanceFailure, match="active models"):
        _run_fixture(tmp_path, value)


def test_file_notify_decoder_handles_multiple_records() -> None:
    first_name = "one.tmp".encode("utf-16-le")
    first_size = 12 + len(first_name)
    aligned_size = (first_size + 3) & ~3
    first = (
        aligned_size.to_bytes(4, "little")
        + (1).to_bytes(4, "little")
        + len(first_name).to_bytes(4, "little")
        + first_name
        + bytes(aligned_size - first_size)
    )
    second_name = "two.tmp".encode("utf-16-le")
    second = (
        (0).to_bytes(4, "little")
        + (3).to_bytes(4, "little")
        + len(second_name).to_bytes(4, "little")
        + second_name
    )
    value = first + second

    assert _decode_notify_buffer(value, len(value)) == (
        (1, "one.tmp"),
        (3, "two.tmp"),
    )


@pytest.mark.skipif(os.name != "nt", reason="requires Windows directory notifications")
def test_windows_change_monitor_detects_a_transient_file(tmp_path: Path) -> None:
    monitor = _WindowsChangeMonitor((tmp_path,))
    try:
        transient = tmp_path / "runtime.partial"
        transient.write_bytes(b"transient")
        transient.unlink()
        deadline = time.monotonic() + 5
        while not monitor.changes() and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        monitor.close()

    paths = {item.relative_path for item in monitor.changes()}
    assert "runtime.partial" in paths
    assert not monitor.errors()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows sharing semantics")
def test_windows_exclusive_open_detects_a_leaked_reader(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    host = WindowsHost()

    assert host.exclusive_open((artifact,)) == 1
    with (
        artifact.open("rb"),
        pytest.raises(AcceptanceFailure, match="cannot be opened exclusively"),
    ):
        host.exclusive_open((artifact,))
