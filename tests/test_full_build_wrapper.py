from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

POWERSHELL = shutil.which("powershell.exe") or shutil.which("pwsh")


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is not installed")
def test_monitoring_arithmetic_uses_int64_without_starting_build(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    missing_project = tmp_path / "project-must-not-be-inspected"
    unused_state = tmp_path / "build-must-not-start"
    command = [
        str(POWERSHELL),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(project / "scripts" / "run_full_build.ps1"),
        "-Project",
        str(missing_project),
        "-BuildState",
        str(unused_state),
        "-MonitoringArithmeticSelfTest",
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "build_started": False,
        "private_bytes": 2_147_487_744,
        "working_set_bytes": 4_294_975_488,
        "peak_private_bytes": 2_147_487_744,
        "peak_working_set_bytes": 4_294_975_488,
        "minimum_free_bytes": 8_589_934_592,
        "openblas_num_threads": "1",
    }
    assert not unused_state.exists()


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is not installed")
def test_recovery_requires_explicit_commit_provenance_before_mutation(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[1]
    unused_state = tmp_path / "recovery-must-not-start"
    command = [
        str(POWERSHELL),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(project / "scripts" / "run_full_build.ps1"),
        "-Project",
        str(project),
        "-BuildState",
        str(unused_state),
        "-RecoverPartial",
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "OriginalBuildCommit" in completed.stderr
    assert not unused_state.exists()


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is not installed")
def test_normal_wrapper_refuses_post_global_partial_before_mutation(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / "data-v1.0.0"
    semantic_partial = output.with_name(output.name + ".partial") / "semantic.partial"
    semantic_partial.mkdir(parents=True)
    sentinel = semantic_partial / "global.usearch"
    sentinel.write_bytes(b"preserve-me")
    unused_state = tmp_path / "normal-build-must-not-start"
    command = [
        str(POWERSHELL),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(project / "scripts" / "run_full_build.ps1"),
        "-Project",
        str(project),
        "-BuildState",
        str(unused_state),
        "-Output",
        str(output),
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "post-global semantic build is already staged" in completed.stderr
    assert sentinel.read_bytes() == b"preserve-me"
    assert not unused_state.exists()
