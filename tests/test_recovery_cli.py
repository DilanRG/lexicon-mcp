from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import scripts.recover_full_corpus as recovery_cli


def test_git_provenance_requires_clean_matching_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = "b" * 40

    def dirty_git(_repository: Path, *arguments: str) -> bytes:
        assert arguments[:2] == ("status", "--porcelain")
        return b" M src/lexicon_mcp/pipeline/orchestrator.py\n"

    monkeypatch.setattr(recovery_cli, "_git_output", dirty_git)
    with pytest.raises(RuntimeError, match="clean Git working tree"):
        recovery_cli._validated_git_provenance(
            tmp_path,
            original_build_commit="a" * 40,
            recovery_commit=head,
        )

    def clean_git(_repository: Path, *arguments: str) -> bytes:
        if arguments[0] == "status":
            return b""
        if arguments[:2] == ("rev-parse", "--verify"):
            return (head + "\n").encode()
        raise AssertionError(arguments)

    monkeypatch.setattr(recovery_cli, "_git_output", clean_git)
    with pytest.raises(RuntimeError, match="must equal current clean Git HEAD"):
        recovery_cli._validated_git_provenance(
            tmp_path,
            original_build_commit="a" * 40,
            recovery_commit="c" * 40,
        )


def test_original_pipeline_identity_is_derived_from_commit_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = "a" * 40
    head = "b" * 40
    blobs = {
        "src/lexicon_mcp/pipeline/a.py": b"first\n",
        "src/lexicon_mcp/pipeline/z.py": b"last\n",
    }

    def git_output(_repository: Path, *arguments: str) -> bytes:
        if arguments[0] == "status":
            return b""
        if arguments[:2] == ("rev-parse", "--verify"):
            return (head + "\n").encode()
        if arguments[:2] == ("cat-file", "-e"):
            return b""
        if arguments[0] == "ls-tree":
            return (
                b"src/lexicon_mcp/pipeline/z.py\n"
                b"src/lexicon_mcp/pipeline/nested/ignored.py\n"
                b"src/lexicon_mcp/pipeline/a.py\n"
            )
        if arguments[0] == "show":
            return blobs[arguments[1].split(":", 1)[1]]
        raise AssertionError(arguments)

    expected = hashlib.sha256()
    for name in ("a.py", "z.py"):
        expected.update(name.encode())
        expected.update(b"\0")
        expected.update(blobs[f"src/lexicon_mcp/pipeline/{name}"])
        expected.update(b"\0")

    monkeypatch.setattr(recovery_cli, "_git_output", git_output)
    assert recovery_cli._validated_git_provenance(
        tmp_path,
        original_build_commit=original,
        recovery_commit=head,
    ) == expected.hexdigest()
