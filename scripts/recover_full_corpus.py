"""Recover a verified post-global Lexicon MCP build without rerunning lexical stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from lexicon_mcp.pipeline import (
    BuildInputs,
    recover_full_corpus_from_semantic_partial,
)


def _git_output(repository: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", b"")
        detail = bytes(stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git provenance check failed: {detail or exc}") from exc
    return result.stdout


def _pipeline_identity_at_commit(repository: Path, commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("original build commit must be a lowercase 40-hex Git commit")
    _git_output(repository, "cat-file", "-e", f"{commit}^{{commit}}")
    listed = _git_output(
        repository,
        "ls-tree",
        "-r",
        "--name-only",
        commit,
        "--",
        "src/lexicon_mcp/pipeline",
        "src/lexicon_mcp/usearch_compat.py",
    ).decode("utf-8")
    directory = PurePosixPath("src/lexicon_mcp/pipeline")
    listed_paths = [PurePosixPath(value) for value in listed.splitlines()]
    paths = [
        path for path in listed_paths if path.parent == directory and path.suffix == ".py"
    ]
    if not paths:
        raise RuntimeError("original build commit has no pipeline Python sources")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(_git_output(repository, "show", f"{commit}:{path.as_posix()}"))
        digest.update(b"\0")
    compatibility = PurePosixPath("src/lexicon_mcp/usearch_compat.py")
    if compatibility in listed_paths:
        digest.update(b"../usearch_compat.py\0")
        digest.update(
            _git_output(repository, "show", f"{commit}:{compatibility.as_posix()}")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _validated_git_provenance(
    repository: Path, *, original_build_commit: str, recovery_commit: str
) -> str:
    dirty = _git_output(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    if dirty.strip():
        raise RuntimeError("recovery requires a clean Git working tree")
    head = _git_output(repository, "rev-parse", "--verify", "HEAD").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise RuntimeError("current Git HEAD is not a full 40-hex commit")
    if recovery_commit != head:
        raise RuntimeError(
            f"--recovery-commit must equal current clean Git HEAD ({head})"
        )
    return _pipeline_identity_at_commit(repository, original_build_commit)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--oewn", type=Path, required=True, help="OEWN 2025 WN-LMF XML[.gz]")
    value.add_argument(
        "--wiktextract",
        type=Path,
        action="append",
        required=True,
        help="Kaikki Wiktextract JSONL[.gz/.zst]; repeat for multiple shards",
    )
    value.add_argument("--conceptnet", type=Path, required=True, help="ConceptNet 5.7 TSV[.gz]")
    value.add_argument(
        "--numberbatch", type=Path, required=True, help="Numberbatch 19.08 text[.gz]"
    )
    value.add_argument("--cmudict", type=Path, required=True, help="Pinned cmudict.dict")
    value.add_argument(
        "--source-lock",
        type=Path,
        required=True,
        help="Pinned schema-v1 sources.lock.json; every source is hash-verified",
    )
    value.add_argument(
        "--notices-dir",
        type=Path,
        required=True,
        help="Directory containing DATA_LICENSES.md and exact licenses/ texts",
    )
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--build-state", type=Path, required=True)
    value.add_argument("--dataset-version", default="data-v1.0.0")
    value.add_argument("--retrieved-at", help="RFC 3339 timestamp recorded for every source")
    value.add_argument(
        "--original-build-commit",
        required=True,
        help="40-hex commit that created the lexical/global partial artifacts",
    )
    value.add_argument(
        "--recovery-commit",
        required=True,
        help="40-hex commit containing the reviewed recovery implementation",
    )
    value.add_argument(
        "--expected-lexicon-sha256",
        required=True,
        help="Immutable SHA-256 anchor for the preserved lexicon.sqlite3",
    )
    value.add_argument(
        "--expected-global-index-sha256",
        required=True,
        help="Immutable SHA-256 anchor for the preserved global.usearch",
    )
    value.add_argument(
        "--expected-global-vectors-sha256",
        required=True,
        help="Immutable SHA-256 anchor for the preserved global.f16",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    original_pipeline_identity = _validated_git_provenance(
        repository,
        original_build_commit=args.original_build_commit,
        recovery_commit=args.recovery_commit,
    )
    result = recover_full_corpus_from_semantic_partial(
        BuildInputs(
            oewn=args.oewn,
            wiktextract=tuple(args.wiktextract),
            conceptnet=args.conceptnet,
            numberbatch=args.numberbatch,
            cmudict=args.cmudict,
            source_lock=args.source_lock,
            notices_dir=args.notices_dir,
        ),
        args.output,
        args.build_state,
        dataset_version=args.dataset_version,
        retrieved_at=args.retrieved_at,
        enforce_corpus_floors=True,
        original_build_commit=args.original_build_commit,
        recovery_commit=args.recovery_commit,
        expected_lexicon_sha256=args.expected_lexicon_sha256,
        expected_global_index_sha256=args.expected_global_index_sha256,
        expected_global_vectors_sha256=args.expected_global_vectors_sha256,
        expected_original_pipeline_identity=original_pipeline_identity,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
