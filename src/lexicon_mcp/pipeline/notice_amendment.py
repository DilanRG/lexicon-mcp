"""Controlled post-promotion amendment of data-bundle notices.

This operation is intentionally narrower than a corpus rebuild or repair.  It
can replace only the canonical data licensing notices after validating the
promoted dataset, its recovery provenance, and immutable large-artifact
anchors.  Corpus artifacts are never opened writable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from lexicon_mcp.data.locking import InstallationLock
from lexicon_mcp.usearch_compat import index_count

from .common import file_sha256, write_json_atomic

_NOTICE_PATHS = (
    "DATA_LICENSES.md",
    "licenses/CC-BY-4.0.txt",
    "licenses/CC-BY-SA-4.0.txt",
    "licenses/CMUDICT.txt",
    "licenses/GFDL-1.3.txt",
    "licenses/OEWN-LICENSE.md",
    "licenses/PRINCETON-WORDNET.txt",
)
_ANCHOR_PATHS = {
    "lexicon.sqlite3": "lexicon.sqlite3",
    "global.usearch": "semantic/indexes/global.usearch",
    "global.f16": "semantic/vectors/global.f16",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TRANSACTION_SCHEMA_VERSION = 1
_TRANSACTION_STATES = frozenset(
    {"prepared", "old_notices_moved", "new_notices_activated", "committed"}
)


def _crash_checkpoint(_name: str) -> None:
    """Test seam for simulating a process loss between durable transitions."""


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


def _validated_clean_head(repository: Path, amendment_commit: str) -> None:
    if not _COMMIT.fullmatch(amendment_commit):
        raise ValueError("amendment_commit must be a lowercase 40-hex Git commit")
    dirty = _git_output(repository, "status", "--porcelain", "--untracked-files=all")
    if dirty.strip():
        raise RuntimeError("notice amendment requires a clean Git working tree")
    head = _git_output(repository, "rev-parse", "--verify", "HEAD").decode().strip()
    if not _COMMIT.fullmatch(head):
        raise RuntimeError("current Git HEAD is not a full 40-hex commit")
    if head != amendment_commit:
        raise RuntimeError(
            f"amendment_commit must equal current clean Git HEAD ({head})"
        )


def _require_sha256(value: str, *, field: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: str, *, field: str) -> str:
    if not _COMMIT.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase 40-hex Git commit")
    return value


def _installed_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _reject_transients(dataset_root: Path) -> None:
    transients: list[str] = []
    symlinks: list[str] = []
    for path in dataset_root.rglob("*"):
        relative = path.relative_to(dataset_root).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
        if path.name.endswith((".partial", "-wal", "-shm")):
            transients.append(relative)
    if symlinks:
        raise RuntimeError(f"promoted dataset contains symbolic links: {symlinks!r}")
    if transients:
        raise RuntimeError(f"promoted dataset contains transient paths: {transients!r}")


def _notice_hashes(root: Path, *, require_exact_tree: bool) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _NOTICE_PATHS:
        path = root / Path(relative)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"required repository notice is missing: {path}")
        result[relative] = file_sha256(path)
    if require_exact_tree:
        allowed = {*_NOTICE_PATHS, "licenses"}
        observed = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
        }
        unexpected = sorted(observed - allowed)
        if unexpected:
            raise RuntimeError(f"dataset notice tree has unexpected paths: {unexpected!r}")
    return result


def _copy_notices(repository: Path, destination: Path) -> dict[str, str]:
    if destination.exists():
        raise FileExistsError(f"notice staging path already exists: {destination}")
    (destination / "licenses").mkdir(parents=True)
    for relative in _NOTICE_PATHS:
        source = repository / Path(relative)
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"required repository notice is missing: {source}")
        target = destination / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomic(target, source.read_bytes())
    return _notice_hashes(destination, require_exact_tree=True)


def _remove_notice_tree(path: Path) -> None:
    """Remove only the known notice files from one internally-created tree."""

    if path.is_symlink():
        raise RuntimeError(f"refusing to remove symbolic-link notice tree: {path}")
    allowed = {*_NOTICE_PATHS, "licenses"}
    unexpected: list[str] = []
    if path.exists():
        for candidate in path.rglob("*"):
            relative = candidate.relative_to(path).as_posix()
            if candidate.is_symlink() or relative not in allowed:
                unexpected.append(relative)
    if unexpected:
        raise RuntimeError(
            f"refusing to remove notice tree with unexpected paths: {sorted(unexpected)!r}"
        )
    for relative in _NOTICE_PATHS:
        candidate = path / Path(relative)
        if candidate.exists() or candidate.is_symlink():
            candidate.unlink()
    licenses = path / "licenses"
    if licenses.exists():
        licenses.rmdir()
    if path.exists():
        path.rmdir()


def _artifact_anchors(
    dataset_root: Path,
    *,
    expected_lexicon_sha256: str,
    expected_global_index_sha256: str,
    expected_global_vectors_sha256: str,
) -> dict[str, str]:
    expected = {
        "lexicon.sqlite3": _require_sha256(
            expected_lexicon_sha256, field="expected_lexicon_sha256"
        ),
        "global.usearch": _require_sha256(
            expected_global_index_sha256, field="expected_global_index_sha256"
        ),
        "global.f16": _require_sha256(
            expected_global_vectors_sha256, field="expected_global_vectors_sha256"
        ),
    }
    observed: dict[str, str] = {}
    for name, relative in _ANCHOR_PATHS.items():
        path = dataset_root / Path(relative)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"immutable amendment anchor is missing: {path}")
        observed[name] = file_sha256(path)
        if observed[name] != expected[name]:
            raise RuntimeError(
                f"notice amendment anchor mismatch for {name}: "
                f"expected {expected[name]}, got {observed[name]}"
            )
    return observed


def _semantic_artifact_hashes(
    dataset_root: Path, *, dataset_version: str
) -> dict[str, str]:
    """Verify and hash the semantic mapping plus every declared language shard."""

    semantic_root = (dataset_root / "semantic").resolve()
    if not semantic_root.is_relative_to(dataset_root) or not semantic_root.is_dir():
        raise RuntimeError("promoted dataset has no safe semantic artifact directory")
    mapping = semantic_root / "mapping.sqlite3"
    if mapping.is_symlink() or not mapping.is_file():
        raise FileNotFoundError(f"semantic mapping is missing: {mapping}")
    connection = sqlite3.connect(
        f"file:{mapping.as_posix()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        quick = connection.execute("PRAGMA quick_check").fetchall()
        if len(quick) != 1 or str(quick[0][0]) != "ok":
            raise RuntimeError(f"semantic mapping quick_check failed: {quick[:3]!r}")
        foreign_key_failure = connection.execute("PRAGMA foreign_key_check").fetchone()
        if foreign_key_failure is not None:
            raise RuntimeError(
                "semantic mapping foreign-key check failed: "
                f"{tuple(foreign_key_failure)!r}"
            )
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM metadata")
        }
        if metadata.get("dataset_version") != dataset_version:
            raise RuntimeError("semantic mapping dataset_version mismatch")
        if metadata.get("schema_version") != "2":
            raise RuntimeError("semantic mapping must use schema version 2")
        try:
            dimensions = int(metadata["dimensions"])
            terms = int(metadata["term_count"])
            connectivity = int(metadata.get("connectivity", "16"))
            expansion_add = int(metadata.get("expansion_add", "256"))
            expansion_search = int(metadata.get("expansion_search", "512"))
            language_count = int(metadata.get("language_count", "-1"))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("semantic mapping metadata is malformed") from exc
        if dimensions < 1 or terms < 1:
            raise RuntimeError("semantic mapping dimensions/term_count are invalid")
        if connectivity != 16 or expansion_add != 256 or expansion_search < 512:
            raise RuntimeError("semantic mapping has an unsupported USearch schema")
        if metadata.get("vector_dtype") != "float16":
            raise RuntimeError("semantic mapping vector dtype must be float16")
        if metadata.get("index_metric") != "cos" or metadata.get("index_dtype") != "i8":
            raise RuntimeError("semantic mapping index schema must be cosine/i8")
        if metadata.get("vector_file") != "vectors/global.f16":
            raise RuntimeError("semantic vector path is not canonical")
        if metadata.get("global_index") != "indexes/global.usearch":
            raise RuntimeError("semantic global-index path is not canonical")
        if metadata.get("language_index_dir") != "indexes/languages":
            raise RuntimeError("semantic language-index directory is not canonical")

        mapped_terms = int(
            connection.execute("SELECT COUNT(*) FROM semantic_terms").fetchone()[0]
        )
        language_rows = connection.execute(
            "SELECT language,index_file,term_count "
            "FROM semantic_languages ORDER BY language"
        ).fetchall()
        if language_count != len(language_rows) or not language_rows:
            raise RuntimeError("semantic language count is not exact")
        recorded_languages = {
            str(row["language"]): int(row["term_count"]) for row in language_rows
        }
        derived_languages = {
            str(row["language"]): int(row["term_count"])
            for row in connection.execute(
                "SELECT term.language,COUNT(*) AS term_count "
                "FROM semantic_terms AS semantic "
                "JOIN lexical_terms AS term ON term.term_id=semantic.term_id "
                "GROUP BY term.language ORDER BY term.language"
            )
        }
        if (
            mapped_terms != terms
            or recorded_languages != derived_languages
            or sum(recorded_languages.values()) != terms
        ):
            raise RuntimeError("semantic language rows do not partition all terms")

        vectors = semantic_root / "vectors" / "global.f16"
        global_index = semantic_root / "indexes" / "global.usearch"
        if vectors.stat().st_size != terms * dimensions * 2:
            raise RuntimeError("semantic vector byte count does not match metadata")
        observed_global_count = index_count(
            global_index,
            dimensions=dimensions,
            metric="cos",
            dtype="i8",
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            expected_count=terms,
        )
        if observed_global_count != terms:
            raise RuntimeError("semantic global index count does not match metadata")

        expected_shards: dict[str, int] = {}
        for row in language_rows:
            language = str(row["language"])
            relative = str(row["index_file"])
            expected_relative = f"indexes/languages/{language.replace('-', '_')}.usearch"
            if relative != expected_relative or relative in expected_shards:
                raise RuntimeError("semantic language index paths are not canonical and unique")
            expected_shards[relative] = int(row["term_count"])
        shard_root = semantic_root / "indexes" / "languages"
        actual_shards = {
            path.relative_to(semantic_root).as_posix()
            for path in shard_root.rglob("*")
            if path.is_file()
        }
        if actual_shards != set(expected_shards):
            raise RuntimeError(
                "semantic language artifact set mismatch: "
                f"missing={sorted(set(expected_shards) - actual_shards)!r}, "
                f"extra={sorted(actual_shards - set(expected_shards))!r}"
            )
        result = {"semantic/mapping.sqlite3": file_sha256(mapping)}
        for relative, expected_count in sorted(expected_shards.items()):
            shard = (semantic_root / relative).resolve()
            if (
                not shard.is_relative_to(semantic_root)
                or shard.is_symlink()
                or not shard.is_file()
            ):
                raise RuntimeError(f"semantic language shard is missing or unsafe: {relative}")
            observed_count = index_count(
                shard,
                dimensions=dimensions,
                metric="cos",
                dtype="i8",
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
                expected_count=expected_count,
            )
            if observed_count != expected_count:
                raise RuntimeError(
                    f"semantic language shard count mismatch for {relative}"
                )
            result[f"semantic/{relative}"] = file_sha256(shard)
        return result
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError(f"semantic artifact verification failed: {exc}") from exc
    finally:
        connection.close()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_manifest(
    dataset_root: Path,
    *,
    dataset_version: str,
    expected_original_build_commit: str,
    expected_recovery_commit: str,
    expected_artifact_sha256: dict[str, str],
) -> tuple[dict[str, Any], str]:
    path = dataset_root / "build-manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid promoted build manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("promoted build manifest must be a JSON object")
    manifest = {str(key): item for key, item in value.items()}
    if manifest.get("schema_version") != 1:
        raise RuntimeError("promoted build manifest must use schema_version 1")
    if manifest.get("dataset_version") != dataset_version:
        raise RuntimeError("promoted build manifest dataset_version mismatch")
    if manifest.get("profile") != "full":
        raise RuntimeError("notice amendment requires a full-profile dataset")
    installed_size = _installed_size(dataset_root)
    if manifest.get("installed_size") != installed_size:
        raise RuntimeError(
            "promoted build manifest installed_size does not match the dataset tree"
        )
    size_limit = manifest.get("installed_size_limit")
    if not isinstance(size_limit, int) or size_limit < installed_size:
        raise RuntimeError("promoted build manifest has an invalid installed-size gate")

    provenance = manifest.get("recovery_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("promoted build manifest has no recovery provenance")
    expected_original = _require_commit(
        expected_original_build_commit, field="expected_original_build_commit"
    )
    expected_recovery = _require_commit(
        expected_recovery_commit, field="expected_recovery_commit"
    )
    if provenance.get("mode") != "validated-post-global-semantic-resume":
        raise RuntimeError("promoted dataset has unsupported recovery provenance mode")
    if provenance.get("original_build_commit") != expected_original:
        raise RuntimeError("promoted dataset original build commit mismatch")
    if provenance.get("recovery_commit") != expected_recovery:
        raise RuntimeError("promoted dataset recovery commit mismatch")
    pipeline_identity = manifest.get("pipeline_identity")
    if not isinstance(pipeline_identity, str) or not _SHA256.fullmatch(pipeline_identity):
        raise RuntimeError("promoted dataset has an invalid pipeline identity")
    if provenance.get("recovery_pipeline_identity") != pipeline_identity:
        raise RuntimeError("recovery pipeline identity does not match build manifest")
    original_identity = provenance.get("original_pipeline_identity")
    if not isinstance(original_identity, str) or not _SHA256.fullmatch(original_identity):
        raise RuntimeError("promoted dataset has an invalid original pipeline identity")
    if provenance.get("artifact_sha256") != expected_artifact_sha256:
        raise RuntimeError("recovery provenance does not match immutable artifact anchors")
    if provenance.get("usearch_version") != "2.26.0":
        raise RuntimeError("recovery provenance does not pin USearch 2.26.0")
    uv_lock_sha256 = provenance.get("uv_lock_sha256")
    if not isinstance(uv_lock_sha256, str) or not _SHA256.fullmatch(uv_lock_sha256):
        raise RuntimeError("recovery provenance has an invalid uv.lock SHA-256")

    amendments = manifest.get("notice_amendments", [])
    if not isinstance(amendments, list) or any(
        not isinstance(item, dict) for item in amendments
    ):
        raise RuntimeError("notice_amendments must be a list of objects")
    return manifest, _canonical_sha256(provenance)


def _write_manifest_converged(path: Path, manifest: dict[str, Any]) -> None:
    for _attempt in range(4):
        write_json_atomic(path, manifest)
        installed_size = _installed_size(path.parent)
        if manifest.get("installed_size") == installed_size:
            return
        manifest["installed_size"] = installed_size
    raise RuntimeError("notice-amended installed_size did not converge")


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validated_notice_hash_map(value: object, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(_NOTICE_PATHS):
        raise RuntimeError(f"notice transaction {field} has an invalid file set")
    result: dict[str, str] = {}
    for relative in _NOTICE_PATHS:
        digest = value.get(relative)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise RuntimeError(
                f"notice transaction {field}.{relative} is not a SHA-256 digest"
            )
        result[relative] = digest
    return result


def _read_transaction_journal(transaction: Path, dataset_root: Path) -> dict[str, Any]:
    journal_path = transaction / "journal.json"
    try:
        value = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid notice-amendment transaction journal: {journal_path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("notice-amendment transaction journal must be an object")
    journal = {str(key): item for key, item in value.items()}
    if journal.get("schema_version") != _TRANSACTION_SCHEMA_VERSION:
        raise RuntimeError("notice-amendment transaction schema is unsupported")
    if journal.get("dataset_root") != str(dataset_root):
        raise RuntimeError("notice-amendment transaction belongs to another dataset")
    if journal.get("state") not in _TRANSACTION_STATES:
        raise RuntimeError("notice-amendment transaction state is invalid")
    amendment_commit = journal.get("amendment_commit")
    if not isinstance(amendment_commit, str) or not _COMMIT.fullmatch(amendment_commit):
        raise RuntimeError("notice-amendment transaction commit is invalid")
    reason = journal.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise RuntimeError("notice-amendment transaction reason is invalid")
    for field in ("original_manifest_sha256", "final_manifest_sha256"):
        digest = journal.get(field)
        if digest is not None and (
            not isinstance(digest, str) or not _SHA256.fullmatch(digest)
        ):
            raise RuntimeError(f"notice-amendment transaction {field} is invalid")
    if not isinstance(journal.get("original_manifest_sha256"), str):
        raise RuntimeError("notice-amendment transaction lacks the original manifest hash")
    if journal["state"] == "committed" and not isinstance(
        journal.get("final_manifest_sha256"), str
    ):
        raise RuntimeError("committed notice transaction lacks the final manifest hash")
    journal["old_notice_sha256"] = _validated_notice_hash_map(
        journal.get("old_notice_sha256"), field="old_notice_sha256"
    )
    journal["new_notice_sha256"] = _validated_notice_hash_map(
        journal.get("new_notice_sha256"), field="new_notice_sha256"
    )
    return journal


def _write_transaction_journal(transaction: Path, journal: dict[str, Any]) -> None:
    write_json_atomic(transaction / "journal.json", journal)


def _remove_transaction_tree(transaction: Path) -> None:
    """Delete only the fixed files that this implementation can create."""

    if not transaction.exists() and not transaction.is_symlink():
        return
    if transaction.is_symlink() or not transaction.is_dir():
        raise RuntimeError(f"unsafe notice-amendment transaction path: {transaction}")
    allowed_files = {
        "build-manifest.old",
        "build-manifest.old.partial",
        "journal.json",
        "journal.json.partial",
    }
    for tree_name in ("notices.new", "notices.old"):
        for relative in _NOTICE_PATHS:
            allowed_files.add(f"{tree_name}/{relative}")
            allowed_files.add(f"{tree_name}/{relative}.partial")
    allowed_directories = {
        "notices.new",
        "notices.new/licenses",
        "notices.old",
        "notices.old/licenses",
    }
    observed_files: list[Path] = []
    for path in transaction.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"transaction contains a symbolic link: {path}")
        if path.is_file():
            relative = path.relative_to(transaction).as_posix()
            if relative not in allowed_files:
                raise RuntimeError(f"transaction contains an unexpected file: {relative}")
            observed_files.append(path)
        elif path.is_dir():
            relative = path.relative_to(transaction).as_posix()
            if relative not in allowed_directories:
                raise RuntimeError(
                    f"transaction contains an unexpected directory: {relative}"
                )
    observed_files.sort(
        key=lambda path: (
            2
            if path.name == "journal.json"
            else 1 if path.name == "journal.json.partial" else 0,
            path.as_posix(),
        )
    )
    for path in observed_files:
        path.unlink()
    directories = sorted(
        (path for path in transaction.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in directories:
        directory.rmdir()
    transaction.rmdir()


def _recover_notice_transaction(
    dataset_root: Path, transaction: Path, transaction_partial: Path
) -> str | None:
    """Roll back an undecided transaction or finish cleanup of a committed one."""

    if transaction_partial.exists() or transaction_partial.is_symlink():
        _remove_transaction_tree(transaction_partial)
    if not transaction.exists() and not transaction.is_symlink():
        return None
    if transaction.is_symlink() or not transaction.is_dir():
        raise RuntimeError(f"unsafe notice-amendment transaction path: {transaction}")
    journal_path = transaction / "journal.json"
    if not journal_path.exists():
        # Cleanup removes the journal last. An otherwise empty tree is therefore
        # an interrupted cleanup, not an undecided data mutation.
        if any(path.is_file() or path.is_symlink() for path in transaction.rglob("*")):
            raise RuntimeError("notice-amendment transaction has no durable journal")
        _remove_transaction_tree(transaction)
        return None
    journal = _read_transaction_journal(transaction, dataset_root)
    notices = dataset_root / "notices"
    manifest_path = dataset_root / "build-manifest.json"
    if journal["state"] == "committed":
        if _notice_hashes(notices, require_exact_tree=True) != journal["new_notice_sha256"]:
            raise RuntimeError("committed notice transaction has mismatched notice bytes")
        if file_sha256(manifest_path) != journal["final_manifest_sha256"]:
            raise RuntimeError("committed notice transaction has a mismatched manifest")
        amendment_commit = str(journal["amendment_commit"])
        _remove_transaction_tree(transaction)
        return amendment_commit

    manifest_backup = transaction / "build-manifest.old"
    if manifest_backup.is_symlink() or not manifest_backup.is_file():
        raise RuntimeError("undecided notice transaction has no manifest backup")
    original_manifest = manifest_backup.read_bytes()
    if hashlib.sha256(original_manifest).hexdigest() != journal["original_manifest_sha256"]:
        raise RuntimeError("notice transaction manifest backup hash mismatch")
    old_notices = transaction / "notices.old"
    if old_notices.is_dir() and not old_notices.is_symlink():
        if _notice_hashes(old_notices, require_exact_tree=True) != journal["old_notice_sha256"]:
            raise RuntimeError("notice transaction backup hashes do not match the journal")
        if notices.exists() or notices.is_symlink():
            _remove_notice_tree(notices)
        os.replace(old_notices, notices)
    elif _notice_hashes(notices, require_exact_tree=True) != journal["old_notice_sha256"]:
        raise RuntimeError("undecided notice transaction cannot restore original notices")
    _write_bytes_atomic(manifest_path, original_manifest)
    if _notice_hashes(notices, require_exact_tree=True) != journal["old_notice_sha256"]:
        raise RuntimeError("rolled-back notice bytes failed verification")
    if file_sha256(manifest_path) != journal["original_manifest_sha256"]:
        raise RuntimeError("rolled-back manifest bytes failed verification")
    _remove_transaction_tree(transaction)
    return None


def amend_promoted_dataset_notices(
    dataset_root: Path,
    repository: Path,
    *,
    dataset_version: str,
    amendment_commit: str,
    reason: str,
    expected_original_build_commit: str,
    expected_recovery_commit: str,
    expected_lexicon_sha256: str,
    expected_global_index_sha256: str,
    expected_global_vectors_sha256: str,
) -> dict[str, Any]:
    """Replace only data notices using a durable, restart-recoverable transaction."""

    dataset_root = dataset_root.resolve()
    repository = repository.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"promoted dataset does not exist: {dataset_root}")
    if dataset_root.name.endswith(".partial"):
        raise RuntimeError("notice amendment refuses a partial dataset directory")
    lingering_partial = dataset_root.with_name(dataset_root.name + ".partial")
    if lingering_partial.exists() or lingering_partial.is_symlink():
        raise RuntimeError(f"notice amendment refuses lingering build state: {lingering_partial}")
    clean_reason = reason.strip()
    if not clean_reason or len(clean_reason) > 1000:
        raise ValueError("reason must contain between 1 and 1000 non-whitespace characters")

    lock_path = dataset_root.parent / f".{dataset_root.name}.notice-amendment.lock"
    transaction = dataset_root.parent / f".{dataset_root.name}.notice-amendment.transaction"
    transaction_partial = transaction.with_name(transaction.name + ".partial")
    with InstallationLock(lock_path):
        # Recovery depends only on the durable transaction and promoted dataset.
        # Complete it before consulting a repository that may have moved, become
        # dirty, or disappeared since the interrupted amendment.
        _recover_notice_transaction(dataset_root, transaction, transaction_partial)
        _validated_clean_head(repository, amendment_commit)
        repository_notice_sha256 = _notice_hashes(
            repository, require_exact_tree=False
        )
        _reject_transients(dataset_root)
        notices = dataset_root / "notices"
        old_notice_sha256 = _notice_hashes(notices, require_exact_tree=True)
        anchors = _artifact_anchors(
            dataset_root,
            expected_lexicon_sha256=expected_lexicon_sha256,
            expected_global_index_sha256=expected_global_index_sha256,
            expected_global_vectors_sha256=expected_global_vectors_sha256,
        )
        semantic_artifact_sha256 = _semantic_artifact_hashes(
            dataset_root, dataset_version=dataset_version
        )
        manifest, provenance_sha256 = _validated_manifest(
            dataset_root,
            dataset_version=dataset_version,
            expected_original_build_commit=expected_original_build_commit,
            expected_recovery_commit=expected_recovery_commit,
            expected_artifact_sha256=anchors,
        )
        amendments = manifest.get("notice_amendments", [])
        matching_amendments = [
            item for item in amendments if item.get("amendment_commit") == amendment_commit
        ]
        if matching_amendments:
            if len(matching_amendments) != 1:
                raise RuntimeError("this amendment commit is recorded more than once")
            recorded = matching_amendments[0]
            if (
                recorded.get("reason") != clean_reason
                or recorded.get("new_notice_sha256") != repository_notice_sha256
                or recorded.get("recovery_provenance_sha256") != provenance_sha256
                or recorded.get("semantic_artifact_sha256")
                != semantic_artifact_sha256
                or old_notice_sha256 != repository_notice_sha256
            ):
                raise RuntimeError("recorded amendment commit does not match current bytes")
            return manifest
        if old_notice_sha256 == repository_notice_sha256:
            raise RuntimeError("repository notices are unchanged; refusing a no-op amendment")

        if transaction.exists() or transaction_partial.exists():
            raise RuntimeError("notice-amendment transaction recovery did not reach a clean state")
        manifest_path = dataset_root / "build-manifest.json"
        original_manifest_bytes = manifest_path.read_bytes()
        try:
            staged_notices = transaction_partial / "notices.new"
            new_notice_sha256 = _copy_notices(repository, staged_notices)
            if new_notice_sha256 != repository_notice_sha256:
                raise RuntimeError("repository notices changed while amendment was staged")
            _crash_checkpoint("during_transaction_prepare")
            _write_bytes_atomic(
                transaction_partial / "build-manifest.old", original_manifest_bytes
            )
            journal: dict[str, Any] = {
                "schema_version": _TRANSACTION_SCHEMA_VERSION,
                "state": "prepared",
                "dataset_root": str(dataset_root),
                "amendment_commit": amendment_commit,
                "reason": clean_reason,
                "old_notice_sha256": old_notice_sha256,
                "new_notice_sha256": new_notice_sha256,
                "original_manifest_sha256": hashlib.sha256(
                    original_manifest_bytes
                ).hexdigest(),
                "final_manifest_sha256": None,
            }
            _write_transaction_journal(transaction_partial, journal)
            _validated_clean_head(repository, amendment_commit)
            if _notice_hashes(repository, require_exact_tree=False) != new_notice_sha256:
                raise RuntimeError("repository notices changed before transaction commit")
            os.replace(transaction_partial, transaction)
            _crash_checkpoint("after_transaction_prepare")

            os.replace(notices, transaction / "notices.old")
            _crash_checkpoint("after_old_notices_move")
            journal["state"] = "old_notices_moved"
            _write_transaction_journal(transaction, journal)

            os.replace(transaction / "notices.new", notices)
            _crash_checkpoint("after_new_notices_move")
            journal["state"] = "new_notices_activated"
            _write_transaction_journal(transaction, journal)

            record = {
                "amendment_commit": amendment_commit,
                "reason": clean_reason,
                "recovery_provenance_sha256": provenance_sha256,
                "old_notice_sha256": old_notice_sha256,
                "new_notice_sha256": new_notice_sha256,
                "semantic_artifact_sha256": semantic_artifact_sha256,
            }
            manifest["notice_amendments"] = [*amendments, record]
            _write_manifest_converged(manifest_path, manifest)
            _crash_checkpoint("after_manifest_update")

            _reject_transients(dataset_root)
            if _notice_hashes(notices, require_exact_tree=True) != new_notice_sha256:
                raise RuntimeError("amended notice bytes failed final verification")
            final_anchors = _artifact_anchors(
                dataset_root,
                expected_lexicon_sha256=expected_lexicon_sha256,
                expected_global_index_sha256=expected_global_index_sha256,
                expected_global_vectors_sha256=expected_global_vectors_sha256,
            )
            if final_anchors != anchors:
                raise RuntimeError("immutable artifact anchors changed during notice amendment")
            final_semantic_artifact_sha256 = _semantic_artifact_hashes(
                dataset_root, dataset_version=dataset_version
            )
            if final_semantic_artifact_sha256 != semantic_artifact_sha256:
                raise RuntimeError("semantic artifacts changed during notice amendment")
            final_manifest, final_provenance_sha256 = _validated_manifest(
                dataset_root,
                dataset_version=dataset_version,
                expected_original_build_commit=expected_original_build_commit,
                expected_recovery_commit=expected_recovery_commit,
                expected_artifact_sha256=anchors,
            )
            if final_provenance_sha256 != provenance_sha256:
                raise RuntimeError("recovery provenance changed during notice amendment")
            journal["state"] = "committed"
            journal["final_manifest_sha256"] = file_sha256(manifest_path)
            _write_transaction_journal(transaction, journal)
            _crash_checkpoint("after_commit_marker")
            _remove_transaction_tree(transaction)
            return final_manifest
        except Exception:
            _recover_notice_transaction(dataset_root, transaction, transaction_partial)
            raise
